//! The M4 acid test: boot the SAME firmware+kernel+dtb image on a 16-lane [`fs_vec::VecSystem`]
//! (every lane cloned byte-identical from one loaded `Cpu`+`Machine`, no per-lane fuzzing input
//! difference at all) and, independently, on a standalone scalar `fs_riscv::Cpu` +
//! `fs_platform::Machine` for the same instruction budget — then assert lane 0's UART console
//! output matches the scalar run's byte-for-byte, and all 16 lanes' UART outputs match each other.
//! This proves `fs-vec` is genuinely full-system (it boots real firmware + a real Linux kernel,
//! privilege modes/CSRs/sv32/traps/CLINT timer interrupts and all) and that `VecSystem::step`'s
//! SIMD fast path never changes the architectural result versus the scalar golden model.
//!
//! Run with `cargo run --release --example boot_vec -p fs-vec [budget_insns]` from the repo root
//! (needs `firmware/{fw_jump.bin,Image,fuzzsoft.dtb}` — see `firmware/` in the repo, or a
//! `firmware` symlink to it). Default budget is 30,000,000 instructions per lane if not given.

use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_platform::Machine;
use fs_riscv::Cpu;
use fs_vec::{VecSystem, LANES};
use std::time::Instant;

const RAM_BASE: u32 = 0x8000_0000;
const RAM_MB: u32 = 128;
const KERNEL_ADDR: u32 = 0x8040_0000;

/// Load firmware/kernel/dtb into one fresh `Cpu`+`Machine`, exactly mirroring `fuzzsoft boot`'s
/// (`crates/fs-cli/src/main.rs::cmd_boot`) memory layout, so this example's result is directly
/// comparable to that known-good scalar path.
fn load_image() -> (Cpu, Machine) {
    let ram_size = RAM_MB * 0x0010_0000;
    let mut m = Machine::new(RAM_BASE, ram_size);
    m.ram.protect(RAM_BASE, ram_size, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let fw = std::fs::read("firmware/fw_jump.bin").expect("read firmware/fw_jump.bin");
    m.ram.map(RAM_BASE, &fw, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let kernel = std::fs::read("firmware/Image").expect("read firmware/Image");
    m.ram.map(KERNEL_ADDR, &kernel, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let dtb = std::fs::read("firmware/fuzzsoft.dtb").expect("read firmware/fuzzsoft.dtb");
    let dtb_addr = RAM_BASE + ram_size - 0x0020_0000; // 2 MiB below the top
    m.ram.map(dtb_addr, &dtb, PERM_READ | PERM_WRITE).unwrap();

    let mut cpu = Cpu::new(RAM_BASE);
    cpu.regs[10] = 0; // a0 = hartid
    cpu.regs[11] = dtb_addr; // a1 = dtb pointer
    (cpu, m)
}

fn main() {
    let budget: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30_000_000);

    eprintln!("boot_vec: loading firmware/fw_jump.bin + firmware/Image + firmware/fuzzsoft.dtb");
    let (cpu, machine) = load_image();

    // --- 16-lane VecSystem, cloned byte-identical from the same loaded image (no fuzzing input
    // difference at all — every lane must therefore produce byte-identical UART output to prove
    // this is a genuine deterministic full-system re-run, not a coincidence). ---
    eprintln!("boot_vec: running VecSystem ({LANES} lanes) for {budget} insns/lane...");
    let mut vs = VecSystem::from_template(&cpu, &machine);
    let t0 = Instant::now();
    let mut steps = 0u64;
    // Driven by a step counter (not any one lane's `insns_retired`): if a lane halts early (e.g. a
    // genuine bug causes it to fault before the others), its own retired-count freezes, but the
    // batch must still stop at the budget rather than spin forever waiting on it.
    while vs.any_active() && steps < budget {
        vs.step();
        steps += 1;
    }
    let vec_elapsed = t0.elapsed();
    let vec_insns = vs.insns_retired(0);

    // --- Standalone scalar oracle: the exact same image, run independently via
    // `fs_platform::run_until` (the same driver loop `fuzzsoft boot`/`fuzzsoft fuzz` use). ---
    eprintln!("boot_vec: running standalone scalar Cpu+Machine for {budget} insns...");
    let (mut scpu, mut sm) = load_image();
    let t1 = Instant::now();
    let stop = fs_platform::run_until(&mut scpu, &mut sm, budget);
    let scalar_elapsed = t1.elapsed();
    let scalar_insns = scpu.insns_retired;

    eprintln!(
        "boot_vec: VecSystem lane0 retired {vec_insns} insns in {:.2}s ({} steps: {} SIMD / {} scalar)",
        vec_elapsed.as_secs_f64(),
        vs.simd_steps + vs.scalar_steps,
        vs.simd_steps,
        vs.scalar_steps,
    );
    eprintln!(
        "boot_vec: scalar oracle retired {scalar_insns} insns in {:.2}s (stop: {stop:?})",
        scalar_elapsed.as_secs_f64()
    );
    let simd_frac = vs.simd_steps as f64 / (vs.simd_steps + vs.scalar_steps).max(1) as f64;
    eprintln!("boot_vec: SIMD fast-path fraction of steps = {:.1}%", simd_frac * 100.0);
    let vec_lane_ips = vec_insns as f64 / vec_elapsed.as_secs_f64();
    let scalar_ips = scalar_insns as f64 / scalar_elapsed.as_secs_f64();
    eprintln!(
        "boot_vec: throughput — VecSystem {:.1}M insns/s (lane0-equivalent, {LANES} lanes total), \
         scalar oracle {:.1}M insns/s",
        vec_lane_ips / 1e6,
        scalar_ips / 1e6,
    );

    // --- The acid test itself: lane 0's console must match the scalar oracle's byte-for-byte, and
    // every lane's console must match every other lane's. ---
    let lane0_uart = vs.uart(0).to_vec();
    let scalar_uart = sm.uart.out.clone();
    let lane0_matches_scalar = lane0_uart == scalar_uart;

    let mut all_lanes_match = true;
    let mut first_divergent_lane = None;
    let mut first_divergent_byte = None;
    for lane in 1..LANES {
        if vs.uart(lane) != lane0_uart.as_slice() {
            all_lanes_match = false;
            if first_divergent_lane.is_none() {
                first_divergent_lane = Some(lane);
                let other = vs.uart(lane);
                first_divergent_byte = lane0_uart
                    .iter()
                    .zip(other.iter())
                    .position(|(a, b)| a != b)
                    .or(Some(lane0_uart.len().min(other.len())));
            }
        }
    }

    let preview: String = String::from_utf8_lossy(&lane0_uart[..lane0_uart.len().min(400)]).into_owned();
    println!("=== first {} chars of lane 0's console ===", preview.len());
    println!("{preview}");
    println!("===");

    let pass = lane0_matches_scalar && all_lanes_match;
    if pass {
        println!(
            "PASS: lane0 UART == scalar UART ({} bytes), all {LANES} lanes identical",
            lane0_uart.len()
        );
    } else {
        println!("FAIL:");
        if !lane0_matches_scalar {
            let byte = lane0_uart
                .iter()
                .zip(scalar_uart.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(lane0_uart.len().min(scalar_uart.len()));
            println!(
                "  lane0 UART ({} bytes) != scalar UART ({} bytes); first differing byte at offset {byte}",
                lane0_uart.len(),
                scalar_uart.len()
            );
        }
        if !all_lanes_match {
            println!(
                "  lane {} diverges from lane 0 at UART byte offset {:?}",
                first_divergent_lane.unwrap(),
                first_divergent_byte
            );
        }
        std::process::exit(1);
    }
}
