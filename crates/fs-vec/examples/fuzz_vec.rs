//! The M4 fuzzing acid test: a vectorized coverage-guided syscall fuzzer built on
//! [`fs_vec::VecSystem`] (16 full-system guest lanes/thread), measured honestly against
//! `fs-cli`'s scalar thread-parallel fuzzer (`fuzzsoft fuzz --jobs N`).
//!
//! Mirrors `fs-cli`'s `cmd_fuzz` exactly for the boot/snapshot/inject/coverage/crash-detection
//! primitives (same guest agent protocol: `HC_EID`/`HC_SNAPSHOT`/`HC_DONE`, same `fs_prog`
//! generate/mutate/lower/to_wire wire format, same `kernel_crash_sig` oracle) — the only thing
//! that differs is the executor: instead of one scalar `Cpu`+`Machine` (or `Cpu`+`CowMachine` per
//! thread), one `VecSystem` runs 16 lanes per batch.
//!
//! CRITICAL correctness-of-approach point (see `VecSystem::step`'s convergence-gated SIMD fast
//! path): the 16 lanes in one batch are NOT 16 unrelated fuzz inputs — they are 16 MUTATIONS OF
//! ONE base program (`fs_prog::mutate` applied 16 times to the same parent). Related inputs keep
//! lanes converged (same `pc`/`privilege`/`satp`) through shared kernel code paths for longer,
//! which is the only way the SIMD/shared-fetch fast paths ever get a chance to fire. Injecting 16
//! independently-generated programs would diverge lanes at the very first differing syscall
//! number and make the fast path fire ~never — measuring nothing but per-lane-scalar overhead.
//!
//! Run with `cargo run --release --example fuzz_vec -p fs-vec -- [--verify] [--batches N]
//! [--case-insns N] [--boot-insns N] [--seed N]` from the repo root (needs
//! `firmware/{fw_jump.bin,Image,fuzzsoft.dtb}`).
//!
//! `--verify` runs one extra correctness batch: the same 16 lowered programs through
//! `VecSystem::run_batch` AND through 16 independent scalar `Cpu`+`CowMachine` runs, asserting
//! each lane's final coverage bitmap + UART output + exit reason are IDENTICAL. This is the
//! non-negotiable gate: a vectorized fuzzer that doesn't compute the same thing as scalar is
//! worthless regardless of throughput.

use fs_cov::{CovBitmap, VirginMap};
use fs_mmu::{Access, Golden, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_platform::{run_until, Clint, CowMachine, Machine, Stop};
use fs_riscv::{Cpu, SysExit};
use fs_vec::system::LaneExit;
use fs_vec::{VecSystem, LANES};
use std::sync::Arc;
use std::time::Instant;

const HC_EID: u32 = 0x0A55_0000;
const HC_SNAPSHOT: u32 = 0;
const HC_DONE: u32 = 1;

const RAM_BASE: u32 = 0x8000_0000;
const KERNEL_ADDR: u32 = 0x8040_0000;

// ---------------------------------------------------------------------------------------------
// Crash oracle — copied verbatim from `fs-cli/src/main.rs` (a binary crate, so its functions
// aren't importable). Small and self-contained by design (decision: `kernel_crash_sig`'s doc
// comment there calls out exactly what it detects and why).
// ---------------------------------------------------------------------------------------------

fn parse_epc(s: &str) -> Option<u32> {
    let i = s.find("epc : ")?;
    let hex: String = s[i + 6..].chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u32::from_str_radix(&hex, 16).ok()
}

fn fnv1a(s: &str) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn report_line_sig(s: &str) -> u32 {
    if let Some(i) = s.find("Allocated in ") {
        let site: &str = s[i + "Allocated in ".len()..].split(['+', ' ', '\n']).next().unwrap_or("");
        if !site.is_empty() {
            return fnv1a(site);
        }
    }
    let line = s
        .lines()
        .map(str::trim)
        .find(|l| {
            l.contains("Redzone")
                || l.contains("Poison")
                || l.contains("Object already free")
                || l.contains("Freepointer")
                || l.contains("Padding overwritten")
                || l.starts_with("BUG")
        })
        .unwrap_or("");
    let normalized: Vec<&str> = line
        .split_whitespace()
        .filter(|t| !t.contains("0x") && !(t.len() >= 6 && t.chars().all(|c| c.is_ascii_hexdigit())))
        .collect();
    fnv1a(&normalized.join(" "))
}

fn kernel_crash_sig(out: &[u8]) -> Option<u32> {
    let s = String::from_utf8_lossy(out);
    let hard_fault = s.contains("Unable to handle kernel")
        || s.contains("KASAN:")
        || s.contains("kernel BUG at")
        || (s.contains("Oops") && !s.contains("Attempted to kill init"));
    let slub_report = s.contains("Redzone overwritten")
        || s.contains("Poison overwritten")
        || s.contains("Object already free")
        || s.contains("Freepointer corrupt")
        || s.contains("Padding overwritten");
    if hard_fault {
        Some(parse_epc(&s).unwrap_or_else(|| report_line_sig(&s)))
    } else if slub_report {
        Some(report_line_sig(&s))
    } else {
        None
    }
}

/// Translate `count` consecutive words starting at guest VA `va` to their guest physical
/// addresses (write access) — mirrors `fs-cli/cmd_fuzz`'s prog/scratch-buffer translation.
fn translate_words(cpu: &mut Cpu, m: &mut Machine, va: u32, count: usize) -> Vec<u32> {
    let mut pas = Vec::with_capacity(count);
    for k in 0..count as u32 {
        pas.push(cpu.xlate(m, va + k * 4, Access::Write).expect("translate guest buffer"));
    }
    pas
}

fn write_words(bus: &mut CowMachine, pas: &[u32], words: &[u32]) {
    use fs_mmu::Bus;
    for (&pa, &w) in pas.iter().zip(words) {
        let _ = bus.store(pa, 4, w);
    }
}

fn write_scratch(bus: &mut CowMachine, pas: &[u32], bytes: &[u8]) {
    use fs_mmu::Bus;
    for (i, &pa) in pas.iter().enumerate() {
        let off = i * 4;
        let mut word = [0u8; 4];
        for (b, wb) in word.iter_mut().enumerate() {
            if let Some(&v) = bytes.get(off + b) {
                *wb = v;
            }
        }
        let _ = bus.store(pa, 4, u32::from_le_bytes(word));
    }
}

/// One independent scalar case run over a `CowMachine`, byte-for-byte the same driver body as
/// `fs-cli`'s `run_case_bus`: per-step CLINT/timer sync, `step_system`, non-fall-through edge
/// recording into `cov`. Used both as `--verify`'s oracle and (implicitly, via `--jobs 1`) as the
/// scalar baseline this whole experiment is measured against.
fn run_case_scalar(cpu: &mut Cpu, bus: &mut CowMachine, cov: &mut CovBitmap, deadline: u64) -> Stop {
    use fs_platform::sync_timer_cow;
    while cpu.insns_retired < deadline {
        bus.clint.mtime = cpu.virtual_time();
        sync_timer_cow(cpu, bus);
        let prev = cpu.pc;
        match cpu.step_system(bus) {
            SysExit::Continue => {
                let cur = cpu.pc;
                if cur != prev.wrapping_add(4) && cur != prev.wrapping_add(2) {
                    cov.record_edge(prev, cur);
                }
            }
            SysExit::Halt(c) => return Stop::Halt(c),
            SysExit::Hypercall(c) => return Stop::Hypercall(c),
        }
    }
    Stop::Budget
}

fn reset_scalar(cpu: &mut Cpu, bus: &mut CowMachine, golden_cpu: &Cpu, golden_clint: &Clint, base_uart: usize) {
    *cpu = golden_cpu.clone();
    bus.ram.reset();
    bus.clint = golden_clint.clone();
    bus.uart.out.truncate(base_uart);
}

struct Args {
    verify: bool,
    batches: u32,
    case_insns: u64,
    boot_insns: u64,
    seed: u32,
    ram_mb: u32,
}

fn parse_args() -> Args {
    let mut a = Args {
        verify: false,
        batches: 200,
        case_insns: 2_000_000,
        boot_insns: 3_000_000_000,
        seed: 1,
        ram_mb: 128,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let val = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--verify" => {
                a.verify = true;
                i += 1;
                continue;
            }
            "--batches" => a.batches = val(i).parse().unwrap_or(a.batches),
            "--case-insns" => a.case_insns = val(i).parse().unwrap_or(a.case_insns),
            "--boot-insns" => a.boot_insns = val(i).parse().unwrap_or(a.boot_insns),
            "--seed" => a.seed = val(i).parse().unwrap_or(a.seed),
            "--ram-mb" => a.ram_mb = val(i).parse().unwrap_or(a.ram_mb),
            other => {
                eprintln!("fuzz_vec: unexpected argument {other:?}");
                std::process::exit(1);
            }
        }
        i += 2;
    }
    a
}

fn main() {
    let args = parse_args();
    let ram_size = args.ram_mb * 0x0010_0000;

    // --- boot scalar to the guest agent's snapshot hypercall (identical to fs-cli::cmd_fuzz). ---
    let mut m = Machine::new(RAM_BASE, ram_size);
    m.ram.protect(RAM_BASE, ram_size, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    for (path, addr) in [("firmware/fw_jump.bin", RAM_BASE), ("firmware/Image", KERNEL_ADDR)] {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        m.ram.map(addr, &b, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    }
    let dtb = std::fs::read("firmware/fuzzsoft.dtb").unwrap_or_default();
    let dtb_addr = RAM_BASE + ram_size - 0x0020_0000;
    m.ram.map(dtb_addr, &dtb, PERM_READ | PERM_WRITE).unwrap();

    let mut cpu = Cpu::new(RAM_BASE);
    cpu.hypercall_eid = Some(HC_EID);
    cpu.regs[10] = 0;
    cpu.regs[11] = dtb_addr;

    eprintln!("fuzz_vec: booting to snapshot (budget {} insns)...", args.boot_insns);
    match run_until(&mut cpu, &mut m, args.boot_insns) {
        Stop::Hypercall(HC_SNAPSHOT) => {}
        other => panic!("fuzz_vec: expected snapshot hypercall, got {other:?} (pc={:#x})", cpu.pc),
    }
    eprintln!("fuzz_vec: snapshot captured at pc={:#010x} after {} insns", cpu.pc, cpu.insns_retired);

    let prog_va = cpu.regs[11];
    let scratch_va = cpu.regs[12];
    let prog_pas = translate_words(&mut cpu, &mut m, prog_va, fs_prog::WIRE_WORDS);
    let scratch_words = (fs_prog::DEFAULT_SCRATCH_CAP / 4) as usize;
    let scratch_pas = translate_words(&mut cpu, &mut m, scratch_va, scratch_words);
    let base_uart = m.uart.out.len();

    // Capture the shared golden image + golden hart/CLINT state — the exact instant every lane
    // (and, in `--verify`, every scalar oracle case) resets back to between batches.
    let golden = Arc::new(Golden::from_mmu(&m.ram));
    let golden_cpu = cpu.clone();
    let golden_clint = m.clint.clone();
    drop(m);

    let mut vs = VecSystem::from_golden_cpu(Arc::clone(&golden), &golden_cpu, RAM_BASE, ram_size);

    let mut rng = fs_prog::Rng::new(args.seed);
    let mut virgin = VirginMap::new();
    let mut corpus: Vec<fs_prog::Prog> = Vec::new();
    let mut crash_sigs = std::collections::HashSet::new();
    let mut crashes = 0u64;
    let mut programs = 0u64;
    let mut done = 0u64;
    let mut budget_hit = 0u64;

    // --- --verify: one batch, cross-checked lane-by-lane against 16 independent scalar runs. ---
    if args.verify {
        eprintln!("fuzz_vec: --verify — running one batch through VecSystem and 16 independent scalar oracles...");
        let base = fs_prog::generate(&mut rng);
        let lane_progs: Vec<fs_prog::Prog> = (0..LANES).map(|_| fs_prog::mutate(&mut rng, &base)).collect();
        let lowered: Vec<fs_prog::Lowered> =
            lane_progs.iter().map(|p| fs_prog::lower(p, scratch_va)).collect();

        vs.reset_batch(&golden_cpu, &golden_clint, base_uart);
        for (lane, low) in lowered.iter().enumerate() {
            vs.inject_lane(lane, &prog_pas, &fs_prog::to_wire(low));
            vs.inject_scratch(lane, &scratch_pas, &low.scratch);
        }
        vs.run_batch(HC_DONE, args.case_insns);

        let mut mismatches = 0;
        for (lane, low) in lowered.iter().enumerate() {
            let mut ocpu = golden_cpu.clone();
            let mut obus = CowMachine::from_golden(Arc::clone(&golden), RAM_BASE, ram_size);
            reset_scalar(&mut ocpu, &mut obus, &golden_cpu, &golden_clint, base_uart);
            write_words(&mut obus, &prog_pas, &fs_prog::to_wire(low));
            write_scratch(&mut obus, &scratch_pas, &low.scratch);
            let mut ocov = CovBitmap::new();
            let deadline = ocpu.insns_retired + args.case_insns;
            let ostop = run_case_scalar(&mut ocpu, &mut obus, &mut ocov, deadline);

            let vec_uart = vs.uart(lane);
            let oracle_uart = &obus.uart.out;
            let uart_match = vec_uart == oracle_uart.as_slice();
            let cov_match = vs.lane_coverage(lane).as_slice() == ocov.as_slice();
            let exit_match = match (vs.exit[lane], ostop) {
                (Some(LaneExit::Hypercall(a)), Stop::Hypercall(b)) => a == b,
                (Some(LaneExit::Halt(a)), Stop::Halt(b)) => a == b,
                (Some(LaneExit::Budget), Stop::Budget) => true,
                (None, Stop::Budget) => true, // both mean "still running at budget" — see note below
                _ => false,
            };
            if !uart_match || !cov_match || !exit_match {
                mismatches += 1;
                eprintln!(
                    "fuzz_vec: VERIFY MISMATCH lane {lane}: uart_match={uart_match} cov_match={cov_match} \
                     exit_match={exit_match} (vec exit={:?}, oracle stop={ostop:?})",
                    vs.exit[lane]
                );
            }
        }
        if mismatches == 0 {
            println!("VERIFY PASS: all {LANES} lanes match the independent scalar oracle (uart + coverage + exit)");
        } else {
            println!("VERIFY FAIL: {mismatches}/{LANES} lanes mismatched — see above");
            std::process::exit(1);
        }
    }

    // --- the fuzzing loop itself ---
    eprintln!(
        "fuzz_vec: fuzzing — {} batches x {LANES} lanes ({} programs), case budget {} insns/lane",
        args.batches,
        args.batches as u64 * LANES as u64,
        args.case_insns
    );
    let t0 = Instant::now();
    for batch in 0..args.batches {
        vs.reset_batch(&golden_cpu, &golden_clint, base_uart);

        // ONE base program per batch; the 16 lanes are 16 mutations of it (related inputs — see
        // module doc comment for why this, not 16 independent programs, is required for the SIMD
        // fast path to have any chance of firing).
        let base = if !corpus.is_empty() && rng.chance(85) {
            corpus[rng.below(corpus.len())].clone()
        } else {
            fs_prog::generate(&mut rng)
        };
        let lane_progs: Vec<fs_prog::Prog> = (0..LANES).map(|_| fs_prog::mutate(&mut rng, &base)).collect();
        let lowered: Vec<fs_prog::Lowered> =
            lane_progs.iter().map(|p| fs_prog::lower(p, scratch_va)).collect();
        for (lane, low) in lowered.iter().enumerate() {
            vs.inject_lane(lane, &prog_pas, &fs_prog::to_wire(low));
            vs.inject_scratch(lane, &scratch_pas, &low.scratch);
        }

        vs.run_batch(HC_DONE, args.case_insns);
        programs += LANES as u64;

        for (lane, prog) in lane_progs.iter().enumerate() {
            match vs.exit[lane] {
                Some(LaneExit::Hypercall(HC_DONE)) => done += 1,
                Some(LaneExit::Budget) => budget_hit += 1,
                _ => {}
            }
            if virgin.has_new_bits(vs.lane_coverage(lane)) {
                corpus.push(prog.clone());
            }
            let uart = vs.uart(lane);
            let out = &uart[base_uart.min(uart.len())..];
            if let Some(sig) = kernel_crash_sig(out) {
                crashes += 1;
                if crash_sigs.insert(sig) {
                    eprintln!("fuzz_vec: [KERNEL CRASH] epc={sig:#010x} batch {batch} lane {lane}");
                }
            }
        }

        if batch % 20 == 19 {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "fuzz_vec: {} programs | {} cov | corpus {} | {} kcrash ({} uniq) | {:.0} progs/s",
                programs,
                virgin.covered_buckets(),
                corpus.len(),
                crashes,
                crash_sigs.len(),
                programs as f64 / elapsed,
            );
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let total_steps = (vs.simd_steps + vs.scalar_steps).max(1);
    let simd_frac = vs.simd_steps as f64 / total_steps as f64;
    let shared_fetch_frac = vs.shared_fetch_steps as f64 / total_steps as f64;
    let progs_per_sec = programs as f64 / elapsed;

    println!("== fuzz_vec complete ==");
    println!("  programs      : {programs}  in {elapsed:.1}s  ({progs_per_sec:.1} programs/sec, {LANES} lanes/thread)");
    println!("  syscalls done : {done}  (budget-hit: {budget_hit})");
    println!("  coverage      : {} bitmap buckets", virgin.covered_buckets());
    println!("  corpus        : {} programs", corpus.len());
    println!("  kernel crashes: {crashes}  ({} unique kernel PCs)", crash_sigs.len());
    println!(
        "  SIMD fast path: {:.2}% of steps ({} simd / {} scalar); shared-fetch {:.2}%",
        simd_frac * 100.0,
        vs.simd_steps,
        vs.scalar_steps,
        shared_fetch_frac * 100.0
    );
    println!("  overlay pages : {} across {LANES} lanes", vs.total_overlay_pages());
    println!(
        "  compare       : ./target/release/fuzzsoft fuzz --jobs 1 --case-insns {} --cases N \
         reports scalar execs/sec on 1 core; {progs_per_sec:.1} programs/sec above is {LANES} \
         lanes on 1 core (thread not spawned) — divide/multiply accordingly for the per-core verdict",
        args.case_insns
    );
}
