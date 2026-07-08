//! fuzzsoft CLI — M0 milestone.
//!
//!   fuzzsoft run <elf> [--max-insns N] [--cov-out FILE]
//!   fuzzsoft gen-elf <out.elf>
//!
//! `run` loads a static RV32IM ELF, executes the scalar interpreter over the soft MMU, records
//! exact edge coverage, and prints the result + coverage. `gen-elf` writes a hermetic,
//! hand-encoded sample program (sum(1..=10) then ecall exit 55) so the whole loop is runnable
//! with no external toolchain (decision #26).

use std::process::ExitCode;

use fs_cov::Coverage;
use fs_loader::Program;
use fs_mmu::{Mmu, PERM_READ, PERM_WRITE};
use fs_riscv::{Cpu, Exit, asm};

const DEFAULT_MAX_INSNS: u64 = 100_000_000;
const STACK_SIZE: u32 = 0x0010_0000; // 1 MiB

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") => cmd_run(&args[2..]),
        Some("gen-elf") => cmd_gen_elf(&args[2..]),
        Some("boot") => cmd_boot(&args[2..]),
        Some("fuzz") => cmd_fuzz(&args[2..]),
        _ => {
            usage();
            ExitCode::FAILURE
        }
    }
}

const HC_EID: u32 = 0x0A55_0000;
const HC_SNAPSHOT: u32 = 0;
const HC_DONE: u32 = 1;

/// Deterministic xorshift32 PRNG (decision #7).
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// Run one fuzz case, recording non-fall-through control-flow edges into an AFL-style bitmap.
fn run_case(
    cpu: &mut fs_riscv::Cpu,
    m: &mut fs_platform::Machine,
    cov: &mut fs_cov::CovBitmap,
    deadline: u64,
) -> fs_platform::Stop {
    use fs_platform::Stop;
    use fs_riscv::SysExit;
    while cpu.insns_retired < deadline {
        m.clint.mtime = cpu.virtual_time();
        fs_platform::sync_timer(cpu, m);
        let prev = cpu.pc;
        match cpu.step_system(m) {
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

/// The faulting kernel PC from an oops register dump ("epc : c00185e0"), for crash dedup.
fn parse_epc(s: &str) -> Option<u32> {
    let i = s.find("epc : ")?;
    let hex: String = s[i + 6..].chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u32::from_str_radix(&hex, 16).ok()
}

/// Classify console output produced during a case. Returns `Some(sig)` only for a genuine KERNEL
/// fault (not a userspace segfault that merely killed init), deduped by faulting kernel PC.
fn kernel_crash_sig(out: &[u8]) -> Option<u32> {
    let s = String::from_utf8_lossy(out);
    let kernel_fault = s.contains("Unable to handle kernel")
        || s.contains("KASAN:")
        || s.contains("kernel BUG at")
        || (s.contains("Oops") && !s.contains("Attempted to kill init"));
    if !kernel_fault {
        return None;
    }
    Some(parse_epc(&s).unwrap_or(0))
}

const MAX_CALLS: usize = 8;

/// One syscall: number + 6 register arguments (a0..a5).
#[derive(Clone)]
struct Call {
    nr: u32,
    args: [u32; 6],
}

/// A fuzz input is a *sequence* of syscalls (syzkaller-style program), so dependencies like
/// open->ioctl->close are reachable (decision #20/#48).
#[derive(Clone)]
struct Prog {
    calls: Vec<Call>,
}

/// Generate a plausible argument from a pool: fds, sentinels, the guest scratch buffer (so
/// pointer args are valid), small ints, and full-random. Valid-ish args reach real kernel paths.
fn gen_arg(rng: &mut Rng, scratch: u32) -> u32 {
    match rng.next() % 8 {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 0xffff_ffff,
        4 => scratch,
        5 => scratch.wrapping_add(rng.next() % 4096),
        6 => rng.next() % 256,
        _ => rng.next(),
    }
}

fn pick_nr(rng: &mut Rng, deny: &[u32]) -> u32 {
    loop {
        let nr = rng.next() % 440;
        if !deny.contains(&nr) {
            return nr;
        }
    }
}

fn gen_call(rng: &mut Rng, scratch: u32, deny: &[u32]) -> Call {
    let mut args = [0u32; 6];
    for a in &mut args {
        *a = gen_arg(rng, scratch);
    }
    Call { nr: pick_nr(rng, deny), args }
}

fn gen_program(rng: &mut Rng, scratch: u32, deny: &[u32]) -> Prog {
    let n = 1 + rng.next() as usize % MAX_CALLS;
    Prog { calls: (0..n).map(|_| gen_call(rng, scratch, deny)).collect() }
}

fn mutate_program(rng: &mut Rng, base: &Prog, scratch: u32, deny: &[u32]) -> Prog {
    let mut p = base.clone();
    match rng.next() % 5 {
        0 if p.calls.len() < MAX_CALLS => {
            let idx = rng.next() as usize % (p.calls.len() + 1);
            p.calls.insert(idx, gen_call(rng, scratch, deny));
        }
        1 if p.calls.len() > 1 => {
            let idx = rng.next() as usize % p.calls.len();
            p.calls.remove(idx);
        }
        2 => {
            let idx = rng.next() as usize % p.calls.len();
            p.calls[idx].nr = pick_nr(rng, deny);
        }
        _ => {
            let idx = rng.next() as usize % p.calls.len();
            let a = rng.next() as usize % 6;
            p.calls[idx].args[a] = match rng.next() % 3 {
                1 => p.calls[idx].args[a] ^ (1 << (rng.next() % 32)),
                _ => gen_arg(rng, scratch),
            };
        }
    }
    if p.calls.is_empty() {
        p.calls.push(gen_call(rng, scratch, deny));
    }
    p
}

/// Write a program into the guest's `prog` buffer via its precomputed physical word addresses:
/// layout is `[count][nr, a0..a5]*`.
fn write_program(m: &mut fs_platform::Machine, prog_pas: &[u32], p: &Prog) {
    use fs_mmu::Bus;
    let n = p.calls.len().min(MAX_CALLS);
    let _ = m.store(prog_pas[0], 4, n as u32);
    for (i, call) in p.calls.iter().take(MAX_CALLS).enumerate() {
        let base = 1 + i * 7;
        let _ = m.store(prog_pas[base], 4, call.nr);
        for (j, &a) in call.args.iter().enumerate() {
            let _ = m.store(prog_pas[base + 1 + j], 4, a);
        }
    }
}

/// Snapshot-based, coverage-guided syscall fuzzer: boot to the agent's snapshot hypercall, then
/// loop reset -> inject (mutate corpus / generate) -> run -> feed coverage back -> detect crashes.
fn cmd_fuzz(args: &[String]) -> ExitCode {
    use fs_cov::{CovBitmap, VirginMap};
    use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_platform::{Machine, Snapshot, Stop, run_until};

    let mut firmware = "firmware/fw_jump.bin";
    let mut dtb = "firmware/fuzzsoft.dtb";
    let mut kernel = "firmware/Image";
    let mut ram_mb = 128u32;
    let mut boot_insns = 3_000_000_000u64;
    let mut case_insns = 2_000_000u64;
    let mut cases = 2000u32;
    let mut seed = 1u32;
    let ram_base = 0x8000_0000u32;
    let kernel_addr = 0x8040_0000u32;

    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let val = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match key {
            "--firmware" => firmware = Box::leak(val(i).into_boxed_str()),
            "--dtb" => dtb = Box::leak(val(i).into_boxed_str()),
            "--kernel" => kernel = Box::leak(val(i).into_boxed_str()),
            "--ram-mb" => ram_mb = val(i).parse().unwrap_or(128),
            "--boot-insns" => boot_insns = val(i).parse().unwrap_or(boot_insns),
            "--case-insns" => case_insns = val(i).parse().unwrap_or(case_insns),
            "--cases" => cases = val(i).parse().unwrap_or(cases),
            "--seed" => seed = val(i).parse().unwrap_or(1),
            other => {
                eprintln!("fuzz: unexpected argument {other:?}");
                return ExitCode::FAILURE;
            }
        }
        i += 2;
    }

    let ram_size = ram_mb * 0x0010_0000;
    let mut m = Machine::new(ram_base, ram_size);
    m.ram.protect(ram_base, ram_size, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    for (path, addr) in [(firmware, ram_base), (kernel, kernel_addr)] {
        match std::fs::read(path) {
            Ok(b) => m.ram.map(addr, &b, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap(),
            Err(e) => {
                eprintln!("fuzz: cannot read {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let dtb_bytes = std::fs::read(dtb).unwrap_or_default();
    let dtb_addr = ram_base + ram_size - 0x0020_0000;
    m.ram.map(dtb_addr, &dtb_bytes, PERM_READ | PERM_WRITE).unwrap();

    let mut cpu = Cpu::new(ram_base);
    cpu.hypercall_eid = Some(HC_EID);
    cpu.regs[10] = 0;
    cpu.regs[11] = dtb_addr;

    // --- boot to the agent's snapshot hypercall ---
    eprintln!("fuzz: booting to snapshot (budget {boot_insns} insns)...");
    match run_until(&mut cpu, &mut m, boot_insns) {
        Stop::Hypercall(HC_SNAPSHOT) => {}
        Stop::Hypercall(other) => {
            eprintln!("fuzz: unexpected hypercall {other} before snapshot");
            return ExitCode::FAILURE;
        }
        Stop::Halt(c) => {
            eprintln!("fuzz: guest halted ({c}) before snapshot");
            return ExitCode::FAILURE;
        }
        Stop::Budget => {
            eprintln!("fuzz: boot budget exhausted before snapshot (pc={:#x})", cpu.pc);
            return ExitCode::FAILURE;
        }
    }
    print!("{}", String::from_utf8_lossy(&m.uart.out));
    println!();
    eprintln!("fuzz: snapshot captured at pc={:#010x} after {} insns", cpu.pc, cpu.insns_retired);

    // The agent passed a1 = program buffer, a2 = scratch buffer (both user VAs). Translate the
    // program buffer's words to physical once (the mapping is stable across resets) so we can
    // write each program cheaply.
    let prog_va = cpu.regs[11];
    let scratch = cpu.regs[12];
    let mut prog_pas = Vec::with_capacity(1 + MAX_CALLS * 7);
    for k in 0..(1 + MAX_CALLS * 7) as u32 {
        match cpu.xlate(&mut m, prog_va + k * 4, fs_mmu::Access::Write) {
            Ok(pa) => prog_pas.push(pa),
            Err(_) => {
                eprintln!("fuzz: could not translate guest program buffer @ {prog_va:#x}");
                return ExitCode::FAILURE;
            }
        }
    }
    eprintln!("fuzz: prog buffer @ {prog_va:#010x}  scratch @ {scratch:#010x}");
    let snap = Snapshot::capture(&cpu, &mut m);
    let base_uart = m.uart.out.len();

    // --- coverage-guided fuzz loop over syscall *programs* ---
    let mut virgin = VirginMap::new(); // accumulated coverage (feedback)
    let mut run_map = CovBitmap::new(); // per-case edge bitmap
    let mut rng = Rng(seed.max(1));
    // Deny syscalls that corrupt the single-process agent's own address space / signal state /
    // lifetime (they crash init as a userspace false-positive rather than stressing the kernel).
    let deny = [
        93u32, 94, 142, // exit, exit_group, reboot
        139, // rt_sigreturn
        214, 215, 216, 222, 226, // brk, munmap, mremap, mmap, mprotect
        132, 133, 134, 135, // sigaltstack, rt_sigtimedwait, rt_sigaction, rt_sigprocmask
        220, 221, 281, 435, // clone, execve, execveat, clone3
    ];
    let mut corpus: Vec<Prog> = Vec::new();
    let mut crash_sigs = std::collections::HashSet::new();
    let mut crashes = 0u32;
    let mut done = 0u32;
    let mut budget_hit = 0u32;
    let mut total_case_insns = 0u64;
    let t0 = std::time::Instant::now();

    for case in 0..cases {
        // Mostly mutate the corpus, sometimes generate fresh (decision #48).
        let prog = if !corpus.is_empty() && rng.next() % 100 < 85 {
            let base = &corpus[(rng.next() as usize) % corpus.len()];
            mutate_program(&mut rng, base, scratch, &deny)
        } else {
            gen_program(&mut rng, scratch, &deny)
        };

        snap.reset(&mut cpu, &mut m);
        let case_start = cpu.insns_retired;
        write_program(&mut m, &prog_pas, &prog);

        run_map.clear();
        let deadline = cpu.insns_retired + case_insns;
        match run_case(&mut cpu, &mut m, &mut run_map, deadline) {
            Stop::Hypercall(HC_DONE) => done += 1,
            Stop::Budget => budget_hit += 1,
            _ => {}
        }
        total_case_insns += cpu.insns_retired - case_start;

        // Coverage feedback (AFL bitmap): a program that lit new buckets joins the corpus.
        if virgin.has_new_bits(&run_map) {
            corpus.push(prog.clone());
        }

        // Crash oracle (decision #19): only genuine KERNEL faults, deduped by faulting kernel PC.
        let out = &m.uart.out[base_uart.min(m.uart.out.len())..];
        if let Some(sig) = kernel_crash_sig(out) {
            crashes += 1;
            if crash_sigs.insert(sig) {
                let nrs: Vec<u32> = prog.calls.iter().map(|c| c.nr).collect();
                eprintln!("fuzz: [KERNEL CRASH] epc={sig:#010x} case {case} nrs={nrs:?}");
                eprintln!("{}", String::from_utf8_lossy(out));
            }
        }

        if case % 500 == 499 {
            eprintln!(
                "fuzz: {} cases | {} cov | corpus {} | {} kcrash ({} uniq) | {:.0} exec/s",
                case + 1,
                virgin.covered_buckets(),
                corpus.len(),
                crashes,
                crash_sigs.len(),
                (case + 1) as f64 / t0.elapsed().as_secs_f64(),
            );
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let execs_per_sec = cases as f64 / elapsed;
    let mips = total_case_insns as f64 / elapsed / 1e6;
    println!("== fuzz complete ==");
    println!("  cases         : {cases}  in {elapsed:.1}s  ({execs_per_sec:.0} execs/sec)");
    println!("  syscalls done : {done}  (budget-hit: {budget_hit})");
    println!("  coverage      : {} bitmap buckets", virgin.covered_buckets());
    println!("  corpus        : {} programs", corpus.len());
    println!("  kernel crashes: {crashes}  ({} unique kernel PCs)", crash_sigs.len());
    println!(
        "  guest speed   : {mips:.0} MIPS ({} insns/case avg)",
        total_case_insns / cases.max(1) as u64
    );
    ExitCode::SUCCESS
}

/// Like `fs_platform::run`, but flushes UART output live and logs privilege/trap transitions —
/// used to watch a long kernel boot as it happens.
fn run_traced(
    cpu: &mut fs_riscv::Cpu,
    m: &mut fs_platform::Machine,
    max: u64,
) -> Option<u32> {
    use fs_riscv::SysExit;
    use std::io::Write;
    let mut printed = 0usize;
    let stdout = std::io::stdout();
    while cpu.insns_retired < max {
        m.clint.mtime = cpu.virtual_time();
        cpu.csr.mtimecmp = m.clint.mtimecmp;
        if m.clint.msip & 1 != 0 {
            cpu.csr.mip |= 1 << 3;
        } else {
            cpu.csr.mip &= !(1 << 3);
        }
        let r = cpu.step_system(m);
        if m.uart.out.len() > printed {
            let mut lock = stdout.lock();
            let _ = lock.write_all(&m.uart.out[printed..]);
            let _ = lock.flush();
            printed = m.uart.out.len();
        }
        if let SysExit::Halt(c) = r {
            return Some(c);
        }
    }
    None
}

/// Boot a flat firmware (OpenSBI fw_jump) in M-mode with a DTB, printing UART console output.
fn cmd_boot(args: &[String]) -> ExitCode {
    use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_platform::Machine;

    let mut firmware: Option<&str> = None;
    let mut dtb: Option<&str> = None;
    let mut kernel: Option<&str> = None;
    let mut kernel_addr = 0x8040_0000u32;
    let mut ram_mb = 128u32;
    let mut max_insns = 200_000_000u64;
    let mut trace = false;
    let ram_base = 0x8000_0000u32;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--firmware" => {
                i += 1;
                firmware = args.get(i).map(String::as_str);
            }
            "--dtb" => {
                i += 1;
                dtb = args.get(i).map(String::as_str);
            }
            "--kernel" => {
                i += 1;
                kernel = args.get(i).map(String::as_str);
            }
            "--kernel-addr" => {
                i += 1;
                kernel_addr = args
                    .get(i)
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .unwrap_or(kernel_addr);
            }
            "--ram-mb" => {
                i += 1;
                ram_mb = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(128);
            }
            "--max-insns" => {
                i += 1;
                max_insns = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(max_insns);
            }
            "--trace" => trace = true,
            other => {
                eprintln!("boot: unexpected argument {other:?}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let (Some(fw_path), Some(dtb_path)) = (firmware, dtb) else {
        eprintln!("usage: fuzzsoft boot --firmware <fw.bin> --dtb <dtb> [--ram-mb N] [--max-insns N]");
        return ExitCode::FAILURE;
    };

    let fw = match std::fs::read(fw_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("boot: cannot read {fw_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let dtb_bytes = match std::fs::read(dtb_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("boot: cannot read {dtb_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ram_size = ram_mb * 0x0010_0000;
    let mut m = Machine::new(ram_base, ram_size);
    // Whole RAM RWX (no RAW oracle on firmware/kernel — decision #13).
    m.ram.protect(ram_base, ram_size, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    m.ram.map(ram_base, &fw, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    // Optional S-mode payload (kernel Image) at the firmware's jump address.
    if let Some(kpath) = kernel {
        match std::fs::read(kpath) {
            Ok(k) => {
                m.ram.map(kernel_addr, &k, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
                eprintln!("loaded kernel {} ({} KiB) @{:#x}", kpath, k.len() / 1024, kernel_addr);
            }
            Err(e) => {
                eprintln!("boot: cannot read kernel {kpath}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // Place the DTB high in RAM, clear of the firmware and the kernel load address.
    let dtb_addr = ram_base + ram_size - 0x0020_0000; // 2 MiB below the top
    m.ram.map(dtb_addr, &dtb_bytes, PERM_READ | PERM_WRITE).unwrap();

    let mut cpu = Cpu::new(ram_base);
    cpu.regs[10] = 0; // a0 = hartid
    cpu.regs[11] = dtb_addr; // a1 = dtb pointer

    eprintln!(
        "booting: fw={} ({} KiB) dtb@{:#x} ram={}MiB max_insns={}",
        fw_path,
        fw.len() / 1024,
        dtb_addr,
        ram_mb,
        max_insns
    );
    let result = if trace {
        run_traced(&mut cpu, &mut m, max_insns)
    } else {
        fs_platform::run(&mut cpu, &mut m, max_insns)
    };

    print!("{}", String::from_utf8_lossy(&m.uart.out));
    println!();
    match result {
        Some(code) => {
            eprintln!("[halted: code {code}, {} insns]", cpu.insns_retired);
            ExitCode::SUCCESS
        }
        None => {
            eprintln!(
                "[instruction budget reached: {} insns, pc={:#010x}, priv={:?}]",
                cpu.insns_retired, cpu.pc, cpu.privilege
            );
            eprintln!(
                "  a0={:#x} a1={:#x} a2={:#x} s7(x23)={:#x}",
                cpu.regs[10], cpu.regs[11], cpu.regs[12], cpu.regs[23]
            );
            eprintln!(
                "  mcause={:#x} mepc={:#010x} mtval={:#010x} mtvec={:#010x}",
                cpu.csr.mcause, cpu.csr.mepc, cpu.csr.mtval, cpu.csr.mtvec
            );
            ExitCode::SUCCESS
        }
    }
}

fn usage() {
    eprintln!("fuzzsoft — vectorized RISC-V syscall fuzzer (M0)");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  fuzzsoft run <elf> [--max-insns N] [--cov-out FILE]");
    eprintln!("  fuzzsoft gen-elf <out.elf>");
}

fn cmd_run(args: &[String]) -> ExitCode {
    let mut path: Option<&str> = None;
    let mut max_insns = DEFAULT_MAX_INSNS;
    let mut cov_out: Option<&str> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--max-insns" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse().ok()) {
                    Some(n) => max_insns = n,
                    None => {
                        eprintln!("error: --max-insns needs a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--cov-out" => {
                i += 1;
                match args.get(i) {
                    Some(s) => cov_out = Some(s),
                    None => {
                        eprintln!("error: --cov-out needs a path");
                        return ExitCode::FAILURE;
                    }
                }
            }
            other => {
                if path.is_some() {
                    eprintln!("error: unexpected argument {other:?}");
                    return ExitCode::FAILURE;
                }
                path = Some(other);
            }
        }
        i += 1;
    }

    let Some(path) = path else {
        eprintln!("error: missing <elf> path");
        usage();
        return ExitCode::FAILURE;
    };

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let prog = match fs_loader::parse(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    match execute(&prog, max_insns) {
        Ok((code, cpu, cov)) => {
            report(path, &prog, code, &cpu, &cov);
            if let Some(out) = cov_out
                && let Err(e) = write_coverage(out, &cov)
            {
                eprintln!("warning: could not write coverage to {out}: {e}");
            }
            // Mirror the guest's exit code into our process exit (clamped to u8).
            ExitCode::from(code as u8)
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Build a guest, load the program, and run to termination.
fn execute(prog: &Program, max_insns: u64) -> Result<(u32, Cpu, Coverage), String> {
    let (lo, hi) = prog.image_bounds();
    let base = lo & !0xfff;
    let top = (hi + 0xfff) & !0xfff;
    let size = (top - base) as usize + STACK_SIZE as usize;

    let mut mmu = Mmu::new(base, size);
    prog.load_into(&mut mmu).map_err(|e| e.to_string())?;

    // A read/write stack above the image; sp points at its top, 16-byte aligned.
    mmu.protect(top, STACK_SIZE, PERM_READ | PERM_WRITE)
        .map_err(|e| e.to_string())?;
    let sp = (base + size as u32) & !0xf;

    let mut cpu = Cpu::new(prog.entry);
    cpu.regs[2] = sp.wrapping_sub(16); // x2 = sp
    cpu.htif_tohost = prog.tohost;

    let mut cov = Coverage::new();
    cov.seed_block(cpu.pc);

    loop {
        if cpu.insns_retired >= max_insns {
            return Err(format!(
                "instruction budget exhausted after {} instructions (possible hang)",
                cpu.insns_retired
            ));
        }
        let pc_before = cpu.pc;
        match cpu.step(&mut mmu).map_err(|t| t.to_string())? {
            Exit::Continue => cov.record_edge(pc_before, cpu.pc),
            Exit::Halt(code) => return Ok((code, cpu, cov)),
            Exit::Ecall => {
                let a7 = cpu.regs[17];
                let a0 = cpu.regs[10];
                if a7 != 93 {
                    eprintln!("note: unhandled ecall a7={a7}; treating as exit");
                }
                return Ok((a0, cpu, cov));
            }
            Exit::Ebreak => return Ok((0, cpu, cov)),
        }
    }
}

fn report(path: &str, prog: &Program, code: u32, cpu: &Cpu, cov: &Coverage) {
    println!("== fuzzsoft run: {path} ==");
    println!("  entry        : {:#010x}", prog.entry);
    if let Some(t) = prog.tohost {
        println!("  htif tohost  : {t:#010x}");
    }
    println!("  exit code    : {code}");
    println!("  insns retired: {}", cpu.insns_retired);
    println!(
        "  coverage     : {} blocks, {} edges",
        cov.num_blocks(),
        cov.num_edges()
    );
    print!("  blocks       :");
    for (n, b) in cov.blocks.iter().enumerate() {
        if n == 16 {
            print!(" …(+{} more)", cov.num_blocks() - 16);
            break;
        }
        print!(" {b:#x}");
    }
    println!();
}

fn write_coverage(path: &str, cov: &Coverage) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# fuzzsoft coverage: {} edges", cov.num_edges());
    for (from, to) in &cov.edges {
        let _ = writeln!(s, "{from:#010x} {to:#010x}");
    }
    std::fs::write(path, s)
}

fn cmd_gen_elf(args: &[String]) -> ExitCode {
    let Some(out) = args.first() else {
        eprintln!("error: missing <out.elf> path");
        return ExitCode::FAILURE;
    };

    use asm::*;
    use fs_riscv::{A0, A7, T0, T1, X0};
    // sum(1..=10) = 55, then ecall exit.
    let prog = [
        addi(A0, X0, 0),
        addi(T0, X0, 1),
        addi(T1, X0, 11),
        bge(T0, T1, 16),
        add(A0, A0, T0),
        addi(T0, T0, 1),
        jal(X0, -12),
        addi(A7, X0, 93),
        ecall(),
    ];
    let mut code = Vec::new();
    for w in prog {
        code.extend_from_slice(&w.to_le_bytes());
    }

    let load_vaddr = 0x8000_0000u32;
    let elf = fs_loader::build_flat_elf(load_vaddr, &code);
    match std::fs::write(out, &elf) {
        Ok(()) => {
            println!("wrote {} ({} bytes) — entry {:#010x}", out, elf.len(), load_vaddr);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: cannot write {out}: {e}");
            ExitCode::FAILURE
        }
    }
}

