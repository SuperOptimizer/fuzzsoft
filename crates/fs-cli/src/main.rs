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

/// Emulator-level sanitizer context: PC-hooks on the kernel allocator + observed-allocation stats.
/// First step is validation-only (prove the hooks fire on the real kernel); redzone poisoning is
/// gated behind the SLUB false-positive analysis (docs/kernel-san.md, pending).
struct SanCtx {
    hooks: fs_san::PcHooks,
    san: fs_san::Sanitizer,
    lm: fs_san::LinearMap,
    poison: bool,
    allocs: u64,
    frees: u64,
    bytes: u64,
}

/// Run one fuzz case, recording non-fall-through control-flow edges into an AFL-style bitmap.
/// When `san` is set, drives the kernel-allocator PC-hooks each retired instruction.
fn run_case(
    cpu: &mut fs_riscv::Cpu,
    m: &mut fs_platform::Machine,
    cov: &mut fs_cov::CovBitmap,
    deadline: u64,
    mut san: Option<&mut SanCtx>,
) -> fs_platform::Stop {
    use fs_platform::Stop;
    use fs_riscv::SysExit;
    while cpu.insns_retired < deadline {
        if let Some(ctx) = san.as_deref_mut()
            && let Some(ev) = ctx.hooks.on_pc(cpu.pc, &cpu.regs)
        {
            match ev {
                fs_san::HookEvent::Alloc { addr, size } => {
                    ctx.allocs += 1;
                    ctx.bytes += size as u64;
                    if ctx.poison && let Some(pa) = ctx.lm.va_to_pa(addr) {
                        // Redzone around the SLUB-bucket-rounded allocation (trailing guard at the
                        // object boundary so legitimate ksize() access doesn't fault).
                        let _ = ctx.san.alloc(&mut m.ram, pa, fs_san::kmalloc_bucket(size));
                    }
                }
                fs_san::HookEvent::Free { addr } => {
                    ctx.frees += 1;
                    if ctx.poison && let Some(pa) = ctx.lm.va_to_pa(addr) {
                        let _ = ctx.san.free(&mut m.ram, pa);
                    }
                }
            }
        }
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

// The typed, resource-threaded program model now lives in `fs-prog` (the syzlang-lite library):
// `fs_prog::generate`/`mutate` build a typed `Prog` (real rv32 syscall descriptions, fd/sock
// resource threading), and `fs_prog::lower`+`to_wire` compile it to the flat wire buffer the guest
// agent (`boot/agent.c`) interprets — call slots plus a resource-fixup table plus a scratch image.
// See `crates/fs-prog/DESIGN.md` for the exact wire contract.

/// Write a slice of `u32` words into guest physical memory via its precomputed per-word physical
/// addresses (`pas[k]` is the physical address of the k-th word). Extra `pas` beyond `words` are
/// left untouched; the guest ignores slots past the counts it reads.
fn write_words(m: &mut fs_platform::Machine, pas: &[u32], words: &[u32]) {
    use fs_mmu::Bus;
    for (&pa, &w) in pas.iter().zip(words) {
        let _ = m.store(pa, 4, w);
    }
}

/// Write a byte image (the lowered scratch region) into guest physical memory word-by-word via the
/// scratch region's precomputed per-word physical addresses. The image is zero-padded up to the
/// number of scratch words actually translated; anything past that is beyond the guest buffer and
/// dropped (the pointers `lower()` handed out never exceed the region cap).
fn write_scratch_bytes(m: &mut fs_platform::Machine, pas: &[u32], bytes: &[u8]) {
    use fs_mmu::Bus;
    for (i, &pa) in pas.iter().enumerate() {
        let off = i * 4;
        let mut word = [0u8; 4];
        for (b, wb) in word.iter_mut().enumerate() {
            if let Some(&v) = bytes.get(off + b) {
                *wb = v;
            }
        }
        let _ = m.store(pa, 4, u32::from_le_bytes(word));
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
    let mut jobs = 1u32;
    let mut sanitize = false;
    let mut san_poison = false;
    let ram_base = 0x8000_0000u32;
    let kernel_addr = 0x8040_0000u32;

    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let val = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match key {
            "--sanitize" => {
                sanitize = true;
                i += 1;
                continue;
            }
            // Experimental: actually poison redzones. Known to false-positive on stock SLUB
            // (packed objects) — see docs/kernel-san.md. Needs slub_debug or a KFENCE-style
            // relocation to be usable; off by default.
            "--san-poison" => {
                sanitize = true;
                san_poison = true;
                i += 1;
                continue;
            }
            "--firmware" => firmware = Box::leak(val(i).into_boxed_str()),
            "--dtb" => dtb = Box::leak(val(i).into_boxed_str()),
            "--kernel" => kernel = Box::leak(val(i).into_boxed_str()),
            "--ram-mb" => ram_mb = val(i).parse().unwrap_or(128),
            "--boot-insns" => boot_insns = val(i).parse().unwrap_or(boot_insns),
            "--case-insns" => case_insns = val(i).parse().unwrap_or(case_insns),
            "--cases" => cases = val(i).parse().unwrap_or(cases),
            "--seed" => seed = val(i).parse().unwrap_or(1),
            // Fuzz with N worker threads (one guest Machine per thread → "many kernels at once"
            // at core granularity; SIMD-lane vectorization is the orthogonal fs-vec axis). Threads
            // share one coverage map + corpus behind a mutex. Incompatible with --sanitize (the
            // per-instruction PC-hook path stays single-threaded).
            "--jobs" => jobs = val(i).parse().unwrap_or(1).max(1),
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

    // The agent passed a1 = program buffer, a2 = scratch buffer (both user VAs). Translate both
    // regions' words to physical once (the mapping is stable across resets) so we can write each
    // program cheaply. The program buffer is `fs_prog::WIRE_WORDS` (186) words; the scratch region
    // is `DEFAULT_SCRATCH_CAP` bytes (32 KiB), where `lower()` places pointer-arg pointee data.
    let prog_va = cpu.regs[11];
    let scratch = cpu.regs[12];
    let scratch_words = (fs_prog::DEFAULT_SCRATCH_CAP / 4) as usize;
    let mut prog_pas = Vec::with_capacity(fs_prog::WIRE_WORDS);
    for k in 0..fs_prog::WIRE_WORDS as u32 {
        match cpu.xlate(&mut m, prog_va + k * 4, fs_mmu::Access::Write) {
            Ok(pa) => prog_pas.push(pa),
            Err(_) => {
                eprintln!("fuzz: could not translate guest program buffer @ {prog_va:#x}");
                return ExitCode::FAILURE;
            }
        }
    }
    let mut scratch_pas = Vec::with_capacity(scratch_words);
    for k in 0..scratch_words as u32 {
        match cpu.xlate(&mut m, scratch + k * 4, fs_mmu::Access::Write) {
            Ok(pa) => scratch_pas.push(pa),
            Err(_) => {
                eprintln!("fuzz: could not translate guest scratch buffer @ {scratch:#x}");
                return ExitCode::FAILURE;
            }
        }
    }
    eprintln!(
        "fuzz: prog buffer @ {prog_va:#010x} ({} words)  scratch @ {scratch:#010x} ({} words)",
        prog_pas.len(),
        scratch_pas.len()
    );
    let snap = Snapshot::capture(&cpu, &mut m);
    let base_uart = m.uart.out.len();

    // Optional emulator-level kernel-allocator sanitizer hooks (validation-only for now).
    let mut san_ctx = if sanitize {
        match std::fs::read_to_string("build/linux-src/System.map") {
            Ok(text) => {
                let syms = fs_san::parse_system_map(&text);
                let mut hooks = fs_san::PcHooks::new();
                fs_san::register_kernel_allocator_hooks(&mut hooks, &syms);
                let lm = fs_san::LinearMap::new(kernel_addr, ram_base, ram_size);
                // Self-check the linear-map offset against _start: it must map to kernel_addr,
                // else a mistranslation would poison unrelated physical memory.
                let self_check = syms
                    .get("_start")
                    .map(|&va| lm.va_to_pa(va) == Some(kernel_addr))
                    .unwrap_or(false);
                let poison = san_poison && self_check;
                if san_poison && !self_check {
                    eprintln!("fuzz: sanitizer VA->PA self-check FAILED — poisoning disabled");
                }
                eprintln!("fuzz: sanitizer ON (poison={poison}) — allocator hooks registered ({} symbols)", syms.len());
                Some(SanCtx {
                    hooks,
                    san: fs_san::Sanitizer::new(fs_san::DEFAULT_REDZONE),
                    lm,
                    poison,
                    allocs: 0,
                    frees: 0,
                    bytes: 0,
                })
            }
            Err(e) => {
                eprintln!("fuzz: --sanitize requested but build/linux-src/System.map unreadable: {e}");
                None
            }
        }
    } else {
        None
    };

    // --- multi-core path: N worker threads, one guest Machine per thread, shared coverage+corpus.
    if jobs > 1 {
        if sanitize {
            eprintln!("fuzz: --jobs > 1 is incompatible with --sanitize (PC-hook path is serial)");
            return ExitCode::FAILURE;
        }
        return run_parallel(
            cpu, m, snap, scratch, prog_pas, scratch_pas, base_uart, case_insns, cases, seed, jobs,
        );
    }

    // --- coverage-guided fuzz loop over syscall *programs* ---
    let mut virgin = VirginMap::new(); // accumulated coverage (feedback)
    let mut run_map = CovBitmap::new(); // per-case edge bitmap
    let mut rng = fs_prog::Rng::new(seed);
    // No syscall deny-list any more: `fs-prog` only generates from its curated table of real rv32
    // syscall descriptions (no address-space/signal/lifetime-destroying calls reach the agent), so
    // the crude number-blacklist the random generator needed is gone.
    let mut corpus: Vec<fs_prog::Prog> = Vec::new();
    let mut crash_sigs = std::collections::HashSet::new();
    let mut crashes = 0u32;
    let mut done = 0u32;
    let mut budget_hit = 0u32;
    let mut total_case_insns = 0u64;
    let t0 = std::time::Instant::now();

    for case in 0..cases {
        // Mostly mutate the corpus, sometimes generate fresh (decision #48).
        let prog = if !corpus.is_empty() && rng.chance(85) {
            let base = &corpus[rng.below(corpus.len())];
            fs_prog::mutate(&mut rng, base)
        } else {
            fs_prog::generate(&mut rng)
        };
        // Compile the typed program to the wire form (call slots + fixup table + scratch image),
        // placing pointer pointees at `scratch`'s guest VA so runtime pointers are valid.
        let lowered = fs_prog::lower(&prog, scratch);

        snap.reset(&mut cpu, &mut m);
        // Reset per-case sanitizer state (perms are restored by snap.reset; clear the tracking).
        if let Some(ctx) = san_ctx.as_mut() {
            ctx.san = fs_san::Sanitizer::new(fs_san::DEFAULT_REDZONE);
            ctx.hooks.clear_pending();
        }
        let case_start = cpu.insns_retired;
        write_words(&mut m, &prog_pas, &fs_prog::to_wire(&lowered));
        write_scratch_bytes(&mut m, &scratch_pas, &lowered.scratch);

        run_map.clear();
        let deadline = cpu.insns_retired + case_insns;
        match run_case(&mut cpu, &mut m, &mut run_map, deadline, san_ctx.as_mut()) {
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
                let names: Vec<&str> = prog.calls.iter().map(|c| c.desc.name).collect();
                let nrs: Vec<u32> = prog.calls.iter().map(|c| c.desc.nr).collect();
                eprintln!("fuzz: [KERNEL CRASH] epc={sig:#010x} case {case} calls={names:?} nrs={nrs:?}");
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
    if let Some(ctx) = &san_ctx {
        println!(
            "  kernel allocs : {} kmalloc ({} bytes), {} kfree  [hooks fired — sanitizer path validated]",
            ctx.allocs, ctx.bytes, ctx.frees
        );
    }
    println!(
        "  guest speed   : {mips:.0} MIPS ({} insns/case avg)",
        total_case_insns / cases.max(1) as u64
    );
    ExitCode::SUCCESS
}

/// Coverage + corpus + crash bookkeeping shared by all worker threads, behind one mutex. The
/// expensive part (emulating a case) happens *outside* the lock; the lock is only taken to pick a
/// mutation base and to fold a finished case's coverage/crash back in — cheap next to millions of
/// guest instructions per case.
struct Shared {
    virgin: fs_cov::VirginMap,
    corpus: Vec<fs_prog::Prog>,
    crash_sigs: std::collections::HashSet<u32>,
    crashes: u32,
    done: u32,
    budget_hit: u32,
    total_case_insns: u64,
    finished: u64, // cases fully folded in (for progress reporting)
}

/// Multi-core coverage-guided fuzzing: boot/snapshot happened once on the caller's thread; here we
/// clone that post-snapshot machine into `jobs` worker threads. Each thread owns its guest state
/// and runs independent cases, claiming case indices from one atomic counter and sharing one
/// [`Shared`] (coverage map + corpus + crash set). This is "fuzz many kernels at once" at *core*
/// granularity — orthogonal to fs-vec's SIMD-lane vectorization (which packs many guests per core).
#[allow(clippy::too_many_arguments)]
fn run_parallel(
    cpu: fs_riscv::Cpu,
    m: fs_platform::Machine,
    snap: fs_platform::Snapshot,
    scratch_va: u32,
    prog_pas: Vec<u32>,
    scratch_pas: Vec<u32>,
    base_uart: usize,
    case_insns: u64,
    cases: u32,
    seed: u32,
    jobs: u32,
) -> ExitCode {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    let shared = Mutex::new(Shared {
        virgin: fs_cov::VirginMap::new(),
        corpus: Vec::new(),
        crash_sigs: std::collections::HashSet::new(),
        crashes: 0,
        done: 0,
        budget_hit: 0,
        total_case_insns: 0,
        finished: 0,
    });
    let counter = AtomicU64::new(0);
    let cases = cases as u64;
    let t0 = std::time::Instant::now();
    eprintln!("fuzz: parallel mode — {jobs} worker threads, {cases} cases total");

    std::thread::scope(|s| {
        for tid in 0..jobs {
            // Each worker gets its own guest state (cloned once from the golden snapshot) and a
            // distinct RNG stream. `snap`/`prog_pas`/`scratch_pas`/`shared`/`counter`/`t0` are
            // shared immutably by reference (thread::scope lets us borrow the stack).
            let mut cpu_t = cpu.clone();
            let mut m_t = m.clone();
            let seed_t = seed.wrapping_add(tid.wrapping_mul(0x9E37_79B9)).max(1);
            let shared = &shared;
            let counter = &counter;
            let snap = &snap;
            let prog_pas = &prog_pas;
            let scratch_pas = &scratch_pas;
            let t0 = &t0;
            s.spawn(move || {
                let mut rng = fs_prog::Rng::new(seed_t);
                let mut run_map = fs_cov::CovBitmap::new();
                loop {
                    let case = counter.fetch_add(1, Ordering::Relaxed);
                    if case >= cases {
                        break;
                    }

                    // Pick a mutation base (or decide to generate) under the lock, then release it
                    // before the expensive emulation.
                    let base = {
                        let sh = shared.lock().unwrap();
                        if !sh.corpus.is_empty() && rng.chance(85) {
                            Some(sh.corpus[rng.below(sh.corpus.len())].clone())
                        } else {
                            None
                        }
                    };
                    let prog = match base {
                        Some(b) => fs_prog::mutate(&mut rng, &b),
                        None => fs_prog::generate(&mut rng),
                    };
                    let lowered = fs_prog::lower(&prog, scratch_va);

                    snap.reset(&mut cpu_t, &mut m_t);
                    let case_start = cpu_t.insns_retired;
                    write_words(&mut m_t, prog_pas, &fs_prog::to_wire(&lowered));
                    write_scratch_bytes(&mut m_t, scratch_pas, &lowered.scratch);

                    run_map.clear();
                    let deadline = cpu_t.insns_retired + case_insns;
                    let stop = run_case(&mut cpu_t, &mut m_t, &mut run_map, deadline, None);
                    let used = cpu_t.insns_retired - case_start;
                    let out = &m_t.uart.out[base_uart.min(m_t.uart.out.len())..];
                    let crash = kernel_crash_sig(out);
                    let crash_console =
                        crash.map(|sig| (sig, String::from_utf8_lossy(out).into_owned()));

                    // Fold results back in under the lock.
                    let mut sh = shared.lock().unwrap();
                    sh.total_case_insns += used;
                    match stop {
                        fs_platform::Stop::Hypercall(HC_DONE) => sh.done += 1,
                        fs_platform::Stop::Budget => sh.budget_hit += 1,
                        _ => {}
                    }
                    if sh.virgin.has_new_bits(&run_map) {
                        sh.corpus.push(prog.clone());
                    }
                    if let Some((sig, console)) = crash_console {
                        sh.crashes += 1;
                        if sh.crash_sigs.insert(sig) {
                            let names: Vec<&str> = prog.calls.iter().map(|c| c.desc.name).collect();
                            eprintln!(
                                "fuzz: [KERNEL CRASH] epc={sig:#010x} thread {tid} calls={names:?}"
                            );
                            eprintln!("{console}");
                        }
                    }
                    sh.finished += 1;
                    if sh.finished.is_multiple_of(2000) {
                        eprintln!(
                            "fuzz: {} cases | {} cov | corpus {} | {} kcrash ({} uniq) | {:.0} exec/s",
                            sh.finished,
                            sh.virgin.covered_buckets(),
                            sh.corpus.len(),
                            sh.crashes,
                            sh.crash_sigs.len(),
                            sh.finished as f64 / t0.elapsed().as_secs_f64(),
                        );
                    }
                }
            });
        }
    });

    let elapsed = t0.elapsed().as_secs_f64();
    let sh = shared.into_inner().unwrap();
    let mips = sh.total_case_insns as f64 / elapsed / 1e6;
    println!("== fuzz complete (parallel, {jobs} threads) ==");
    println!(
        "  cases         : {}  in {elapsed:.1}s  ({:.0} execs/sec)",
        sh.finished,
        sh.finished as f64 / elapsed
    );
    println!("  syscalls done : {}  (budget-hit: {})", sh.done, sh.budget_hit);
    println!("  coverage      : {} bitmap buckets", sh.virgin.covered_buckets());
    println!("  corpus        : {} programs", sh.corpus.len());
    println!("  kernel crashes: {}  ({} unique kernel PCs)", sh.crashes, sh.crash_sigs.len());
    println!(
        "  guest speed   : {mips:.0} MIPS aggregate ({} insns/case avg)",
        sh.total_case_insns / sh.finished.max(1)
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

