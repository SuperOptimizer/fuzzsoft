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
use std::sync::Arc;

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

fn fnv1a(s: &str) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// A stable 32-bit crash signature for reports that lack an `epc :` register dump (SLUB debug
/// prints a `BUG <cache> …` banner + call trace but no epc). Keyed on the *allocation call site* —
/// `SLAB_STORE_USER` prints `Allocated in <symbol>+0x…`, and that symbol (e.g.
/// `___se_sys_memfd_create`) is identical across runs regardless of the object's runtime address,
/// so it both dedups distinct bug sites and stays constant during minimization (whereas the
/// `Redzone <addr>:` line embeds a per-allocation address that would make every run look unique).
/// Falls back to the address-stripped `BUG`/report line when no allocation site is recorded.
fn report_line_sig(s: &str) -> u32 {
    if let Some(i) = s.find("Allocated in ") {
        let site: &str = s[i + "Allocated in ".len()..]
            .split(['+', ' ', '\n'])
            .next()
            .unwrap_or("");
        if !site.is_empty() {
            return fnv1a(site);
        }
    }
    // Fallback: the report line with hex-looking tokens (addresses) removed, so it's run-stable.
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
        .filter(|t| {
            !t.contains("0x") && !(t.len() >= 6 && t.chars().all(|c| c.is_ascii_hexdigit()))
        })
        .collect();
    fnv1a(&normalized.join(" "))
}

/// Classify console output produced during a case. Returns `Some(sig)` only for a genuine KERNEL
/// fault (not a userspace segfault that merely killed init), deduped by a crash signature.
///
/// Two families are detected: (1) hard CPU faults / assertions (`Unable to handle kernel …`,
/// `Oops`, `kernel BUG at`, `KASAN:`) — deduped by faulting kernel PC (`epc`); and (2) *allocator*
/// self-check reports emitted by a `CONFIG_SLUB_DEBUG_ON` kernel (`Redzone overwritten`, `Poison
/// overwritten`, `Object already free`, `Freepointer corrupt`, `Padding overwritten`) — deduped by
/// a hash of the report's `BUG <cache> …` line, since those don't print an `epc` register dump.
/// The second family is what makes `firmware/Image.slubdebug`/`Image.buggy` heap-bug detection work.
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
        // Prefer the epc; fall back to the report-line hash if this oops lacks a register dump.
        Some(parse_epc(&s).unwrap_or_else(|| report_line_sig(&s)))
    } else if slub_report {
        Some(report_line_sig(&s))
    } else {
        None
    }
}

// The typed, resource-threaded program model now lives in `fs-prog` (the syzlang-lite library):
// `fs_prog::generate`/`mutate` build a typed `Prog` (real rv32 syscall descriptions, fd/sock
// resource threading), and `fs_prog::lower`+`to_wire` compile it to the flat wire buffer the guest
// agent (`boot/agent.c`) interprets — call slots plus a resource-fixup table plus a scratch image.
// See `crates/fs-prog/DESIGN.md` for the exact wire contract.

// ---------------------------------------------------------------------------------------------
// `GuestBus`: the per-case driver capability shared by `Machine` and `CowMachine` — CLINT
// timer-sync + UART readback layered on top of `fs_mmu::Bus`. Lets program injection, the per-case
// step loop, crash minimization, and seed replay all run generically over either backing store, so
// the parallel (`--jobs`) path can drive `CowMachine` (shared-golden copy-on-write RAM,
// `docs/cow-shared-ram.md`) through the exact same code the serial path drives `Machine` through.
// `--sanitize` needs direct access to `Machine.ram: Mmu` for the sanitizer's poison/alloc
// primitives, so it stays on the concrete `Machine`/`run_case` path, untouched by this trait.
// ---------------------------------------------------------------------------------------------
trait GuestBus: fs_mmu::Bus {
    fn clint_mtime_set(&mut self, t: u64);
    fn sync_timer(&self, cpu: &mut fs_riscv::Cpu);
    fn uart_out(&self) -> &[u8];
}

impl GuestBus for fs_platform::Machine {
    fn clint_mtime_set(&mut self, t: u64) {
        self.clint.mtime = t;
    }
    fn sync_timer(&self, cpu: &mut fs_riscv::Cpu) {
        fs_platform::sync_timer(cpu, self);
    }
    fn uart_out(&self) -> &[u8] {
        &self.uart.out
    }
}

impl GuestBus for fs_platform::CowMachine {
    fn clint_mtime_set(&mut self, t: u64) {
        self.clint.mtime = t;
    }
    fn sync_timer(&self, cpu: &mut fs_riscv::Cpu) {
        fs_platform::sync_timer_cow(cpu, self);
    }
    fn uart_out(&self) -> &[u8] {
        &self.uart.out
    }
}

/// Write a slice of `u32` words into guest physical memory via its precomputed per-word physical
/// addresses (`pas[k]` is the physical address of the k-th word). Extra `pas` beyond `words` are
/// left untouched; the guest ignores slots past the counts it reads.
fn write_words<B: fs_mmu::Bus>(m: &mut B, pas: &[u32], words: &[u32]) {
    for (&pa, &w) in pas.iter().zip(words) {
        let _ = m.store(pa, 4, w);
    }
}

/// Write a byte image (the lowered scratch region) into guest physical memory word-by-word via the
/// scratch region's precomputed per-word physical addresses. The image is zero-padded up to the
/// number of scratch words actually translated; anything past that is beyond the guest buffer and
/// dropped (the pointers `lower()` handed out never exceed the region cap).
fn write_scratch_bytes<B: fs_mmu::Bus>(m: &mut B, pas: &[u32], bytes: &[u8]) {
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

/// Generic per-case step loop over any `GuestBus` — the `CowMachine`-capable twin of `run_case`,
/// used by the parallel `--jobs` path (and by crash minimization / seed replay, which are generic
/// over `GuestBus` too). No sanitizer-hook support: `--sanitize` stays on `run_case`/`Machine`.
fn run_case_bus<B: GuestBus>(
    cpu: &mut fs_riscv::Cpu,
    m: &mut B,
    cov: &mut fs_cov::CovBitmap,
    deadline: u64,
) -> fs_platform::Stop {
    use fs_platform::Stop;
    use fs_riscv::SysExit;
    while cpu.insns_retired < deadline {
        m.clint_mtime_set(cpu.virtual_time());
        m.sync_timer(cpu);
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

/// Reset (via the caller-supplied `reset` closure — `Snapshot::reset` for `Machine`, `reset_cow`
/// for `CowMachine`), inject one program, run it to completion/deadline, and report the crash
/// signature (faulting kernel PC) with the console text if it faulted. The single primitive both
/// the fuzz loops and the crash minimizer use to execute a candidate program, generic over
/// `GuestBus` so it drives the serial `Machine` path and the parallel `CowMachine` path identically.
#[allow(clippy::too_many_arguments)]
fn inject_and_run<B: GuestBus>(
    cpu: &mut fs_riscv::Cpu,
    m: &mut B,
    mut reset: impl FnMut(&mut fs_riscv::Cpu, &mut B),
    prog: &fs_prog::Prog,
    scratch_va: u32,
    prog_pas: &[u32],
    scratch_pas: &[u32],
    case_insns: u64,
    base_uart: usize,
    run_map: &mut fs_cov::CovBitmap,
) -> (fs_platform::Stop, u64, Option<(u32, String)>) {
    let lowered = fs_prog::lower(prog, scratch_va);
    reset(cpu, m);
    let start = cpu.insns_retired;
    write_words(m, prog_pas, &fs_prog::to_wire(&lowered));
    write_scratch_bytes(m, scratch_pas, &lowered.scratch);
    run_map.clear();
    let deadline = cpu.insns_retired + case_insns;
    let stop = run_case_bus(cpu, m, run_map, deadline);
    let used = cpu.insns_retired - start;
    let uart = m.uart_out();
    let out = &uart[base_uart.min(uart.len())..];
    let crash = kernel_crash_sig(out).map(|sig| (sig, String::from_utf8_lossy(out).into_owned()));
    (stop, used, crash)
}

/// Rebuild a program from an ordered subset of its calls, keeping resource threading valid: a
/// `Produced` reference to a kept producer is re-indexed to its new position; a reference to a
/// *dropped* producer degrades to a harmless seed fd (so the program still lowers and is
/// well-formed). This is what lets the minimizer delete calls out of the middle of a chain.
fn subset_prog(prog: &fs_prog::Prog, keep: &[usize]) -> fs_prog::Prog {
    use fs_prog::{ArgValue, ResRef, TypedCall};
    let mut remap = std::collections::HashMap::new();
    for (new_i, &old_i) in keep.iter().enumerate() {
        remap.insert(old_i, new_i as u16);
    }
    let mut calls = Vec::with_capacity(keep.len());
    for &old_i in keep {
        let tc = &prog.calls[old_i];
        let mut args = tc.args.clone();
        for av in &mut args {
            if let ArgValue::Res(ResRef::Produced { call_idx, .. }) = av {
                match remap.get(&(*call_idx as usize)) {
                    Some(&new_idx) => *call_idx = new_idx,
                    None => *av = ArgValue::Res(ResRef::Seed(-100)), // producer dropped → bogus fd
                }
            }
        }
        calls.push(TypedCall { desc: tc.desc, args });
    }
    fs_prog::Prog { calls }
}

/// Delta-debug a crashing program down to a minimal call subset that still reproduces the *same*
/// kernel crash signature. O(n²) re-runs, n ≤ MAX_CALLS = 8, so ≤ ~28 executions — cheap. Returns
/// the minimized program plus the console text of its final reproducing run.
#[allow(clippy::too_many_arguments)]
fn minimize_program<B: GuestBus>(
    cpu: &mut fs_riscv::Cpu,
    m: &mut B,
    mut reset: impl FnMut(&mut fs_riscv::Cpu, &mut B),
    prog: &fs_prog::Prog,
    target_sig: u32,
    scratch_va: u32,
    prog_pas: &[u32],
    scratch_pas: &[u32],
    case_insns: u64,
    base_uart: usize,
) -> (fs_prog::Prog, Option<String>) {
    let mut run_map = fs_cov::CovBitmap::new();
    let mut keep: Vec<usize> = (0..prog.calls.len()).collect();
    let mut console: Option<String> = None;
    let mut changed = true;
    while changed && keep.len() > 1 {
        changed = false;
        for pos in 0..keep.len() {
            let mut cand = keep.clone();
            cand.remove(pos);
            let sub = subset_prog(prog, &cand);
            let (_, _, crash) = inject_and_run(
                cpu, m, &mut reset, &sub, scratch_va, prog_pas, scratch_pas, case_insns, base_uart,
                &mut run_map,
            );
            if let Some((sig, out)) = crash
                && sig == target_sig
            {
                keep = cand;
                console = Some(out);
                changed = true;
                break;
            }
        }
    }
    (subset_prog(prog, &keep), console)
}

/// Emit a standalone, compilable C reproducer for a (typically minimized) program. Mirrors
/// `boot/agent.c`'s interpreter exactly: a `scratch[]` image, per-call `results[]`, resource args
/// threaded from earlier calls' return values (Reg) or kernel-written out-buffers (Mem), and raw
/// `syscall(nr, …)` invocations. Faithful because it reuses the same `lower()` output the emulator
/// injects — pointers become `scratch + offset`, resources become `r[k]`.
fn emit_c_reproducer(prog: &fs_prog::Prog) -> String {
    use fs_prog::{ArgValue, FixupSrc};
    use std::fmt::Write as _;
    // Lower with scratch base 0 so a pointer arg's concrete value *is* its scratch byte offset.
    let low = fs_prog::lower(prog, 0);
    let cap = fs_prog::DEFAULT_SCRATCH_CAP as usize;

    let mut s = String::new();
    let _ = writeln!(s, "/* fuzzsoft crash reproducer — auto-generated. Build for the guest:");
    let _ = writeln!(
        s,
        " *   clang --target=riscv32 -march=rv32imac -mabi=ilp32 -static -O2 -o repro repro.c */"
    );
    let _ = writeln!(s, "#include <sys/syscall.h>");
    let _ = writeln!(s, "#include <unistd.h>");
    let _ = writeln!(s, "#include <string.h>\n");
    let _ = writeln!(s, "static unsigned char scratch[{cap}];");

    // The scratch initializer bytes (only the written prefix; rest stays zero).
    let _ = write!(s, "static const unsigned char scratch_init[] = {{");
    for (i, b) in low.scratch.iter().enumerate() {
        if i % 16 == 0 {
            let _ = write!(s, "\n  ");
        }
        let _ = write!(s, "0x{b:02x},");
    }
    let _ = writeln!(s, "\n}};\n");

    let _ = writeln!(s, "int main(void) {{");
    if !low.scratch.is_empty() {
        let _ = writeln!(s, "  memcpy(scratch, scratch_init, sizeof scratch_init);");
    }
    let _ = writeln!(s, "  long r[{}];", prog.calls.len().max(1));
    let _ = writeln!(s, "  (void)r;");

    for (i, (tc, cc)) in prog.calls.iter().zip(&low.calls).enumerate() {
        // Build the six argument expressions.
        let mut argexpr: [String; 6] = Default::default();
        for (j, ae) in argexpr.iter_mut().enumerate() {
            // A fixup for (i, j) overrides the concrete placeholder with a runtime value.
            let fx = low
                .fixups
                .iter()
                .find(|f| f.dst_call as usize == i && f.dst_arg as usize == j);
            *ae = if let Some(f) = fx {
                match f.src {
                    FixupSrc::Reg(src) => format!("r[{src}]"),
                    FixupSrc::Mem(off) => format!("*(unsigned *)(scratch + {off})"),
                }
            } else if matches!(tc.args.get(j), Some(ArgValue::Ptr(_))) {
                // Pointer arg: concrete value is the scratch byte offset (base was 0).
                format!("(long)(scratch + {})", cc.args[j])
            } else {
                format!("{}u", cc.args[j])
            };
        }
        let _ = writeln!(
            s,
            "  r[{i}] = syscall({}, {}, {}, {}, {}, {}, {}); /* {} */",
            cc.nr, argexpr[0], argexpr[1], argexpr[2], argexpr[3], argexpr[4], argexpr[5], tc.desc.name
        );
    }
    let _ = writeln!(s, "  return 0;\n}}");
    s
}

/// Minimize a fresh kernel crash and write two artifacts under `crashes/`: a human-readable trace
/// (`crash_<sig>.txt`, syscall names + the oops console) and a compilable C reproducer
/// (`crash_<sig>.c`). Best-effort — reports what it wrote to stderr.
#[allow(clippy::too_many_arguments)]
fn handle_new_crash<B: GuestBus>(
    cpu: &mut fs_riscv::Cpu,
    m: &mut B,
    reset: impl FnMut(&mut fs_riscv::Cpu, &mut B),
    prog: &fs_prog::Prog,
    sig: u32,
    scratch_va: u32,
    prog_pas: &[u32],
    scratch_pas: &[u32],
    case_insns: u64,
    base_uart: usize,
) {
    let before = prog.calls.len();
    let (minimal, console) = minimize_program(
        cpu, m, reset, prog, sig, scratch_va, prog_pas, scratch_pas, case_insns, base_uart,
    );
    let names: Vec<&str> = minimal.calls.iter().map(|c| c.desc.name).collect();
    eprintln!(
        "fuzz: minimized crash epc={sig:#010x} from {before} → {} calls: {names:?}",
        minimal.calls.len()
    );

    if std::fs::create_dir_all("crashes").is_err() {
        return;
    }
    let mut trace = format!(
        "fuzzsoft kernel crash\n  epc (faulting kernel PC): {sig:#010x}\n  minimized to {} calls (from {before}):\n",
        minimal.calls.len()
    );
    for (i, c) in minimal.calls.iter().enumerate() {
        trace.push_str(&format!("    {i}: {} (nr {})\n", c.desc.name, c.desc.nr));
    }
    if let Some(out) = &console {
        trace.push_str("\n--- kernel console ---\n");
        trace.push_str(out);
    }
    let txt = format!("crashes/crash_{sig:08x}.txt");
    let cfile = format!("crashes/crash_{sig:08x}.c");
    let _ = std::fs::write(&txt, trace);
    let _ = std::fs::write(&cfile, emit_c_reproducer(&minimal));
    eprintln!("fuzz: wrote {txt} and {cfile}");
}

// ---- Corpus persistence (decision #34): make campaigns cumulative. ----
//
// A typed `fs_prog::Prog` is serialized to a compact, whitespace-tokenized text form using only
// fs-prog's *public* API (call names + a pre-order `ArgValue` token stream), so a `--corpus-dir`
// of `.prog` files survives across runs. Deserialization maps each call name back to its
// `&'static SyscallDesc` via `fs_prog::SYSCALLS` and rebuilds the typed arg tree, rejecting
// anything that fails `Prog::is_well_formed` — so a corrupt/stale file can never inject an invalid
// program.

/// Append the pre-order token encoding of one `ArgValue` to `out`.
fn enc_arg(av: &fs_prog::ArgValue, out: &mut Vec<String>) {
    use fs_prog::{ArgValue, ResRef};
    match av {
        ArgValue::Imm(v) => out.push(format!("i{v}")),
        ArgValue::Res(ResRef::Seed(s)) => out.push(format!("s{s}")),
        ArgValue::Res(ResRef::Produced { call_idx, slot }) => {
            out.push(format!("r{call_idx}.{slot}"))
        }
        ArgValue::Bytes(b) => {
            let mut h = String::from("b");
            if b.is_empty() {
                h.push('-');
            } else {
                for byte in b {
                    h.push_str(&format!("{byte:02x}"));
                }
            }
            out.push(h);
        }
        ArgValue::Ptr(inner) => {
            out.push("p".to_string());
            enc_arg(inner, out);
        }
        ArgValue::Struct(vals) => {
            out.push(format!("t{}", vals.len()));
            for v in vals {
                enc_arg(v, out);
            }
        }
    }
}

/// Decode one `ArgValue` (pre-order) from a token iterator; `None` on any malformed token.
fn dec_arg<'a, I: Iterator<Item = &'a str>>(it: &mut I) -> Option<fs_prog::ArgValue> {
    use fs_prog::{ArgValue, ResRef};
    let tok = it.next()?;
    let (tag, rest) = tok.split_at(1);
    Some(match tag {
        "i" => ArgValue::Imm(rest.parse().ok()?),
        "s" => ArgValue::Res(ResRef::Seed(rest.parse().ok()?)),
        "r" => {
            let (c, slot) = rest.split_once('.')?;
            ArgValue::Res(ResRef::Produced {
                call_idx: c.parse().ok()?,
                slot: slot.parse().ok()?,
            })
        }
        "b" => {
            if rest == "-" {
                ArgValue::Bytes(Vec::new())
            } else {
                let mut bytes = Vec::with_capacity(rest.len() / 2);
                let hb = rest.as_bytes();
                if !hb.len().is_multiple_of(2) {
                    return None;
                }
                for pair in hb.chunks(2) {
                    bytes.push(u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?);
                }
                ArgValue::Bytes(bytes)
            }
        }
        "p" => ArgValue::Ptr(Box::new(dec_arg(it)?)),
        "t" => {
            let n: usize = rest.parse().ok()?;
            let mut vals = Vec::with_capacity(n);
            for _ in 0..n {
                vals.push(dec_arg(it)?);
            }
            ArgValue::Struct(vals)
        }
        _ => return None,
    })
}

/// Serialize a program to the corpus text form: `FSCORPUS1 <n> (CALL <name> <nargs> <argtokens…>)*`.
fn serialize_prog(p: &fs_prog::Prog) -> String {
    let mut toks: Vec<String> = vec!["FSCORPUS1".into(), p.calls.len().to_string()];
    for c in &p.calls {
        toks.push("CALL".into());
        toks.push(c.desc.name.to_string());
        toks.push(c.args.len().to_string());
        for a in &c.args {
            enc_arg(a, &mut toks);
        }
    }
    toks.join(" ")
}

/// Parse the corpus text form back into a `Prog`; `None` unless it round-trips to a well-formed
/// program whose calls all resolve to known syscall descriptions.
fn deserialize_prog(s: &str) -> Option<fs_prog::Prog> {
    use fs_prog::TypedCall;
    let mut it = s.split_whitespace();
    if it.next()? != "FSCORPUS1" {
        return None;
    }
    let ncalls: usize = it.next()?.parse().ok()?;
    let mut calls = Vec::with_capacity(ncalls);
    for _ in 0..ncalls {
        if it.next()? != "CALL" {
            return None;
        }
        let name = it.next()?;
        let desc = fs_prog::SYSCALLS.iter().find(|d| d.name == name)?;
        let nargs: usize = it.next()?.parse().ok()?;
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(dec_arg(&mut it)?);
        }
        calls.push(TypedCall { desc, args });
    }
    let prog = fs_prog::Prog { calls };
    prog.is_well_formed().then_some(prog)
}

/// Load every `*.prog` file in `dir` into a list of valid programs (silently skipping unparseable
/// or ill-formed ones). Missing directory → empty list.
fn load_corpus(dir: &str) -> Vec<fs_prog::Prog> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("prog") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&path)
            && let Some(p) = deserialize_prog(&text)
        {
            out.push(p);
        }
    }
    out
}

/// Persist a corpus to `dir` (created if missing), one `<hash>.prog` per program; content-addressed
/// so re-saving an unchanged corpus is idempotent. Returns how many new files were written.
fn save_corpus(dir: &str, corpus: &[fs_prog::Prog]) -> usize {
    if std::fs::create_dir_all(dir).is_err() {
        return 0;
    }
    let mut written = 0;
    for p in corpus {
        let text = serialize_prog(p);
        let name = format!("{dir}/{:08x}.prog", fnv1a(&text));
        if !std::path::Path::new(&name).exists() && std::fs::write(&name, &text).is_ok() {
            written += 1;
        }
    }
    written
}

/// Replay loaded seed programs once each to rebuild coverage feedback, returning the accumulated
/// virgin map and the coverage-minimized corpus (only programs that lit new buckets are kept — the
/// same admission rule the main loop uses). Runs on one guest; each seed resets the snapshot.
#[allow(clippy::too_many_arguments)]
fn replay_seeds<B: GuestBus>(
    cpu: &mut fs_riscv::Cpu,
    m: &mut B,
    mut reset: impl FnMut(&mut fs_riscv::Cpu, &mut B),
    seeds: &[fs_prog::Prog],
    scratch_va: u32,
    prog_pas: &[u32],
    scratch_pas: &[u32],
    case_insns: u64,
    base_uart: usize,
) -> (fs_cov::VirginMap, Vec<fs_prog::Prog>) {
    let mut virgin = fs_cov::VirginMap::new();
    let mut run_map = fs_cov::CovBitmap::new();
    let mut corpus = Vec::new();
    for p in seeds {
        let (_, _, _) = inject_and_run(
            cpu, m, &mut reset, p, scratch_va, prog_pas, scratch_pas, case_insns, base_uart,
            &mut run_map,
        );
        if virgin.has_new_bits(&run_map) {
            corpus.push(p.clone());
        }
    }
    (virgin, corpus)
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
    let mut corpus_dir: Option<String> = None;
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
            // Persist/seed the corpus across runs (decision #34): load all *.prog files from DIR at
            // start (replayed to rebuild coverage), and save the final corpus back to DIR.
            "--corpus-dir" => corpus_dir = Some(val(i)),
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
    let base_uart = m.uart.out.len();

    // --- multi-core path: N worker threads sharing ONE golden RAM image (`Arc<Golden>`) behind
    // per-thread copy-on-write overlays (`CowMachine`), instead of N independent ~256 MB `Machine`
    // clones (`docs/cow-shared-ram.md`, PR1-4, already on `main`). Handled first (and returns)
    // because it must NOT build a `Snapshot` (another full golden RAM copy the parallel path has
    // no use for) — it captures its own `Golden` straight from `m.ram` instead.
    if jobs > 1 {
        if sanitize {
            eprintln!("fuzz: --jobs > 1 is incompatible with --sanitize (PC-hook path is serial)");
            return ExitCode::FAILURE;
        }
        // Captured right here — after boot AND after the prog/scratch address translation above
        // (which can set PTE A/D bits) — the exact instant `Snapshot::capture` would otherwise
        // capture for the serial path below. `m` is dropped immediately after: `Golden` holds its
        // own copy of the mem/perm planes, so the original `Machine`'s ~256 MB is no longer needed.
        let golden = Arc::new(fs_mmu::Golden::from_mmu(&m.ram));
        let golden_cpu = cpu.clone();
        let golden_clint = m.clint.clone();
        let golden_ram_base = m.ram.base();
        let golden_ram_size = m.ram.size() as u32;
        drop(m);

        // Seed the corpus from a persisted --corpus-dir, replayed on one throwaway `CowMachine`
        // over the same golden image (cheap: shares the `Arc`, only the seeds' dirtied pages
        // allocate overlay storage) using the exact same reset primitive the workers use below.
        let (seed_virgin, seed_corpus) = if let Some(dir) = &corpus_dir {
            let seeds = load_corpus(dir);
            if seeds.is_empty() {
                eprintln!("fuzz: corpus-dir {dir} — no seeds loaded (fresh start)");
                (VirginMap::new(), Vec::new())
            } else {
                let mut seed_cpu = golden_cpu.clone();
                let mut seed_m = fs_platform::CowMachine::from_golden(
                    Arc::clone(&golden),
                    golden_ram_base,
                    golden_ram_size,
                );
                let (v, c) = replay_seeds(
                    &mut seed_cpu,
                    &mut seed_m,
                    |cpu, m| reset_cow(cpu, m, &golden_cpu, &golden_clint, base_uart),
                    &seeds,
                    scratch,
                    &prog_pas,
                    &scratch_pas,
                    case_insns,
                    base_uart,
                );
                eprintln!(
                    "fuzz: corpus-dir {dir} — loaded {} seeds, {} kept after coverage replay ({} buckets)",
                    seeds.len(),
                    c.len(),
                    v.covered_buckets()
                );
                (v, c)
            }
        } else {
            (VirginMap::new(), Vec::new())
        };

        return run_parallel(
            golden, golden_cpu, golden_clint, golden_ram_base, golden_ram_size, scratch, prog_pas,
            scratch_pas, base_uart, case_insns, cases, seed, jobs, corpus_dir, seed_virgin,
            seed_corpus,
        );
    }

    // --- serial path (also `--sanitize`'s only path — it needs `Machine.ram: Mmu` directly for
    // the sanitizer's poison/alloc primitives): a single `Machine` + golden `Snapshot`. ---
    let snap = Snapshot::capture(&cpu, &mut m);

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

    // Seed the corpus from a persisted --corpus-dir (decision #34): load saved programs and replay
    // them once to rebuild coverage feedback, so a campaign resumes where the last one left off.
    let (seed_virgin, seed_corpus) = if let Some(dir) = &corpus_dir {
        let seeds = load_corpus(dir);
        if seeds.is_empty() {
            eprintln!("fuzz: corpus-dir {dir} — no seeds loaded (fresh start)");
            (VirginMap::new(), Vec::new())
        } else {
            let (v, c) = replay_seeds(
                &mut cpu, &mut m, |cpu, m| snap.reset(cpu, m), &seeds, scratch, &prog_pas,
                &scratch_pas, case_insns, base_uart,
            );
            eprintln!(
                "fuzz: corpus-dir {dir} — loaded {} seeds, {} kept after coverage replay ({} buckets)",
                seeds.len(),
                c.len(),
                v.covered_buckets()
            );
            (v, c)
        }
    } else {
        (VirginMap::new(), Vec::new())
    };

    // --- coverage-guided fuzz loop over syscall *programs* ---
    let mut virgin = seed_virgin; // accumulated coverage (feedback), warm-started from the corpus
    let mut run_map = CovBitmap::new(); // per-case edge bitmap
    let mut rng = fs_prog::Rng::new(seed);
    // No syscall deny-list any more: `fs-prog` only generates from its curated table of real rv32
    // syscall descriptions (no address-space/signal/lifetime-destroying calls reach the agent), so
    // the crude number-blacklist the random generator needed is gone.
    let mut corpus: Vec<fs_prog::Prog> = seed_corpus;
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
        let crash = kernel_crash_sig(out);
        if let Some(sig) = crash {
            crashes += 1;
            if crash_sigs.insert(sig) {
                let names: Vec<&str> = prog.calls.iter().map(|c| c.desc.name).collect();
                let nrs: Vec<u32> = prog.calls.iter().map(|c| c.desc.nr).collect();
                eprintln!("fuzz: [KERNEL CRASH] epc={sig:#010x} case {case} calls={names:?} nrs={nrs:?}");
                eprintln!("{}", String::from_utf8_lossy(out));
                // Minimize + emit a C reproducer (resets the snapshot internally; safe mid-loop).
                handle_new_crash(
                    &mut cpu, &mut m, |cpu, m| snap.reset(cpu, m), &prog, sig, scratch, &prog_pas,
                    &scratch_pas, case_insns, base_uart,
                );
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
    if let Some(dir) = &corpus_dir {
        let n = save_corpus(dir, &corpus);
        println!("  corpus saved  : {n} new programs → {dir}");
    }
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

/// Reset a per-worker `CowMachine` case to the golden post-boot state — the `CowMachine` analogue
/// of `Snapshot::reset`, kept behavior-identical by construction: restore the hart to the golden
/// clone, drop this lane's RAM overlays back to golden (`CowRam::reset` — O(dirty) directory
/// entries dropped, zero byte copy-back, since golden is never mutated — cheaper even than
/// `Mmu::reset_dirty`'s O(dirty) copy-back), restore the CLINT to its golden clone, and truncate
/// UART output back to the pre-case length. Used for every per-case reset in the parallel path:
/// worker cases, corpus-dir seed replay, and crash minimization re-runs alike.
fn reset_cow(
    cpu: &mut fs_riscv::Cpu,
    m: &mut fs_platform::CowMachine,
    golden_cpu: &fs_riscv::Cpu,
    golden_clint: &fs_platform::Clint,
    base_uart: usize,
) {
    *cpu = golden_cpu.clone();
    m.ram.reset();
    m.clint = golden_clint.clone();
    m.uart.out.truncate(base_uart);
}

/// Multi-core coverage-guided fuzzing: boot/snapshot happened once on the caller's thread, which
/// captured one immutable `Arc<Golden>` RAM image (`docs/cow-shared-ram.md`). Here `jobs` worker
/// threads each build their OWN `CowMachine` — a small per-thread page directory + overlay pages —
/// over that SAME shared golden image (the memory dedup: one ~256 MB golden image process-wide
/// instead of one independent ~256 MB `Machine` clone per thread). Each thread owns its hart +
/// `CowMachine` and runs independent cases, claiming case indices from one atomic counter and
/// sharing one [`Shared`] (coverage map + corpus + crash set). This is "fuzz many kernels at once"
/// at *core* granularity — orthogonal to fs-vec's SIMD-lane vectorization (which packs many guests
/// per core).
#[allow(clippy::too_many_arguments)]
fn run_parallel(
    golden: Arc<fs_mmu::Golden>,
    golden_cpu: fs_riscv::Cpu,
    golden_clint: fs_platform::Clint,
    ram_base: u32,
    ram_size: u32,
    scratch_va: u32,
    prog_pas: Vec<u32>,
    scratch_pas: Vec<u32>,
    base_uart: usize,
    case_insns: u64,
    cases: u32,
    seed: u32,
    jobs: u32,
    corpus_dir: Option<String>,
    seed_virgin: fs_cov::VirginMap,
    seed_corpus: Vec<fs_prog::Prog>,
) -> ExitCode {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    let shared = Mutex::new(Shared {
        virgin: seed_virgin,
        corpus: seed_corpus,
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
    eprintln!(
        "fuzz: parallel mode — {jobs} worker threads, {cases} cases total (shared golden RAM, CoW overlays)"
    );

    std::thread::scope(|s| {
        for tid in 0..jobs {
            // Each worker builds its own CowMachine sharing `golden` (the memory dedup — one
            // golden RAM image behind every thread's small directory + overlay pages) and clones
            // the golden hart; a distinct RNG stream per thread. `golden_cpu`/`golden_clint`/
            // `prog_pas`/`scratch_pas`/`shared`/`counter`/`t0` are shared immutably by reference
            // (thread::scope lets us borrow the stack).
            let mut cpu_t = golden_cpu.clone();
            let mut m_t = fs_platform::CowMachine::from_golden(Arc::clone(&golden), ram_base, ram_size);
            let seed_t = seed.wrapping_add(tid.wrapping_mul(0x9E37_79B9)).max(1);
            let shared = &shared;
            let counter = &counter;
            let golden_cpu = &golden_cpu;
            let golden_clint = &golden_clint;
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

                    reset_cow(&mut cpu_t, &mut m_t, golden_cpu, golden_clint, base_uart);
                    let case_start = cpu_t.insns_retired;
                    write_words(&mut m_t, prog_pas, &fs_prog::to_wire(&lowered));
                    write_scratch_bytes(&mut m_t, scratch_pas, &lowered.scratch);

                    run_map.clear();
                    let deadline = cpu_t.insns_retired + case_insns;
                    let stop = run_case_bus(&mut cpu_t, &mut m_t, &mut run_map, deadline);
                    let used = cpu_t.insns_retired - case_start;
                    let uart = m_t.uart_out();
                    let out = &uart[base_uart.min(uart.len())..];
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
                    let mut new_crash: Option<u32> = None;
                    if let Some((sig, console)) = crash_console {
                        sh.crashes += 1;
                        if sh.crash_sigs.insert(sig) {
                            let names: Vec<&str> = prog.calls.iter().map(|c| c.desc.name).collect();
                            eprintln!(
                                "fuzz: [KERNEL CRASH] epc={sig:#010x} thread {tid} calls={names:?}"
                            );
                            eprintln!("{console}");
                            new_crash = Some(sig); // minimize *after* dropping the lock
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
                    drop(sh);

                    // Minimize + write the reproducer outside the lock (it re-runs the guest ~n²
                    // times on this thread's own state, dropping overlays between candidates via
                    // `reset_cow`; other threads keep fuzzing meanwhile).
                    if let Some(sig) = new_crash {
                        handle_new_crash(
                            &mut cpu_t,
                            &mut m_t,
                            |cpu, m| reset_cow(cpu, m, golden_cpu, golden_clint, base_uart),
                            &prog,
                            sig,
                            scratch_va,
                            prog_pas,
                            scratch_pas,
                            case_insns,
                            base_uart,
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
    if let Some(dir) = &corpus_dir {
        let n = save_corpus(dir, &sh.corpus);
        println!("  corpus saved  : {n} new programs → {dir}");
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `subset_prog` keeps any ordered subset well-formed: dropped resource producers degrade to
    /// seed fds, kept producers are re-indexed, and the result still lowers to a valid wire buffer.
    #[test]
    fn subset_prog_stays_well_formed_and_lowers() {
        for seed in 1..200u32 {
            let mut rng = fs_prog::Rng::new(seed);
            let prog = fs_prog::generate(&mut rng);
            let n = prog.calls.len();
            if n < 2 {
                continue;
            }
            // Drop the first call (most likely to be a producer others depend on).
            let keep: Vec<usize> = (1..n).collect();
            let sub = subset_prog(&prog, &keep);
            assert_eq!(sub.calls.len(), n - 1);
            assert!(sub.is_well_formed(), "seed {seed}: subset not well-formed");
            let low = fs_prog::lower(&sub, 0x1000);
            assert!(fs_prog::to_wire(&low).len() == fs_prog::WIRE_WORDS);
        }
    }

    /// The singleton keep-set edge case doesn't panic and stays well-formed.
    #[test]
    fn subset_prog_singleton_ok() {
        let mut rng = fs_prog::Rng::new(42);
        let prog = fs_prog::generate(&mut rng);
        if !prog.calls.is_empty() {
            let sub = subset_prog(&prog, &[prog.calls.len() - 1]);
            assert_eq!(sub.calls.len(), 1);
            assert!(sub.is_well_formed());
        }
    }

    /// The C reproducer is structurally sound: one `syscall(nr, …)` per call, a `main`, and a
    /// `scratch` buffer.
    #[test]
    fn c_reproducer_emits_one_syscall_per_call() {
        let mut rng = fs_prog::Rng::new(7);
        // Prefer a program with at least one resource fixup so we exercise the r[k] path.
        let mut prog = fs_prog::generate(&mut rng);
        for _ in 0..500 {
            let low = fs_prog::lower(&prog, 0);
            if !low.fixups.is_empty() && prog.calls.len() >= 2 {
                break;
            }
            prog = fs_prog::generate(&mut rng);
        }
        let c = emit_c_reproducer(&prog);
        assert!(c.contains("int main"));
        assert!(c.contains("static unsigned char scratch"));
        assert_eq!(c.matches("syscall(").count(), prog.calls.len());
        for call in &prog.calls {
            assert!(c.contains(&format!("{}", call.desc.nr)));
        }
    }

    /// Corpus serialization round-trips: for many generated programs, serialize → deserialize
    /// yields a program with identical calls (names + args) that is still well-formed.
    #[test]
    fn corpus_serialization_round_trips() {
        for seed in 1..300u32 {
            let mut rng = fs_prog::Rng::new(seed);
            let prog = fs_prog::generate(&mut rng);
            let text = serialize_prog(&prog);
            let back = deserialize_prog(&text).expect("must deserialize");
            assert_eq!(back.calls.len(), prog.calls.len(), "seed {seed}");
            for (a, b) in prog.calls.iter().zip(&back.calls) {
                assert_eq!(a.desc.name, b.desc.name, "seed {seed}");
                assert_eq!(a.args, b.args, "seed {seed} args mismatch");
            }
            // And it re-serializes identically (canonical form is stable).
            assert_eq!(serialize_prog(&back), text, "seed {seed}");
        }
    }

    /// A malformed / unknown-syscall corpus line is rejected, never panics.
    #[test]
    fn corpus_deserialize_rejects_garbage() {
        assert!(deserialize_prog("").is_none());
        assert!(deserialize_prog("NOPE 1 CALL foo 0").is_none());
        assert!(deserialize_prog("FSCORPUS1 1 CALL not_a_real_syscall 0").is_none());
        assert!(deserialize_prog("FSCORPUS1 99 CALL").is_none());
    }

    /// The fs-cli retrofit's correctness gate: the parallel (`--jobs`) path's new plumbing —
    /// `GuestBus`, the generic `run_case_bus` step loop, and `reset_cow` (the `CowMachine` analogue
    /// of `Snapshot::reset`) — must drive a `CowMachine` case to results byte-identical to
    /// `run_case`/`Snapshot::reset` driving the same case over a `Machine`. `CowMachine` is already
    /// proven byte-exact to `Machine` at the `Bus` level for arbitrary access streams
    /// (`fs-platform`'s own `tests/cow_machine.rs`, PR2 of `docs/cow-shared-ram.md`, untouched by
    /// this change); THIS test proves fs-cli's driver plumbing on top of that preserves the
    /// equivalence — same coverage bitmap, same final registers/PC/insns_retired, same UART output,
    /// same RAM contents — and does so across TWO resets in a row, so the reset step itself (not
    /// just a fresh backing store) is proven behavior-identical, not merely first-run-identical.
    #[test]
    fn cow_machine_case_matches_machine_case() {
        use fs_mmu::{Bus, PERM_EXEC, PERM_READ, PERM_WRITE};

        let base = 0x8000_0000u32;
        let size = 0x0001_0000u32; // 64 KiB — plenty for a tiny hand-assembled program.
        let data_addr = base + 0x1000; // 4 KiB-aligned so a bare `lui` loads it whole.
        let tohost = base + 0x2000;
        const T2: u8 = 7; // x7 — not one of the named ABI aliases, used as a scratch pointer reg.
        const T3: u8 = 28;
        const T4: u8 = 29;

        // A tiny deterministic "case": a 3-iteration counted store loop (so `run_case_bus` records
        // at least one non-fall-through coverage edge — the backward branch), then an HTIF halt.
        let mut code = Vec::new();
        for w in [
            asm::addi(fs_riscv::T0, fs_riscv::X0, 0), // t0 = 0 (counter)
            asm::addi(fs_riscv::T1, fs_riscv::X0, 3), // t1 = 3 (bound)
            asm::lui(T2, data_addr),                  // t2 = data_addr
            asm::sw(T2, fs_riscv::T0, 0),              // loop: [t2] = t0
            asm::addi(fs_riscv::T0, fs_riscv::T0, 1),  // t0 += 1
            asm::bne(fs_riscv::T0, fs_riscv::T1, -8),  // back to the sw while t0 != 3
            asm::lui(T3, tohost),
            asm::addi(T4, fs_riscv::X0, 1),
            asm::sw(T3, T4, 0), // HTIF halt (a write to `tohost` stops the hart)
        ] {
            code.extend_from_slice(&w.to_le_bytes());
        }

        // --- `Machine` + `Snapshot`: the serial path's backing store. ---
        let mut m = fs_platform::Machine::new(base, size);
        m.ram.protect(base, size, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        m.ram.map(base, &code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cpu_m = fs_riscv::Cpu::new(base);
        cpu_m.htif_tohost = Some(tohost);
        let snap = fs_platform::Snapshot::capture(&cpu_m, &mut m);
        let base_uart = m.uart.out.len();

        // --- `CowMachine` over a `Golden` captured from that SAME state: the parallel path's
        // backing store, built exactly as `cmd_fuzz`'s `--jobs > 1` branch builds it. ---
        let golden = Arc::new(fs_mmu::Golden::from_mmu(&m.ram));
        let golden_cpu = cpu_m.clone();
        let golden_clint = m.clint.clone();
        let mut cm = fs_platform::CowMachine::from_golden(Arc::clone(&golden), base, size);
        let mut cpu_c = golden_cpu.clone();

        // Run the identical case twice on each backing store — the second iteration's reset
        // (`snap.reset` vs `reset_cow`) must land both machines back on identical golden state.
        for iter in 0..2 {
            snap.reset(&mut cpu_m, &mut m);
            let mut cov_m = fs_cov::CovBitmap::new();
            let deadline_m = cpu_m.insns_retired + 1000;
            let stop_m = run_case_bus(&mut cpu_m, &mut m, &mut cov_m, deadline_m);

            reset_cow(&mut cpu_c, &mut cm, &golden_cpu, &golden_clint, base_uart);
            let mut cov_c = fs_cov::CovBitmap::new();
            let deadline_c = cpu_c.insns_retired + 1000;
            let stop_c = run_case_bus(&mut cpu_c, &mut cm, &mut cov_c, deadline_c);

            assert_eq!(stop_m, stop_c, "iter {iter}: Stop mismatch");
            assert_eq!(cpu_m.regs, cpu_c.regs, "iter {iter}: register mismatch");
            assert_eq!(cpu_m.pc, cpu_c.pc, "iter {iter}: pc mismatch");
            assert_eq!(
                cpu_m.insns_retired, cpu_c.insns_retired,
                "iter {iter}: insns_retired mismatch"
            );
            assert_eq!(cov_m.as_slice(), cov_c.as_slice(), "iter {iter}: coverage bitmap mismatch");
            assert_eq!(m.uart.out, cm.uart.out, "iter {iter}: uart mismatch");
            assert_eq!(
                m.load(data_addr, 4).unwrap(),
                cm.load(data_addr, 4).unwrap(),
                "iter {iter}: final RAM content mismatch"
            );
            // The loop actually ran (sanity: this isn't vacuously comparing two no-ops).
            assert_eq!(m.load(data_addr, 4).unwrap(), 2, "iter {iter}: loop didn't run as expected");
        }
    }
}

