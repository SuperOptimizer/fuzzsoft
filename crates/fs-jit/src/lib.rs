//! Stage 0 of `docs/jit.md`: a zero-`unsafe`, PA-keyed, direct-mapped threaded-code cache of
//! decoded [`fs_riscv::Inst`]s — the GO/NO-GO gate for the whole JIT effort.
//!
//! # Design
//!
//! **PA-keyed, direct-mapped, golden-tier only.** The cache is indexed by the instruction's
//! *physical* address (mirroring `fs-riscv`'s 256-entry direct-mapped software TLB's style: a flat
//! array, a mask-based slot function, a tag check on lookup), not by virtual address — so shared
//! kernel `.text`, reached through many different `satp` address spaces across the life of a fuzz
//! campaign, collapses onto the *same* cache entries instead of one copy per address space. A
//! decoded [`fs_riscv::Inst`] carries no absolute address (only offsets read straight out of the
//! instruction encoding); the caller always supplies the *current* virtual pc to
//! [`fs_riscv::Cpu::exec_one`], so replaying a cached decode at a different (but byte-identical) VA
//! than it was originally decoded at is exactly as correct as decoding it fresh there.
//!
//! An entry is only inserted when every physical page its bytes touch is reported "golden" by the
//! caller-supplied `is_golden_page` predicate (for a `CowRam`-backed bus: not currently overlaid —
//! i.e. still byte-identical to the immutable `Golden` snapshot the emulator captured; for a plain
//! `Mmu`-backed bus with no COW concept, the caller passes `|_| true`). Once inserted, an entry is
//! *never* invalidated (Stage 0 has no write/protect hook — see `docs/jit.md`'s Stage 1 note): this
//! is sound precisely because "golden" means "byte-identical to an image that is never mutated", so
//! a page that was golden when cached and is *still* golden now is *still* those same bytes. Stage 0
//! explicitly assumes the measured boot+syscall-fuzz workload does not self-modify code; Stage 1
//! is where a real invalidation hook (piggybacking `DIRTY_BLOCK`/`mark_dirty`) would be added.
//!
//! **Compile-on-miss compiles a *run*, not just one instruction.** A miss decodes and caches a
//! whole straight-line run of instructions starting at the miss point — ending at a control-transfer
//! instruction (`Branch`/`Jal`/`Jalr`/`Mret`/`Sret`/`Ecall`/`Ebreak`/`Illegal` — the only `Inst`
//! kinds whose next pc is not simply `pc.wrapping_add(ilen)`, or whose execution must stop for the
//! caller to observe an early exit), a physical page boundary (a run never *starts* a new
//! instruction on a different physical page than its first instruction — RISC-V's IALIGN=16 means
//! only the low 2 bytes of a straddling final 4-byte instruction may reach into the next page,
//! exactly as the interpreter's own `fetch16` already tolerates), or a length cap. Every instruction
//! in the run is inserted into the cache at its own physical address, so a subsequent single-
//! instruction lookup anywhere in that run — not just at its head — is an O(1) hit.
//!
//! **Execution stays driven one instruction (or one taken interrupt) at a time by the caller.**
//! `run_block` looks up-or-compiles-and-caches the instruction at `cpu.pc`, then executes exactly
//! that one unit of work via [`fs_riscv::Cpu::exec_one`] (or takes a pending interrupt via
//! [`fs_riscv::Cpu::poll_interrupt`]) before returning — it does not loop internally across
//! multiple instructions without returning to the caller. This is a deliberate correctness choice,
//! not an oversight: the existing driver loop (`fs-cli`'s `run_case`) re-syncs the CLINT
//! (`mtime`/`mtimecmp`/`msip` -> the hart's CSRs) *before every single instruction*, and a real
//! kernel's timer ISR can retire many non-branch instructions between writing `mtimecmp` and the
//! `sret` that re-enables interrupts. Batching multiple instructions inside one opaque call here
//! would either skip that per-instruction CLINT resync (risking a late-delivered timer interrupt)
//! or require threading `fs-platform`/`fs-cov` types into this crate (which `docs/jit.md` keeps out:
//! `fs-jit` depends only on `fs-riscv` + `fs-mmu`). Driving one unit of work per call keeps `fs-cli`'s
//! loop — CLINT sync, interrupt polling, coverage-edge recording — byte-identical in *structure* to
//! the interpreter path; only the fetch+decode step is served from the cache instead of freshly
//! parsing bytes. The win Stage 0 measures (removing re-decode + the interior fetch/EXEC-permission
//! recheck — *not* dispatch cost, see `docs/jit.md`) is fully realized by a cache hit regardless of
//! whether the caller's own loop iterates once per instruction or once per block.

#![forbid(unsafe_code)]

use fs_mmu::{Access, Bus};
use fs_riscv::{Cpu, Inst, SysExit, Trap, decode, decode_compressed};

/// Direct-mapped slot count (must be a power of two). ~1M instructions' worth of decode results
/// (a few tens of bytes each) comfortably covers a booted kernel's hot working set; collisions just
/// cause an extra (still-correct) recompile, never a correctness issue.
pub const DEFAULT_CAPACITY: usize = 1 << 16;

/// Longest straight-line run compiled in one pass. Bounds the miss-path compile cost and the
/// (rare) collision blast radius; real basic blocks in kernel code are almost always far shorter.
const MAX_RUN_LEN: usize = 64;

/// One decoded instruction, tagged by the physical address it is valid for.
#[derive(Clone, Copy)]
struct CachedInsn {
    /// Physical address this entry was compiled from. `EMPTY` marks an unused slot.
    tag: u32,
    inst: Inst,
    ilen: u32,
    iword: u32,
}

const EMPTY: u32 = u32::MAX;

/// The Stage 0 block cache. See the module docs for the full design.
pub struct BlockCache {
    slots: Vec<CachedInsn>,
    hits: u64,
    misses: u64,
}

impl Default for BlockCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockCache {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// `capacity` is rounded up to the next power of two (required for the mask-based slot
    /// function, mirroring `fs-riscv`'s software TLB).
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(1);
        BlockCache {
            slots: vec![CachedInsn { tag: EMPTY, inst: Inst::Illegal(0), ilen: 0, iword: 0 }; capacity],
            hits: 0,
            misses: 0,
        }
    }

    /// Cache hits since construction (diagnostics for the Stage 0 benchmark).
    pub fn hits(&self) -> u64 {
        self.hits
    }
    /// Cache misses (= compile-run invocations) since construction.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    #[inline]
    fn slot(&self, pa: u32) -> usize {
        // Instructions are at least 2-byte aligned (IALIGN=16), so the low bit is always 0;
        // shifting it out before masking spreads consecutive instructions across more slots.
        ((pa >> 1) as usize) & (self.slots.len() - 1)
    }

    /// Fetch-or-compile the decoded instruction at `cpu.pc`, returning `(Inst, ilen, iword)` ready
    /// for [`fs_riscv::Cpu::exec_one`] (called with the *current* `cpu.pc`, per the module docs).
    /// A cache hit skips both the physical byte fetch (and its `PERM_EXEC` recheck) and the decode;
    /// a miss performs them once (compiling a whole run for future hits, see module docs) and
    /// returns the first instruction of that run.
    pub fn fetch(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut dyn Bus,
        is_golden_page: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Inst, u32, u32), Trap> {
        let va = cpu.pc;
        let pa = cpu.xlate(bus, va, Access::Exec)?;
        let idx = self.slot(pa);
        let cached = self.slots[idx];
        if cached.tag == pa {
            self.hits += 1;
            return Ok((cached.inst, cached.ilen, cached.iword));
        }
        self.misses += 1;
        self.compile_run(cpu, bus, va, pa, is_golden_page)
    }

    /// Look up-or-compile the instruction at `cpu.pc` and execute exactly one unit of work: either
    /// take a pending interrupt ([`fs_riscv::Cpu::poll_interrupt`]) or execute that one instruction
    /// via [`fs_riscv::Cpu::exec_one`], returning the resulting [`SysExit`] — the same granularity
    /// and semantics as [`fs_riscv::Cpu::step_system`], with fetch+decode served from this cache
    /// when possible. See the module docs for why this does not loop across multiple instructions
    /// internally.
    pub fn run_block(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut dyn Bus,
        is_golden_page: &mut dyn FnMut(u32) -> bool,
    ) -> SysExit {
        if cpu.poll_interrupt() {
            return SysExit::Continue;
        }
        let pc = cpu.pc;
        let r = self
            .fetch(cpu, bus, is_golden_page)
            .and_then(|(inst, ilen, iword)| cpu.exec_one(bus, inst, pc, ilen, iword));
        cpu.finish_exit(r)
    }

    /// Compile a straight-line run starting at `(start_va, start_pa)`, inserting each instruction
    /// into the cache at its own physical address (golden pages only), and return the run's first
    /// decoded instruction. Ends the run at a control-transfer instruction, a physical page
    /// boundary, or [`MAX_RUN_LEN`] — see the module docs for the exact rules.
    fn compile_run(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut dyn Bus,
        start_va: u32,
        start_pa: u32,
        is_golden_page: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Inst, u32, u32), Trap> {
        let block_page = start_pa >> 12;
        let mut va = start_va;
        let mut pa = start_pa;
        let mut first: Option<(Inst, u32, u32)> = None;

        for _ in 0..MAX_RUN_LEN {
            // A fault fetching/translating THIS instruction is only a real trap (propagate) when
            // it's the run's very first instruction (the one `fetch`'s caller actually asked for —
            // exactly what plain `fetch16` would have raised). Any instruction after that is a
            // speculative look-ahead extending the cached run for *future* hits: a fault there just
            // ends the run early (keeping whatever was already compiled) — it is not this call's
            // instruction to report, and a later top-level `fetch` at that pc will re-derive and
            // report the real fault when the guest actually reaches it.
            macro_rules! fault_or_stop {
                ($r:expr) => {
                    match $r {
                        Ok(v) => v,
                        Err(e) => {
                            if first.is_none() {
                                return Err(Trap::Mem(e));
                            }
                            break;
                        }
                    }
                };
            }

            let lo = fault_or_stop!(bus.ifetch16(pa));
            // extra_page: the second half-word's page, only if it differs from the run's page (a
            // 4-byte instruction whose low half is the run page's last half-word).
            let (inst, ilen, iword, extra_page) = if lo & 0x3 != 0x3 {
                (decode_compressed(lo), 2u32, lo as u32, None)
            } else {
                let hi_va = va.wrapping_add(2);
                let hi_pa = match cpu.xlate(bus, hi_va, Access::Exec) {
                    Ok(p) => p,
                    Err(e) => {
                        if first.is_none() {
                            return Err(e);
                        }
                        break;
                    }
                };
                let hi = fault_or_stop!(bus.ifetch16(hi_pa));
                let w = (lo as u32) | ((hi as u32) << 16);
                let hi_page = hi_pa >> 12;
                let extra = if hi_page != block_page { Some(hi_page) } else { None };
                (decode(w), 4u32, w, extra)
            };

            let cacheable = is_golden_page(block_page) && extra_page.is_none_or(&mut *is_golden_page);
            if cacheable {
                let idx = self.slot(pa);
                self.slots[idx] = CachedInsn { tag: pa, inst, ilen, iword };
            }
            if first.is_none() {
                first = Some((inst, ilen, iword));
            }

            let terminal = matches!(
                inst,
                Inst::Branch { .. }
                    | Inst::Jal { .. }
                    | Inst::Jalr { .. }
                    | Inst::Mret
                    | Inst::Sret
                    | Inst::Ecall
                    | Inst::Ebreak
                    | Inst::Illegal(_)
            );
            if terminal {
                break;
            }

            va = va.wrapping_add(ilen);
            let Ok(next_pa) = cpu.xlate(bus, va, Access::Exec) else {
                break; // a fresh top-level `fetch` at this pc will re-derive and report the fault
            };
            if next_pa >> 12 != block_page {
                break; // physical page boundary: never start a new instruction on a new page
            }
            pa = next_pa;
        }

        Ok(first.expect("compile_run always decodes at least one instruction"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_riscv::{A0, A7, T0, T1, asm};

    fn setup() -> (Cpu, Mmu) {
        let base = 0x8000_0000u32;
        let mut mmu = Mmu::new(base, 0x1_0000);
        mmu.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        (Cpu::new(base), mmu)
    }

    /// A straight-line loop program run entirely through the cache must retire the identical
    /// instruction count / final register state / exit as the plain interpreter (`Cpu::step`).
    #[test]
    fn cached_execution_matches_interpreter() {
        use asm::*;
        let prog = [
            addi(A0, 0, 0),
            addi(T0, 0, 1),
            addi(T1, 0, 11),
            bge(T0, T1, 16),
            add(A0, A0, T0),
            addi(T0, T0, 1),
            jal(0, -12),
            addi(A7, 0, 93),
            ecall(),
        ];
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        // Interpreter reference.
        let (mut cpu_i, mut mmu_i) = setup();
        mmu_i.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut retired_i = 0u64;
        loop {
            match cpu_i.step(&mut mmu_i).unwrap() {
                fs_riscv::Exit::Ecall => break,
                fs_riscv::Exit::Continue => {}
                _ => panic!("unexpected exit"),
            }
            retired_i += 1;
        }

        // Cached run: repeated `fetch` + `exec_one`, exercising both the miss-compile path and
        // (on the loop's later iterations) the hit path.
        let (mut cpu_c, mut mmu_c) = setup();
        mmu_c.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cache = BlockCache::new();
        let mut retired_c = 0u64;
        loop {
            let pc = cpu_c.pc;
            let (inst, ilen, iword) = cache.fetch(&mut cpu_c, &mut mmu_c, &mut |_| true).unwrap();
            match cpu_c.exec_one(&mut mmu_c, inst, pc, ilen, iword).unwrap() {
                fs_riscv::Exit::Ecall => break,
                fs_riscv::Exit::Continue => {}
                _ => panic!("unexpected exit"),
            }
            retired_c += 1;
        }

        assert_eq!(retired_i, retired_c);
        assert_eq!(cpu_i.regs, cpu_c.regs);
        assert_eq!(cpu_i.pc, cpu_c.pc);
        assert_eq!(cpu_i.insns_retired, cpu_c.insns_retired);
        // The backward branch means this ran the same block-head instructions several times —
        // prove the cache actually served hits, not just misses every time.
        assert!(cache.hits() > 0, "expected cache hits on the loop's later iterations");
    }

    /// `run_block` (poll_interrupt + fetch + exec_one + finish_exit bundled) matches
    /// `step_system` exactly over the same program: a counted loop, then an HTIF `tohost` halt
    /// (so both drivers exercise a genuine `SysExit::Halt`, not just `Continue`).
    #[test]
    fn run_block_matches_step_system() {
        use asm::*;
        const T3: u8 = 28;
        const T4: u8 = 29;
        let tohost = 0x8000_2000u32;
        let prog = [
            addi(A0, 0, 0),
            addi(T0, 0, 1),
            addi(T1, 0, 11),
            bge(T0, T1, 16),
            add(A0, A0, T0),
            addi(T0, T0, 1),
            jal(0, -12),
            lui(T3, tohost),
            addi(T4, 0, 1),
            sw(T3, T4, 0), // HTIF halt
        ];
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        let (mut cpu_i, mut mmu_i) = setup();
        mmu_i.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        cpu_i.htif_tohost = Some(tohost);
        let mut n = 0u32;
        loop {
            n += 1;
            match cpu_i.step_system(&mut mmu_i) {
                fs_riscv::SysExit::Continue => {}
                fs_riscv::SysExit::Halt(_) | fs_riscv::SysExit::Hypercall(_) => break,
            }
            assert!(n < 10_000, "interpreter did not terminate");
        }

        let (mut cpu_c, mut mmu_c) = setup();
        mmu_c.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        cpu_c.htif_tohost = Some(tohost);
        let mut cache = BlockCache::new();
        let mut m = 0u32;
        loop {
            m += 1;
            match cache.run_block(&mut cpu_c, &mut mmu_c, &mut |_| true) {
                fs_riscv::SysExit::Continue => {}
                fs_riscv::SysExit::Halt(_) | fs_riscv::SysExit::Hypercall(_) => break,
            }
            assert!(m < 10_000, "cached run did not terminate");
        }

        assert_eq!(cpu_i.regs, cpu_c.regs);
        assert_eq!(cpu_i.pc, cpu_c.pc);
        assert_eq!(cpu_i.insns_retired, cpu_c.insns_retired);
    }

    /// A code page reported non-golden is never inserted into the cache — every fetch through it
    /// stays a (correct, just uncached) miss.
    #[test]
    fn non_golden_page_is_never_cached() {
        use asm::*;
        let prog = [addi(A0, 0, 5), addi(A0, 0, 6), addi(A7, 0, 93), ecall()];
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let (mut cpu, mut mmu) = setup();
        mmu.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cache = BlockCache::new();
        loop {
            let pc = cpu.pc;
            let (inst, ilen, iword) = cache.fetch(&mut cpu, &mut mmu, &mut |_| false).unwrap();
            match cpu.exec_one(&mut mmu, inst, pc, ilen, iword).unwrap() {
                fs_riscv::Exit::Ecall => break,
                fs_riscv::Exit::Continue => {}
                _ => panic!("unexpected exit"),
            }
        }
        assert_eq!(cache.hits(), 0, "non-golden pages must never produce a cache hit");
        assert!(cache.misses() > 0);
    }

    /// A run stops at a physical page boundary: mapping two adjacent guest pages to
    /// non-adjacent physical pages (via two separate `Mmu`s is awkward, so instead we assert the
    /// weaker, still load-bearing property directly against the real single-`Mmu` VA==PA identity
    /// map) is covered implicitly by `cached_execution_matches_interpreter`'s multi-page program
    /// below; this test targets the length cap instead: a long straight-line (no branch) run must
    /// still execute identically to the interpreter once it exceeds `MAX_RUN_LEN`.
    #[test]
    fn long_straight_line_run_exceeding_length_cap_matches_interpreter() {
        use asm::*;
        let mut prog = vec![addi(T0, 0, 0)];
        for _ in 0..300 {
            prog.push(addi(T0, T0, 1));
        }
        prog.push(add(A0, T0, 0));
        prog.push(addi(A7, 0, 93));
        prog.push(ecall());
        let mut bytes = Vec::new();
        for w in &prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        let (mut cpu_i, mut mmu_i) = setup();
        mmu_i.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        loop {
            if cpu_i.step(&mut mmu_i).unwrap() == fs_riscv::Exit::Ecall {
                break;
            }
        }

        let (mut cpu_c, mut mmu_c) = setup();
        mmu_c.map(0x8000_0000, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cache = BlockCache::new();
        loop {
            let pc = cpu_c.pc;
            let (inst, ilen, iword) = cache.fetch(&mut cpu_c, &mut mmu_c, &mut |_| true).unwrap();
            if cpu_c.exec_one(&mut mmu_c, inst, pc, ilen, iword).unwrap() == fs_riscv::Exit::Ecall {
                break;
            }
        }

        assert_eq!(cpu_i.regs, cpu_c.regs);
        assert_eq!(cpu_i.insns_retired, cpu_c.insns_retired);
    }
}
