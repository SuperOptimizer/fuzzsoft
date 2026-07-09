//! Phase 1+2 of `docs/jit-scalar-design.md`: a chained ALU/branch/Load/Store native compiler,
//! memory-resident regs, admission-guarded. See the design doc in full for the rationale.
//!
//! **Scope.** A compiled chain is a straight-line run of `Lui`/`Auipc`/`OpImm`/`Op`/`Fence`/`Load`/
//! `Store`, optionally terminated by exactly one `Branch`/`Jal`/`Jalr` resolved with a branchless
//! `cmov` (no *ahead-of-time-target* jump — see the Phase 2 note below on the one kind of jump
//! this module *does* emit). Every other `Inst` kind (`Mul`, `LrW`, `ScW`, `AmoW`, `Ecall`,
//! `Ebreak`, `Csr`, `Mret`, `Sret`, `Wfi`, `SfenceVma`, `Illegal`) simply ends the chain *before*
//! itself — that instruction is never compiled, and the native code, having retired everything
//! before it, returns with `cpu.pc` already pointing at it so the ordinary interpreter
//! (`Cpu::exec_one`, via the embedded Stage 0 [`crate::BlockCache`]) handles it exactly as today.
//!
//! **Phase 1 (ALU/branch) vs Phase 2 (Load/Store): why a chain isn't uniformly "can't fail".**
//! None of `Lui`/`Auipc`/`OpImm`/`Op`/`Fence`/`Branch`/`Jal`/`Jalr` can return `Err` from
//! [`fs_riscv::Cpu::exec_one`], raise `Exit::Halt`, or need `Cpu::finish_exit`'s trap-vectoring —
//! so a chain built from *only* those never needs to report anything but the plain `Continue` tag.
//! `Load`/`Store` break that: a translation/permission fault is a real, data-dependent possibility
//! on every access, and a `Store` to HTIF `tohost` can halt the case. Each Load/Store call-out
//! (`sys.rs`'s `jit_load_*`/`jit_store_*` shims, wrapping the byte-for-byte-shared
//! `fs_riscv::load_impl`/`store_impl`) returns a packed `u64` — see `sys.rs`'s `TAG_TRAP`/
//! `TAG_HALT`/`TAG_REPOLL` doc for the exact bit layout — and `emit_step`'s Load/Store arms contain
//! the only conditional control flow this module ever emits (see "The one kind of jump" below).
//!
//! **Why the chain's entry pc must be read at runtime, not baked in as a compile-time constant.**
//! `ChainCache` is PA-keyed (mirroring [`crate::BlockCache`]'s Stage 0 rationale): the same
//! physical page of kernel `.text`, reached through different `satp` address spaces across a fuzz
//! campaign, can be executed at different virtual addresses. A compiled chain's `Auipc`/`Jal`
//! target and its `Branch`/`Jal`/`Jalr` fallthrough/link addresses are all `entry_pc +
//! static_offset [+ imm]` — `static_offset` (and `imm`) are compile-time constants (baked into a
//! `lea`'s disp32), but `entry_pc` (the *current* VA, i.e. `cpu.pc` when the chain is dispatched)
//! is read fresh into a register (`R8`) at the top of every call, so the very same compiled bytes
//! are correct no matter which VA this physical page happens to be mapped at this time.
//!
//! **The one kind of jump this module emits (Phase 2's `emit_skip`).** A Load/Store call-out's
//! outcome is checked with a single `test rax,rax` (after the call), and the RARE (trap/halt/
//! repoll) path is placed *inline*, guarded by a forward `Jcc` that skips over it in the common
//! (successful, non-CLINT) case. Crucially this needs **no label table or backpatching**: the rare
//! block is built into its own temporary [`Asm`] buffer first, so its exact length is known before
//! the `Jcc`'s `rel32` is emitted — see [`emit_skip`]. This is a narrower, purpose-built mechanism
//! than a general jump-target/relocation system, deliberately: every jump this module ever emits
//! has exactly one, immediately-following, statically-known-length target.
//!
//! **Register convention** (see `emit.rs`'s doc comment for why no SIB byte is ever needed):
//! `RDI` = cpu pointer (base for every `[rdi+disp32]` memory operand); `RSI`/`RDX` = the
//! decomposed `&mut dyn Bus` fat pointer (Phase 2: forwarded into every Load/Store call-out, per
//! `sys.rs`'s `JitFn` doc); `R8` = this call's entry pc, loaded once. **All four** of
//! `RDI`/`RSI`/`RDX`/`R8` are caller-saved per the SysV ABI — a real `call` is free to clobber any
//! of them (Phase 1 had no `call` at all, so this wasn't yet a concern; an early Phase 2 bug
//! protected only `R8` and segfaulted the moment a chain's `RDI` got clobbered mid-chain by a
//! Load/Store call-out, since every subsequent memory operand dereferences whatever garbage `RDI`
//! is left holding) — so [`emit_call_preserving_regs`] `push`es all four (plus one alignment-
//! padding register, `R9`) immediately before every Load/Store `call` and `pop`s them back
//! immediately after, restoring exactly the values every later chain instruction (and any
//! subsequent Load/Store call-out) needs. `RAX`/`RCX` = the two-operand scratch pair for every ALU
//! op (`RAX`=lhs, `RCX`=rhs — also why `RCX` is a natural choice for `Op`'s register-count shifts,
//! which x86 hardwires to `CL`) and, for Load/Store, `RCX` doubles as the call-out's `va` argument
//! (4th SysV integer arg, set up BEFORE the protecting pushes since it's freshly computed each
//! time, not carried across the call) and `RAX` receives its packed `u64` return (deliberately
//! left unprotected by `emit_call_preserving_regs` — it's the one register the caller WANTS
//! clobbered, with the call's result); `R9`/`R10` = branch-resolution scratch (`setcc`/`cmovcc`
//! targets), used only within a chain's single terminal step so a preceding Load/Store `call` can
//! never observe them live (this is also why `R9` is a safe, meaningless-to-preserve choice for
//! `emit_call_preserving_regs`'s alignment-padding push); `R11` = scratch for the call-out's
//! absolute address (`movabs`+`call`, never a `rel32` direct call — the shim's fixed address can be
//! arbitrarily far from the mmap'd arena). A `Store`'s `val` argument (5th SysV integer arg) is
//! loaded directly into `R8` *after* `emit_call_preserving_regs` has already pushed the real `R8` —
//! safe precisely because it's restored by the matching pop right after the call returns.

#![forbid(unsafe_code)]

use crate::emit::{Alu2, Asm, Cc, Reg};
use crate::sys::{self, Arena, TAG_HALT, TAG_REPOLL, TAG_TRAP};
use crate::BlockCache;
use fs_mmu::{Access, Bus};
use fs_riscv::{AluOp, BranchOp, Cpu, Inst, SysExit, decode, decode_compressed};

/// Longest straight-line chain compiled in one pass — bounds compile cost and code size, mirroring
/// [`crate::MAX_RUN_LEN`]-style caps elsewhere in this crate. Real kernel basic blocks are almost
/// always far shorter.
const MAX_CHAIN_LEN: usize = 64;

/// Direct-mapped chain-metadata slot count. Matches [`crate::DEFAULT_CAPACITY`]'s order of
/// magnitude (Stage 0's own per-instruction cache) rather than a smaller guess: the
/// boot+syscall-fuzz benchmark workload touches a very large number of distinct chain-head
/// addresses, and a too-small direct-mapped table thrashes (collision-evicts a still-hot chain,
/// forcing a real recompile) long before the arena itself would fill.
const DEFAULT_CHAIN_CAPACITY: usize = 1 << 20;

const EMPTY: u32 = u32::MAX;

/// `Cpu`'s field offsets, computed **once** via `offset_of!` (never hand-derived) so a future
/// field reorder in `fs-riscv` is automatically picked up here — there is nothing to silently
/// drift, unlike a hand-copied constant. All three fields are `pub` on `Cpu`, so this works across
/// the crate boundary without any `fs-riscv` change.
const REGS_OFF: i32 = std::mem::offset_of!(Cpu, regs) as i32;
const PC_OFF: i32 = std::mem::offset_of!(Cpu, pc) as i32;
const INSNS_RETIRED_OFF: i32 = std::mem::offset_of!(Cpu, insns_retired) as i32;
/// Phase 3 (`docs/jit-scalar-design.md`) fast-path instrumentation counters — bumped directly by
/// emitted code, mirroring `INSNS_RETIRED_OFF`'s bump (see `Cpu::fast_path_hits`'s doc comment).
const FAST_PATH_HITS_OFF: i32 = std::mem::offset_of!(Cpu, fast_path_hits) as i32;
const FAST_PATH_BAILS_OFF: i32 = std::mem::offset_of!(Cpu, fast_path_bails) as i32;

#[inline]
fn reg_off(i: u8) -> i32 {
    REGS_OFF + 4 * i as i32
}

/// One decoded chain step: the instruction plus its offset (in bytes) from the chain's entry pc —
/// the compile-time-known half of every pc-relative computation (see the module doc).
#[derive(Clone, Copy)]
struct Step {
    inst: Inst,
    ilen: u32,
    static_offset: u32,
}

/// A compiled chain's metadata, direct-mapped by the physical address it starts at.
#[derive(Clone, Copy)]
struct ChainSlot {
    tag: u32,
    /// Byte offset into the arena. Meaningless when `static_len == 0` (nothing compileable here —
    /// still a genuine, cacheable fact: don't retry decoding this pc as a chain head again).
    code_off: u32,
    /// Number of guest instructions this chain retires when run to completion — the admission
    /// guard's `chain.static_len` and the histogram bucket key.
    static_len: u32,
    /// `(byte offset from the chain's entry pc, ilen)` of the terminal `Branch`/`Jal`/`Jalr`, if
    /// the chain ends in one. `None` for a chain that ends because the next instruction wasn't
    /// compileable (pure ALU fallthrough — never a real control transfer, never an edge).
    ///
    /// Used only for [`ChainCache::take_last_edge`]'s coverage-edge bookkeeping — deliberately
    /// **not** a runtime "was the branch actually taken" signal (which the native code could
    /// easily compute and report back, e.g. via a return-value bit). The existing per-instruction
    /// interpreter driver's own edge check (`cur != prev.wrapping_add(ilen)`, `fs-cli`'s
    /// `run_case`/`run_case_jit`) is blind to a degenerate "taken but the target happens to equal
    /// the fallthrough address" branch/jal/jalr (e.g. `beq x1,x2,+4` when `x1==x2`): it would
    /// report "no edge" purely because the *address* didn't change, even though a real transfer
    /// happened. Matching that exact (address-based, not semantics-based) heuristic — bug-for-bug
    /// — rather than a more "correct" runtime-taken signal is the deliberate choice here, so
    /// `--jit-chain`'s coverage bitmap stays comparable to `--jit`/the plain interpreter's, instead
    /// of silently becoming a superset that grows the corpus differently. See
    /// `docs/jit-scalar-design.md`'s Phase 2 "coverage-edge parity" discussion, which this
    /// generalizes to Phase 1's multi-instruction-chain case.
    terminal: Option<(u32, u32)>,
}

/// Coarse chain-length histogram buckets for the benchmark report (`docs/jit-scalar-design.md`'s
/// "hit/miss/chain-length histograms").
pub const CHAIN_LEN_BUCKETS: [(u32, u32); 7] =
    [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 16), (17, u32::MAX)];

/// The Phase 1 chain cache. Wraps a Stage 0 [`BlockCache`] (reused, untouched) for: (a) the
/// sanitizer-gated fallback (KMSAN/CMPLOG/UBSAN route around the compiled path entirely), (b) the
/// admission-guard-failed / zero-length-chain single-step fallback (both are exactly "run one
/// instruction through the ordinary per-instruction cache+interpreter path", which is precisely
/// what `BlockCache::run_block`'s per-instruction building blocks (`fetch`+`exec_one`) already do).
pub struct ChainCache {
    interp: BlockCache,
    slots: Vec<ChainSlot>,
    arena: Arena,
    /// Native chain dispatches.
    chain_hits: u64,
    /// Chain compile attempts (cache misses at the chain-metadata level).
    chain_misses: u64,
    /// Admission-guard failures or zero-length-chain single-steps (Stage-0-style fallback calls).
    fallbacks: u64,
    /// Compiles that found a non-empty, cacheable chain but the arena was full — see
    /// `lookup_or_compile`'s doc comment on why these are still cached as `static_len: 0`.
    arena_full: u64,
    /// Histogram of `static_len` for every natively-dispatched chain, indexed by
    /// [`CHAIN_LEN_BUCKETS`].
    len_hist: [u64; CHAIN_LEN_BUCKETS.len()],
    /// The most recent `run_block` call's control-flow edge, if it produced one — see
    /// [`ChainCache::take_last_edge`].
    last_edge: Option<(u32, u32)>,
}

impl ChainCache {
    pub fn new() -> Self {
        Self::with_capacity(crate::DEFAULT_CAPACITY, DEFAULT_CHAIN_CAPACITY)
    }

    pub fn with_capacity(interp_capacity: usize, chain_capacity: usize) -> Self {
        let chain_capacity = chain_capacity.next_power_of_two().max(1);
        ChainCache {
            interp: BlockCache::with_capacity(interp_capacity),
            slots: vec![
                ChainSlot { tag: EMPTY, code_off: 0, static_len: 0, terminal: None };
                chain_capacity
            ],
            arena: Arena::new().expect("mmap for JIT arena failed"),
            chain_hits: 0,
            chain_misses: 0,
            fallbacks: 0,
            arena_full: 0,
            len_hist: [0; CHAIN_LEN_BUCKETS.len()],
            last_edge: None,
        }
    }

    /// The control-flow edge (if any) produced by the most recent `run_block` call, consuming it
    /// (like `Cpu::cmplog_take`'s drain style) — mirrors what the plain interpreter's own
    /// `cur != prev.wrapping_add(ilen)` check would have recorded, generalized to a chain that may
    /// have retired many instructions in one call. `None` means the call was pure ALU fallthrough
    /// (however many instructions it batched) — not a real control-transfer, so the caller should
    /// not record a coverage edge for it, unlike the naive "compare final pc against the call's
    /// entry pc" heuristic Stage 0's single-instruction granularity got away with (which would
    /// misfire on every multi-instruction chain — `docs/jit-scalar-design.md`'s "coverage-edge
    /// parity" concern, addressed here rather than deferred).
    pub fn take_last_edge(&mut self) -> Option<(u32, u32)> {
        self.last_edge.take()
    }

    pub fn chain_hits(&self) -> u64 {
        self.chain_hits
    }
    pub fn chain_misses(&self) -> u64 {
        self.chain_misses
    }
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks
    }
    /// Chains that would have compiled but the (fixed-size, Phase 1) arena was already full —
    /// diagnostic for whether [`crate::sys`]'s arena capacity is undersized for a given workload.
    pub fn arena_full_count(&self) -> u64 {
        self.arena_full
    }
    pub fn len_histogram(&self) -> &[u64; CHAIN_LEN_BUCKETS.len()] {
        &self.len_hist
    }
    /// The embedded Stage 0 cache's own hit/miss counters (per-instruction decode cache, shared by
    /// the sanitizer-gated and admission-guard-failed fallback paths).
    pub fn interp_hits(&self) -> u64 {
        self.interp.hits()
    }
    pub fn interp_misses(&self) -> u64 {
        self.interp.misses()
    }
    /// Bytes of the (fixed-size, Phase 1) executable arena committed so far — diagnostic for the
    /// benchmark report (how close a run came to exhausting it).
    pub fn arena_bytes_used(&self) -> usize {
        self.arena.used()
    }
    /// Total fixed arena capacity — diagnostic for the benchmark report.
    pub fn arena_capacity(&self) -> usize {
        self.arena.capacity()
    }

    #[inline]
    fn slot(&self, pa: u32) -> usize {
        ((pa >> 1) as usize) & (self.slots.len() - 1)
    }

    fn record_len(&mut self, static_len: u32) {
        for (i, &(lo, hi)) in CHAIN_LEN_BUCKETS.iter().enumerate() {
            if static_len >= lo && static_len <= hi {
                self.len_hist[i] += 1;
                return;
            }
        }
    }

    /// Look up-or-compile-and-cache the chain starting at (physical) `pa`/(virtual) `va`, or
    /// single-step one instruction and return `None` if nothing was (or could be) compiled there.
    /// This is a purely speculative *decode* pass ([`decode_chain`] never executes anything, so it
    /// never itself needs to surface a `Trap`) — it never needs a `Result` return of its own: any
    /// real fault at `va`, whether at COMPILE time (this function just stops the speculative
    /// decode early) or at RUNTIME inside a Load/Store call-out (Phase 2's `TAG_TRAP`, handled by
    /// `run_block` after the compiled chain returns), is re-derived/reported for real by the
    /// fallback path's own `fetch`+`exec_one` call or by `run_block`'s trap-vectoring, exactly as
    /// it would be without this cache at all.
    fn lookup_or_compile(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut dyn Bus,
        va: u32,
        pa: u32,
        is_golden_page: &mut dyn FnMut(u32) -> bool,
    ) -> Option<ChainSlot> {
        let idx = self.slot(pa);
        if self.slots[idx].tag == pa {
            return Some(self.slots[idx]);
        }
        self.chain_misses += 1;

        let block_page = pa >> 12;
        if !is_golden_page(block_page) {
            // Never cache/compile over a non-golden page (Stage 0's exact rule) — every call here
            // re-attempts, which is correct (if a little wasteful) and matches
            // `BlockCache::compile_run`'s own non-golden handling.
            return None;
        }

        let steps = decode_chain(cpu, bus, va, pa, block_page, is_golden_page);
        let slot = if steps.is_empty() {
            ChainSlot { tag: pa, code_off: 0, static_len: 0, terminal: None }
        } else {
            let terminal = steps
                .last()
                .filter(|s| is_terminal(&s.inst))
                .map(|s| (s.static_offset, s.ilen));
            let code = codegen(&steps);
            match self.arena.write(&code).expect("arena write failed") {
                Some(off) => ChainSlot { tag: pa, code_off: off, static_len: steps.len() as u32, terminal },
                None => {
                    // Arena full: cache a `static_len: 0` ("nothing compiled here") marker just
                    // like the `steps.is_empty()` case above — critically, still INSERTED into
                    // `self.slots`, not merely returned for this one call. Without this, every
                    // future dispatch at this pc (i.e. effectively every miss for the rest of the
                    // run, once the fixed-size arena fills) would redo this exact decode+codegen
                    // work for nothing every single time — a real, measured performance cliff (not
                    // just "a little wasteful"): the fallback path is still fully correct on its
                    // own, so there is no reason to keep re-deriving that fact. This does mean an
                    // arena-full pc is "stuck" un-compiled for the rest of the run even if the
                    // arena later had room (it never does — Phase 1 never evicts) — an accepted
                    // Phase 1 limitation (see `sys::Arena::write`'s doc comment).
                    self.arena_full += 1;
                    ChainSlot { tag: pa, code_off: 0, static_len: 0, terminal: None }
                }
            }
        };
        self.slots[idx] = slot;
        Some(slot)
    }

    /// Public contract unchanged from Stage 0's `BlockCache::run_block` (see
    /// `docs/jit-scalar-design.md`'s integration seam): one call does one "unit of work" — an
    /// interrupt take, one native chain, or (sanitizers on / admission guard failed / nothing
    /// compileable) one interpreted instruction — and returns.
    pub fn run_block(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut dyn Bus,
        is_golden_page: &mut dyn FnMut(u32) -> bool,
    ) -> SysExit {
        let entry_pc = cpu.pc;
        self.last_edge = None;

        // Sanitizer/introspection gate: checked on EVERY dispatch, never cached, so a hypothetical
        // future mid-case toggle can't reactivate a stale compiled path (design doc's mandate).
        if cpu.kmsan_enabled() || cpu.cmplog_enabled() || cpu.ubsan_enabled() {
            let exit = self.interp.run_block(cpu, bus, is_golden_page);
            // `interp` is always single-instruction (or interrupt-take) granularity, so the
            // classic Stage 0 heuristic is exactly correct here (no chain batching to confuse it).
            if exit == SysExit::Continue {
                self.last_edge = edge_if_not_fallthrough(entry_pc, cpu.pc);
            }
            return exit;
        }
        if cpu.poll_interrupt() {
            // Always a real redirect (pc now points at the trap vector).
            self.last_edge = Some((entry_pc, cpu.pc));
            return SysExit::Continue;
        }
        let pc = cpu.pc;
        let pa = match cpu.xlate(bus, pc, Access::Exec) {
            Ok(pa) => pa,
            Err(trap) => {
                let exit = cpu.finish_exit(Err(trap));
                if exit == SysExit::Continue {
                    self.last_edge = Some((entry_pc, cpu.pc)); // a fault always vectors a trap
                }
                return exit;
            }
        };
        let chain = self.lookup_or_compile(cpu, bus, pc, pa, is_golden_page);
        let static_len = chain.map_or(0, |c| c.static_len);

        // Admission guard: provably safe to run the whole chain in one native call iff no pending
        // timer deadline falls strictly inside it (`docs/jit-scalar-design.md`'s Phase 1 section).
        let budget =
            cpu.csr.mtimecmp.min(cpu.csr.stimecmp).saturating_sub(cpu.virtual_time());
        let admitted = static_len > 0 && budget >= static_len as u64;

        if !admitted {
            self.fallbacks += 1;
            let fetched = self.interp.fetch(cpu, bus, is_golden_page);
            let ilen = fetched.as_ref().map(|(_, ilen, _)| *ilen).unwrap_or(0);
            let r = fetched.and_then(|(inst, ilen, iword)| cpu.exec_one(bus, inst, pc, ilen, iword));
            let exit = cpu.finish_exit(r);
            if exit == SysExit::Continue {
                self.last_edge = edge_if_not_fallthrough_by(pc, cpu.pc, ilen);
            }
            return exit;
        }

        let slot = chain.expect("admitted implies a compiled chain");
        self.chain_hits += 1;
        self.record_len(slot.static_len);
        let (bus_data, bus_vtable) = sys::decompose_bus(bus);
        let ret = self.arena.call(slot.code_off, cpu as *mut Cpu, bus_data, bus_vtable);

        if ret & TAG_TRAP != 0 {
            // A Load/Store call-out stashed the real `Trap` in `cpu.jit_pending_trap` just before
            // returning this sentinel (`sys.rs`'s tag doc) — vector it exactly as the interpreter
            // would have propagated the same `Err(trap)` from `exec_one`.
            let trap = cpu.jit_pending_trap.take().expect("TAG_TRAP implies jit_pending_trap is set");
            let exit = cpu.finish_exit(Err(trap));
            if exit == SysExit::Continue {
                self.last_edge = Some((entry_pc, cpu.pc)); // a fault always vectors a trap
            }
            return exit;
        }
        if ret & TAG_HALT != 0 {
            // HTIF `tohost` halt, forwarded through the Store call-out exactly as `exec_one`'s
            // `Store` arm does today — `insns_retired`/`pc` were already committed by the chain
            // before it returned this tag (see `emit_step`'s `Store` arm).
            return SysExit::Halt((ret & 0xffff_ffff) as u32);
        }
        if ret & TAG_REPOLL != 0 {
            // A CLINT-range store retired but the chain stopped itself immediately (before
            // reaching any statically-known terminal `Branch`/`Jal`/`Jalr` this compiled chain may
            // have had) so the driver's per-instruction CLINT resync runs before anything else
            // executes. This early exit's own pc transition is always a plain fallthrough (a
            // `Store` never redirects control) — `slot.terminal` must NOT be consulted here, or a
            // branch/jal/jalr that was never actually reached this call would be misreported as a
            // coverage edge (`last_edge` is already `None` from the top of this function).
            return SysExit::Continue;
        }

        // Normal completion: the compiled chain ran to its full static length. Coverage edge,
        // computed the SAME (address-based, not runtime-taken-based) way the interpreter's own
        // heuristic would — see `ChainSlot::terminal`'s doc comment.
        if let Some((terminal_offset, terminal_ilen)) = slot.terminal {
            let terminal_pc = pc.wrapping_add(terminal_offset);
            self.last_edge = edge_if_not_fallthrough_by(terminal_pc, cpu.pc, terminal_ilen);
        }
        SysExit::Continue
    }
}

/// `cur` vs. the single-instruction fallthrough of an instruction at `prev` with length 2 or 4 —
/// the classic Stage 0 / plain-interpreter edge heuristic, valid whenever exactly one instruction
/// (or zero, for an interrupt-take handled elsewhere) was retired by the call being measured.
fn edge_if_not_fallthrough(prev: u32, cur: u32) -> Option<(u32, u32)> {
    (cur != prev.wrapping_add(2) && cur != prev.wrapping_add(4)).then_some((prev, cur))
}

/// Same idea but against a specific known `ilen` (used by the fallback path, which always knows
/// exactly which single instruction it just ran).
fn edge_if_not_fallthrough_by(prev: u32, cur: u32, ilen: u32) -> Option<(u32, u32)> {
    (cur != prev.wrapping_add(ilen)).then_some((prev, cur))
}

impl Default for ChainCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Is this `Inst` compileable as a non-terminal chain link (falls straight through)? Phase 2 adds
/// `Load`/`Store` to Phase 1's ALU/branch set (`docs/jit-scalar-design.md`'s Phase 2 scope) — both
/// fall straight through by `ilen` just like an ALU op in the no-trap, no-CLINT-repoll case; their
/// call-out's rare paths (trap/halt/repoll) are handled by an early `ret` from within `emit_step`
/// itself, not by ending the chain's static shape here.
fn is_alu_link(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Lui { .. }
            | Inst::Auipc { .. }
            | Inst::OpImm { .. }
            | Inst::Op { .. }
            | Inst::Fence
            | Inst::Load { .. }
            | Inst::Store { .. }
    )
}

/// Is this `Inst` a valid chain *terminator* (at most one, always last)?
fn is_terminal(inst: &Inst) -> bool {
    matches!(inst, Inst::Branch { .. } | Inst::Jal { .. } | Inst::Jalr { .. })
}

/// Decode a straight-line, compileable chain starting at `(va, pa)`. Purely advisory/planning —
/// see the module doc for why this never needs to report a `Trap`: any decode/fetch failure here
/// (even on the very first instruction) just yields a shorter (possibly empty) chain, and the
/// caller's real fault reporting always happens later, for real, via `Cpu::exec_one`.
fn decode_chain(
    cpu: &mut Cpu,
    bus: &mut dyn Bus,
    va: u32,
    pa: u32,
    block_page: u32,
    is_golden_page: &mut dyn FnMut(u32) -> bool,
) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut va = va;
    let mut pa = pa;
    let mut offset = 0u32;

    for _ in 0..MAX_CHAIN_LEN {
        let Ok(lo) = bus.ifetch16(pa) else { break };
        let (inst, ilen, extra_page) = if lo & 0x3 != 0x3 {
            (decode_compressed(lo), 2u32, None)
        } else {
            let hi_va = va.wrapping_add(2);
            let Ok(hi_pa) = cpu.xlate(bus, hi_va, Access::Exec) else { break };
            let Ok(hi) = bus.ifetch16(hi_pa) else { break };
            let w = (lo as u32) | ((hi as u32) << 16);
            let hi_page = hi_pa >> 12;
            let extra = if hi_page != block_page { Some(hi_page) } else { None };
            (decode(w), 4u32, extra)
        };

        if let Some(p) = extra_page
            && !is_golden_page(p)
        {
            break; // straddles into a non-golden page: don't trust it as stably re-executable
        }

        if is_alu_link(&inst) {
            steps.push(Step { inst, ilen, static_offset: offset });
        } else if is_terminal(&inst) {
            steps.push(Step { inst, ilen, static_offset: offset });
            break; // terminal: always the chain's last instruction
        } else {
            break; // Load/Store/Mul/... : chain ends BEFORE this instruction (not included)
        }

        if steps.len() >= MAX_CHAIN_LEN {
            break;
        }
        va = va.wrapping_add(ilen);
        offset += ilen;
        let Ok(next_pa) = cpu.xlate(bus, va, Access::Exec) else { break };
        if next_pa >> 12 != block_page {
            break; // never start a new instruction on a different physical page
        }
        pa = next_pa;
    }

    steps
}

/// Emit native code for a non-empty, already-validated chain. `steps.last()` is either a terminal
/// `Branch`/`Jal`/`Jalr` (chain ends via its own control transfer) or a plain ALU op (chain ends
/// because the next instruction wasn't compileable / a boundary was hit — the final `pc` in that
/// case is simply "right after the last compiled instruction").
fn codegen(steps: &[Step]) -> Vec<u8> {
    let mut a = Asm::new();
    // Entry pc, read once (see the module doc for why this can't be a compile-time constant).
    a.mov_r32_mem(Reg::R8, Reg::RDI, PC_OFF);

    let last = steps.len() - 1;
    for (i, step) in steps.iter().enumerate() {
        emit_step(&mut a, step, i == last);
        // Load/Store bump `insns_retired` THEMSELVES, conditionally (only on the paths where the
        // instruction actually retired — never on a trap) — see their `emit_step` arms. Every
        // other instruction kind can never fail, so the blanket bump here is exactly right for
        // them, unconditionally, matching Phase 1.
        if !matches!(step.inst, Inst::Load { .. } | Inst::Store { .. }) {
            a.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
        }
    }

    let last_step = &steps[last];
    if !is_terminal(&last_step.inst) {
        // Fell off the end of the chain without a control transfer: pc becomes "right after the
        // last compiled instruction" — a plain `lea`+store, same shape as every fallthrough above.
        let next_off = (last_step.static_offset + last_step.ilen) as i32;
        a.lea_r32_mem(Reg::RAX, Reg::R8, next_off);
        a.mov_mem_r32(Reg::RDI, PC_OFF, Reg::RAX);
    }
    a.zero(Reg::RAX); // Continue tag (0) — always; see `ChainSlot::terminal`'s doc comment for why
    // "was the branch actually taken" is deliberately NOT threaded through this return value.
    a.ret();
    a.buf
}

/// Load register `i`'s value into `dst`: `x0` is elided at compile time into `xor dst,dst` (no
/// memory access — see the design doc), matching `Cpu::rd_reg`'s semantics with zero runtime `if`.
fn load_reg(a: &mut Asm, dst: Reg, i: u8) {
    if i == 0 {
        a.zero(dst);
    } else {
        a.mov_r32_mem(dst, Reg::RDI, reg_off(i));
    }
}

/// Store `src` into register `i`: a write to `x0` is omitted entirely (no store emitted at all),
/// matching `Cpu::wr_reg`'s semantics — strictly cheaper than the interpreter's runtime check.
fn store_reg(a: &mut Asm, i: u8, src: Reg) {
    if i != 0 {
        a.mov_mem_r32(Reg::RDI, reg_off(i), src);
    }
}

fn alu2_of(op: AluOp) -> Option<Alu2> {
    match op {
        AluOp::Add => Some(Alu2::Add),
        AluOp::Sub => Some(Alu2::Sub),
        AluOp::Xor => Some(Alu2::Xor),
        AluOp::Or => Some(Alu2::Or),
        AluOp::And => Some(Alu2::And),
        AluOp::Slt | AluOp::Sltu | AluOp::Sll | AluOp::Srl | AluOp::Sra => None,
    }
}

/// Emit `dst = alu(op, lhs, rhs)` where `lhs` is already in `RAX` and, for the register-rhs forms,
/// `rhs` is already in `RCX` (required for the shift ops — x86 hardwires the shift count to `CL`).
/// The immediate-rhs forms (`OpImm`) pass `rhs_imm` instead and never touch `RCX`.
fn emit_alu_reg_result(a: &mut Asm, op: AluOp, rhs_imm: Option<u32>) {
    if let Some(op2) = alu2_of(op) {
        match rhs_imm {
            Some(imm) => a.alu_r32_imm32(op2, Reg::RAX, imm),
            None => a.alu_r32_r32(op2, Reg::RAX, Reg::RCX),
        }
        return;
    }
    match op {
        AluOp::Slt | AluOp::Sltu => {
            match rhs_imm {
                Some(imm) => a.alu_r32_imm32(Alu2::Cmp, Reg::RAX, imm),
                None => a.alu_r32_r32(Alu2::Cmp, Reg::RAX, Reg::RCX),
            }
            let cc = if op == AluOp::Slt { Cc::L } else { Cc::B };
            a.setcc(cc, Reg::R9);
            a.movzx_r32_r8(Reg::RAX, Reg::R9);
        }
        AluOp::Sll | AluOp::Srl | AluOp::Sra => {
            match rhs_imm {
                // OpImm's shamt is a compile-time-known 0..31 constant (decoded straight into
                // `imm` by `fs_riscv::decode`) — use the immediate-count form directly.
                Some(imm) => {
                    let shamt = (imm & 0x1f) as u8;
                    match op {
                        AluOp::Sll => a.shl_imm8(Reg::RAX, shamt),
                        AluOp::Srl => a.shr_imm8(Reg::RAX, shamt),
                        _ => a.sar_imm8(Reg::RAX, shamt),
                    }
                }
                // Op's shift count is a runtime register value, already loaded into RCX by the
                // caller — x86 masks it to 5 bits in hardware, matching `shamt & 31`.
                None => match op {
                    AluOp::Sll => a.shl_cl(Reg::RAX),
                    AluOp::Srl => a.shr_cl(Reg::RAX),
                    _ => a.sar_cl(Reg::RAX),
                },
            }
        }
        _ => unreachable!("alu2_of covers every other AluOp"),
    }
}

fn emit_step(a: &mut Asm, step: &Step, is_last: bool) {
    match step.inst {
        Inst::Lui { rd, imm } => {
            if rd != 0 {
                a.mov_r32_imm32(Reg::RAX, imm);
                store_reg(a, rd, Reg::RAX);
            }
        }
        Inst::Auipc { rd, imm } => {
            if rd != 0 {
                let disp = step.static_offset.wrapping_add(imm) as i32;
                a.lea_r32_mem(Reg::RAX, Reg::R8, disp);
                store_reg(a, rd, Reg::RAX);
            }
        }
        Inst::OpImm { op, rd, rs1, imm } => {
            load_reg(a, Reg::RAX, rs1);
            emit_alu_reg_result(a, op, Some(imm as u32));
            store_reg(a, rd, Reg::RAX);
        }
        Inst::Op { op, rd, rs1, rs2 } => {
            load_reg(a, Reg::RAX, rs1);
            load_reg(a, Reg::RCX, rs2);
            emit_alu_reg_result(a, op, None);
            store_reg(a, rd, Reg::RAX);
        }
        Inst::Fence => {}
        Inst::Jal { rd, imm } => {
            debug_assert!(is_last);
            let link_disp = (step.static_offset + step.ilen) as i32;
            if rd != 0 {
                a.lea_r32_mem(Reg::RAX, Reg::R8, link_disp);
                store_reg(a, rd, Reg::RAX);
            }
            let target_disp = step.static_offset.wrapping_add(imm as u32) as i32;
            a.lea_r32_mem(Reg::RCX, Reg::R8, target_disp);
            a.mov_mem_r32(Reg::RDI, PC_OFF, Reg::RCX);
        }
        Inst::Jalr { rd, rs1, imm } => {
            debug_assert!(is_last);
            load_reg(a, Reg::RCX, rs1);
            a.alu_r32_imm32(Alu2::Add, Reg::RCX, imm as u32);
            a.alu_r32_imm32(Alu2::And, Reg::RCX, 0xffff_fffe);
            if rd != 0 {
                let link_disp = (step.static_offset + step.ilen) as i32;
                a.lea_r32_mem(Reg::RAX, Reg::R8, link_disp);
                store_reg(a, rd, Reg::RAX);
            }
            a.mov_mem_r32(Reg::RDI, PC_OFF, Reg::RCX);
        }
        Inst::Branch { op, rs1, rs2, imm } => {
            debug_assert!(is_last);
            load_reg(a, Reg::RAX, rs1);
            load_reg(a, Reg::RCX, rs2);
            a.alu_r32_r32(Alu2::Cmp, Reg::RAX, Reg::RCX);
            let not_taken_disp = (step.static_offset + step.ilen) as i32;
            let taken_disp = step.static_offset.wrapping_add(imm as u32) as i32;
            a.lea_r32_mem(Reg::R9, Reg::R8, not_taken_disp);
            a.lea_r32_mem(Reg::R10, Reg::R8, taken_disp);
            let cc = match op {
                BranchOp::Eq => Cc::E,
                BranchOp::Ne => Cc::Ne,
                BranchOp::Lt => Cc::L,
                BranchOp::Ge => Cc::Ge,
                BranchOp::Ltu => Cc::B,
                BranchOp::Geu => Cc::Ae,
            };
            a.cmovcc(cc, Reg::R9, Reg::R10);
            a.mov_mem_r32(Reg::RDI, PC_OFF, Reg::R9);
        }
        Inst::Load { op, rd, rs1, imm } => {
            // va = rs1 + imm, into RCX (the shim's 4th SysV arg; RDI/RSI/RDX already hold
            // cpu/bus_data/bus_vtable, the shim's first three args, untouched).
            load_reg(a, Reg::RCX, rs1);
            a.alu_r32_imm32(Alu2::Add, Reg::RCX, imm as u32);

            // Phase 3 fast path (`docs/jit-scalar-design.md`): ask for a direct host pointer first
            // — a narrow, provably-safe check ([`fs_riscv::Cpu::fast_pa`] + [`fs_mmu::Bus::fast_ptr`],
            // see their docs) that NEVER traps by construction (any doubt at all just returns 0,
            // meaning "bail to the unchanged Phase 2 call-out below"). `RCX` (va) is preserved
            // across this call by `emit_fast_path_call_preserving_va` since the slow branch needs
            // it again unchanged.
            emit_fast_path_call_preserving_va(a, |a| {
                a.mov_r64_imm64(Reg::R11, sys::fast_load_ptr_addr(op));
                a.call_r64(Reg::R11);
            });
            a.test_r64_r64(Reg::RAX, Reg::RAX);
            emit_if_else(
                a,
                Cc::Ne, // RAX != 0 => a real host pointer (fast-path hit — never NULL, see sys.rs)
                |a| {
                    // FAST: direct sized/signed load from the host pointer (RAX) — no call-out.
                    // `RCX` (va) is no longer needed, so it doubles as the load's destination.
                    a.add_qword_mem_imm8(Reg::RDI, FAST_PATH_HITS_OFF, 1);
                    emit_sized_load_from_ptr(a, op, Reg::RCX, Reg::RAX);
                    a.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
                    store_reg(a, rd, Reg::RCX);
                },
                |a| {
                    // SLOW (bail): Phase 2's existing call-out path, byte-for-byte unchanged —
                    // `RCX` still holds `va` (restored by `emit_fast_path_call_preserving_va`'s pop).
                    a.add_qword_mem_imm8(Reg::RDI, FAST_PATH_BAILS_OFF, 1);
                    // cpu.pc = this instruction's OWN address, BEFORE the call: `fs_riscv::take_trap`
                    // reads `self.pc` as the faulting epc, and (mirroring `exec_one`'s structure,
                    // where `self.pc` is never advanced until an instruction fully commits) that
                    // must be exactly this instruction's address if the call-out reports a trap.
                    a.lea_r32_mem(Reg::RAX, Reg::R8, step.static_offset as i32);
                    a.mov_mem_r32(Reg::RDI, PC_OFF, Reg::RAX);
                    emit_call_preserving_regs(a, |a| {
                        a.mov_r64_imm64(Reg::R11, sys::load_shim_addr(op));
                        a.call_r64(Reg::R11);
                    });
                    a.test_r64_r64(Reg::RAX, Reg::RAX);
                    // Bit 63 (sign) set => TAG_TRAP: stop the chain now, passing the shim's packed
                    // value straight through as this whole compiled chain's own return value (same
                    // bit layout — see `sys.rs`'s tag doc). `pc` is already correct (just set
                    // above); `insns_retired` must NOT be bumped (a faulting instruction never
                    // retires, matching `exec_one`).
                    emit_skip(a, Cc::Ns, |rare| rare.ret());
                    // No trap: this instruction retired. Bump `insns_retired` and store the loaded
                    // value (RAX's low 32 bits — always a clean `u32` in this path, see `sys.rs`'s
                    // tag doc) into `rd` (elided for `x0`).
                    a.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
                    store_reg(a, rd, Reg::RAX);
                },
            );
        }
        Inst::Store { op, rs1, rs2, imm } => {
            load_reg(a, Reg::RCX, rs1);
            a.alu_r32_imm32(Alu2::Add, Reg::RCX, imm as u32); // va

            // Phase 3 fast path: same shape as Load's, above.
            emit_fast_path_call_preserving_va(a, |a| {
                a.mov_r64_imm64(Reg::R11, sys::fast_store_ptr_addr(op));
                a.call_r64(Reg::R11);
            });
            a.test_r64_r64(Reg::RAX, Reg::RAX);
            emit_if_else(
                a,
                Cc::Ne,
                |a| {
                    // FAST: `fast_ptr`'s contract guarantees this can only ever resolve to plain
                    // RAM (never CLINT/MMIO — see `sys.rs`'s `fast_store_ptr_common` doc), so no
                    // `store_may_assert_interrupt`/`TAG_REPOLL`-equivalent check is needed here at
                    // all: this Store structurally cannot newly assert an interrupt. `rs2`'s value
                    // is loaded fresh here (cheap — a single `mov` from the register array) rather
                    // than preserved across the fast-ptr call, since nothing needed it before now.
                    a.add_qword_mem_imm8(Reg::RDI, FAST_PATH_HITS_OFF, 1);
                    load_reg(a, Reg::RCX, rs2);
                    emit_sized_store_to_ptr(a, op, Reg::RAX, Reg::RCX);
                    a.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
                },
                |a| {
                    // SLOW (bail): Phase 2's existing call-out path, byte-for-byte unchanged.
                    a.add_qword_mem_imm8(Reg::RDI, FAST_PATH_BAILS_OFF, 1);
                    a.lea_r32_mem(Reg::RAX, Reg::R8, step.static_offset as i32);
                    a.mov_mem_r32(Reg::RDI, PC_OFF, Reg::RAX);
                    emit_call_preserving_regs(a, |a| {
                        // val (the shim's 5th SysV arg, R8) — safe: the real R8 is already saved
                        // by `emit_call_preserving_regs`, restored by its matching pop right after
                        // the call.
                        load_reg(a, Reg::R8, rs2);
                        a.mov_r64_imm64(Reg::R11, sys::store_shim_addr(op));
                        a.call_r64(Reg::R11);
                    });
                    a.test_r64_r64(Reg::RAX, Reg::RAX);
                    // Bit 63 set => trap: identical early-`ret` to the Load case above.
                    emit_skip(a, Cc::Ns, |rare| rare.ret());
                    // Not a trap. `rax==0` => plain continue (fast path, falls through below);
                    // `rax!=0` (bit 62 halt | bit 61 repoll — the only two other TAG_* values a
                    // Store can produce) => this store STILL retired, so bump `insns_retired` and
                    // advance `pc` to this instruction's fallthrough (mirroring the interpreter's
                    // normal per-instruction commit) before returning the shim's tag unchanged —
                    // the SAME "retire, stop the chain" shape for both halt and repoll, since only
                    // the already-embedded tag value distinguishes them one level up
                    // (`ChainCache::run_block`), not anything computed here.
                    let next_off = (step.static_offset + step.ilen) as i32;
                    emit_skip(a, Cc::E, |rare| {
                        rare.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
                        rare.lea_r32_mem(Reg::RCX, Reg::R8, next_off);
                        rare.mov_mem_r32(Reg::RDI, PC_OFF, Reg::RCX);
                        rare.ret();
                    });
                    // Plain continue: retired normally, nothing else to do (no `rd` for `Store`).
                    a.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
                },
            );
        }
        _ => unreachable!("decode_chain never includes a non-ALU, non-terminal instruction"),
    }
}

/// Emit `a.jcc_rel32(skip_if, len(inner))` followed by `inner`'s bytes, where `inner` is built by
/// `build` into its own temporary buffer first (so its exact length is known up front — see the
/// module doc's "one kind of jump" note). When `skip_if` holds at runtime, execution jumps past
/// `inner` entirely to whatever `a` emits next; otherwise it falls straight into `inner` (which,
/// in every call site in this module, ends in its own `ret` — `inner` is always a "handle the rare
/// case and return" block, never a block meant to fall through to `a`'s continuation).
fn emit_skip(a: &mut Asm, skip_if: Cc, build: impl FnOnce(&mut Asm)) {
    let mut inner = Asm::new();
    build(&mut inner);
    a.jcc_rel32(skip_if, inner.buf.len() as i32);
    a.buf.extend_from_slice(&inner.buf);
}

/// Phase 3 (`docs/jit-scalar-design.md`): an if/else with exactly one runtime branch decision,
/// built the same "measure each block first" way [`emit_skip`] does (no label table, no
/// backpatching) — but unlike `emit_skip`'s inner block (which always ends in its own `ret`),
/// BOTH `then_` and `else_` here are expected to fall through to whatever `a` emits next: this is
/// the mechanism the Phase 3 fast/slow Load/Store split needs, since either branch must continue
/// into the REST of the chain (bump `insns_retired`, move on to the next `Step`), not return from
/// it. Compiles to:
/// ```text
/// jcc  cc, else_len + 5      ; jump straight into `then_` when `cc` holds
/// <else_ bytes>               ; falls straight through here when `cc` does NOT hold
/// jmp  then_len               ; ...then jumps over `then_` to land after it
/// <then_ bytes>
/// ```
fn emit_if_else(a: &mut Asm, cc: Cc, then_: impl FnOnce(&mut Asm), else_: impl FnOnce(&mut Asm)) {
    let mut then_buf = Asm::new();
    then_(&mut then_buf);
    let mut else_buf = Asm::new();
    else_(&mut else_buf);
    // Jump straight to `then_` when `cc` holds, skipping `else_`'s block AND the `jmp` that
    // immediately follows it (5 bytes: `E9 rel32`).
    a.jcc_rel32(cc, else_buf.buf.len() as i32 + 5);
    a.buf.extend_from_slice(&else_buf.buf);
    a.jmp_rel32(then_buf.buf.len() as i32);
    a.buf.extend_from_slice(&then_buf.buf);
}

/// Like [`emit_call_preserving_regs`], but preserves `RCX` (the freshly computed `va`, needed
/// again by the slow-path call-out if the Phase 3 fast-path attempt bails) instead of the
/// meaningless `R9` alignment padding — same push count (5, still odd — required for 16-byte
/// `RSP` alignment immediately before the `call`, see `emit_call_preserving_regs`'s doc), just a
/// padding register whose value the caller actually wants back this time. `RAX` (the fast-ptr
/// shim's return value — 0 or a real pointer) is deliberately left unprotected, exactly like
/// `emit_call_preserving_regs`'s `RAX`.
fn emit_fast_path_call_preserving_va(a: &mut Asm, emit_call: impl FnOnce(&mut Asm)) {
    a.push_r64(Reg::RCX);
    a.push_r64(Reg::RDI);
    a.push_r64(Reg::RSI);
    a.push_r64(Reg::RDX);
    a.push_r64(Reg::R8);
    emit_call(a);
    a.pop_r64(Reg::R8);
    a.pop_r64(Reg::RDX);
    a.pop_r64(Reg::RSI);
    a.pop_r64(Reg::RDI);
    a.pop_r64(Reg::RCX);
}

/// Phase 3: emit the correctly-sized, correctly-signed load `dst = *(ptr-sized-by-op)` straight
/// from a host pointer already in `ptr_reg` — the ONE place a `LoadOp`'s signedness matters for
/// the fast path (the shim that produced `ptr_reg` doesn't know or care about it, see
/// `sys.rs`'s `fast_load_ptr_addr` doc). Mirrors exactly what `fs_riscv::load_impl`'s
/// sign/zero-extension match does for each `LoadOp`, just performed by the CPU's own load
/// instruction instead of Rust arithmetic.
fn emit_sized_load_from_ptr(a: &mut Asm, op: fs_riscv::LoadOp, dst: Reg, ptr_reg: Reg) {
    use fs_riscv::LoadOp;
    match op {
        LoadOp::Lb => a.movsx_r32_mem8(dst, ptr_reg, 0),
        LoadOp::Lbu => a.movzx_r32_mem8(dst, ptr_reg, 0),
        LoadOp::Lh => a.movsx_r32_mem16(dst, ptr_reg, 0),
        LoadOp::Lhu => a.movzx_r32_mem16(dst, ptr_reg, 0),
        LoadOp::Lw => a.mov_r32_mem(dst, ptr_reg, 0),
    }
}

/// Phase 3: emit the correctly-sized store `*(ptr-sized-by-op) = src` straight to a host pointer
/// already in `ptr_reg`. `src`'s low 1/2/4 bytes are stored, matching `fs_riscv::store_impl`'s
/// truncating byte-wise write for `Sb`/`Sh` and the full word for `Sw`.
fn emit_sized_store_to_ptr(a: &mut Asm, op: fs_riscv::StoreOp, ptr_reg: Reg, src: Reg) {
    use fs_riscv::StoreOp;
    match op {
        StoreOp::Sb => a.mov_mem8_r8(ptr_reg, 0, src),
        StoreOp::Sh => a.mov_mem16_r16(ptr_reg, 0, src),
        StoreOp::Sw => a.mov_mem_r32(ptr_reg, 0, src),
    }
}

/// Protect every register a Load/Store call-out's `call` is free to clobber (RDI/RSI/RDX/R8 are
/// ALL caller-saved per the SysV ABI — not just R8; RDI is the cpu pointer and RSI/RDX are the
/// `&mut dyn Bus` fat pointer, both needed by every instruction/call-out for the rest of the
/// chain, not merely R8's entry pc) around `emit_call` (which sets up any call-specific argument —
/// e.g. Store's `val` into R8 — and emits the `movabs`+`call` itself), then restores them in
/// reverse order. `R9` is pushed first purely as **16-byte-alignment padding**: the SysV ABI
/// requires `RSP % 16 == 0` immediately before a `call` (so the callee sees `RSP % 16 == 8` at its
/// own entry, matching how *this* chain itself was called); a chain's `RSP` is `entry_rsp` (≡ 8
/// mod 16) at the top of every Load/Store step, and exactly **5** pushes (an odd count) restores
/// 16-byte alignment before the `call` (4 pushes — one per real register — would leave it
/// misaligned). `R9`'s own value doesn't need preserving (it's only ever live within a chain's
/// single terminal step, never across a Load/Store `call`), but pushing-then-popping it is both
/// the simplest way to get the required odd push count AND, incidentally, still round-trips
/// whatever was in it for free. RAX (the call's return value) is deliberately left untouched by
/// any of this.
fn emit_call_preserving_regs(a: &mut Asm, emit_call: impl FnOnce(&mut Asm)) {
    a.push_r64(Reg::R9);
    a.push_r64(Reg::RDI);
    a.push_r64(Reg::RSI);
    a.push_r64(Reg::RDX);
    a.push_r64(Reg::R8);
    emit_call(a);
    a.pop_r64(Reg::R8);
    a.pop_r64(Reg::RDX);
    a.pop_r64(Reg::RSI);
    a.pop_r64(Reg::RDI);
    a.pop_r64(Reg::R9);
}

