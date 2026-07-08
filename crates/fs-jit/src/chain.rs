//! Phase 1 of `docs/jit-scalar-design.md`: a chained ALU/branch native compiler, memory-resident
//! regs, admission-guarded. See the design doc in full for the rationale; this module implements
//! exactly its "Phase 1" section.
//!
//! **Scope.** A compiled chain is a straight-line run of `Lui`/`Auipc`/`OpImm`/`Op`/`Fence`,
//! optionally terminated by exactly one `Branch`/`Jal`/`Jalr` resolved with a branchless `cmov` (no
//! emitted conditional jumps — no label table, no backpatching, no relocation). Every other `Inst`
//! kind (`Load`, `Store`, `Mul`, `LrW`, `ScW`, `AmoW`, `Ecall`, `Ebreak`, `Csr`, `Mret`, `Sret`,
//! `Wfi`, `SfenceVma`, `Illegal`) simply ends the chain *before* itself — that instruction is never
//! compiled, and the native code, having retired everything before it, returns with `cpu.pc`
//! already pointing at it so the ordinary interpreter (`Cpu::exec_one`, via the embedded Stage 0
//! [`crate::BlockCache`]) handles it exactly as today.
//!
//! **Why a compiled Phase 1 chain can never trap or halt.** None of `Lui`/`Auipc`/`OpImm`/`Op`/
//! `Fence`/`Branch`/`Jal`/`Jalr` can return `Err` from [`fs_riscv::Cpu::exec_one`], raise
//! `Exit::Halt`, or need `Cpu::finish_exit`'s trap-vectoring (that all only happens via `Load`/
//! `Store`/`Mul`/CSR/`Ecall`/`Ebreak`/`Mret`/`Sret`/`Illegal`, none of which are ever compiled into
//! a chain). So a Phase 1 `JitFn` call always completes normally and its packed `u64` result is
//! always the `Continue` tag (`0`) — there is no `TrapPending`/`Halt` tag to interpret yet (those
//! arrive with Phase 2's `Load`/`Store`, per the design doc's ABI note). This is *why* Phase 1
//! needs no `Cpu::jit_pending_trap` field and no `fs-riscv` changes at all: every field this
//! module reads (`regs`, `pc`, `insns_retired`, `csr.mtimecmp`/`csr.stimecmp`,
//! `kmsan_enabled`/`cmplog_enabled`/`ubsan_enabled`, `virtual_time`) was already `pub`.
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
//! **Register convention** (see `emit.rs`'s doc comment for why no SIB byte is ever needed):
//! `RDI` = cpu pointer (base for every `[rdi+disp32]` memory operand, never overwritten); `R8` =
//! this call's entry pc (loaded once, read-only for the rest of the chain); `RAX`/`RCX` = the
//! two-operand scratch pair for every ALU op (`RAX`=lhs, `RCX`=rhs — this is also why `RCX` is a
//! natural choice for `Op`'s register-count shifts, which x86 hardwires to `CL`); `R9`/`R10` =
//! branch-resolution scratch (`setcc`/`cmovcc` targets). `RSI`/`RDX` are never touched, even though
//! Phase 1 doesn't need them, to stay Phase-2-ABI-compatible (see `sys.rs`'s `JitFn` doc).

#![forbid(unsafe_code)]

use crate::emit::{Alu2, Asm, Cc, Reg};
use crate::sys::Arena;
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
    /// See the module doc for why a chain never traps: this never needs to surface a `Trap` of its
    /// own — any real fault at `va` is re-derived for real by the fallback path's own
    /// `fetch`+`exec_one` call, exactly as it would be without this cache at all.
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
            match self.arena.write(&code).expect("mprotect failed") {
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
        let tag = self.arena.call(slot.code_off, cpu as *mut Cpu);
        debug_assert_eq!(tag, 0, "Phase 1 chains only ever produce the Continue tag");
        // Coverage edge, computed the SAME (address-based, not runtime-taken-based) way the
        // interpreter's own heuristic would — see `ChainSlot::terminal`'s doc comment.
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

/// Is this `Inst` compileable as a non-terminal chain link (falls straight through)?
fn is_alu_link(inst: &Inst) -> bool {
    matches!(inst, Inst::Lui { .. } | Inst::Auipc { .. } | Inst::OpImm { .. } | Inst::Op { .. } | Inst::Fence)
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
        a.add_qword_mem_imm8(Reg::RDI, INSNS_RETIRED_OFF, 1);
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
        _ => unreachable!("decode_chain never includes a non-ALU, non-terminal instruction"),
    }
}
