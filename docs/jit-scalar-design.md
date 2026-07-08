# fuzzsoft scalar x86-64 block JIT ("Stage 1") — design blueprint

Status: design, not yet implemented. Builds on the shipped Stage 0 threaded-code cache
(`crates/fs-jit/src/lib.rs`, measured **~1.26x** over the raw interpreter — commit `711dd33`),
which is the baseline every number below is measured against. Companion doc: `docs/jit.md` (the
overall staged plan this document narrows into a concrete Stage-1 build order).

## 0. Verdict on the four proposals

| Proposal | Correctness risk (5=safest) | Expected speedup over Stage 0 | Impl effort (5=least) | Testability / incrementality | Vector-extensibility |
|---|---|---|---|---|---|
| A. Minimal-risk, per-instruction stub (memory-backed regs) | **5** | 2 (1.05–1.3x) | 4 | **5** | 4 |
| B. Register-cached hot path (GPR-pinned basic blocks) | 2 | 4 (2.5–4x, unproven) | 2 | 3 | **2** |
| C. Pragmatic ALU-only ceiling measurement | **5** | 2 (1.1–1.4x, by design) | **5** | **5** | 4 |
| D. Aggressive inlined soft-MMU fast path | 1 | **5** (1.5–3x) | 1 | 2 | 4 |

**Verdict.** The prompt's prior is correct and this blueprint follows it: cheaply validate the
native-codegen path and measure the real ceiling before spending risk budget on register
allocation (B) or a duplicated TLB/perm oracle (D) — both are correctly identified by their own
authors as the highest-effort, highest-risk levers, and D is explicitly what `docs/jit.md` calls
"the single highest-severity correctness risk" territory. Neither A nor C alone is the final
answer, though: A's per-instruction granularity (no chaining) risks measuring noise, since Stage
0 already proved Rust's `match` dispatch is cheap (1.26x, not the a-priori 1.5–2.5x) — a
non-chained native stub mostly removes the same thing Stage 0 already removed. C's chained
ALU-only run is the one design in the whole set that gets a *real, batched* speedup number
**for free, with a provable safety argument**: RISC-V ALU/branch instructions can never trap or
touch `mip`/`mie`/CSRs, and C's runs never contain a `Store` (so `msip` cannot change), so the
only way a pending timer interrupt could newly appear strictly inside a chained run is the
passage of virtual time — which is `insns_retired`, a pure host-side counter checkable with one
comparison before entering the run. That closes the interrupt-timing gap B has to invent new
machinery for, at zero cost. **This blueprint's Phase 1 is C's design.** Phase 2 then extends the
*same* chain compiler to Load/Store using A's best asset — byte-for-byte extraction of
`Cpu::load`/`Cpu::store` into shared helpers so the soft-MMU oracle has zero drift — plus one
narrow idea borrowed from B (detect a CLINT-range store at the call-out and force an early chain
exit) to close the one new gap Load/Store reopens. B's GPR pinning and D's inlined TLB are both
explicitly **deferred**, not rejected: gated behind Phase 1+2's measured numbers, exactly as
`docs/jit.md`'s own stage-gating discipline demands. B is a weak vector-extensibility bet on top
of that (AVX-512 has 32 zmm lanes for 32 guest GPRs — the scarcity problem B solves mostly
disappears there), so even if Phase 1+2 shows residual register-traffic cost, D or a much simpler
"chain across Load/Store already gets you most of it" re-measurement should be tried before B.

## 1. Phased build order

Each phase is independently mergeable and independently Spike-testable (fs-diff) before the next
starts. Every phase keeps `fs-jit`'s existing public contract unchanged:
`BlockCache::run_block(cpu: &mut Cpu, bus: &mut dyn Bus, is_golden_page: &mut dyn FnMut(u32) -> bool) -> SysExit`
— so `fs-cli`'s `run_case_jit` (`crates/fs-cli/src/main.rs:286-311`) needs **zero changes** at
any phase; the JIT-vs-interpreter decision stays entirely inside `fs-jit`.

### Phase 1 — chained ALU/branch run, memory-resident regs, admission-guarded (the cheap ceiling number)

**Scope.** Compile `Lui, Auipc, OpImm, Op, Fence` (as a true no-op — no semantic effect in this
deterministic single-hart core) as a straight-line chain, optionally terminated by exactly one
`Branch`/`Jal`/`Jalr` resolved via `cmov`/`setcc` (never an internal jump — see below). Everything
else (`Load, Store, Mul, LrW, ScW, AmoW, Ecall, Ebreak, Csr, Mret, Sret, Wfi, SfenceVma, Illegal`)
is **not compiled**: the chain simply stops being extended there and the native code returns that
instruction's (compile-time-known) address with a plain `Continue` tag, and the existing
interpreter (`Cpu::exec_one`) handles it exactly as today.

**Why chaining is safe here, precisely.** A compiled chain never contains a `Store`, `Csr`, or any
instruction that can change `mip`/`mie`/CSR state, so nothing inside the chain can make a
currently-disabled-or-not-yet-due interrupt newly deliverable *except* the passage of virtual time
crossing `mtimecmp`/`stimecmp` (`Cpu::virtual_time()` is exactly `insns_retired`,
`crates/fs-riscv/src/lib.rs:1644`). That is a pure, deterministic, host-side computation available
*before* the chain runs. So: at dispatch, if `min(mtimecmp, stimecmp).saturating_sub(cpu.virtual_time()) >= chain.static_len`,
it is **provably** identical to run the whole chain natively in one call vs. single-stepping it
through the interpreter — no interrupt can be delivered late. If the budget check fails (rare,
only near a timer deadline), fall back to single-stepping that chain's cached instructions one at
a time via the existing Stage 0 `fetch`+`exec_one` path — slow, correct, and rare enough not to
matter for throughput. **Documented forward-compat risk:** this argument depends on the current
codebase having no asynchronous external-interrupt source (`fs-platform`'s own doc comment:
"PLIC/UART/virtio come later"); adding one later requires re-deriving this invariant.

**No internal jumps needed.** Because a chain has at most one control-transfer instruction and it
is always the *last* thing in the chain, every ALU op falls straight through sequentially and the
terminal `Branch`/`Jal`/`Jalr` is resolved with a **branchless `cmov`** (not an emitted conditional
jump): compute both the taken-target and not-taken-target as plain values, `cmp` the operands, and
`cmovcc` the winner into the return-value register. This means Phase 1's emitter needs **no label
table, no jump-target backpatching, and no relocation machinery at all** — a major scope reduction
worth calling out explicitly to the implementer.

**Register strategy.** `Cpu.regs: [u32; 32]` stays exactly where it is — no host GPR is ever
dedicated to a guest register (that is Phase-B-style register allocation, explicitly deferred).
Every operand is `mov reg32, [cpu_ptr + REGS_OFF + 4*i]` / `mov [cpu_ptr + REGS_OFF + 4*i], reg32`,
using an `i` that is a compile-time-known `u8` baked from the decoded `Inst` (no runtime register
dispatch, no bounds check needed — `i` is provably in `0..32`). `REGS_OFF` (and `PC_OFF`,
`INSNS_RETIRED_OFF`) must be computed via `std::mem::offset_of!(Cpu, regs)` etc. **once**, not
hand-derived, so a future field reorder fails a `const_assert`/compile check instead of silently
corrupting codegen. `x0` is elided at **compile time**: a read of `x0` becomes an immediate
`xor reg,reg` (no load emitted); a write to `x0` is omitted entirely (no store emitted) — strictly
cheaper than the interpreter's `rd_reg`/`wr_reg` runtime `if i==0` check
(`crates/fs-riscv/src/lib.rs:920-929`).

**`insns_retired` bookkeeping.** Increment it by **one, per retired instruction, inside the chain**
(a single `inc`/`add` after each instruction) rather than lump-summing the whole chain's length at
the end. This costs one cheap host instruction per guest instruction but means `insns_retired` is
always exactly correct even on the chain's early-exit paths added in Phase 2 — no separate
"partial chain" accounting logic is ever needed.

**Sanitizer/introspection gate.** Before dispatch, check
`!cpu.kmsan_enabled() && !cpu.cmplog_enabled() && !cpu.ubsan_enabled()` (all three are existing
`pub` accessors, `crates/fs-riscv/src/lib.rs:870,892,915`) — if any is on, this whole call falls
back to the Stage 0 path (plain `exec_one` per instruction). This is a cheap (three
`Option::is_some`) check, done on **every** dispatch, not cached, so a hypothetical future
mid-case toggle can't reactivate a stale compiled path. `Branch` and `Op::{Sub,Xor}` record CMPLOG
pairs and `Branch` can raise `Trap::KmsanTainted` when KMSAN is on
(`crates/fs-riscv/src/lib.rs:1306-1320`) — none of that is replicated in asm; it's simply routed
around the compiled path entirely.

**x86-64 emitter scope for Phase 1** (the complete instruction set needed, nothing more):
- `mov r32, [base+disp8/disp32]` / `mov [base+disp8/disp32], r32` — register-array read/write.
- `mov r32, imm32` — `Lui`, immediate materialization.
- `lea r32, [base+disp32]` — `Auipc`/`Jal`/`Jalr` link-address and pc-relative target arithmetic
  (the compile-time-known `entry_pc + static_offset + imm` sum, added against the one runtime
  value — the chain's entry pc, read once from `[cpu_ptr+PC_OFF]` into a register at chain entry).
- `add/sub/and/or/xor/cmp r32, r32` and `r32, imm32` — `Op`/`OpImm` ALU ops (x86's 32-bit forms
  already wrap mod 2^32, matching RV32's `wrapping_add`/`wrapping_sub` with zero extra codegen).
- `shl/shr/sar r32, cl` — shifts; x86 masks the count to 5 bits by hardware definition for a
  32-bit destination, matching RV32's `shamt & 31` with zero extra masking codegen.
- `cmp r32,r32` + `setcc`/`cmovcc` (`sete/setne/setl/setge/setb/setae` and the `cmov` equivalents)
  — `Slt`/`Sltu` and every `BranchOp` (signed via `l`/`ge`, unsigned via `b`/`ae`).
  `Jalr`'s `& !1` masking is a plain `and r32, 0xfffffffe`.
- `inc`/`add [base+disp], imm` — `insns_retired` bump.
- `ret`.

No `call` instruction is needed in Phase 1 at all (that arrives in Phase 2) — this keeps the
mmap(RW)→write→mprotect(R-X) arena, the fn-pointer ABI, and the emitter all genuinely minimal for
this first increment.

**Benchmark target.** Re-run the exact Stage 0 harness (boot + syscall-fuzz workload) with this
path enabled, reporting hit/miss/chain-length histograms (extending `BlockCache::hits()`/
`misses()`) alongside the headline number. **Go/no-go:** if this doesn't clear roughly **1.1–1.15x
over Stage 0** on the real workload, that is itself useful evidence (ALU/branch code is a small
fraction of retired-instruction time in this kernel-fuzzing workload) — proceed to Phase 2 anyway
(it targets the other, larger fraction) but downgrade expectations for Phase 3.

### Phase 2 — extend the chain to Load/Store (the real Stage-1 ISA scope)

**Scope.** `Load`/`Store` join the compiled, *continuable* (non-terminal) set — a chain may now
contain any interleaving of `{Lui,Auipc,OpImm,Op,Fence,Load,Store}` followed by at most one
terminal `Branch`/`Jal`/`Jalr`. This reaches `docs/jit.md`'s full Stage-1 ISA list
(`Lui/Auipc/Jal/Jalr/Branch/OpImm/Op/Load/Store`).

**ABI change.** The `JitFn` signature is fixed **now**, in a form Phase 1 already returns (unused
arguments cost nothing) so Phase 2 needs no ABI break:

```rust
/// SAFETY-relevant ABI, isolated to fs-jit's sys.rs (mirrors fs-hostmem's unsafe-surface style).
/// `bus_data`/`bus_vtable` are the two words of a decomposed `&mut dyn Bus` fat pointer, produced
/// by exactly one audited helper, never interpreted by emitted code itself — only forwarded
/// unchanged into call-outs.
pub type JitFn = unsafe extern "C" fn(cpu: *mut fs_riscv::Cpu, bus_data: *mut (), bus_vtable: *const ()) -> u64;
```

`rdi`=cpu ptr, `rsi`/`rdx` = the decomposed `dyn Bus` fat pointer (SysV integer-arg registers).
Phase 1's codegen already lives entirely off `rdi`; Phase 2 additionally needs `rsi`/`rdx` alive at
every `call` site, so they are **not** used as ALU scratch registers by the emitter (use
`rax/rcx/r8-r11`, all caller-saved and free, for temporaries).

**Load/Store call-out.** Extract `Cpu::load`/`Cpu::store` (`crates/fs-riscv/src/lib.rs:1154-1190`)
bodies **byte-for-byte** (not reimplemented) into two new `pub(crate)` functions callable both from
`exec_one` (updated to call them — zero behavior change) and from the JIT, guaranteeing the
byte-granular soft-MMU oracle (perm checks, sv32 `xlate`, `dyn Bus` dispatch, RAW-upgrade) has
provably zero drift from the interpreter. Each returns a packed `u64`: bit 63 set = trap pending
(the real `Trap` value — too large to pack — was written into a new `Cpu::jit_pending_trap: Option<Trap>`
field just before returning the sentinel); else the low bits carry the loaded value (Load) or a
`ContinueNeedsRepoll` flag (Store — see below). The native chain computes the address inline
(`mov`+`add`, pure ALU, identical cost to Phase 1's arithmetic), then does a `movabs`+`call` to the
call-out's fixed process-lifetime-stable address (Rust function addresses don't move; never a
`rel32` direct call since the mmap'd arena may be far from it in the address space).

**Closing the CLINT/`msip` gap Load/Store reopens.** Phase 1's admission-guard argument relied on
"no `Store` in the chain, so `msip` can't change mid-chain." Phase 2 breaks that assumption: a
guest `Store` to the CLINT MMIO window mid-chain **can** newly assert a software interrupt that,
today, is only observed at the top of `fs-cli`'s per-instruction loop
(`m.clint.mtime = cpu.virtual_time(); fs_platform::sync_timer(cpu, m);`,
`crates/fs-cli/src/main.rs:296-297`). Fix (borrowed narrowly from proposal B, not its register
pinning): the extracted store call-out additionally tests whether the resolved address falls in
the CLINT MMIO window — a check it can perform cheaply since it already has the physical address
in hand — and if so, returns a distinct `ContinueNeedsRepoll` tag instead of the ordinary
`Continue` tag. The native chain tests this tag after every Store call and, on
`ContinueNeedsRepoll`, **immediately returns** (`pc` = the address right after that store,
`insns_retired` already correctly incremented) rather than continuing to chain further
instructions — forcing the outer driver's per-instruction CLINT resync to run before anything
else executes. This is a runtime check paid only on an actual Store (and only meaningfully costs
anything on the rare CLINT-targeting one), not a new per-ALU-instruction tax, so it doesn't erode
Phase 1's win on ALU-heavy code. The `mtimecmp`/`stimecmp` admission guard is otherwise unaffected
(it's still a pure function of the chain's static instruction count).

**Coverage-edge parity.** `fs-cli`'s `cov.record_edge(prev, cur)` only fires on a
non-fall-through transition (`crates/fs-cli/src/main.rs:238`, `304-306`) and is checked once per
outer-loop call. A compiled chain retires zero or more ALU/Load/Store instructions (which always
fall through by exactly `ilen`, never generating an edge on their own) followed by at most one
control-transfer — so the single `(entry_pc, returned_next_pc)` pair the driver observes after one
chain call is still the semantically correct edge. This needs an explicit test (interleaved
ALU+Load+Store+Branch program, compiled-chain run vs. interpreter, asserting identical `cov`
bitmap), because it is exactly the kind of invariant a future change (e.g., letting a chain
continue past a taken branch into a second basic block) would silently break.

**Emitter scope addition:** `movabs r64, imm64` + `call r64` (indirect call to the fixed call-out
address), plus testing the packed return tag (`test`/`bt` on the high bits) to select
continue-in-chain vs. early-return vs. trap-forward vs. halt.

### Phase 3 — contingent: inline the TLB/perm fast path (only if the numbers say so)

Not started until Phase 1+2's measured hit/miss/chain-length data answers `docs/jit.md`'s own open
question: *is the load/store dispatch/call overhead the remaining bottleneck, or is the
`xlate`+`dyn Bus` cost itself the bottleneck?* If profiling shows the call-out overhead (not the
translation/permission-check work itself) still dominates retired-instruction time on the
kernel-fuzzing workload, build proposal D's inlined TLB-hit + byte-perm fast path — but only with
its required exhaustive property-differential harness (random `(regs, perms, TLB-state, CSR)`
combinations proving bit-for-bit agreement with calling the real `xlate`+`Bus::load/store`, not
just example programs) before trusting it, per `docs/jit.md`'s explicit warning that a drifted
duplicate oracle is "a correctness landmine for a bug-attribution fuzzer." If instead the
`xlate`/`Bus` cost itself dominates (docs/jit.md's own stated cap — this doesn't speed up no matter
how the call is dispatched), **stop here**: Phase 1+2 already banked the available win, and neither
Phase 3 nor proposal B's register pinning is worth the added risk. Register pinning (proposal B)
is only worth revisiting if the profiling instead shows register-array *traffic* (not memory
access) is the dominant residual cost on ALU/branch-heavy code specifically — a narrower, cheaper
hypothesis than committing to full basic-block-level GPR allocation up front.

## 2. Integration seam (concrete)

**Unchanged public contract** (this is the point — no caller outside `fs-jit` needs to change):

```rust
// crates/fs-jit/src/lib.rs
impl BlockCache {
    pub fn run_block(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut dyn Bus,
        is_golden_page: &mut dyn FnMut(u32) -> bool,
    ) -> SysExit { .. }
}
```

Internally, `run_block` gains, ahead of today's `poll_interrupt`+`fetch`+`exec_one`+`finish_exit`
sequence:

1. **Sanitizer gate**: `if cpu.kmsan_enabled() || cpu.cmplog_enabled() || cpu.ubsan_enabled() { return self.run_block_interpreted(cpu, bus, is_golden_page); }` (today's exact Stage 0 body, renamed).
2. **poll_interrupt()** — unchanged, first, as today.
3. **Lookup-or-compile a chain** at `cpu.pc` (an extension of `compile_run`, `crates/fs-jit/src/lib.rs:175-268`, that *also* emits native code for the ALU/branch/Load/Store prefix of the run it already decodes and caches — the existing per-instruction `CachedInsn` caching is untouched and still serves the interpreter fallback and any chain miss).
4. **Admission guard**: `if min(cpu.csr.mtimecmp, cpu.csr.stimecmp).saturating_sub(cpu.virtual_time()) < chain.static_len { /* fall back: single-step this chain's cached instructions via fetch+exec_one, Stage-0-style */ }`.
5. **Dispatch**: decompose `bus: &mut dyn Bus` into its two words via the one audited `fs_jit::sys` helper, call `chain.entry(cpu as *mut Cpu, bus_data, bus_vtable)`, interpret the packed `u64`:
   - `Continue` (low bits unused): `cpu.pc`/`cpu.insns_retired` already correct — return `SysExit::Continue`.
   - `TrapPending`: `let trap = cpu.jit_pending_trap.take().expect(..); cpu.finish_exit(Err(trap))`.
   - `Halt(code)`: `SysExit::Halt(code)` (from the HTIF `tohost` store path, forwarded through the extracted store call-out exactly as `exec_one`'s `Store` arm does today, `crates/fs-riscv/src/lib.rs:1368-1372`).

`Cpu` gains one new field: `pub(crate) jit_pending_trap: Option<Trap>` (or `pub` if `fs-jit` needs
direct access — either way, funneled through one audited accessor, not a scattered `pub` field).
`fs-riscv` gains `pub(crate) fn load_jit(...)`/`store_jit(...)` — the byte-for-byte extractions —
plus `std::mem::offset_of!`-derived constants for `regs`/`pc`/`insns_retired`/`jit_pending_trap`,
exposed to `fs-jit` through a small `jit_abi` module (mirrors `fs-hostmem`'s isolated-unsafe-surface
convention). `fs-jit` becomes the one unsafe-carrying crate from this point on (`sys.rs`: mmap(RW)
→ write → mprotect(R-X) → transmute-to-fn-ptr → call → munmap, plus the one fat-pointer
decompose/reconstruct helper, each with a dedicated round-trip unit test) — `fs-cli` keeps
`forbid(unsafe_code)`, per `docs/jit.md`'s crate-boundary decision.

## 3. Validation per phase

- **Phase 1**: (a) differential unit tests, every `{AluOp}` × `{rd==0, rs1==0, rs2==0, rd==rs1, rd==rs2, rs1==rs2}` combination, against `Cpu::exec_one`, bit-for-bit on `regs`/`pc`/`insns_retired`; (b) every `BranchOp` × taken/not-taken × forward/backward/page-adjacent target; (c) a purpose-built "timer fires mid-chain" program (set `mtimecmp` to land inside what would otherwise be a long compiled ALU run) proving the admission guard produces identical trap-delivery timing to the interpreter — this is the one scenario ordinary fuzzing is unlikely to exercise densely, so it must be a dedicated test, not incidental coverage; (d) the coverage-edge-parity test described above; (e) head-to-head benchmark vs. Stage 0 on the real boot+syscall-fuzz workload, reporting hit/miss/chain-length histograms.
- **Phase 2**: all of Phase 1's suite, plus (a) exhaustive Load/Store fault-boundary differential tests — byte-perm faults, RAW faults, unaligned, page-crossing, page-boundary-adjacent chains, HTIF `tohost` store; (b) the CLINT-range-store-mid-chain scenario, for real, proving `ContinueNeedsRepoll` reproduces the interpreter's timer-interrupt-observability exactly; (c) full `fs-diff` differential run vs. Spike over a real kernel boot with this path enabled; (d) the adversarial self-modifying-code/W^X test `docs/jit.md` requires for every stage (the golden-tier invalidation-hook audit from Stage 1's original scope still applies here — piggyback `DIRTY_BLOCK`/`mark_dirty`, now also invalidating compiled chains, not just decoded `Inst`s).
- **Phase 3** (if pursued): the exhaustive random-state `(regs, perms, TLB, CSR)` property-differential harness proposal D specifies, run to a much higher iteration count than example-program testing, before trusting the inlined fast path at all.

## 4. Fork point to the vectorized (AVX-512 SoA) JIT

The fork is exactly `docs/jit.md`'s **Stage 2**: once Phase 1+2 above are shipped and validated,
wire this scalar chain-JIT as `VecCpu`'s per-lane divergent-fallback (replacing the interpreter
fallback it uses today) and **re-run the exact 0.83x vectorization benchmark** — per `docs/jit.md`,
this is "the single most important number in the plan." No new vector codegen is written at this
fork point; only the call target for a masked-out divergent lane changes.

**What carries over unchanged:**
- The `JitFn(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const ()) -> u64` ABI becomes literally
  the per-lane call target Stage 4's k-mask-predicated divergent-lane execution routes to.
- The memory-resident, fixed-struct-offset register addressing pattern (`[cpu_ptr+REGS_OFF+4*i]`)
  generalizes directly to a `[[u32;16];32]` SoA array at Stage 4 — same "operand lives at a
  fixed offset, computed, written back" shape, just multiplied by a 16-lane stride
  (`[soa_ptr + i*64 + 4*lane]` or a zmm-per-register layout), instead of needing to be un-designed.
- The byte-for-byte extracted `load_jit`/`store_jit` helpers become exactly the per-lane fallback
  logic Stage 4 needs regardless of what the convergent vector path looks like — built as a
  byproduct of Phase 2, not a separate effort.

**What does *not* carry over, and must be redesigned at the fork:**
- Phase 1's branchless `cmov` resolution of a chain's terminal branch is fundamentally a
  single-target-per-call trick; 16 SIMD lanes can each want a *different* next-pc on a divergent
  branch, which is exactly why `docs/jit.md` defers true k-mask-predicated divergent execution to
  Stage 4 rather than trying to extend `cmov` — the convergent vector path's branch handling is new
  work, not a scalar carryover.
- Proposal B's host-GPR pinning was declined for this exact reason among others: AVX-512 gives 32
  zmm registers for RV32's 32 architectural GPRs, so the scarcity problem GPR pinning exists to
  solve mostly disappears in the vector version, and SysV's zmm16-31 have no standard preservation
  guarantee across a `call` (unlike the scalar callee-saved GPRs B relies on) — a vector kernel
  calling into the per-lane scalar fallback would have to pay an explicit full-register-file spill
  that this blueprint's scalar design never has to.

## 5. Summary of explicit deferrals (not rejections)

- **Register pinning across a compiled unit** (proposal B, in full) — deferred; revisit only if
  Phase 1+2 profiling shows register-array *traffic*, not memory access or dispatch, is the
  dominant residual cost on ALU/branch-heavy code, and even then budget for its added CLINT/`msip`
  correctness surface and its weak vector-fork payoff.
- **Inlined TLB-hit + byte-perm fast path** (proposal D) — deferred; gated on Phase 1+2's
  profiling showing call-out/dispatch overhead (not `xlate`/`dyn Bus` cost itself) dominates, and
  on building the exhaustive property-differential harness first, per `docs/jit.md`'s own warning.
- **Trace/superblock compilation across taken branches** — out of scope for all of Phase 1-3;
  chains stop at the first control-transfer, matching Stage 0's existing `compile_run` terminal
  set. Worth revisiting only after Stage 2 (the vector fork) is measured, since it changes the
  scalar JIT's basic-block-cache interaction with the golden/per-case tier invalidation story that
  is already flagged as the single highest-severity correctness risk in `docs/jit.md`.
