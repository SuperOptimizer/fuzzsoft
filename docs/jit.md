# fuzzsoft JIT — staged plan

From a 3-way independent-design workflow + adversarial synthesis (2026-07-08). The vectorized-fuzzer
prototype proved the interpreter's **per-lane dispatch cost** makes divergent kernel fuzzing a net
loss (0.83×) — a JIT is the identified unlock. This is the M5 endgame, built in **measurement-gated
stages** so no expensive stage is committed on paper estimates.

Name: **threaded-code-to-native, PA-keyed, escape-to-interpreter, scalar-JIT-feeds-the-vector-fallback.**

## Load-bearing design decisions (grounded in the repo)

- **PA-keyed block cache**, not VA+satp. Instruction fetch doesn't honour MPRV, so a VA→PA is stable
  while the mapping is unchanged → shared kernel `.text` collapses onto **one** cache entry across
  every address space, maximizing the snapshot-reset amortization (kernel code is stable post-boot, so
  a block compiles once and is reused across millions of case resets). Two tiers mapped onto the
  existing `Golden`/`CowRam` split: a golden tier (process-lifetime, valid while no spanned page is
  overlaid) that survives every reset, and a per-case tier cleared with `CowRam::reset()`.
- **SFENCE.VMA does NOT invalidate the cache** (it only flushes the VA→PA soft-TLB; PA-keyed *bytes*
  don't change because a mapping changed). What invalidates a golden entry: any `Mmu::write`/`protect`/
  `poison` (+ CowRam equivalents) touching a byte in the block's PA range — hooked into the write path
  (piggyback the existing `DIRTY_BLOCK`/`mark_dirty` machinery). **This invalidation-site audit is the
  single highest-severity correctness risk** — an incomplete hook list = stale compiled code.
- **Soft-MMU byte-perms are preserved by construction**: JIT'd loads/stores compute the address inline
  (pure ALU) but **call out to the unmodified `Bus::load`/`store`/`xlate`** for the actual access — so
  the byte-granular RAW/redzone oracle (decision #19) has *zero* drift. Guard-page/host-MMU tricks are
  rejected (page-granular, would regress the byte oracle). Inlining the perm fast-path into emitted asm
  is a *later* PR, gated on an exhaustive differential proof it agrees with the Rust fast path (a
  drifted duplicate oracle is a correctness landmine for a bug-attribution fuzzer).
- **Hand-rolled x86-64, not Cranelift**: the codegen surface (RV32IM ALU/branch/addr, non-privileged)
  is small; the host is pinned (decision #52, `target-cpu=native`); a small encoder is more auditable
  and mirrors `fs-hostmem`'s already-accepted small unsafe surface. (Reversible: fall back to Cranelift
  at Stage 1 if hand-encoder bug rate is too costly.)
- **Honest correction**: `Cpu::step` takes `bus: &mut dyn Bus` — the interpreter *already* pays dynamic
  dispatch on every memory op, so a JIT calling through `dyn Bus` is no regression but also not a free
  "direct call". Whether to monomorphize over a concrete `Machine`/`CowMachine` is an open question.
- **New crate `fs-jit`** (Stage 0 `forbid(unsafe_code)`; from Stage 1 the one unsafe-carrying crate:
  mmap(RW)+write+mprotect(R-X)+transmute-to-fn-ptr+call+munmap, isolated `sys.rs` with SAFETY comments,
  mirroring fs-hostmem). Depends on fs-riscv + fs-mmu; they never depend back → their forbid stays.

## Stages (each independently validated bit-exact vs the interpreter: differential + Spike/fs-diff + boot + an adversarial self-modifying-code/W^X test)

- **Stage 0 — threaded-code Inst-cache (days, zero unsafe, GO/NO-GO GATE).** Extract
  `exec_one(&mut Cpu, &mut dyn Bus, Inst) -> Result<Exit,Trap>` from `Cpu::step` (pure refactor, tests
  unchanged). New `fs-jit`: PA-keyed direct-mapped `Vec<Inst>` block cache (golden-tier, no
  invalidation), wired into the scalar cores-first runner only. Benchmark vs the interpreter.
  **Est ~1.5–2.5×** (removes re-decode/RVC-immediate-extraction + interior fetch-perm rechecks — *not*
  dispatch; Rust's `match` is already a jump table). **If <1.3×, decode/dispatch was never the dominant
  cost → re-diagnose before the much larger Stage 1.** This is the number every later stage is sized
  against.
- **Stage 1 — hand-rolled scalar x86-64 JIT (the real unlock).** Reuse Stage 0's cache; replace
  `Vec<Inst>` replay with emitted machine code for Lui/Auipc/Jal/Jalr/Branch/OpImm/Op/Load/Store
  (Mul/AMO/CSR/Fence/Ecall/Mret/Sret/Wfi/SfenceVma terminate a block → interpreter for that one insn).
  Regs memory-resident first, block-local reg-alloc later; loads/stores call the unmodified soft-MMU;
  add the write/protect invalidation hook + per-case CowRam tier. **Est ~2–4×** over the interpreter
  (capped: load/store call-outs and paged xlate walks don't speed up). × 32 threads → double-digit-to-
  ~100× aggregate over a single-thread interpreter.
- **Stage 2 — scalar JIT as VecCpu's divergent-lane fallback (THE measurement).** Make `VecSystem`'s
  per-lane divergent fallback call Stage 1's compiled block instead of the interpreter, and **re-run the
  exact 0.83× benchmark.** No new vector codegen. This is the single most important number in the plan:
  does removing per-lane dispatch flip vectorized fuzzing to a net win? (Speculatively 1.2–2×, MUST be
  measured before Stage 3/4 — repeating the un-measured-vectorization mistake here would be the worst
  failure available.)
- **Stage 3 (only if Stage 2 justifies) — closure-threaded SIMD templates** over fs-vec's proven
  `try_simd_*` set (zero unsafe) — removes per-step decode/match on the convergent path.
- **Stage 4 (only if Stage 3 justifies) — hand-rolled AVX-512 EVEX** in a new `fs-vjit` crate, one
  template per Inst kind over the resident `[[u32;16];32]` SoA regs; divergent lanes masked out and run
  by calling Stage 1's per-lane scalar JIT (never re-entering the interpreter) — both tiers share one
  PA-keyed cache + invalidation. This is the only place true k-mask-predicated divergent-lane execution
  (gamozolabs' technique) is built, deferred until measurement justifies it.

## Open questions (measure, don't assume)

Stage 0 & Stage 2 are explicit go/no-go gates. Also: profile the load/store vs ALU/branch fraction of
hot kernel/syscall-fuzz code (sets Stage 1's ceiling); whether to monomorphize over `dyn Bus`; the
exact write/protect invalidation-site checklist (top correctness risk); hand-encoder-vs-Cranelift as a
reversible Stage-1 decision; whether `.ko` module loading churns the golden cache.

## First increment

Extract `exec_one` (pure refactor) + `fs-jit` golden-tier `Vec<Inst>` cache wired into the scalar
runner + the head-to-head benchmark. Independently mergeable, zero new unsafe, and it produces the one
go/no-go number the entire rest of the plan is sized against.
