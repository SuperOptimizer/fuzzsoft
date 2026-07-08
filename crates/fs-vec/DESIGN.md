# fs-vec — design notes

`fs-vec` is the M4 vectorization foundation (`docs/architecture.md` §2, §8; `docs/decisions.md`
#3, #4, #23, #45). What's implemented *today* is a safe-Rust SoA executor with a real SIMD fast
path for fetch, ALU, and same-address memory, backed by a genuinely shared interleaved memory
model (`VecMmu`) rather than one `fs_mmu::Mmu` per lane. This document records what's vectorized
now, the AVX-512 target the code is shaped for, and exactly what changes when we get there.

## What exists now

- `VecCpu::regs: [[u32; LANES]; 32]` — SoA register file, `regs[reg][lane]`.
- `VecCpu::pc: [u32; LANES]`, `VecCpu::active: [bool; LANES]` — per-lane PC and active mask.
- **`VecMmu`** (`src/vec_mmu.rs`) — ONE shared interleaved store for all `LANES` lanes, replacing
  the original one-`fs_mmu::Mmu`-per-lane model. For guest word `W`, lane `L`'s copy of that word's
  content lives at `content[W*LANES + L]` and its parallel byte-permission plane (RWX+RAW, same
  bit encoding as `fs_mmu`) lives at `perms[W*LANES + L]` — exactly architecture.md §2/§3's
  `host_word_index = guest_word*16 + lane` layout, so all `LANES` lanes' copies of one guest word
  are contiguous. `map`/`protect` broadcast identically to every lane (the byte-identical starting
  snapshot every fuzz case begins from); per-lane divergence only ever arises from lanes
  subsequently *executing* differently, never from a different starting image.
- `VecCpu::step` first tries `try_simd_fast_step` (fetch once for the whole converged group, then
  dispatch to the ALU, branch/jump, same-address-memory, or gather/scatter-memory payload); if that
  declines (only for MUL/DIV/REM, atomics, system instructions, or pc divergence), it falls back to
  `for lane in 0..LANES { if active[lane] { step_lane(lane, ...) } }`, each lane now against the
  shared `VecMmu` via its per-lane accessors (`load_lane`/`store_lane`/`ifetch16_lane`) instead of
  a private `fs_mmu::Mmu`.
- `alu`/`muldiv`/`branch_taken` are byte-for-byte copies of `fs_riscv`'s private functions (same
  edge cases: DIV/0 → `0xffff_ffff`, REM/0 → dividend, `INT_MIN / -1` → DIV=`0x8000_0000`/REM=`0`,
  MULH* via i64/u64 widening). `simd_alu`/`simd_branch_taken` are the `Simd<u32, LANES>`-packed
  twins of `alu`/`branch_taken`, fuzz-tested against them
  (`simd_alu_fast_path_matches_scalar_alu_exhaustively`,
  `simd_branch_taken_matches_scalar_branch_taken_exhaustively`).
- `#![forbid(unsafe_code)]` everywhere in this crate — including `VecMmu` and every fast path:
  `#![feature(portable_simd)]` + `std::simd` is 100% safe Rust (decision #23), so no `unsafe` was
  needed to add any of this.

## What's SIMD today vs. still scalar

**SIMD (`try_simd_fast_step` and its payloads):**
- **Fetch** ([`fetch_converged`]): every active lane shares the same `pc`
  (`VecCpu::lanes_converged`), so the whole group fetches its instruction word from the shared
  `VecMmu` in ONE `VecMmu::ifetch16_same` call (two for a 32-bit instruction) instead of `LANES`
  independent per-lane fetches — checked, not assumed: declines if `VecMmu` itself declines
  (misalignment/OOB/any active lane missing `PERM_EXEC`) or if the active lanes' fetched bytes are
  not byte-identical (a lane whose memory has diverged, e.g. self-modifying code, must not be
  silently treated as converged).
- **ALU** (`try_simd_alu`): `Inst::OpImm`/`Inst::Op` for the ten `AluOp`s with a clean packed
  form — add, sub, and, or, xor, sll, srl, sra, slt, sltu (covers both 32-bit and C-extension
  forms, since `decode_compressed` already canonicalizes e.g. `c.addi`/`c.and`/`c.srli` down to
  `Inst::OpImm`/`Inst::Op`). Decoded once (shared with the other payloads), executed as one masked
  `Simd<u32, 16>` op (`simd_alu`).
- **Branch/jump** (`try_simd_branch`/`try_simd_jal`/`try_simd_jalr`): `Inst::Branch` computes every
  active lane's own taken/not-taken condition from its own `rs1`/`rs2` (`simd_branch_taken`, the
  packed twin of `branch_taken`) and selects each lane's own next `pc` — lanes are explicitly
  allowed to *disagree* on the outcome within this one converged step (that is the ordinary "loop
  exit condition differs per lane" case), it costs nothing extra to compute every lane's own answer
  rather than requiring agreement first. `Inst::Jal`'s target is identical across lanes (`pc+imm`,
  both converged) so only the link register varies per lane; `Inst::Jalr`'s target is per-lane
  (`rs1+imm`, snapshotted before `rd` is written, matching `step_lane`'s ordering exactly). None of
  these three ever decline once dispatched — every `BranchOp`/jump form has a clean packed form.
- **Same-address memory** (`try_simd_load`/`try_simd_store`): `Inst::Load`/`Inst::Store` whose
  effective address (`rs1 + imm`, computed as one `Simd<u32, LANES>` add) also agrees across every
  active lane — the common case for lockstep lanes running the same code against byte-identical
  memory. One `VecMmu::load_same`/`store_same` call services the whole group (permission-checked
  as one masked compare across the 16-lane line) instead of `LANES` separate loads/stores.
- **Gather/scatter memory** (`try_simd_gather_load`/`try_simd_gather_store`): when the effective
  address diverges (or `VecMmu::load_same`/`store_same` itself declines, e.g. misalignment/OOB/
  permission), `try_simd_load`/`try_simd_store` now fall through to a single
  `VecMmu::load_gather`/`store_scatter` batch call instead of declining outright — the shared
  fetch/decode/address-computation is kept; only the actual memory access degrades to `LANES`
  independent per-lane checked accesses (the safe-Rust stand-in for `vpgatherdd`/`vpscatterdd`). A
  lane whose access faults is halted individually (`VecCpu::halt`) without disturbing any other
  lane's `pc`/registers/mask; these two never decline (unlike the same-address pair) since
  per-lane faults are a fully-handled outcome, not a further reason to fall back.
- `VecCpu::simd_alu_steps`/`simd_branch_steps`/`simd_mem_steps`/`simd_gather_steps` count how many
  `step()` calls took each fast path — diagnostic counters used by the fast-path tests and
  `examples/bench.rs`, not part of the correctness contract.

**Still scalar (`step_lane`, now against the shared `VecMmu` via `load_lane`/`store_lane`/
`ifetch16_lane` instead of a private `fs_mmu::Mmu`):**
- Any pc or instruction-word divergence (every fast-path function returns `false` having mutated
  nothing on decline, so falling through and re-fetching/re-computing is free of side effects until
  the point a store actually commits).
- `LrW`/`ScW`/`AmoW` — always scalarized regardless of convergence; no same-address fast path for
  atomics yet (see "Remaining gap" below).
- `Mul`/muldiv (MUL/MULH*/DIV/DIVU/REM/REMU) — no packed form exists for these regardless of
  convergence (architecture.md §2); always scalarized, exactly as designed originally.
- `Ecall`/`Ebreak`/`Fence`/CSR-privileged/`Illegal` — one-off control instructions, not worth a
  vector form.

## Benchmark (`examples/bench.rs`)

`cargo run --release --example bench -p fs-vec` now runs **two** RV32IM counting loops (each with
a fixed trip count → lanes stay pc-converged throughout) on the same two engines as before —
`VecCpu::step` vs. `LANES` independent `fs_riscv::Cpu`s each stepped one instruction at a time —
but now both engines share the exact same shared-`VecMmu`/per-lane-`fs_mmu::Mmu` split the
production code uses:

1. **ALU+fetch-bound loop** (~83% of dynamic instructions are ALU, no memory ops in the loop body)
   — isolates the fetch-deduplication + packed-ALU win.
2. **Same-address memory loop** (same shape, but the per-lane accumulator round-trips through ONE
   shared guest address every iteration instead of a register) — isolates the new same-address
   `load_same`/`store_same` win.

Measured on this machine (release build, `std::time::Instant`, 16 lanes × 2000 loop iterations ×
40 repeats per program, both engines verified to retire the same lane-instruction count before
computing a rate):

```
ALU+fetch-bound loop:
  SIMD fast path:    15,363,200 lane-instructions in ~62–91ms    ≈ 169M–247M lane-instr/sec
  scalar-over-lanes: 15,363,200 lane-instructions in ~229–329ms  ≈  47M– 67M lane-instr/sec
  speedup:           ~3.6x–3.9x across repeated runs

Same-address memory loop:
  SIMD fast path:    7,684,480 lane-instructions in ~33–40ms     ≈ 193M–236M lane-instr/sec
  scalar-over-lanes: 7,684,480 lane-instructions in ~143–146ms   ≈  53M– 53M lane-instr/sec
  speedup:           ~3.7x–4.4x across repeated runs
```

**This beats the previous ~2.55x–2.8x** (recorded when `Branch`/`Jal`/`Jalr` always scalarized
even on fully pc-converged loops — see git history for the prior numbers) by roughly 35-60%,
consistent with removing exactly the bottleneck DESIGN.md itself used to call out as "likely
higher-value than gather/scatter": both benchmarked loops execute one `bge` and one `jal` every
iteration, and — before `try_simd_branch`/`try_simd_jal` existed — *every single iteration* fell
back to `step_lane`'s fully-scalar per-lane fetch/decode/execute for those two instructions, even
though pc stayed perfectly converged the entire run. `examples/bench.rs`'s own diagnostic step
counters now confirm this directly: the ALU+fetch-bound loop takes zero `step_lane` fallbacks for
its `bge`/`jal` (`branch/jump-path` step count equals the loop's trip count), leaving only
MUL/DIV/REM and atomics as the categories that must always scalarize regardless of convergence.
The remaining gap to a much larger speedup is now driven by the loop overhead outside the
benchmarked instruction mix (the outer Rust harness, `insns_retired` bookkeeping per step, and
`std::simd`'s `Simd<u32,16>`/`Mask<i32,16>` codegen not being confirmed to lower to genuine
`zmm`/`k`-register AVX-512 instructions on this host — see "Remaining gap" below) rather than any
still-scalarized instruction in these two loop bodies specifically.

## The AVX-512 target (architecture.md §2)

### 1. Register file as ZMM transpose

Today: `regs[reg][lane]` is a plain `[u32; 16]` array per register. Mechanically, this *is* the
layout of one ZMM register's 16 packed `u32` lanes — `regs[reg]` loads directly into a ZMM with
`vmovdqa32 zmm, [regs + reg*64]` (each register occupies one 64-byte-aligned cache line). No
transpose logic needs to change; only the *consumer* of `regs[reg]` changes from "a Rust `[u32;
16]` read in a loop" to "a `__m512i` loaded once".

`pc` and `active` get the same treatment: `pc` becomes a ZMM (`vmovdqa32`), and `active` becomes
a 16-bit AVX-512 k-mask (`__mmask16`) rather than `[bool; 16]` — `active[lane] as bool` maps 1:1
onto k-mask bit `lane`.

### 2. Lockstep fetch/decode — now implemented in safe Rust

**Implemented.** Converged lanes (same `pc`, checked by `VecCpu::lanes_converged`) fetch and
decode **once**, not once per lane, via `fetch_converged` + `VecMmu::ifetch16_same` — the whole
point of vectorizing, and previously the biggest gap in this crate (the original SIMD fast path
still fetched `LANES` times because each lane owned an independent `fs_mmu::Mmu`). `decode`/
`decode_compressed` still run scalar (they are control-flow-heavy bit-twiddling, not a good SIMD
target), but they run *once per converged group* instead of once per lane. Divergent lanes (a
k-mask subset whose `pc` disagrees with the group) fall out of the vector step and are
scalar-executed via the same `step_lane` that exists today, then re-converge naturally once their
`pc` catches back up to the majority (or they get folded into a new converged group next step).

The `vpcmpeqd` + `kortestw`-against-a-broadcast that architecture.md envisions for convergence
detection is today's `Mask::from_array` + `.simd_eq(...).any()` idiom throughout `vec_mmu.rs` and
`lib.rs` — safe `std::simd` compiles this down to roughly the same instruction shape without an
explicit intrinsic.

### 3. Memory: same-address fast path vs. gather/scatter — both paths now implemented and wired in

**Implemented (both same-address and gather/scatter):** `VecMmu` (`src/vec_mmu.rs`) is the one
shared interleaved memory region architecture.md §3/§4 calls for, replacing the original
one-`fs_mmu::Mmu`-per-lane model:

- `content[guest_word * LANES + lane]` / `perms[guest_word * LANES + lane]` — all `LANES` lanes'
  copies of one guest word are contiguous, exactly as specified. Each `u32` in both planes packs
  its four guest bytes little-endian, matching `fs_mmu::Mmu`'s own byte order.
- **Fast path** (`VecMmu::load_same`/`store_same`/`ifetch16_same`): if every active lane's
  effective address is identical, one contiguous `Simd<u32, LANES>` read/write of the interleaved
  line services all 16 lanes, with the whole 16-lane permission line checked in one masked compare
  (`(perm_line & need).simd_eq(need)`) before any lane's content is touched. This is the safe-Rust
  stand-in for `vmovdqa32` + `vpcmpd`/`vptestmd`; only on a check failure does the caller fall
  through to the gather/scatter path below.
- **Divergent path, now wired into the hot path** (`VecMmu::load_gather`/`store_scatter`, called
  from `VecCpu::try_simd_gather_load`/`try_simd_gather_store`): still a scalar loop over active
  lanes underneath (decision #45), but *structured* for the eventual `vpgatherdd`/`vpscatterdd`:
  `load_gather`/`store_scatter` take one address (and, for stores, one value) *per lane* and return
  one `Result` *per lane* — a real gather instruction's calling convention, with the per-lane
  permission check taking the place of the mask register a hardware gather would use to suppress
  faulting lanes. `try_simd_load`/`try_simd_store` now call these instead of declining outright on
  address divergence (or on a same-address decline), so the shared fetch/decode/address-computation
  from the converged group is kept either way — only the underlying `VecMmu` access itself degrades
  from one shared read/write to `LANES` independent per-lane ones. A lane whose access faults is
  halted individually, without disturbing any lane that didn't fault.
- Divergence is still the primary cost lever to minimize (not just tolerate): the interleaved
  layout and the same-address check exist specifically so the *common* case (lockstep,
  byte-identical lanes) stays on the cheapest path, and only genuinely-divergent lanes pay the
  scalar-loop memory-access cost (though they now still share fetch/decode/address-computation).
  **Re-benchmark the ~4x same-address-vs-divergent gap architecture.md cites (Falk's Xeon Phi
  numbers) on the actual host (7945HX) once real AVX-512 intrinsics replace this safe-Rust
  stand-in** — do not assume Falk's 35-vs-157-cycle numbers transfer, and note that a safe-Rust
  scalar loop is not yet a fair proxy for `vpgatherdd`'s real relative cost either way.

### 4. k-mask predication

Every vector ALU/load/store op in this crate's fast paths is already predicated by the current
active mask (`Mask<i32, LANES>`, via `.select(...)` on both the register writeback and, for
memory, both the content and permission planes) — the AVX-512 target's real `__mmask16` is a
drop-in replacement for the same `[bool; 16]`-backed `Mask` this crate already threads through
every payload. `VecCpu::active` is designed to become exactly this mask (see §1).

### 5. Masked scalarize-16 fallback for DIV/REM/MULH*

DIV/DIVU/REM/REMU have no packed AVX-512 integer-divide instruction, and MULH/MULHSU/MULHU need a
64-bit-widened intermediate with no clean 32-lane packed form either. These always go through a
**masked scalar loop over all 16 lanes** (`muldiv` in this crate, extracting each lane's `a`/`b`
from the ZMM, computing scalar, and inserting back) — never a vector instruction. This crate's
`muldiv` is written to be exactly that per-lane body already; the AVX-512 version differs only in
how `a`/`b` are extracted (`vpextrd`/lane-extract from the two source ZMMs) and how the 16 results
are re-packed (`vpinsrd`/scatter back into a destination ZMM), not in the arithmetic itself. This
is why requirement 2 insists on copying `fs_riscv`'s DIV/0, `INT_MIN/-1`, and MULH* edge cases
byte-for-byte now: get that exact once, in one place, and the AVX-512 scalarize path inherits it
unchanged.

### 6. The hybrid plan (decision #3)

End state per decision #3: AVX-512 SIMD on hot common paths (converged lockstep execution against
byte-identical post-boot snapshots), scalar-execute divergent tails (the k-mask'd-out lanes, plus
the masked scalarize-16 fallback above), re-converge at common PCs. This crate's
scalar-`step_lane` *is* what the "scalar-execute divergent tails" path calls — it does not go
away when AVX-512 lands, it becomes the minority-lane and DIV/REM/MULH* fallback path invoked
from inside an otherwise-vectorized `step`.

## Remaining gap to true AVX-512

Fetch, ALU, branches/jumps, and both same-address and gather/scatter memory are all vectorized now
(in safe Rust); what's left:

1. **Masked scalarize-16 for MULH*/DIV/REM.** Still exactly the `muldiv` scalar loop, as designed
   from the start (§5 above) — there is no packed form to move to regardless of convergence, only
   the *extraction*/*repacking* glue changes when this becomes real AVX-512 (`vpextrd`/`vpinsrd`
   instead of Rust array indexing).
2. **No same-address fast path for `LrW`/`ScW`/`AmoW`.** These always scalarize via `step_lane`
   today, converged or not; a same-address `AmoW` (RMW) fast path is possible following the same
   shape as `try_simd_load`/`try_simd_store` but was not attempted here — atomics are rarer on a
   hot ALU/memory loop than plain load/store, so this was deprioritized.
3. **Gather/scatter is still `LANES` genuinely-independent per-lane `VecMmu` accesses.**
   `try_simd_gather_load`/`try_simd_gather_store` share fetch/decode/address-computation across the
   converged group, but the actual memory access underneath (`VecMmu::load_gather`/
   `store_scatter`) is still a scalar Rust loop, not a real `vpgatherdd`/`vpscatterdd`. This only
   matters for programs whose *addresses* diverge while `pc` stays converged — neither of
   `examples/bench.rs`'s two loops exercise this (both use a shared or purely-register-resident
   accumulator), so it doesn't show up in the numbers above; a benchmark with genuinely
   lane-scattered memory access would be needed to measure its real-world impact.
4. **Real AVX-512 registers/intrinsics.** `std::simd::Simd<u32, 16>` is portable — it does not
   guarantee it compiles to a single `zmm` register + AVX-512 instructions on this host; it may
   lower to multiple narrower vector ops depending on target features enabled at compile time.
   Confirming/forcing actual `zmm` codegen (`RUSTFLAGS="-C target-feature=+avx512f"` or explicit
   intrinsics) is unverified — see "Where `unsafe`/intrinsics will go" below.

## Where `unsafe`/intrinsics will go

This crate is `#![forbid(unsafe_code)]` — `VecMmu` and every fast path added above (fetch, ALU,
branches/jumps, same-address memory, gather/scatter memory) are built entirely on safe
`std::simd`/portable_simd (decision #23 explicitly allows this over raw intrinsics, and none of it
needed `unsafe`). When the full AVX-512 executor is built out with real hardware intrinsics,
decision #23 still allows either staying on `std::simd`/portable_simd on nightly (which does have
safe
`Simd::gather_or`/`Simd::scatter` helpers, but they index into a plain Rust slice by offset, not a
fault-checked guest address through the permission/RAW-tracking `VecMmu` this crate needs), or
dropping to raw `core::arch::x86_64` AVX-512 intrinsics for the actual `vpgatherdd`/`vpscatterdd`
hardware instructions once real codegen is verified:

- A new module (e.g. `fs_vec::avx512`, likely a separate crate or `#[cfg(target_feature =
  "avx512f")]`-gated module) will hold the only `unsafe` blocks in the vectorized path: the
  `_mm512_load_epi32`/`_mm512_store_epi32` same-address fast path (this crate's `load_same`/
  `store_same` are the safe-Rust reference implementation these intrinsics must match),
  `_mm512_i32gather_epi32`/`_mm512_i32scatter_epi32` for the divergent path (`load_gather`/
  `store_scatter` are the reference here), `_mm512_cmpeq_epi32_mask`/`_mm512_kortestc` for
  convergence/fast-path detection, and `_mm512_mask_*` predicated ALU ops.
- Everything feeding those intrinsics (decode, address computation, the k-mask *value* itself as
  `[bool; 16]` vs. `__mmask16`) stays safe Rust; only the raw intrinsic calls are `unsafe`, kept
  behind a narrow, audited boundary — matching decision #45 ("targeted `unsafe` in profiled hot
  spots ... only when justified", "never in the decoder ... without a fight").
- DIV/REM/MULH* scalarization and any lane that falls out of the k-mask remain calls into this
  crate's existing safe `step_lane`/`muldiv` — no unsafe code is needed for the scalar fallback
  itself, only for the gather/scatter/compare glue around it.
