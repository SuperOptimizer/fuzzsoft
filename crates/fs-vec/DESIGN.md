# fs-vec — design notes

`fs-vec` is the M4 vectorization foundation (`docs/architecture.md` §2, §8; `docs/decisions.md`
#3, #4, #23, #45). What's implemented *today* is a safe-Rust SoA executor with a real (if partial)
SIMD fast path on top of a scalar-over-lanes fallback: it gets the state layout and masking
contract right so that growing the SIMD path towards real AVX-512 later is mechanical, not a
rewrite. This document records what's vectorized now, the AVX-512 target the code is shaped for,
and exactly what changes when we get there.

## What exists now

- `VecCpu::regs: [[u32; LANES]; 32]` — SoA register file, `regs[reg][lane]`.
- `VecCpu::pc: [u32; LANES]`, `VecCpu::active: [bool; LANES]` — per-lane PC and active mask.
- `VecCpu::step` first tries `try_simd_alu_step` (below); if that declines, it falls back to
  `for lane in 0..LANES { if active[lane] { step_lane(lane, ...) } }`, each lane against its own
  `fs_mmu::Mmu`.
- `alu`/`muldiv` are byte-for-byte copies of `fs_riscv`'s private functions (same edge cases:
  DIV/0 → `0xffff_ffff`, REM/0 → dividend, `INT_MIN / -1` → DIV=`0x8000_0000`/REM=`0`, MULH* via
  i64/u64 widening). `simd_alu` is the `Simd<u32, LANES>`-packed twin of `alu`, fuzz-tested against
  it (`simd_alu_fast_path_matches_scalar_alu_exhaustively`).
- `#![forbid(unsafe_code)]` everywhere in this crate — including the new SIMD path:
  `#![feature(portable_simd)]` + `std::simd` is 100% safe Rust (decision #23), so no `unsafe` was
  needed to add it.

This used to be *not yet vectorized at all* — the scaffolding decision #45 asked for
("correctness-first ... targeted `unsafe` in profiled hot spots only when justified"). It now has
one real vectorized path (converged-lane ALU) built on that scaffolding, still with zero `unsafe`.

## What's SIMD today vs. still scalar

**SIMD (`try_simd_alu_step`, `std::simd::Simd<u32, 16>`/`Mask<i32, 16>`):**
- Precondition: every active lane shares the same `pc` (`VecCpu::lanes_converged`) *and*
  independently fetches the identical instruction word from its own `Mmu` at that `pc` (checked,
  not assumed — see "Why fetch is still per-lane" below).
- Payload: `Inst::OpImm`/`Inst::Op` for the ten `AluOp`s with a clean packed form — add, sub, and,
  or, xor, sll, srl, sra, slt, sltu (this covers both 32-bit and their C-extension forms, since
  `decode_compressed` already canonicalizes e.g. `c.addi`/`c.and`/`c.srli` down to
  `Inst::OpImm`/`Inst::Op`). Decoded once, executed as one masked `Simd<u32, 16>` op
  (`simd_alu`), pc advanced for all active lanes as one masked vector add.
- `VecCpu::simd_alu_steps` counts how many `step()` calls took this path — a diagnostic used by
  the fast-path test and `examples/bench.rs`, not part of the correctness contract.

**Still scalar (`step_lane`, unchanged):**
- Any pc or instruction-word divergence (`try_simd_alu_step` returns `false` having mutated
  nothing, so falling through and re-fetching is free of side effects — `ifetch16` is a pure load).
- All memory ops (`Load`/`Store`/`LrW`/`ScW`/`AmoW`) — no shared/interleaved MMU yet (see below).
- All control flow (`Branch`/`Jal`/`Jalr`) — even when lanes agree on the branch outcome today,
  computing "did every lane agree" and then packing the pc update is future work, not yet done.
- `Mul`/muldiv (MUL/MULH*/DIV/DIVU/REM/REMU) — no packed form exists for these regardless of
  convergence (architecture.md §2); always scalarized, exactly as designed originally.
- `Ecall`/`Ebreak`/`Fence`/CSR-privileged/`Illegal` — one-off control instructions, not worth a
  vector form.

**Why fetch is still per-lane (the biggest remaining gap):** `try_simd_alu_step` calls
`bus.ifetch16` once per *active lane* (not once for the whole group), because each lane still owns
an independent `fs_mmu::Mmu` (requirement 3's original "simplest correct model", never revisited).
It only saves the *decode* (run once, not `LANES` times) and the *ALU op itself* (one packed op,
not `LANES` scalar ones) — not the fetch. Since fetch is a significant fraction of per-instruction
cost, this caps the realistic speedup well below `LANES`x until the interleaved shared-memory
design in "3. Memory" below lands and a converged group can fetch once for the whole group instead
of once per lane.

## Benchmark (`examples/bench.rs`)

`cargo run --release --example bench -p fs-vec` runs the same RV32IM counting loop (fixed trip
count → lanes stay pc-converged throughout; ~83% of dynamic instructions are ALU) on two engines
that share the exact same decode/execute code paths:
- `VecCpu::step` (SIMD fast path engages for every converged ALU instruction), vs.
- `LANES` independent `fs_riscv::Cpu`s stepped one instruction at a time (what `VecCpu::step` did
  before this fast path existed).

Measured on this machine (release build, `std::time::Instant`, 16 lanes × 2000 loop iterations ×
40 repeats = 15,363,200 lane-instructions on both sides — verified equal before computing a rate):

```
SIMD fast path:    15,363,200 lane-instructions in ~113–163ms  ≈ 95M–136M lane-instr/sec
scalar-over-lanes: 15,363,200 lane-instructions in ~233–336ms  ≈ 46M– 66M lane-instr/sec
speedup:           ~1.9x–2.1x across repeated runs
```

A ~2x, not ~16x, speedup is expected and consistent with "why fetch is still per-lane" above: the
fetch — still done `LANES` times regardless of the fast path — is not eliminated yet, so this
measures the win from eliminating `LANES`-fold redundant decode + scalarized ALU execution alone.

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

### 2. Lockstep fetch/decode

Converged lanes (same `pc`, checked today by `VecCpu::lanes_converged`, which will become a
`vpcmpeqd` + `kortestw` against a broadcast of lane 0's `pc`) fetch and decode **once**, not once
per lane — the whole point of vectorizing. `decode`/`decode_compressed` still run scalar (they are
control-flow-heavy bit-twiddling, not a good SIMD target), but they run *once per converged group*
instead of once per lane. Divergent lanes (a k-mask subset whose `pc` disagrees with the group)
fall out of the vector step and are scalar-executed via the same `step_lane` that exists today,
then re-converge naturally once their `pc` catches back up to the majority (or they get folded
into a new converged group next step).

### 3. Memory: same-address fast path vs. gather/scatter

Today each lane owns an independent `fs_mmu::Mmu` (requirement 3's "simplest correct model").
The AVX-512 target instead uses **one shared interleaved memory region** (architecture.md §3,
§4), because the whole reset/dirty-tracking story depends on lanes sharing physical layout:

- `host_word_index = guest_word * 16 + lane` — all 16 lanes' copies of one guest word are
  contiguous, so a single `vmovdqa32`/`vmovdqa32` store touches all 16 lanes' copy of that word
  in one instruction.
- Two interleaved planes per guest word: one 64-byte line of 16 permission `u32`s, immediately
  followed by one 64-byte line of the 16 content `u32`s. A single `vpcmpd`/`vptestmd` against the
  permission line validates all 16 lanes' R/W/X/RAW bits at once; only on failure do we fall back
  to a per-lane fault path (masked, using the failing lanes' k-mask).
- **Fast path** (architecture.md §2): if every active lane's effective address for this
  instruction is identical (the common case — lockstep lanes running the same code against
  byte-identical snapshots almost always compute the same address), a single aligned
  `vmovdqa32`/`vmovdqa32` load/store services all 16 lanes. Detecting this is one
  `vpbroadcastd` of lane 0's address + `vpcmpeqd` + `kortestw` (all-lanes-equal check) before
  issuing the memory op.
- **Divergent path**: when addresses differ across lanes, fall back to
  `vpgatherdd`/`vpscatterdd` with the address vector and the active k-mask as the gather/scatter
  mask. Zen 4 caveat (architecture.md §2, §27): gather/scatter decode to many more uops than on
  the Xeon Phi Falk benchmarked; **re-benchmark the ~4x same-address-vs-divergent gap on the
  actual host (7945HX) before tuning around it** — do not assume Falk's 35-vs-157-cycle numbers
  transfer.
- Divergence is the primary cost lever to minimize (not just tolerate): the interleaved layout
  and the same-address check exist specifically so the *common* case (lockstep, byte-identical
  lanes) stays on the cheap path, and only genuinely-divergent lanes (different syscall args,
  different branch outcomes touching different addresses) pay the gather/scatter cost.

### 4. k-mask predication

Every vector ALU/load/store op is predicated by the current active k-mask (`__mmask16`), so a
masked-off lane's destination register/memory is untouched by that instruction — the AVX-512
equivalent of today's `if self.active[lane] { step_lane(...) }` guard, done as one masked vector
instruction instead of a per-lane Rust `if`. `VecCpu::active` is designed to become exactly this
mask (see §1).

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

The converged-lane ALU fast path proves the shape works (decode once, execute once as a packed
op, mask-predicate the writeback) but is still a long way from the architecture.md §2 target:

1. **Interleaved shared MMU** (biggest gap, see "Why fetch is still per-lane" above). Each lane
   still owns an independent `fs_mmu::Mmu`, so fetch — and every load/store — is `LANES` separate
   calls regardless of convergence. The AVX-512 target's interleaved memory (§3 below) turns a
   converged group's fetch into one shared read instead of 16 identical ones.
2. **Gather/scatter for divergent memory** (§3 below). Not attempted at all yet — `Load`/`Store`/
   `LrW`/`ScW`/`AmoW` always fall through to scalar `step_lane`, converged or not.
3. **k-mask-predicated branches/jumps**. `Branch`/`Jal`/`Jalr` always scalarize today even when
   every lane agrees on the outcome; packing "did every lane branch the same way" + a masked pc
   update is straightforward follow-on work using the same `Mask<i32, LANES>` machinery
   `try_simd_alu_step` already has.
4. **Masked scalarize-16 for MULH*/DIV/REM**. Still exactly the `muldiv` scalar loop, as designed
   from the start (§5 below) — there is no packed form to move to regardless of convergence, only
   the *extraction*/*repacking* glue changes when this becomes real AVX-512 (`vpextrd`/`vpinsrd`
   instead of Rust array indexing).
5. **Real AVX-512 registers/intrinsics**. `std::simd::Simd<u32, 16>` is portable — it does not
   guarantee it compiles to a single `zmm` register + AVX-512 instructions on this host; it may
   lower to multiple narrower vector ops depending on target features enabled at compile time.
   Confirming/forcing actual `zmm` codegen (`RUSTFLAGS="-C target-feature=+avx512f"` or explicit
   intrinsics) is unverified — see "Where `unsafe`/intrinsics will go" below.

## Where `unsafe`/intrinsics will go

This crate is `#![forbid(unsafe_code)]` — the converged-lane ALU fast path added above is built
entirely on safe `std::simd`/portable_simd (decision #23 explicitly allows this over raw
intrinsics, and it needed zero `unsafe`). When the full AVX-512 executor is built out (the
interleaved MMU, divergent-address gather/scatter, k-mask-predicated branches), decision #23 still
allows either staying on `std::simd`/portable_simd on nightly (which does have safe
`Simd::gather_or`/`Simd::scatter` helpers, but they index into a plain Rust slice by offset, not a
fault-checked guest address through the permission/RAW-tracking `Mmu` this crate needs), or
dropping to raw `core::arch::x86_64` AVX-512 intrinsics for the actual `vpgatherdd`/`vpscatterdd`
hardware instructions once the interleaved memory layout exists to gather/scatter against:

- A new module (e.g. `fs_vec::avx512`, likely a separate crate or `#[cfg(target_feature =
  "avx512f")]`-gated module) will hold the only `unsafe` blocks in the vectorized path: the
  `_mm512_load_epi32`/`_mm512_store_epi32` same-address fast path, `_mm512_i32gather_epi32`/
  `_mm512_i32scatter_epi32` for the divergent path, `_mm512_cmpeq_epi32_mask`/`_mm512_kortestc`
  for convergence/fast-path detection, and `_mm512_mask_*` predicated ALU ops.
- Everything feeding those intrinsics (decode, address computation, the k-mask *value* itself as
  `[bool; 16]` vs. `__mmask16`) stays safe Rust; only the raw intrinsic calls are `unsafe`, kept
  behind a narrow, audited boundary — matching decision #45 ("targeted `unsafe` in profiled hot
  spots ... only when justified", "never in the decoder ... without a fight").
- DIV/REM/MULH* scalarization and any lane that falls out of the k-mask remain calls into this
  crate's existing safe `step_lane`/`muldiv` — no unsafe code is needed for the scalar fallback
  itself, only for the gather/scatter/compare glue around it.
