# fs-vec — design notes

`fs-vec` is the M4 vectorization foundation (`docs/architecture.md` §2, §8; `docs/decisions.md`
#3, #4, #23, #45). What's implemented *today* is a safe-Rust, scalar-over-lanes SoA executor: it
gets the state layout and masking contract right so that dropping in real AVX-512 execution later
is mechanical, not a rewrite. This document records the AVX-512 target the current code is
shaped for, and exactly what changes when we get there.

## What exists now

- `VecCpu::regs: [[u32; LANES]; 32]` — SoA register file, `regs[reg][lane]`.
- `VecCpu::pc: [u32; LANES]`, `VecCpu::active: [bool; LANES]` — per-lane PC and active mask.
- `VecCpu::step` loops `for lane in 0..LANES { if active[lane] { step_lane(lane, ...) } }`,
  each lane against its own `fs_mmu::Mmu`.
- `alu`/`muldiv` are byte-for-byte copies of `fs_riscv`'s private functions (same edge cases:
  DIV/0 → `0xffff_ffff`, REM/0 → dividend, `INT_MIN / -1` → DIV=`0x8000_0000`/REM=`0`, MULH* via
  i64/u64 widening).
- `#![forbid(unsafe_code)]` everywhere in this crate.

This is intentionally *not yet vectorized* — it is the scaffolding decision #45 asks for
("correctness-first ... targeted `unsafe` in profiled hot spots only when justified").

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

## Where `unsafe`/intrinsics will go

This crate is `#![forbid(unsafe_code)]` today — there is no SIMD yet, only the SoA shape. When
the AVX-512 executor is built (decision #23 allows either `std::simd`/portable_simd on nightly or
raw `core::arch::x86_64` AVX-512 intrinsics):

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
