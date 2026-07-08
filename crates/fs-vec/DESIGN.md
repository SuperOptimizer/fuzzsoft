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
  permission), `try_simd_load`/`try_simd_store` now fall through to these instead of declining
  outright — the shared fetch/decode/address-computation is kept either way. **Goal 2 (this
  round): the common case — every active lane's own address in-bounds, correctly aligned, and
  permission-checked — now stays fully vectorized too**, via `VecMmu::load_gather_fast`/
  `store_scatter_fast`: one `Simd<usize, LANES>` host-index computation (`word*LANES+lane`,
  vectorized) plus one `Simd::gather_select`/`scatter_select` call each against `content`/`perms`
  (portable_simd's safe, bounds-checked-internally stand-in for `vpgatherdd`/`vpscatterdd` — no
  `unsafe` needed) — no per-lane Rust loop at all when every lane succeeds. Only a lane that
  actually fails (out-of-bounds/misaligned/missing permission — the rare case) falls back to the
  exact per-lane `VecMmu::load_lane`/`store_lane` to get its precise `Fault`; that lane is halted
  individually (`VecCpu::halt`) without disturbing any other lane's `pc`/registers/mask. These two
  never decline as a whole (unlike the same-address pair) since per-lane faults are a
  fully-handled outcome, not a further reason to fall back.
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

`cargo run --release --example bench -p fs-vec` runs **three** RV32IM counting loops (each with a
fixed trip count → lanes stay pc-converged throughout) on the same two engines —
`VecCpu::step` vs. `LANES` independent `fs_riscv::Cpu`s each stepped one instruction at a time —
sharing the exact same shared-`VecMmu`/per-lane-`fs_mmu::Mmu` split the production code uses:

1. **ALU+fetch-bound loop** (~83% of dynamic instructions are ALU, no memory ops in the loop body)
   — isolates the fetch-deduplication + packed-ALU win.
2. **Same-address memory loop** (same shape, but the per-lane accumulator round-trips through ONE
   shared guest address every iteration instead of a register) — isolates the same-address
   `load_same`/`store_same` win.
3. **Divergent-address (gather/scatter) memory loop** (same shape, but each lane's accumulator
   round-trips through a DISTINCT per-lane guest address instead) — isolates
   `load_gather_fast`/`store_scatter_fast`'s win (see "Goal 2" below).

### ⚠️ Default `cargo build --release` does NOT use AVX-512 on this host — read this first

**Verified by disassembly** (`objdump -d --demangle`, `target-feature` grep), not assumed. A plain
`cargo build -p fs-vec --example bench --release` on this host (AMD Ryzen 9 7945HX,
`avx512f/bw/dq/vl/cd/vnni` all present per `lscpu`) targets the **generic x86-64 baseline**
(`rustc --print cfg -O` under that build shows only `sse`/`sse2`/`fxsr` — no `avx`, no `avx2`, no
`avx512*`). Disassembling `<fs_vec::VecCpu>::step` (which inlines `try_simd_alu`,
`fetch_converged`, `load_same`/`store_same`, etc.) from that binary: **682 `xmm` (128-bit)
instructions, 0 `ymm`, 0 `zmm`, 0 `%k0`–`%k7` mask registers.** Every one of this crate's
`Simd<u32, 16>` ops is being lowered to **4× 128-bit SSE2 operations**, not one 512-bit one — the
"16-wide vector" is real in the Rust type system but not on the actual hardware register file
under this build.

Building instead with `RUSTFLAGS="-C target-cpu=native"` (or explicitly
`-C target-feature=+avx512f,+avx512bw,+avx512dq,+avx512vl`) changes this completely: `rustc --print
cfg -O -C target-cpu=native` now reports `avx`, `avx2`, `avx512f`, `avx512bw`, `avx512dq`,
`avx512vl`, `avx512cd`, `avx512vnni`, etc., and the *same* `<fs_vec::VecCpu>::step` disassembles to
**39 `zmm` instructions and 17 uses of `%k0`/`%k1`**, including real masked/predicated AVX-512:

```
vpbroadcastd %eax,%zmm0
vpandd (%r14,%rdi,4),%zmm0,%zmm1
vpcmpneqd %zmm0,%zmm1,%k0        ; convergence/permission compare -> k-mask
ktestw %k1,%k0
vpaddd %zmm1,%zmm0,%zmm0{%k1}    ; k-mask-PREDICATED add (masked ALU writeback)
vmovdqu32 %zmm0,(%rax,%rdi,1){%k1}  ; k-mask-PREDICATED store
vpcmpltud %zmm0,%zmm1,%k0        ; unsigned branch/gather-bounds compare -> k-mask
vpsrlvd/vpsravd/vpsllvd          ; per-lane variable shift (Sll/Srl/Sra)
```

This is genuine EVEX-encoded AVX-512 with k-mask predication — not a fluke of one instruction, and
not just in `simd_alu`: the permission-check/gather-index machinery this task's Goal 2 added
(`load_gather_fast`/`store_scatter_fast`, below) also compiles down to `vpcmpneqd`/`vpcmpltud`/
`vpcmpgtd`/`vpmovm2d`/masked `vmovdqu32`/`vpaddd{%k1}`.

**Bottom line: `std::simd::Simd<u32, 16>` on this nightly does NOT default to real AVX-512 zmm
codegen — you must opt in with `-C target-cpu=native` (or explicit `+avx512*` features) or you are
silently running 4-lane SSE2 under a 16-lane API.** Because this crate cannot add a workspace-wide
`.cargo/config.toml` (out of scope — would affect every other crate in the workspace), there is no
in-repo mechanism forcing this today; anyone benchmarking or shipping `fs-vec` must set
`RUSTFLAGS="-C target-cpu=native"` explicitly (or the crate silently reverts to the SSE2 numbers
below).

### Measured numbers: default (SSE2) build vs. `target-cpu=native` (AVX-512) build

Measured on this machine (release build, `std::time::Instant`, 16 lanes × 2000 loop iterations ×
40 repeats per program, both engines verified to retire the same lane-instruction count before
computing a rate). Two full build+run passes, one per `RUSTFLAGS` setting:

```
=== Default build (no RUSTFLAGS — SSE2 only, 0 zmm/k-mask in the disassembly) ===
ALU+fetch-bound loop:
  SIMD fast path:    15,363,200 lane-instructions ≈ 170M–248M lane-instr/sec
  scalar-over-lanes: 15,363,200 lane-instructions ≈  46M– 67M lane-instr/sec
  speedup:           ~3.7x

Same-address memory loop:
  SIMD fast path:    7,684,480 lane-instructions  ≈ 165M–238M lane-instr/sec
  scalar-over-lanes: 7,684,480 lane-instructions  ≈  48M– 53M lane-instr/sec
  speedup:           ~3.2x–4.5x

Divergent-address (gather/scatter) memory loop:
  SIMD fast path:    7,683,200 lane-instructions  ≈ 172M–197M lane-instr/sec
  scalar-over-lanes: 7,683,200 lane-instructions  ≈  53M lane-instr/sec
  speedup:           ~3.25x–3.72x

=== RUSTFLAGS="-C target-cpu=native" build (real AVX-512 — 39 zmm + 17 k-mask ops confirmed) ===
ALU+fetch-bound loop:
  SIMD fast path:    15,363,200 lane-instructions ≈ 406M–591M lane-instr/sec
  scalar-over-lanes: 15,363,200 lane-instructions ≈  45M– 66M lane-instr/sec
  speedup:           ~8.9x

Same-address memory loop:
  SIMD fast path:    7,684,480 lane-instructions  ≈ 565M–610M lane-instr/sec
  scalar-over-lanes: 7,684,480 lane-instructions  ≈  43M– 53M lane-instr/sec
  speedup:           ~9.6x–11.5x

Divergent-address (gather/scatter) memory loop:
  SIMD fast path:    7,683,200 lane-instructions  ≈ 307M–431M lane-instr/sec
  scalar-over-lanes: 7,683,200 lane-instructions  ≈  52M lane-instr/sec
  speedup:           ~5.9x–8.4x
```

**The `target-cpu=native` build is ~2.2x–2.5x faster in absolute lane-instr/sec than the default
build** (e.g. ALU+fetch: ~591M vs. ~248M at their respective best runs) — this is the real,
hardware-confirmed AVX-512 win; the ~3.2x–4.5x speedups reported by earlier revisions of this
document were all measured on the **default SSE2 build** and, while still a genuine win (from
fetch/decode deduplication and per-lane-`Mmu`-overhead removal — see below — not from vector
width), significantly *understate* what this code is actually capable of once it is compiled to
target the hardware it is nominally "16-wide" for.

**A caution the numbers above also surface**: even the SSE2 (4-lane-wide, non-AVX-512) default
build gets a ~3.2x–4.5x speedup over the scalar-over-lanes baseline, despite processing the same
16 logical lanes in groups of 4 instead of one group of 16. This means a meaningful share of this
crate's speedup — enough to produce a "looks like it's working" ~3-4x number even with zero real
512-bit hardware parallelism — comes from **fetch/decode deduplication and avoiding `LANES`
separate `fs_mmu::Mmu` structs' overhead**, not from vector width per se. The `target-cpu=native`
delta (~2.2x–2.5x on top of that) is the part that is specifically attributable to real AVX-512,
and is the number to trust when reasoning about "how much wider hardware parallelism buys us."

**This beats the original ~2.55x–2.8x** (recorded when `Branch`/`Jal`/`Jalr` always scalarized even
on fully pc-converged loops — see git history for the prior numbers), consistent with removing
exactly the bottleneck DESIGN.md itself used to call out as "likely higher-value than
gather/scatter": both the ALU and same-address loops execute one `bge` and one `jal` every
iteration, and — before `try_simd_branch`/`try_simd_jal` existed — *every single iteration* fell
back to `step_lane`'s fully-scalar per-lane fetch/decode/execute for those two instructions, even
though pc stayed perfectly converged the entire run. `examples/bench.rs`'s own diagnostic step
counters confirm this directly: the ALU+fetch-bound loop takes zero `step_lane` fallbacks for its
`bge`/`jal` (`branch/jump-path` step count equals the loop's trip count), leaving only MUL/DIV/REM
and atomics as the categories that must always scalarize regardless of convergence.

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
- **Divergent path — vectorized for the common case (Goal 2, this round)**
  (`VecMmu::load_gather_fast`/`store_scatter_fast`, called from
  `VecCpu::try_simd_gather_load`/`try_simd_gather_store`): computes every active lane's host index
  (`word*LANES+lane`, `Self::load_lane`'s own indexing, vectorized as one `Simd<usize, LANES>` —
  invalid lanes' `word` clamped to `0` first so the multiply/add can never overflow) and issues ONE
  `Simd::gather_select`/`scatter_select` against `content`/`perms` — `portable_simd`'s safe
  (bounds-checked internally against the slice length, per its own doc contract),
  no-`unsafe`-required stand-in for `vpgatherdd`/`vpscatterdd` — instead of `LANES` independent
  per-lane `load_lane`/`store_lane` calls, for the case where every active lane is in-bounds,
  correctly `size`-aligned, and permission-checked (verified in the disassembly: this compiles to
  real `vpcmpneqd`/`vpcmpltud`/`vpcmpgtd`/`vpmovm2d`/masked `vmovdqu32` zmm instructions under
  `-C target-cpu=native` — see the Benchmark section above). Only a lane that fast path couldn't
  resolve (the rare case: out-of-bounds/misaligned/unpermitted) is re-serviced individually via the
  exact per-lane `load_lane`/`store_lane` to get its precise `Fault`, exactly as the original
  fully-scalar `load_gather`/`store_scatter` (kept, unchanged, as this rare-path fallback and still
  directly exercised/tested) did for every lane before. `try_simd_load`/`try_simd_store` still call
  into this instead of declining outright on address divergence (or a same-address decline), so the
  shared fetch/decode/address-computation from the converged group is kept either way. A lane whose
  access faults is halted individually, without disturbing any lane that didn't fault.
  `crates/fs-vec/src/vec_mmu.rs`'s `gather_fast_and_scatter_fast_never_disagree_with_the_scalar_per_lane_reference`
  property test fuzzes random per-lane addresses (in/out-of-bounds, aligned/misaligned,
  permitted/unpermitted, mixed within one call) to prove the vectorized fast path never disagrees
  with the scalar `load_lane`/`store_lane` reference it replaces, on both the success mask and the
  loaded/stored values.
- Divergence is still the primary cost lever to minimize (not just tolerate): the interleaved
  layout and the same-address check exist specifically so the *common* case (lockstep,
  byte-identical lanes) stays on the cheapest path. Measured on this host (7945HX,
  `examples/bench.rs`'s new divergent-address loop): the vectorized gather/scatter fast path is
  ~1.15x faster than the pre-Goal-2 fully-scalar loop on the default (SSE2) build (~3.25x → ~3.72x
  speedup over scalar-over-lanes) and ~1.4x faster under `-C target-cpu=native` (~5.87x → ~8.36x) —
  a real, measured win, though (as expected) still behind the same-address fast path's ~11x under
  AVX-512, since gather/scatter's permission/bounds-check-then-conditionally-touch-memory shape is
  inherently costlier than same-address's single shared read/write even when both are vectorized.

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

Fetch, ALU, branches/jumps, same-address memory, and now the common case of gather/scatter memory
are all vectorized (in safe Rust) and **confirmed to compile to real AVX-512 zmm/k-mask
instructions under `-C target-cpu=native`** (see the Benchmark section's disassembly evidence).
What's left:

1. **Masked scalarize-16 for MULH*/DIV/REM.** Still exactly the `muldiv` scalar loop, as designed
   from the start (§5 above) — there is no packed form to move to regardless of convergence, only
   the *extraction*/*repacking* glue changes when this becomes real AVX-512 (`vpextrd`/`vpinsrd`
   instead of Rust array indexing).
2. **No same-address fast path for `LrW`/`ScW`/`AmoW`.** These always scalarize via `step_lane`
   today, converged or not; a same-address `AmoW` (RMW) fast path is possible following the same
   shape as `try_simd_load`/`try_simd_store` but was not attempted here — atomics are rarer on a
   hot ALU/memory loop than plain load/store, so this was deprioritized.
3. **Gather/scatter's rare-fault lanes are still serviced one at a time.** ~~Gather/scatter is
   still `LANES` genuinely-independent per-lane `VecMmu` accesses~~ — **closed for the common case
   this round (Goal 2)**: `VecMmu::load_gather_fast`/`store_scatter_fast` now resolve every active
   lane in one `Simd::gather_select`/`scatter_select` call when every lane is in-bounds/aligned/
   permitted (measured ~1.15x–1.4x faster than the old fully-scalar loop — see the Benchmark
   section's new divergent-address-loop numbers). What remains scalar is only the genuinely rare
   per-lane fault case (out-of-bounds/misaligned/unpermitted), which still calls `load_lane`/
   `store_lane` individually to get the exact `Fault` — by construction this is the cold path, not
   the common one, so it was not worth vectorizing further.
4. **Real AVX-512 registers/intrinsics — now verified, not just possible.** `std::simd::Simd<u32,
   16>` is portable — it does NOT default to a single `zmm` register + AVX-512 instructions on this
   host: a plain `cargo build --release` compiles every fast path in this crate down to 128-bit
   `xmm` (SSE2) ops only (0 `zmm`, 0 k-mask registers in the disassembly — verified this round, see
   Benchmark section). `RUSTFLAGS="-C target-cpu=native"` (or explicit
   `-C target-feature=+avx512f,+avx512bw,+avx512dq,+avx512vl`) is REQUIRED to get real `zmm`/k-mask
   codegen (confirmed: 39 zmm instructions + 17 k-mask ops in `<fs_vec::VecCpu>::step` under that
   flag) and is worth ~2.2x–2.5x in absolute throughput on top of the default build's already-real
   ~3.2x–4.5x speedup (which comes mostly from fetch/decode deduplication and avoiding `LANES`
   separate `fs_mmu::Mmu`s' overhead, not from vector width — see Benchmark section for the full
   breakdown). This crate cannot add a workspace-wide `.cargo/config.toml` (would affect every other
   crate); until upstream infrastructure changes, anyone building/benchmarking `fs-vec` for real
   must set `RUSTFLAGS="-C target-cpu=native"` explicitly or silently get the SSE2 numbers instead.
   Raw hardware intrinsics (`core::arch::x86_64`) remain unnecessary for now — portable_simd's safe
   `Simd::gather_select`/`scatter_select` (used by Goal 2's new fast path) already lower to real
   `vpgatherdd`/`vpscatterdd`-adjacent EVEX code without any `unsafe` — see "Where `unsafe`/
   intrinsics will go" below for when raw intrinsics might still become necessary.

## Where `unsafe`/intrinsics will go

This crate is `#![forbid(unsafe_code)]` — `VecMmu` and every fast path added above (fetch, ALU,
branches/jumps, same-address memory, gather/scatter memory) are built entirely on safe
`std::simd`/portable_simd (decision #23 explicitly allows this over raw intrinsics, and none of it
needed `unsafe`) and, verified this round, this genuinely does compile down to real EVEX-encoded
`zmm`/k-mask AVX-512 under `-C target-cpu=native` (see the Benchmark section) — including the
gather/scatter path: `Simd::gather_select`/`scatter_select` (used by `load_gather_fast`/
`store_scatter_fast`, Goal 2) are safe, bounds-checked-against-the-slice-length by construction, and
still lower to real gather/scatter-family EVEX instructions with `vpcmpltud`/`vpcmpgtd`/`vpmovm2d`
building the enable mask. This supersedes an earlier (pre-verification) note here claiming
`Simd::gather_or`/`scatter` couldn't service a fault-checked guest address — they can, once the
bounds/permission check is done as a separate vectorized step first (exactly what
`gather_addr_plan` + the `valid`/`ok` masks in `vec_mmu.rs` now do) rather than expecting the
gather itself to carry permission semantics.

Decision #23 still allows dropping to raw `core::arch::x86_64` AVX-512 intrinsics later if
profiling ever shows portable_simd's codegen leaving real performance on the table versus hand-written
intrinsics (e.g. if `gather_select`'s internal bounds-check overhead turns out to cost more than a
hardware gather's own fault-suppression would) — that remains a possible future step, not a known
current gap:

- A new module (e.g. `fs_vec::avx512`, likely a separate crate or `#[cfg(target_feature =
  "avx512f")]`-gated module) would hold the only `unsafe` blocks in the vectorized path: the
  `_mm512_load_epi32`/`_mm512_store_epi32` same-address fast path (this crate's `load_same`/
  `store_same` are the safe-Rust reference implementation these intrinsics must match),
  `_mm512_i32gather_epi32`/`_mm512_i32scatter_epi32` for the divergent path (`load_gather_fast`/
  `store_scatter_fast` are the reference here now), `_mm512_cmpeq_epi32_mask`/`_mm512_kortestc` for
  convergence/fast-path detection, and `_mm512_mask_*` predicated ALU ops.
- Everything feeding those intrinsics (decode, address computation, the k-mask *value* itself as
  `[bool; 16]` vs. `__mmask16`) stays safe Rust; only the raw intrinsic calls are `unsafe`, kept
  behind a narrow, audited boundary — matching decision #45 ("targeted `unsafe` in profiled hot
  spots ... only when justified", "never in the decoder ... without a fight").
- DIV/REM/MULH* scalarization and any lane that falls out of the k-mask remain calls into this
  crate's existing safe `step_lane`/`muldiv` — no unsafe code is needed for the scalar fallback
  itself, only for the gather/scatter/compare glue around it.
