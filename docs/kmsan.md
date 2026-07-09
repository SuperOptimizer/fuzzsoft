# Emulator-native KMSAN — staged plan

Value-taint tracking (uninitialized-*value* use) on **uninstrumented** guests. From a 2-lens design
workflow + adversarial synthesis (2026-07-08), verified against the actual source. Mode-gated
(default runs pay nothing), deterministic, **byte-granular** (no finer ground-truth claim than
`PERM_RAW`). Distinct from `PERM_RAW`'s ASAN-strict fault-on-first-read: KMSAN *permits* reading
uninit and reports only at **consumption** — so the two are separate campaigns, never simultaneous on
one access.

## Load-bearing decisions (verified against the repo)

- **Register shadow = `Cpu.regs_taint: Option<Box<[u32;32]>>`** (byte-taint: bit at 0/8/16/24 = that
  byte of the value is uninit), mirroring the existing `cmplog: Option<Vec<..>>` idiom. NOT a bare
  `[u32;32]` — `Cpu` is `clone()`d **every fuzz case** (`main.rs` `*cpu = golden_cpu.clone()`), so a
  bare array is a 128-byte memcpy per case even with KMSAN off; the `Option` keeps the off-cost at one
  pointer. `VecCpu.regs_taint: [[u32;16];32]` (SoA twin; one packed OR in `try_simd_alu`).
- **Memory shadow = a spare bit in the existing `perms` byte, `PERM_VTAINT = 1<<5`** — NOT a separate
  `Vec<u8>` plane. Decisive: `fs-loader::load_into` sets `perms[off] = perm` **directly** (bypassing
  `Mmu::write`, the only RAW-clearing path), so a *separate* shadow array would default to all-tainted
  for the whole loaded kernel image → a day-one false-positive storm. A reused bit is safe by
  construction (existing callers never set bit 5; `perms`≥0 for any accessible byte leaves taint=0 for
  free). Zero extra memory, and CowRam's page-lazy-alloc already covers it. `Mmu::write`'s unconditional
  `(*p|READ)&!RAW` stays untouched; a KMSAN-gated `write_shadow` fires alongside the store. Replicate
  across `Mmu` / `Golden` (read side) / `CowRam` (both).
- **Report-and-stop** (`Trap::KmsanTainted{pc,rs1,rs2,taint_a,taint_b}`), NOT an accumulating log —
  matches the codebase's fault-shaped bug-oracle idiom (cmplog's drain is a coverage side-channel, not
  an oracle).
- **exec_one seam**: KMSAN wires taint arms next to the existing `wr_reg`/`load`/`store` in `step`'s
  match — the same lines the JIT Stage 0 `exec_one` extraction touches. Land KMSAN build **after** JIT
  Stage 0 so it rides `exec_one` (a rebase-coordination issue, not a design conflict).

## Propagation (Stage 0 = pure `alu_taint`/`muldiv_taint(op, a_val,a_taint,b_val,b_taint)`, unit-testable)

- v0 (sound-enough, ship first): Xor/shift/Slt/Add/Sub = OR of operand byte-masks; And/Or = OR (accept
  documented over-taint); Mul/Div/Rem = taint-all if either operand tainted; Lui/Auipc = 0; Load =
  gather byte-taint from `PERM_RAW` (Stage 1) then `|PERM_VTAINT` (Stage 2), sign/zero-extend the taint
  mirroring the value; Store = scatter the low-N reg byte-taint.
- **v1 refinements (Stage 3, not someday — both designs flag these as the dominant FP sources):**
  - **Add/Sub: byte-position carry-smear** — plain OR is UNSOUND (a tainted low byte's carry can flip a
    "clean" high byte). Smear the byte-mask upward (`m|=m<<1; m|=m<<2` over the 4 byte-positions) —
    Valgrind-Memcheck-style, sound, byte-granular.
  - **And/Or: known-byte clearing** — a byte forced by a known-0 (AND) / known-1 (OR) operand byte is
    untainted regardless of the other operand.
  - **Register-shift (`Op` form): if the shift-amount `rs2` is tainted → taint-all** (any of 32 amounts
    could apply); immediate-form shifts stay exact.
  - **Sra / sign-extending load**: propagate the shadow's top-byte taint into filled high bytes.

## Stages (each gated on a synthetic RV32 test with a POSITIVE and NEGATIVE control — asymmetric testing is how the loader-seeding FP storm would slip through)

- **Stage 0** — pure `alu_taint`/`muldiv_taint` functions + per-op unit tests. Zero Cpu/Mmu touch.
- **Stage 1 (FIRST INCREMENT)** — `regs_taint: Option<Box<[u32;32]>>` + `set_kmsan`/`kmsan_enabled`
  (mirror `set_cmplog`), wire v0 rules into `step`'s Op/OpImm/Mul arms, load-taint via a new additive
  `Mmu::read_raw_state` (does NOT touch `read`/`read_bytewise`), branch-on-tainted → `Trap::KmsanTainted`.
  NO memory shadow, NO refinements, NO checkpoints beyond Branch. Validate: positive (RAW byte →
  Lw → Add → Branch fires), negative (store-first → doesn't fire), clean-regression (off = zero behavior
  change + clone cost unchanged). Independently mergeable, touches no write path.
- **Stage 2** — `PERM_VTAINT` spare bit + `write_shadow`; loads OR in both RAW and VTAINT (this is what
  makes it *value* taint, not just byte taint). Extend to store-then-reload-then-branch round trip.
- **Stage 3** — the v1 precision refinements above (carry-smear, known-byte, tainted-shift, Sra).
- **Stage 4** — more checkpoints (independently landable once Stage 2 exists): Jalr-target-tainted,
  syscall-return (`a0`) tainted, and `copy_to_user`/`put_user` **sink hooks** via a new
  `fs-san::PcHooks` hook kind (OR-reduce the memory shadow over `[ptr,ptr+len)`).

Origin tracking (which alloc first introduced a taint) is out of scope (a 4th-plane stretch).

## T3.1 outcome — live oracle wired, but Stage 1 is DORMANT on a real kernel (2026-07-08)

`--kmsan` is wired end-to-end (`Cpu::kmsan_hit: Option<KmsanReport>` stashed by `finish_exit`
alongside its `SysExit::Halt(pc)`, drained by the fuzz loop, minimized + reproduced via the same
`crashes/` machinery as the kernel-crash oracle, mutually exclusive with `--sanitize`). The unit-level
positive/negative controls (synthetic `PERM_RAW` injected directly via `Mmu::protect`, bypassing any
allocator) all pass, proving the checkpoint → stash → report pipeline is correct when taint exists.

**But the empirical false-positive measurement surfaced a bigger finding than expected: it isn't a
false-positive storm, it's zero hits of *any kind* — and that's for a structural reason, not a
precision one.** `Cpu::regs_taint` can only become nonzero via the `Load` arm's gather from
`PERM_RAW` (Stage 1 has no memory shadow yet, no other entry point). `PERM_RAW` itself is set in
exactly one place reachable in this codebase: `fs-san`'s allocator hooks (`fs-san/src/alloc.rs`),
gated behind `--sanitize` — which `--kmsan` is (by this same design doc) forbidden to run
alongside. `fs-cli`'s boot sequence `protect()`s the *entire* guest RAM `READ|WRITE|EXEC` (no RAW)
before anything loads, and `fs-san/src/pages.rs` deliberately stamps page-granularity
(re)allocations `READ|WRITE` with no RAW too (its own false-positive-avoidance for the OOB/UAF
oracle). So a `--kmsan`-only run has **no live taint source at all**: 0 hits over 4000 clean-kernel
cases, and — confirmed by a one-off diagnostic run with `--sanitize` also enabled (its slack-byte
`PERM_RAW` stamping is the one real producer) — still 0 hits over 3000 more cases, because legitimate
kernel code never reads that OOB slack padding. The observed zero rate is therefore **guaranteed by
construction, not evidence Stage 1's v0 over-tainting rules are precise enough** — Stage 1 cannot
currently produce a true positive OR a false positive on a live kernel; it is a correctly-wired but
dormant detector.

**Implication for Stage 2:** the priority isn't the Stage 3 precision refinements (carry-smear,
known-byte clearing) this doc originally slated next — a live kernel never reaches them because no
register ever gets tainted in the first place. **Stage 2's `PERM_VTAINT` memory shadow is the
blocking increment**, not just for round-trip store/load taint fidelity, but because it needs its
*own* independent seeding path (e.g. tag freshly-`kmalloc`'d/`alloc_pages`'d payload bytes VTAINT at
allocation, independent of `--sanitize`'s RAW/OOB bookkeeping) to give `--kmsan` any live signal at
all. Re-run this same false-positive measurement once Stage 2 lands — that is the first point at
which "0 hits" or "hit storm" becomes a meaningful precision result rather than a tautology.

## T3.2 outcome — Stage 2 landed, live source confirmed, ONE root-caused FP class dominates (2026-07-08)

Stage 2 shipped exactly as scoped: `PERM_VTAINT` (fs-mmu), `Bus::write_shadow` (store-scatter,
exact overwrite) + `read_raw_state` now ORs RAW|VTAINT (load-gather), `Cpu::store_taint` wired into
`exec_one`'s `Store` arm, and a brand-new independent seeding path in `fs-cli`'s `KmsanCtx`: it
reuses `fs-san`'s existing `kmalloc`/`kmem_cache_alloc` PC-hooks (a new `PcHooks::on_kmsan_alloc_pc`
query, `fs-san/src/hooks.rs`, captures `gfp_flags` from the conventional `a1` register alongside the
existing size/cache-pointer capture) and stamps the returned, not-yet-written payload `PERM_VTAINT`
via a new `Mmu::set_vtaint` bulk primitive — skipping `__GFP_ZERO` allocations (`fs_san::GFP_ZERO =
0x100`, pinned against this build's `gfp_types.h`). **A real, load-bearing bug was found and fixed
along the way**: `fs_platform::Machine`/`CowMachine` (the actual `Bus` impls `run_case` drives)
never overrode `Bus::read_raw_state`/`write_shadow`, so they silently fell back to the trait's
default (always-clean/no-op) — meaning `--kmsan`'s load-taint gather was **structurally dead on any
real kernel run, independent of T3.1's taint-source finding**. Fixed by forwarding both methods to
`self.ram` (mirroring the existing `fast_ptr` forwarding pattern) — without this, Stage 2 would have
measured "0 hits" for a THIRD, unrelated reason.

**The false-positive measurement (the real deliverable): two independent clean-kernel runs, 8000
cases total (`firmware/Image`, `seed 1 × 5000` + `seed 2 × 3000`), converge on exactly ONE
root-caused finding, not a storm of distinct ones:**

- Run 1 (seed 1, 5000 cases): 364 `KmsanTainted` halts, **1 unique pc**; 8825 allocations tainted
  (2.98 MB), 1073 `__GFP_ZERO` allocations correctly skipped.
- Run 2 (seed 2, 3000 cases): 548 halts, **the same 1 unique pc**; 7591 tainted (3.41 MB), 1018
  skipped.
- Combined: 912/8000 cases (~11%) halt on KMSAN — but every single one is the SAME faulting `pc`,
  triggered by different socket-creating syscalls each time (`accept4`+`socketpair`+...,
  `listen`+`socketpair`+`bind`, ...) — i.e. **one recurring code path, not many distinct bugs**, and
  the corpus mutator naturally re-explores it once discovered (explaining the high recurrence rate
  from a single cause).

**Root-caused sample** (disassembled via `llvm-objdump --triple=riscv32` against
`build/linux-src/vmlinux`, symbol-resolved via `build/linux-src/System.map`): the faulting pc
(`0xc0b0581e`) is inside `_raw_spin_lock_irq`'s ticket-lock fast-path check —
`amoadd.w.aqrl a3,a1,(a0)` reads the lock word (this becomes `old` in `Cpu`'s `AmoW` arm, shadowed
from `Bus::read_raw_state` exactly like a plain load per Stage 1's existing rule); `a2 = a3 >> 16`
(the ticket we just reserved) and `a3 &= 0xffff` (the current head) are then compared
(`beq a2, a3`). Both registers are reported fully tainted (`0x0101_0101`) — and the *propagation
itself is exact*: `Srl`/`And`'s v0 OR-of-masks rule is sound for these ops, so this is not a Stage 3
carry/known-byte precision gap. **The bug is entirely in the SEED**: the lock lives inside a
heap object (a `struct sock`-shaped allocation is the common case for the triggering syscalls, but
the exact struct varies run to run — the class is the load-bearing fact, not one specific field) whose
initializer follows a Linux convention this pass's `KmsanCtx` doesn't yet model — confirmed directly
in `net/core/sock.c`: `sk_prot_alloc()` *always* strips `__GFP_ZERO` before calling the real
`kmem_cache_alloc()` (so the hook legitimately observes "not zeroed" and taints it, per its own
documented logic), then separately either zeroes non-lock fields via `sk_prot_clear_nulls()` or
(`sk_clone`'s path) `memcpy`s everything **except** a compiler-enforced `sk_dontcopy_begin`/
`sk_dontcopy_end` byte range that includes the lock — relying on a *later*, explicit
`sock_lock_init()` call to re-establish it. This is the same general class this doc's own `KmsanCtx`
doc comment pre-flagged before the measurement ran: **`kmem_cache` allocations whose lock/sync
fields are established by a `ctor` (persists correctly across every *reused* allocation from that
slab, never re-triggered by a plain `kmem_cache_alloc` return) or by an allocator-internal
"don't-copy-this-region, a dedicated init call handles it" convention are invisible to a seeding
model that only asks "was `__GFP_ZERO` set?"** — real Linux KMSAN's actual `kmsan_slab_alloc()` hook
independently special-cases exactly this (`cache->ctor` -> skip poisoning), which this finding
rediscovers empirically rather than by having read that code.

**Positive control:** the fs-riscv unit tests (`kmsan_stage2_store_scatter_then_reload_then_branch_fires`
/ `..._store_of_clean_value_clears_stale_vtaint` / `kmsan_disabled_never_sets_vtaint`) are the
clean, deterministic positive/negative/off controls for the mechanism itself (store→reload→branch
round trip Stage 1 could not do). The live-kernel finding above is the END-TO-END positive control
for the full pipeline (independent alloc-hook seeding → real kernel code → store/load propagation →
live oracle checkpoint) — it is a real, reproducible, root-caused signal, just one whose correct
disposition is "known seeding gap," not "kernel bug."

**Verdict: campaign-ready for the MECHANISM, not yet for unattended bug-hunting.** The shadow,
propagation, and independent seed all work exactly as designed, and — unlike a storm of many
distinct unexplained hits — this is ONE well-understood, tractable false-positive class with an
identified fix (extend `KmsanCtx` to skip tainting `kmem_cache_alloc` returns whose cache has a
ctor, mirroring the `__GFP_ZERO` check; needs a `kmem_cache::ctor` guest-memory read + sanity bound,
the same shape as `KMEM_CACHE_OBJECT_SIZE_OFFSET`). Until that lands, a live campaign would spend a
meaningful fraction of cases (~11% here) re-discovering this one class rather than finding new
signal — usable for validating the mechanism, not yet for autonomous triage. This reframes Stage 3:
the ALU precision refinements (carry-smear, known-byte clearing) are NOT what's blocking real usage
— the seed's allocator-convention coverage is.

## T3.2.5 outcome — ctor-skip closes the one root-caused FP class (2026-07-08)

Closed exactly the gap T3.2's outcome section identified: `kmem_cache_alloc` returns whose cache
has a non-null constructor are now skipped by `--kmsan`'s seeding, mirroring the existing
`__GFP_ZERO` skip and real Linux KMSAN's own `kmsan_slab_alloc()` special-case.

**`ctor` offset derivation.** `struct kmem_cache` (`build/linux-src/mm/slab.h`) lists, after
`object_size` (offset 16, the pre-existing `KMEM_CACHE_OBJECT_SIZE_OFFSET`, unchanged): `struct
reciprocal_value reciprocal_size` (8B — `u32 m` + `u8 sh1` + `u8 sh2`, padded to 4-byte alignment),
`unsigned int offset`, `unsigned int sheaf_capacity`, `struct kmem_cache_order_objects oo` (4B, one
`unsigned int x`), `struct kmem_cache_order_objects min`, `gfp_t allocflags`, `int refcount`, then
`void (*ctor)(void *object)` — no `CONFIG_`-gated fields intervene. Hand-summing RV32 sizes
(4-byte pointers/`unsigned int`/`unsigned long`, natural alignment) gives offset 16 + 4 + 8 + 4 + 4
+ 4 + 4 + 4 + 4 = **52**. Cross-checked (no RV32 cross-compiler available in this environment) by
compiling an equivalent struct with the host `gcc`, using fixed-width `uint32_t` stand-ins for
every RV32 4-byte type (pointers, `unsigned long`, `gfp_t`, `slab_flags_t`) and reading
`offsetof(...)`: this independently reproduced `object_size` at 16 (matching the pre-existing,
already-trusted constant — a self-consistency check on the method itself) and `ctor` at 52. Pinned
as `fs_san::linux::KMEM_CACHE_CTOR_OFFSET = 52`, with a unit test
(`kmem_cache_ctor_offset_matches_documented_derivation`) that also re-derives it arithmetically from
`KMEM_CACHE_OBJECT_SIZE_OFFSET` plus the documented per-field byte count, so the two constants can't
silently drift apart. Re-derive both for any other kernel version/config, exactly like the existing
`object_size`/`GFP_ZERO` constants' doc comments already require.

**The seeding-skip change** (`crates/fs-cli/src/main.rs`, localized to `KmsanCtx`'s
`KmsanAllocEvent::Cache` arm in `run_case` — the same block that already reads `object_size` and
checks `__GFP_ZERO`): after the `object_size` sanity-bounded read succeeds, additionally read
`cachep->ctor` at `KMEM_CACHE_CTOR_OFFSET` and treat it as "has a constructor" if the read value is
`>= fs_san::linux::PAGE_OFFSET` (a real kernel VA never being dereferenced, only compared as an
integer — no unsafe, no indirection through it). If so, count it in a new `ctor_skipped` stat and
skip the `set_vtaint` call entirely (same `else if` ladder as the `__GFP_ZERO` check, so a
`__GFP_ZERO` + ctor cache is counted once, under `zeroed_skipped`). The plain `kmalloc`
(`KmsanAllocEvent::Sized`) path is untouched — it has no `cache` pointer and therefore no `ctor` to
check, exactly as scoped.

**FP measurement — BEFORE vs. AFTER, same methodology as T3.2** (clean `firmware/Image`, `--kmsan`,
seed 1 × 5000 + seed 2 × 3000 = 8000 cases total, run by invoking the built `fuzzsoft` binary with
cwd = the main checkout so the default `firmware/`/`build/linux-src/System.map` relative paths
resolve — `build/`/`firmware/` are gitignored artifacts that live only in the main checkout, not
this worktree):

| | seed 1 (5000 cases) | seed 2 (3000 cases) | combined |
|---|---|---|---|
| **BEFORE (T3.2)** | 364 halts, 1 unique pc | 548 halts, 1 unique pc | 912/8000 (~11%) |
| **AFTER (T3.2.5)** | **0 halts, 0 unique pc** | **0 halts, 0 unique pc** | **0/8000 (0%)** |

Seeding stats confirm the mechanism is still live, just narrower: seed 1 tainted 12103 allocations
(3.04 MB), skipped 1514 for `__GFP_ZERO`, and now additionally skips 657 for a ctor-having cache;
seed 2 tainted 10148 (2.86 MB), skipped 1295 `__GFP_ZERO`, 924 ctor. `kmem_cache_alloc`
size-unavailable stayed 0 in both runs (the pre-existing `object_size` offset is unaffected).
Coverage/corpus/kernel-crash counts are bit-for-bit identical to the equivalent `--sanitize` run at
the same seed/cases (14892 buckets / 958 corpus for seed 1), confirming this change altered only
KMSAN's seeding decision, nothing about execution. **The `pc=0xc0b0581e` `_raw_spin_lock_irq` FP
class from T3.2 is gone: 0 recurrences in either run.**

**Is the skip too broad?** Cross-checked against a same-seed/same-cases `--sanitize` run (mutually
exclusive with `--kmsan`, so run separately): 13670 total `kmem_cache_alloc` events were observed at
seed 1/5000-cases (`sanitizer: ... 13670 kmem_cache_alloc ...`), against which the 657 ctor-skips
measured on the identical seed/cases under `--kmsan` are **~4.8%** — a narrow, plausible fraction,
not a blanket "cache allocs never get tainted anymore." The large majority of `kmem_cache_alloc`
returns are still exposed to VTAINT seeding.

**Positive control — survives.** The fs-riscv unit-level mechanism tests
(`kmsan_stage2_store_scatter_then_reload_then_branch_fires`,
`kmsan_stage2_store_of_clean_value_clears_stale_vtaint`, `kmsan_positive_branch_on_uninitialized_load_traps`,
`kmsan_negative_load_after_store_does_not_trap`, `kmsan_disabled_never_sets_vtaint`,
`kmsan_disabled_is_behavior_preserving_and_never_allocates_regs_taint`,
`kmsan_live_oracle_stashes_report_in_finish_exit`) all still pass unmodified — none of them route
through `KmsanCtx`/the allocator hooks this change touches, so they're an independent confirmation
that the shadow, propagation, and checkpoint-trap plumbing are untouched by the ctor-skip; the skip
only narrows *seeding*, not detection. A genuinely fresh live "seeded-uninit, non-ctor" positive
control on the clean kernel was not constructed in this pass (0/8000 hits is the expected outcome
on a bug-free stock kernel by construction, not evidence against the mechanism — see the unit tests
above for that); building a deliberately-planted uninitialized-value kernel bug (parallel to
`scripts/build-buggy-kernel.sh`'s planted OOB-write bug, but for KMSAN) is flagged as a good
follow-up if an end-to-end "found a real bug" demonstration is wanted, but was out of scope here.

**Verdict: `--kmsan` is now campaign-ready.** The one root-caused FP class T3.2 measured (~11% of
clean-kernel cases, entirely one recurring code path) is eliminated (0/8000), the fix is narrowly
scoped (confirmed via the `--sanitize` cross-check, not a blanket cache-alloc taint suppression),
and the underlying mechanism (shadow, propagation, live oracle) is unmodified and still passes its
own unit tests. A live campaign can now run `--kmsan` unattended without spending cases
re-discovering a known, understood non-bug.

## Open questions (decide, don't assume)
Byte-taint packing (bits 0-3 vs byte-aligned 0/8/16/24 — recommend byte-aligned for Stage-3 shift math);
mode flag as runtime `Option` (start here, matches cmplog) vs const-generic (only if profiled);
coordinate the `exec_one` rebase with JIT Stage 0; Stage-2 `read_raw_state` interaction with the
software TLB (caches perms at translate time); Stage-4 syscall-return check keyed off `fs-san`'s
`Convention` table to avoid firing on void-returning syscalls.
