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

## Open questions (decide, don't assume)
Byte-taint packing (bits 0-3 vs byte-aligned 0/8/16/24 — recommend byte-aligned for Stage-3 shift math);
mode flag as runtime `Option` (start here, matches cmplog) vs const-generic (only if profiled);
coordinate the `exec_one` rebase with JIT Stage 0; Stage-2 `read_raw_state` interaction with the
software TLB (caches perms at translate time); Stage-4 syscall-return check keyed off `fs-san`'s
`Convention` table to avoid firing on void-returning syscalls.
