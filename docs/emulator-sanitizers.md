# Emulator-native sanitizers (uninstrumented targets)

The vision (decision #19): sanitization as an **emulator feature** on **uninstrumented** guest binaries —
no recompile, no source — so the same fuzzer can eventually target Windows/closed-source. The
kernel-instrumented builds (`Image.slubdebug`, `Image.dpalloc`, `Image.buggy`) are **validation
oracles only** — they need a recompile, so they can never be the strategy; they exist to test
*ourselves*. This doc records what is *correctly* buildable emulator-side, and — honestly — what is
not.

## KASAN (OOB + UAF) — correct-but-limited; a hard ceiling exists

Two tempting approaches are **rejected** on correctness grounds:
- **KFENCE-style relocation** (redirect `kmalloc`'s returned pointer to a guard-flanked region):
  rejected. The `kfree(Q)` pointer-rewrite is fixable, but SLUB's *own* internals compute
  `virt_to_head_page(ptr)` / `virt_to_phys(ptr)` on many paths *before* any hook runs (kfree's first
  act, kmemleak, compound/NUMA lookups, DMA-map). For those to be coherent, the redirected address
  needs a real, correctly-populated `struct page`/`slab` in the kernel's `mem_map` — which only
  exists for genuine physical frames the kernel already knows about. Forging that is
  kernel-version-specific struct reverse-engineering (real KFENCE reserves such a pool *at boot,
  inside the kernel*). Not purely-emulator-side.
- **In-place cross-object redzones**: empirically ~40% false positives (`docs/kernel-san.md`), and the
  mechanism is *fundamental* — stock SLUB packs objects with zero gap, so any guard byte past the
  bucket boundary is the first byte of a live neighbor. Real KASAN escapes this only by **widening
  `cache->size` at cache creation** — i.e. it is exactly as allocator-cooperative as `SLUB_DEBUG_ON`.

**Recommended (correct, zero false positives, buildable):**
1. **UAF via quarantine** (already built): on `kfree`, poison `[addr, addr+size)` no-access until the
   *same physical address* is handed back by a later `kmalloc`. Only ever touches an object's *own*
   payload after the kernel declared it dead → zero spatial false-positive surface. (Prereqs already
   flagged: free-at-return, exclude `SLAB_TYPESAFE_BY_RCU`.)
2. **Slack-only OOB**: poison *only* `[addr+req_size, addr+bucket_size)` — the object's own rounding
   slack, never a neighbor — plus a `ksize()`/`krealloc` hook to re-open it when the kernel
   legitimately grows into it. Catches e.g. `kmalloc(30)` writes to byte 30/31 of a 32-byte bucket.
3. **Stretch — emulator-native page-granularity UAF**: PC-hook the *page* allocator
   (`alloc_pages`/`__free_pages`, learn PFN+order from registers like the existing kmalloc table),
   poison/unpoison whole physical pages. Zero false-positive (whole pages), portable to a
   Windows-pool target — the un-forged equivalent of `Image.dpalloc`, needing no kernel config.

**The honest ceiling (load-bearing):** a purely-emulator-side sanitizer **cannot** catch an overflow
that crosses the bucket boundary into a neighbor on a *packed* allocator — that requires the allocator
to insert a real gap. This **includes `Image.buggy`'s planted bug** (`kmalloc(32)` write at
object+32: no slack in a 32-byte bucket, and it's neighbor territory, not a UAF). Only the
kernel-cooperative `SLUB_DEBUG_ON` oracle catches that class. So: **emulator-native KASAN (UAF +
slack + page-granularity) coexists with, but does not replace, the kernel-cooperative validation
oracles.** For targets where *we* control allocation (userspace, or an OS we can relocate within),
relocation/redzones would work fully — the limit is specific to fuzzing an allocator that does its
own physical-page bookkeeping.

## KMSAN (uninitialized-value taint) — a real new subsystem; SoA-cheap

Today's `PERM_RAW` is *memory*-granular and ASAN-strict (faults on first read of a never-written
byte). Real KMSAN *permits* reading uninit memory and reports only at *consumption* (branch on a
tainted value, `copy_to_user`, syscall-return, DMA) — a different oracle. Build a **shadow register
file** alongside the value file: `regs_taint: [u32;32]` (scalar) / `[[u32;LANES];32]` (SoA — exactly
gamozolabs' taint-tracking shape, one extra `Simd<u32,16>` op per ALU op, near-free). Propagation
keyed off the `AluOp`/`MulOp`/`Load`/`Store` enums: OR-combine for add/sub/bitwise (sound), shift the
shadow for shifts, taint-all for mul/div, load = source bytes' RAW state; **stores** need a new
mode-gated byte-shadow plane (default `write()` clears RAW, wrong for value-taint) so keep it behind a
distinct mode. Reporting needs consumption checkpoints (branch-on-tainted + sink hooks). Medium-high
effort, its **own phase**. First increment: shadow arrays + ALU OR-combine + load-taint only,
validated on a synthetic uninit-use-then-branch program before the kernel.

## UBSAN — three detectable, the rest need type info

- **Div-by-zero — build now** (lowest-effort, highest-confidence win in the report): RISC-V *defines*
  DIV/0 → all-ones, REM/0 → dividend (no trap), so it needs an explicit check. One choke point,
  `muldiv` (shared by the scalar interp and fs-vec's masked-scalarized divide): `if b == 0 {
  report(pc) }`. Kernel code doesn't intentionally rely on the saturating value (unlike unsigned
  wraparound), so low false-positive risk.
- **Misalignment** — already have `FaultKind::Unaligned` (pre-flight: verify we're not over-reporting
  vs the guest kernel's own misaligned-fixup path).
- **Null / low-VA deref** — likely already free: the zero page is unmapped in the kernel's page
  tables, so the Sv32 walk faults → guest page fault → console oops → existing oracle (verify).
- **Out of scope uninstrumented** (permanent, need C type info): signed overflow and shift-out-of-range
  (RV add/shift are bit-identical for signed vs unsigned; unsigned wraparound is legal+common →
  flagging every case is a false-positive firehose), array-bounds, CFI/provenance.

## KCSAN — parked (needs SMP)

Data races can't occur in the current single-hart `nosmp` model (each lane is an *independent* kernel
instance, not multiple harts sharing one image) — accesses are serialized by construction. KCSAN is
gated on a machine-model decision (true multi-hart SMP emulation, a larger effort), not sanitizer
work. Park it; its watchpoint idea ports over directly once/if SMP lands.

## Buildable increments (queued, in order)

1. **UBSAN div-by-zero** — a few lines in `muldiv`. Trivial, do first.
2. **KASAN slack-only + `ksize` hook** — finish `docs/kernel-san.md`'s plan calling `alloc_with_slack`
   instead of the cross-object guard; re-run the 500-case smoke test expecting **0% FP** (was ~40%).
3. **KASAN page-granularity UAF** — new page-allocator PC-hook table (medium).
4. **KMSAN taint** — its own phase (shadow regs + propagation + checkpoints).

The emulator-native sanitizers are correct and uninstrumented (the Windows-path virtue), but narrow on
packed-kernel-heap OOB — a real, honest limitation, not an effort gap. The kernel-cooperative oracles
stay as the complementary "catch the packed-neighbor overflow" tool for self-validation.
