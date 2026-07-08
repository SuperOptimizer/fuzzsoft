# fuzzsoft: wiring fs-san to the RV32 Linux kernel heap

*Design note from a research workflow (verified against the actual code + kernel build). Guides turning `--sanitize` from counting into catching real kernel heap bugs.*

# Wiring fs-san to the RV32 Linux kmalloc/kfree — concrete design

Grounded directly in the current code: `crates/fs-san/src/{alloc,hooks,linux}.rs`,
`crates/fs-mmu/src/lib.rs`, `crates/fs-platform/src/lib.rs`, `crates/fs-cli/src/main.rs`, and the
actual kernel build (`build/linux-src/System.map`, `build/linux-src/arch/riscv/include/asm/page.h`,
`.config`). Current state (verified by reading the code, not assumed): `fs-cli/src/main.rs:391-407`
+ `55-60` + `74-84` is **validation-only** — `SanCtx` just counts `allocs`/`frees`/`bytes` from
`PcHooks::on_pc`; it never calls `Sanitizer::alloc`/`free`, never touches `Mmu`. This is the exact
point where real wiring needs to go in.

## 0. One correction to feed into everything else: the offset is *not* `PAGE_OFFSET - ram_base`

`fs-cli/src/main.rs` sets `ram_base = 0x8000_0000` (lines 300, 548) but loads the **kernel Image**
at `kernel_addr = 0x8040_0000` (lines 301, 544) — 4 MiB into RAM, leaving room for firmware at
`ram_base`. `System.map` confirms `_start` (the kernel's first instruction, i.e. VA of
`kernel_addr`) is `c0000000`. Real RISC-V `setup_vm()` computes
`va_pa_offset = PAGE_OFFSET - kernel_map.phys_addr`, where `phys_addr` is *where the Image was
loaded*, not the start of RAM. So for this build:

```
va_pa_offset = 0xc0000000 (PAGE_OFFSET, arch/riscv/include/asm/page.h, !CONFIG_64BIT)
             - 0x80400000 (kernel_addr — fs-cli's own loader constant)
             = 0x3fc00000
```

**not** `0x4000_0000`. Verify: `_start` VA `0xc0000000` → PA `0xc0000000 - 0x3fc00000 = 0x80400000`
= `kernel_addr`. ✓. Using the naive `PAGE_OFFSET - ram_base` offset would silently mistranslate
every kmalloc/kfree pointer by 4 MiB — landing redzone stamps on unrelated physical memory and
guaranteeing false positives at whatever *actually* lives there. This is exactly the class of bug
the design must guard against structurally, not just get right once by hand.

## 1. VA→PA method: fixed affine offset, computed from the loader's own constants, self-checked once at boot — no page-table walk on the hot path

**Two different address domains are in play here and must not be conflated:**

- **PC domain** (`cpu.pc`, `hook.entry_pc` from `System.map`, the `ret_pc` stashed in
  `PcHooks::pending`): these are all kernel **VAs**, compared VA-to-VA in
  `hooks.rs::on_pc` (`self.allocs.get(&pc)`, `self.pending.get_mut(&pc)`). **No translation is
  needed or wanted here** — `cpu.step_system` already does the real Sv32 fetch-translation
  internally to execute at that VA; `cpu.pc` itself is never made physical. Do not add
  VA→PA translation into the hook-matching path.
- **Data-pointer domain** (`a0`/`a1` register values captured as the alloc size-return or
  free-argument — i.e. `HookEvent::Alloc{addr}` / `HookEvent::Free{addr}` from `hooks.rs`): these
  are addresses *of memory*, and everything downstream (`Sanitizer::alloc`/`free`, `Mmu::poison`/
  `protect`) operates on **physical** addresses (`fs_mmu::Mmu` is a flat physical window,
  `base=0x8000_0000`, confirmed by `Mmu::offset`'s `addr - self.base` and every call site in
  `fs-cli/src/main.rs`). **Only this domain needs translation**, and it needs it exactly once,
  right where `HookEvent` is turned into a `Sanitizer` call — i.e. inside (a new, corrected)
  `run_case` in `fs-cli/src/main.rs`, not inside `hooks.rs` (keep `hooks.rs` ISA/domain-agnostic
  per its own doc comment).

**Why a fixed affine offset is architecturally correct here, not a heuristic:** `kmalloc`/
`__kmalloc`/`kmem_cache_alloc`/`kfree`/`kmem_cache_free` operands are *always* linear-map
(`__va`/`__pa`-reachable) addresses in stock SLUB — slab pages come from the buddy allocator with
`GFP_KERNEL` (no `__GFP_HIGHMEM`), never from `vmalloc`. This holds even if the target kernel later
enables `CONFIG_HIGHMEM` for RV32 (that only affects `alloc_pages(__GFP_HIGHMEM)` callers, not
kmalloc). So a single global subtraction is not a shortcut that "usually works" — it is the
literal implementation of `__pa()` for this address class, on every current and plausible future
build of this target.

**Implementation:**

```rust
// New: crates/fs-san/src/linux.rs (or fs-cli, but linux.rs is the right home — it already
// owns "RV32 Linux wiring" per its module doc).
pub const PAGE_OFFSET: u32 = 0xc000_0000; // arch/riscv/include/asm/page.h, !CONFIG_64BIT — fixed
                                           // for every RV32 Linux boot, not runtime-computed
                                           // (unlike RV64's kernel_map.page_offset).

pub struct LinearMap { va_pa_offset: u32, pa_lo: u32, pa_hi: u32 }

impl LinearMap {
    /// `kernel_load_pa` is fs-cli's own loader constant (`kernel_addr`) — it is *known exactly*,
    /// not discovered, because fuzzsoft is both the loader and the emulator.
    pub fn new(kernel_load_pa: u32, ram_base: u32, ram_size: u32) -> Self {
        Self {
            va_pa_offset: PAGE_OFFSET.wrapping_sub(kernel_load_pa),
            pa_lo: ram_base,
            pa_hi: ram_base + ram_size,
        }
    }

    /// Translate a kmalloc/kfree-hook-observed VA to PA, or reject it. Never guesses outside the
    /// validated linear-map window — an out-of-window VA means a wrong hook/register, not
    /// something to translate speculatively.
    pub fn va_to_pa(&self, va: u32) -> Option<u32> {
        if va < PAGE_OFFSET { return None; }               // not in the linear map at all
        let pa = va.wrapping_sub(self.va_pa_offset);
        if pa >= self.pa_lo && pa < self.pa_hi { Some(pa) } else { None }
    }
}
```

**One-time self-check at boot** (cheap, run once after `Snapshot::capture` in
`fs-cli/src/main.rs`, before the fuzz loop starts): resolve `_start` from the parsed
`System.map` (`syms["_start"]`), confirm `linear_map.va_to_pa(syms["_start"]) == Some(kernel_addr)`.
If it doesn't match, disable `--sanitize` for the run and print a hard error — a mistranslated
offset is worse than no sanitizer (it stamps redzones over live unrelated kernel memory).
Optionally cross-check a second symbol from a different region (e.g. a `.bss` symbol) to catch
the (currently impossible, but future-proof against build changes) case of a non-contiguous
linear map.

**Do not use `cpu.xlate`'s Sv32 walker (`fs-riscv/src/lib.rs:610-660`) on the hot path.** It's the
right tool for general VA translation (and is already used correctly for the *user*-space program
buffer at `fs-cli/src/main.rs:378`, which really can be anywhere in the address space), but for
kmalloc/kfree pointers it's strictly more machinery than the guaranteed-linear-map case needs, and
it introduces new failure modes here that the affine offset doesn't (which hart's `satp`? mid-walk
if the hook fires from an interrupt context?). Reserve it for a one-time cross-validation if you
want extra paranoia: at boot, walk `_start`'s VA through `cpu.xlate` using the kernel's S-mode
`satp` and confirm it agrees with `LinearMap::va_to_pa`.

## 2. Redzone / quarantine / size-rounding policy

Current `fs-san/src/alloc.rs` already has the right *mechanism* (`DEFAULT_REDZONE = 16`,
`Sanitizer::alloc` poisons `[addr-16,addr)` and `[addr+size,addr+size+16)`, stamps payload
`WRITE|RAW`; `Sanitizer::free` poisons the payload and moves it to a FIFO quarantine capped at
`DEFAULT_QUARANTINE_CAP = 4096`). Three policy changes are needed for kernel-heap correctness:

1. **Round the captured size up to the real SLUB bucket before computing the trailing redzone
   boundary**, not the raw requested size. `linux.rs::KNOWN_SYMBOLS` captures the *requested*
   size (`a0` for `kmalloc`/`__kmalloc`/`__kmalloc_node`, `a2` for `kmalloc_trace`) — but SLUB
   rounds every request up to a fixed bucket (RV32 default: 8, 16, 32, 64, 96, 128, 192, 256, 512,
   1024, 2048, 4096, 8192 — check `build/linux-src/.config` for `CONFIG_SLAB_BUCKETS`/
   `KMALLOC_MIN_SIZE` to confirm the exact table for this build), and the kernel legitimately
   touches the whole bucket via `ksize()`, in-place `krealloc()` growth, and
   `kmalloc_size_roundup()`-then-populate idioms. Add a `kmalloc_bucket_size(requested: u32) -> u32`
   table lookup in `linux.rs` and call it before `Sanitizer::alloc` stamps the trailing redzone —
   i.e. `Sanitizer::alloc` should poison `[addr+size, addr+bucket_size)` as *slack* (not directly
   accessible per SanError, but readable) and `[addr+bucket_size, addr+bucket_size+16)` as the
   *real* redzone. Simplest correct approximation given current `Sanitizer::alloc`'s one-`size`
   API: just pass `bucket_size` (not `requested_size`) as `size` into `Sanitizer::alloc`, and keep
   `requested_size` around only for the crash-report string. This trades a few missed
   exact-bucket-boundary 1-byte overflows for zero false positives on rounding slack — the right
   asymmetry for an unattended fuzzer oracle.
2. **Hook `ksize()` and unpoison up to the bucket size on call** (add `("ksize",
   Convention::???)` — `ksize()` takes the pointer as `a0` and needs a re-`protect(addr, bucket-
   original_size, WRITE)` style "open the slack" call, not a redzone stamp). Without this, the
   common kernel pattern of allocating conservatively then calling `ksize()` to discover and use
   the real bucket size (skb, crypto, mm code) will false-positive against the *real* redzone
   moved to the bucket boundary. If bucket rounding is applied per (1), `ksize()` accesses inside
   `[addr, addr+bucket_size)` already succeed for free — but if a later change tightens the
   redzone back to `requested_size`, `ksize()` handling becomes mandatory again; keep the two
   changes conceptually paired.
3. **Quarantine must be soft/advisory, never sticky, and it is `Sanitizer::alloc`'s existing
   `evict_quarantine(&addr)` call (line 168) that already implements this correctly** — verify
   this stays true after adding VA→PA translation: the *real* SLUB freelist can and will reuse a
   physical address for a brand-new object with zero notice to fs-san other than the next
   kmalloc-hook firing at that same `pa`. `Sanitizer::alloc` already un-quarantines on `alloc()`
   before re-stamping — this is correct, keep it, and do **not** add any check that treats
   "already quarantined" as suspicious when a fresh `alloc()` targets it; that would reintroduce
   the exact false-positive KASAN's in-band quarantine doesn't have to worry about (its quarantine
   is authoritative; fs-san's is not).

## 3. False positives to suppress, and exactly how

In order of how likely each is to actually fire on this target, based on reading the current code:

1. **`kfree`/`kmem_cache_free` poisoning at hook *entry* faults on SLUB's own legitimate write —
   this is a live, guaranteed-to-fire bug in the current hook design, not a hypothetical.**
   `hooks.rs`'s own doc comment says free-shaped hooks fire "immediately" at entry because
   "freeing typically only touches allocator metadata before the payload, not the payload bytes
   themselves" — true for the toy bump allocator this framework was generalized from, **false for
   SLUB**: `set_freepointer()` writes the intrusive freelist "next" pointer directly into the
   freed object's own payload bytes (`s->offset` into the object; this is the exact bug the real
   upstream patch "slub: Actually fix freelist pointer vs redzoning" addressed). If
   `Sanitizer::free()` poisons the payload the instant PC hits `kfree`'s entry — before the callee
   body that writes the freelist pointer has even executed — SLUB's own write then lands on
   already-poisoned bytes and faults as a false UAF on essentially every `kfree()` call.
   **Fix:** extend `hooks.rs`'s existing stash-until-return machinery (already built for allocs —
   `pending: HashMap<u32, Vec<u32>>` keyed by return address) to frees too: at a `FreeHook` entry,
   capture `(addr, ra)` and stash it instead of emitting `HookEvent::Free` immediately; emit the
   event when PC reaches the stashed `ra`. This is a small, mechanical change to `PcHooks::on_pc`
   (add a `pending_frees` map parallel to `pending`) and costs nothing extra — no new state
   machinery, just reusing the pattern already proven for allocs.
2. **VA outside the validated linear-map window → skip, log, don't translate.** Any hook-observed
   `addr` that fails `LinearMap::va_to_pa` (wrong register captured, an inlined/tail-called
   variant hit at a hooked symbol name, a future symbol whose pointer isn't linear-map) must be
   dropped with a log line, never blindly `wrapping_sub`'d — a mistranslated PA corrupts the
   permission plane at unrelated physical memory and produces "spurious fault at a site unrelated
   to the real bug," which is strictly worse than a missed detection.
3. **NULL and `ZERO_SIZE_PTR` (`(void*)16`).** Special-case both in the VA→PA translation layer
   before calling `Sanitizer::alloc`/`free` at all — `kzalloc(0, ...)` legitimately returns
   `ZERO_SIZE_PTR`, and translating `0x10` through the linear-map offset would compute a bogus PA
   near the bottom of RAM and poison whatever's actually there (page tables, `_start`'s own
   region, etc.).
4. **`kmem_cache_alloc`/`kmem_cache_alloc_noprof` are already correctly *not* hooked**
   (`linux.rs::Convention::AllocSizeUnavailable`) — this is a coverage gap, not a false-positive
   source, and the doc comment already explains why (`cachep->object_size` isn't in a register).
   Leave as-is for now; if coverage matters later, the fix is a guest-memory read of
   `cachep->object_size` at a kernel-version-specific struct offset, which is a bigger change than
   this design covers — flag it as a known, accepted gap rather than silently forgetting it.
5. **`SLAB_TYPESAFE_BY_RCU` caches.** `kmem_cache_free`'s hook already captures `cachep` (`a1`,
   per `linux.rs`'s `FreePtrReg` convention table — note it's `a1` for `kmem_cache_free`, `a0` for
   plain `kfree`/`kfree_sensitive`). Since fuzzsoft owns the kernel build, statically grep the
   kernel source at build time for every `kmem_cache_create(..., SLAB_TYPESAFE_BY_RCU, ...)` call
   site, resolve each cache's name, and at boot (once, via a `/proc/slabinfo` read through the
   guest agent, or a one-time symbol/struct walk) map name → runtime `struct kmem_cache*` address.
   At the (now return-deferred, per #1) `kmem_cache_free` hook, check `cachep` against this list
   and **skip poisoning** (not the whole event — just the poison — so quarantine bookkeeping stays
   consistent) for RCU-typesafe caches, since lock-free readers are contractually allowed to touch
   the object during the grace period. This is a narrow, enumerable set (typically single digits
   of caches in a kernel build) — do not skip this for the sake of "some UAF coverage is better
   than none," because on this class of cache it's guaranteed spurious, not occasional.
6. **`kfree_bulk`/`kmem_cache_free_bulk` are absent from `KNOWN_SYMBOLS`** — a silent
   false-*negative* gap, not a false positive, but flag it in run output ("N known-unhooked bulk
   frees skipped" — cheap to detect by also hooking these entries and just logging a counter, even
   before implementing real per-pointer poisoning) so a clean run on bulk-free-heavy code doesn't
   read as "no bugs" when it's really "not tested."
7. **The snapshot/reset lifecycle desync — the most likely to actually surface once wiring goes
   live, and structurally different from the others.** `fs-platform::Snapshot` (lib.rs:153-179)
   already snapshots+restores `Mmu`'s memory and permission planes every case
   (`machine.ram.enable_dirty_tracking()` at capture, `machine.ram.reset_dirty(...)` at
   `snap.reset()`, called at `fs-cli/src/main.rs:439` inside the `for case in 0..cases` loop). But
   `Sanitizer`'s own bookkeeping (`live: HashMap`, `quarantine_order: VecDeque`,
   `quarantined: HashMap` — all currently non-`Clone`, confirmed by reading `alloc.rs`: no
   `#[derive(Clone)]`, no `reset`/snapshot API at all) is **host-side state completely outside
   this snapshot**. Two broken options if this isn't fixed:
   - Leave `Sanitizer` un-reset across cases → case N+1's kernel legitimately reallocates a
     physical address that's still marked `live` from case N (routine, since `Mmu` rolled the
     actual bytes back to golden state but `Sanitizer` didn't) → spurious `SanError::DoubleAlloc`,
     **and** `Sanitizer::alloc` returns early on that error *before* stamping any redzone/RAW
     permissions — so the new object silently gets zero sanitizer coverage for the rest of that
     case. A bookkeeping false positive causing a fail-open safety regression.
   - Naively clear `Sanitizer` to empty every case → every legitimate `kfree()` in the new case on
     an object allocated during kernel *boot* (before `Snapshot::capture` even ran) looks like a
     free of an untracked pointer → spurious `SanError::InvalidFree`.

   **Fix:** add `#[derive(Clone)]` to `Sanitizer` (free — its fields are plain `HashMap`/`VecDeque`
   of `Copy` structs). Capture one `Sanitizer` clone alongside `Snapshot::capture` (right after
   `let snap = Snapshot::capture(&cpu, &mut m);` at `fs-cli/src/main.rs:387`, once the boot-time
   allocator hooks have been driven forward to that point so boot-time kmallocs are already
   reflected in `live`). Restore that clone — not a fresh `Sanitizer::default()` — every case, in
   lockstep with `snap.reset(&mut cpu, &mut m)` at line 439. Concretely: change `SanCtx` to hold
   `sanitizer: Sanitizer` alongside `hooks: PcHooks`, add a `golden: Sanitizer` field or thread the
   clone through the loop, and do `san_ctx.sanitizer = golden_sanitizer.clone();` right next to
   `snap.reset(...)`. Once this is in place, `DEFAULT_QUARANTINE_CAP` can be raised for the
   kernel-heap sanitizer specifically (its true lifetime is now "one case's churn," not an
   unbounded campaign, so the original unbounded-growth concern the cap exists for doesn't apply
   in the same way).

## 4. Step-by-step fs-cli run-loop integration plan

1. **`crates/fs-san/src/linux.rs`**: add `PAGE_OFFSET` const, `LinearMap` struct + `va_to_pa`
   (§1), and a `kmalloc_bucket_size(u32) -> u32` table (§2.1). Add `ksize` to `KNOWN_SYMBOLS` as
   its own new `Convention` variant that means "unpoison slack" rather than alloc/free-shaped.
2. **`crates/fs-san/src/hooks.rs`**: extend `PcHooks` with a `pending_frees: HashMap<u32,
   Vec<u32>>` mirroring the existing alloc-return stash, so `FreeHook` entries stash `(addr, ra)`
   and `HookEvent::Free` only fires at the matching return PC (§3.1). Keep `hooks.rs` otherwise
   ISA/domain-agnostic — still just PC + register file in, `HookEvent` out, no VA/PA knowledge
   here.
3. **`crates/fs-san/src/alloc.rs`**: add `#[derive(Clone)]` to `Sanitizer` (§3.7). Change
   `Sanitizer::alloc`'s size-rounding call site (or add a thin wrapper) to accept/compute bucket
   size before stamping the trailing redzone (§2.1). No other structural change needed —
   `poison_guard_before`/`after`'s existing best-effort `in_bounds` skip logic is already exactly
   the right discipline to reuse for VA→PA rejects.
4. **`crates/fs-cli/src/main.rs`**:
   a. After `let syms = fs_san::parse_system_map(&text);` (line 394), build
      `LinearMap::new(kernel_addr, ram_base, ram_size)` and run the `_start` self-check (§1);
      abort `--sanitize` with a clear error on mismatch.
   b. Change `SanCtx` (line 55-60) to hold `hooks: PcHooks`, `sanitizer: Sanitizer`, `linear_map:
      LinearMap`, plus a `rcu_typesafe_caches: HashSet<u32>` (§3.5) — drop the placeholder
      `allocs`/`frees`/`bytes` counters or keep them as separate stats fed *after* real
      `Sanitizer` calls succeed.
   c. In `run_case` (line 64-100), replace the `match ev { HookEvent::Alloc{size,..} => { ctx.allocs
      += 1; ... } ... }` stub with: translate `addr` via `ctx.linear_map.va_to_pa(addr)`; on `None`,
      log-and-skip (§3.2/§3.3); on `Some(pa)`, for `Alloc` round `size` to bucket (§2.1) and call
      `ctx.sanitizer.alloc(m /* &mut Mmu, i.e. &mut machine.ram — thread it in */, pa, bucket_size)`;
      for `Free`, check `ctx.rcu_typesafe_caches` membership via the free's `cachep`/`a1` register
      (needs `PcHooks`/`hooks.rs` to also surface which register produced the free's cache pointer
      when applicable — thread the raw regs through, or extend `HookEvent::Free` with an optional
      `cachep` field) and skip poisoning (but still evict from `live`) if typesafe-RCU; otherwise
      call `ctx.sanitizer.free(m, pa)`. Route `Sanitizer::alloc`/`free`'s `Err(SanError::...)`
      results into the same crash-reporting path `kernel_crash_sig` already feeds (a `SanError` is
      exactly as real a bug signal as a kernel oops string) — do not `let _ = ...` swallow them as
      the current `DESIGN.md` sketch does.
   d. Right after `let snap = Snapshot::capture(&cpu, &mut m);` (line 387) — and after driving the
      allocator hooks through the remaining boot instructions if any boot-time kmallocs still need
      to land in `sanitizer.live` — take `let golden_sanitizer = san_ctx.as_ref().map(|c|
      c.sanitizer.clone());` (§3.7).
   e. Inside `for case in 0..cases { snap.reset(&mut cpu, &mut m); ... }` (line 439), immediately
      add: `if let (Some(ctx), Some(g)) = (san_ctx.as_mut(), &golden_sanitizer) { ctx.sanitizer =
      g.clone(); }` — restoring `Sanitizer` state in lockstep with `Mmu`'s dirty-block reset every
      single case, per §3.7.
   f. Wire `ksize()` hits (once added to `KNOWN_SYMBOLS`/`PcHooks` per step 1) to call
      `mmu.protect(pa, bucket_size, PERM_WRITE)` (re-opening the slack, not poisoning) rather than
      going through `Sanitizer::alloc`/`free` at all.
5. **Smoke-test order** (cheapest-to-verify-first): (i) run with `--sanitize` and confirm the
   `_start` self-check passes and the boot-time `va_to_pa` translations land inside
   `[ram_base, ram_base+ram_size)`; (ii) run a handful of cases and confirm `Sanitizer::alloc`/
   `free` return `Ok` for the overwhelming majority of observed kmalloc/kfree pairs (a high
   `SanError` rate signals a wiring bug, e.g. §3.1's entry-vs-return ordering, not real kernel
   bugs); (iii) only once steady-state `SanError` rate is near zero, start trusting
   `SanError`/redzone faults as genuine crash signal alongside `kernel_crash_sig`.

---

## Experiment result (2026-07-08): naive redzone poisoning false-positives on stock SLUB

Wired the above (correct VA→PA offset 0x3fc00000 with `_start` self-check, SLUB size rounding via `kmalloc_bucket`, free-at-return) and ran `fuzzsoft fuzz --san-poison`:

- **Without poison (counting only):** 0 kernel crashes over hundreds of programs (hooks fire: ~315 kmalloc / 1741 kfree observed). ✓ the hook + VA→PA path is validated.
- **With poison:** ~195 "kernel crashes" in 500 programs (≈40%), all one signature, at ~13 exec/s.

Conclusion: as predicted, poisoning guard bytes around a stock-SLUB allocation stamps memory that belongs to **adjacent live objects** (SLUB packs objects tight; kmalloc caches have no inter-object gaps), so the kernel faults on legitimate neighbor access. `kmalloc_bucket` fixes the *trailing* ksize() case but the guard past the bucket boundary is still the next object.

**So poisoning is off by default** (`--sanitize` = validate/count; `--san-poison` = experimental). The infrastructure (correct translation, free-at-return, size rounding, quarantine) is retained and correct.

### Viable paths to real kernel-heap detection (future work)
1. **slub_debug kernel** — build with `CONFIG_SLUB_DEBUG_ON` / `slub_debug=FZ`: the *allocator* inserts redzones + poison and self-checks them on free, printing "Redzone overwritten" oopses the fuzzer already detects via the console oracle. Kernel-cooperative, but zero emulator poisoning and immediately usable on this target.
2. **KFENCE-in-emulator** — on a sampled fraction of kmalloc returns, the PC-hook *rewrites a0* to relocate the allocation onto a dedicated guard-page-flanked region in emulator memory (remapping the page tables), so guards never touch neighbors. This is the true uninstrumented path (and the model for the eventual Windows-pool case).

---

## Kernel-side SLUB debugging (viable now) — shipped 2026-07-08

Path #1 above is now built and validated: a **second** RV32 Linux kernel Image, identical to the
stock fuzzer kernel except that the *allocator itself* red-zones/poisons every slab object and
self-checks on free. This turns the fuzzer's existing console crash oracle (`kernel_crash_sig`)
into a real kernel-heap-bug detector with **zero emulator-side poisoning** — sidestepping the
adjacent-object false positives that killed `--san-poison` on stock SLUB (see the experiment
result above).

### What was built

- **Config:** started from the stock `build/linux-src/.config` and flipped on
  `CONFIG_SLUB_DEBUG_ON=y` (on top of the already-set `CONFIG_SLUB=y` / `CONFIG_SLUB_DEBUG=y`).
  `CONFIG_SLUB_DEBUG_ON` enables red-zoning + object poisoning + freelist/padding sanity checks
  **by default for every slab cache**, with no `slub_debug=` cmdline argument required (equivalent
  to booting `slub_debug=FZPU` unconditionally). `CONFIG_INITRAMFS_SOURCE` still points at
  `boot/initramfs.spec`, so the guest fuzzing agent (`build/agent`, packed as `/init`) boots and
  reaches the snapshot exactly as on the stock kernel — the snapshot/hypercall wire protocol is
  unchanged.
- **Out-of-tree build:** built with `O=build/linux-slubdebug` from a *clean second git worktree*
  `build/linux-slubdebug-src` (Kbuild refuses an `O=` build against `build/linux-src` because that
  tree already holds the stock in-tree build; and we must not `mrproper` it). Both worktrees sit on
  the same kernel commit, so it is the same source, just a pristine checkout to build from.
- **Outputs:** `firmware/Image.slubdebug` (26,930,176 bytes) and `firmware/System.map.slubdebug`
  (4,160,369 bytes). The stock `firmware/Image` and `build/linux-src/System.map` (the files the
  main fuzzer uses) are **untouched**.
- **Reproduce:** `scripts/build-slubdebug-kernel.sh` (bash, `set -euo pipefail`) does the whole
  thing deterministically — creates the worktree if missing, seeds + `olddefconfig`s the `.config`,
  flips the SLUB debug symbols, verifies they stuck, builds `Image`, and publishes the two
  `firmware/*.slubdebug` files.

### How it catches bugs

When guest kernel code overruns a slab allocation, writes through a dangling/freed pointer, or
otherwise corrupts a slab object, **SLUB's own free-time checks fire** and the kernel prints a
report such as `Redzone overwritten`, `Poison overwritten`, `Object already free`, or an outright
`BUG`/oops with a call trace. All of these land on the guest console (UART), which is exactly the
stream the fuzzer's `kernel_crash_sig` oracle already scans — so no new detection code is needed
on the emulator side. A hit is a **true positive** (the allocator, not a guessed emulator redzone,
declared the corruption), unlike the stock-SLUB emulator-poisoning approach.

### How to run the fuzzer against it

```
./target/release/fuzzsoft fuzz --cases N --seed S --kernel firmware/Image.slubdebug
```

Run it **without** `--sanitize`/`--san-poison` — the kernel self-checks now; emulator poisoning is
neither needed nor wanted here (it would re-introduce the adjacent-object false positives). Any
SLUB corruption report shows up in the normal crash count/console dump.

### Tradeoff

Slower boot and slower steady-state than the stock kernel — `SLUB_DEBUG_ON` makes every alloc/free
walk red-zones and poison the full object, and boot to snapshot is visibly longer (guest boot
reaches `Run /init` around the ~207s guest-time mark vs. the stock kernel's much earlier point).
Smoke test (2026-07-08): `fuzz --cases 50 --seed 1 --kernel firmware/Image.slubdebug` reached
`snapshot captured`, completed all 50 cases at ~139 execs/sec (~21 guest MIPS), 3403 coverage
buckets, 28-program corpus, **0 kernel crashes and no Redzone/Poison/BUG/oops** across the run (as
expected — 50 random seed-1 programs don't corrupt the heap; the point is the oracle is now armed).
The asymmetry is the whole trade: slower, but a crash here is a real kernel heap bug, not a
sanitizer false positive. Use this Image for heap-focused campaigns; keep the stock `firmware/Image`
for raw-throughput coverage growth.
