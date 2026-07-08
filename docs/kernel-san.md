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

---

## Oracle validation: planted-bug kernel

The `firmware/Image.slubdebug` smoke test above is necessary but not sufficient: 0 crashes over 50
clean-kernel programs proves the oracle *doesn't false-positive*, not that it can *detect* a real
kernel heap bug (it has never been given one to catch). This section closes that gap with a
**third** kernel Image, `firmware/Image.buggy`, that plants a deliberate, clearly-fenced,
allocator-detected slab overwrite reachable only from a specific fuzzer syscall — proof the
`kernel_crash_sig` console oracle actually fires on real corruption, not just that it stays quiet.

**THIS IMAGE IS FOR ORACLE VALIDATION ONLY — never a real fuzzing target.** It has a deliberate,
unconditional kernel heap bug and must never be confused with `firmware/Image` or
`firmware/Image.slubdebug`.

### The planted bug

Built from a third clean worktree, `build/linux-buggy-src` (same pattern as
`build/linux-slubdebug-src`: a pristine second checkout of the `linux/` repo at the same commit as
`build/linux-src`, since Kbuild's `O=` build refuses to run against an already-in-tree-built
source). The patch, saved as `scripts/planted-bug.patch` and applied only to that worktree, adds a
fenced block at the top of `SYSCALL_DEFINE2(memfd_create, ...)` in `mm/memfd.c` (this kernel version
keeps `memfd_create` in `mm/memfd.c`, not `fs/memfd.c`):

```c
	/* FUZZSOFT PLANTED BUG — NOT FOR PRODUCTION
	 * Deliberate slab out-of-bounds write, planted here to validate that
	 * the fuzzsoft crash oracle (console scan for SLUB_DEBUG reports)
	 * actually detects real kernel heap corruption. This is reachable
	 * only via the memfd_create() syscall, which the fuzzer's own boot
	 * path never calls, so it does not affect boot-to-snapshot. Do not
	 * upstream, do not merge into a real kernel build.
	 */
	{
		char *fuzzsoft_p = kmalloc(32, GFP_KERNEL);
		if (fuzzsoft_p) {
			/* First byte immediately past the 32-byte object. For a
			 * SLUB_DEBUG_ON kmalloc-32 cache, s->object_size=32 and
			 * s->inuse=36 (object_size rounded to word size, then +4
			 * for a non-empty Right Redzone since SLAB_RED_ZONE and
			 * size==object_size) — so bytes [32,36) are the live,
			 * checked "Right Redzone". Writing offset 32 lands
			 * squarely inside it (an offset like +16 lands inside
			 * SLAB_STORE_USER's alloc/free track metadata instead,
			 * which SLUB does NOT sanity-check on free — verified by
			 * smoke test to produce zero detections).
			 */
			fuzzsoft_p[32] = 0x41; /* 1 byte past the 32-byte object: SLUB Right Redzone corruption, caught on kfree() by SLUB_DEBUG */
			kfree(fuzzsoft_p);
		}
	}
	/* END FUZZSOFT PLANTED BUG */
```

Design notes:

- **Boot-safe:** `memfd_create` is never called during plain kernel boot/init, only when the
  fuzzer's guest agent explicitly emits it as one of its curated syscalls (`nr` 279, confirmed
  against `include/uapi/asm-generic/unistd.h`'s `#define __NR_memfd_create 279` and
  `crates/fs-prog/src/syscalls.rs`'s `SyscallDesc { name: "memfd_create", nr: 279, ... }`). Boot to
  snapshot on `Image.buggy` is unaffected — it reaches `Run /init` exactly like `Image.slubdebug`.
- **Fuzzer-reachable:** `memfd_create` is in `fs-prog`'s curated syscall table and is a common,
  dependency-free pick in generated programs (it produces an `FD` resource other calls consume).
- **Unconditional and unattached to user input:** the write is a fixed offset into a fixed-size
  kernel allocation, not derived from any user-controlled pointer/length — this is a clean
  allocator-detected corruption on the very first `memfd_create` call, not a wild/user-triggerable
  fault, so detection doesn't depend on which arguments the fuzzer happened to generate.
- **Offset choice matters, and the first attempt (offset 48) was wrong** — see the note in the
  patch comment. A first pass at this experiment planted the write at `p[48]` (following the
  original design intent: "16 bytes past a 32-byte object"). Built, booted, and ran 400 cases:
  **zero crashes.** Reading `calculate_sizes()` in `mm/slub.c` explains why: for a `SLUB_DEBUG_ON`
  kmalloc-32 cache, the *checked* Right Redzone is only `s->inuse - s->object_size` = 4 bytes
  (`[32, 36)`) — byte 48 instead lands inside the `SLAB_STORE_USER` alloc/free `struct track`
  metadata region, which SLUB writes and reads but never sanity-checks for corruption on free. That
  overwrite was real but silent — a good reminder that "past the object" isn't automatically "in a
  checked byte range" once `SLAB_STORE_USER`/`SLAB_RED_ZONE` reshuffle the object's tail layout.
  Moving the write to offset 32 (the first byte of the guaranteed-checked Right Redzone) fixed it.

### Build

`scripts/build-buggy-kernel.sh` reproduces the whole thing deterministically: creates
`build/linux-buggy-src` (clean worktree) if missing, applies `scripts/planted-bug.patch` if not
already applied (checked via the fenced marker, so it's idempotent), seeds + `olddefconfig`s the
`.config` from the stock build, flips `CONFIG_SLUB_DEBUG_ON`, verifies both that and
`CONFIG_INITRAMFS_SOURCE` stuck, builds `Image`, and publishes `firmware/Image.buggy` +
`firmware/System.map.buggy`. Output sizes: **`Image.buggy` 26,930,176 bytes**, **`System.map.buggy`
4,160,369 bytes** — identical sizes to the `.slubdebug` build, as expected (same config, one
fenced-off ~20-line function body change).

### Smoke test — the oracle fires

```
./target/release/fuzzsoft fuzz --cases 400 --seed 1 --kernel firmware/Image.buggy
```

Result: **`[KERNEL CRASH]` fired at case 91** (first of 5 distinct crash signatures across 9 total
crashes in the 400-case run — `memfd_create` gets picked often enough that several *different*
surrounding call sequences each independently trip the same planted bug and get deduped by faulting
PC into 5 unique signatures). Exact console excerpt from the first hit:

```
fuzz: [KERNEL CRASH] epc=0x7e9d0013 case 91 calls=["memfd_create", "dup", "statx", "ioctl$TCSETS", "recvmsg", "pidfd_getfd", "epoll_pwait", "close"] nrs=[279, 23, 291, 29, 212, 438, 22, 57]
[  207.377195] [Right Redzone overwritten] 0xc2e33dc0-0xc2e33dc0 @offset=3520. First byte 0x41 instead of 0xcc
[  207.379899] =============================================================================
[  207.381986] BUG kmalloc-32 (Not tainted): Object corrupt
[  207.383741] -----------------------------------------------------------------------------
[  207.385996] Allocated in ___se_sys_memfd_create+0x32/0x1cc age=0 cpu=0 pid=1
[  207.388530]  __kmalloc_cache_noprof+0x110/0x29c
[  207.390717]  ___se_sys_memfd_create+0x32/0x1cc
[  207.392703]  __riscv_sys_memfd_create+0x12/0x1c
[  207.394707]  syscall_handler+0x1c/0x28
[  207.396719]  do_trap_ecall_u+0x108/0x238
[  207.398631]  handle_exception+0xd4/0xe2
[  207.400483] Freed in kobject_uevent_env+0x128/0x1bc age=88 cpu=0 pid=1
...
[  207.430653] Slab 0xc7afd7f8 objects=32 used=28 fp=0xc2e33e20 flags=0x80000200(workingset|section=16|zone=0)
[  207.433455] Object 0xc2e33da0 @offset=3488 fp=0xc2e33e20
[  207.435589] Redzone  c2e33d80: cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc  ................
[  207.437936] Redzone  c2e33d90: cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc cc  ................
[  207.440304] Object   c2e33da0: 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b  kkkkkkkkkkkkkkkk
[  207.442658] Object   c2e33db0: 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b 6b a5  kkkkkkkkkkkkkkk.
[  207.445007] Redzone  c2e33dc0: 41 cc cc cc                                      A...
[  207.447219] Padding  c2e33df4: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a              ZZZZZZZZZZZZ
[  207.449459] Disabling lock debugging due to kernel taint
[  207.452346] ------------[ cut here ]------------
[  207.452602] WARNING: mm/slub.c:1233 at object_err+0xa0/0xb0, CPU#0: init/1
```

The `Redzone c2e33dc0: 41 cc cc cc` line is a direct, byte-level confirmation: `0x41` (`'A'`, our
planted write) sitting where the pristine `0xcc` redzone fill should be, at exactly `object_addr +
32` (`0xc2e33da0 + 32 = 0xc2e33dc0`). SLUB's own allocator — not a guessed emulator redzone —
declared `BUG kmalloc-32 (Not tainted): Object corrupt` and correctly attributed the allocation to
`___se_sys_memfd_create+0x32/0x1cc`, i.e. our planted code. `kernel_crash_sig`'s `slub_report` branch
matches this via the `"Redzone overwritten"` substring (the actual format string is `"[%s
overwritten]"` with `%s = "Right Redzone"`, so it substring-matches as designed) and reports it as a
genuine `[KERNEL CRASH]`, deduped by a report-line hash since this class of report has no `epc:`
register dump line to key on.

**Conclusion: the fuzzer's crash oracle is validated end-to-end.** Given a real, allocator-detected
kernel heap corruption reachable through a fuzzer-emitted syscall, `kernel_crash_sig` catches it,
`[KERNEL CRASH]` fires, and the console report is captured verbatim — on the very first fenced-bug
kernel tried, well within the first few hundred cases. This closes the loop the `Image.slubdebug`
smoke test (0 crashes on a clean kernel) left open: the oracle isn't just quiet on clean kernels, it
is loud on planted ones.

---

## KASAN / KFENCE investigation (2026-07-08): neither is available on rv32 — built DEBUG_PAGEALLOC instead

`Image.slubdebug` (above) only catches heap corruption **at `kfree()` time** (SLUB's own
redzone/poison self-check). It does **not** catch an out-of-bounds **read**, and it does not catch
a use-after-free the instant it happens — only when the object is eventually freed and SLUB
happens to check it. KASAN and KFENCE are the real answer to that gap (shadow-memory / guard-page
techniques that fault on the actual bad access), so this investigation tried to build one of them
for the rv32 target, in the order the design called for: KASAN first, KFENCE as fallback.

### Result: both are architecturally unavailable on 32-bit RISC-V in this kernel tree

This isn't a missed config flag — it's a hard Kconfig gate, confirmed by reading `arch/riscv/Kconfig`
in the kernel worktree (`build/linux-kasan-src`, same commit as every other worktree, `0e35b9b6ec0f`,
kernel 7.2.0-rc2) and then **empirically proving it** by trying to enable each option:

```
arch/riscv/Kconfig:136:  select HAVE_ARCH_KASAN        if MMU && 64BIT
arch/riscv/Kconfig:137:  select HAVE_ARCH_KASAN_VMALLOC if MMU && 64BIT
arch/riscv/Kconfig:138:  select HAVE_ARCH_KFENCE       if MMU && 64BIT
```

`lib/Kconfig.kasan`'s `menuconfig KASAN` depends on `HAVE_ARCH_KASAN` (generic mode) or
`HAVE_ARCH_KASAN_SW_TAGS` (arm64-only) or `HAVE_ARCH_KASAN_HW_TAGS` (arm64 MTE-only) — none of
which rv32 can ever select. `lib/Kconfig.kfence`'s `menuconfig KFENCE` depends on
`HAVE_ARCH_KFENCE` alone — same story. This build runs `CONFIG_ARCH_RV32I=y` with `CONFIG_64BIT`
unset, so both gates are permanently closed.

Proof, reproduced against a fresh `O=` config seeded from the stock `.config`
(`build/linux-kasan/.config`):

```
$ scripts/config --file .config -e KASAN -e KASAN_GENERIC -e KASAN_INLINE
$ make O=build/linux-kasan ARCH=riscv LLVM=1 olddefconfig
$ grep -E '^CONFIG_KASAN|^CONFIG_HAVE_ARCH_KASAN' .config
(no output — the symbol isn't even present as "# CONFIG_KASAN is not set";
 olddefconfig silently drops a selection whose `depends on` can never be true)

$ scripts/config --file .config -e KFENCE
$ make O=build/linux-kasan ARCH=riscv LLVM=1 olddefconfig
$ grep -E '^CONFIG_KFENCE|^CONFIG_HAVE_ARCH_KFENCE' .config
(no output — same result)
```

Both attempts leave zero trace of the symbol in the resulting `.config` — Kconfig doesn't even
offer a disabled prompt, because the `depends on` chain is unsatisfiable on this arch/bitness
combination. There is no rv32 arch support for KASAN or KFENCE anywhere in this kernel source tree
to fall back onto; implementing it would mean writing `arch_kfence_init_pool()`/shadow-memory
offset math for Sv32 page tables from scratch — arch bring-up work, not a config change, and well
outside the scope of this build task.

### What was built instead: `firmware/Image.dpalloc`

The strongest oracle that **is** actually available on rv32 with no arch bring-up:
`CONFIG_DEBUG_PAGEALLOC` + `CONFIG_PAGE_POISONING`, on top of the same `CONFIG_SLUB_DEBUG_ON` used
by `Image.slubdebug`. `arch/riscv/Kconfig` selects `ARCH_SUPPORTS_DEBUG_PAGEALLOC if MMU` — no
`64BIT` restriction — so this one is real on rv32.

- **`CONFIG_DEBUG_PAGEALLOC`**: unmaps a page from the kernel's linear map immediately when
  `free_pages()` returns it to the buddy allocator. Any subsequent access — read *or* write —
  to that page is a genuine CPU page fault (`Unable to handle kernel paging request at virtual
  address ...`, `arch/riscv/mm/fault.c`), not a check that only runs at some later free. This is
  the "immediate UAF" property KASAN/KFENCE have, just at page granularity.
- **`CONFIG_PAGE_POISONING`**: fills freed pages with a poison pattern and verifies it on the next
  `alloc_pages()`, catching corruption even in the (rare) case a stale mapping elsewhere still
  allowed a write that `DEBUG_PAGEALLOC`'s unmap didn't intercept.
- **`CONFIG_SLUB_DEBUG_ON`** stays on for the free-time slab redzone/poison checks `Image.slubdebug`
  already provides.

Config verified stuck (`build/linux-kasan/.config`):

```
CONFIG_ARCH_SUPPORTS_DEBUG_PAGEALLOC=y
CONFIG_DEBUG_PAGEALLOC=y
CONFIG_SLUB_DEBUG=y
CONFIG_SLUB_DEBUG_ON=y
CONFIG_PAGE_POISONING=y
CONFIG_INITRAMFS_SOURCE="/home/forrest/fuzzsoft/boot/initramfs.spec"
```
(`CONFIG_KASAN` / `CONFIG_KFENCE` are absent, as established above.)

Built via a fourth clean worktree `build/linux-kasan-src` + `O=build/linux-kasan`, same recipe
pattern as `Image.slubdebug`/`Image.buggy`. Outputs: **`firmware/Image.dpalloc`** (26,930,176
bytes — same size class as the other custom builds) and **`firmware/System.map.dpalloc`**
(4,161,596 bytes). Reproduce with `scripts/build-kasan-kernel.sh` (kept that filename since it's
the prescribed build-script name for this investigation; the script itself tries KASAN, then
KFENCE, then falls back to this, and picks its own output suffix — `kasan`/`kfence`/`dpalloc` —
based on what actually stuck, so it stays correct if a future kernel version ever adds rv32
KASAN/KFENCE support).

### Honest limitation: this is a real but *narrower* oracle than KASAN/KFENCE would have been

`DEBUG_PAGEALLOC`/`PAGE_POISONING` operate at **page granularity**. They catch immediate UAF/OOB
on whole pages once those pages are returned to the buddy allocator — `vmalloc` frees, order>0
allocations, a fully-emptied SLUB slab page reclaimed back to the page allocator. They do **not**
give KASAN/KFENCE's byte-level, every-object redzone coverage: SLUB packs multiple small kmalloc
objects per page, and the page stays mapped (and un-poisoned) as long as *any* object on it is
still live. So a typical small-object kmalloc overflow/UAF — like the `Image.buggy` planted bug
above (`kmalloc(32)`, 1-byte overflow) — would **not** be caught by this oracle; it still needs
`Image.slubdebug`'s free-time redzone check for that class of bug. `Image.dpalloc` mainly adds
value for large/`order>0`/`vmalloc`-backed allocations and use-after-free of pages returned to the
buddy allocator. It is the strongest *available* oracle on this target, not a full KASAN/KFENCE
substitute — that substitute does not exist for rv32 in this kernel tree.

### Smoke test

```
./target/release/fuzzsoft fuzz --cases 100 --seed 1 --kernel firmware/Image.dpalloc
```

Reached `snapshot captured` at the **default** `--boot-insns` budget (3,000,000,000) — actual boot
took 2,114,096,390 insns, no bump needed (unlike the KASAN/KFENCE slow-boot warning this
investigation was scoped to expect — `DEBUG_PAGEALLOC`/`PAGE_POISONING` are much cheaper than full
shadow-memory instrumentation). Guest boot reached `Run /init` around the ~206s guest-time mark
(comparable to `Image.slubdebug`'s ~207s). Completed all 100 cases at ~133 execs/sec (~33 guest
MIPS), 6082 coverage buckets, 38-program corpus, **0 kernel crashes** (expected — clean kernel,
100 random seed-1 programs; the point is the oracle is armed, not that it fires here).

### Report-string prefix for the fuzzer's crash oracle — no new matcher needed

Unlike a real KASAN/KFENCE build (which would need a new `"KFENCE:"` substring matcher in
`kernel_crash_sig`, `crates/fs-cli/src/main.rs:167-186` — that function currently matches
`"KASAN:"` and the SLUB debug strings but not `"KFENCE"`), `DEBUG_PAGEALLOC`'s fault is a plain
kernel Oops. `kernel_crash_sig` **already** matches it with zero code changes needed:

```
s.contains("Unable to handle kernel")   // arch/riscv/mm/fault.c: "Unable to handle kernel %s at virtual address ..."
```

i.e. the exact banner prefix is:

```
Unable to handle kernel paging request at virtual address <addr>
```

followed by the usual RISC-V Oops dump (`epc :`, register file, call trace), which
`kernel_crash_sig`'s existing `hard_fault` branch already dedupes by `epc`. So `Image.dpalloc` is
usable against the fuzzer today with no oracle-side changes — the caveat above (page-granularity
only) is the real cost, not any missing plumbing.

**If rv32 KASAN/KFENCE support is ever added upstream** and `scripts/build-kasan-kernel.sh` is
re-run, it would publish `firmware/Image.kasan`/`firmware/Image.kfence` instead and the *new*
matcher work would be: KASAN's report banner is `"BUG: KASAN: <bug-type> in <function>"` (already
covered by the existing `s.contains("KASAN:")` check); KFENCE's is `"BUG: KFENCE: <bug-type> in
<function>"` (per `mm/kfence/report.c`'s `kfence_report_error()` format string) — that one **is
not yet matched** (`kernel_crash_sig` checks `"KASAN:"` but not `"KFENCE"`), so add
`|| s.contains("KFENCE:")` to the `hard_fault` condition at that point, but not before rv32 KFENCE
actually exists to test it against.

---

## Emulator-native sanitizer fuzz-loop wiring — shipped 2026-07-08

The slack-only kmalloc OOB/UAF core (`Sanitizer::alloc_with_slack`/`reopen_slack`) and the
page-granularity core (`PageSanitizer::alloc_pages`/`free_pages`) built in `fs-san` (see
`docs/emulator-sanitizers.md`) are now wired into `fuzzsoft fuzz`'s actual run loop
(`crates/fs-cli/src/main.rs`), replacing the old validation-only/`--san-poison` counting stub.
`--sanitize` now poisons unconditionally (the old separate `--san-poison` gate is a deprecated
alias — poisoning is zero-FP now, so there's no reason to keep it opt-in twice). `--jobs > 1` stays
incompatible (PC-hook path is serial-only, unchanged).

**Wiring, per `docs/emulator-sanitizers.md`'s KASAN section item list:** `run_case`'s per-instruction
dispatch calls, alongside the existing `hooks.on_pc`: `hooks.ksize_hit(pc, &cpu.regs)` ->
`Sanitizer::reopen_slack` (re-opens rounding slack on `ksize()`/`__ksize()`), and
`hooks.on_page_pc(pc, &cpu.regs)` -> `PageSanitizer::alloc_pages`/`free_pages` (whole-page
UAF/OOB). `SanCtx` now holds `page_san: PageSanitizer` alongside `san: Sanitizer`, both re-created
fresh at the top of every case (mirroring the existing `Sanitizer` reset), and `SanError`s from
either core are counted in `san_errors` rather than silently dropped.

**Two real false-positive mechanisms were found and fixed during validation — both fixed entirely
from `fs-cli`, with zero changes to `fs-san`/`fs-mmu` (as scoped):**

1. **`Mmu::protect`/`poison` aren't tracked by `Mmu`'s dirty-block reset.** `Snapshot::reset`'s
   O(dirty) restore only reverts blocks marked dirty by the *content-writing* path
   (`write`/`store`); `protect`/`poison` are permission-*only* mutations and never call
   `mark_dirty`. Every sanitizer-poisoned byte therefore leaked permanently across every
   subsequent case, until the guest allocator eventually reused that physical address for an
   unrelated, live object in a later case — a real, delayed false positive with no clean way to
   attribute it back to the sanitizer at a glance. Fixed by having `SanCtx` record every `(addr,
   len)` range it touches each case (`SanCtx::mark_dirtied`) and manually restoring it byte-for-
   byte from a golden permission-plane copy (`SanCtx::restore_dirtied_perms`, using only the
   already-public `Mmu::protect`/`Mmu::planes`) right after `snap.reset()`, before the next case
   (and also before crash minimization, which replays candidates via the sanitizer-free
   `run_case_bus`/`snap.reset` path and would otherwise inherit the same-case leak).
2. **`Sanitizer::alloc`/`alloc_with_slack` stamp `WRITE|RAW` unconditionally, colliding with
   `kmalloc(..., __GFP_ZERO)`/`kzalloc()`'s in-call zeroing.** The alloc-shaped PC-hook fires at
   the callee's *return* (the pointer isn't known until then — see `hooks.rs`), by which point
   SLUB has already run its own `GFP_ZERO`/`init_on_alloc` zeroing memset *inside* the call
   (`mm/slub.c`'s `slab_want_init_on_alloc`) — a real, legitimate write that happened before our
   hook ever ran. Stamping the payload `RAW` (unwritten) at that point hides that write and faults
   the very next legitimate read. Confirmed on the very first kmalloc the fuzzer's own corpus
   reached: `sk_prot_alloc()`'s `kmalloc(prot->obj_size, priority | __GFP_ZERO)` fallback (used by
   `netlink_create` -> `sk_alloc`, since `netlink_proto` has no dedicated `kmem_cache`) — verified
   deterministically reproducible with sanitizer poisoning on and absent with it off, isolating it
   to the sanitizer rather than a kernel/emulator bug. Fixed by dropping the `RAW` bit (keeping
   `WRITE|READ`) on the live payload immediately after `alloc_with_slack` succeeds, and again on
   the whole bucket after `reopen_slack` succeeds — the alloc-side analogue of `hooks.rs`'s
   existing free-side "delay to return" fix for SLUB's freelist-pointer write. This keeps the
   slack/OOB and free/quarantine/UAF protection (the actual ask), and only drops the
   uninitialized-read oracle for kmalloc payloads specifically — which real KASAN doesn't have
   either (that's KMSAN's separate, not-yet-built job per `docs/emulator-sanitizers.md`), so this
   is not a regression against the thing being approximated.

**Result (with both fixes in place):** `fuzz --sanitize --cases 500 --seed 1` and `--cases 2000
--seed 1` against `firmware/Image` both completed with **0 kernel crashes** — and, notably, with
coverage-bucket counts, corpus size, and syscalls-done/budget-hit counts **byte-identical** to the
same seed/case-count run with `--sanitize` entirely absent (2000-case run: 12088 coverage buckets,
421-program corpus, 1839 done/161 budget-hit in both cases). That exact match is strong evidence
the sanitizer's poisoning no longer perturbs legitimate guest execution at all on this campaign —
the 0% this task set out to confirm, replacing the old in-place cross-object redzone's measured
~40% false-positive rate. Sanitizer activity over the 2000-case run: 73 kmalloc / 6738 kfree
tracked, 0 page-alloc / 8 page-free (the page-alloc/free asymmetry is the documented
`alloc_pages`/`struct page*` coverage gap, not a false positive — `PageSanitizer::free_pages` on an
address it never saw allocated safely reports `SanError::InvalidFree` without touching memory),
5504 total `SanError`s (overwhelmingly `kfree`/`kmem_cache_free` of objects allocated via the
documented-unhooked `kmem_cache_alloc` path — a coverage gap, not a spatial false positive, since
`Sanitizer::free` returns before poisoning anything when the address isn't tracked live).
Before the RAW-vs-`GFP_ZERO` fix, the identical 500-case run produced 4 crashes (1 unique
signature) — the false positive this section documents catching and fixing.

Before this task, div-by-zero (RISC-V's other from-`docs/emulator-sanitizers.md` "build now" item)
had no emulator-side detection at all, since RV32 DIV/0 and REM/0 are defined (saturating result,
no trap). `fs-riscv`'s `Cpu` now carries an `Option<Vec<u32>>` UBSAN log mirroring the existing
CMPLOG mechanism exactly (`set_ubsan`/`ubsan_take`, zero cost when disabled); `fuzz --ubsan` arms
it once (before `Snapshot::capture`, so every case's reset restores a clean armed log for free) and
reports a deduped-by-pc finding count. A 2000-case run against the same clean kernel found 0
div-by-zero hits (an honest result — this kernel build's exercised syscall surface didn't reach a
reachable div/0 in that budget, not a wiring failure) with coverage identical to the non-`--ubsan`
baseline, confirming it's purely observational when armed and free when not.
