//! Page-granularity UAF/OOB sanitizer: the emulator-native equivalent of
//! `CONFIG_DEBUG_PAGEALLOC` (`firmware/Image.dpalloc`, `docs/kernel-san.md`'s KASAN/KFENCE
//! investigation) — recommended as the "stretch" item in `docs/emulator-sanitizers.md`'s KASAN
//! section: "PC-hook the *page* allocator ... poison/unpoison whole physical pages. Zero
//! false-positive (whole pages), portable to a Windows-pool target".
//!
//! **Deliberately a separate type from [`crate::Sanitizer`]** (the kmalloc/kfree byte-granularity
//! slack-only sanitizer in `alloc.rs`), not a new method bolted onto it: the two track different
//! address *domains* over the same physical memory — `Sanitizer` keys on arbitrary kmalloc object
//! addresses, `PageSanitizer` keys on page-aligned page-range base addresses spanning
//! `PAGE << order` bytes. Sharing one `live` map would risk a kmalloc object address coinciding
//! with an unrelated page-range base and producing a bogus `DoubleAlloc`/`InvalidFree` — sharing
//! bookkeeping across two different granularities is exactly the kind of accidental coupling a
//! zero-false-positive design should not introduce. Keeping them as separate types with separate
//! bookkeeping (this module's [`PageSanitizer`] vs [`crate::Sanitizer`]) makes that impossible by
//! construction, mirroring the very "never touch a neighbor" property whole-page poisoning itself
//! is built on.
//!
//! **Why whole pages, never sub-page, is zero-false-positive by construction:** the guest's own
//! page allocator (the buddy allocator) never hands out two live, independent allocations sharing
//! one physical page — that invariant is the allocator's entire job. So poisoning
//! `[base_pa, base_pa + (PAGE << order))` on [`PageSanitizer::free_pages`] and un-poisoning the
//! identical range on the next [`PageSanitizer::alloc_pages`] of that exact range can never stamp
//! a byte that legitimately belongs to some other, still-live allocation — unlike a cross-object
//! byte-granular redzone on a *packed* allocator (SLUB kmalloc caches), which is exactly the ~40%
//! false-positive mechanism `docs/kernel-san.md`'s experiment measured. This is a **different**,
//! complementary class to [`crate::Sanitizer::alloc_with_slack`]'s slack-only kmalloc OOB: it
//! catches immediate UAF/OOB on `order>0` allocations, `vmalloc`-backed pages, and fully-emptied
//! SLUB slab pages reclaimed back to the page allocator — not small in-slab kmalloc overflows
//! (SLUB packs several kmalloc objects per page, and the page stays mapped as long as *any*
//! object on it is live; that class still needs `firmware/Image.slubdebug`'s kernel-cooperative
//! free-time redzone check — see `docs/emulator-sanitizers.md`'s honest ceiling section).
//!
//! **Permission model, and why it differs from `Sanitizer::alloc`'s `WRITE | RAW`:** a freshly
//! (re)allocated page range is stamped plain `READ | WRITE`, with **no** [`fs_mmu::PERM_RAW`].
//! This mirrors real `CONFIG_DEBUG_PAGEALLOC` semantics exactly — it only ever unmaps a page on
//! free and remaps it (present, normally accessible) on the next allocation; it has no separate
//! uninitialized-read check layered on top. Reusing `Sanitizer::alloc`'s `WRITE | RAW` convention
//! here would be a *new* false-positive surface this design does not want: a whole guest page is
//! legitimately read by kernel code in patterns that don't hold to a "write before read" discipline
//! at page granularity (DMA buffers, driver-mapped pages, pages a get_zeroed_page caller reads
//! before writing every byte, etc.) — the object-level uninitialized-read oracle is a reasonable
//! assumption for a single kmalloc object; it is not a reasonable assumption for an entire
//! page's worth of bytes.
//!
//! **Free of an untracked page range and quarantine's meaning here:** [`PageSanitizer::free_pages`]
//! reports [`SanError::InvalidFree`] if the given `base_pa` is not currently tracked as live —
//! exactly [`crate::Sanitizer::free`]'s discipline, for the same reason (a real bug signal in the
//! fully-covered case). In practice, until a future follow-up also hooks the `struct page*`-based
//! `alloc_pages`/`__alloc_pages`/`__free_pages` family (deferred — see `crate::linux`'s
//! `Convention::PageStructUnavailable` doc comment) and handles boot-time pages allocated before
//! hooks were armed, some legitimate frees of pages this sanitizer never saw allocated will
//! surface as `InvalidFree` — a coverage gap, not a spatial false positive, exactly analogous to
//! the already-documented `kmem_cache_alloc` gap and the snapshot/reset lifecycle desync in
//! `docs/kernel-san.md` §3 point 7 (which applies to [`PageSanitizer`] too: a future run-loop
//! integration must `Clone` and restore it every case in lockstep with `Sanitizer` and `Mmu`'s own
//! snapshot reset — `PageSanitizer` already derives `Clone` for exactly this reason).

use std::collections::{HashMap, VecDeque};

use fs_mmu::{Mmu, PERM_READ, PERM_WRITE};

use crate::alloc::SanError;

/// Page size for this target: `arch/riscv/include/asm/page.h`'s `PAGE_SIZE` for RV32 (Sv32,
/// 4 KiB pages) — fixed, like `linux::PAGE_OFFSET`.
pub const PAGE: u32 = 4096;

/// How many freed page ranges to remember before evicting the oldest from *tracking* (their bytes
/// stay poisoned regardless — mirrors [`crate::DEFAULT_QUARANTINE_CAP`]'s discipline exactly, see
/// that constant's doc comment for the full rationale).
pub const DEFAULT_PAGE_QUARANTINE_CAP: usize = 4096;

/// Bookkeeping for one live page range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LivePageRange {
    /// The order this range was (re)allocated/freed with — `PAGE << order` bytes starting at the
    /// range's base address.
    order: u32,
}

/// The page-granularity sanitizer: whole-page poison-on-free + quarantine over a [`fs_mmu::Mmu`],
/// independent of [`crate::Sanitizer`]'s byte-granular kmalloc/kfree bookkeeping. See the module
/// doc comment for the full design rationale.
///
/// Carries no reference to the `Mmu` itself, exactly like [`crate::Sanitizer`] — every method
/// takes `&mut Mmu` explicitly.
#[derive(Clone)]
pub struct PageSanitizer {
    quarantine_cap: usize,
    live: HashMap<u32, LivePageRange>,
    /// FIFO of quarantined page-range base addresses, oldest first.
    quarantine_order: VecDeque<u32>,
    quarantined: HashMap<u32, LivePageRange>,
}

impl Default for PageSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PageSanitizer {
    /// A page sanitizer with the default quarantine bound.
    pub fn new() -> Self {
        Self::with_quarantine_cap(DEFAULT_PAGE_QUARANTINE_CAP)
    }

    /// Full control over the quarantine bound (mainly for tests wanting a tiny cap).
    pub fn with_quarantine_cap(quarantine_cap: usize) -> Self {
        Self {
            quarantine_cap,
            live: HashMap::new(),
            quarantine_order: VecDeque::new(),
            quarantined: HashMap::new(),
        }
    }

    /// True if `base_pa` is a currently-live page range's base address.
    pub fn is_live(&self, base_pa: u32) -> bool {
        self.live.contains_key(&base_pa)
    }

    /// True if `base_pa` is a freed page range still sitting in quarantine (poisoned, not yet
    /// reused). As with [`crate::Sanitizer::is_quarantined`], quarantine bytes stay poisoned even
    /// after eviction from *tracking* — this only reflects whether we still recognize `base_pa`
    /// specifically as a freed range.
    pub fn is_quarantined(&self, base_pa: u32) -> bool {
        self.quarantined.contains_key(&base_pa)
    }

    /// The order a currently-live page range at `base_pa` was allocated with, if any.
    pub fn live_order(&self, base_pa: u32) -> Option<u32> {
        self.live.get(&base_pa).map(|r| r.order)
    }

    /// `PAGE << order` as a byte length, or `None` if that would overflow a `u32`. `order` is
    /// bounded at 19 (`PAGE << 19` is `2^31`, the largest shift that still fits `u32` headroom for
    /// this target's 32-bit address space) — no real buddy-allocator order ever comes remotely
    /// close to that (Linux's own `MAX_PAGE_ORDER` is in the low teens at most), so an `order`
    /// this large is a strong signal of a garbage register (wrong hook wiring, or the hook firing
    /// on an unrelated call), not a real allocation. Public so a caller computing the same length
    /// elsewhere (e.g. the fs-cli follow-up, when logging) doesn't duplicate this bound.
    pub fn page_range_len(order: u32) -> Option<u32> {
        if order > 19 {
            return None;
        }
        Some(PAGE << order)
    }

    /// Record a new page-range allocation of `PAGE << order` bytes at `base_pa` (the guest's own
    /// buddy allocator decided the address; like [`crate::Sanitizer::alloc`], this sanitizer never
    /// allocates address space itself, only annotates what the guest allocator already decided).
    ///
    /// Effects:
    /// - `[base_pa, base_pa + (PAGE << order))` is stamped `READ | WRITE` (no `RAW` — see the
    ///   module doc comment for why page granularity does not layer the uninitialized-read oracle
    ///   on top).
    /// - If `base_pa` was in quarantine, it is evicted from quarantine tracking (a real allocator
    ///   handing the same freed page range back out is routine, not suspicious — exactly
    ///   [`crate::Sanitizer::alloc`]'s reasoning).
    ///
    /// Errors: [`SanError::DoubleAlloc`] if `base_pa` is already live; [`SanError::InvalidPageOrder`]
    /// if `order` is unreasonably large (see [`Self::page_range_len`]).
    pub fn alloc_pages(&mut self, mmu: &mut Mmu, base_pa: u32, order: u32) -> Result<(), SanError> {
        if self.live.contains_key(&base_pa) {
            return Err(SanError::DoubleAlloc { addr: base_pa });
        }
        let len = Self::page_range_len(order).ok_or(SanError::InvalidPageOrder {
            addr: base_pa,
            order,
        })?;

        mmu.protect(base_pa, len, PERM_READ | PERM_WRITE)?;

        self.evict_quarantine(&base_pa);
        self.live.insert(base_pa, LivePageRange { order });
        Ok(())
    }

    /// Free the page range at `base_pa`: poison `[base_pa, base_pa + (PAGE << order))` no-access
    /// and move it into quarantine, so a subsequent access — before this exact range is ever
    /// reused by [`Self::alloc_pages`] — faults as a page-granularity use-after-free instead of
    /// silently succeeding. This is the direct emulator-native analogue of
    /// `CONFIG_DEBUG_PAGEALLOC` unmapping a page the instant `free_pages()` returns it to the
    /// buddy allocator.
    ///
    /// The `order` passed here — the guest's own `free_pages(addr, order)` argument — is what
    /// determines the poisoned length, not whatever order [`Self::alloc_pages`] originally
    /// recorded (if any): `free_pages()` is the guest's own authoritative declaration of how many
    /// pages are being returned to the allocator right now, so it is trusted directly, exactly as
    /// [`crate::Sanitizer::free`] trusts its caller's pointer without re-deriving a size.
    ///
    /// Errors: [`SanError::InvalidFree`] if `base_pa` is not currently live (see the module doc
    /// comment's honest note on why this can include legitimate frees of pages this sanitizer
    /// never saw allocated, until the `alloc_pages`/`struct page*` gap is closed);
    /// [`SanError::InvalidPageOrder`] if `order` is unreasonably large.
    pub fn free_pages(&mut self, mmu: &mut Mmu, base_pa: u32, order: u32) -> Result<(), SanError> {
        let Some(_prev) = self.live.remove(&base_pa) else {
            return Err(SanError::InvalidFree { addr: base_pa });
        };
        let len = Self::page_range_len(order).ok_or(SanError::InvalidPageOrder {
            addr: base_pa,
            order,
        })?;

        mmu.poison(base_pa, len)?;

        if self.quarantine_order.len() >= self.quarantine_cap
            && let Some(oldest) = self.quarantine_order.pop_front()
        {
            self.quarantined.remove(&oldest);
        }
        self.quarantine_order.push_back(base_pa);
        self.quarantined.insert(base_pa, LivePageRange { order });
        Ok(())
    }

    fn evict_quarantine(&mut self, base_pa: &u32) {
        if self.quarantined.remove(base_pa).is_some() {
            self.quarantine_order.retain(|a| a != base_pa);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::FaultKind;

    fn mmu() -> Mmu {
        // Plenty of room for several page-aligned ranges plus neighbor pages either side.
        Mmu::new(0x8000_0000, 0x0010_0000)
    }

    #[test]
    fn alloc_pages_is_accessible_read_write_no_raw_oracle() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8000_1000;
        san.alloc_pages(&mut mmu, base, 0).unwrap();

        // Unlike `Sanitizer::alloc`, an unwritten byte does NOT fault: no RAW stamped at page
        // granularity (see module doc comment).
        assert_eq!(mmu.read_u8(base).unwrap(), 0);
        mmu.write(base, &[1, 2, 3, 4]).unwrap();
        let mut buf = [0u8; 4];
        mmu.read(base, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
        assert!(san.is_live(base));
        assert_eq!(san.live_order(base), Some(0));
    }

    #[test]
    fn free_pages_poisons_exactly_the_whole_page_and_no_further() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8000_2000;
        san.alloc_pages(&mut mmu, base, 0).unwrap();
        mmu.write(base, &[0xAA; 16]).unwrap();

        san.free_pages(&mut mmu, base, 0).unwrap();

        // Every byte across the whole freed page faults (use-after-free-of-page).
        for off in [0u32, 1, 2, 100, PAGE / 2, PAGE - 1] {
            assert_eq!(
                mmu.read_u8(base + off).unwrap_err().kind,
                FaultKind::Permission,
                "byte at offset {off} should be poisoned"
            );
            assert_eq!(
                mmu.write_u8(base + off, 0x41).unwrap_err().kind,
                FaultKind::Permission,
                "byte at offset {off} should be poisoned"
            );
        }

        // The byte exactly at base + PAGE — the first byte of the *next* page — is completely
        // untouched: still in its pristine pre-poison state (perm 0, same as a fresh Mmu), proving
        // this call never wrote past the exact page boundary.
        assert_eq!(mmu.perm_at(base + PAGE), Some(0));
        // A legitimate neighbor allocation on that next page is unaffected by this free.
        mmu.protect(base + PAGE, 4, PERM_READ | PERM_WRITE).unwrap();
        mmu.write(base + PAGE, &[9, 9, 9, 9]).unwrap();
        let mut nbuf = [0u8; 4];
        mmu.read(base + PAGE, &mut nbuf).unwrap();
        assert_eq!(nbuf, [9, 9, 9, 9]);

        assert!(san.is_quarantined(base));
        assert!(!san.is_live(base));
    }

    #[test]
    fn use_after_free_of_page_faults() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8000_3000;
        san.alloc_pages(&mut mmu, base, 0).unwrap();
        mmu.write(base, &[7; 8]).unwrap();

        san.free_pages(&mut mmu, base, 0).unwrap();

        // A stale pointer read/write after the free faults, the whole-page UAF oracle.
        assert_eq!(mmu.read_u8(base).unwrap_err().kind, FaultKind::Permission);
        assert_eq!(
            mmu.write_u8(base, 0x41).unwrap_err().kind,
            FaultKind::Permission
        );
    }

    #[test]
    fn realloc_of_the_same_page_range_unpoisons_it() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8000_4000;
        san.alloc_pages(&mut mmu, base, 0).unwrap();
        mmu.write(base, &[1; 16]).unwrap();
        san.free_pages(&mut mmu, base, 0).unwrap();
        assert!(san.is_quarantined(base));

        // The buddy allocator hands the same page range back out (routine under fuzzing-scale
        // churn) — a fresh alloc_pages() must fully un-poison it.
        san.alloc_pages(&mut mmu, base, 0).unwrap();
        assert!(!san.is_quarantined(base));
        assert!(san.is_live(base));

        // Freshly (re)allocated: readable/writable again, not somehow still poisoned. (Contents
        // are whatever was last written — page granularity has no RAW/zero-on-alloc oracle, see
        // the module doc comment — only the *permission* is restored.)
        assert!(mmu.read_u8(base).is_ok());
        mmu.write(base, &[2; 16]).unwrap();
        let mut buf = [0u8; 16];
        mmu.read(base, &mut buf).unwrap();
        assert_eq!(buf, [2; 16]);
    }

    #[test]
    fn order_greater_than_zero_covers_all_pages_in_the_range() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8001_0000; // page-aligned
        let order = 3u32; // 8 pages = 32 KiB
        san.alloc_pages(&mut mmu, base, order).unwrap();

        let len = PageSanitizer::page_range_len(order).unwrap();
        assert_eq!(len, PAGE * 8);

        // Every page across the whole multi-page range is accessible immediately after alloc.
        for page in 0..8u32 {
            let addr = base + page * PAGE;
            mmu.write(addr, &[page as u8; 4]).unwrap();
            let mut buf = [0u8; 4];
            mmu.read(addr, &mut buf).unwrap();
            assert_eq!(buf, [page as u8; 4]);
        }

        san.free_pages(&mut mmu, base, order).unwrap();

        // Every page across the whole multi-page range is poisoned after free...
        for page in 0..8u32 {
            let addr = base + page * PAGE;
            assert_eq!(
                mmu.read_u8(addr).unwrap_err().kind,
                FaultKind::Permission,
                "page index {page} should be poisoned after free_pages(order=3)"
            );
        }
        // ...but the page immediately after the whole 8-page range is untouched.
        assert_eq!(mmu.perm_at(base + PAGE * 8), Some(0));
    }

    #[test]
    fn poisoned_pages_neighbor_page_is_independently_accessible() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let victim = 0x8002_0000;
        let neighbor = victim + PAGE;

        // Both pages are live, independent allocations.
        san.alloc_pages(&mut mmu, victim, 0).unwrap();
        san.alloc_pages(&mut mmu, neighbor, 0).unwrap();
        mmu.write(victim, &[1; 8]).unwrap();
        mmu.write(neighbor, &[2; 8]).unwrap();

        // Freeing only the victim page must not disturb the neighbor at all.
        san.free_pages(&mut mmu, victim, 0).unwrap();

        assert_eq!(mmu.read_u8(victim).unwrap_err().kind, FaultKind::Permission);
        let mut nbuf = [0u8; 8];
        mmu.read(neighbor, &mut nbuf).unwrap();
        assert_eq!(nbuf, [2; 8]);
        mmu.write(neighbor, &[3; 8]).unwrap();
        assert!(san.is_live(neighbor));
        assert!(!san.is_live(victim));
    }

    #[test]
    fn double_alloc_is_reported() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8003_0000;
        san.alloc_pages(&mut mmu, base, 0).unwrap();
        assert_eq!(
            san.alloc_pages(&mut mmu, base, 0).unwrap_err(),
            SanError::DoubleAlloc { addr: base }
        );
    }

    #[test]
    fn double_free_is_reported() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8004_0000;
        san.alloc_pages(&mut mmu, base, 0).unwrap();
        san.free_pages(&mut mmu, base, 0).unwrap();
        assert_eq!(
            san.free_pages(&mut mmu, base, 0).unwrap_err(),
            SanError::InvalidFree { addr: base }
        );
    }

    #[test]
    fn free_of_untracked_page_is_reported_not_silently_ignored() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8005_0000;
        assert_eq!(
            san.free_pages(&mut mmu, base, 0).unwrap_err(),
            SanError::InvalidFree { addr: base }
        );
    }

    #[test]
    fn absurd_order_is_reported_not_wrapped() {
        let mut mmu = mmu();
        let mut san = PageSanitizer::new();
        let base = 0x8006_0000;
        assert_eq!(
            san.alloc_pages(&mut mmu, base, 4_000_000_000).unwrap_err(),
            SanError::InvalidPageOrder {
                addr: base,
                order: 4_000_000_000
            }
        );
        // page_range_len itself is the same guard, independently testable.
        assert_eq!(PageSanitizer::page_range_len(20), None);
        assert_eq!(PageSanitizer::page_range_len(19), Some(PAGE << 19));
    }

    #[test]
    fn quarantine_cap_bounds_tracking_but_pages_stay_poisoned() {
        let mut mmu = Mmu::new(0x8000_0000, 0x0010_0000);
        let mut san = PageSanitizer::with_quarantine_cap(2);

        let a = 0x8000_0000u32;
        let b = 0x8000_1000u32;
        let c = 0x8000_2000u32;
        for addr in [a, b, c] {
            san.alloc_pages(&mut mmu, addr, 0).unwrap();
            san.free_pages(&mut mmu, addr, 0).unwrap();
        }

        // Cap is 2: the oldest (`a`) was evicted from *tracking* once `c` was freed...
        assert!(!san.is_quarantined(a));
        assert!(san.is_quarantined(b));
        assert!(san.is_quarantined(c));

        // ...but the page stays poisoned regardless of tracking (safety never regresses).
        assert_eq!(mmu.read_u8(a).unwrap_err().kind, FaultKind::Permission);
    }
}
