//! Kernel-heap sanitization for the RV32 Linux target: turn a `System.map` into `PcHooks`
//! registrations for the kernel's slab allocator, so `hooks.rs`'s PC-hook mechanism (§3b of
//! `DESIGN.md`) can learn `kmalloc`/`kfree` call sites **without any kernel instrumentation** —
//! no `CONFIG_KASAN`, no recompilation, just a symbol table the kernel build already produces.
//!
//! This module is the "worked example" DESIGN.md promises for the closed-source-adjacent case:
//! we have kernel *source* (it's our own build), but we deliberately do not want to touch it —
//! the whole point is that the same mechanism also reaches a kernel (or driver) we have no source
//! for at all, provided we can resolve its allocator entry points to addresses some other way
//! (DWARF, a leaked PDB, ...). `System.map` is simply the friendliest available symbol source for
//! our own build.
//!
//! Two things vary across kernel versions/configs and both are handled here:
//! 1. **Which symbol names exist.** Slab allocator entry points have been renamed and split
//!    several times (e.g. the `_noprof` variants added by the allocation-profiling feature).
//!    [`register_kernel_allocator_hooks`] tries every name it knows and registers whichever are
//!    actually present in the supplied symbol table.
//! 2. **Where the size argument is**, which is *not* uniformly "a0" despite that being the common
//!    case — see the per-symbol comments below. Getting this wrong doesn't crash anything, it
//!    just makes `fs-san` learn the wrong size for an allocation (a false-positive/false-negative
//!    redzone), so it's worth being precise rather than assuming every function looks like
//!    `kmalloc(size, flags)`.

use std::collections::HashMap;

use crate::hooks::{
    AllocHook, CacheAllocHook, FreeHook, KsizeHook, PageAllocHook, PageFreeHook, PcHooks,
};

/// Parse a Linux `System.map` (or an equivalent `nm -n vmlinux`-style listing) into a
/// symbol-name -> address table.
///
/// Expected line shape: `HEXADDR TYPE SYMBOL` (whitespace-separated), e.g.:
/// ```text
/// c0123456 T kmalloc
/// c01a2b3c t some_static_helper
/// ```
/// `TYPE` is the single-character nm/System.map symbol type (`T`/`t` = text/local-text,
/// `W`/`w` = weak, `D`/`d` = data, `B`/`b` = bss, ...) — both cases are accepted and the letter
/// itself is not otherwise inspected, since all we need is "does this name resolve to an
/// address," not its section. Addresses are parsed as plain (no `0x` prefix) hex, matching
/// `System.map`'s format, and truncated to 32 bits (this is an RV32 target).
///
/// Malformed lines (blank, comments, fewer than three fields, a non-hex address field) are
/// silently skipped rather than erroring — a `System.map` is large and we only care about the
/// handful of allocator symbols we go looking for afterwards.
pub fn parse_system_map(text: &str) -> HashMap<String, u32> {
    let mut map = HashMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let (Some(addr_field), Some(ty_field), Some(name_field)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        // The type letter must be present and alphabetic; we don't otherwise filter on it (see
        // doc comment above) but a sanity check here rejects lines that merely *look* like a
        // symbol line (e.g. two hex numbers with no name).
        if !ty_field.chars().all(|c| c.is_ascii_alphabetic()) || ty_field.is_empty() {
            continue;
        }
        let Ok(addr) = u32::from_str_radix(addr_field, 16) else {
            continue;
        };
        map.insert(name_field.to_string(), addr);
    }
    map
}

/// One kernel allocator symbol's calling convention, as known to
/// [`register_kernel_allocator_hooks`].
enum Convention {
    /// Alloc-shaped: size argument lives in this register index (RISC-V `a0..a7` = x10..x17).
    AllocSizeReg(usize),
    /// `kmem_cache_alloc`-shaped: the size is not an argument at all (it's `cachep->object_size`,
    /// a guest-memory field), but the *cache pointer* is — this register index. Handled by
    /// [`crate::hooks::PcHooks::hook_cache_alloc`]/`on_cache_alloc_pc` (see
    /// [`crate::hooks::CacheAllocHook`]'s doc comment for the full mechanism: the cache pointer
    /// is stashed here and the actual `object_size` guest-memory read happens at the call site
    /// that owns an `Mmu`, not in this register-file-only crate module).
    CacheAllocCacheReg(usize),
    /// Free-shaped: pointer argument lives in this register index.
    FreePtrReg(usize),
    /// `ksize()`-shaped: the guest is reporting/re-opening the usable size of a live allocation.
    /// Pointer argument lives in this register index. See `docs/emulator-sanitizers.md`'s KASAN
    /// section and [`crate::Sanitizer::reopen_slack`] — this is the hook half of that primitive.
    KsizePtrReg(usize),
    /// Page-allocator alloc-shaped: `fn(..., order) -> VA`, tractable via [`LinearMap`] because
    /// the return value is *already* a linear-map virtual address (unlike `alloc_pages`/
    /// `__alloc_pages`, see [`Convention::PageStructUnavailable`]). Order argument lives in this
    /// register index, or `None` if the function has no order argument at all (always order 0).
    /// Handled by [`crate::hooks::PcHooks::hook_page_alloc`]/`on_page_pc` — see
    /// `docs/emulator-sanitizers.md`'s KASAN section, item (d) ("emulator-native page-granularity
    /// UAF") — rather than `hook_alloc`/`on_pc`, since page-granularity events are a separate,
    /// independently-queried family (see [`crate::hooks::PageHookEvent`]'s doc comment).
    PageAllocOrderReg(Option<usize>),
    /// Page-allocator free-shaped: `fn free_pages(addr, order)`. `addr` is already a linear-map VA
    /// (the free-side mirror of `PageAllocOrderReg`'s return value). Pointer argument register,
    /// then order argument register (or `None` for an implicit order 0). Handled by
    /// [`crate::hooks::PcHooks::hook_page_free`]/`on_page_pc`.
    PageFreeAddrOrderReg(usize, Option<usize>),
    /// Alloc- or free-shaped in principle (`alloc_pages`/`__alloc_pages`/`__free_pages`), but the
    /// function takes/returns a `struct page *`, not a linear-map virtual address — turning that
    /// into a physical page requires `page_to_pfn`/`mem_map` arithmetic (a guest-memory struct
    /// walk at a kernel-version-specific offset, a *much* larger/riskier reverse-engineering task
    /// than [`Convention::CacheAllocCacheReg`]'s single fixed-field-offset read — see that
    /// variant's doc comment for the one this project *did* take on). Listed here purely as
    /// documentation of a considered-but-deferred symbol, never hooked. This is the honest gap
    /// `docs/emulator-sanitizers.md`'s KASAN section flags: `__get_free_pages`/`free_pages` (the
    /// VA-returning family, handled by the two variants above) are the tractable case;
    /// `alloc_pages`/`struct page*` is the harder follow-up.
    PageStructUnavailable,
}

/// Every kernel slab-allocator entry point this module knows how to hook, with its calling
/// convention. Register order/index follows the standard RISC-V integer calling convention:
/// `a0`=x10, `a1`=x11, `a2`=x12, ...
///
/// Kept as a plain slice (not a `HashMap`) since it's small, iterated once per call to
/// [`register_kernel_allocator_hooks`], and its order is itself documentation.
const KNOWN_SYMBOLS: &[(&str, Convention)] = &[
    // --- Allocators -----------------------------------------------------------------------
    // `void *kmalloc(size_t size, gfp_t flags)` -> size is the first argument, a0.
    ("kmalloc", Convention::AllocSizeReg(10)),
    // `void *__kmalloc(size_t size, gfp_t flags)` -> same shape as kmalloc; the underscored
    // name is the out-of-line slow path kmalloc()'s inline fast path falls back to.
    ("__kmalloc", Convention::AllocSizeReg(10)),
    // `void *__kmalloc_noprof(size_t size, gfp_t flags)` -> allocation-profiling
    // (CONFIG_MEM_ALLOC_PROFILING) build of `__kmalloc`; profiling metadata is threaded through
    // a separate mechanism (a per-callsite tag), not an extra argument here, so size stays a0.
    ("__kmalloc_noprof", Convention::AllocSizeReg(10)),
    // `void *kmalloc_noprof(size_t size, gfp_t flags)` -> allocation-profiling build of
    // `kmalloc`; size stays a0.
    ("kmalloc_noprof", Convention::AllocSizeReg(10)),
    // `void *__kmalloc_node(size_t size, gfp_t flags, int node)` -> size is still the first
    // argument (the NUMA node is appended at the end), so a0.
    ("__kmalloc_node", Convention::AllocSizeReg(10)),
    // `void *kmalloc_trace(struct kmem_cache *s, gfp_t flags, size_t size)` -> this one does
    // *not* follow the "size in a0" pattern: it's called from kmalloc()'s inline fast path once
    // the compile-time-constant size has already been mapped to a kmem_cache, so the cache
    // pointer leads and the (still separately tracked, for /proc/slabinfo accounting) requested
    // size is the *third* argument -> a2 (x12), not a0.
    ("kmalloc_trace", Convention::AllocSizeReg(12)),
    // `void *kmem_cache_alloc(struct kmem_cache *cachep, gfp_t flags)` -> the requested size is
    // NOT an argument at all; it's an intrinsic property of `cachep` (its
    // `object_size` field), which this register-file-only PC-hook layer cannot dereference
    // directly. Closed via `Convention::CacheAllocCacheReg`: the cache pointer (a0) is stashed
    // instead of a size, and the call site that owns an `Mmu` (fs-cli's `run_case`) reads
    // `cachep->object_size` itself — see `crate::hooks::CacheAllocHook`'s doc comment and
    // `KMEM_CACHE_OBJECT_SIZE_OFFSET` below for the guest-memory offset this depends on.
    ("kmem_cache_alloc", Convention::CacheAllocCacheReg(10)),
    // `void *kmem_cache_alloc_noprof(struct kmem_cache *cachep, gfp_t flags)` -> same shape as
    // `kmem_cache_alloc` (allocation-profiling build).
    ("kmem_cache_alloc_noprof", Convention::CacheAllocCacheReg(10)),
    // --- Deallocators -----------------------------------------------------------------------
    // `void kfree(const void *objp)` -> the only argument is the pointer, a0.
    ("kfree", Convention::FreePtrReg(10)),
    // `void kmem_cache_free(struct kmem_cache *cachep, void *objp)` -> NOTE this is the one
    // free-shaped symbol whose pointer is *not* in a0: the cache leads, so the pointer being
    // freed is the second argument -> a1 (x11).
    ("kmem_cache_free", Convention::FreePtrReg(11)),
    // `void kfree_sensitive(const void *objp)` -> like kfree but zeroes the memory first
    // (formerly `kzfree`); single pointer argument, a0.
    ("kfree_sensitive", Convention::FreePtrReg(10)),
    // --- Usable-size query / slack re-open ---------------------------------------------------
    // `size_t ksize(const void *objp)` -> the only argument is the pointer, a0. Called by the
    // kernel (skb, crypto, mm code) to discover and legitimately use the full rounded-up bucket
    // size of an allocation — must re-open `alloc_with_slack`'s poisoned slack, not fault it.
    ("ksize", Convention::KsizePtrReg(10)),
    // `size_t __ksize(const void *objp)` -> same shape; the internal helper `ksize()` itself (and
    // some direct callers, e.g. mm/slab_common.c) resolve to.
    ("__ksize", Convention::KsizePtrReg(10)),
    // --- Page allocator (whole-page granularity; docs/emulator-sanitizers.md's KASAN "stretch"
    // item (d), the emulator-native equivalent of CONFIG_DEBUG_PAGEALLOC / firmware/Image.dpalloc)
    // -----------------------------------------------------------------------------------------
    // `unsigned long __get_free_pages(gfp_t gfp_mask, unsigned int order)` -> mm/page_alloc.c;
    // the return value is already a linear-map VA (this is the tractable, VA-returning half of
    // the page allocator, unlike alloc_pages/struct page* below). order is the *second* argument,
    // a1 (x11).
    ("__get_free_pages", Convention::PageAllocOrderReg(Some(11))),
    // `unsigned long get_zeroed_page(gfp_t gfp_mask)` -> mm/page_alloc.c; same VA-returning shape
    // as __get_free_pages, but it always requests order 0 internally
    // (`__get_free_pages(gfp_mask | __GFP_ZERO, 0)`) -- there is no order argument to read here.
    ("get_zeroed_page", Convention::PageAllocOrderReg(None)),
    // `#define __get_free_page(gfp_mask) __get_free_pages((gfp_mask), 0)` and the equivalent
    // `get_free_page` spelling some call sites use -- in mainline these are header-inline macros
    // that expand directly to a call to __get_free_pages, so they have no symbol of their own in
    // a normal build (`register_kernel_allocator_hooks` simply won't find them in `syms`, at zero
    // cost). Listed anyway, same reasoning `KNOWN_SYMBOLS`'s doc comment gives for `kmem_cache_alloc`:
    // documents the name was considered, and costs nothing if some future kernel version/config
    // ever gives one of these a real out-of-line definition.
    ("__get_free_page", Convention::PageAllocOrderReg(None)),
    ("get_free_page", Convention::PageAllocOrderReg(None)),
    // `void free_pages(unsigned long addr, unsigned int order)` -> mm/page_alloc.c; addr is
    // already a linear-map VA (the free-side mirror of __get_free_pages), a0; order is a1. Fires
    // immediately at entry (not delayed to return like `kfree`): the page allocator's own free
    // path does not write into the freed page's payload before this entry PC, so there is no
    // SLUB-style race to guard against here (see `hooks.rs`'s `PageFreeHook` doc comment).
    ("free_pages", Convention::PageFreeAddrOrderReg(10, Some(11))),
    // `#define free_page(addr) free_pages((addr), 0)` -- likewise usually a macro with no symbol
    // of its own; listed for the same reason as `__get_free_page` above.
    ("free_page", Convention::PageFreeAddrOrderReg(10, None)),
    // `struct page *alloc_pages(gfp_t gfp, unsigned int order)` / `struct page
    // *__alloc_pages(gfp_t gfp, unsigned int order, int preferred_nid, nodemask_t *nodemask)` ->
    // return a `struct page *`, not a linear-map VA -- see `Convention::PageStructUnavailable`'s
    // doc comment. Deliberately never hooked.
    ("alloc_pages", Convention::PageStructUnavailable),
    ("__alloc_pages", Convention::PageStructUnavailable),
    // `void __free_pages(struct page *page, unsigned int order)` -> the free-side mirror of
    // alloc_pages, with the identical struct-page limitation.
    ("__free_pages", Convention::PageStructUnavailable),
];

/// Register PC hooks for every kernel slab-allocator symbol in `KNOWN_SYMBOLS` that is present
/// in `syms` (as produced by [`parse_system_map`]). Symbol names and even calling conventions
/// drift across kernel versions/configs, so this tries each known name independently rather than
/// requiring a fixed set — whichever a given kernel build actually has get hooked.
///
/// `kmem_cache_alloc`/`kmem_cache_alloc_noprof`, if present, are registered as
/// [`crate::hooks::CacheAllocHook`]s (see [`Convention::CacheAllocCacheReg`]): the cache pointer is
/// captured here, but the actual `object_size` guest-memory read (and the `Sanitizer::alloc_with_slack`
/// call it feeds) is the caller's job, since this function only has a register file, not an `Mmu`.
/// Their frees (`kmem_cache_free`) are hooked normally as before: `Sanitizer::free` only needs the
/// pointer, not the original size.
pub fn register_kernel_allocator_hooks(hooks: &mut PcHooks, syms: &HashMap<String, u32>) {
    for (name, convention) in KNOWN_SYMBOLS {
        let Some(&entry_pc) = syms.get(*name) else {
            continue;
        };
        match *convention {
            Convention::AllocSizeReg(size_reg) => {
                hooks.hook_alloc(AllocHook { entry_pc, size_reg });
            }
            Convention::CacheAllocCacheReg(cache_reg) => {
                hooks.hook_cache_alloc(CacheAllocHook {
                    entry_pc,
                    cache_reg,
                });
            }
            Convention::FreePtrReg(ptr_reg) => {
                hooks.hook_free(FreeHook { entry_pc, ptr_reg });
            }
            Convention::KsizePtrReg(ptr_reg) => {
                hooks.hook_ksize(KsizeHook { entry_pc, ptr_reg });
            }
            Convention::PageAllocOrderReg(order_reg) => {
                hooks.hook_page_alloc(PageAllocHook {
                    entry_pc,
                    order_reg,
                });
            }
            Convention::PageFreeAddrOrderReg(addr_reg, order_reg) => {
                hooks.hook_page_free(PageFreeHook {
                    entry_pc,
                    addr_reg,
                    order_reg,
                });
            }
            Convention::PageStructUnavailable => {
                // Deliberately not hooked; see KNOWN_SYMBOLS/Convention doc comments.
            }
        }
    }
}

/// RV32 Linux `PAGE_OFFSET` (`arch/riscv/include/asm/page.h`, `!CONFIG_64BIT`) — fixed, not
/// runtime-computed on RV32.
pub const PAGE_OFFSET: u32 = 0xc000_0000;

/// The kernel lowmem linear map: a fixed affine VA↔PA offset. kmalloc/kfree pointers are always
/// linear-map addresses in stock SLUB (slab pages come from the buddy allocator, never vmalloc),
/// so one subtraction is the literal implementation of `__pa()` for this address class. The offset
/// is `PAGE_OFFSET - kernel_load_pa` (the address the Image was loaded at), NOT `- ram_base`.
pub struct LinearMap {
    va_pa_offset: u32,
    pa_lo: u32,
    pa_hi: u32,
}

impl LinearMap {
    pub fn new(kernel_load_pa: u32, ram_base: u32, ram_size: u32) -> Self {
        Self {
            va_pa_offset: PAGE_OFFSET.wrapping_sub(kernel_load_pa),
            pa_lo: ram_base,
            pa_hi: ram_base.wrapping_add(ram_size),
        }
    }

    /// Translate a kmalloc/kfree-observed kernel VA to a physical address, or reject it (NULL,
    /// `ZERO_SIZE_PTR` = 0x10, user addresses, or anything outside the mapped physical window —
    /// never blindly subtract, or a mistranslation poisons unrelated memory).
    pub fn va_to_pa(&self, va: u32) -> Option<u32> {
        if va < PAGE_OFFSET {
            return None;
        }
        let pa = va.wrapping_sub(self.va_pa_offset);
        (pa >= self.pa_lo && pa < self.pa_hi).then_some(pa)
    }
}

/// Byte offset of `struct kmem_cache::object_size` (`mm/slab.h`) for **this project's own kernel
/// build** (`build/linux-src`, confirmed against its `mm/slab.h` at the revision that build was
/// compiled from). Unlike `PAGE_OFFSET` (an ABI constant fixed by the RISC-V port itself), this is
/// a *struct layout* fact — it depends on the exact set of preceding fields and any
/// `CONFIG_`-gated ones among them, so it must be re-derived (re-read `mm/slab.h`, recompute the
/// RV32 field layout: 4-byte pointers/`unsigned int`/`unsigned long`, natural alignment, no
/// padding needed since every field before `object_size` is exactly 4 bytes) for any other kernel
/// version or `.config`. Field layout on this build's `struct kmem_cache` (RV32: 4-byte pointers,
/// no gaps since each field below is a 4-byte-aligned 4-byte value):
/// ```text
/// struct kmem_cache {
///     struct slub_percpu_sheaves __percpu *cpu_sheaves; // offset  0 (pointer, 4B)
///     slab_flags_t flags;                               // offset  4 (`unsigned int`, 4B)
///     unsigned long min_partial;                        // offset  8 (4B on RV32)
///     unsigned int size;                                // offset 12
///     unsigned int object_size;                          // offset 16  <-- this constant
///     ...
/// };
/// ```
/// Never trusted blindly at runtime: the call site that reads this (fs-cli's `run_case`) applies
/// [`KMEM_CACHE_OBJECT_SIZE_MAX`] as a sanity bound and skips the hook for that one call (falling
/// back to today's "invisible to the sanitizer" behavior, never a guessed/garbage size) if the
/// read value fails it — the same "getting this wrong doesn't crash, it just silently mis-sizes an
/// allocation" risk this module's own doc comment already calls out for symbol/convention drift,
/// just here for a struct-layout fact instead of a calling convention.
pub const KMEM_CACHE_OBJECT_SIZE_OFFSET: u32 = 16;

/// Sanity bound for a guest-memory-read `object_size` (see [`KMEM_CACHE_OBJECT_SIZE_OFFSET`]'s
/// doc comment): no real `kmem_cache` object is anywhere near this large, so a value at or above
/// it is a strong signal of a wrong offset (wrong kernel build/version/config) or a garbage
/// `cachep` pointer — not a real cache — and the read should be treated as unusable rather than
/// fed to the sanitizer.
pub const KMEM_CACHE_OBJECT_SIZE_MAX: u32 = 0x0010_0000; // 1 MiB

/// Round a kmalloc request up to its SLUB bucket, so the trailing redzone lands at the object
/// boundary (the kernel may legitimately access up to `ksize()` = the full bucket size).
pub fn kmalloc_bucket(size: u32) -> u32 {
    const BUCKETS: &[u32] = &[8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192];
    for &b in BUCKETS {
        if size <= b {
            return b;
        }
    }
    size.next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookEvent;

    #[test]
    fn linear_map_offset_and_window() {
        // kernel loaded at 0x8040_0000, RAM 0x8000_0000..+128MiB. _start VA 0xc0000000 -> PA.
        let lm = LinearMap::new(0x8040_0000, 0x8000_0000, 0x0800_0000);
        assert_eq!(lm.va_to_pa(0xc000_0000), Some(0x8040_0000)); // _start
        assert_eq!(lm.va_to_pa(0x0000_0000), None); // NULL
        assert_eq!(lm.va_to_pa(0x0000_0010), None); // ZERO_SIZE_PTR
        assert_eq!(lm.va_to_pa(0x1234_5678), None); // user address
        assert_eq!(kmalloc_bucket(30), 32);
        assert_eq!(kmalloc_bucket(64), 64);
        assert_eq!(kmalloc_bucket(100), 128);
    }

    /// Pins `KMEM_CACHE_OBJECT_SIZE_OFFSET`'s documented derivation (mm/slab.h field layout) as a
    /// regression guard — if this ever needs to change for a different kernel build, the change
    /// should be a deliberate edit to the constant *and* this test, not a silent drift.
    #[test]
    fn kmem_cache_object_size_offset_matches_documented_derivation() {
        assert_eq!(KMEM_CACHE_OBJECT_SIZE_OFFSET, 16);
        assert_eq!(KMEM_CACHE_OBJECT_SIZE_MAX, 0x0010_0000);
    }

    /// A tiny excerpt in real `System.map` shape: two allocator symbols we care about, plus
    /// assorted noise lines (blank, other symbol types, an unrelated function) that a real map
    /// is full of and that parsing must simply ignore.
    const SNIPPET: &str = "\
c0100000 T _stext
c0100004 t local_helper

c0123456 T kmalloc
c0123500 T kfree
c0140000 t some_bss_helper
c0150000 D some_data_symbol
not_a_valid_line at all
c0160000 W weak_symbol
";

    #[test]
    fn parse_system_map_finds_symbols_and_skips_noise() {
        let syms = parse_system_map(SNIPPET);
        assert_eq!(syms.get("kmalloc"), Some(&0xc0123456));
        assert_eq!(syms.get("kfree"), Some(&0xc0123500));
        assert_eq!(syms.get("_stext"), Some(&0xc0100000));
        assert_eq!(syms.get("local_helper"), Some(&0xc0100004));
        assert_eq!(syms.get("some_bss_helper"), Some(&0xc0140000));
        assert_eq!(syms.get("some_data_symbol"), Some(&0xc0150000));
        assert_eq!(syms.get("weak_symbol"), Some(&0xc0160000));
        // The malformed line contributed nothing and did not panic.
        assert_eq!(syms.len(), 7);
    }

    #[test]
    fn end_to_end_kmalloc_kfree_via_system_map() {
        let syms = parse_system_map(SNIPPET);
        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        let kmalloc_pc = syms["kmalloc"];
        let kfree_pc = syms["kfree"];

        // Hit kmalloc's entry: a0 = requested size, ra = return address.
        let mut regs = [0u32; 32];
        regs[10] = 48; // size
        regs[crate::hooks::REG_RETURN_ADDR] = 0xc020_0000; // ra
        assert_eq!(hooks.on_pc(kmalloc_pc, &regs), None);
        assert_eq!(hooks.pending_returns(), 1);

        // Hit the stashed return address: a0 now holds the allocated pointer.
        let mut ret_regs = [0u32; 32];
        ret_regs[crate::hooks::REG_RETURN_VALUE] = 0x8010_0000;
        assert_eq!(
            hooks.on_pc(0xc020_0000, &ret_regs),
            Some(HookEvent::Alloc {
                addr: 0x8010_0000,
                size: 48
            })
        );

        // kfree's entry: a0 = ptr, ra = return; the free is delayed to the return.
        let mut free_regs = [0u32; 32];
        free_regs[10] = 0x8010_0000;
        free_regs[crate::REG_RETURN_ADDR] = 0xc040_0000;
        assert_eq!(hooks.on_pc(kfree_pc, &free_regs), None);
        assert_eq!(
            hooks.on_pc(0xc040_0000, &[0u32; 32]),
            Some(HookEvent::Free {
                addr: 0x8010_0000
            })
        );
    }

    #[test]
    fn kmem_cache_alloc_is_now_registered_as_a_cache_alloc_hook_not_on_pc() {
        let mut syms = HashMap::new();
        syms.insert("kmem_cache_alloc".to_string(), 0xc030_0000u32);
        syms.insert("kmem_cache_alloc_noprof".to_string(), 0xc030_1000u32);
        syms.insert("kmem_cache_free".to_string(), 0xc030_2000u32);

        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        // kmem_cache_alloc(cachep, flags): a0 = cachep. Its size is not a register at all (see
        // Convention::CacheAllocCacheReg), so it must surface only through on_cache_alloc_pc, not
        // through the unrelated on_pc/HookEvent path.
        let mut entry = [0u32; 32];
        entry[10] = 0xc070_0000; // cachep
        entry[crate::REG_RETURN_ADDR] = 0xc031_0000;
        assert_eq!(hooks.on_pc(0xc030_0000, &entry), None);
        assert_eq!(hooks.on_cache_alloc_pc(0xc030_0000, &entry), None);
        assert_eq!(hooks.cache_alloc_pending_returns(), 1);

        let mut ret = [0u32; 32];
        ret[crate::hooks::REG_RETURN_VALUE] = 0x8060_0000;
        assert_eq!(
            hooks.on_cache_alloc_pc(0xc031_0000, &ret),
            Some(crate::hooks::CacheAllocEvent::Alloc {
                addr: 0x8060_0000,
                cache_ptr: 0xc070_0000
            })
        );
        // Never surfaced through on_pc.
        assert_eq!(hooks.on_pc(0xc031_0000, &ret), None);

        // The _noprof variant behaves identically.
        let mut entry2 = [0u32; 32];
        entry2[10] = 0xc070_1000;
        entry2[crate::REG_RETURN_ADDR] = 0xc031_1000;
        assert_eq!(hooks.on_cache_alloc_pc(0xc030_1000, &entry2), None);
        let mut ret2 = [0u32; 32];
        ret2[crate::hooks::REG_RETURN_VALUE] = 0x8060_1000;
        assert_eq!(
            hooks.on_cache_alloc_pc(0xc031_1000, &ret2),
            Some(crate::hooks::CacheAllocEvent::Alloc {
                addr: 0x8060_1000,
                cache_ptr: 0xc070_1000
            })
        );

        // Its matching free IS (and always was) hooked via the ordinary free path, pointer is the
        // *second* argument (a1), not a0, unlike every other free-shaped hook here.
        let mut free_regs = [0u32; 32];
        free_regs[11] = 0x8020_0000; // a1 = objp
        free_regs[crate::REG_RETURN_ADDR] = 0xc040_1000;
        assert_eq!(hooks.on_pc(0xc030_2000, &free_regs), None);
        assert_eq!(
            hooks.on_pc(0xc040_1000, &[0u32; 32]),
            Some(HookEvent::Free {
                addr: 0x8020_0000
            })
        );
    }

    #[test]
    fn missing_symbols_register_nothing() {
        let syms = HashMap::new();
        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);
        assert_eq!(hooks.pending_returns(), 0);
        // No panics, no hooks fire on arbitrary PCs.
        assert_eq!(hooks.on_pc(0x1234, &[0u32; 32]), None);
    }

    #[test]
    fn end_to_end_page_allocator_via_system_map() {
        let mut syms = HashMap::new();
        syms.insert("__get_free_pages".to_string(), 0xc060_0000u32);
        syms.insert("free_pages".to_string(), 0xc060_1000u32);
        syms.insert("get_zeroed_page".to_string(), 0xc060_2000u32);

        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        // __get_free_pages(gfp_mask, order): order is a1, pointer known only at return.
        let mut entry_regs = [0u32; 32];
        entry_regs[11] = 2; // order
        entry_regs[crate::hooks::REG_RETURN_ADDR] = 0xc070_0000;
        assert_eq!(hooks.on_page_pc(syms["__get_free_pages"], &entry_regs), None);
        assert_eq!(hooks.page_pending_returns(), 1);

        let mut ret_regs = [0u32; 32];
        ret_regs[crate::hooks::REG_RETURN_VALUE] = 0xc010_0000;
        assert_eq!(
            hooks.on_page_pc(0xc070_0000, &ret_regs),
            Some(crate::hooks::PageHookEvent::Alloc {
                addr: 0xc010_0000,
                order: 2
            })
        );

        // get_zeroed_page(gfp_mask): no order argument at all, always order 0.
        let mut gzp_entry = [0u32; 32];
        gzp_entry[crate::hooks::REG_RETURN_ADDR] = 0xc070_1000;
        assert_eq!(hooks.on_page_pc(syms["get_zeroed_page"], &gzp_entry), None);
        let mut gzp_ret = [0u32; 32];
        gzp_ret[crate::hooks::REG_RETURN_VALUE] = 0xc010_2000;
        assert_eq!(
            hooks.on_page_pc(0xc070_1000, &gzp_ret),
            Some(crate::hooks::PageHookEvent::Alloc {
                addr: 0xc010_2000,
                order: 0
            })
        );

        // free_pages(addr, order): fires immediately at entry, no return-address stash.
        let mut free_regs = [0u32; 32];
        free_regs[10] = 0xc010_0000; // addr
        free_regs[11] = 2; // order
        assert_eq!(
            hooks.on_page_pc(syms["free_pages"], &free_regs),
            Some(crate::hooks::PageHookEvent::Free {
                addr: 0xc010_0000,
                order: 2
            })
        );

        // None of this ever surfaced through the unrelated kmalloc-shaped `on_pc`/`ksize_hit`.
        assert_eq!(hooks.on_pc(syms["__get_free_pages"], &entry_regs), None);
        assert_eq!(hooks.pending_returns(), 0);
    }

    #[test]
    fn alloc_pages_and_free_pages_struct_page_family_is_never_registered() {
        let mut syms = HashMap::new();
        syms.insert("alloc_pages".to_string(), 0xc080_0000u32);
        syms.insert("__alloc_pages".to_string(), 0xc080_1000u32);
        syms.insert("__free_pages".to_string(), 0xc080_2000u32);

        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        // Hitting any of these entries is simply a no-op, not a crash or a spuriously-emitted
        // event -- confirmed via both query methods since these could in principle have been
        // mis-registered into either family.
        let regs = [0u32; 32];
        for &pc in syms.values() {
            assert_eq!(hooks.on_page_pc(pc, &regs), None);
            assert_eq!(hooks.on_pc(pc, &regs), None);
        }
        assert_eq!(hooks.page_pending_returns(), 0);
    }

    #[test]
    fn ksize_and_dunder_ksize_are_registered_and_fire_at_entry() {
        let mut syms = HashMap::new();
        syms.insert("ksize".to_string(), 0xc050_0000u32);
        syms.insert("__ksize".to_string(), 0xc050_1000u32);

        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        let mut regs = [0u32; 32];
        regs[10] = 0x8010_0000; // a0 = ptr being ksize()'d

        // ksize hits are surfaced via the independent `ksize_hit` query (see `hooks.rs`'s
        // `HookEvent` doc comment for why this is kept out of `on_pc`'s `HookEvent`).
        assert_eq!(hooks.ksize_hit(0xc050_0000, &regs), Some(0x8010_0000));
        assert_eq!(hooks.ksize_hit(0xc050_1000, &regs), Some(0x8010_0000));
        assert_eq!(hooks.on_pc(0xc050_0000, &regs), None);
        // Neither ksize hook stashes anything awaiting a return.
        assert_eq!(hooks.pending_returns(), 0);
    }

    /// End-to-end **positive control** for the `kmem_cache_alloc` coverage gap this module closes:
    /// drives the full `System.map` -> `PcHooks` -> `Sanitizer` pipeline (mirroring fs-cli's
    /// `run_case`, minus VA/PA translation, which is orthogonal to what this proves) for
    /// `kmem_cache_alloc` -> `kmem_cache_free` -> `kmem_cache_free` (a double-free bug).
    ///
    /// Before this feature, `kmem_cache_alloc` was never hooked at all (`AllocSizeUnavailable`),
    /// so the sanitizer never learned this address was live. That made a genuine double-free of a
    /// `kmem_cache_alloc`-sourced object **indistinguishable from ordinary correct usage**: the
    /// very first (legitimate) `kmem_cache_free` already reported `SanError::InvalidFree`, purely
    /// because the matching alloc was invisible — exactly the noise this test's "before" half
    /// reproduces directly against `Sanitizer` (no coverage gap to simulate; it's simply never
    /// calling `alloc`, which is the truth of today's behavior). After this feature, the first
    /// free is silently accepted (the object really was live) and only the second, genuinely
    /// buggy free is flagged — the bug the current (pre-this-change) code missed.
    #[test]
    fn kmem_cache_alloc_hook_turns_a_noisy_invalid_free_into_a_real_double_free_signal() {
        use crate::Sanitizer;

        let mut syms = HashMap::new();
        syms.insert("kmem_cache_alloc".to_string(), 0xc090_0000u32);
        syms.insert("kmem_cache_free".to_string(), 0xc090_1000u32);

        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        let cachep: u32 = 0xc0a0_0000;
        let obj_addr: u32 = 0x8005_0000;
        let ret_pc: u32 = 0xc091_0000;
        let free_ret_pc: u32 = 0xc091_1000;

        // --- "Before": kmem_cache_alloc invisible (today's behavior without this feature) ---
        // No alloc() is ever recorded for `obj_addr` (that's the literal pre-existing gap this
        // feature closes) -- so even a single, entirely legitimate free already reports
        // InvalidFree, indistinguishable from a real double-free.
        {
            let mut mmu = fs_mmu::Mmu::new(0x8000_0000, 0x0010_0000);
            let mut san = Sanitizer::new(crate::DEFAULT_REDZONE);
            assert_eq!(
                san.free(&mut mmu, obj_addr).unwrap_err(),
                crate::SanError::InvalidFree { addr: obj_addr },
                "today: kmem_cache_alloc is invisible, so even the FIRST, legitimate free \
                 already looks like an error -- a real double-free would look identical"
            );
        }

        // --- "After": this feature's full pipeline ---
        let mut mmu = fs_mmu::Mmu::new(0x8000_0000, 0x0010_0000);
        let mut san = Sanitizer::new(crate::DEFAULT_REDZONE);

        // kmem_cache_alloc(cachep, flags) entry: a0 = cachep, ra = ret_pc.
        let mut entry = [0u32; 32];
        entry[10] = cachep;
        entry[crate::hooks::REG_RETURN_ADDR] = ret_pc;
        assert_eq!(hooks.on_cache_alloc_pc(syms["kmem_cache_alloc"], &entry), None);

        // Return: a0 = the allocated pointer.
        let mut ret = [0u32; 32];
        ret[crate::hooks::REG_RETURN_VALUE] = obj_addr;
        let ev = hooks.on_cache_alloc_pc(ret_pc, &ret);
        let crate::hooks::CacheAllocEvent::Alloc {
            addr,
            cache_ptr: got_cachep,
        } = ev.expect("cache alloc event must fire at the stashed return address");
        assert_eq!(addr, obj_addr);
        assert_eq!(got_cachep, cachep);

        // fs-cli's real integration reads `cachep->object_size` out of guest memory here; this
        // test isn't about that read (covered elsewhere), so it supplies the size directly, the
        // same way a successful guest-memory read would.
        san.alloc_exact_no_uaf_guard(&mut mmu, addr, 96).unwrap();
        assert!(san.is_live(addr));

        // kmem_cache_free(cachep, objp): a1 = objp (NOT a0 -- see KNOWN_SYMBOLS' own note on this).
        let mut free1 = [0u32; 32];
        free1[11] = addr;
        free1[crate::hooks::REG_RETURN_ADDR] = free_ret_pc;
        assert_eq!(hooks.on_pc(syms["kmem_cache_free"], &free1), None);
        let HookEvent::Free { addr: freed_addr } = hooks
            .on_pc(free_ret_pc, &[0u32; 32])
            .expect("free event must fire at the stashed return address")
        else {
            panic!("expected a Free event");
        };
        assert_eq!(freed_addr, addr);

        // The first, legitimate free succeeds silently now -- the bug this feature fixes.
        san.free(&mut mmu, freed_addr).unwrap();
        assert!(!san.is_live(freed_addr));

        // A second free of the same pointer (the actual bug under test) is now the ONE thing
        // that reports InvalidFree -- a genuine double-free signal, not noise.
        assert_eq!(
            san.free(&mut mmu, freed_addr).unwrap_err(),
            crate::SanError::InvalidFree { addr: freed_addr },
            "after: the sanitizer now distinguishes a real double-free from ordinary usage"
        );
    }
}
