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

use crate::hooks::{AllocHook, FreeHook, PcHooks};

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
    /// Alloc-shaped in principle, but the size is not among the function's arguments at all —
    /// see the `kmem_cache_alloc*` entries below. We still list the symbol (so callers can see
    /// it was considered) but never register a hook for it.
    AllocSizeUnavailable,
    /// Free-shaped: pointer argument lives in this register index.
    FreePtrReg(usize),
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
    // NOT an argument at all; it's an intrinsic property of `cachep` (its fixed object size,
    // `cachep->size`), which isn't available at this PC-hook layer since we only read registers,
    // not dereference guest structures. LIMITATION: to hook this precisely you would need to
    // read `cachep->object_size` out of guest memory at entry (offset is
    // kernel-version-specific) and stash *that* as the pending size instead of a register value.
    // We deliberately do not implement that guest-memory read here (it would break the "only
    // needs a PC and a register file" ISA-agnostic property `hooks.rs` is built on) — so this
    // symbol is intentionally never registered as an alloc hook. It is still listed here as
    // documentation of the limitation and so a caller scanning `KNOWN_SYMBOLS` sees it was
    // considered, not forgotten.
    ("kmem_cache_alloc", Convention::AllocSizeUnavailable),
    // `void *kmem_cache_alloc_noprof(struct kmem_cache *cachep, gfp_t flags)` -> same limitation
    // as `kmem_cache_alloc` (allocation-profiling build).
    ("kmem_cache_alloc_noprof", Convention::AllocSizeUnavailable),
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
];

/// Register PC hooks for every kernel slab-allocator symbol in `KNOWN_SYMBOLS` that is present
/// in `syms` (as produced by [`parse_system_map`]). Symbol names and even calling conventions
/// drift across kernel versions/configs, so this tries each known name independently rather than
/// requiring a fixed set — whichever a given kernel build actually has get hooked.
///
/// `kmem_cache_alloc`/`kmem_cache_alloc_noprof` are never registered even if present, since their
/// size is not available from the register file at entry — see the doc comment on
/// [`Convention::AllocSizeUnavailable`]. Their frees (`kmem_cache_free`) are still hooked
/// normally: `Sanitizer::free` only needs the pointer, not the original size.
pub fn register_kernel_allocator_hooks(hooks: &mut PcHooks, syms: &HashMap<String, u32>) {
    for (name, convention) in KNOWN_SYMBOLS {
        let Some(&entry_pc) = syms.get(*name) else {
            continue;
        };
        match *convention {
            Convention::AllocSizeReg(size_reg) => {
                hooks.hook_alloc(AllocHook { entry_pc, size_reg });
            }
            Convention::AllocSizeUnavailable => {
                // Deliberately not hooked; see KNOWN_SYMBOLS doc comment.
            }
            Convention::FreePtrReg(ptr_reg) => {
                hooks.hook_free(FreeHook { entry_pc, ptr_reg });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::HookEvent;

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

        // Hit kfree's entry: a0 = pointer being freed, fires immediately (no return-wait).
        let mut free_regs = [0u32; 32];
        free_regs[10] = 0x8010_0000;
        assert_eq!(
            hooks.on_pc(kfree_pc, &free_regs),
            Some(HookEvent::Free {
                addr: 0x8010_0000
            })
        );
    }

    #[test]
    fn kmem_cache_alloc_is_never_registered() {
        let mut syms = HashMap::new();
        syms.insert("kmem_cache_alloc".to_string(), 0xc030_0000u32);
        syms.insert("kmem_cache_alloc_noprof".to_string(), 0xc030_1000u32);
        syms.insert("kmem_cache_free".to_string(), 0xc030_2000u32);

        let mut hooks = PcHooks::new();
        register_kernel_allocator_hooks(&mut hooks, &syms);

        // Hitting the (unhooked) kmem_cache_alloc entry PC is simply a no-op, not a crash or a
        // spuriously-sized alloc event.
        let regs = [0u32; 32];
        assert_eq!(hooks.on_pc(0xc030_0000, &regs), None);
        assert_eq!(hooks.on_pc(0xc030_1000, &regs), None);

        // But its matching free IS hooked, and the pointer is the *second* argument (a1), not
        // a0, unlike every other free-shaped hook here.
        let mut free_regs = [0u32; 32];
        free_regs[11] = 0x8020_0000; // a1 = objp
        assert_eq!(
            hooks.on_pc(0xc030_2000, &free_regs),
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
}
