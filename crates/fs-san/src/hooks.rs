//! PC-hook framework: learn guest allocator calls by watching the program counter, with **no**
//! guest cooperation and **no** recompilation. This is the path that works on uninstrumented and
//! closed-source binaries (the Windows kernel-pool case in `DESIGN.md`) where hypercalls
//! (`hypercall.rs`) are not an option because the guest was never built with a sanitizer agent.
//!
//! The idea, generalized from any "entry args in registers, return value in a fixed register"
//! calling convention (RISC-V: `a0..a7` = x10..x17, return address in `ra` = x1, return value in
//! `a0`; the same shape holds for x86-64 fastcall/Win64: args in registers, return address on the
//! stack, return value in `rax`):
//!
//! 1. Register the **entry PC** of a known allocator function plus which argument register holds
//!    the size (`alloc`) or pointer (`free`).
//! 2. Every time the interpreter's PC lands on a hooked entry address, [`PcHooks::on_pc`]
//!    inspects the register file:
//!    - **Free-shaped hook:** the pointer argument is already known at entry -> emit
//!      [`HookEvent::Free`] immediately (freeing typically only touches allocator metadata
//!      *before* the payload, not the payload itself, so poisoning at entry is safe).
//!    - **Alloc-shaped hook:** the *pointer* isn't known yet — the callee hasn't run — only the
//!      *size* is. Stash `(return_pc, size)` and wait.
//!    - **Return-address match:** once PC reaches a previously-stashed `return_pc`, the callee
//!      just returned; the ABI's return-value register now holds the allocated pointer. Pop the
//!      matching pending size and emit [`HookEvent::Alloc`].
//!
//! The pending-size stash is a small `Vec` per return address (LIFO), so recursive/re-entrant
//! calls through the *same* call site are handled correctly — the return that fires first pairs
//! with the size that was pushed most recently.
//!
//! **Framework vs. worked example:** this module is ISA-agnostic (it only needs "a PC" and "a
//! flat register file" — no dependency on `fs-riscv`). `DESIGN.md` gives the concrete RISC-V
//! `kmalloc`/`kfree` wiring as the worked example, and sketches the Windows
//! `ExAllocatePoolWithTag`/`ExFreePool` case as the return-value-capture path this same code
//! already implements, just with different register-index constants.
//!
//! **`kmem_cache_alloc` ([`CacheAllocHook`]):** one alloc-shaped kernel entry point does not fit
//! "the size is an argument register" at all — `kmem_cache_alloc(cachep, flags)`'s size is
//! `cachep->object_size`, a guest-*memory* field, not a register. Reading that would need an
//! `Mmu` reference this framework deliberately does not carry (see above). So `CacheAllocHook`
//! stashes the **cache pointer** at entry (a plain register value, exactly like [`AllocHook`]
//! stashes a size) and [`PcHooks::on_cache_alloc_pc`] hands the caller both the returned object
//! pointer and that cache pointer at return; the caller — which already owns an `Mmu` for every
//! other sanitizer call — reads `object_size` itself. This keeps the "PC + register file only"
//! property intact for `hooks.rs` while still closing the `kmem_cache_alloc` coverage gap
//! `linux.rs`/`DESIGN.md` previously documented as permanently unavailable.

use std::collections::HashMap;

/// Register index conventionally holding the return address at function entry.
/// RISC-V: `ra` = x1.
pub const REG_RETURN_ADDR: usize = 1;
/// Register index conventionally holding a function's return value once it's back.
/// RISC-V: `a0` = x10 (also the first argument register, per the standard calling convention).
pub const REG_RETURN_VALUE: usize = 10;

/// A monitored allocator-entry function: `fn alloc(..., size, ...) -> *mut u8`.
#[derive(Debug, Clone, Copy)]
pub struct AllocHook {
    /// Guest PC of the function's first instruction.
    pub entry_pc: u32,
    /// Register index holding the size argument at entry (e.g. `kmalloc(size_t size, ...)` ->
    /// the RISC-V `a0` register, index 10).
    pub size_reg: usize,
}

/// A monitored deallocator-entry function: `fn free(ptr, ...)`.
#[derive(Debug, Clone, Copy)]
pub struct FreeHook {
    /// Guest PC of the function's first instruction.
    pub entry_pc: u32,
    /// Register index holding the pointer argument at entry.
    pub ptr_reg: usize,
}

/// A monitored "report/re-open usable size" entry function: `fn ksize(ptr) -> size_t` (also
/// covers `__ksize`/`krealloc`'s in-place-grow path). Unlike [`FreeHook`], firing at entry is
/// always safe here — `ksize()` only *reads* allocator metadata to compute a size, it never
/// writes into the object's payload the way SLUB's `kfree()` writes its intrusive freelist
/// pointer — so there is no SLUB-internal-write race to guard against and no return-address
/// stash is needed; see [`PcHooks::ksize_hit`].
#[derive(Debug, Clone, Copy)]
pub struct KsizeHook {
    /// Guest PC of the function's first instruction.
    pub entry_pc: u32,
    /// Register index holding the pointer argument at entry.
    pub ptr_reg: usize,
}

/// A sanitizer-relevant event learned by watching the guest PC, ready to hand to
/// [`crate::Sanitizer::alloc`] / [`crate::Sanitizer::free`].
///
/// Deliberately does **not** carry a `ksize()`-shaped variant: `HookEvent` is already matched
/// exhaustively by existing callers (the fs-cli run loop), so adding a new required-to-handle
/// variant here would be a breaking change to every such match, not an additive one. The
/// `ksize()` query is exposed as its own independent method, [`PcHooks::ksize_hit`], precisely so
/// a caller can adopt it whenever it wires up [`crate::Sanitizer::reopen_slack`] without that
/// forcing a change everywhere `on_pc`'s return value is already matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    Alloc { addr: u32, size: u32 },
    Free { addr: u32 },
}

/// A monitored page-allocator entry function whose *pointer* is discovered at return
/// (`__get_free_pages`/`get_zeroed_page`-shaped): `fn alloc_pages(gfp_mask, order) -> VA`. Mirrors
/// [`AllocHook`]'s entry-then-return timing exactly, just with an *order* captured at entry
/// instead of a byte size — this module stays unit-agnostic (converting `order` to
/// `PAGE << order` bytes is [`crate::PageSanitizer`]'s job, not this framework's), same discipline
/// [`AllocHook`] already applies to its raw `size`.
#[derive(Debug, Clone, Copy)]
pub struct PageAllocHook {
    /// Guest PC of the function's first instruction.
    pub entry_pc: u32,
    /// Register index holding the order argument at entry, or `None` if the function has no
    /// order argument at all — it is always order 0 (e.g. `get_zeroed_page(gfp_mask)`, which has
    /// only the flags argument and calls `__get_free_pages(gfp_mask, 0)` internally).
    pub order_reg: Option<usize>,
}

/// A monitored page-allocator deallocator entry function (`free_pages`-shaped): `fn
/// free_pages(addr, order)`. Unlike [`FreeHook`] (which must delay to the return address because
/// SLUB's intrusive freelist pointer write happens *during* the call), the page allocator's own
/// free path does not write into the freed page's payload before this hook's entry PC fires —
/// poisoning immediately at entry is safe, so [`PcHooks::on_page_pc`] fires this with **no**
/// return-address stash, mirroring the free-at-entry design this module's own doc comment
/// describes as the general shape (the SLUB-specific exception is what forced [`FreeHook`] to
/// delay; the page allocator has no equivalent exception).
#[derive(Debug, Clone, Copy)]
pub struct PageFreeHook {
    /// Guest PC of the function's first instruction.
    pub entry_pc: u32,
    /// Register index holding the pointer argument at entry.
    pub addr_reg: usize,
    /// Register index holding the order argument at entry, or `None` if the function has no
    /// order argument at all (always order 0).
    pub order_reg: Option<usize>,
}

/// An event learned from watching the page-allocator PC entries, ready to hand to
/// [`crate::PageSanitizer::alloc_pages`] / [`crate::PageSanitizer::free_pages`].
///
/// Kept entirely separate from [`HookEvent`]/[`PcHooks::on_pc`] for the exact reason documented on
/// that enum: `HookEvent` is matched exhaustively by existing callers, so a new event family must
/// be its own independent query rather than a new variant — [`PcHooks::ksize_hit`] established this
/// pattern for `ksize()`, and [`PcHooks::on_page_pc`] follows it here for the page allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageHookEvent {
    Alloc { addr: u32, order: u32 },
    Free { addr: u32, order: u32 },
}

/// A monitored `kmem_cache_alloc`-shaped entry function: `fn kmem_cache_alloc(cachep, flags) ->
/// *mut u8`. Closes the gap `DESIGN.md`/`linux.rs` previously documented as
/// `Convention::AllocSizeUnavailable`: the requested size is not an argument at all, it's
/// `cachep->object_size` — a guest-memory field this register-file-only framework cannot read.
/// Rather than break that "only a PC and a register file" property (this module's whole
/// ISA-agnostic-design point), this hook only ever stashes the **cache pointer** here — a plain
/// register value, exactly like [`AllocHook`] stashes a size — and leaves the guest-memory read of
/// `object_size` to the caller (which already owns an `Mmu` reference for every other sanitizer
/// call, so it's a natural, minimal place for it; see `docs/emulator-sanitizers.md`).
#[derive(Debug, Clone, Copy)]
pub struct CacheAllocHook {
    /// Guest PC of the function's first instruction.
    pub entry_pc: u32,
    /// Register index holding the `struct kmem_cache *` argument at entry (e.g.
    /// `kmem_cache_alloc(struct kmem_cache *cachep, gfp_t flags)` -> `a0`, index 10).
    pub cache_reg: usize,
}

/// An event learned from watching [`CacheAllocHook`]-registered entries: the returned object
/// pointer plus the cache pointer captured at entry, so the caller can read `cachep->object_size`
/// itself and hand `(addr, object_size)` to [`crate::Sanitizer::alloc_with_slack`]. Kept as its own
/// independent query ([`PcHooks::on_cache_alloc_pc`]) for the same reason [`PageHookEvent`] is:
/// `HookEvent` is matched exhaustively elsewhere and must not gain a new required-to-handle
/// variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAllocEvent {
    Alloc { addr: u32, cache_ptr: u32 },
}

/// Registry of PC hooks plus the small amount of state needed to bridge an alloc call's entry
/// (where the size is known) to its return (where the pointer is known).
#[derive(Default)]
pub struct PcHooks {
    allocs: HashMap<u32, AllocHook>,
    frees: HashMap<u32, FreeHook>,
    /// `ksize()`-shaped hooks, queried independently via [`PcHooks::ksize_hit`] rather than
    /// through `on_pc`'s [`HookEvent`] (see that enum's doc comment for why).
    ksizes: HashMap<u32, KsizeHook>,
    /// Sizes awaiting a return, keyed by the call's return address. A `Vec` (used as a stack)
    /// per address handles recursion/re-entrancy through the same call site.
    pending: HashMap<u32, Vec<u32>>,
    /// Free pointers awaiting a return, keyed by return address. Frees are delayed to the return
    /// (not emitted at entry) because SLUB writes its intrusive freelist pointer *into* the freed
    /// object during the call — poisoning at entry would fault SLUB's own legitimate write.
    pending_frees: HashMap<u32, Vec<u32>>,
    /// Page-allocator hooks, queried independently via [`PcHooks::on_page_pc`] rather than through
    /// `on_pc`'s [`HookEvent`] (see [`PageHookEvent`]'s doc comment for why).
    page_allocs: HashMap<u32, PageAllocHook>,
    page_frees: HashMap<u32, PageFreeHook>,
    /// Orders awaiting a return, keyed by the call's return address — the page-allocator analogue
    /// of `pending`.
    page_pending: HashMap<u32, Vec<u32>>,
    /// `kmem_cache_alloc`-shaped hooks, queried independently via [`PcHooks::on_cache_alloc_pc`].
    cache_allocs: HashMap<u32, CacheAllocHook>,
    /// Cache pointers awaiting a return, keyed by return address — the `kmem_cache_alloc`
    /// analogue of `pending`, except the stashed payload is a cache pointer, not a size (see
    /// [`CacheAllocHook`]'s doc comment for why).
    cache_pending: HashMap<u32, Vec<u32>>,
}

impl PcHooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Watch `entry_pc` as an allocator function entry.
    pub fn hook_alloc(&mut self, hook: AllocHook) {
        self.allocs.insert(hook.entry_pc, hook);
    }

    /// Watch `entry_pc` as a deallocator function entry.
    pub fn hook_free(&mut self, hook: FreeHook) {
        self.frees.insert(hook.entry_pc, hook);
    }

    /// Watch `entry_pc` as a `ksize()`-shaped ("report/re-open usable size") function entry.
    pub fn hook_ksize(&mut self, hook: KsizeHook) {
        self.ksizes.insert(hook.entry_pc, hook);
    }

    /// Watch `entry_pc` as a page-allocator function entry (`__get_free_pages`-shaped).
    pub fn hook_page_alloc(&mut self, hook: PageAllocHook) {
        self.page_allocs.insert(hook.entry_pc, hook);
    }

    /// Watch `entry_pc` as a page-deallocator function entry (`free_pages`-shaped).
    pub fn hook_page_free(&mut self, hook: PageFreeHook) {
        self.page_frees.insert(hook.entry_pc, hook);
    }

    /// Watch `entry_pc` as a `kmem_cache_alloc`-shaped function entry.
    pub fn hook_cache_alloc(&mut self, hook: CacheAllocHook) {
        self.cache_allocs.insert(hook.entry_pc, hook);
    }

    /// Independent query: is `pc` a registered `ksize()`-shaped entry, and if so, what pointer is
    /// being queried? Kept separate from [`PcHooks::on_pc`]/[`HookEvent`] deliberately (see
    /// `HookEvent`'s doc comment) — callers that want `ksize()` support call this alongside
    /// `on_pc` and feed a `Some(addr)` to [`crate::Sanitizer::reopen_slack`]. Never mutates any
    /// pending-return state: firing is always safe immediately, no stash needed (see
    /// [`KsizeHook`]'s doc comment).
    pub fn ksize_hit(&self, pc: u32, regs: &[u32; 32]) -> Option<u32> {
        self.ksizes.get(&pc).map(|hook| regs[hook.ptr_reg])
    }

    /// Number of alloc-call returns currently awaited (i.e. calls whose entry we saw but whose
    /// return we have not yet observed). Exposed mainly for tests/diagnostics.
    pub fn pending_returns(&self) -> usize {
        self.pending.values().map(|v| v.len()).sum()
    }

    /// Feed the current guest program counter and register file. Call this once per retired
    /// instruction (or at minimum, once for every distinct PC the interpreter executes) — a hook
    /// firing is purely a function of "did PC land on an address we're watching", so the caller
    /// controls the granularity/cost by how often it calls this.
    ///
    /// Returns at most one event per call. If a PC is simultaneously an alloc entry, a free
    /// entry, and a pending return (pathological but not impossible with hand-picked addresses in
    /// a test), alloc-entry takes priority, then free-entry, then return-match — entries are
    /// checked before returns so a hook that is *both* an entry and someone else's return address
    /// still registers the entry.
    ///
    /// This does not check `ksize()`-shaped hooks — call [`PcHooks::ksize_hit`] separately for
    /// those (see `HookEvent`'s doc comment for why they're independent).
    pub fn on_pc(&mut self, pc: u32, regs: &[u32; 32]) -> Option<HookEvent> {
        if let Some(hook) = self.allocs.get(&pc) {
            let size = regs[hook.size_reg];
            let ret_pc = regs[REG_RETURN_ADDR];
            self.pending.entry(ret_pc).or_default().push(size);
            return None; // The pointer isn't known until the call returns.
        }
        if let Some(hook) = self.frees.get(&pc) {
            // Delay the free to the return: SLUB writes its freelist pointer into the object
            // *during* the call, so we must not poison the payload at entry.
            let addr = regs[hook.ptr_reg];
            let ret_pc = regs[REG_RETURN_ADDR];
            self.pending_frees.entry(ret_pc).or_default().push(addr);
            return None;
        }
        if let Some(sizes) = self.pending.get_mut(&pc)
            && let Some(size) = sizes.pop()
        {
            if sizes.is_empty() {
                self.pending.remove(&pc);
            }
            let addr = regs[REG_RETURN_VALUE];
            return Some(HookEvent::Alloc { addr, size });
        }
        if let Some(ptrs) = self.pending_frees.get_mut(&pc)
            && let Some(addr) = ptrs.pop()
        {
            if ptrs.is_empty() {
                self.pending_frees.remove(&pc);
            }
            return Some(HookEvent::Free { addr });
        }
        None
    }

    /// Independent page-allocator query, mirroring [`PcHooks::on_pc`]'s entry/return dance but for
    /// the [`PageAllocHook`]/[`PageFreeHook`] family — kept separate from `on_pc` so its
    /// [`PageHookEvent`] never needs to join `HookEvent`'s exhaustively-matched set (see
    /// `PageHookEvent`'s doc comment). Call this alongside `on_pc` (and `ksize_hit`) once per PC, in
    /// addition to it, not instead of it.
    ///
    /// Same entry-before-return priority discipline as `on_pc`: an alloc entry is checked first,
    /// then a free entry (which fires immediately — no return-address stash needed, see
    /// [`PageFreeHook`]'s doc comment), then a pending-return match.
    pub fn on_page_pc(&mut self, pc: u32, regs: &[u32; 32]) -> Option<PageHookEvent> {
        if let Some(hook) = self.page_allocs.get(&pc) {
            let order = hook.order_reg.map_or(0, |r| regs[r]);
            let ret_pc = regs[REG_RETURN_ADDR];
            self.page_pending.entry(ret_pc).or_default().push(order);
            return None; // The pointer isn't known until the call returns.
        }
        if let Some(hook) = self.page_frees.get(&pc) {
            let addr = regs[hook.addr_reg];
            let order = hook.order_reg.map_or(0, |r| regs[r]);
            return Some(PageHookEvent::Free { addr, order });
        }
        if let Some(orders) = self.page_pending.get_mut(&pc)
            && let Some(order) = orders.pop()
        {
            if orders.is_empty() {
                self.page_pending.remove(&pc);
            }
            let addr = regs[REG_RETURN_VALUE];
            return Some(PageHookEvent::Alloc { addr, order });
        }
        None
    }

    /// Number of page-alloc-call returns currently awaited. Exposed mainly for tests/diagnostics,
    /// mirroring [`PcHooks::pending_returns`].
    pub fn page_pending_returns(&self) -> usize {
        self.page_pending.values().map(|v| v.len()).sum()
    }

    /// Independent `kmem_cache_alloc` query, mirroring [`PcHooks::on_pc`]'s alloc entry/return
    /// dance but stashing a **cache pointer** at entry instead of a size (see [`CacheAllocHook`]'s
    /// doc comment for why) — kept separate from `on_pc` so [`CacheAllocEvent`] never needs to
    /// join `HookEvent`'s exhaustively-matched set, the same reasoning [`PageHookEvent`] and
    /// [`PcHooks::ksize_hit`] already establish. Call this alongside `on_pc`/`ksize_hit`/
    /// `on_page_pc`, once per PC, in addition to them, not instead of them.
    pub fn on_cache_alloc_pc(&mut self, pc: u32, regs: &[u32; 32]) -> Option<CacheAllocEvent> {
        if let Some(hook) = self.cache_allocs.get(&pc) {
            let cache_ptr = regs[hook.cache_reg];
            let ret_pc = regs[REG_RETURN_ADDR];
            self.cache_pending.entry(ret_pc).or_default().push(cache_ptr);
            return None; // The object pointer isn't known until the call returns.
        }
        if let Some(ptrs) = self.cache_pending.get_mut(&pc)
            && let Some(cache_ptr) = ptrs.pop()
        {
            if ptrs.is_empty() {
                self.cache_pending.remove(&pc);
            }
            let addr = regs[REG_RETURN_VALUE];
            return Some(CacheAllocEvent::Alloc { addr, cache_ptr });
        }
        None
    }

    /// Number of `kmem_cache_alloc`-call returns currently awaited. Exposed mainly for
    /// tests/diagnostics, mirroring [`PcHooks::pending_returns`]/[`PcHooks::page_pending_returns`].
    pub fn cache_alloc_pending_returns(&self) -> usize {
        self.cache_pending.values().map(|v| v.len()).sum()
    }

    /// Drop all in-flight alloc/free calls awaiting a return. Call between snapshot-fuzzing cases
    /// so a call left mid-flight by one case's reset doesn't leak into the next.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
        self.pending_frees.clear();
        self.page_pending.clear();
        self.cache_pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regs_with(mut set: impl FnMut(&mut [u32; 32])) -> [u32; 32] {
        let mut r = [0u32; 32];
        set(&mut r);
        r
    }

    #[test]
    fn alloc_hook_waits_for_return_to_learn_pointer() {
        let mut hooks = PcHooks::new();
        hooks.hook_alloc(AllocHook {
            entry_pc: 0x1000,
            size_reg: 10, // a0 = size at entry
        });

        // Entry: a0 = requested size (64), ra = return address (0x2000).
        let entry_regs = regs_with(|r| {
            r[10] = 64;
            r[REG_RETURN_ADDR] = 0x2000;
        });
        assert_eq!(hooks.on_pc(0x1000, &entry_regs), None);
        assert_eq!(hooks.pending_returns(), 1);

        // Some unrelated PCs execute in between; nothing fires.
        assert_eq!(hooks.on_pc(0x1004, &entry_regs), None);

        // Return: PC lands on the stashed return address, a0 now holds the pointer.
        let ret_regs = regs_with(|r| {
            r[REG_RETURN_VALUE] = 0x8000_1000;
        });
        assert_eq!(
            hooks.on_pc(0x2000, &ret_regs),
            Some(HookEvent::Alloc {
                addr: 0x8000_1000,
                size: 64
            })
        );
        assert_eq!(hooks.pending_returns(), 0);
    }

    #[test]
    fn free_hook_fires_at_return_not_entry() {
        let mut hooks = PcHooks::new();
        hooks.hook_free(FreeHook {
            entry_pc: 0x3000,
            ptr_reg: 10,
        });
        // Entry: a0 = ptr, ra = return address. No event yet — SLUB writes its freelist pointer
        // into the object during the call, so we must not poison the payload before the call runs.
        let entry = regs_with(|r| {
            r[10] = 0x8000_2000;
            r[REG_RETURN_ADDR] = 0x3100;
        });
        assert_eq!(hooks.on_pc(0x3000, &entry), None);
        // Return: emit Free with the pointer captured at entry.
        assert_eq!(
            hooks.on_pc(0x3100, &regs_with(|_| {})),
            Some(HookEvent::Free { addr: 0x8000_2000 })
        );
    }

    #[test]
    fn recursive_calls_through_same_site_pair_lifo() {
        let mut hooks = PcHooks::new();
        hooks.hook_alloc(AllocHook {
            entry_pc: 0x1000,
            size_reg: 10,
        });

        let outer_entry = regs_with(|r| {
            r[10] = 16;
            r[REG_RETURN_ADDR] = 0x2000;
        });
        hooks.on_pc(0x1000, &outer_entry);

        let inner_entry = regs_with(|r| {
            r[10] = 32;
            r[REG_RETURN_ADDR] = 0x2000;
        });
        hooks.on_pc(0x1000, &inner_entry);
        assert_eq!(hooks.pending_returns(), 2);

        // First return to 0x2000 pairs with the most recently pushed (inner, size=32) call.
        let ret1 = regs_with(|r| r[REG_RETURN_VALUE] = 0x9000_0000);
        assert_eq!(
            hooks.on_pc(0x2000, &ret1),
            Some(HookEvent::Alloc {
                addr: 0x9000_0000,
                size: 32
            })
        );
        let ret2 = regs_with(|r| r[REG_RETURN_VALUE] = 0x9000_1000);
        assert_eq!(
            hooks.on_pc(0x2000, &ret2),
            Some(HookEvent::Alloc {
                addr: 0x9000_1000,
                size: 16
            })
        );
        assert_eq!(hooks.pending_returns(), 0);
    }

    #[test]
    fn unrelated_pc_is_a_no_op() {
        let mut hooks = PcHooks::new();
        hooks.hook_alloc(AllocHook {
            entry_pc: 0x1000,
            size_reg: 10,
        });
        let regs = [0u32; 32];
        assert_eq!(hooks.on_pc(0x4242, &regs), None);
    }

    #[test]
    fn ksize_hook_fires_at_entry_via_its_own_independent_query() {
        let mut hooks = PcHooks::new();
        hooks.hook_ksize(KsizeHook {
            entry_pc: 0x5000,
            ptr_reg: 10, // a0 = ptr
        });
        // Unlike FreeHook, ksize is safe to fire immediately at entry — no return-address stash.
        let entry = regs_with(|r| {
            r[10] = 0x8000_3000;
            r[REG_RETURN_ADDR] = 0x5100;
        });
        assert_eq!(hooks.ksize_hit(0x5000, &entry), Some(0x8000_3000));
        // `on_pc` itself never surfaces ksize hits (kept out of `HookEvent`, see its doc comment)
        // and a ksize entry doesn't stash anything awaiting a return either.
        assert_eq!(hooks.on_pc(0x5000, &entry), None);
        assert_eq!(hooks.pending_returns(), 0);
        // An unrelated PC (including the never-stashed "return address") is a plain no-op.
        assert_eq!(hooks.ksize_hit(0x5100, &regs_with(|_| {})), None);
        assert_eq!(hooks.on_pc(0x5100, &regs_with(|_| {})), None);
    }

    // -- Page-allocator hooks (PageAllocHook/PageFreeHook/on_page_pc) --

    #[test]
    fn page_alloc_hook_waits_for_return_to_learn_pointer() {
        let mut hooks = PcHooks::new();
        hooks.hook_page_alloc(PageAllocHook {
            entry_pc: 0x6000,
            order_reg: Some(11), // a1 = order at entry, like __get_free_pages(gfp_mask, order)
        });

        let entry_regs = regs_with(|r| {
            r[11] = 2; // order 2 -> 4 pages
            r[REG_RETURN_ADDR] = 0x6100;
        });
        assert_eq!(hooks.on_page_pc(0x6000, &entry_regs), None);
        assert_eq!(hooks.page_pending_returns(), 1);
        // `on_pc`/`ksize_hit` must not see this event family at all.
        assert_eq!(hooks.on_pc(0x6000, &entry_regs), None);

        let ret_regs = regs_with(|r| r[REG_RETURN_VALUE] = 0x8100_0000);
        assert_eq!(
            hooks.on_page_pc(0x6100, &ret_regs),
            Some(PageHookEvent::Alloc {
                addr: 0x8100_0000,
                order: 2
            })
        );
        assert_eq!(hooks.page_pending_returns(), 0);
    }

    #[test]
    fn page_alloc_hook_with_no_order_register_is_always_order_zero() {
        let mut hooks = PcHooks::new();
        hooks.hook_page_alloc(PageAllocHook {
            entry_pc: 0x6200,
            order_reg: None, // get_zeroed_page(gfp_mask): no order argument, always order 0
        });
        let entry_regs = regs_with(|r| r[REG_RETURN_ADDR] = 0x6300);
        assert_eq!(hooks.on_page_pc(0x6200, &entry_regs), None);
        let ret_regs = regs_with(|r| r[REG_RETURN_VALUE] = 0x8110_0000);
        assert_eq!(
            hooks.on_page_pc(0x6300, &ret_regs),
            Some(PageHookEvent::Alloc {
                addr: 0x8110_0000,
                order: 0
            })
        );
    }

    #[test]
    fn page_free_hook_fires_at_entry_not_return() {
        let mut hooks = PcHooks::new();
        hooks.hook_page_free(PageFreeHook {
            entry_pc: 0x7000,
            addr_reg: 10,       // a0 = addr
            order_reg: Some(11), // a1 = order
        });
        // Unlike FreeHook, this fires immediately: no SLUB-style write-into-payload race for the
        // page allocator's own free path.
        let entry_regs = regs_with(|r| {
            r[10] = 0x8120_0000;
            r[11] = 1;
            r[REG_RETURN_ADDR] = 0x7100;
        });
        assert_eq!(
            hooks.on_page_pc(0x7000, &entry_regs),
            Some(PageHookEvent::Free {
                addr: 0x8120_0000,
                order: 1
            })
        );
        assert_eq!(hooks.page_pending_returns(), 0);
        // The return address never got a stash, so hitting it is a plain no-op.
        assert_eq!(hooks.on_page_pc(0x7100, &regs_with(|_| {})), None);
    }

    #[test]
    fn page_hooks_do_not_leak_into_or_from_the_kmalloc_hook_family() {
        let mut hooks = PcHooks::new();
        hooks.hook_alloc(AllocHook {
            entry_pc: 0x1000,
            size_reg: 10,
        });
        hooks.hook_page_alloc(PageAllocHook {
            entry_pc: 0x6000,
            order_reg: Some(11),
        });
        let regs = regs_with(|r| {
            r[10] = 64;
            r[11] = 3;
            r[REG_RETURN_ADDR] = 0x2000;
        });
        // Hitting the kmalloc-shaped entry only stashes a `pending` (byte-size) entry, never a
        // `page_pending` (order) entry, and vice versa.
        assert_eq!(hooks.on_pc(0x1000, &regs), None);
        assert_eq!(hooks.pending_returns(), 1);
        assert_eq!(hooks.page_pending_returns(), 0);

        assert_eq!(hooks.on_page_pc(0x6000, &regs), None);
        assert_eq!(hooks.pending_returns(), 1);
        assert_eq!(hooks.page_pending_returns(), 1);
    }

    #[test]
    fn clear_pending_drops_in_flight_page_allocs_too() {
        let mut hooks = PcHooks::new();
        hooks.hook_page_alloc(PageAllocHook {
            entry_pc: 0x6000,
            order_reg: Some(11),
        });
        let regs = regs_with(|r| {
            r[11] = 0;
            r[REG_RETURN_ADDR] = 0x6100;
        });
        hooks.on_page_pc(0x6000, &regs);
        assert_eq!(hooks.page_pending_returns(), 1);
        hooks.clear_pending();
        assert_eq!(hooks.page_pending_returns(), 0);
        // The stashed return no longer fires anything.
        assert_eq!(
            hooks.on_page_pc(0x6100, &regs_with(|r| r[REG_RETURN_VALUE] = 0x8000_0000)),
            None
        );
    }

    // -- kmem_cache_alloc hook (CacheAllocHook/CacheAllocEvent/on_cache_alloc_pc) --

    #[test]
    fn cache_alloc_hook_waits_for_return_to_learn_pointer_and_carries_the_cache_ptr() {
        let mut hooks = PcHooks::new();
        hooks.hook_cache_alloc(CacheAllocHook {
            entry_pc: 0x8000,
            cache_reg: 10, // a0 = cachep at entry
        });

        let entry_regs = regs_with(|r| {
            r[10] = 0xc040_0000; // cachep
            r[REG_RETURN_ADDR] = 0x8100;
        });
        assert_eq!(hooks.on_cache_alloc_pc(0x8000, &entry_regs), None);
        assert_eq!(hooks.cache_alloc_pending_returns(), 1);
        // Not visible through any other query.
        assert_eq!(hooks.on_pc(0x8000, &entry_regs), None);
        assert_eq!(hooks.on_page_pc(0x8000, &entry_regs), None);

        let ret_regs = regs_with(|r| r[REG_RETURN_VALUE] = 0x8030_0000);
        assert_eq!(
            hooks.on_cache_alloc_pc(0x8100, &ret_regs),
            Some(CacheAllocEvent::Alloc {
                addr: 0x8030_0000,
                cache_ptr: 0xc040_0000
            })
        );
        assert_eq!(hooks.cache_alloc_pending_returns(), 0);
    }

    #[test]
    fn cache_alloc_recursive_calls_through_same_site_pair_lifo() {
        let mut hooks = PcHooks::new();
        hooks.hook_cache_alloc(CacheAllocHook {
            entry_pc: 0x8000,
            cache_reg: 10,
        });

        let outer = regs_with(|r| {
            r[10] = 0xc040_0000;
            r[REG_RETURN_ADDR] = 0x8100;
        });
        hooks.on_cache_alloc_pc(0x8000, &outer);
        let inner = regs_with(|r| {
            r[10] = 0xc050_0000;
            r[REG_RETURN_ADDR] = 0x8100;
        });
        hooks.on_cache_alloc_pc(0x8000, &inner);
        assert_eq!(hooks.cache_alloc_pending_returns(), 2);

        let ret1 = regs_with(|r| r[REG_RETURN_VALUE] = 0x9000_0000);
        assert_eq!(
            hooks.on_cache_alloc_pc(0x8100, &ret1),
            Some(CacheAllocEvent::Alloc {
                addr: 0x9000_0000,
                cache_ptr: 0xc050_0000
            })
        );
        let ret2 = regs_with(|r| r[REG_RETURN_VALUE] = 0x9000_1000);
        assert_eq!(
            hooks.on_cache_alloc_pc(0x8100, &ret2),
            Some(CacheAllocEvent::Alloc {
                addr: 0x9000_1000,
                cache_ptr: 0xc040_0000
            })
        );
        assert_eq!(hooks.cache_alloc_pending_returns(), 0);
    }

    #[test]
    fn clear_pending_drops_in_flight_cache_allocs_too() {
        let mut hooks = PcHooks::new();
        hooks.hook_cache_alloc(CacheAllocHook {
            entry_pc: 0x8000,
            cache_reg: 10,
        });
        let regs = regs_with(|r| {
            r[10] = 0xc040_0000;
            r[REG_RETURN_ADDR] = 0x8100;
        });
        hooks.on_cache_alloc_pc(0x8000, &regs);
        assert_eq!(hooks.cache_alloc_pending_returns(), 1);
        hooks.clear_pending();
        assert_eq!(hooks.cache_alloc_pending_returns(), 0);
        assert_eq!(
            hooks.on_cache_alloc_pc(0x8100, &regs_with(|r| r[REG_RETURN_VALUE] = 0x9000_0000)),
            None
        );
    }

    #[test]
    fn cache_alloc_hooks_do_not_leak_into_kmalloc_or_page_families() {
        let mut hooks = PcHooks::new();
        hooks.hook_alloc(AllocHook {
            entry_pc: 0x1000,
            size_reg: 10,
        });
        hooks.hook_page_alloc(PageAllocHook {
            entry_pc: 0x6000,
            order_reg: Some(11),
        });
        hooks.hook_cache_alloc(CacheAllocHook {
            entry_pc: 0x8000,
            cache_reg: 10,
        });
        let regs = regs_with(|r| {
            r[10] = 64;
            r[11] = 2;
            r[REG_RETURN_ADDR] = 0x2000;
        });
        assert_eq!(hooks.on_cache_alloc_pc(0x1000, &regs), None);
        assert_eq!(hooks.cache_alloc_pending_returns(), 0);
        assert_eq!(hooks.on_cache_alloc_pc(0x6000, &regs), None);
        assert_eq!(hooks.cache_alloc_pending_returns(), 0);

        assert_eq!(hooks.on_cache_alloc_pc(0x8000, &regs), None);
        assert_eq!(hooks.cache_alloc_pending_returns(), 1);
        assert_eq!(hooks.pending_returns(), 0);
        assert_eq!(hooks.page_pending_returns(), 0);
    }
}
