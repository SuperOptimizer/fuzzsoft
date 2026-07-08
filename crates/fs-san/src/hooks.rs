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

    /// Drop all in-flight alloc/free calls awaiting a return. Call between snapshot-fuzzing cases
    /// so a call left mid-flight by one case's reset doesn't leak into the next.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
        self.pending_frees.clear();
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
}
