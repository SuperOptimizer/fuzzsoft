//! fs-san: the sanitizer policy layer over `fs-mmu`'s soft MMU.
//!
//! Turns the byte-granular permission plane described in `docs/architecture.md` §3 into a
//! practical memory-safety oracle (decision #19: "soft-MMU as primary sanitizer") *without any
//! guest instrumentation* — the guest binary is never recompiled, patched, or linked against a
//! sanitizer runtime. Two independent ways to learn the addresses/sizes involved are provided:
//!
//! - [`hypercall`] — a cooperative guest agent reports `malloc`/`free` via a reserved `ecall`.
//!   Cheap and precise, but requires a guest we can add a few lines of code to.
//! - [`hooks`] — a PC-hook framework that intercepts known allocator entry points by address and
//!   reads arguments/return values out of the register file per the calling convention. Works on
//!   binaries we cannot recompile, including closed-source ones (the Windows kernel-pool case).
//!
//! Both paths feed the same [`Sanitizer`] in [`alloc`], which is the only place that actually
//! touches `Mmu` permissions. See `DESIGN.md` for the full rationale and the KASAN/ASAN contrast.
//!
//! - [`linux`] — wires the PC-hook path to the RV32 Linux kernel target specifically: parses a
//!   `System.map` and registers hooks for whichever slab-allocator symbols the kernel build
//!   actually has, with no kernel-side instrumentation required.

mod alloc;
pub mod hooks;
pub mod hypercall;
pub mod linux;

pub use alloc::{DEFAULT_QUARANTINE_CAP, DEFAULT_REDZONE, SanError, Sanitizer};
pub use hooks::{
    AllocHook, FreeHook, HookEvent, KsizeHook, PcHooks, REG_RETURN_ADDR, REG_RETURN_VALUE,
};
pub use linux::{LinearMap, kmalloc_bucket, parse_system_map, register_kernel_allocator_hooks};
