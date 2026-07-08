//! `fs-prog` — the fuzzsoft syzlang-lite typed syscall-program model.
//!
//! A self-contained, pure-data library implementing the design in `docs/syzlang.md`: a small
//! typed vocabulary for syscall arguments (`ArgType`), a resource model with fd/sock subtyping
//! (`ResourceKind`), a starter table of real rv32 syscall descriptions (`SYSCALLS`), a program
//! IR that threads resources between calls (`Prog`/`TypedCall`/`ArgValue`/`ResRef`), a
//! deterministic generator/mutator (`generate`/`mutate`), and a lowering pass that compiles a
//! typed program down to the concrete wire form fuzzsoft's guest agent understands (`lower`,
//! `to_wire`).
//!
//! See `DESIGN.md` in this crate for the exact guest-agent wire protocol.

pub mod genr;
pub mod lower;
pub mod mutate;
pub mod prog;
pub mod resource;
pub mod rng;
pub mod syscalls;
pub mod types;

pub use genr::generate;
pub use lower::{
    CALL_WORDS, ConcreteCall, DEFAULT_SCRATCH_CAP, FIXUP_WORDS, Fixup, FixupSrc, Lowered,
    MAX_FIXUPS, ScratchWriter, WIRE_WORDS, lower, lower_with_cap, to_wire,
};
pub use mutate::mutate;
pub use prog::{ArgValue, MAX_CALLS, Prog, ResRef, TypedCall};
pub use resource::{FD, RESOURCES, ResourceDef, ResourceKind, SOCK, VMA, kind_compat, seeds_for};
pub use rng::Rng;
pub use syscalls::SYSCALLS;
pub use types::{ArgType, Dir, Field, LenSpec, Produces, SyscallDesc};
