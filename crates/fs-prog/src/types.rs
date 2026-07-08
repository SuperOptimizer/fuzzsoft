//! The syzlang-lite type vocabulary: the smallest set of shapes needed to describe real
//! syscall arguments with types (resources, lengths, pointers, structs, flags) while staying
//! plain Rust data — no DSL, no compiler. See `docs/syzlang.md` §1.

use crate::resource::ResourceKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    In,
    Out,
    InOut,
}

/// How many bytes a `Buffer` payload should have.
#[derive(Clone, Copy, Debug)]
pub enum LenSpec {
    Fixed(u16),
    Range(u16, u16),
}

#[derive(Debug)]
pub struct Field {
    pub name: &'static str,
    pub ty: &'static ArgType,
}

/// The syzlang-lite type vocabulary. Deliberately flat — no arrays-of-structs, no unions,
/// no bitfields inside ints. Everything a syscall arg can be is one of these.
#[derive(Debug)]
pub enum ArgType {
    /// A single fixed value (e.g. an F_SETFL command number).
    Const(u32),
    /// An arbitrary-ish scalar; generator biases toward {0,1,2,-1,boundary,small,random}.
    Int { bits: u8, signed: bool },
    /// An enum (`bitmask:false`, pick one of `vals`) or bitmask (`bitmask:true`, OR a random
    /// subset of `vals`) — replaces raw random ints on flag-shaped args.
    Flags { vals: &'static [u32], bitmask: bool },
    /// A resource-typed argument: consumed from a live producer of a compatible kind in the
    /// same program, or a seed literal from `ResourceDef::seeds`.
    Res(ResourceKind),
    /// Byte-length of the (serialized) sibling arg at index `of` in the same call. Resolved
    /// to a literal at *generation* time so it can later be independently mutated.
    Len { of: u8 },
    /// A pointer into the scratch region. `nullable` lets the generator emit a literal NULL
    /// instead of allocating scratch (e.g. sendto's optional sockaddr).
    Ptr {
        dir: Dir,
        inner: &'static ArgType,
        nullable: bool,
    },
    /// Raw bytes (mutated buffer contents), sized per `LenSpec`.
    Buffer { len: LenSpec },
    /// A packed struct, fields laid out in C/RV32-ILP32 order (natural alignment, capped at
    /// 4 bytes except explicitly 8-byte-aligned 64-bit fields e.g. timespec64).
    Struct(&'static [Field]),
    /// Pick one of a small literal string pool (paths, memfd names); NUL-terminated when
    /// serialized.
    StringConst(&'static [&'static str]),
}

/// What new resource(s), if any, a call produces.
#[derive(Clone, Copy, Debug)]
pub enum Produces {
    None,
    /// Return value (guest a0 right after the ecall) is a new resource of `kind`.
    Ret(ResourceKind),
    /// The `arg_idx`-th arg (a `Ptr{Out,...}`) is an out-array; after the call, `count`
    /// consecutive u32 words at that scratch address are each a new resource of `kind`.
    OutArray {
        arg_idx: u8,
        kind: ResourceKind,
        count: u8,
    },
}

impl Produces {
    /// Number of resource "slots" this call produces (0, 1, or `count`).
    pub fn slot_count(&self) -> u8 {
        match self {
            Produces::None => 0,
            Produces::Ret(_) => 1,
            Produces::OutArray { count, .. } => *count,
        }
    }

    /// The kind produced at `slot`, if any.
    pub fn kind_at(&self, slot: u8) -> Option<ResourceKind> {
        match self {
            Produces::None => None,
            Produces::Ret(k) => (slot == 0).then_some(*k),
            Produces::OutArray { kind, count, .. } => (slot < *count).then_some(*kind),
        }
    }
}

/// One syscall description (analogous to a syzkaller `.txt` entry / `prog.Syscall`).
#[derive(Debug)]
pub struct SyscallDesc {
    pub name: &'static str, // may include a "$variant" suffix, e.g. "fcntl64$dupfd"
    pub nr: u32,            // REAL rv32 nr from generated unistd_32.h — never randomly guessed
    pub args: &'static [ArgType],
    pub produces: Produces,
}
