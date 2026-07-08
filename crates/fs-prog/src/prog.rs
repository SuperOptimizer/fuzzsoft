//! The typed program IR: the tree the corpus stores and the mutator edits. Analogous to
//! syzkaller's `prog.Prog`/`prog.Arg`, but flat (one call = one flat arg list, no nested arg
//! trees beyond `Struct`/`Ptr` payloads). See `docs/syzlang.md` §2.

use crate::types::SyscallDesc;

/// Maximum calls per program (matches the guest agent's `MAX_CALLS`).
pub const MAX_CALLS: usize = 8;

/// A resource-typed arg value: either a seed literal, or "whatever call `call_idx` produced,
/// resource slot `slot`" (slot 0 for `Produces::Ret`, slot `j` for the `j`-th entry of a
/// `Produces::OutArray`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResRef {
    Seed(i64),
    Produced { call_idx: u16, slot: u8 },
}

/// The value bound to one `ArgType` slot in a `TypedCall`. Structurally mirrors `ArgType` one
/// level deep (a `Ptr` wraps its pointee's `ArgValue`; a `Struct` is a vec of per-field
/// `ArgValue`s in field order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgValue {
    Imm(u64),              // Const / Int / Flags / Len (resolved) / literal NULL for a Ptr slot
    Res(ResRef),           // Res(kind) slot
    Bytes(Vec<u8>),        // Buffer contents / chosen StringConst (NUL-terminated)
    Struct(Vec<ArgValue>), // one entry per Field, same order as the ArgType::Struct
    Ptr(Box<ArgValue>),    // Ptr's pointee (Bytes/Struct/Imm); Imm(0) at a Ptr slot means NULL
}

#[derive(Clone, Debug)]
pub struct TypedCall {
    pub desc: &'static SyscallDesc,
    pub args: Vec<ArgValue>, // args.len() == desc.args.len()
}

/// A whole fuzz input: a sequence of typed calls with resources threaded between them. This is
/// the corpus + mutation unit; `MAX_CALLS` is enforced on insert.
#[derive(Clone, Debug, Default)]
pub struct Prog {
    pub calls: Vec<TypedCall>,
}

impl Prog {
    pub fn new() -> Self {
        Prog { calls: Vec::new() }
    }

    /// True iff every `Res` arg's `Produced` reference points strictly backward (`call_idx <
    /// i`) at a call whose produced kind is compatible with what the consumer wants. This is
    /// the well-formedness invariant the generator/mutator must preserve (`docs/syzlang.md`
    /// §2 "Threading invariant").
    pub fn is_well_formed(&self) -> bool {
        for (i, call) in self.calls.iter().enumerate() {
            if call.args.len() != call.desc.args.len() {
                return false;
            }
            for (aty, av) in call.desc.args.iter().zip(&call.args) {
                let crate::types::ArgType::Res(want) = aty else {
                    continue;
                };
                let ArgValue::Res(rref) = av else {
                    return false;
                };
                if let ResRef::Produced { call_idx, slot } = rref {
                    let call_idx = *call_idx as usize;
                    if call_idx >= i {
                        return false; // forward or self reference
                    }
                    let Some(have) = self.calls[call_idx].desc.produces.kind_at(*slot) else {
                        return false; // producer doesn't actually make a resource at that slot
                    };
                    if !crate::resource::kind_compat(*want, have) {
                        return false;
                    }
                }
            }
        }
        true
    }
}
