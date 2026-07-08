//! Lowering: `Prog -> Lowered`. The one-shot pass run immediately before injecting a case —
//! never during mutation, so mutation stays cheap and purely on the typed tree. Produces the
//! concrete wire form the guest agent understands: up to `MAX_CALLS` `(nr, args[6])` tuples, a
//! scratch byte image for pointee data, and a resource-fixup table recording which arg slots
//! must be patched at runtime with an earlier call's result. See `docs/syzlang.md` §3-4 and
//! this crate's `DESIGN.md`.

use crate::prog::{ArgValue, MAX_CALLS, Prog, ResRef};
use crate::types::{ArgType, Field, Produces};

/// Bump allocator for the guest scratch region. Deterministically truncates (never
/// panics/errors) if a write would exceed `cap` — same "clamp, don't crash" policy as the rest
/// of fuzzsoft's generation pipeline.
#[derive(Debug)]
pub struct ScratchWriter {
    pub bytes: Vec<u8>,
    pub cursor: u32,
    pub cap: u32,
}

fn round_up(x: u32, align: u32) -> u32 {
    let align = align.max(1);
    x.div_ceil(align) * align
}

impl ScratchWriter {
    pub fn new(cap: u32) -> Self {
        ScratchWriter {
            bytes: Vec::new(),
            cursor: 0,
            cap,
        }
    }

    /// Bump-allocate `data`, aligned to `align`; returns the byte offset. If the aligned start
    /// is already past `cap`, nothing is written and `cap` is returned (a caller adding this to
    /// `scratch_base_va` gets an address one-past-the-end — never a crash). If `data` partially
    /// overflows `cap`, it is truncated in place (deterministic, not an error).
    pub fn write(&mut self, data: &[u8], align: u32) -> u32 {
        let off = round_up(self.cursor, align);
        if off >= self.cap {
            return self.cap;
        }
        if (self.bytes.len() as u32) < off {
            self.bytes.resize(off as usize, 0);
        }
        let room = (self.cap - off) as usize;
        let n = data.len().min(room);
        self.bytes.extend_from_slice(&data[..n]);
        self.cursor = off + n as u32;
        off
    }
}

fn int_bytes(bits: u8) -> u32 {
    (bits.max(8) as u32) / 8
}

fn int_align(bits: u8) -> u32 {
    match bits {
        64 => 8,
        32 => 4,
        16 => 2,
        _ => 1,
    }
}

/// `(size, align)` in bytes for one value as it would be serialized into scratch — used both
/// to size a `Ptr`'s pointee (for `Len{of}` resolution) and to lay out `Struct` fields.
pub fn value_size_align(ty: &ArgType, val: &ArgValue) -> (u32, u32) {
    match (ty, val) {
        (ArgType::Const(_), _) => (4, 4),
        (ArgType::Int { bits, .. }, _) => (int_bytes(*bits), int_align(*bits)),
        (ArgType::Flags { .. }, _) => (4, 4),
        (ArgType::Res(_), _) => (4, 4),
        (ArgType::Len { .. }, _) => (4, 4),
        (ArgType::Ptr { .. }, _) => (4, 4), // a pointer value itself, if ever nested
        (ArgType::Buffer { .. }, ArgValue::Bytes(b)) => (b.len() as u32, 1),
        (ArgType::StringConst(_), ArgValue::Bytes(b)) => (b.len() as u32, 1),
        (ArgType::Struct(fields), ArgValue::Struct(vals)) => {
            let (_, total, align) = struct_layout(fields, vals);
            (total, align)
        }
        _ => (0, 1),
    }
}

/// Byte size a `Ptr`'s pointee will occupy once serialized — what `Len{of}` args measure.
pub fn ptr_size_of(ty: &ArgType, val: &ArgValue) -> u32 {
    value_size_align(ty, val).0
}

/// Per-field byte offsets, the struct's total (padded) size, and its own alignment — natural
/// alignment, capped at 4 bytes except explicitly 8-byte-aligned 64-bit fields (RV32-ILP32).
pub fn struct_layout(fields: &[Field], vals: &[ArgValue]) -> (Vec<u32>, u32, u32) {
    let mut off = 0u32;
    let mut offsets = Vec::with_capacity(fields.len());
    let mut max_align = 1u32;
    for (f, v) in fields.iter().zip(vals) {
        let (size, align) = value_size_align(f.ty, v);
        let align = align.max(1);
        max_align = max_align.max(align);
        off = round_up(off, align);
        offsets.push(off);
        off += size;
    }
    let total = round_up(off, max_align);
    (offsets, total, max_align)
}

/// Serialize one value's bytes standalone (i.e. as it would appear once placed at an aligned
/// scratch offset) — recurses into `Struct` fields, writing them at their own laid-out offsets.
///
/// Takes `w`/`scratch_base_va` because a `Struct` field may itself be a `Ptr` (e.g. `msghdr`'s
/// `msg_iov`/`msg_name`/`msg_control`): such a nested pointee is bump-allocated into `w` *before*
/// the enclosing struct's own bytes are written (so it lands at a lower scratch offset than its
/// parent), and the 4-byte pointer value embedded in the parent's buffer is
/// `scratch_base_va + that offset` — exactly the same rule `lower_arg` applies to a top-level
/// `Ptr` arg, just recursively.
fn build_bytes(w: &mut ScratchWriter, scratch_base_va: u32, ty: &ArgType, val: &ArgValue) -> Vec<u8> {
    match (ty, val) {
        (ArgType::Const(_), ArgValue::Imm(v)) => (*v as u32).to_le_bytes().to_vec(),
        (ArgType::Int { bits, .. }, ArgValue::Imm(v)) => {
            let n = int_bytes(*bits) as usize;
            v.to_le_bytes()[..n.min(8)].to_vec()
        }
        (ArgType::Flags { .. }, ArgValue::Imm(v)) => (*v as u32).to_le_bytes().to_vec(),
        (ArgType::Res(_), ArgValue::Res(ResRef::Seed(s))) => {
            (*s as i32 as u32).to_le_bytes().to_vec()
        }
        (ArgType::Res(_), ArgValue::Res(ResRef::Produced { .. })) => 0u32.to_le_bytes().to_vec(),
        (ArgType::Buffer { .. }, ArgValue::Bytes(b)) => b.clone(),
        (ArgType::StringConst(_), ArgValue::Bytes(b)) => b.clone(),
        (ArgType::Ptr { nullable, .. }, ArgValue::Imm(0)) if *nullable => {
            0u32.to_le_bytes().to_vec()
        }
        (ArgType::Ptr { inner, .. }, ArgValue::Ptr(pointee)) => {
            let off = serialize_into(w, scratch_base_va, inner, pointee);
            scratch_base_va.wrapping_add(off).to_le_bytes().to_vec()
        }
        (ArgType::Struct(fields), ArgValue::Struct(vals)) => {
            let (offsets, total, _) = struct_layout(fields, vals);
            let mut buf = vec![0u8; total as usize];
            for ((f, v), off) in fields.iter().zip(vals).zip(offsets) {
                let fb = build_bytes(w, scratch_base_va, f.ty, v);
                let off = off as usize;
                let n = fb.len().min(buf.len().saturating_sub(off));
                buf[off..off + n].copy_from_slice(&fb[..n]);
            }
            buf
        }
        _ => Vec::new(),
    }
}

fn serialize_into(w: &mut ScratchWriter, scratch_base_va: u32, ty: &ArgType, val: &ArgValue) -> u32 {
    let (_, align) = value_size_align(ty, val);
    let bytes = build_bytes(w, scratch_base_va, ty, val);
    w.write(&bytes, align)
}

/// Where a masked arg's *real* value comes from, resolved at runtime by the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixupSrc {
    Reg(u16), // results[call_idx] — that call's a0 return value
    Mem(u32), // *(u32*)(scratch_base + byte_offset) — kernel-written out-value
}

/// One resource-fixup: at runtime, before executing `dst_call`, overwrite its `dst_arg`-th
/// register with the value produced by `source_call_index`'s `source_slot`-th resource slot
/// (`src` is that lookup already resolved to a concrete `Reg`/`Mem` source, ready for wire
/// encoding — see `to_wire`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixup {
    pub dst_call: u8,
    pub dst_arg: u8,
    pub source_call_index: u16,
    pub source_slot: u8,
    pub src: FixupSrc,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConcreteCall {
    pub nr: u32,
    pub args: [u32; 6], // args[j] is a placeholder (0) if a Fixup targets it
}

#[derive(Clone, Debug, Default)]
pub struct Lowered {
    pub calls: Vec<ConcreteCall>,
    pub fixups: Vec<Fixup>,
    pub scratch: Vec<u8>, // byte image to write into the guest scratch region
}

/// Default scratch region size (32 KiB, per `docs/syzlang.md` §3).
pub const DEFAULT_SCRATCH_CAP: u32 = 32 * 1024;

pub fn lower(p: &Prog, scratch_base_va: u32) -> Lowered {
    lower_with_cap(p, scratch_base_va, DEFAULT_SCRATCH_CAP)
}

pub fn lower_with_cap(p: &Prog, scratch_base_va: u32, scratch_cap: u32) -> Lowered {
    let mut w = ScratchWriter::new(scratch_cap);
    let mut fixups = Vec::new();
    let mut out_array_off: Vec<Option<u32>> = vec![None; p.calls.len()];
    let mut calls = Vec::with_capacity(p.calls.len());

    for (i, tc) in p.calls.iter().enumerate() {
        let mut args = [0u32; 6];
        for (j, (aty, av)) in tc.desc.args.iter().zip(&tc.args).enumerate() {
            args[j] = lower_arg(
                p,
                aty,
                av,
                &mut w,
                scratch_base_va,
                i,
                j,
                tc.desc.produces,
                &mut fixups,
                &mut out_array_off,
            );
        }
        calls.push(ConcreteCall {
            nr: tc.desc.nr,
            args,
        });
    }
    Lowered {
        calls,
        fixups,
        scratch: w.bytes,
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_arg(
    p: &Prog,
    aty: &ArgType,
    av: &ArgValue,
    w: &mut ScratchWriter,
    scratch_base_va: u32,
    call_i: usize,
    arg_j: usize,
    this_produces: Produces,
    fixups: &mut Vec<Fixup>,
    out_array_off: &mut [Option<u32>],
) -> u32 {
    match (aty, av) {
        (ArgType::Res(_), ArgValue::Res(ResRef::Seed(s))) => *s as i32 as u32,
        (ArgType::Res(_), ArgValue::Res(ResRef::Produced { call_idx, slot })) => {
            let src_call = *call_idx as usize;
            let src = match p.calls.get(src_call).map(|c| c.desc.produces) {
                Some(Produces::OutArray { .. }) => {
                    let base = out_array_off[src_call].unwrap_or(0);
                    FixupSrc::Mem(base + 4 * (*slot as u32))
                }
                _ => FixupSrc::Reg(*call_idx),
            };
            fixups.push(Fixup {
                dst_call: call_i as u8,
                dst_arg: arg_j as u8,
                source_call_index: *call_idx,
                source_slot: *slot,
                src,
            });
            0 // placeholder; the guest agent overwrites this per the fixup table
        }
        (ArgType::Ptr { nullable, .. }, ArgValue::Imm(0)) if *nullable => 0,
        (ArgType::Ptr { inner, .. }, ArgValue::Ptr(pointee)) => {
            let off = serialize_into(w, scratch_base_va, inner, pointee);
            if let Produces::OutArray { arg_idx, .. } = this_produces
                && arg_idx as usize == arg_j
            {
                out_array_off[call_i] = Some(off);
            }
            scratch_base_va.wrapping_add(off)
        }
        (_, ArgValue::Imm(v)) => *v as u32,
        _ => 0,
    }
}

/// Max fixups per program (generous: worst case is every one of `MAX_CALLS` calls' 6 args
/// resource-typed = 48, which never actually happens with the starter descriptions).
pub const MAX_FIXUPS: usize = 32;
/// Words per call slot in the wire buffer: `nr, a0..a5`.
pub const CALL_WORDS: usize = 7;
/// Words per fixup slot in the wire buffer: `dst_call, dst_arg, src_kind, src_val`.
pub const FIXUP_WORDS: usize = 4;
/// Total word count of the wire buffer `to_wire` produces — matches the guest agent's
/// `prog[]` array size exactly (`docs/syzlang.md` §4 / this crate's `DESIGN.md`).
pub const WIRE_WORDS: usize = 1 + MAX_CALLS * CALL_WORDS + 1 + MAX_FIXUPS * FIXUP_WORDS;

/// Encode a `Lowered` program into the flat `u32` wire buffer the guest agent reads directly:
/// `[n][nr,a0..a5]*MAX_CALLS[nfix][dst_call,dst_arg,src_kind,src_val]*MAX_FIXUPS`.
/// `src_kind` is `0` for `Reg(call_idx)` (value = call_idx) or `1` for `Mem(byte_offset)`
/// (value = byte offset into scratch). Calls/fixups beyond the fixed slot counts are dropped
/// (never happens in practice: `MAX_CALLS`/`MAX_FIXUPS` are sized generously).
pub fn to_wire(lowered: &Lowered) -> Vec<u32> {
    let mut buf = vec![0u32; WIRE_WORDS];
    let n = lowered.calls.len().min(MAX_CALLS);
    buf[0] = n as u32;
    for (i, c) in lowered.calls.iter().take(MAX_CALLS).enumerate() {
        let base = 1 + i * CALL_WORDS;
        buf[base] = c.nr;
        buf[base + 1..base + 1 + 6].copy_from_slice(&c.args);
    }
    let fixup_base = 1 + MAX_CALLS * CALL_WORDS;
    let nfix = lowered.fixups.len().min(MAX_FIXUPS);
    buf[fixup_base] = nfix as u32;
    for (k, f) in lowered.fixups.iter().take(MAX_FIXUPS).enumerate() {
        let base = fixup_base + 1 + k * FIXUP_WORDS;
        let (kind, val) = match f.src {
            FixupSrc::Reg(c) => (0u32, c as u32),
            FixupSrc::Mem(o) => (1u32, o),
        };
        buf[base] = f.dst_call as u32;
        buf[base + 1] = f.dst_arg as u32;
        buf[base + 2] = kind;
        buf[base + 3] = val;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_writer_aligns_and_bumps() {
        let mut w = ScratchWriter::new(64);
        let a = w.write(&[1, 2, 3], 4);
        assert_eq!(a, 0);
        let b = w.write(&[9], 4);
        assert_eq!(b, 4); // padded up to 4-byte alignment
        assert_eq!(&w.bytes[0..3], &[1, 2, 3]);
        assert_eq!(w.bytes[4], 9);
    }

    #[test]
    fn scratch_writer_truncates_deterministically_on_overflow() {
        let mut w = ScratchWriter::new(4);
        let off = w.write(&[1, 2, 3, 4, 5, 6], 1);
        assert_eq!(off, 0);
        assert_eq!(w.bytes.len(), 4); // truncated to cap
        let off2 = w.write(&[7, 8], 1);
        assert_eq!(off2, 4); // == cap: no room, offset clamped, nothing written
        assert_eq!(w.bytes.len(), 4);
    }
}
