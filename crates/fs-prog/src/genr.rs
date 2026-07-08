//! Deterministic PRNG-driven generation of well-formed `Prog`s, with a resource pool threading
//! producers (openat/socket/pipe2 -> fd/sock) into consumers (read/ioctl/close). See
//! `docs/syzlang.md` §2.

use crate::lower::ptr_size_of;
use crate::prog::{ArgValue, MAX_CALLS, Prog, ResRef, TypedCall};
use crate::resource::{ResourceKind, kind_compat, seeds_for};
use crate::rng::Rng;
use crate::syscalls::SYSCALLS;
use crate::types::{ArgType, Field, LenSpec, SyscallDesc};

/// One live resource in the program-under-construction: `desc.produces`' `slot`-th resource,
/// produced by call `call_idx`. `pub(crate)` so `mutate` can share this exact pool
/// representation instead of re-deriving it.
#[derive(Clone, Copy)]
pub(crate) struct PoolEntry {
    pub(crate) call_idx: u16,
    pub(crate) slot: u8,
    pub(crate) kind: ResourceKind,
}

pub(crate) fn build_pool(calls: &[TypedCall]) -> Vec<PoolEntry> {
    let mut pool = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        for slot in 0..c.desc.produces.slot_count() {
            if let Some(kind) = c.desc.produces.kind_at(slot) {
                pool.push(PoolEntry {
                    call_idx: i as u16,
                    slot,
                    kind,
                });
            }
        }
    }
    pool
}

/// Generate a fresh `Prog` of 1..=MAX_CALLS calls, each a random `SyscallDesc` from the starter
/// table with type-directed argument generation and resource threading against earlier calls.
pub fn generate(rng: &mut Rng) -> Prog {
    let n = 1 + rng.below(MAX_CALLS);
    let mut calls: Vec<TypedCall> = Vec::with_capacity(n);
    for _ in 0..n {
        let desc = pick_desc(rng);
        let args = generate_args(rng, desc, &calls);
        calls.push(TypedCall { desc, args });
    }
    Prog { calls }
}

pub fn pick_desc(rng: &mut Rng) -> &'static SyscallDesc {
    rng.pick(SYSCALLS)
}

/// Generate a full args vector for `desc`, given the calls already placed before it (used both
/// for resource threading and `Len{of}` resolution).
pub fn generate_args(
    rng: &mut Rng,
    desc: &'static SyscallDesc,
    existing: &[TypedCall],
) -> Vec<ArgValue> {
    let pool = build_pool(existing);
    let mut args: Vec<ArgValue> = Vec::with_capacity(desc.args.len());
    for aty in desc.args {
        let av = match aty {
            ArgType::Len { of } => {
                let of = *of as usize;
                let sz = match (desc.args.get(of), args.get(of)) {
                    (Some(oty), Some(oval)) => len_of_arg(oty, oval),
                    _ => 0,
                };
                ArgValue::Imm(sz as u64)
            }
            _ => gen_arg_value(rng, aty, &pool),
        };
        args.push(av);
    }
    args
}

/// The byte length `lower()` will actually give the value at `(aty, av)` — what a sibling
/// `Len{of}` arg should be generated as so lengths start out correct (they may later be
/// independently mutated to desync on purpose).
fn len_of_arg(aty: &ArgType, av: &ArgValue) -> u32 {
    match (aty, av) {
        (ArgType::Ptr { .. }, ArgValue::Imm(_)) => 0, // NULL pointer
        (ArgType::Ptr { inner, .. }, ArgValue::Ptr(pointee)) => ptr_size_of(inner, pointee),
        (ArgType::Buffer { .. }, ArgValue::Bytes(b)) => b.len() as u32,
        (ArgType::StringConst(_), ArgValue::Bytes(b)) => b.len() as u32,
        _ => 0,
    }
}

pub(crate) fn gen_arg_value(rng: &mut Rng, aty: &ArgType, pool: &[PoolEntry]) -> ArgValue {
    match aty {
        ArgType::Const(v) => ArgValue::Imm(*v as u64),
        ArgType::Int { bits, signed } => ArgValue::Imm(gen_int(rng, *bits, *signed)),
        ArgType::Flags { vals, bitmask } => ArgValue::Imm(gen_flags(rng, vals, *bitmask) as u64),
        ArgType::Res(kind) => ArgValue::Res(pick_res(rng, *kind, pool)),
        ArgType::Len { .. } => ArgValue::Imm(0), // resolved by generate_args; never reached directly
        ArgType::Ptr {
            inner, nullable, ..
        } => {
            if *nullable && rng.chance(20) {
                ArgValue::Imm(0)
            } else {
                ArgValue::Ptr(Box::new(gen_arg_value(rng, inner, pool)))
            }
        }
        ArgType::Buffer { len } => {
            let n = gen_len(rng, *len);
            let bytes = (0..n).map(|_| rng.next() as u8).collect();
            ArgValue::Bytes(bytes)
        }
        ArgType::Struct(fields) => ArgValue::Struct(gen_struct_fields(rng, fields, pool)),
        ArgType::StringConst(pool_strs) => {
            let s = rng.pick(pool_strs);
            let mut b = s.as_bytes().to_vec();
            b.push(0);
            ArgValue::Bytes(b)
        }
    }
}

fn gen_struct_fields(rng: &mut Rng, fields: &'static [Field], pool: &[PoolEntry]) -> Vec<ArgValue> {
    fields
        .iter()
        .map(|f| gen_arg_value(rng, f.ty, pool))
        .collect()
}

fn gen_len(rng: &mut Rng, spec: LenSpec) -> u32 {
    match spec {
        LenSpec::Fixed(n) => n as u32,
        LenSpec::Range(lo, hi) => {
            let (lo, hi) = (lo as u32, hi as u32);
            if hi <= lo {
                lo
            } else {
                lo + rng.below((hi - lo + 1) as usize) as u32
            }
        }
    }
}

fn mask_to_bits(v: u64, bits: u8) -> u64 {
    if bits >= 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    }
}

/// Biased scalar generation: {0, 1, 2, -1, boundary, small, full-random}, matching the flavor
/// of today's `gen_arg` pool, per-argument-typed now.
fn gen_int(rng: &mut Rng, bits: u8, signed: bool) -> u64 {
    let raw: u64 = match rng.below(7) {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => {
            if signed {
                (-1i64) as u64
            } else {
                u64::MAX
            }
        }
        4 => {
            // boundary value for this width
            if bits >= 64 {
                if signed { i64::MAX as u64 } else { u64::MAX }
            } else if signed {
                (1u64 << (bits - 1)) - 1 // i.e. INT_MAX for this width
            } else {
                (1u64 << bits) - 1 // UINT_MAX for this width
            }
        }
        5 => rng.next() as u64 % 256,
        _ => rng.next_u64(),
    };
    mask_to_bits(raw, bits)
}

fn gen_flags(rng: &mut Rng, vals: &[u32], bitmask: bool) -> u32 {
    if !bitmask {
        return *rng.pick(vals);
    }
    if rng.bool() {
        *rng.pick(vals)
    } else {
        vals.iter()
            .fold(0u32, |acc, &v| if rng.bool() { acc | v } else { acc })
    }
}

/// 70% pick an existing compatible producer earlier in the program-under-construction (if
/// any), else fall back to a seed literal.
pub(crate) fn pick_res(rng: &mut Rng, want: ResourceKind, pool: &[PoolEntry]) -> ResRef {
    if rng.chance(70) {
        let compatible: Vec<&PoolEntry> =
            pool.iter().filter(|e| kind_compat(want, e.kind)).collect();
        if !compatible.is_empty() {
            let e = **rng.pick(&compatible);
            return ResRef::Produced {
                call_idx: e.call_idx,
                slot: e.slot,
            };
        }
    }
    let seeds = seeds_for(want);
    if seeds.is_empty() {
        ResRef::Seed(-1)
    } else {
        ResRef::Seed(*rng.pick(seeds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_yields_well_formed_programs_within_call_limit() {
        for seed in 1..200u32 {
            let mut rng = Rng::new(seed);
            let p = generate(&mut rng);
            assert!(!p.calls.is_empty());
            assert!(p.calls.len() <= MAX_CALLS);
            assert!(
                p.is_well_formed(),
                "seed {seed} produced ill-formed program"
            );
        }
    }

    #[test]
    fn generated_args_match_desc_arity() {
        let mut rng = Rng::new(3);
        let p = generate(&mut rng);
        for c in &p.calls {
            assert_eq!(c.args.len(), c.desc.args.len());
        }
    }

    #[test]
    fn len_args_match_actual_serialized_sibling_size() {
        // Force read(fd,buf,len) shape checks across many seeds by scanning generated progs.
        let mut rng = Rng::new(99);
        let mut checked = 0;
        for _ in 0..500 {
            let p = generate(&mut rng);
            for c in &p.calls {
                for (j, aty) in c.desc.args.iter().enumerate() {
                    if let ArgType::Len { of } = aty {
                        let of = *of as usize;
                        let expect = len_of_arg(&c.desc.args[of], &c.args[of]);
                        if let ArgValue::Imm(v) = &c.args[j] {
                            assert_eq!(*v as u32, expect);
                            checked += 1;
                        } else {
                            panic!("Len arg wasn't Imm");
                        }
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn resource_pool_prefers_compatible_earlier_producers() {
        // Directly test pick_res determinism/behavior with a synthetic pool.
        use crate::resource::{FD, SOCK};
        let pool = vec![PoolEntry {
            call_idx: 0,
            slot: 0,
            kind: SOCK,
        }];
        let mut rng = Rng::new(5);
        let mut saw_produced = false;
        for _ in 0..200 {
            if let ResRef::Produced { call_idx, slot } = pick_res(&mut rng, FD, &pool) {
                assert_eq!(call_idx, 0);
                assert_eq!(slot, 0);
                saw_produced = true;
            }
        }
        assert!(
            saw_produced,
            "SOCK should satisfy a Res(FD) consumer at least once in 200 tries"
        );
    }
}
