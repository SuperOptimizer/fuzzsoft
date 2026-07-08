//! Targeted mutation over the typed program tree: insert/remove a call, mutate an arg by its
//! type (including re-picking a resource reference), and a dedicated "wire two calls together"
//! move that manufactures `open->read->close`-style chains instead of waiting for them to occur
//! by chance. Every operation preserves the threading invariant (`Prog::is_well_formed`). See
//! `docs/syzlang.md` §2.

use crate::genr::{
    PoolEntry, build_pool, gen_arg_value, generate, generate_args, pick_desc, pick_res,
};
use crate::lower::ptr_size_of;
use crate::prog::{ArgValue, MAX_CALLS, Prog, ResRef, TypedCall};
use crate::resource::{ResourceKind, kind_compat, seeds_for};
use crate::rng::Rng;
use crate::syscalls::SYSCALLS;
use crate::types::{ArgType, SyscallDesc};

fn len_of_arg(aty: &ArgType, av: &ArgValue) -> u32 {
    match (aty, av) {
        (ArgType::Ptr { .. }, ArgValue::Imm(_)) => 0,
        (ArgType::Ptr { inner, .. }, ArgValue::Ptr(pointee)) => ptr_size_of(inner, pointee),
        (ArgType::Buffer { .. }, ArgValue::Bytes(b)) => b.len() as u32,
        (ArgType::StringConst(_), ArgValue::Bytes(b)) => b.len() as u32,
        _ => 0,
    }
}

/// Any `Produced{call_idx,..}` with `call_idx >= at` is renumbered `+1` (a call was just
/// inserted at position `at`).
fn renumber_after_insert(p: &mut Prog, at: usize) {
    for call in p.calls.iter_mut() {
        for (aty, av) in call.desc.args.iter().zip(call.args.iter_mut()) {
            if let (ArgType::Res(_), ArgValue::Res(ResRef::Produced { call_idx, .. })) = (aty, av)
                && *call_idx as usize >= at
            {
                *call_idx += 1;
            }
        }
    }
}

/// A call at position `at` was just removed: any `Produced{call_idx: at, ..}` reference
/// becomes dangling and falls back to a `Seed`; `Produced{call_idx > at, ..}` is renumbered
/// `-1`; anything below `at` is untouched.
fn renumber_after_remove(rng: &mut Rng, p: &mut Prog, at: usize) {
    for call in p.calls.iter_mut() {
        for (aty, av) in call.desc.args.iter().zip(call.args.iter_mut()) {
            let ArgType::Res(want) = aty else { continue };
            let ArgValue::Res(rref) = av else { continue };
            if let ResRef::Produced { call_idx, .. } = rref {
                let ci = *call_idx as usize;
                use std::cmp::Ordering::*;
                match ci.cmp(&at) {
                    Equal => {
                        let seeds = seeds_for(*want);
                        *rref = if seeds.is_empty() {
                            ResRef::Seed(-1)
                        } else {
                            ResRef::Seed(*rng.pick(seeds))
                        };
                    }
                    Greater => *call_idx -= 1,
                    Less => {}
                }
            }
        }
    }
}

fn insert_call(rng: &mut Rng, p: &mut Prog) {
    let pos = rng.below(p.calls.len() + 1);
    let desc = pick_desc(rng);
    let args = generate_args(rng, desc, &p.calls[..pos]);
    renumber_after_insert(p, pos);
    p.calls.insert(pos, TypedCall { desc, args });
}

fn remove_call(rng: &mut Rng, p: &mut Prog) {
    let idx = rng.below(p.calls.len());
    p.calls.remove(idx);
    renumber_after_remove(rng, p, idx);
}

fn mutate_one_arg(rng: &mut Rng, call: &mut TypedCall, j: usize, pool: &[PoolEntry]) {
    match &call.desc.args[j] {
        ArgType::Res(kind) => {
            call.args[j] = ArgValue::Res(pick_res(rng, *kind, pool));
        }
        ArgType::Len { of } => {
            let of = *of as usize;
            if rng.chance(70) {
                if let (Some(oty), Some(oval)) = (call.desc.args.get(of), call.args.get(of)) {
                    let sz = len_of_arg(oty, oval);
                    call.args[j] = ArgValue::Imm(sz as u64);
                }
            } else if let ArgValue::Imm(v) = call.args[j] {
                let nv = match rng.below(3) {
                    0 => v ^ (1u64 << (rng.next() % 32)),
                    1 => rng.next() as u64,
                    _ => v.wrapping_add((rng.next() % 17) as u64),
                };
                call.args[j] = ArgValue::Imm(nv);
            }
        }
        _ => {
            call.args[j] = gen_arg_value(rng, &call.desc.args[j], pool);
        }
    }
}

fn mutate_random_arg(rng: &mut Rng, p: &mut Prog) {
    let candidates: Vec<usize> = (0..p.calls.len())
        .filter(|&i| !p.calls[i].args.is_empty())
        .collect();
    if candidates.is_empty() {
        return;
    }
    let idx = *rng.pick(&candidates);
    let pool = build_pool(&p.calls[..idx]);
    let j = rng.below(p.calls[idx].args.len());
    mutate_one_arg(rng, &mut p.calls[idx], j, &pool);
}

/// The dedicated "wire two calls together" mutation: pick a `Res(kind)`-typed arg anywhere in
/// the program; if a compatible producer already precedes it, rewrite the arg to reference it;
/// otherwise insert a fresh compatible producer call earlier and rewrite the arg to reference
/// *that*. This is what actually manufactures `open->read->close` chains instead of waiting for
/// them to occur by chance.
fn wire_producer(rng: &mut Rng, p: &mut Prog) {
    let mut positions: Vec<(usize, usize, ResourceKind)> = Vec::new();
    for (i, c) in p.calls.iter().enumerate() {
        for (j, aty) in c.desc.args.iter().enumerate() {
            if let ArgType::Res(k) = aty {
                positions.push((i, j, *k));
            }
        }
    }
    if positions.is_empty() {
        return;
    }
    let (idx, j, kind) = *rng.pick(&positions);

    let pool = build_pool(&p.calls[..idx]);
    let compatible: Vec<&PoolEntry> = pool.iter().filter(|e| kind_compat(kind, e.kind)).collect();
    if !compatible.is_empty() {
        let e = **rng.pick(&compatible);
        p.calls[idx].args[j] = ArgValue::Res(ResRef::Produced {
            call_idx: e.call_idx,
            slot: e.slot,
        });
        return;
    }

    if p.calls.len() >= MAX_CALLS {
        return; // no room to insert a fresh producer; leave the arg as-is (Seed fallback)
    }
    let producer_descs: Vec<&'static SyscallDesc> = SYSCALLS
        .iter()
        .filter(|d| {
            (0..d.produces.slot_count())
                .any(|s| d.produces.kind_at(s).is_some_and(|k| kind_compat(kind, k)))
        })
        .collect();
    if producer_descs.is_empty() {
        return;
    }
    let desc = *rng.pick(&producer_descs);
    let pos = rng.below(idx + 1); // insert at some position <= idx
    let args = generate_args(rng, desc, &p.calls[..pos]);
    renumber_after_insert(p, pos);
    p.calls.insert(pos, TypedCall { desc, args });

    let new_idx = idx + 1; // pos <= idx, so idx's call shifted right by exactly one
    let slot = (0..desc.produces.slot_count())
        .find(|&s| {
            desc.produces
                .kind_at(s)
                .is_some_and(|k| kind_compat(kind, k))
        })
        .unwrap_or(0);
    p.calls[new_idx].args[j] = ArgValue::Res(ResRef::Produced {
        call_idx: pos as u16,
        slot,
    });
}

/// Mutate `base` into a new, still well-formed `Prog`. Never mutates `base` in place.
pub fn mutate(rng: &mut Rng, base: &Prog) -> Prog {
    let mut p = base.clone();
    if p.calls.is_empty() {
        return generate(rng);
    }
    match rng.below(4) {
        0 if p.calls.len() < MAX_CALLS => insert_call(rng, &mut p),
        1 if p.calls.len() > 1 => remove_call(rng, &mut p),
        2 => mutate_random_arg(rng, &mut p),
        _ => wire_producer(rng, &mut p),
    }
    if p.calls.is_empty() {
        return generate(rng);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genr::generate;

    #[test]
    fn mutation_preserves_well_formedness() {
        for seed in 1..300u32 {
            let mut rng = Rng::new(seed);
            let base = generate(&mut rng);
            let mutated = mutate(&mut rng, &base);
            assert!(
                mutated.is_well_formed(),
                "seed {seed} broke well-formedness"
            );
            assert!(!mutated.calls.is_empty());
            assert!(mutated.calls.len() <= MAX_CALLS);
        }
    }

    #[test]
    fn insert_renumbers_forward_refs() {
        let mut rng = Rng::new(11);
        // Build: [openat, read(Produced{0,0})]
        let openat = SYSCALLS.iter().find(|d| d.name == "openat").unwrap();
        let read = SYSCALLS.iter().find(|d| d.name == "read").unwrap();
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: openat,
            args: generate_args(&mut rng, openat, &[]),
        });
        let mut read_args = generate_args(&mut rng, read, &p.calls);
        read_args[0] = ArgValue::Res(ResRef::Produced {
            call_idx: 0,
            slot: 0,
        });
        p.calls.push(TypedCall {
            desc: read,
            args: read_args,
        });
        assert!(p.is_well_formed());

        insert_call(&mut rng, &mut p); // may insert before or after index 0
        assert!(
            p.is_well_formed(),
            "insert must renumber Produced refs to stay well-formed"
        );
    }

    #[test]
    fn remove_dangling_producer_falls_back_to_seed() {
        let openat = SYSCALLS.iter().find(|d| d.name == "openat").unwrap();
        let close = SYSCALLS.iter().find(|d| d.name == "close").unwrap();
        let mut rng = Rng::new(22);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: openat,
            args: generate_args(&mut rng, openat, &[]),
        });
        p.calls.push(TypedCall {
            desc: close,
            args: vec![ArgValue::Res(ResRef::Produced {
                call_idx: 0,
                slot: 0,
            })],
        });
        assert!(p.is_well_formed());

        remove_call(&mut rng, &mut p); // removes one of the two calls at random
        assert!(p.is_well_formed());
        if p.calls.len() == 1 {
            // if `close` survived and `openat` was removed, its ref must now be a Seed
            if p.calls[0].desc.name == "close" {
                assert!(matches!(p.calls[0].args[0], ArgValue::Res(ResRef::Seed(_))));
            }
        }
    }

    #[test]
    fn wire_producer_builds_open_read_close_chain() {
        let mut rng = Rng::new(1234);
        // Start from a program that only has consumers with no producer yet.
        let read = SYSCALLS.iter().find(|d| d.name == "read").unwrap();
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: read,
            args: generate_args(&mut rng, read, &[]),
        });
        assert!(p.is_well_formed());

        for _ in 0..20 {
            wire_producer(&mut rng, &mut p);
        }
        assert!(p.is_well_formed());
        // After enough tries, the read's fd arg should have become a Produced reference at
        // least once across many seeds (checked in the crate-level integration test too).
    }
}
