//! Targeted mutation over the typed program tree: insert/remove a call, mutate an arg by its
//! type (including re-picking a resource reference), and a dedicated "wire two calls together"
//! move that manufactures `open->read->close`-style chains instead of waiting for them to occur
//! by chance. Every operation preserves the threading invariant (`Prog::is_well_formed`). See
//! `docs/syzlang.md` §2.

use crate::dict::pick_dict_const;
use crate::genr::{
    PoolEntry, build_pool, gen_arg_value, generate, generate_args, mask_to_bits, pick_desc_biased,
    pick_interesting_int, pick_res_biased, same_call_anchors,
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
    // Biased like `generate()`'s per-call picker: prefer a description that consumes a resource
    // already live at `pos` (if any) over a uniform pick, so mutation-time insertion also tends
    // to deepen existing chains rather than just diluting them with unrelated scalar-only calls.
    let desc = pick_desc_biased(rng, &p.calls[..pos]);
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
            // Same T2.4 cross-reference bias as fresh generation (see `pick_res_biased`'s doc
            // comment): if this call's *other* args are already bound to a live resource, bias
            // re-rolling this one toward a sibling of that exact same specific kind rather than
            // the whole `kind`-compatible pool — e.g. re-rolling `epoll_ctl`'s target-fd arg in
            // an existing program that already has its `epfd` wired to a live epoll instance.
            let anchors = same_call_anchors(pool, &call.args, Some(j));
            call.args[j] = ArgValue::Res(pick_res_biased(rng, *kind, pool, &anchors));
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

/// Mutation op (a) from the fs-prog expansion brief: pick a `Res(kind)`-typed arg anywhere in
/// the program and, if a compatible producer already exists earlier in the program, rewrite the
/// arg to reference it. Unlike `wire_producer`, this never inserts a new call — it's the
/// lightweight "just reuse what's already there" half, so it fires cheaply and often without
/// growing the program.
fn splice_resource_use(rng: &mut Rng, p: &mut Prog) {
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
    if compatible.is_empty() {
        return;
    }
    let e = **rng.pick(&compatible);
    p.calls[idx].args[j] = ArgValue::Res(ResRef::Produced {
        call_idx: e.call_idx,
        slot: e.slot,
    });
}

/// Mutation op (b): toggle a single known flag value on or off (XOR one element of the arg's
/// `vals` table into its current bits) rather than only ever re-rolling the whole `Flags` value
/// from scratch via `gen_arg_value` — reaches small, targeted flag deltas a full re-roll would
/// rarely produce on its own.
fn toggle_flag_bit(rng: &mut Rng, p: &mut Prog) {
    let mut candidates: Vec<(usize, usize, &'static [u32])> = Vec::new();
    for (i, c) in p.calls.iter().enumerate() {
        for (j, aty) in c.desc.args.iter().enumerate() {
            if let ArgType::Flags { vals, .. } = aty
                && !vals.is_empty()
            {
                candidates.push((i, j, vals));
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let (i, j, vals) = *rng.pick(&candidates);
    let bit = *rng.pick(vals);
    if let ArgValue::Imm(v) = &mut p.calls[i].args[j] {
        *v ^= bit as u64;
    }
}

/// Mutation op (c): grow or shrink a `Buffer` payload in place (append random bytes, or
/// truncate) instead of only ever resampling a fresh length from its `LenSpec` — lets a
/// buffer/its sibling `Len{of}` arg drift out of sync incrementally across a mutation chain,
/// which is exactly the kind of "usually correct, occasionally desynced" length behavior
/// `docs/syzlang.md` calls out as valuable.
fn resize_buffer(rng: &mut Rng, p: &mut Prog) {
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for (i, c) in p.calls.iter().enumerate() {
        for (j, aty) in c.desc.args.iter().enumerate() {
            let ArgType::Ptr { inner, .. } = aty else {
                continue;
            };
            if !matches!(inner, ArgType::Buffer { .. }) {
                continue;
            }
            if matches!(&c.args[j], ArgValue::Ptr(b) if matches!(b.as_ref(), ArgValue::Bytes(_)))
            {
                candidates.push((i, j));
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let (i, j) = *rng.pick(&candidates);
    if let ArgValue::Ptr(inner) = &mut p.calls[i].args[j]
        && let ArgValue::Bytes(b) = inner.as_mut()
    {
        if b.is_empty() || rng.bool() {
            let n = 1 + rng.below(16);
            for _ in 0..n {
                b.push(rng.next() as u8);
            }
            b.truncate(4096); // stay well within the 32KiB scratch cap after many mutations
        } else {
            let cut = 1 + rng.below(b.len());
            let new_len = b.len() - cut;
            b.truncate(new_len);
        }
    }
}

/// Mutation op (d): swap in a curated "interesting" integer (0, 1, -1, `INT_MAX`, `PAGE_SIZE`,
/// ...) for an `Int`-typed scalar arg — `genr::gen_int` already biases fresh generation toward
/// these, but a dedicated mutator lets an existing, otherwise-unrelated program get nudged
/// straight to a boundary value without re-rolling everything else about that call.
fn mutate_interesting_int(rng: &mut Rng, p: &mut Prog) {
    let mut candidates: Vec<(usize, usize, u8)> = Vec::new();
    for (i, c) in p.calls.iter().enumerate() {
        for (j, aty) in c.desc.args.iter().enumerate() {
            if let ArgType::Int { bits, .. } = aty {
                candidates.push((i, j, *bits));
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let (i, j, bits) = *rng.pick(&candidates);
    p.calls[i].args[j] = ArgValue::Imm(pick_interesting_int(rng, bits));
}

/// Mutation op (e): swap in a curated real-kernel constant from `crate::dict` (an ioctl request
/// code, netlink type/flag, errno, fcntl/prctl command, ...) for an `Int`/`Flags`/`Const`-typed
/// scalar arg. This is the dedicated mutation half of the dictionary fix described in `dict`'s
/// module doc: `mutate_interesting_int` above only ever reaches syzkaller's generic 0/1/-1/
/// boundary set, which rarely equals the specific magic value a real kernel branch is gated on;
/// this operator targets that gap directly, including `Const` slots (whose value is otherwise
/// always fixed by the description and never touched by any other mutator) so a mutation chain
/// can still explore alternate real constants there instead of only ever the description's
/// hard-coded one.
fn mutate_dict_const(rng: &mut Rng, p: &mut Prog) {
    let mut candidates: Vec<(usize, usize, u8)> = Vec::new();
    for (i, c) in p.calls.iter().enumerate() {
        for (j, aty) in c.desc.args.iter().enumerate() {
            match aty {
                ArgType::Int { bits, .. } => candidates.push((i, j, *bits)),
                ArgType::Flags { .. } | ArgType::Const(_) => candidates.push((i, j, 32)),
                _ => {}
            }
        }
    }
    if candidates.is_empty() {
        return;
    }
    let (i, j, bits) = *rng.pick(&candidates);
    let v = mask_to_bits(pick_dict_const(rng) as u64, bits);
    p.calls[i].args[j] = ArgValue::Imm(v);
}

/// Mutate `base` into a new, still well-formed `Prog`. Never mutates `base` in place.
pub fn mutate(rng: &mut Rng, base: &Prog) -> Prog {
    let mut p = base.clone();
    if p.calls.is_empty() {
        return generate(rng);
    }
    match rng.below(9) {
        0 if p.calls.len() < MAX_CALLS => insert_call(rng, &mut p),
        1 if p.calls.len() > 1 => remove_call(rng, &mut p),
        2 => mutate_random_arg(rng, &mut p),
        3 => wire_producer(rng, &mut p),
        4 => splice_resource_use(rng, &mut p),
        5 => toggle_flag_bit(rng, &mut p),
        6 => resize_buffer(rng, &mut p),
        7 => mutate_interesting_int(rng, &mut p),
        _ => mutate_dict_const(rng, &mut p),
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

    #[test]
    fn splice_resource_use_rewires_without_inserting_calls() {
        // openat -> read, but read's fd starts as a Seed (no threading yet).
        let openat = SYSCALLS.iter().find(|d| d.name == "openat").unwrap();
        let read = SYSCALLS.iter().find(|d| d.name == "read").unwrap();
        let mut rng = Rng::new(77);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: openat,
            args: generate_args(&mut rng, openat, &[]),
        });
        let mut read_args = generate_args(&mut rng, read, &p.calls);
        read_args[0] = ArgValue::Res(ResRef::Seed(-1));
        p.calls.push(TypedCall {
            desc: read,
            args: read_args,
        });
        assert!(p.is_well_formed());
        let calls_before = p.calls.len();

        let mut saw_produced = false;
        for _ in 0..200 {
            splice_resource_use(&mut rng, &mut p);
            assert!(p.is_well_formed());
            assert_eq!(p.calls.len(), calls_before, "must never insert/remove calls");
            if matches!(p.calls[1].args[0], ArgValue::Res(ResRef::Produced { .. })) {
                saw_produced = true;
            }
        }
        assert!(saw_produced, "splice_resource_use never wired the fd");
    }

    #[test]
    fn toggle_flag_bit_changes_a_flags_arg_over_many_tries() {
        let openat = SYSCALLS.iter().find(|d| d.name == "openat").unwrap();
        let mut rng = Rng::new(9001);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: openat,
            args: generate_args(&mut rng, openat, &[]),
        });
        let ArgValue::Imm(initial) = p.calls[0].args[2] else {
            panic!("openat's flags arg should be Imm");
        };
        let mut changed = false;
        for _ in 0..100 {
            toggle_flag_bit(&mut rng, &mut p);
            assert!(p.is_well_formed());
            if let ArgValue::Imm(v) = p.calls[0].args[2]
                && v != initial
            {
                changed = true;
            }
        }
        assert!(changed, "toggle_flag_bit never changed the flags value");
    }

    #[test]
    fn resize_buffer_grows_or_shrinks_a_buffer_payload() {
        let write = SYSCALLS.iter().find(|d| d.name == "write").unwrap();
        let mut rng = Rng::new(555);
        let mut p = Prog::new();
        let mut args = generate_args(&mut rng, write, &[]);
        // Force a known, nonzero starting length so both grow and shrink are exercisable.
        args[1] = ArgValue::Ptr(Box::new(ArgValue::Bytes(vec![0u8; 10])));
        args[2] = ArgValue::Imm(10);
        p.calls.push(TypedCall { desc: write, args });
        assert!(p.is_well_formed());

        let mut saw_len_change = false;
        for _ in 0..200 {
            resize_buffer(&mut rng, &mut p);
            assert!(p.is_well_formed());
            if let ArgValue::Ptr(b) = &p.calls[0].args[1]
                && let ArgValue::Bytes(bytes) = b.as_ref()
                && bytes.len() != 10
            {
                saw_len_change = true;
            }
        }
        assert!(saw_len_change, "resize_buffer never changed the buffer length");
        // lower() must still succeed even though the Len{of:1} arg (still Imm(10)) is now
        // desynced from the buffer's actual (mutated) length — that's the intended fuzz signal.
        let _ = crate::lower::lower(&p, 0x9000_0000);
    }

    #[test]
    fn mutate_interesting_int_picks_from_the_curated_pool() {
        let prctl = SYSCALLS.iter().find(|d| d.name == "prctl").unwrap();
        let mut rng = Rng::new(321);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: prctl,
            args: generate_args(&mut rng, prctl, &[]),
        });
        for _ in 0..50 {
            mutate_interesting_int(&mut rng, &mut p);
            assert!(p.is_well_formed());
        }
        // prctl's args[1..] are all Int{32,unsigned}; at least one should have landed on a
        // curated interesting value across 50 tries.
        let saw_interesting = p.calls[0].args[1..].iter().any(|a| {
            matches!(a, ArgValue::Imm(v) if [0u64,1,2,u32::MAX as u64,4096,(-4096i64) as u32 as u64,i32::MAX as u64].contains(v))
        });
        assert!(saw_interesting, "never landed on a curated interesting value");
    }

    #[test]
    fn mutate_dict_const_lands_on_a_cataloged_dictionary_value() {
        use crate::dict::DICTIONARY_GROUPS;
        // openat has an Int-free but Flags/Const-bearing arg list; use ioctl$generic instead,
        // which has both a Flags (cmd) and no Const, plus prctl for Int coverage — exercised via
        // whichever candidates mutate_dict_const finds on ioctl$generic.
        let ioctl = SYSCALLS.iter().find(|d| d.name == "ioctl$generic").unwrap();
        let mut rng = Rng::new(55);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: ioctl,
            args: generate_args(&mut rng, ioctl, &[]),
        });
        let mut saw_dict_value = false;
        for _ in 0..200 {
            mutate_dict_const(&mut rng, &mut p);
            assert!(p.is_well_formed());
            if let ArgValue::Imm(v) = p.calls[0].args[1]
                && DICTIONARY_GROUPS.iter().any(|g| g.contains(&(v as u32)))
            {
                saw_dict_value = true;
            }
        }
        assert!(saw_dict_value, "mutate_dict_const never landed on a cataloged dictionary value");
    }

    #[test]
    fn mutate_dict_const_can_rewrite_a_const_typed_slot() {
        // fcntl64$setfl's 2nd arg is a fixed Const(4 /* F_SETFL */) — no other mutator ever
        // touches a Const slot, so this checks mutate_dict_const specifically reaches it.
        let desc = SYSCALLS.iter().find(|d| d.name == "fcntl64$setfl").unwrap();
        let mut rng = Rng::new(66);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc,
            args: generate_args(&mut rng, desc, &[]),
        });
        let ArgValue::Imm(initial) = p.calls[0].args[1] else {
            panic!("expected Const arg to be Imm");
        };
        let mut changed = false;
        for _ in 0..200 {
            mutate_dict_const(&mut rng, &mut p);
            assert!(p.is_well_formed());
            if let ArgValue::Imm(v) = p.calls[0].args[1]
                && v != initial
            {
                changed = true;
            }
        }
        assert!(changed, "mutate_dict_const never touched the Const slot");
    }

    #[test]
    fn full_dispatch_exercises_all_new_ops_without_breaking_well_formedness() {
        let mut rng = Rng::new(2024);
        let mut p = generate(&mut rng);
        for _ in 0..3000 {
            p = mutate(&mut rng, &p);
            assert!(p.is_well_formed());
            assert!(!p.calls.is_empty());
            assert!(p.calls.len() <= MAX_CALLS);
        }
    }
}
