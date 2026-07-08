//! SIMD-shaped batch mutation: the input-side lever that keeps a batch of guest VMs
//! *control-flow convergent* so the vectorized emulator's SIMD lanes actually pay off.
//!
//! A vectorized emulator runs `N` guest lanes in lockstep; SIMD only helps when every lane is at
//! the *same* program counter with *different* data (single instruction, multiple data). If a
//! batch of `N` fuzz inputs is mutated freely (different syscalls, different call sequences,
//! different resource threading), the lanes take different kernel code paths almost immediately
//! and all SIMD benefit is lost.
//!
//! The fix implemented here: given one parent `Prog`, generate `N` sibling programs that share
//! the *exact same control-flow skeleton* — same calls in the same order, same `SyscallDesc`
//! (hence same `nr`) per call, same `ArgType`s, same `Res`/`ResRef` threading — and differ *only*
//! in leaf DATA: `ArgValue::Imm` values sitting in `Const`/`Int`/`Flags`/`Len`-typed slots, and
//! `Bytes` contents sitting in `Buffer`/`StringConst`-typed slots (including inside nested
//! `Ptr`/`Struct` payloads). All `N` guests then execute the identical kernel control-flow path
//! and diverge only at data-dependent branches — exactly the coverage frontier worth exploring,
//! and exactly the input-side complement to the emulator's execution-side lane-divergence
//! handling.
//!
//! "Skeleton" vs. "data", precisely:
//! - Skeleton (held fixed): `Prog.calls.len()`, each call's `desc` pointer (name/nr/arg types),
//!   and every `ArgValue::Res(ResRef)` (`Seed` or `Produced{call_idx,slot}`) — i.e. everything
//!   `Prog::is_well_formed` cares about plus the call list itself.
//! - Data (the only thing mutated here): every other `ArgValue` leaf — `Imm` scalars and `Bytes`
//!   buffers, including ones nested inside a `Ptr` pointee or a `Struct`'s fields.
//!
//! Mutation is biased toward "influential" leaves — `Flags`, `Len`, `Const`, and the first few
//! bytes of `Buffer`/`Struct` payloads (headers) — because those are what data-dependent kernel
//! branches actually test; a byte that's merely copied through and never branched on gives no
//! productive divergence between lanes.

use crate::dict::pick_dict_const;
use crate::genr::{mask_to_bits, pick_interesting_int};
use crate::prog::{ArgValue, Prog};
use crate::rng::Rng;
use crate::types::ArgType;

/// One step of a path from a call's top-level arg down into nested `Ptr`/`Struct` payloads.
#[derive(Clone, Copy)]
enum PathStep {
    PtrInner,
    StructField(usize),
}

/// What kind of leaf sits at a `LeafSite`, plus whatever static data (flag value table) is
/// needed to mutate it without re-walking the `ArgType` tree.
#[derive(Clone)]
enum LeafKind {
    Flags { vals: &'static [u32], bitmask: bool },
    Const,
    Int { bits: u8 },
    Len,
    Bytes,
}

/// One mutable leaf-data site found somewhere in a `Prog`: `calls[call_idx].args[arg_idx]`,
/// descended via `path` (empty for a top-level scalar/bytes arg).
struct LeafSite {
    call_idx: usize,
    arg_idx: usize,
    path: Vec<PathStep>,
    kind: LeafKind,
    /// Relative selection weight; higher = more likely to be picked. Biases toward
    /// control-flow-influential args (see module doc).
    weight: u32,
}

/// Recursively walk one (type, value) pair — a top-level arg or something nested inside a
/// `Ptr`/`Struct` — collecting every leaf-data site reachable from it. Never descends into or
/// emits a site for `Res`-typed values: resource threading is skeleton, not data.
fn collect_leaves_in_value(
    aty: &ArgType,
    av: &ArgValue,
    path: &mut Vec<PathStep>,
    out: &mut Vec<(Vec<PathStep>, LeafKind, u32)>,
) {
    match (aty, av) {
        (ArgType::Res(_), _) => {} // resource refs are skeleton, never touched here
        (ArgType::Flags { vals, bitmask }, ArgValue::Imm(_)) => out.push((
            path.clone(),
            LeafKind::Flags {
                vals,
                bitmask: *bitmask,
            },
            30,
        )),
        (ArgType::Const(_), ArgValue::Imm(_)) => out.push((path.clone(), LeafKind::Const, 15)),
        (ArgType::Int { bits, .. }, ArgValue::Imm(_)) => {
            out.push((path.clone(), LeafKind::Int { bits: *bits }, 20))
        }
        (ArgType::Len { .. }, ArgValue::Imm(_)) => out.push((path.clone(), LeafKind::Len, 20)),
        (ArgType::Buffer { .. }, ArgValue::Bytes(_))
        | (ArgType::StringConst(_), ArgValue::Bytes(_)) => {
            out.push((path.clone(), LeafKind::Bytes, 25))
        }
        (ArgType::Ptr { inner, .. }, ArgValue::Ptr(pointee)) => {
            path.push(PathStep::PtrInner);
            collect_leaves_in_value(inner, pointee, path, out);
            path.pop();
        }
        (ArgType::Ptr { .. }, ArgValue::Imm(_)) => {} // NULL pointer, nothing to mutate
        (ArgType::Struct(fields), ArgValue::Struct(vals)) => {
            for (i, (f, v)) in fields.iter().zip(vals.iter()).enumerate() {
                path.push(PathStep::StructField(i));
                collect_leaves_in_value(f.ty, v, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

/// Collect every mutable leaf-data site across the whole program.
fn collect_all_leaves(p: &Prog) -> Vec<LeafSite> {
    let mut sites = Vec::new();
    for (ci, c) in p.calls.iter().enumerate() {
        for (ai, (aty, av)) in c.desc.args.iter().zip(&c.args).enumerate() {
            let mut path = Vec::new();
            let mut local = Vec::new();
            collect_leaves_in_value(aty, av, &mut path, &mut local);
            for (path, kind, weight) in local {
                sites.push(LeafSite {
                    call_idx: ci,
                    arg_idx: ai,
                    path,
                    kind,
                    weight,
                });
            }
        }
    }
    sites
}

/// Descend from a top-level `ArgValue` through `path`, returning a mutable reference to the leaf
/// it names. `path` is always constructed by `collect_leaves_in_value` walking the very same
/// value, so every step is guaranteed to match.
fn navigate_mut<'a>(av: &'a mut ArgValue, path: &[PathStep]) -> &'a mut ArgValue {
    let mut cur = av;
    for step in path {
        cur = match (step, cur) {
            (PathStep::PtrInner, ArgValue::Ptr(b)) => b.as_mut(),
            (PathStep::StructField(i), ArgValue::Struct(vals)) => &mut vals[*i],
            (_, other) => other, // unreachable given how paths are built
        };
    }
    cur
}

/// Full re-roll of a `Flags` value from its own `vals`/`bitmask` shape (mirrors
/// `genr::gen_flags`'s policy; duplicated locally to keep this module additive/self-contained).
fn reroll_flags(rng: &mut Rng, vals: &[u32], bitmask: bool) -> u64 {
    if vals.is_empty() {
        return 0;
    }
    if !bitmask {
        *rng.pick(vals) as u64
    } else {
        vals.iter().fold(0u32, |acc, &v| if rng.bool() { acc | v } else { acc }) as u64
    }
}

/// Perturb a scalar `Imm` value in place: swap in a curated "interesting" value (0, 1, -1,
/// `INT_MAX`, `PAGE_SIZE`, ...), swap in a real curated kernel constant from `crate::dict`
/// (ioctl cmd, netlink type, errno, ...) — see that module's doc for why SIMD-batch data
/// diversity benefits from real constants too, not just syzkaller's generic boundary set — flip
/// a single bit, or draw a fresh random value. Masked to `bits`.
fn mutate_scalar_bits(rng: &mut Rng, v: &mut u64, bits: u8) {
    let bits_nonzero = bits.max(1);
    let nv = match rng.below(4) {
        0 => pick_interesting_int(rng, bits),
        1 => pick_dict_const(rng) as u64,
        2 => *v ^ (1u64 << (rng.next() % bits_nonzero as u32).min(63)),
        _ => rng.next_u64(),
    };
    *v = mask_to_bits(nv, bits);
}

/// Perturb one byte of a buffer in place, biased toward the first few bytes (headers/tags —
/// exactly what data-dependent parsers tend to switch on) over the tail.
fn mutate_bytes(rng: &mut Rng, b: &mut [u8]) {
    if b.is_empty() {
        return;
    }
    let head = 4.min(b.len());
    let idx = if b.len() > head && rng.chance(70) {
        rng.below(head)
    } else {
        rng.below(b.len())
    };
    match rng.below(3) {
        0 => b[idx] = rng.next() as u8,
        1 => b[idx] ^= 0xFF,
        _ => b[idx] = b[idx].wrapping_add(1),
    }
}

/// Apply the mutation appropriate to `kind` at `av` (which must be the `ArgValue` `kind` was
/// derived from by `collect_leaves_in_value`).
fn apply_leaf_mutation(rng: &mut Rng, av: &mut ArgValue, kind: &LeafKind) {
    match (kind, av) {
        (LeafKind::Flags { vals, bitmask }, ArgValue::Imm(v)) => {
            if !vals.is_empty() {
                if rng.chance(60) {
                    let bit = *rng.pick(vals);
                    *v ^= bit as u64;
                } else {
                    *v = reroll_flags(rng, vals, *bitmask);
                }
            }
        }
        (LeafKind::Const, ArgValue::Imm(v)) | (LeafKind::Len, ArgValue::Imm(v)) => {
            mutate_scalar_bits(rng, v, 32);
        }
        (LeafKind::Int { bits }, ArgValue::Imm(v)) => {
            mutate_scalar_bits(rng, v, *bits);
        }
        (LeafKind::Bytes, ArgValue::Bytes(b)) => {
            mutate_bytes(rng, b);
        }
        _ => {} // unreachable given how LeafSite kinds are derived from the matching ArgValue
    }
}

/// Mutate one randomly (weighted) chosen leaf-data site in `p`, in place. No-op if `p` has no
/// mutable leaf-data sites at all (e.g. an all-resource, empty, or degenerate program).
fn mutate_data_in_place(rng: &mut Rng, p: &mut Prog) {
    let sites = collect_all_leaves(p);
    if sites.is_empty() {
        return;
    }
    let total_weight: u32 = sites.iter().map(|s| s.weight).sum();
    let mut r = rng.below(total_weight.max(1) as usize) as u32;
    let mut chosen_idx = sites.len() - 1;
    for (i, s) in sites.iter().enumerate() {
        if r < s.weight {
            chosen_idx = i;
            break;
        }
        r -= s.weight;
    }
    let LeafSite {
        call_idx,
        arg_idx,
        path,
        kind,
        ..
    } = &sites[chosen_idx];
    let av = navigate_mut(&mut p.calls[*call_idx].args[*arg_idx], path);
    apply_leaf_mutation(rng, av, kind);
}

/// Mutation op (2) from the SIMD-batch-mutation brief: mutate *only* leaf data in `base` —
/// `ArgValue::Imm` values in `Const`/`Int`/`Flags`/`Len`-typed slots and `Bytes` contents in
/// `Buffer`/`StringConst`-typed slots, including inside nested `Ptr`/`Struct` payloads — without
/// touching the call list, `nr`s, `ArgType`s, or `Res`/`ResRef` threading. Never mutates `base`
/// in place. See the module doc for why this is the SIMD-batch-friendly mutation surface, and
/// which leaves are biased toward.
pub fn mutate_data(rng: &mut Rng, base: &Prog) -> Prog {
    let mut p = base.clone();
    mutate_data_in_place(rng, &mut p);
    p
}

/// Generate `n` sibling `Prog`s from `parent`, all sharing the identical control-flow skeleton
/// (same calls, same order, same `desc` per call — hence identical `nr`s/`ArgType`s — and
/// identical `Res`/`ResRef` threading) but differing in leaf DATA (see `mutate_data`).
///
/// This is the "SIMD-shaped batch mutation": pair the returned `Vec<Prog>` 1:1 with the
/// vectorized emulator's `N` lanes and every lane runs the same kernel control-flow path,
/// diverging only where the kernel branches on data — which is exactly the productive,
/// SIMD-preserving divergence to explore. Deterministic: the same `rng` state and `n` always
/// produce the same batch.
///
/// Each sibling gets its own accumulated combination of leaf tweaks (1-3 mutation rounds) so
/// that, across `n` siblings, there is real data diversity rather than every sibling landing on
/// the exact same single site/value. If `parent` has no mutable leaf-data sites at all (e.g. an
/// empty program), every sibling is simply an identical clone of `parent` — the skeleton
/// guarantee still holds, just with no data to diversify.
pub fn mutate_batch(rng: &mut Rng, parent: &Prog, n: usize) -> Vec<Prog> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut sibling = parent.clone();
        let rounds = 1 + rng.below(3);
        for _ in 0..rounds {
            mutate_data_in_place(rng, &mut sibling);
        }
        out.push(sibling);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genr::generate;
    use crate::lower::lower;
    use crate::prog::{MAX_CALLS, ResRef};
    use crate::rng::Rng;

    /// Everything `mutate_batch` must hold fixed across siblings, extracted from a `Prog` for
    /// comparison: per-call `(name, nr)`, per-arg `ArgType` discriminant tags, and the full
    /// `Res`/`ResRef` structure. Also carries the lowered fixup table + wire nrs, since the brief
    /// asks us to verify skeleton identity "by comparing lower()'s fixup table + the call nrs".
    #[derive(Debug, PartialEq, Eq)]
    struct Skeleton {
        call_names_nrs: Vec<(&'static str, u32)>,
        arg_type_tags: Vec<Vec<&'static str>>,
        res_refs: Vec<Vec<Option<ResRef>>>,
        wire_nrs: Vec<u32>,
        fixups: Vec<(u8, u8, u16, u8)>, // (dst_call, dst_arg, source_call_index, source_slot)
    }

    fn arg_type_tag(aty: &ArgType) -> &'static str {
        match aty {
            ArgType::Const(_) => "Const",
            ArgType::Int { .. } => "Int",
            ArgType::Flags { .. } => "Flags",
            ArgType::Res(_) => "Res",
            ArgType::Len { .. } => "Len",
            ArgType::Ptr { .. } => "Ptr",
            ArgType::Buffer { .. } => "Buffer",
            ArgType::Struct(_) => "Struct",
            ArgType::StringConst(_) => "StringConst",
        }
    }

    fn skeleton_of(p: &Prog) -> Skeleton {
        let call_names_nrs = p.calls.iter().map(|c| (c.desc.name, c.desc.nr)).collect();
        let arg_type_tags = p
            .calls
            .iter()
            .map(|c| c.desc.args.iter().map(arg_type_tag).collect())
            .collect();
        let res_refs = p
            .calls
            .iter()
            .map(|c| {
                c.args
                    .iter()
                    .map(|av| match av {
                        ArgValue::Res(r) => Some(*r),
                        _ => None,
                    })
                    .collect()
            })
            .collect();
        let lowered = lower(p, 0x9000_0000);
        let wire_nrs = lowered.calls.iter().map(|c| c.nr).collect();
        let fixups = lowered
            .fixups
            .iter()
            .map(|f| (f.dst_call, f.dst_arg, f.source_call_index, f.source_slot))
            .collect();
        Skeleton {
            call_names_nrs,
            arg_type_tags,
            res_refs,
            wire_nrs,
            fixups,
        }
    }

    /// (a) `mutate_batch` returns siblings with skeletons identical to the parent's (and hence
    /// to each other), across many parent seeds.
    #[test]
    fn batch_siblings_share_identical_skeleton() {
        for seed in 1..150u32 {
            let mut gen_rng = Rng::new(seed);
            let parent = generate(&mut gen_rng);
            let parent_skel = skeleton_of(&parent);

            let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(3));
            let batch = mutate_batch(&mut rng, &parent, 16);
            assert_eq!(batch.len(), 16);
            for (i, sib) in batch.iter().enumerate() {
                assert_eq!(
                    skeleton_of(sib),
                    parent_skel,
                    "seed {seed} sibling {i} skeleton diverged from parent"
                );
            }
        }
    }

    /// (b) The siblings differ in at least some leaf data — real data diversity, not a batch of
    /// identical clones.
    #[test]
    fn batch_siblings_have_data_diversity() {
        let mut saw_diversity_in_a_seed = false;
        for seed in 1..150u32 {
            let mut gen_rng = Rng::new(seed);
            let parent = generate(&mut gen_rng);
            // Skip degenerate parents with no mutable leaf-data sites at all.
            if collect_all_leaves(&parent).is_empty() {
                continue;
            }
            let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(3));
            let batch = mutate_batch(&mut rng, &parent, 16);
            let first_args: Vec<Vec<ArgValue>> =
                batch.iter().map(|p| p.calls.iter().flat_map(|c| c.args.clone()).collect()).collect();
            let all_identical = first_args.iter().all(|a| *a == first_args[0]);
            if !all_identical {
                saw_diversity_in_a_seed = true;
            }
        }
        assert!(
            saw_diversity_in_a_seed,
            "mutate_batch never produced any data diversity across 150 parent seeds"
        );
    }

    /// (c) Every sibling stays well-formed and lowers to exactly `WIRE_WORDS` with
    /// `nfix <= MAX_FIXUPS`.
    #[test]
    fn batch_siblings_are_well_formed_and_lower_cleanly() {
        use crate::lower::{MAX_FIXUPS, WIRE_WORDS, to_wire};
        for seed in 1..150u32 {
            let mut gen_rng = Rng::new(seed);
            let parent = generate(&mut gen_rng);
            let mut rng = Rng::new(seed.wrapping_mul(31).wrapping_add(11));
            let batch = mutate_batch(&mut rng, &parent, 16);
            for sib in &batch {
                assert!(sib.is_well_formed(), "seed {seed}: sibling not well-formed");
                assert!(!sib.calls.is_empty());
                assert!(sib.calls.len() <= MAX_CALLS);
                let lowered = lower(sib, 0x9000_0000);
                assert!(lowered.fixups.len() <= MAX_FIXUPS);
                let wire = to_wire(&lowered);
                assert_eq!(wire.len(), WIRE_WORDS);
            }
        }
    }

    /// (d) Determinism: same seed -> same batch (call-for-call, byte-for-byte).
    #[test]
    fn batch_is_deterministic_for_same_seed() {
        for seed in [1u32, 2, 42, 12345, 999_999] {
            let mut gen_rng = Rng::new(seed);
            let parent = generate(&mut gen_rng);

            let mut rng_a = Rng::new(seed);
            let batch_a = mutate_batch(&mut rng_a, &parent, 16);
            let mut rng_b = Rng::new(seed);
            let batch_b = mutate_batch(&mut rng_b, &parent, 16);

            assert_eq!(batch_a.len(), batch_b.len());
            for (a, b) in batch_a.iter().zip(&batch_b) {
                for (ca, cb) in a.calls.iter().zip(&b.calls) {
                    assert_eq!(ca.desc.name, cb.desc.name);
                    assert_eq!(ca.args, cb.args);
                }
            }
        }
    }

    /// `mutate_data` alone: leaf-only mutation preserves skeleton and well-formedness.
    #[test]
    fn mutate_data_preserves_skeleton_and_well_formedness() {
        for seed in 1..200u32 {
            let mut gen_rng = Rng::new(seed);
            let base = generate(&mut gen_rng);
            let base_skel = skeleton_of(&base);
            let mut rng = Rng::new(seed.wrapping_add(500));
            let mutated = mutate_data(&mut rng, &base);
            assert_eq!(skeleton_of(&mutated), base_skel, "seed {seed}");
            assert!(mutated.is_well_formed());
        }
    }

    /// `mutate_data` never touches an empty program (nothing to mutate, no panic).
    #[test]
    fn mutate_data_on_empty_program_is_a_noop() {
        let empty = Prog::new();
        let mut rng = Rng::new(1);
        let mutated = mutate_data(&mut rng, &empty);
        assert!(mutated.calls.is_empty());
    }

    /// `mutate_batch`'s leaf-data diversification reaches `crate::dict`'s curated constants too
    /// (not just `pick_interesting_int`'s generic 0/1/boundary set) — across enough siblings some
    /// scalar leaf should land on a cataloged dictionary value.
    #[test]
    fn mutate_batch_siblings_sometimes_carry_a_dictionary_constant() {
        use crate::dict::DICTIONARY_GROUPS;
        let is_dict_value = |v: u32| DICTIONARY_GROUPS.iter().any(|g| g.contains(&v));

        let mut saw_dict_value = false;
        'seeds: for seed in 1..200u32 {
            let mut gen_rng = Rng::new(seed);
            let parent = generate(&mut gen_rng);
            let mut rng = Rng::new(seed.wrapping_mul(101).wrapping_add(1));
            let batch = mutate_batch(&mut rng, &parent, 32);
            for sib in &batch {
                for c in &sib.calls {
                    for av in &c.args {
                        if let ArgValue::Imm(v) = av
                            && is_dict_value(*v as u32)
                        {
                            saw_dict_value = true;
                            break 'seeds;
                        }
                    }
                }
            }
        }
        assert!(
            saw_dict_value,
            "mutate_batch never produced a dictionary-constant leaf across 200 parent seeds x 32 siblings"
        );
    }
}
