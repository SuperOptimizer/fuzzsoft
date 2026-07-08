//! CMPLOG (comparison-coverage) mutation operator — the RedQueen/AFL++-CMPLOG technique for
//! solving data-dependent "magic value" branches (`if (x == MAGIC)`) that random mutation would
//! need up to 2^32 tries to hit by chance.
//!
//! Companion to `fs-riscv`'s optional per-`Cpu` comparison log (`Cpu::set_cmplog`/
//! `cmplog_take`): the fuzzer runs a candidate program once with recording on, collects the
//! `(a, b)` operand pairs observed at every executed branch (and `Sub`/`Xor`, the compiler's
//! usual equality-compare lowering), then calls [`mutate_cmplog`] here to scan the SAME
//! program's own data (`Imm` args and `Buffer`/`StringConst` bytes, including nested inside
//! `Ptr`/`Struct` payloads) for a value matching one side of a logged pair. Wherever a match is
//! found, the OTHER side (and its ±1 neighbours, in case the real check is `<`/`<=`/`>`/`>=`
//! rather than exact equality) is substituted in — "guessing" the constant a kernel check wants
//! in one mutation step instead of waiting on random search.
//!
//! This module parallels `batch.rs`'s leaf-data traversal (same `Ptr`/`Struct` walk, same
//! "resource refs are skeleton, never touched" rule) but is driven by cmp pairs instead of a bare
//! RNG, and is kept independently self-contained (its own small `PathStep`/navigate helpers)
//! rather than reusing batch.rs's private types, so it can be read/reviewed on its own.

use crate::prog::{ArgValue, Prog};
use crate::rng::Rng;
use crate::types::ArgType;

/// Bound on how many substitution candidates we ever collect for one `mutate_cmplog` call — caps
/// the worst-case cost regardless of how many cmp pairs were logged or how large the program's
/// buffers are.
const MAX_CANDIDATES: usize = 256;
/// Bound on how many *unique* logged pairs we scan against — a tight compare loop logs the same
/// pair thousands of times; dedup + this cap keeps the scan cheap without losing coverage of the
/// distinct constants actually compared against.
const MAX_UNIQUE_PAIRS: usize = 128;
/// Bound on how many leading bytes of a `Buffer`/`StringConst` we scan for an embedded magic
/// value — data-dependent parsers overwhelmingly switch on a small header/tag, not tail bytes, so
/// this trades a little theoretical reach for a lot of practical speed on large buffers.
const BUFFER_SCAN_CAP: usize = 256;

/// One step of a path from a call's top-level arg down into nested `Ptr`/`Struct` payloads
/// (mirrors `batch::PathStep`; duplicated to keep this module independently reviewable).
#[derive(Clone, Copy)]
enum PathStep {
    PtrInner,
    StructField(usize),
}

/// The concrete substitution one [`Candidate`] would make.
enum Replacement {
    /// Overwrite an `Imm` scalar outright (RV32 register/arg values are always 32-bit, so no
    /// separate bit-width bookkeeping is needed here — see the module doc).
    Imm(u32),
    /// Splice `bytes` into a `Bytes` buffer starting at byte offset `off`.
    Bytes { off: usize, bytes: Vec<u8> },
}

/// One candidate substitution: where in the program tree, and what to put there.
struct Candidate {
    call_idx: usize,
    arg_idx: usize,
    path: Vec<PathStep>,
    repl: Replacement,
}

fn navigate<'a>(av: &'a ArgValue, path: &[PathStep]) -> &'a ArgValue {
    let mut cur = av;
    for step in path {
        cur = match (step, cur) {
            (PathStep::PtrInner, ArgValue::Ptr(b)) => b.as_ref(),
            (PathStep::StructField(i), ArgValue::Struct(vals)) => &vals[*i],
            (_, other) => other, // unreachable given how paths are built
        };
    }
    cur
}

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

/// Recursively collect every `Imm`/`Bytes` leaf reachable from `(aty, av)` as `(path, is_bytes)`,
/// skipping `Res` (resource threading is skeleton, never a cmplog substitution target).
fn collect_leaves(
    aty: &ArgType,
    av: &ArgValue,
    path: &mut Vec<PathStep>,
    out: &mut Vec<(Vec<PathStep>, bool)>,
) {
    match (aty, av) {
        (ArgType::Res(_), _) => {}
        (ArgType::Ptr { inner, .. }, ArgValue::Ptr(pointee)) => {
            path.push(PathStep::PtrInner);
            collect_leaves(inner, pointee, path, out);
            path.pop();
        }
        (ArgType::Struct(fields), ArgValue::Struct(vals)) => {
            for (i, (f, v)) in fields.iter().zip(vals.iter()).enumerate() {
                path.push(PathStep::StructField(i));
                collect_leaves(f.ty, v, path, out);
                path.pop();
            }
        }
        (_, ArgValue::Imm(_)) => out.push((path.clone(), false)),
        (_, ArgValue::Bytes(_)) => out.push((path.clone(), true)),
        _ => {}
    }
}

/// `x`, `x+1`, `x-1` — covers an off-by-one relational check (`<`, `<=`, `>`, `>=`) in addition to
/// exact equality.
fn near(x: u32) -> [u32; 3] {
    [x, x.wrapping_add(1), x.wrapping_sub(1)]
}

fn near_masked(x: u32, mask: u32) -> [u32; 3] {
    near(x).map(|v| v & mask)
}

/// Scan one `Imm` leaf's value against every logged pair; push a candidate substitution for each
/// side that matches (using the OTHER side, ± a neighbour, as the new value).
fn scan_imm(
    call_idx: usize,
    arg_idx: usize,
    path: &[PathStep],
    v: u64,
    pairs: &[(u32, u32)],
    out: &mut Vec<Candidate>,
) {
    let v32 = v as u32;
    for &(a, b) in pairs {
        if v32 == a && b != a {
            for nb in near(b) {
                out.push(Candidate {
                    call_idx,
                    arg_idx,
                    path: path.to_vec(),
                    repl: Replacement::Imm(nb),
                });
            }
        }
        if v32 == b && a != b {
            for na in near(a) {
                out.push(Candidate {
                    call_idx,
                    arg_idx,
                    path: path.to_vec(),
                    repl: Replacement::Imm(na),
                });
            }
        }
        if out.len() >= MAX_CANDIDATES {
            return;
        }
    }
}

/// Scan one `Bytes` leaf for any 1/2/4-byte little-endian window matching either side of a
/// logged pair, over the leaf's first [`BUFFER_SCAN_CAP`] bytes; push a candidate splicing in the
/// other side (± a neighbour) at that offset.
fn scan_bytes(
    call_idx: usize,
    arg_idx: usize,
    path: &[PathStep],
    buf: &[u8],
    pairs: &[(u32, u32)],
    out: &mut Vec<Candidate>,
) {
    let scan_len = buf.len().min(BUFFER_SCAN_CAP);
    for width in [4usize, 2, 1] {
        if scan_len < width {
            continue;
        }
        let mask = if width == 4 { u32::MAX } else { (1u32 << (8 * width)) - 1 };
        for off in 0..=(scan_len - width) {
            let mut cur = 0u32;
            for k in 0..width {
                cur |= (buf[off + k] as u32) << (8 * k);
            }
            for &(a, b) in pairs {
                let (aw, bw) = (a & mask, b & mask);
                if cur == aw && bw != aw {
                    for nb in near_masked(bw, mask) {
                        out.push(Candidate {
                            call_idx,
                            arg_idx,
                            path: path.to_vec(),
                            repl: Replacement::Bytes { off, bytes: nb.to_le_bytes()[..width].to_vec() },
                        });
                    }
                }
                if cur == bw && aw != bw {
                    for na in near_masked(aw, mask) {
                        out.push(Candidate {
                            call_idx,
                            arg_idx,
                            path: path.to_vec(),
                            repl: Replacement::Bytes { off, bytes: na.to_le_bytes()[..width].to_vec() },
                        });
                    }
                }
            }
            if out.len() >= MAX_CANDIDATES {
                return;
            }
        }
    }
}

fn apply(p: &mut Prog, cand: &Candidate) {
    let av = navigate_mut(&mut p.calls[cand.call_idx].args[cand.arg_idx], &cand.path);
    match (&cand.repl, av) {
        (Replacement::Imm(v), ArgValue::Imm(cur)) => *cur = *v as u64,
        (Replacement::Bytes { off, bytes }, ArgValue::Bytes(buf)) => {
            let end = (*off + bytes.len()).min(buf.len());
            let n = end.saturating_sub(*off);
            buf[*off..*off + n].copy_from_slice(&bytes[..n]);
        }
        _ => {} // unreachable: `repl`'s variant is derived from the matching leaf's own kind
    }
}

/// Mutate `base` by substituting a CMPLOG-derived "guessed" value at one leaf-data site,
/// choosing uniformly among every substitution [`scan_imm`]/[`scan_bytes`] can find from `pairs`
/// (a comparison-operand log recorded by running `base` — or a close relative of it — with
/// `fs_riscv::Cpu::set_cmplog(true)`). Returns `None` if `pairs` is empty or no leaf in `base`
/// matches either side of any logged pair (the caller should then fall back to an ordinary
/// mutation, e.g. [`crate::mutate::mutate`]). Never mutates `base` in place; the skeleton (call
/// list, `nr`s, `ArgType`s, `Res`/`ResRef` threading) is always preserved, since only `Imm`/
/// `Bytes` leaves are ever touched — same invariant `mutate_data` guarantees.
pub fn mutate_cmplog(rng: &mut Rng, base: &Prog, pairs: &[(u32, u32)]) -> Option<Prog> {
    if pairs.is_empty() {
        return None;
    }
    let mut uniq: Vec<(u32, u32)> = pairs.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    uniq.truncate(MAX_UNIQUE_PAIRS);

    let mut candidates: Vec<Candidate> = Vec::new();
    'outer: for (ci, c) in base.calls.iter().enumerate() {
        for (ai, (aty, av)) in c.desc.args.iter().zip(&c.args).enumerate() {
            let mut path = Vec::new();
            let mut leaves = Vec::new();
            collect_leaves(aty, av, &mut path, &mut leaves);
            for (leaf_path, is_bytes) in &leaves {
                let leaf = navigate(av, leaf_path);
                if *is_bytes {
                    if let ArgValue::Bytes(b) = leaf {
                        scan_bytes(ci, ai, leaf_path, b, &uniq, &mut candidates);
                    }
                } else if let ArgValue::Imm(v) = leaf {
                    scan_imm(ci, ai, leaf_path, *v, &uniq, &mut candidates);
                }
                if candidates.len() >= MAX_CANDIDATES {
                    break 'outer;
                }
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }
    let idx = rng.below(candidates.len());
    let mut p = base.clone();
    apply(&mut p, &candidates[idx]);
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genr::generate;
    use crate::prog::{ArgValue, TypedCall};
    use crate::syscalls::SYSCALLS;

    /// No pairs -> no mutation, always `None`.
    #[test]
    fn empty_pairs_yields_none() {
        let mut rng = Rng::new(1);
        let prog = generate(&mut rng);
        assert!(mutate_cmplog(&mut rng, &prog, &[]).is_none());
    }

    /// The core efficacy property: a program contains an `Imm` arg whose current value (0) was
    /// logged as one side of a compare against a MAGIC constant on the other side. `mutate_cmplog`
    /// must find it and substitute the magic value (or a ±1 neighbour) in one step. Every other
    /// `Imm` leaf on the call is pinned to a distinct sentinel so the single match is unambiguous.
    #[test]
    fn finds_and_substitutes_a_logged_magic_value() {
        const MAGIC: u32 = 0xC0DE_1234;
        let prctl = SYSCALLS.iter().find(|d| d.name == "prctl").unwrap();
        let mut rng = Rng::new(7);
        let mut p = crate::prog::Prog::new();
        let mut args = crate::genr::generate_args(&mut rng, prctl, &[]);
        for (idx, av) in args.iter_mut().enumerate() {
            if let ArgValue::Imm(v) = av {
                *v = 1000 + idx as u64; // distinct sentinel: never 0, never MAGIC(±1)
            }
        }
        // prctl's args[1] is Int{32,..}; pin it to 0, the value we'll claim was compared to MAGIC.
        args[1] = ArgValue::Imm(0);
        p.calls.push(TypedCall { desc: prctl, args });

        let pairs = [(0u32, MAGIC)];
        let mutated = mutate_cmplog(&mut rng, &p, &pairs).expect("must find the magic-value match");
        let ArgValue::Imm(v) = mutated.calls[0].args[1] else {
            panic!("arg should stay Imm");
        };
        assert!(
            [MAGIC, MAGIC.wrapping_add(1), MAGIC.wrapping_sub(1)].contains(&(v as u32)),
            "expected MAGIC (±1), got {v:#x}"
        );
        // Every other arg (the sentinels) must be untouched — exactly one leaf was substituted.
        for (idx, av) in mutated.calls[0].args.iter().enumerate() {
            if idx != 1
                && let ArgValue::Imm(v) = av
            {
                assert_eq!(*v, 1000 + idx as u64, "sentinel at arg {idx} was disturbed");
            }
        }
        assert!(mutated.is_well_formed());
    }

    /// Same idea, but the matched value is embedded inside a `Buffer` at a known offset, not a
    /// top-level `Imm` — proves the byte-buffer scan path also works end to end. The surrounding
    /// bytes are chosen so no other offset/width in the buffer coincidentally matches, but a
    /// match at the true offset is legitimately found at more than one width (1/2/4 bytes all
    /// alias the same physical location) — so we only assert the match landed at the right
    /// offset and left everything outside it alone, not which exact width was chosen.
    #[test]
    fn finds_and_substitutes_a_magic_value_embedded_in_a_buffer() {
        const MAGIC: u32 = 0xDEAD_BEEF;
        let write = SYSCALLS.iter().find(|d| d.name == "write").unwrap();
        let mut rng = Rng::new(9);
        let mut p = crate::prog::Prog::new();
        let mut args = crate::genr::generate_args(&mut rng, write, &[]);
        // args[1] is Ptr{Buffer}; embed v0 = 0x11223344 (LE) uniquely at offset 2, surrounded by
        // bytes that never coincide with any 1/2/4-byte truncation of v0 or MAGIC.
        let buf = vec![0x71u8, 0x82, 0x44, 0x33, 0x22, 0x11, 0x93, 0xA4];
        args[1] = ArgValue::Ptr(Box::new(ArgValue::Bytes(buf.clone())));
        args[2] = ArgValue::Imm(buf.len() as u64);
        p.calls.push(TypedCall { desc: write, args });
        assert!(p.is_well_formed());

        let pairs = [(0x1122_3344u32, MAGIC)];
        let mutated = mutate_cmplog(&mut rng, &p, &pairs).expect("must find the embedded magic value");
        let ArgValue::Ptr(inner) = &mutated.calls[0].args[1] else {
            panic!("arg should stay Ptr");
        };
        let ArgValue::Bytes(b) = inner.as_ref() else {
            panic!("pointee should stay Bytes");
        };
        assert_eq!(&b[0..2], &buf[0..2], "bytes before the match must be untouched");
        assert_eq!(&b[6..8], &buf[6..8], "bytes after the match must be untouched");
        assert_ne!(&b[2..6], &buf[2..6], "the embedded value at offset 2 must have changed");
        assert!(mutated.is_well_formed());
    }

    /// No leaf in the program matches either side of any logged pair -> `None` (unrelated pairs
    /// shouldn't spuriously invent a "match").
    #[test]
    fn no_match_yields_none() {
        let close = SYSCALLS.iter().find(|d| d.name == "close").unwrap();
        let mut rng = Rng::new(3);
        let mut p = crate::prog::Prog::new();
        p.calls.push(TypedCall {
            desc: close,
            args: crate::genr::generate_args(&mut rng, close, &[]),
        });
        // close's only arg is Res(FD) — never a cmplog target — so no pair can ever match.
        let pairs = [(0xDEAD_BEEFu32, 0x1234_5678u32)];
        assert!(mutate_cmplog(&mut rng, &p, &pairs).is_none());
    }

    /// Property test across many generated programs and synthetic pairs derived from the
    /// program's own data: whenever a substitution is found, the result stays well-formed and
    /// keeps the same call/skeleton shape (only `Imm`/`Bytes` leaves may differ).
    #[test]
    fn mutate_cmplog_preserves_well_formedness_across_many_seeds() {
        for seed in 1..300u32 {
            let mut rng = Rng::new(seed);
            let base = generate(&mut rng);
            // Synthesize pairs from the base's own Imm leaves (guaranteed matches for some seeds)
            // plus a couple of arbitrary ones (guaranteed near-misses for others).
            let mut pairs = Vec::new();
            for c in &base.calls {
                for av in &c.args {
                    if let ArgValue::Imm(v) = av {
                        pairs.push((*v as u32, 0x1122_3344u32));
                    }
                }
            }
            pairs.push((rng.next(), rng.next()));
            if let Some(mutated) = mutate_cmplog(&mut rng, &base, &pairs) {
                assert!(mutated.is_well_formed(), "seed {seed}");
                assert_eq!(mutated.calls.len(), base.calls.len(), "seed {seed}");
                for (bc, mc) in base.calls.iter().zip(&mutated.calls) {
                    assert_eq!(bc.desc.name, mc.desc.name, "seed {seed}");
                }
            }
        }
    }
}
