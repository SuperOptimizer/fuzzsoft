//! Deterministic PRNG-driven generation of well-formed `Prog`s, with a resource pool threading
//! producers (openat/socket/pipe2 -> fd/sock) into consumers (read/ioctl/close). See
//! `docs/syzlang.md` §2.

use crate::dict::pick_dict_const;
use crate::lower::ptr_size_of;
use crate::prog::{ArgValue, MAX_CALLS, Prog, ResRef, TypedCall};
use crate::resource::{ResourceKind, kind_compat, seeds_for};
use crate::rng::Rng;
use crate::syscalls::SYSCALLS;
use crate::types::{ArgType, Field, LenSpec, SyscallDesc};

/// Chance (out of 100) that a fresh `Int`/`Flags` value is drawn from [`crate::dict`]'s curated
/// "interesting" constants instead of the type's own biased/curated generation — see `dict`'s
/// module doc for why this exists (cmplog needs real constants to already be in the corpus
/// before it has anything to substitute/log against). Applied at [`gen_arg_value`]'s `Int`/
/// `Flags` cases, so it reaches every description's scalar args uniformly, not just a hand-picked
/// subset.
pub(crate) const DICT_BIAS_PCT: u32 = 15;

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

/// A small number of hand-authored, real, bug-prone-subsystem call *sequences* (by `SyscallDesc`
/// name, including `$variant` suffixes). `generate()` occasionally builds one of these verbatim
/// (in order) instead of picking every call uniformly at random — this manufactures specific
/// deep chains (tmpfs-backed mmap lifecycle, epoll/eventfd wiring, socket option + connect,
/// double-close, pidfd duplication, ...) that are individually valuable but each too specific to
/// reliably assemble by chance even with the resource-threading bias below. Resource threading
/// *within* a recipe still goes through the normal `generate_args`/`pick_res` machinery (each
/// call sees every earlier call in the recipe as its `existing` pool), so the fd/sock/vma
/// produced by an early recipe call gets threaded into later recipe calls exactly like organic
/// generation — recipes only fix the call *order*, not the argument values.
pub static RECIPES: &[&[&str]] = &[
    // tmpfs-backed vma lifecycle: memfd growth -> map -> reprotect -> advise -> unmap.
    &[
        "memfd_create",
        "ftruncate64",
        "mmap2",
        "mprotect",
        "madvise",
        "munmap",
    ],
    // file lifecycle with a positioned write and an flag change before close.
    &["openat", "read", "pwrite64", "fcntl64$setfl", "close"],
    // TCP-ish socket lifecycle: option set, connect, option probe.
    &[
        "socket",
        "setsockopt$tcp_nodelay",
        "connect$inet",
        "getsockopt",
        "close",
    ],
    // listening socket lifecycle.
    &["socket", "bind$inet", "listen", "getsockname", "close"],
    // socketpair depth: both ends get used before either is torn down.
    &["socketpair", "sendmsg", "recvmsg", "shutdown"],
    // classic double-close / use-after-close signal via pipe2's two-fd OutArray.
    &["pipe2", "write", "read", "close", "close"],
    // epoll wiring: create the epoll fd, create something to watch, register it, wait.
    &[
        "epoll_create1",
        "eventfd2",
        "epoll_ctl",
        "epoll_pwait",
        "close",
    ],
    // fd-table depth: dup, query flags, lock, close.
    &["openat", "dup3", "fcntl64$getfd", "flock", "close"],
    // pidfd cross-process fd duplication.
    &["pidfd_open", "pidfd_getfd", "close"],
    // memfd + mmap + vectored IO against the same backing fd.
    &["memfd_create", "mmap2", "readv", "writev", "munmap"],
    // T2.1 wave 12: pipe-based data movement — two pipes so tee/splice see two distinct fds
    // instead of degenerating to a single pipe's own two ends every time.
    &["pipe2", "pipe2", "vmsplice", "tee", "splice", "close", "close"],
    // T2.1 wave 13: acquire a namespace fd, join it, then also try the flags-only unshare path.
    &["openat$ns", "setns", "unshare"],
    // T2.1 wave 14: add a key, then run it through the read-only keyctl ops before revoking it —
    // the classic add/describe/read/revoke key lifecycle.
    &["add_key", "keyctl$describe", "keyctl$read", "keyctl$revoke"],
    // T2.1 wave 15: cross-process memory access, both directions back to back.
    &["process_vm_writev", "process_vm_readv"],
];

fn find_desc(name: &str) -> Option<&'static SyscallDesc> {
    SYSCALLS.iter().find(|d| d.name == name)
}

/// Build a `Prog` by generating each call of `recipe` in order (via the normal
/// `generate_args`/`pick_res` path, so resource threading is organic, not forced). Returns `None`
/// if a name in `recipe` doesn't match any current `SyscallDesc` (defensive against a future
/// rename; `generate()` just falls back to uniform-random generation in that case).
pub(crate) fn generate_from_recipe(rng: &mut Rng, recipe: &[&str]) -> Option<Prog> {
    let mut calls: Vec<TypedCall> = Vec::with_capacity(recipe.len().min(MAX_CALLS));
    for name in recipe.iter().take(MAX_CALLS) {
        let desc = find_desc(name)?;
        let args = generate_args(rng, desc, &calls);
        calls.push(TypedCall { desc, args });
    }
    Some(Prog { calls })
}

/// Generate a fresh `Prog` of 1..=MAX_CALLS calls, each a random `SyscallDesc` from the starter
/// table with type-directed argument generation and resource threading against earlier calls.
///
/// 20% of the time, build one of `RECIPES` verbatim instead — a deliberately deep, real,
/// bug-prone-subsystem call chain (see `RECIPES`'s doc comment). The remaining 80% (or if the
/// chosen recipe somehow doesn't resolve) falls back to per-call `pick_desc_biased`, which itself
/// increasingly prefers descriptions that consume an already-live resource once the
/// program-under-construction has produced one — see that function's doc comment for why this is
/// what actually deepens organically-generated chains too.
pub fn generate(rng: &mut Rng) -> Prog {
    if rng.chance(20) {
        let recipe: &'static [&'static str] = RECIPES[rng.below(RECIPES.len())];
        if let Some(p) = generate_from_recipe(rng, recipe) {
            return p;
        }
    }
    let n = 1 + rng.below(MAX_CALLS);
    let mut calls: Vec<TypedCall> = Vec::with_capacity(n);
    for _ in 0..n {
        let desc = pick_desc_biased(rng, &calls);
        let args = generate_args(rng, desc, &calls);
        calls.push(TypedCall { desc, args });
    }
    Prog { calls }
}

pub fn pick_desc(rng: &mut Rng) -> &'static SyscallDesc {
    rng.pick(SYSCALLS)
}

/// Like `pick_desc`, but once the program-under-construction (`existing`) has produced at least
/// one live resource, 55% of the time prefer a `SyscallDesc` that actually *consumes* a
/// compatible resource kind from that pool over picking uniformly at random from the whole
/// table. This is the "generation bias" half of deepening chains (the other half is `RECIPES`
/// above and the pre-existing 70%-prefer-a-producer bias inside `pick_res`): without it, a
/// produced fd/sock/vma competes on equal footing with every scalar-only description (`prctl`,
/// `getcwd`, ...) for each subsequent call slot, so long dependent chains are diluted rather than
/// compounded. Falls back to `pick_desc` whenever the pool is empty, no compatible consumer
/// exists, or the 45% complement rolls — so untyped/no-resource descriptions still appear
/// regularly and every description in `SYSCALLS` remains reachable.
pub(crate) fn pick_desc_biased(rng: &mut Rng, existing: &[TypedCall]) -> &'static SyscallDesc {
    if !existing.is_empty() && rng.chance(55) {
        let pool = build_pool(existing);
        if !pool.is_empty() {
            let consumers: Vec<&'static SyscallDesc> = SYSCALLS
                .iter()
                .filter(|d| {
                    d.args.iter().any(|a| {
                        matches!(a, ArgType::Res(want) if pool.iter().any(|e| kind_compat(*want, e.kind)))
                    })
                })
                .collect();
            if !consumers.is_empty() {
                return consumers[rng.below(consumers.len())];
            }
        }
    }
    pick_desc(rng)
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
        ArgType::Int { bits, signed } => {
            // Dictionary bias: a fraction of the time, draw a real, cited kernel constant
            // (ioctl cmd, netlink type, errno, ...) instead of the usual 0/1/-1/boundary/random
            // pool — see `dict`'s module doc and `DICT_BIAS_PCT`.
            if rng.chance(DICT_BIAS_PCT) {
                ArgValue::Imm(mask_to_bits(pick_dict_const(rng) as u64, *bits))
            } else {
                ArgValue::Imm(gen_int(rng, *bits, *signed))
            }
        }
        ArgType::Flags { vals, bitmask } => {
            if rng.chance(DICT_BIAS_PCT) {
                let v = pick_dict_const(rng);
                // For a bitmask arg, OR the dictionary constant into a normal roll rather than
                // replacing it outright, so the description's own curated bits (e.g. O_CREAT)
                // usually still survive alongside the injected constant.
                let v = if *bitmask { v | gen_flags(rng, vals, *bitmask) } else { v };
                ArgValue::Imm(v as u64)
            } else {
                ArgValue::Imm(gen_flags(rng, vals, *bitmask) as u64)
            }
        }
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

pub(crate) fn mask_to_bits(v: u64, bits: u8) -> u64 {
    if bits >= 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    }
}

/// Curated "interesting" scalar values a mutator can swap into an `Int`/`Flags`/`Len` slot —
/// syzkaller's classic 0/1/-1/boundary/page-size set, masked to the field's actual bit width.
/// Shared with `mutate::mutate_interesting_int`.
pub(crate) const INTERESTING_INTS: &[i64] = &[
    0,
    1,
    2,
    -1,
    4096,             // PAGE_SIZE
    -4096,
    i32::MAX as i64,  // INT_MAX
    i32::MIN as i64,  // INT_MIN
    u16::MAX as i64,
];

pub(crate) fn pick_interesting_int(rng: &mut Rng, bits: u8) -> u64 {
    let v = *rng.pick(INTERESTING_INTS);
    mask_to_bits(v as u64, bits)
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
/// any), else fall back to a seed literal. Among compatible producers, 60% of the time prefer
/// the *most recently* produced one (`pool` is built in call order, so that's `compatible`'s
/// last entry) rather than picking uniformly — this is what actually makes multi-call chains
/// like `openat->read->close` or `socket->setsockopt->bind` form densely instead of scattering
/// references thinly across every producer seen so far in a long program.
pub(crate) fn pick_res(rng: &mut Rng, want: ResourceKind, pool: &[PoolEntry]) -> ResRef {
    if rng.chance(70) {
        let compatible: Vec<&PoolEntry> =
            pool.iter().filter(|e| kind_compat(want, e.kind)).collect();
        if !compatible.is_empty() {
            let e = if compatible.len() > 1 && rng.chance(60) {
                *compatible[compatible.len() - 1]
            } else {
                **rng.pick(&compatible)
            };
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

/// Prepend the fail_nth arming preamble — `openat$fail_nth` -> `write$fail_nth`, with the
/// second call's fd arg *guaranteed* (not left to `pick_res`'s probabilistic seed-vs-produced
/// choice) to thread from the first call's own `Produces::Ret(FD)` — onto `prog`. See
/// `docs/bug-finding.md`'s "FAULT INJECTION FIRST": this is the 2-call preamble a `--fail-inject`
/// generator bias prepends to a fraction of generated programs so the fuzzer can arm a specific
/// kernel allocation to fail before running the rest of the program against it.
///
/// Every pre-existing `ResRef::Produced` reference in `prog` is shifted forward by 2 call slots
/// (the preamble now occupies indices 0-1), and the combined program is truncated to `MAX_CALLS`
/// from the tail if it would otherwise overflow. Truncating only the tail is always safe: a
/// well-formed program's resource references only ever point strictly backward (see
/// `Prog::is_well_formed`), so no surviving call can have referenced one of the dropped ones.
pub fn prepend_fail_inject(rng: &mut Rng, prog: Prog) -> Prog {
    let openat_desc = SYSCALLS
        .iter()
        .find(|d| d.name == "openat$fail_nth")
        .expect("openat$fail_nth description must exist");
    let write_desc = SYSCALLS
        .iter()
        .find(|d| d.name == "write$fail_nth")
        .expect("write$fail_nth description must exist");

    let openat_call = TypedCall {
        desc: openat_desc,
        args: generate_args(rng, openat_desc, &[]),
    };

    let mut write_args = generate_args(rng, write_desc, std::slice::from_ref(&openat_call));
    write_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    });
    let write_call = TypedCall {
        desc: write_desc,
        args: write_args,
    };

    let mut calls: Vec<TypedCall> = Vec::with_capacity(2 + prog.calls.len());
    calls.push(openat_call);
    calls.push(write_call);
    for mut call in prog.calls {
        for av in &mut call.args {
            shift_produced_call_idx(av, 2);
        }
        calls.push(call);
    }
    calls.truncate(MAX_CALLS);
    Prog { calls }
}

/// Recursively add `shift` to every `ResRef::Produced`'s `call_idx` found in `av` — walks into
/// `Struct` fields and `Ptr` pointees since a future description could nest a `Res` arg inside
/// either (none of today's `SYSCALLS` do, but this stays correct if one ever does).
fn shift_produced_call_idx(av: &mut ArgValue, shift: u16) {
    match av {
        ArgValue::Res(ResRef::Produced { call_idx, .. }) => *call_idx += shift,
        ArgValue::Struct(fields) => {
            for f in fields.iter_mut() {
                shift_produced_call_idx(f, shift);
            }
        }
        ArgValue::Ptr(inner) => shift_produced_call_idx(inner, shift),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_recipe_resolves_to_real_descriptions_and_lowers_cleanly() {
        // Each RECIPES entry must name real, currently-existing SyscallDescs (a typo/rename would
        // otherwise silently fall back to uniform generation forever without ever failing a
        // test), and the resulting program must be well-formed and lowerable regardless of how
        // the RNG resolves each recipe call's non-recipe-fixed argument content.
        for (i, recipe) in RECIPES.iter().enumerate() {
            for seed in [1u32, 2, 3, 42, 12345] {
                let mut rng = Rng::new(seed.wrapping_add(i as u32 * 1000 + 1));
                let p = generate_from_recipe(&mut rng, recipe)
                    .unwrap_or_else(|| panic!("recipe {i} {recipe:?} failed to resolve"));
                assert_eq!(p.calls.len(), recipe.len());
                assert!(p.is_well_formed(), "recipe {i} {recipe:?} ill-formed");
                let _ = crate::lower::lower(&p, 0xA000_0000);
            }
        }
    }

    #[test]
    fn generate_sometimes_builds_a_recipe_verbatim() {
        // Across many seeds, `generate()`'s 20% recipe path must actually fire at least once
        // (call sequence matches one of RECIPES exactly) — otherwise the recipe mechanism would
        // be dead code that never contributes to the corpus.
        let mut saw_recipe = false;
        'seeds: for seed in 1..500u32 {
            let mut rng = Rng::new(seed);
            let p = generate(&mut rng);
            let names: Vec<&str> = p.calls.iter().map(|c| c.desc.name).collect();
            for recipe in RECIPES {
                if names == *recipe {
                    saw_recipe = true;
                    break 'seeds;
                }
            }
        }
        assert!(saw_recipe, "generate() never produced a verbatim recipe in 500 seeds");
    }

    #[test]
    fn pick_desc_biased_prefers_a_resource_consumer_when_pool_nonempty() {
        // Build a one-call pool that only produces FD (openat), then check that biased picking
        // returns an FD/SOCK-consuming description noticeably more often than plain uniform
        // `pick_desc` would (uniform draws a consumer roughly `consumers/total` of the time;
        // biased should draw one distinctly more often across many tries).
        let openat = SYSCALLS.iter().find(|d| d.name == "openat").unwrap();
        let mut rng = Rng::new(777);
        let existing = vec![TypedCall {
            desc: openat,
            args: generate_args(&mut rng, openat, &[]),
        }];

        let consumes_fd = |d: &SyscallDesc| {
            d.args
                .iter()
                .any(|a| matches!(a, ArgType::Res(k) if crate::resource::kind_compat(*k, crate::resource::FD)))
        };

        let mut biased_hits = 0u32;
        let mut uniform_hits = 0u32;
        const TRIES: u32 = 2000;
        for _ in 0..TRIES {
            if consumes_fd(pick_desc_biased(&mut rng, &existing)) {
                biased_hits += 1;
            }
            if consumes_fd(pick_desc(&mut rng)) {
                uniform_hits += 1;
            }
        }
        assert!(
            biased_hits > uniform_hits,
            "biased picker ({biased_hits}/{TRIES}) should beat uniform ({uniform_hits}/{TRIES})"
        );
    }

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

    /// `prepend_fail_inject` must produce a well-formed program whose first two calls are the
    /// fail_nth preamble, with call 1's fd arg *guaranteed* to thread from call 0 (a `Reg` fixup,
    /// not a seed literal), and the whole thing must still lower cleanly to the fixed wire size.
    #[test]
    fn prepend_fail_inject_builds_a_well_formed_armed_preamble_that_lowers_cleanly() {
        for seed in [1u32, 2, 3, 42, 999] {
            let mut rng = Rng::new(seed);
            let base = generate(&mut rng);
            let armed = prepend_fail_inject(&mut rng, base);

            assert!(armed.calls.len() >= 2);
            assert!(armed.calls.len() <= MAX_CALLS);
            assert_eq!(armed.calls[0].desc.name, "openat$fail_nth");
            assert_eq!(armed.calls[1].desc.name, "write$fail_nth");
            assert!(armed.is_well_formed(), "seed {seed}: armed program ill-formed");
            assert_eq!(
                armed.calls[1].args[0],
                ArgValue::Res(ResRef::Produced {
                    call_idx: 0,
                    slot: 0
                }),
                "seed {seed}: write$fail_nth's fd arg must thread from openat$fail_nth"
            );

            let lowered = crate::lower::lower(&armed, 0xA000_0000);
            let has_expected_fixup = lowered.fixups.iter().any(|f| {
                f.dst_call == 1
                    && f.dst_arg == 0
                    && matches!(f.src, crate::lower::FixupSrc::Reg(0))
            });
            assert!(
                has_expected_fixup,
                "seed {seed}: expected a Reg(0) fixup threading call 0's fd into call 1's fd arg"
            );

            let wire = crate::lower::to_wire(&lowered);
            assert_eq!(wire.len(), crate::lower::WIRE_WORDS);
        }
    }

    /// Prepending onto an already-`MAX_CALLS`-long base program must truncate the tail (never
    /// panic, never drop the preamble itself) and stay well-formed.
    #[test]
    fn prepend_fail_inject_truncates_a_full_length_base_program() {
        let mut rng = Rng::new(5);
        let mut base = generate(&mut rng);
        let mut tries = 0;
        while base.calls.len() < MAX_CALLS && tries < 500 {
            base = generate(&mut rng);
            tries += 1;
        }
        assert_eq!(base.calls.len(), MAX_CALLS, "never sampled a full-length base program");

        let armed = prepend_fail_inject(&mut rng, base);
        assert_eq!(armed.calls.len(), MAX_CALLS);
        assert_eq!(armed.calls[0].desc.name, "openat$fail_nth");
        assert_eq!(armed.calls[1].desc.name, "write$fail_nth");
        assert!(armed.is_well_formed());
        let _ = crate::lower::lower(&armed, 0xA000_0000);
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

    /// Dictionary bias validation (a): across a generated corpus, `Int`/`Flags`-typed args carry
    /// a dictionary constant at a rate broadly consistent with `DICT_BIAS_PCT` — proves the
    /// `dict` wiring in `gen_arg_value` actually fires during ordinary generation, not just when
    /// called directly.
    #[test]
    fn generated_corpus_carries_dictionary_constants_at_a_measurable_rate() {
        use crate::dict::DICTIONARY_GROUPS;
        let is_dict_value = |v: u32| DICTIONARY_GROUPS.iter().any(|g| g.contains(&v));

        let mut total_scalar_args = 0u64;
        let mut dict_hits = 0u64;
        let mut rng = Rng::new(4242);
        for _ in 0..3000 {
            let p = generate(&mut rng);
            for c in &p.calls {
                for (aty, av) in c.desc.args.iter().zip(&c.args) {
                    let is_scalar_slot = matches!(aty, ArgType::Int { .. } | ArgType::Flags { .. });
                    if !is_scalar_slot {
                        continue;
                    }
                    let ArgValue::Imm(v) = av else { continue };
                    total_scalar_args += 1;
                    if is_dict_value(*v as u32) {
                        dict_hits += 1;
                    }
                }
            }
        }
        assert!(total_scalar_args > 1000, "too few scalar args sampled");
        // Some of these "hits" are coincidental (e.g. plain `gen_int`'s own 0/1/boundary pool
        // overlapping a dictionary value), so this only checks for a clearly nonzero, measurable
        // rate — not a tight match to DICT_BIAS_PCT.
        let rate = dict_hits as f64 / total_scalar_args as f64;
        assert!(
            rate > 0.02,
            "dictionary constants appeared in only {rate:.4} of {total_scalar_args} scalar args"
        );
    }

    /// Dictionary bias validation, `Flags` bitmask case specifically: an injected dictionary
    /// constant must survive as a set bit even when OR'd with the description's own curated
    /// `vals` roll (not silently lost/masked away).
    #[test]
    fn dict_bias_ors_into_bitmask_flags_without_losing_the_injected_bit() {
        // openat's 3rd arg is `Flags{vals: OPEN_FLAGS, bitmask: true}`.
        let openat = SYSCALLS.iter().find(|d| d.name == "openat").unwrap();
        let mut rng = Rng::new(1);
        let mut saw_a_dict_only_bit = false;
        for _ in 0..3000 {
            let args = generate_args(&mut rng, openat, &[]);
            let ArgValue::Imm(v) = args[2] else { continue };
            let v = v as u32;
            // A bit is "dict-only" if it's set in `v` but not producible by ORing any subset of
            // OPEN_FLAGS's own vals table.
            let openat_flags_union: u32 = match &openat.args[2] {
                ArgType::Flags { vals, .. } => vals.iter().fold(0u32, |a, b| a | b),
                _ => 0,
            };
            if v & !openat_flags_union != 0 {
                saw_a_dict_only_bit = true;
                break;
            }
        }
        assert!(
            saw_a_dict_only_bit,
            "never observed a dictionary-injected bit outside openat's own OPEN_FLAGS union"
        );
    }
}
