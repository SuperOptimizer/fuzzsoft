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

/// Chance (out of 100) that, when generating/mutating a `Res`-typed arg in a call that has
/// *already* bound an earlier `Res` arg to some specific live resource (an "anchor" — see
/// [`same_call_anchors`]), this arg instead wires to a *different* live resource of that exact
/// same specific kind (a "sibling"), when one exists — rather than falling through to
/// [`pick_res`]'s undiscriminating whole-kind pool. This is the T2.4 same-kind cross-reference /
/// cycle-building bias (`docs/roadmap.md` T2.4, `docs/bug-finding.md`'s epoll loop-check
/// overflow): without it, a call shaped like `epoll_ctl(epfd, op, fd, event)` treats every fd
/// producer in the pool as fungible, so an epoll_create1-produced epfd's *target* fd argument is
/// diluted across every other unrelated fd producer (openat/socket/pipe2/...) instead of
/// preferentially wiring to another live epoll instance — the shape needed for epoll->epoll
/// nesting (and, wired right across several such calls, a genuine containment cycle). Kept a
/// bias, not mandatory, so ordinary "wire to an unrelated fd" diversity is preserved. See
/// [`pick_res_biased`].
pub(crate) const CROSS_REF_BIAS_PCT: u32 = 55;

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
    // T2.4: three same-kind epoll instances back to back, then three epoll_ctl calls. This
    // doesn't by itself force the exact A->B->C->A containment cycle (call *values* are still
    // resolved organically by generate_args/pick_res_biased per the RECIPES doc comment above),
    // but it puts >=2 sibling epoll instances in the pool *before* any epoll_ctl call generates,
    // which is the precondition `pick_res_biased`'s cross-reference bias needs to have anything
    // to wire epoll_ctl's target-fd toward. Without this shape, getting >=3 epoll_create1 calls
    // into one organically-generated 8-call program is vanishingly rare (measured ~0 in T4.2's
    // 300k-case campaign corpus) since epoll_create1 competes uniformly with 90+ other
    // descriptions. See docs/roadmap.md T2.4 / docs/bug-finding.md's epoll loop-check overflow.
    &[
        "epoll_create1",
        "epoll_create1",
        "epoll_create1",
        "epoll_ctl",
        "epoll_ctl",
        "epoll_ctl",
    ],
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
/// `RING_FORCE_PCT` of the time (checked first), force-build a complete, closed same-kind
/// resource ring via `build_epoll_ring` — T2.4.5's directed cycle-forcing recipe (see that
/// function's doc comment): unlike everything else in this function, this fixes actual argument
/// *values*/resource wiring, not just call order, so the CVE's exact minimal trigger shape
/// (a closed N-node epoll containment cycle) assembles deterministically rather than by chance.
///
/// Otherwise, 20% of the time, build one of `RECIPES` verbatim instead — a deliberately deep,
/// real, bug-prone-subsystem call chain (see `RECIPES`'s doc comment). The remaining fraction (or
/// if the chosen recipe/ring somehow doesn't resolve) falls back to per-call `pick_desc_biased`,
/// which itself increasingly prefers descriptions that consume an already-live resource once the
/// program-under-construction has produced one — see that function's doc comment for why this is
/// what actually deepens organically-generated chains too.
pub fn generate(rng: &mut Rng) -> Prog {
    if rng.chance(RING_FORCE_PCT) {
        let n = 3 + rng.below(MAX_CALLS / 2 - 2); // 3..=MAX_CALLS/2 (3 or 4 today)
        if let Some(p) = build_epoll_ring(rng, n) {
            return p;
        }
    }
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
            // Special-cased (rather than falling through to `gen_arg_value`) so the
            // cross-reference bias can see what this same call has already bound to an earlier
            // `Res` arg — see `CROSS_REF_BIAS_PCT`/`pick_res_biased`.
            ArgType::Res(kind) => {
                let anchors = same_call_anchors(&pool, &args, None);
                ArgValue::Res(pick_res_biased(rng, *kind, &pool, &anchors))
            }
            _ => gen_arg_value(rng, aty, &pool),
        };
        args.push(av);
    }
    args
}

/// The live resources this same call has already bound in an earlier `Res` arg (excluding
/// `exclude_idx`, if given — used when re-rolling one arg of an already-fully-populated call
/// during mutation, so the arg's own current/stale binding doesn't count as its own anchor).
/// These are the "anchors" [`pick_res_biased`] tries to wire a *sibling* same-kind resource
/// against — e.g. once `epoll_ctl`'s `epfd` arg (index 0) is bound to a live `epoll_create1`
/// output, that becomes an anchor when generating/mutating the `fd` arg (index 2) right after.
pub(crate) fn same_call_anchors(
    pool: &[PoolEntry],
    call_args: &[ArgValue],
    exclude_idx: Option<usize>,
) -> Vec<PoolEntry> {
    call_args
        .iter()
        .enumerate()
        .filter(|(i, _)| exclude_idx != Some(*i))
        .filter_map(|(_, av)| match av {
            ArgValue::Res(ResRef::Produced { call_idx, slot }) => pool
                .iter()
                .find(|e| e.call_idx == *call_idx && e.slot == *slot)
                .copied(),
            _ => None,
        })
        .collect()
}

/// Like [`pick_res`], but first tries the T2.4 same-kind cross-reference bias: if `anchors` is
/// nonempty and the bias roll (`CROSS_REF_BIAS_PCT`) fires, look for a live resource in `pool`
/// that shares an anchor's *exact* kind (not just `kind_compat`-compatible with `want` — the
/// point is connecting two resources of the *same specific* kind, e.g. two epoll instances, not
/// just two arbitrary fds) and isn't the anchor itself (so a call never wires an arg to the
/// literal same resource its own sibling arg already used — e.g. `epoll_ctl(A, ADD, A)`, which
/// the kernel rejects anyway). Falls back to plain `pick_res` whenever no anchor's kind is even
/// `want`-compatible, no such sibling exists, or the bias roll doesn't fire — so ordinary
/// resource threading is unaffected for the common case (calls with 0 or 1 `Res` arg, which is
/// most of `SYSCALLS`).
pub(crate) fn pick_res_biased(
    rng: &mut Rng,
    want: ResourceKind,
    pool: &[PoolEntry],
    anchors: &[PoolEntry],
) -> ResRef {
    if !anchors.is_empty() && rng.chance(CROSS_REF_BIAS_PCT) {
        for anchor in anchors {
            if !kind_compat(want, anchor.kind) {
                continue;
            }
            let siblings: Vec<&PoolEntry> = pool
                .iter()
                .filter(|e| {
                    e.kind == anchor.kind
                        && !(e.call_idx == anchor.call_idx && e.slot == anchor.slot)
                })
                .collect();
            if !siblings.is_empty() {
                let e = **rng.pick(&siblings);
                return ResRef::Produced {
                    call_idx: e.call_idx,
                    slot: e.slot,
                };
            }
        }
    }
    pick_res(rng, want, pool)
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

/// Chance (out of 100) that `generate()` force-builds a complete, closed same-kind resource ring
/// (see [`build_same_kind_ring`]/[`build_epoll_ring`]) instead of an ordinary RECIPES draw or
/// uniform-random generation. T2.4.5 (`docs/roadmap.md`): T2.4 made an epoll->epoll link
/// expressible and even let a closed cycle assemble *by chance* occasionally (~0.7% of
/// programs, `t24_probe::probe_full_cycle_rate`), but the CVE's exact minimal trigger — a
/// specific, fully-closed >=3-node ring where *every* producer and *every* link lands exactly on
/// its ring neighbor — needs all of that wiring to line up simultaneously, which compounds down
/// to ~0.004% under the probabilistic bias alone. This constant instead *forces* the whole ring's
/// wiring outright (mirrors `prepend_fail_inject`'s style: fixed argument values/resource wiring,
/// not just call order), while staying a modest bias so ordinary RECIPES/uniform-random diversity
/// remains the common case.
pub(crate) const RING_FORCE_PCT: u32 = 10;

/// How a `linker` call's own `Res` args wire the directed edge between two ring nodes, plus any
/// extra forced values, for [`build_same_kind_ring`]. Bundled into one struct (rather than
/// several loose parameters) so the builder function stays under clippy's arg-count limit.
pub(crate) struct RingLinkSpec<'a> {
    /// Index of the linker's `Res` arg wired to the ring *predecessor* (node `i`).
    pub(crate) anchor_arg_idx: usize,
    /// Index of the linker's `Res` arg wired to the ring *successor* (node `(i+1) mod n`).
    pub(crate) target_arg_idx: usize,
    /// Additional `(arg_idx, ArgValue)` overrides applied to *every* linker call after the ring
    /// wiring (e.g. forcing epoll_ctl's `op` arg to `EPOLL_CTL_ADD` rather than leaving it to
    /// land on DEL/MOD by chance) — the "fixing argument values, not just call order" recipe
    /// style `prepend_fail_inject` established.
    pub(crate) fixed_args: &'a [(usize, ArgValue)],
    /// If given, guarantees the linker's arg at that index (which must be an `ArgType::Ptr`) is
    /// never the nullable-NULL case — needed for epoll_ctl's `event` arg, since `EPOLL_CTL_ADD`
    /// with a NULL event fails `copy_from_user` before ever reaching the vulnerable insert path,
    /// which would silently break the ring 20% of the time (`ArgType::Ptr`'s own nullable roll)
    /// if left to chance.
    pub(crate) force_nonnull_ptr_idx: Option<usize>,
}

/// Build a `Prog` that force-closes a complete, directed N-node ring over some same-kind
/// resource: `n` calls to `producer` (each of which must `Produces::Ret(kind)` exactly one
/// resource per call, at slot 0), followed by `n` calls to `linker`, with the `i`-th linker
/// call's `link.anchor_arg_idx` argument wired to producer call `i`'s output and its
/// `link.target_arg_idx` argument wired to producer call `(i+1) mod n`'s output — i.e.
/// `linker(P_i, ..., P_{(i+1) mod n}, ...)` for every `i`, closing the ring
/// `P0 -> P1 -> ... -> P(n-1) -> P0`. General over *any* same-kind resource graph (not
/// epoll-specific): the caller supplies the producer/linker description names and which of the
/// linker's own `Res` arg slots are the "from"/"to" ends of the edge (plus any forced values, via
/// `link` — see [`RingLinkSpec`]). Every other argument (both producer and linker) is left to
/// ordinary `generate_args`/`pick_res_biased`, so only the ring's own shape is forced, not the
/// whole program.
///
/// Returns `None` if `producer`/`linker` don't name a live `SyscallDesc`, if either arg index is
/// out of range for `linker`, or if `n < 2` or `2*n > MAX_CALLS` (the caller should clamp `n`
/// first; this is just a defensive backstop matching `generate_from_recipe`'s style).
pub(crate) fn build_same_kind_ring(
    rng: &mut Rng,
    producer_name: &str,
    linker_name: &str,
    link: &RingLinkSpec,
    n: usize,
) -> Option<Prog> {
    if n < 2 || 2 * n > MAX_CALLS {
        return None;
    }
    let producer_desc = find_desc(producer_name)?;
    let linker_desc = find_desc(linker_name)?;
    if link.anchor_arg_idx >= linker_desc.args.len() || link.target_arg_idx >= linker_desc.args.len()
    {
        return None;
    }

    let mut calls: Vec<TypedCall> = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let args = generate_args(rng, producer_desc, &calls);
        calls.push(TypedCall {
            desc: producer_desc,
            args,
        });
    }
    for i in 0..n {
        let mut args = generate_args(rng, linker_desc, &calls);
        args[link.anchor_arg_idx] = ArgValue::Res(ResRef::Produced {
            call_idx: i as u16,
            slot: 0,
        });
        args[link.target_arg_idx] = ArgValue::Res(ResRef::Produced {
            call_idx: ((i + 1) % n) as u16,
            slot: 0,
        });
        for (idx, val) in link.fixed_args {
            args[*idx] = val.clone();
        }
        if let Some(idx) = link.force_nonnull_ptr_idx
            && let ArgType::Ptr { inner, .. } = &linker_desc.args[idx]
        {
            args[idx] = ArgValue::Ptr(Box::new(gen_arg_value(rng, inner, &build_pool(&calls))));
        }
        calls.push(TypedCall {
            desc: linker_desc,
            args,
        });
    }
    Some(Prog { calls })
}

/// The concrete T2.4.5 deliverable: force-build a closed N-node `epoll_create1` ring —
/// `epoll_create1`x`n` -> E0..E(n-1), then `epoll_ctl(E_i, EPOLL_CTL_ADD, E_{(i+1) mod n}, event)`
/// for every `i` — the exact minimal shape that overflows `ep_loop_check_proc`'s nesting-depth
/// counter (`docs/bug-finding.md`'s epoll loop-check CVE / `scripts/cve-epoll-loop-bug.patch`).
/// `epoll_ctl`'s args are `[Res(EPOLL) epfd, Flags op, Res(FD) fd, Ptr event]` (see
/// `syscalls.rs`), so `anchor_arg_idx=0` wires `epfd` to the ring predecessor and
/// `target_arg_idx=2` wires the target `fd` to the ring successor; `fixed_args` forces `op`
/// (index 1) to `EPOLL_CTL_ADD=1` so every link actually attempts an insert (an
/// organically-rolled DEL/MOD wouldn't reach `ep_loop_check_proc` at all), and
/// `force_nonnull_ptr_idx=Some(3)` guarantees `event` (index 3) is never NULL, since ADD requires
/// a real event struct to pass `copy_from_user` before reaching the loop check.
///
/// `n` is clamped to `[3, MAX_CALLS/2]`: 3 is the CVE's documented minimal cycle length (see
/// `scripts/cve-epoll-loop-seed.prog`'s 3-epoll_create1/3-epoll_ctl shape), and `MAX_CALLS/2`
/// (4, since `MAX_CALLS=8`) is the largest ring `2*n` calls can fit in one program — `n=5` (10
/// calls) would overflow `MAX_CALLS`, so it's capped down to 4 rather than failing outright.
pub(crate) fn build_epoll_ring(rng: &mut Rng, n: usize) -> Option<Prog> {
    let n = n.clamp(3, MAX_CALLS / 2);
    const EPOLL_CTL_ADD: u64 = 1;
    let link = RingLinkSpec {
        anchor_arg_idx: 0,
        target_arg_idx: 2,
        fixed_args: &[(1, ArgValue::Imm(EPOLL_CTL_ADD))],
        force_nonnull_ptr_idx: Some(3),
    };
    build_same_kind_ring(rng, "epoll_create1", "epoll_ctl", &link, n)
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

    /// Returns `true` iff `p` contains an `epoll_ctl` call whose target-fd arg (index 2) is a
    /// `Produced` reference to a live `epoll_create1` call — i.e. a genuine epoll->epoll
    /// cross-reference, the shape the epoll loop-check overflow (docs/bug-finding.md) needs.
    fn has_epoll_into_epoll_link(p: &Prog) -> bool {
        p.calls.iter().any(|c| {
            c.desc.name == "epoll_ctl"
                && matches!(
                    c.args[2],
                    ArgValue::Res(ResRef::Produced { call_idx, .. })
                        if p.calls[call_idx as usize].desc.name == "epoll_create1"
                )
        })
    }

    /// T2.4 minimum deliverable, tier (a): the generator must be able to emit a genuine
    /// epoll->epoll cross-reference (an `epoll_ctl` whose target-fd binds to a live
    /// `epoll_create1` output) within a small, fixed seed budget — before this change, T4.2's
    /// 300k-case corpus measurement found this shape *zero* times organically.
    #[test]
    fn generator_can_emit_an_epoll_into_epoll_epoll_ctl_within_n_seeds() {
        const N: u32 = 200;
        let mut found = false;
        for seed in 1..=N {
            let mut rng = Rng::new(seed);
            let p = generate(&mut rng);
            if has_epoll_into_epoll_link(&p) {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "generate() never produced an epoll->epoll epoll_ctl link within {N} seeds"
        );
    }

    /// T2.4 quantification: across a large generated sample, measure what fraction of programs
    /// contain a genuine epoll->epoll link. T4.2's real-campaign corpus measured this at ~0/18646
    /// (0%) before this change. This asserts a clearly-nonzero, measurable rate — the exact
    /// number is reported (not hardcoded to a tight bound) since it depends on the RECIPES/bias
    /// tuning constants and shouldn't be pinned brittlely.
    #[test]
    fn measurable_fraction_of_generated_programs_contain_an_epoll_into_epoll_link() {
        const N: u32 = 5000;
        let mut hits = 0u32;
        let mut three_plus_epoll = 0u32;
        for seed in 1..=N {
            let mut rng = Rng::new(seed + 1_000_000); // disjoint seed space from other tests
            let p = generate(&mut rng);
            if has_epoll_into_epoll_link(&p) {
                hits += 1;
            }
            let epoll_count = p
                .calls
                .iter()
                .filter(|c| c.desc.name == "epoll_create1")
                .count();
            if epoll_count >= 3 {
                three_plus_epoll += 1;
            }
        }
        let rate = f64::from(hits) / f64::from(N);
        eprintln!(
            "T2.4: {hits}/{N} ({:.2}%) generated programs contain an epoll->epoll link; \
             {three_plus_epoll}/{N} have >=3 live epoll_create1 calls",
            rate * 100.0
        );
        assert!(
            hits > 0,
            "expected a measurably nonzero rate of epoll->epoll links, got 0/{N}"
        );
    }

    /// The cross-reference bias's core claim, isolated from the epoll-specific recipe: given a
    /// call-local anchor bound to one of several same-kind (`EPOLL`) pool entries, `pick_res_biased`
    /// must noticeably prefer a *different* same-kind sibling over uniform `pick_res`'s
    /// undiscriminating whole-`FD`-kind pool.
    #[test]
    fn pick_res_biased_prefers_a_same_kind_sibling_over_the_anchor_itself() {
        use crate::resource::{EPOLL, FD};
        // Pool: three EPOLL entries (calls 0,1,2) and one unrelated FD entry (call 3, e.g.
        // openat) — mimics "epoll_create1 x3, openat" before an epoll_ctl call.
        let pool = vec![
            PoolEntry {
                call_idx: 0,
                slot: 0,
                kind: EPOLL,
            },
            PoolEntry {
                call_idx: 1,
                slot: 0,
                kind: EPOLL,
            },
            PoolEntry {
                call_idx: 2,
                slot: 0,
                kind: EPOLL,
            },
            PoolEntry {
                call_idx: 3,
                slot: 0,
                kind: FD,
            },
        ];
        // Anchor: epfd already bound to call 0's epoll instance.
        let anchors = vec![pool[0]];

        let mut rng = Rng::new(4242);
        let mut sibling_hits = 0u32;
        let mut anchor_or_other_hits = 0u32;
        const TRIES: u32 = 2000;
        for _ in 0..TRIES {
            match pick_res_biased(&mut rng, FD, &pool, &anchors) {
                ResRef::Produced { call_idx, .. } if call_idx == 1 || call_idx == 2 => {
                    sibling_hits += 1;
                }
                // Anchor (call 0) or the unrelated FD (call 3) or a seed literal: still possible
                // via the plain `pick_res` fallback whenever the bias roll doesn't fire — the
                // self-loop *avoidance* only applies within the biased branch itself, not the
                // fallback, so this isn't a bug, just the complement of the bias.
                _ => anchor_or_other_hits += 1,
            }
        }
        assert!(
            sibling_hits > anchor_or_other_hits,
            "expected the same-kind sibling bias to dominate: {sibling_hits} sibling hits vs \
             {anchor_or_other_hits} anchor/other/seed hits over {TRIES} tries"
        );
    }

    /// Without an anchor (e.g. generating the *first* `Res` arg of a call), `pick_res_biased`
    /// must behave exactly like plain `pick_res` — the bias only ever engages once a same-call
    /// sibling binding exists.
    #[test]
    fn pick_res_biased_matches_plain_pick_res_with_no_anchors() {
        use crate::resource::FD;
        let pool = vec![PoolEntry {
            call_idx: 0,
            slot: 0,
            kind: FD,
        }];
        let mut rng_a = Rng::new(9);
        let mut rng_b = Rng::new(9);
        for _ in 0..500 {
            assert_eq!(
                pick_res_biased(&mut rng_a, FD, &pool, &[]),
                pick_res(&mut rng_b, FD, &pool)
            );
        }
    }

    // ---- T2.4.5: ring-forcing recipe (build_same_kind_ring / build_epoll_ring) ----

    /// Returns `Some(order)` iff `p` is *exactly* the shape `build_epoll_ring(n)` produces: the
    /// first `n` calls are all `epoll_create1`, the remaining calls are all `epoll_ctl` with
    /// op == EPOLL_CTL_ADD (1), a non-NULL event, and epfd/target-fd wiring that forms a single
    /// closed directed ring over the `n` epoll_create1 calls (every node has out-degree and
    /// in-degree exactly 1) — `order` is the ring's node order starting from call 0. Used both to
    /// check `build_epoll_ring`'s own output and to detect when `generate()`'s ring-force path
    /// fired (as opposed to RECIPES/uniform-random happening to look similar).
    fn exact_closed_epoll_ring(p: &Prog, n: usize) -> bool {
        if p.calls.len() != 2 * n {
            return false;
        }
        for c in &p.calls[..n] {
            if c.desc.name != "epoll_create1" {
                return false;
            }
        }
        let mut next: Vec<Option<usize>> = vec![None; n];
        for (j, c) in p.calls[n..].iter().enumerate() {
            if c.desc.name != "epoll_ctl" {
                return false;
            }
            let ArgValue::Res(ResRef::Produced { call_idx: epfd_idx, .. }) = c.args[0] else {
                return false;
            };
            let ArgValue::Imm(op) = c.args[1] else {
                return false;
            };
            if op != 1 {
                return false; // must be EPOLL_CTL_ADD
            }
            let ArgValue::Res(ResRef::Produced { call_idx: fd_idx, .. }) = c.args[2] else {
                return false;
            };
            if matches!(c.args[3], ArgValue::Imm(0)) {
                return false; // event must be non-NULL
            }
            let epfd_idx = epfd_idx as usize;
            let fd_idx = fd_idx as usize;
            if epfd_idx >= n || fd_idx >= n {
                return false;
            }
            // The linker call at position j (0-indexed among the n linker calls) is expected to
            // wire epoll_create1 call `j` -> `(j+1) mod n`, but only the *shape* (a single closed
            // ring covering all n nodes) is asserted here, not the exact call order, since that's
            // exactly what this function independently re-derives via `next`.
            let _ = j;
            if next[epfd_idx].is_some() {
                return false; // out-degree > 1: not a simple ring
            }
            next[epfd_idx] = Some(fd_idx);
        }
        // Walk the ring starting at node 0 and confirm it visits every node exactly once before
        // returning to 0.
        let mut visited = vec![false; n];
        let mut cur = 0usize;
        for _ in 0..n {
            if visited[cur] {
                return false;
            }
            visited[cur] = true;
            let Some(nxt) = next[cur] else {
                return false;
            };
            cur = nxt;
        }
        cur == 0 && visited.iter().all(|&v| v)
    }

    /// `build_epoll_ring` must, for every supported ring size (3 and 4 — the range `[3, MAX_CALLS
    /// / 2]`), produce a program that is well-formed, lowers cleanly to the fixed wire size, and
    /// is *exactly* a closed N-node epoll containment ring per `exact_closed_epoll_ring` — not
    /// just "contains an epoll link somewhere", the full CVE-minimal shape every time.
    #[test]
    fn build_epoll_ring_closes_an_exact_n_node_cycle_and_lowers_cleanly() {
        for n in [3usize, 4] {
            for seed in [1u32, 2, 3, 42, 12345, 999999] {
                let mut rng = Rng::new(seed.wrapping_add(n as u32 * 7919));
                let p = build_epoll_ring(&mut rng, n)
                    .unwrap_or_else(|| panic!("build_epoll_ring({n}) returned None (seed {seed})"));
                assert_eq!(p.calls.len(), 2 * n);
                assert!(p.is_well_formed(), "n={n} seed={seed}: ring program ill-formed");
                assert!(
                    exact_closed_epoll_ring(&p, n),
                    "n={n} seed={seed}: build_epoll_ring did not close an exact {n}-node ring: {:?}",
                    p.calls.iter().map(|c| c.desc.name).collect::<Vec<_>>()
                );
                let lowered = crate::lower::lower(&p, 0xA000_0000);
                let wire = crate::lower::to_wire(&lowered);
                assert_eq!(wire.len(), crate::lower::WIRE_WORDS);
            }
        }
    }

    /// `build_epoll_ring` must ask for at least the CVE's documented minimal cycle length (n=3):
    /// n=2 is a valid ring shape-wise but not the target trigger, so the clamp floor matters.
    /// Also checks the clamp ceiling (`n=5`, which would need 10 calls, gets capped to 4).
    #[test]
    fn build_epoll_ring_clamps_n_into_the_max_calls_budget() {
        let mut rng = Rng::new(1);
        let p_small = build_epoll_ring(&mut rng, 1).expect("clamped to 3");
        assert_eq!(p_small.calls.len(), 6, "n=1 should clamp up to 3 (6 calls)");
        let p_big = build_epoll_ring(&mut rng, 5).expect("clamped to 4");
        assert_eq!(p_big.calls.len(), 8, "n=5 should clamp down to 4 (8 calls, MAX_CALLS)");
        assert!(p_big.calls.len() <= MAX_CALLS);
    }

    /// `build_same_kind_ring`'s defensive backstops: an unknown producer/linker name, an
    /// out-of-range arg index, or an `n` that can't fit `2*n` calls in `MAX_CALLS` must all
    /// return `None` rather than panicking.
    #[test]
    fn build_same_kind_ring_rejects_bad_input_defensively() {
        let mut rng = Rng::new(1);
        let link = RingLinkSpec {
            anchor_arg_idx: 0,
            target_arg_idx: 2,
            fixed_args: &[],
            force_nonnull_ptr_idx: None,
        };
        assert!(build_same_kind_ring(&mut rng, "no_such_syscall", "epoll_ctl", &link, 3).is_none());
        assert!(
            build_same_kind_ring(&mut rng, "epoll_create1", "no_such_syscall", &link, 3).is_none()
        );
        let bad_anchor = RingLinkSpec {
            anchor_arg_idx: 99,
            target_arg_idx: 2,
            fixed_args: &[],
            force_nonnull_ptr_idx: None,
        };
        assert!(
            build_same_kind_ring(&mut rng, "epoll_create1", "epoll_ctl", &bad_anchor, 3).is_none(),
            "out-of-range anchor_arg_idx must fail"
        );
        assert!(
            build_same_kind_ring(&mut rng, "epoll_create1", "epoll_ctl", &link, 5).is_none(),
            "n=5 needs 10 calls > MAX_CALLS and must fail (unclamped API)"
        );
    }

    /// T2.4.5's core engagement claim: across many seeds, `generate()`'s `RING_FORCE_PCT` path
    /// must actually fire at a measurable, roughly-proportional rate — i.e. this isn't dead code,
    /// and it isn't so rare that it wouldn't matter in a real campaign. Checked against a wide
    /// tolerance band (not tightly pinned to `RING_FORCE_PCT`) since RECIPES/uniform-random could
    /// coincidentally also produce this exact shape a small extra fraction of the time.
    #[test]
    fn generate_engages_the_ring_force_bias_at_a_measurable_rate() {
        const N: u32 = 20_000;
        let mut hits = 0u32;
        for seed in 1..=N {
            let mut rng = Rng::new(seed.wrapping_mul(2_654_435_761).wrapping_add(11));
            let p = generate(&mut rng);
            let n = p.calls.len() / 2;
            if (3..=4).contains(&n) && exact_closed_epoll_ring(&p, n) {
                hits += 1;
            }
        }
        let rate = f64::from(hits) / f64::from(N);
        eprintln!(
            "T2.4.5: {hits}/{N} ({:.2}%) generate() calls produced an exact closed epoll ring \
             (RING_FORCE_PCT={RING_FORCE_PCT})",
            rate * 100.0
        );
        // RING_FORCE_PCT=10 means ~10% * (fraction that resolves, always here since epoll_ctl/
        // epoll_create1 always exist) should hit; assert a generous lower bound well below that
        // to avoid RNG-offset flakiness while still proving the path is very much alive.
        assert!(
            rate > 0.03,
            "expected a robustly measurable ring-force engagement rate, got {rate:.4} ({hits}/{N})"
        );
    }
}

#[cfg(test)]
mod t24_probe {
    use super::*;
    use std::collections::HashMap;

    /// Builds a directed graph over `epoll_create1` call indices from every `epoll_ctl` call
    /// whose epfd AND target fd both resolve to a live `epoll_create1` output (a genuine
    /// epoll->epoll containment edge, `epfd contains fd`).
    fn epoll_edges(p: &Prog) -> HashMap<u16, Vec<u16>> {
        let mut edges: HashMap<u16, Vec<u16>> = HashMap::new();
        for c in &p.calls {
            if c.desc.name != "epoll_ctl" {
                continue;
            }
            let ArgValue::Res(ResRef::Produced { call_idx: epfd_idx, .. }) = c.args[0] else {
                continue;
            };
            if p.calls[epfd_idx as usize].desc.name != "epoll_create1" {
                continue;
            }
            let ArgValue::Res(ResRef::Produced { call_idx: fd_idx, .. }) = c.args[2] else {
                continue;
            };
            if p.calls[fd_idx as usize].desc.name != "epoll_create1" {
                continue;
            }
            edges.entry(epfd_idx).or_default().push(fd_idx);
        }
        edges
    }

    /// Returns the shortest closed cycle's node count (>=2) in `edges`, if any — lets the probe
    /// distinguish a 2-node mutual-nesting cycle from a >=3-node cycle (the CVE's documented
    /// minimal trigger shape: `epoll_create1`x3 -> A,B,C; A contains B; B contains C; C contains
    /// A). Exploratory probe only, not a committed correctness assertion (full closed cycles are
    /// much rarer than a mere 2-node link, so this just measures the rate via simple DFS).
    fn min_cycle_len(edges: &HashMap<u16, Vec<u16>>) -> Option<usize> {
        fn dfs(node: u16, edges: &HashMap<u16, Vec<u16>>, visiting: &mut Vec<u16>) -> Option<usize> {
            if let Some(pos) = visiting.iter().position(|&n| n == node) {
                return Some(visiting.len() - pos);
            }
            visiting.push(node);
            let mut best: Option<usize> = None;
            if let Some(next) = edges.get(&node) {
                for &n in next {
                    if let Some(len) = dfs(n, edges, visiting) {
                        best = Some(best.map_or(len, |b: usize| b.min(len)));
                    }
                }
            }
            visiting.pop();
            best
        }
        let mut best: Option<usize> = None;
        for &start in edges.keys() {
            let mut visiting = Vec::new();
            if let Some(len) = dfs(start, edges, &mut visiting) {
                best = Some(best.map_or(len, |b: usize| b.min(len)));
            }
        }
        best
    }

    /// Exploratory measurement (not a hard-pinned assertion, since the exact rate depends on
    /// bias/recipe tuning constants): across a large generated sample, count programs with (a)
    /// at least one epoll->epoll link, (b) *any* closed containment cycle (>=2 nodes), and (c)
    /// specifically a >=3-node cycle — the CVE's documented minimal trigger shape
    /// (`epoll_create1`x3 -> A,B,C; A contains B; B contains C; C contains A).
    #[test]
    fn probe_full_cycle_rate() {
        const N: u32 = 200_000;
        let mut link_hits = 0u32;
        let mut cycle_hits = 0u32;
        let mut cycle_ge3_hits = 0u32;
        for seed in 1..=N {
            let mut rng = Rng::new(seed.wrapping_mul(2_654_435_761).wrapping_add(7));
            let p = generate(&mut rng);
            let edges = epoll_edges(&p);
            if !edges.is_empty() {
                link_hits += 1;
            }
            if let Some(len) = min_cycle_len(&edges) {
                cycle_hits += 1;
                if len >= 3 {
                    cycle_ge3_hits += 1;
                }
            }
        }
        eprintln!(
            "T2.4 probe: {link_hits}/{N} programs contain an epoll->epoll link; \
             {cycle_hits}/{N} contain a closed containment CYCLE (any length); \
             {cycle_ge3_hits}/{N} contain a >=3-node cycle (the CVE's minimal trigger shape)"
        );
        // Hard gate: the generator must organically close *some* epoll containment cycle at a
        // robust, non-flaky rate (measured ~0.7% — well above what a single unlucky seed offset
        // could zero out). The >=3-node-specific count is reported but not gated on: it's real
        // and nonzero at this N (measured 8/200000), but rare enough that pinning a hard minimum
        // here would risk RNG-offset flakiness without adding meaningful signal beyond the
        // any-length gate.
        assert!(
            cycle_hits > 0,
            "generate() never closed an epoll containment cycle in {N} seeds"
        );
    }
}
