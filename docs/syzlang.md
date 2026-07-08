# fuzzsoft syzlang-lite — typed syscall-description design

*Design produced by a research+synthesis workflow (syzkaller study → fuzzsoft-specific spec). Not yet implemented; the current fuzzer uses raw `Call{nr,args[6]}`.*

## fuzzsoft syzlang-lite: a minimal typed syscall-description model

Grounded in the current implementation (`crates/fs-cli/src/main.rs`: `Call{nr:u32,args:[u32;6]}`, `Prog{calls:Vec<Call>}`, `gen_arg`/`mutate_program`/`pick_nr`/`write_program`, `cmd_fuzz`) and the guest interpreter (`boot/agent.c`: snapshot hypercall hands over `prog`+`scratch` VAs once, then a tight loop does `do_syscall(nr,a0..a5)` `n` times, then `HC_DONE`). Today `nr` is `rng.next()%440` (mostly invalid/denied numbers on this kernel) and every arg is drawn from one flat 8-way pool with no notion of type, fd, buffer, or length — so `open→read→close` chains, well-formed structs, and correct length fields are effectively never generated. This document specifies a typed replacement that still compiles down to exactly the same wire shape fuzzsoft already knows how to inject (nr + up to 6 `u32` register args, plus a scratch region for pointees), extended with one small, general fixup mechanism to thread resources across calls.

Design goal: **syzlang-lite** — the smallest type vocabulary that gives us syzkaller's core wins (typed args, resource threading incl. multi-resource producers like `pipe2`, struct/buffer serialization, length derivation, targeted mutation) without syzkaller's compiler, IL, or cross-arch generality. Everything here is plain Rust data (`&'static` tables + small enums), no DSL parser required (though a DSL front-end could later compile into this same IR).

---

### 1. Type system

```rust
/// A named resource kind (syzkaller's `resource fd[int32]: ...`).
/// Subtyping is a single flat parent link: `sock` is-a `fd`, so a consumer that
/// declares `Res(FD)` accepts values produced as FD *or* SOCK.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResourceKind(pub &'static str);

pub const FD:  ResourceKind = ResourceKind("fd");
pub const SOCK: ResourceKind = ResourceKind("sock"); // subtype of fd
pub const VMA: ResourceKind = ResourceKind("vma");

pub struct ResourceDef {
    pub kind: ResourceKind,
    pub subtype_of: Option<ResourceKind>,
    /// Seed literals usable even with no live producer (syzkaller's `-1, AT_FDCWD`).
    pub seeds: &'static [i64],
}

pub static RESOURCES: &[ResourceDef] = &[
    ResourceDef { kind: FD,   subtype_of: None,      seeds: &[-1, -100 /*AT_FDCWD*/, 0, 1, 2] },
    ResourceDef { kind: SOCK, subtype_of: Some(FD),   seeds: &[-1] },
    ResourceDef { kind: VMA,  subtype_of: None,      seeds: &[0, -1] }, // 0 = let kernel pick, -1 = deliberately bad
];

/// Whether a value produced as `have` may satisfy a consumer that wants `want`.
pub fn kind_compat(want: ResourceKind, have: ResourceKind) -> bool {
    if want == have { return true; }
    RESOURCES.iter().find(|r| r.kind == have)
        .and_then(|r| r.subtype_of).is_some_and(|p| kind_compat(want, p))
}

#[derive(Clone, Copy)]
pub enum Dir { In, Out, InOut }

/// How many bytes a `Buffer` payload should have.
pub enum LenSpec { Fixed(u16), Range(u16, u16) }

pub struct Field { pub name: &'static str, pub ty: &'static ArgType }

/// The syzlang-lite type vocabulary. Deliberately flat — no arrays-of-structs,
/// no unions, no bitfields inside ints. Everything a syscall arg can be is one of these.
pub enum ArgType {
    /// A single fixed value (e.g. an F_SETFL command number).
    Const(u32),
    /// An arbitrary-ish scalar; generator biases toward {0,1,2,-1,boundary,small,random}
    /// the same way today's `gen_arg` pool does, just per-argument-position now.
    Int { bits: u8, signed: bool },
    /// An enum (`bitmask:false`, pick one of `vals`) or bitmask (`bitmask:true`, OR a
    /// random subset of `vals`) — replaces raw random ints on flag-shaped args.
    Flags { vals: &'static [u32], bitmask: bool },
    /// A resource-typed argument: consumed from a live producer of a compatible kind
    /// in the same program, or a seed literal from `ResourceDef::seeds`.
    Res(ResourceKind),
    /// Byte-length of the (serialized) sibling arg at index `of` in the same call.
    /// Resolved to a literal at *generation* time (not a special runtime type) so it
    /// can later be independently mutated — this is how we get "usually correct,
    /// occasionally desynced" length fields for free.
    Len { of: u8 },
    /// A pointer into the scratch region. `nullable` lets the generator emit a literal
    /// NULL instead of allocating scratch (e.g. sendto's optional sockaddr).
    Ptr { dir: Dir, inner: &'static ArgType, nullable: bool },
    /// Raw bytes (mutated buffer contents), sized per `LenSpec`.
    Buffer { len: LenSpec },
    /// A packed struct, fields laid out in C/RV32-ILP32 order (natural alignment,
    /// capped at 4 bytes except explicitly 8-byte-aligned 64-bit fields e.g. timespec64).
    Struct(&'static [Field]),
    /// Pick one of a small literal string pool (paths, memfd names); NUL-terminated
    /// when serialized.
    StringConst(&'static [&'static str]),
}

/// What new resource(s), if any, a call produces — beyond the plain return value
/// being "just an int", most producers return a resource in `a0`; a few (pipe2) fill
/// an *array* of resources into an out-buffer instead.
pub enum Produces {
    None,
    /// Return value (guest a0 right after the ecall) is a new resource of `kind`.
    Ret(ResourceKind),
    /// The `arg_idx`-th arg (a `Ptr{Out,...}`) is an out-array; after the call, `count`
    /// consecutive u32 words at that scratch address are each a new resource of `kind`.
    OutArray { arg_idx: u8, kind: ResourceKind, count: u8 },
}

/// One syscall description (analogous to a syzkaller `.txt` entry / `prog.Syscall`).
pub struct SyscallDesc {
    pub name: &'static str,   // may include a "$variant" suffix, e.g. "fcntl64$dupfd"
    pub nr: u32,              // REAL rv32 nr from generated unistd_32.h — never randomly guessed
    pub args: &'static [ArgType],
    pub produces: Produces,
}
```

Notes / deliberate omissions (kept minimal on purpose):
- No generic tagged unions (syzkaller's polymorphic `fcntl`-style args). Cmd-dependent syscalls are modeled as separate `$variant` `SyscallDesc`s with `Const` cmd fields, exactly matching syzkaller's own convention (see `fcntl64$dupfd` / `fcntl64$setfl` below).
- No arrays-of-arbitrary-type, only the one hard-coded `OutArray` producer shape needed for `pipe2`. If a future syscall needs a general `ptr[out,array[T]]`, extend `ArgType` then — don't build the general case speculatively.
- A resource does not carry metadata (e.g. mmap2's length is not attached to the `vma` it returns) — `munmap`/`mprotect`/`madvise` take an independently-generated `len` arg rather than a length recalled from the producing `mmap2` call. This is a real simplification vs. syzkaller (which *can* propagate such metadata); documented here as a deliberate scope cut. If wanted later, add `ArgType::ResMeta{kind, meta: &'static str}` plus a metadata slot on `ResRef::Produced`.

---

### 2. Resource model + threading (corpus/mutation IR)

This is the tree the corpus stores and the mutator edits — analogous to syzkaller's `prog.Prog`/`prog.Arg`, but flat (one call = one flat arg list, no nested arg trees beyond `Struct`/`Ptr` payloads).

```rust
/// A resource-typed arg value: either a seed literal, or "whatever call `call_idx`
/// produced, resource slot `slot`" (slot 0 for `Produces::Ret`, slot `j` for the
/// `j`-th entry of a `Produces::OutArray`).
pub enum ResRef {
    Seed(i64),
    Produced { call_idx: u16, slot: u8 },
}

/// The value bound to one `ArgType` slot in a `TypedCall`. Structurally mirrors
/// `ArgType` one level deep (a `Ptr` wraps its pointee's `ArgValue`; a `Struct` is a
/// vec of per-field `ArgValue`s in field order).
pub enum ArgValue {
    Imm(u64),                 // Const / Int / Flags / Len (resolved) / literal NULL for a Ptr slot
    Res(ResRef),               // Res(kind) slot
    Bytes(Vec<u8>),            // Buffer contents / chosen StringConst (NUL-terminated)
    Struct(Vec<ArgValue>),     // one entry per Field, same order as the ArgType::Struct
    Ptr(Box<ArgValue>),        // Ptr's pointee (Bytes/Struct); Imm(0) at a Ptr slot means NULL
}

pub struct TypedCall {
    pub desc: &'static SyscallDesc,
    pub args: Vec<ArgValue>,   // args.len() == desc.args.len()
}

pub struct TypedProgram {
    pub calls: Vec<TypedCall>, // corpus + mutation unit; MAX_CALLS (8) enforced on insert
}
```

**Threading invariant (the part that actually matters):** a `ResRef::Produced{call_idx,slot}` inside `calls[i].args[..]` is only valid if `call_idx < i` **and** `kind_compat(consumer_kind, calls[call_idx].desc's produced kind at that slot)` holds. The mutator must preserve this on every edit:

- **Insert** a call at position `p`: any existing `Produced{call_idx,..}` with `call_idx >= p` is renumbered (`+1`); a newly inserted resource-consuming call may pick `Produced{call_idx: k}` only for `k < p`.
- **Remove** call at position `p`: any `Produced{call_idx: p, ..}` reference elsewhere becomes dangling — replace with `ResRef::Seed(...)` (fallback to the kind's seed pool) rather than deleting the whole call chain; `Produced{call_idx>p,..}` is renumbered (`-1`).
- **Splice** (crossover between two corpus programs): re-resolve every `Produced` ref in the spliced-in suffix against the *new* program's indices; anything unresolvable (kind mismatch, no compatible earlier producer) falls back to `Seed`.
- **Havoc on a `Res` arg**: with small probability, swap `Produced{k}` for a different compatible earlier producer, or for a `Seed`, or (if a compatible producer exists later after some other insert) leave as-is — never point forward.
- A dedicated **"wire two calls together" mutation** (the syzkaller-style improvement over pure random insert) explicitly: pick a resource-producing `SyscallDesc` (e.g. `openat`), insert it at a random earlier position if none exists yet, then rewrite a randomly chosen existing `Res(FD)`-typed arg later in the program to `Produced{that call}` — this is what actually manufactures `open→read→close` chains instead of waiting for them to occur by chance.

Generation of a *fresh* `TypedCall` for arg slot `Res(kind)`: 70% pick an existing compatible producer earlier in the program-under-construction (if any), else fall back to `Seed` from `ResourceDef::seeds`.

---

### 3. Lowering: `TypedProgram → ConcreteProgram` (register injection + scratch)

This is the one-shot pass run immediately before injecting a case — never during mutation, so mutation stays cheap and purely on the typed tree.

```rust
pub struct ScratchWriter { pub bytes: Vec<u8>, pub cursor: u32, pub cap: u32 }
impl ScratchWriter {
    /// Bump-allocate `data`, aligned to `align`; deterministically truncates (never
    /// panics/errors) if it would exceed `cap` — same "clamp, don't crash" policy as
    /// the rest of fuzzsoft's generation pipeline.
    pub fn write(&mut self, data: &[u8], align: u32) -> u32 { /* returns byte offset */ todo!() }
}

/// Where a masked arg's *real* value comes from, resolved at runtime by the guest.
pub enum FixupSrc {
    Reg(u8),   // results[call_idx] — that call's a0 return value
    Mem(u32),  // *(u32*)(scratch_base + byte_offset) — kernel-written out-value
}
pub struct Fixup { pub dst_call: u8, pub dst_arg: u8, pub src: FixupSrc }

pub struct ConcreteCall { pub nr: u32, pub args: [u32; 6] } // args[j] is a placeholder (0) if a Fixup targets it
pub struct ConcreteProgram {
    pub calls: Vec<ConcreteCall>,
    pub fixups: Vec<Fixup>,
    pub scratch: Vec<u8>,   // byte image to write into the guest scratch region
}

pub fn lower(p: &TypedProgram, scratch_base_va: u32, scratch_cap: u32) -> ConcreteProgram {
    let mut w = ScratchWriter { bytes: Vec::new(), cursor: 0, cap: scratch_cap };
    let mut calls = Vec::with_capacity(p.calls.len());
    let mut fixups = Vec::new();
    // out_array_off[i] = scratch byte offset allocated for call i's OutArray ptr, if any
    let mut out_array_off: Vec<Option<u32>> = vec![None; p.calls.len()];

    for (i, tc) in p.calls.iter().enumerate() {
        let mut args = [0u32; 6];
        for (j, (aty, av)) in tc.desc.args.iter().zip(&tc.args).enumerate() {
            args[j] = lower_arg(aty, av, &mut w, scratch_base_va, i, j, &mut fixups, &out_array_off);
        }
        if let Produces::OutArray { arg_idx, .. } = tc.desc.produces {
            // record where that Ptr's pointee landed, so later `Produced{i, slot}` refs
            // can compute Mem(offset + slot*4) — filled in by lower_arg via a side channel.
        }
        calls.push(ConcreteCall { nr: tc.desc.nr, args });
    }
    ConcreteProgram { calls, fixups, scratch: w.bytes }
}
```

Per-`ArgType` lowering rules:

| ArgType | ArgValue | Lowered `u32` |
|---|---|---|
| `Const`/`Int`/`Flags` | `Imm(v)` | `v as u32` |
| `Len{of}` | `Imm(v)` (computed at *generation* time from sibling's serialized size, independently mutable afterward) | `v as u32` |
| `Res(kind)` | `Res(Seed(s))` | `s as u32` (literal, e.g. `-1`, `AT_FDCWD`) |
| `Res(kind)` | `Res(Produced{call_idx,slot})` | placeholder `0` in `args[j]`, **plus** a `Fixup{dst_call:i, dst_arg:j, src}` where `src = Reg(call_idx)` if `calls[call_idx].desc.produces == Ret(_)`, else `Mem(out_array_off[call_idx].unwrap() + 4*slot)` if `OutArray{..}` |
| `Ptr{dir,inner,nullable:true}` | `Imm(0)` | `0` (NULL, no scratch used) |
| `Ptr{dir,inner,..}` | `Ptr(inner_val)` | serialize `inner_val` per `inner`'s shape into `w`, get byte `off`; lowered value = `scratch_base_va + off`. If this `Ptr` is the `arg_idx` of a `Produces::OutArray`, record `off` into `out_array_off[i]`. |
| `Buffer{..}` (only ever reached via a `Ptr`) | `Bytes(b)` | write `b` verbatim |
| `Struct(fields)` (only via `Ptr`) | `Struct(vals)` | write each field at its natural-aligned offset (RV32 ILP32: align = size, capped at 4, except 8-byte fields like the two `i64`s in `timespec64`/`llseek`'s `loff_t*` result which align to 8) |
| `StringConst` (only via `Ptr`) | `Bytes(s)` | write bytes + trailing `0` |

Scratch sizing: grow the guest's flat scratch array from today's 4 KiB to **32 KiB** (`static char scratch[32768] __attribute__((aligned(64)))` in `boot/agent.c`), same translate-once-at-snapshot pattern already used for `prog_pas`. 32 KiB comfortably holds several structs/buffers across an 8-call program; overflow truncates deterministically (last writes get 0-length buffers) rather than erroring, matching fuzzsoft's existing "always produce something injectable" philosophy.

---

### 4. Program wire encoding (host → guest `prog` buffer)

Two arrays, fixed max sizes, still one flat `u32` buffer exactly like today — just longer:

```
prog[0]                                   = n   (number of calls, <= MAX_CALLS)
prog[1 .. 1+MAX_CALLS*7)                  = MAX_CALLS call-slots, 7 words each: nr, a0..a5
prog[1+MAX_CALLS*7]                        = nfix  (number of fixups, <= MAX_FIXUPS)
prog[2+MAX_CALLS*7 .. 2+MAX_CALLS*7+MAX_FIXUPS*4)
                                           = MAX_FIXUPS fixup-slots, 4 words each:
                                             dst_call, dst_arg, src_kind(0=Reg,1=Mem), src_val
```

`MAX_CALLS = 8` (unchanged), `MAX_FIXUPS = 32` (generous — worst case is every one of 8 calls' 6 args resource-typed, 48, but that never actually happens; unused slots are zeroed and ignored via `nfix`). Total buffer = `1 + 8*7 + 1 + 32*4 = 186` words = 744 bytes — negligible.

This deliberately **replaces** the bit-packed "6-bit mask + 4-bit-index-per-slot" meta-word approach (which only has 4 bits of index headroom — enough for `Reg(call_idx)` up to 16 calls, but not enough for `Mem(scratch_offset)`, which needs to address up to 32 KiB / 4 = 8192 words = 13 bits). An explicit side-table of `{dst_call, dst_arg, src_kind, src_val}` records is simpler, has no headroom limit, and is what actually makes `pipe2`'s two-resources-from-one-out-buffer case (and any future multi-resource producer) work without a wire-format redesign.

Why a side-table doesn't need per-call ordering games: fixups are applied to call `i` strictly using data produced by calls `< i` (mutator invariant, §2), so by the time the guest is about to execute call `i`, every `Reg`/`Mem` source it might reference has already been written by an earlier iteration of the same loop.

---

### 5. Guest agent execution (`boot/agent.c`)

Still exactly one hypercall pair per case (`SNAPSHOT` then `DONE`) — no new hypercalls, no VM-exit multiplication. The interpreter loop grows from ~5 lines to ~15:

```c
#define MAX_CALLS  8
#define MAX_FIXUPS 32
#define CALL_WORDS 7      /* nr, a0..a5 */
#define FIXUP_WORDS 4     /* dst_call, dst_arg, src_kind, src_val */
#define SCRATCH_SIZE (32 * 1024)

static volatile unsigned prog[1 + MAX_CALLS * CALL_WORDS + 1 + MAX_FIXUPS * FIXUP_WORDS];
static char scratch[SCRATCH_SIZE] __attribute__((aligned(64)));
static unsigned results[MAX_CALLS];   /* per-call return values (a0), reused each program */

/* ... inside the existing for(;;) { hypercall(SNAPSHOT,...); ... } loop ... */
unsigned n = prog[0];
if (n > MAX_CALLS) n = MAX_CALLS;

unsigned fixup_base = 1 + MAX_CALLS * CALL_WORDS;
unsigned nfix = prog[fixup_base];
if (nfix > MAX_FIXUPS) nfix = MAX_FIXUPS;

for (unsigned i = 0; i < n; i++) {
    volatile unsigned *c = &prog[1 + i * CALL_WORDS];
    unsigned a[6] = { c[1], c[2], c[3], c[4], c[5], c[6] };

    for (unsigned f = 0; f < nfix; f++) {
        volatile unsigned *fr = &prog[fixup_base + 1 + f * FIXUP_WORDS];
        if (fr[0] != i) continue;
        unsigned v = fr[2] == 0
            ? results[fr[3]]                                   /* Reg(src_val) */
            : *(volatile unsigned *)(scratch + fr[3]);          /* Mem(src_val) */
        a[fr[1]] = v;
    }

    results[i] = (unsigned)do_syscall(c[0], a[0], a[1], a[2], a[3], a[4], a[5]);
}
hypercall(HC_DONE, 0, 0);
```

`results[]` doesn't need explicit zeroing between programs: any fixup referencing call `k` is only emitted by `lower()` when `k < i` and call `k` is guaranteed to run (calls always execute in array order, unconditionally — fuzzsoft doesn't branch on syscall failure), so `results[k]` is always freshly written before it's read. A failed producer (negative errno) threads forward as-is — a useful negative-testing signal, and keeps the interpreter branch-free on success/failure exactly as today.

Cost: `O(MAX_CALLS * MAX_FIXUPS)` = at most 256 integer compares per program, dwarfed by syscall overhead; no additional hypercalls, no change to snapshot/restore, no change to `prog_pas`-style physical-address pre-translation (just one more array, `scratch`, whose physical addresses are already translated today — only its size grows).

---

### 6. End-to-end pipeline summary

```
corpus: Vec<TypedProgram>                    (typed, resource-linked, mutated here)
            │  mutate (insert/remove/splice/havoc-scalar/havoc-res/"wire producer")
            ▼
   TypedProgram (one case)
            │  lower(scratch_base_va, 32KiB)
            ▼
   ConcreteProgram { calls, fixups, scratch }
            │  write into prog_pas[] / scratch_pas[] via Bus::store  (mechanically
            │  identical to today's write_program, just two regions instead of one)
            ▼
   guest agent: for i in 0..n { apply fixups targeting i; do_syscall; record results[i] }
            │
            ▼
   host: coverage bitmap + kernel-crash oracle (unchanged from today)
```

Only `fs-cli`'s generation/mutation/injection code changes (replacing `gen_arg`/`pick_nr`/`mutate_program`/`write_program` with the typed IR + `lower()`), plus the ~15-line `boot/agent.c` interpreter extension and the scratch buffer size bump. `Snapshot::capture`/`reset`, the coverage bitmap, and the crash oracle are untouched.


---

# Starter descriptions

Concrete starter table (~20 descriptions) in the Rust literal form defined above. `nr` values are read directly from this repo's own generated `build/linux-src/arch/riscv/include/generated/uapi/asm/unistd_32.h` (verified against `qemu/linux-headers/asm-riscv/unistd_32.h`, which agrees). rv32-specific ABI shapes (no plain `lseek`/`mmap`/`fcntl`/legacy `futex`) are modeled with their *real* rv32 signatures, not the generic/x86_64 ones.

```rust
use ArgType::*;
use Dir::*;

// ---------------- flag/const tables ----------------

pub const OPEN_FLAGS: &[u32] = &[
    0o0,        // O_RDONLY
    0o1,        // O_WRONLY
    0o2,        // O_RDWR
    0o100,      // O_CREAT
    0o1000,     // O_TRUNC
    0o2000,     // O_APPEND
    0o200000,   // O_DIRECTORY
    0o2000000,  // O_CLOEXEC
    0o4000,     // O_NONBLOCK
];
pub const OPEN_MODE: &[u32]   = &[0o600, 0o644, 0o755, 0o777, 0];
pub const MMAP_PROT: &[u32]   = &[0 /*NONE*/, 1 /*READ*/, 2 /*WRITE*/, 4 /*EXEC*/, 1|2, 1|4];
pub const MMAP_FLAGS: &[u32]  = &[1 /*SHARED*/, 2 /*PRIVATE*/, 0x10 /*FIXED*/, 0x20 /*ANONYMOUS*/, 2|0x20];
pub const MADV_ADVICE: &[u32] = &[0 /*NORMAL*/, 1 /*RANDOM*/, 4 /*DONTNEED*/, 8 /*FREE*/];
pub const AF_FAMILY: &[u32]   = &[1 /*AF_UNIX*/, 2 /*AF_INET*/];
pub const SOCK_TYPE: &[u32]   = &[1 /*SOCK_STREAM*/, 2 /*SOCK_DGRAM*/];
pub const SEND_FLAGS: &[u32]  = &[0, 0x40 /*MSG_DONTWAIT*/, 0x4000 /*MSG_NOSIGNAL*/];
pub const SEEK_WHENCE: &[u32] = &[0 /*SET*/, 1 /*CUR*/, 2 /*END*/];
pub const O_FLAGS_SETFL: &[u32] = &[0o2000 /*APPEND*/, 0o4000 /*NONBLOCK*/, 0o10000 /*ASYNC*/];
pub const MEMFD_FLAGS: &[u32] = &[0 /*none*/, 1 /*MFD_CLOEXEC*/, 2 /*MFD_ALLOW_SEALING*/, 3];
pub const PRCTL_OPTION: &[u32] = &[
    15 /*PR_SET_NAME*/, 16 /*PR_GET_NAME*/, 38 /*PR_SET_NO_NEW_PRIVS*/, 4 /*PR_SET_DUMPABLE*/,
];
pub const IOCTL_CMD: &[u32] = &[0x541B /*FIONREAD*/, 0x5421 /*FIONBIO*/, 0x5401 /*TCGETS*/];
pub const FUTEX2_FLAGS: &[u32] = &[0 /*FUTEX2_SIZE_U32*/, 2 /*FUTEX2_PRIVATE*/];
pub const CLOCKIDS: &[u32] = &[0 /*CLOCK_REALTIME*/, 1 /*CLOCK_MONOTONIC*/];
pub const PATH_POOL: &[&str] = &["/", "/dev/null", "/proc/self/maps", "/tmp/x"];
pub const MEMFD_NAMES: &[&str] = &["a", "fuzz", ""];

// struct sockaddr (generic, 16 bytes: u16 family + 14 bytes data — enough for AF_UNIX/AF_INET)
static SOCKADDR_FIELDS: &[Field] = &[
    Field { name: "family", ty: &Flags { vals: AF_FAMILY, bitmask: false } },
    Field { name: "data",   ty: &Buffer { len: LenSpec::Fixed(14) } },
];
static SOCKADDR: ArgType = Struct(SOCKADDR_FIELDS);

// struct timespec64 { i64 tv_sec; i64 tv_nsec; } — both 8-byte aligned on rv32
static TIMESPEC64_FIELDS: &[Field] = &[
    Field { name: "tv_sec",  ty: &Int { bits: 64, signed: true } },
    Field { name: "tv_nsec", ty: &Int { bits: 64, signed: true } },
];
static TIMESPEC64: ArgType = Struct(TIMESPEC64_FIELDS);

// ---------------- syscall descriptions ----------------

pub static SYSCALLS: &[SyscallDesc] = &[

    // 1. openat(56): dirfd:AT_FDCWD-or-fd, path, flags, mode -> fd
    SyscallDesc {
        name: "openat", nr: 56,
        args: &[
            Res(FD),                                   // dirfd (seed AT_FDCWD=-100 covers the common case)
            Ptr { dir: In, inner: &StringConst(PATH_POOL), nullable: false },
            Flags { vals: OPEN_FLAGS, bitmask: true },
            Flags { vals: OPEN_MODE, bitmask: false },
        ],
        produces: Produces::Ret(FD),
    },

    // 2. read(63): fd, buf(out), count=len(buf) -> ssize
    SyscallDesc {
        name: "read", nr: 63,
        args: &[
            Res(FD),
            Ptr { dir: Out, inner: &Buffer { len: LenSpec::Range(0, 256) }, nullable: false },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },

    // 3. write(64): fd, buf(in), count=len(buf) -> ssize
    SyscallDesc {
        name: "write", nr: 64,
        args: &[
            Res(FD),
            Ptr { dir: In, inner: &Buffer { len: LenSpec::Range(0, 256) }, nullable: false },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },

    // 4. close(57): fd -> int   (consumes the resource; enables double-close mutants)
    SyscallDesc { name: "close", nr: 57, args: &[Res(FD)], produces: Produces::None },

    // 5. llseek(62): REAL rv32 5-arg shape, not generic lseek.
    //    fd, off_hi, off_lo, result:ptr[out,i64], whence -> int32
    SyscallDesc {
        name: "llseek", nr: 62,
        args: &[
            Res(FD),
            Int { bits: 32, signed: true },   // offset_high
            Int { bits: 32, signed: true },   // offset_low
            Ptr { dir: Out, inner: &Int { bits: 64, signed: true }, nullable: false }, // loff_t *result
            Flags { vals: SEEK_WHENCE, bitmask: false },
        ],
        produces: Produces::None,
    },

    // 6. ioctl(29): fd, cmd, arg (generic hand-picked cmd set; extend per-device as
    //    the initramfs grows char devices — do NOT try to model cmd/arg as a union).
    SyscallDesc {
        name: "ioctl$generic", nr: 29,
        args: &[
            Res(FD),
            Flags { vals: IOCTL_CMD, bitmask: false },
            Ptr { dir: InOut, inner: &Buffer { len: LenSpec::Fixed(64) }, nullable: true },
        ],
        produces: Produces::None,
    },

    // 7. mmap2(222): addr,len,prot,flags,fd,pgoff(page units!) -> vma
    SyscallDesc {
        name: "mmap2", nr: 222,
        args: &[
            Int { bits: 32, signed: false },               // addr hint (usually 0)
            Int { bits: 32, signed: false },                // len (independent of any later consumer's len)
            Flags { vals: MMAP_PROT, bitmask: true },
            Flags { vals: MMAP_FLAGS, bitmask: true },
            Res(FD),                                        // fd, or -1 seed for MAP_ANONYMOUS
            Int { bits: 32, signed: false },                 // pgoff — GOTCHA: page units, not bytes
        ],
        produces: Produces::Ret(VMA),
    },

    // 8. munmap(215): addr:vma, len -> int32   (consumes vma)
    SyscallDesc {
        name: "munmap", nr: 215,
        args: &[Res(VMA), Int { bits: 32, signed: false }],
        produces: Produces::None,
    },

    // 9. mprotect(226): addr:vma, len, prot -> int32
    SyscallDesc {
        name: "mprotect", nr: 226,
        args: &[Res(VMA), Int { bits: 32, signed: false }, Flags { vals: MMAP_PROT, bitmask: true }],
        produces: Produces::None,
    },

    // 10. madvise(233): addr:vma, len, advice -> int32
    SyscallDesc {
        name: "madvise", nr: 233,
        args: &[Res(VMA), Int { bits: 32, signed: false }, Flags { vals: MADV_ADVICE, bitmask: false }],
        produces: Produces::None,
    },

    // 11. socket(198): domain, type, proto -> sock (fd subtype)
    SyscallDesc {
        name: "socket", nr: 198,
        args: &[
            Flags { vals: AF_FAMILY, bitmask: false },
            Flags { vals: SOCK_TYPE, bitmask: false },
            Const(0),
        ],
        produces: Produces::Ret(SOCK),
    },

    // 12. bind(200): fd:sock, addr(in), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "bind", nr: 200,
        args: &[
            Res(SOCK),
            Ptr { dir: In, inner: &SOCKADDR, nullable: false },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },

    // 13. connect(203): fd:sock, addr(in), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "connect", nr: 203,
        args: &[
            Res(SOCK),
            Ptr { dir: In, inner: &SOCKADDR, nullable: false },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },

    // 14. sendto(206): EXACTLY 6 args — fd, buf, len=len(buf), flags, addr(in,nullable), addrlen=len(addr)
    SyscallDesc {
        name: "sendto", nr: 206,
        args: &[
            Res(SOCK),
            Ptr { dir: In, inner: &Buffer { len: LenSpec::Range(0, 128) }, nullable: false },
            Len { of: 1 },
            Flags { vals: SEND_FLAGS, bitmask: true },
            Ptr { dir: In, inner: &SOCKADDR, nullable: true },
            Len { of: 4 },
        ],
        produces: Produces::None,
    },

    // 15. pipe2(59): fds:ptr[out,array[fd,2]], flags -> int32
    //     Multi-resource producer: TWO new fd resources come from an out-buffer, not a0.
    SyscallDesc {
        name: "pipe2", nr: 59,
        args: &[
            Ptr { dir: Out, inner: &Buffer { len: LenSpec::Fixed(8) }, nullable: false }, // 2x i32 fds
            Flags { vals: OPEN_FLAGS, bitmask: true }, // O_CLOEXEC/O_NONBLOCK subset in practice
        ],
        produces: Produces::OutArray { arg_idx: 0, kind: FD, count: 2 },
    },

    // 16. dup(23): oldfd:fd -> fd
    SyscallDesc { name: "dup", nr: 23, args: &[Res(FD)], produces: Produces::Ret(FD) },

    // 17. dup3(24): oldfd:fd, newfd:int, flags -> fd
    SyscallDesc {
        name: "dup3", nr: 24,
        args: &[
            Res(FD),
            Int { bits: 32, signed: false },  // newfd (small ints most interesting; generator biases low)
            Flags { vals: &[0o2000000 /*O_CLOEXEC*/], bitmask: true },
        ],
        produces: Produces::Ret(FD),
    },

    // 18a. fcntl64$dupfd(25): fd, cmd=F_DUPFD, arg:int (min new fd) -> fd
    SyscallDesc {
        name: "fcntl64$dupfd", nr: 25,
        args: &[Res(FD), Const(0 /*F_DUPFD*/), Int { bits: 32, signed: false }],
        produces: Produces::Ret(FD),
    },
    // 18b. fcntl64$setfl(25): fd, cmd=F_SETFL, arg:flags -> int32
    SyscallDesc {
        name: "fcntl64$setfl", nr: 25,
        args: &[Res(FD), Const(4 /*F_SETFL*/), Flags { vals: O_FLAGS_SETFL, bitmask: true }],
        produces: Produces::None,
    },

    // 19. getdents64(61): fd (ideally O_DIRECTORY-opened), buf(out), count=len(buf) -> int32
    SyscallDesc {
        name: "getdents64", nr: 61,
        args: &[
            Res(FD),
            Ptr { dir: Out, inner: &Buffer { len: LenSpec::Range(32, 512) }, nullable: false },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },

    // 20. memfd_create(279): name(in,string), flags -> fd. Zero filesystem dependency.
    SyscallDesc {
        name: "memfd_create", nr: 279,
        args: &[
            Ptr { dir: In, inner: &StringConst(MEMFD_NAMES), nullable: false },
            Flags { vals: MEMFD_FLAGS, bitmask: true },
        ],
        produces: Produces::Ret(FD),
    },

    // 21. prctl(167): all-scalar, no resources/pointers.
    SyscallDesc {
        name: "prctl", nr: 167,
        args: &[
            Flags { vals: PRCTL_OPTION, bitmask: false },
            Int { bits: 32, signed: false },
            Int { bits: 32, signed: false },
            Int { bits: 32, signed: false },
            Int { bits: 32, signed: false },
        ],
        produces: Produces::None,
    },

    // 22a. futex_wake(454): modern split op, NOT legacy futex(). uaddr, mask, nr, flags -> int32
    SyscallDesc {
        name: "futex_wake", nr: 454,
        args: &[
            Ptr { dir: InOut, inner: &Int { bits: 32, signed: false }, nullable: false },
            Int { bits: 32, signed: false },              // mask
            Int { bits: 32, signed: false },               // nr to wake
            Flags { vals: FUTEX2_FLAGS, bitmask: true },
        ],
        produces: Produces::None,
    },
    // 22b. futex_wait(455): uaddr, val, mask, flags, timeout(in,nullable), clockid -> int32
    SyscallDesc {
        name: "futex_wait", nr: 455,
        args: &[
            Ptr { dir: InOut, inner: &Int { bits: 32, signed: false }, nullable: false },
            Int { bits: 32, signed: false },               // val
            Int { bits: 32, signed: false },                // mask
            Flags { vals: FUTEX2_FLAGS, bitmask: true },
            Ptr { dir: In, inner: &TIMESPEC64, nullable: true },
            Flags { vals: CLOCKIDS, bitmask: false },
        ],
        produces: Produces::None,
    },
];
```

**Resource threading realized by this table**: `openat`/`socket`/`memfd_create`/`dup`/`dup3`/`fcntl64$dupfd` all `Produces::Ret(FD|SOCK)`; `pipe2` is the one multi-resource producer (`Produces::OutArray{count:2}`); `mmap2` produces `VMA`. Consumers (`read`/`write`/`close`/`llseek`/`ioctl$generic`/`fcntl64*`/`getdents64`/`bind`/`connect`/`sendto`/`munmap`/`mprotect`/`madvise`) all declare `Res(FD)`/`Res(SOCK)`/`Res(VMA)` args, so the mutator's "wire two calls together" move can build `openat→read→close`, `socket→bind→sendto`, `pipe2→write(fds[1])→read(fds[0])→close→close`, and `mmap2→madvise→munmap` chains directly, plus double-close / use-after-close mutants by leaving a stale `Produced` ref after the mutator's remove-with-fallback path chooses *not* to repair it (a deliberately occasional, not-always-repaired case worth keeping for UAF-style coverage).

**Landing order** (matches the zero-filesystem-dependency-first recommendation): wave 1 = `memfd_create, pipe2, dup, dup3, socket, bind, connect, sendto, mmap2, munmap, mprotect, madvise, futex_wake, futex_wait, prctl, fcntl64$dupfd, fcntl64$setfl` (need no guest path corpus); wave 2 = `openat, read, write, close, llseek, ioctl$generic, getdents64` once `openat`'s `O_CREAT` + the small `PATH_POOL` are confirmed to work against the current initramfs.

