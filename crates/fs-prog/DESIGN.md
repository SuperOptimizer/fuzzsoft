# fs-prog: guest-agent wire protocol

This crate lowers a typed `Prog` into a `Lowered` (calls + scratch bytes + a resource-fixup
table) via `lower()`, then `to_wire()` packs that into the flat `u32` buffer the guest agent
(`boot/agent.c`) reads. This document is the exact contract the lead should implement on the
`boot/agent.c` side (full rationale in `docs/syzlang.md` §3-5; this is the condensed spec).

## 1. Wire buffer layout

One flat array of `u32`, replacing today's `[count][nr,a0..a5]*`:

```
prog[0]                                    = n        (number of calls, <= MAX_CALLS)
prog[1 .. 1+MAX_CALLS*7)                   = MAX_CALLS call-slots, 7 words each:
                                              nr, a0, a1, a2, a3, a4, a5
prog[1+MAX_CALLS*7]                        = nfix      (number of fixups, <= MAX_FIXUPS)
prog[2+MAX_CALLS*7 .. 2+MAX_CALLS*7+MAX_FIXUPS*4)
                                            = MAX_FIXUPS fixup-slots, 4 words each:
                                              dst_call, dst_arg, src_kind, src_val
```

Constants (must match `fs-prog`'s `lower`/`to_wire` exactly — see `crates/fs-prog/src/lower.rs`):

```c
#define MAX_CALLS   8
#define MAX_FIXUPS  32
#define CALL_WORDS  7    /* nr, a0..a5 */
#define FIXUP_WORDS 4    /* dst_call, dst_arg, src_kind, src_val */
```

Total buffer size: `1 + MAX_CALLS*CALL_WORDS + 1 + MAX_FIXUPS*FIXUP_WORDS` = `1 + 56 + 1 + 128`
= **186 words = 744 bytes** (`fs_prog::WIRE_WORDS`).

Unused call/fixup slots beyond `n`/`nfix` are zeroed by `to_wire()` and MUST be ignored by the
agent (it only reads `n` call-slots and `nfix` fixup-slots).

## 2. Scratch region

Grows from today's 4 KiB to **32 KiB**:

```c
static char scratch[32 * 1024] __attribute__((aligned(64)));
```

`fs-prog`'s `ScratchWriter` bump-allocates into this region and deterministically truncates
(never panics/errors) on overflow — pointer args may end up pointing at 0-length data past a
certain point in a case, never at an invalid/unmapped address, since the address handed back is
always `scratch_base_va + offset` with `offset <= cap`.

## 3. Fixup semantics: `dst_call, dst_arg, src_kind, src_val`

Each fixup slot says: *"before executing call `dst_call`, overwrite its `dst_arg`-th register
(a0..a5) with a value looked up at runtime."* `src_kind` selects how:

- `src_kind == 0` (`Reg`): the value is `results[src_val]` — the a0 return value call `src_val`
  produced (`src_val` is a call index). Used when the source call's `Produces` is `Ret(kind)`
  (e.g. `openat`, `socket`, `dup`, `memfd_create`).
- `src_kind == 1` (`Mem`): the value is the `u32` word at `*(scratch + src_val)` — `src_val` is a
  byte offset into the scratch region (NOT a call index). Used when the source call's
  `Produces` is `OutArray{..}` (currently only `pipe2`, which writes two fds into an out-buffer
  rather than returning one in a0); `src_val` is precomputed by `lower()` as
  `out_array_offset + 4*slot`.

`args[dst_arg]` in the call-slot itself is a placeholder `0` wherever a fixup targets it — the
agent must apply all fixups for a call *before* invoking that call's syscall.

## 4. Agent interpreter loop (replaces `boot/agent.c`'s current ~5-line loop)

```c
#define MAX_CALLS   8
#define MAX_FIXUPS  32
#define CALL_WORDS  7
#define FIXUP_WORDS 4
#define SCRATCH_SIZE (32 * 1024)

static volatile unsigned prog[1 + MAX_CALLS * CALL_WORDS + 1 + MAX_FIXUPS * FIXUP_WORDS];
static char scratch[SCRATCH_SIZE] __attribute__((aligned(64)));
static unsigned results[MAX_CALLS];   /* per-call a0, reused each program */

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
        if (fr[0] != i) continue;                             /* dst_call != this call */
        unsigned v = fr[2] == 0
            ? results[fr[3]]                                   /* Reg(src_val=call_idx) */
            : *(volatile unsigned *)(scratch + fr[3]);         /* Mem(src_val=byte_offset) */
        a[fr[1]] = v;                                          /* dst_arg */
    }

    results[i] = (unsigned)do_syscall(c[0], a[0], a[1], a[2], a[3], a[4], a[5]);
}
hypercall(HC_DONE, 0, 0);
```

Notes:

- `results[]` needs no explicit zeroing between programs: a fixup referencing call `k` is only
  ever emitted by `lower()` when `k < dst_call`, and calls always execute in array order
  unconditionally (fuzzsoft never branches on syscall failure), so `results[k]` is always
  freshly written before it's read.
- A failed producer (negative errno in `results[k]`) threads forward as-is into the consumer —
  useful negative-testing signal, and keeps the interpreter branch-free on success/failure.
- Cost: `O(MAX_CALLS * MAX_FIXUPS)` = at most 256 integer compares per program — negligible next
  to syscall overhead. No new hypercalls; the existing `SNAPSHOT`/`DONE` pair is unchanged.
- `prog[]`'s and `scratch[]`'s physical addresses are pre-translated once at snapshot time
  exactly as today (only `scratch`'s size changes, from 4 KiB to 32 KiB, and `prog[]`'s array
  size grows to 186 words — both are just bigger flat arrays, no new translation logic needed on
  the host side).

## 5. Host-side integration (`fs-cli::cmd_fuzz`)

Replace:

- `Call`/`Prog` (raw `nr`+`args[6]`) → `fs_prog::Prog` (typed, resource-linked).
- `gen_program`/`gen_call`/`gen_arg`/`pick_nr` → `fs_prog::generate(&mut rng)`.
- `mutate_program` → `fs_prog::mutate(&mut rng, &prog)`.
- `write_program` (writes `[count][nr,a0..a5]*` into `prog_pas`) → `fs_prog::lower(&prog,
  scratch_base_va)` then `fs_prog::to_wire(&lowered)`, writing the resulting `Vec<u32>` into the
  (now 186-word) `prog_pas[]` word-by-word, **plus** writing `lowered.scratch` into a newly
  precomputed `scratch_pas[]` (physical addresses for the scratch region, translated once at
  snapshot time exactly like `prog_pas` is today).

`fs-prog` never touches `fs_riscv`/`fs_mmu`/`fs_platform` — it is pure data plus the
generate/mutate/lower pipeline. All guest-memory writes stay in `fs-cli`, mechanically identical
to today's `write_program`, just writing two regions (`prog`, `scratch`) of known word counts
instead of one.
