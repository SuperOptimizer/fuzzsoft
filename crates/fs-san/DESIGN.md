# fs-san — Design

`fs-san` is the sanitizer *policy* layer on top of `fs-mmu`'s soft MMU (`docs/architecture.md`
§3). It is the concrete realization of decision #19 ("soft-MMU as primary sanitizer for the whole
kernel — hook guest allocator to stamp guard/poison perms — plus panic/oops/WARN/BUG detection").
This document covers the allocator-hook half of #19 (the panic/oops/BUG-symbol watching half is a
separate, simpler concern for whoever owns the run loop, and is not part of this crate).

## 1. The core idea: the MMU is already the sanitizer

`fs-mmu::Mmu` carries one permission byte per guest byte: `READ | WRITE | EXEC | RAW | ACC`. Two
properties of that plane, which already exist and are untouched by this crate, do all the actual
bug-detection work:

- Any access to a byte lacking the required permission bit faults `FaultKind::Permission`,
  distinguishable from `FaultKind::Unmapped` (outside the guest window at all).
- Allocation stamps `RAW | WRITE` (no `READ`); the first *write* to a byte clears `RAW` and sets
  `READ`. A *read* before that first write faults. This is the existing uninitialized-memory
  oracle — `fs-san` does not add a new mechanism for it, it just stamps allocations this way.

`fs-san`'s entire job is: **decide which bytes should be poisoned or RAW-stamped, and when** —
i.e. learn `(addr, size)` for every `malloc`/`free` the guest performs, without the guest having
been built with any sanitizer runtime. Once `fs-san` calls `Mmu::protect`/`Mmu::poison`, the
faults themselves are just the MMU doing what it already does. This is the whole thesis: **a
sanitizer is not a separate detection engine bolted onto the interpreter — it's metadata fed to a
detector that was already there for free.**

```
crates/fs-san/src/
  alloc.rs      Sanitizer: redzone alloc()/free(), quarantine bookkeeping, SanError
  hooks.rs      PcHooks: ISA-agnostic PC-hook framework (learn calls with NO guest changes)
  hypercall.rs  Cooperative hypercall sub-protocol (learn calls WITH a guest agent)
  linux.rs      RV32 Linux wiring: System.map -> PcHooks for the kernel slab allocator
  lib.rs        Re-exports + crate-level overview
```

## 2. Redzones and quarantine, mechanically

`Sanitizer::alloc(mmu, addr, size)`:

1. Poison `[addr - redzone, addr)` and `[addr + size, addr + size + redzone)` (best-effort —
   skipped where it would fall outside the mapped window, since a payload can legitimately sit at
   the very edge of guest RAM).
2. Stamp `[addr, addr + size)` as `WRITE | RAW` (no `READ`).
3. Record `addr` as live; if it was sitting in quarantine, evict it from quarantine tracking (see
   below — this models a real allocator reusing a freed address).

Any read/write that walks past the payload lands on a poisoned guard byte -> `Permission` fault.
That is the entire OOB detector: no bounds check is written anywhere in `fs-san`; the MMU's
existing per-byte permission check *is* the bounds check, and the redzone is just data placed
next to the payload that happens to always fail that check.

`Sanitizer::free(mmu, addr)`:

1. Look up `addr` in the live-allocation map. Not found -> `SanError::InvalidFree` (double-free or
   a wild/unknown pointer — itself a bug worth surfacing to the fuzzer).
2. Poison the payload bytes (the guard bytes were already poisoned from step 1 above and are left
   alone — poisoning is idempotent).
3. Move `addr` into a bounded FIFO "quarantine" of freed-and-tracked addresses.

Any subsequent access to a freed pointer — including from a stale alias kept around by buggy guest
code — lands on now-poisoned payload bytes -> `Permission` fault: use-after-free, detected the
same way as OOB, because it *is* the same mechanism (poisoned bytes), just applied to freed rather
than never-allocated space.

**Quarantine is bounded** (`DEFAULT_QUARANTINE_CAP`, mirroring the "bounded dirty list" discipline
in decision #11): once the FIFO is full, the oldest entry is evicted from *tracking* only. Its
bytes stay poisoned regardless — eviction never un-poisons anything, it only means `fs-san` can no
longer report "this address was specifically freed" for that entry; a later access to it still
faults, just without that extra classification detail. Safety therefore never regresses; only bug
triage detail is bounded.

**A subsequent `alloc()` at a quarantined address is allowed and expected.** Guest allocators
(kernel slabs, bump allocators) routinely hand the same address back out after some churn — that
is not a sanitizer bug, it's the allocator working normally. `alloc()` un-poisons and re-stamps the
region, exactly as if it had never been freed; the *quarantine window itself* (accesses between
free and the next matching alloc) is where UAF detection lives.

## 3. Two ways to *learn* `(addr, size)` — the actual novelty here

Redzones and quarantine over a byte-permission plane are not new (this is Falk's design, and it's
also conceptually what ASAN's shadow memory does). The interesting fuzzsoft-specific problem is:
**how do you learn a guest's malloc/free call sites and arguments without being able to
recompile the guest** — because the end goal (per the task's motivating example) is fuzzing
closed-source, uninstrumented binaries, e.g. a Windows kernel driver, where there is no source to
add ASAN/KASAN to and no build to relink against a sanitizer runtime.

### 3a. Cooperative hypercall path (`hypercall.rs`) — when you *can* touch the guest

fuzzsoft already has a hypercall channel: `Cpu.hypercall_eid: Option<u32>` plus
`SysExit::Hypercall(a0)` in `fs-riscv` (decision #6's mechanism, reused here). `fs-san` defines two
sub-commands within that channel:

| `a0` (cmd)        | `a1`        | `a2`   | Effect                     |
|-------------------|-------------|--------|----------------------------|
| `CMD_MALLOC`      | returned ptr| size   | `Sanitizer::alloc(a1, a2)` |
| `CMD_FREE`         | ptr         | —      | `Sanitizer::free(a1)`      |

`SysExit::Hypercall` only surfaces `a0`; `a1`/`a2` are read directly out of `Cpu::regs` (a public
field) by whoever owns the run loop, then handed to `fs_san::hypercall::dispatch`. This requires a
tiny guest-side agent (a few lines wrapping the real allocator to also `ecall` with the sub-command
before returning) — cheap and exact, but only possible when the guest is ours to modify. This is
the path for our own RV32 Linux target once `kmalloc`/`kfree` wrappers are added.

### 3b. PC-hook path (`hooks.rs`) — the path for uninstrumented / closed-source binaries

This is the path that generalizes to "cannot recompile the guest at all," including a Windows
target running under a (hypothetical, future) x86 backend. `PcHooks` is deliberately ISA-agnostic:
it only needs "a PC" and "a flat register file," no dependency on `fs-riscv`.

**The mechanism, independent of ISA:** every calling convention that matters here (RISC-V's
standard `a0..a7`/`ra` convention; x86-64 Win64/fastcall; System V AMD64) shares the same shape —
arguments arrive in registers (or, for some x86 conventions, on the stack, which is just "a fixed
offset from the stack pointer register" and equally readable), and the return value comes back in
one fixed register (`a0` on RISC-V, `rax` on x86-64). So the framework is:

1. **Register known allocator entry points by address.** For our Linux kernel target:
   `kmalloc`/`__kmalloc`/`kmem_cache_alloc` (alloc-shaped: size in an argument register) and
   `kfree` (free-shaped: pointer in an argument register). For a Windows kernel target (documented
   here as the design target, not yet implemented since fuzzsoft has no x86 backend):
   `ExAllocatePoolWithTag(PoolType, NumberOfBytes, Tag)` (size is the *second* argument) and
   `ExFreePool(P)` / `ExFreePoolWithTag(P, Tag)`. Entry addresses come from the target's symbol
   table (kernel `System.map`/DWARF, or a Windows PDB) — no source or recompilation needed, only a
   symbol resolution step, which is exactly the sense in which this works on closed-source code:
   you need to *know where the allocator is*, not *read its source*.
2. **At an alloc-shaped entry**, read the size argument immediately (it's already in a register).
   The pointer is *not* known yet — the callee hasn't executed — so stash `(return_address, size)`
   and wait. The return address is itself just a register at entry (RISC-V `ra`) or the top of
   stack at entry (x86 `call` convention) — both are "read one fixed location at the moment
   control enters the function," which is the crux of why this generalizes.
3. **At a free-shaped entry**, the pointer is already available — fire the free event immediately.
   (Freeing typically only touches allocator/slab metadata *before* the user payload, not the
   payload bytes themselves, so poisoning at entry — rather than waiting for the free function to
   finish — is safe and simpler.)
4. **When PC reaches a previously-stashed return address**, the callee just returned: read the
   fixed return-value location (RISC-V `a0`, x86-64 `rax`) to learn the pointer, pop the matching
   stashed size (LIFO per return address, so recursive/re-entrant calls through the same call site
   still pair correctly), and fire the alloc event.

This is fully implemented for the RISC-V register-convention case in `hooks.rs` (`PcHooks`,
`AllocHook`, `FreeHook`, `HookEvent`) — it is not a stub. What's future work, out of this crate's
scope because fuzzsoft has no x86 emulation core, is: (a) an x86-64/Win64 register-convention
instantiation of the same `on_pc` logic (same algorithm, different register indices, and a stack
read instead of a register read for the argument that Win64 spills), and (b) a symbolizer that
turns a Windows PDB into `entry_pc` values automatically instead of a hand-written config. The
kernel `System.map` half of (b) — for our own RV32 Linux target specifically — is implemented
now, in `linux.rs`; see §3c.

### 3c. `linux.rs` — wiring the PC-hook path to the RV32 Linux kernel target

`linux.rs` is where §3b's generic mechanism meets our actual target: it turns a kernel build's
`System.map` into `PcHooks` registrations for the slab allocator, with zero kernel changes.

**End-to-end flow:**

```
build kernel (build/linux-src)
        |
        v
System.map on disk  --parse_system_map()-->  HashMap<symbol name, u32 address>
        |
        v
register_kernel_allocator_hooks(&mut hooks, &syms)
        |   (tries every known kmalloc/kfree-family name; registers whichever the
        |    running kernel build actually has — names drift across versions/configs)
        v
PcHooks now watches kmalloc/kfree entry (and kmalloc's matching return) addresses
        |
        v
fuzz driver's step loop, once per retired instruction:
    if let Some(event) = hooks.on_pc(cpu.pc, &cpu.regs) {
        match event {
            HookEvent::Alloc { addr, size } => { let _ = sanitizer.alloc(&mut mmu, addr, size); }
            HookEvent::Free { addr }        => { let _ = sanitizer.free(&mut mmu, addr); }
        }
    }
        |
        v
Sanitizer stamps/poisons bytes in the soft MMU (§2) exactly as in the userspace case
        |
        v
Kernel OOB/UAF in *unmodified, non-KASAN* kernel code now faults at the byte-granular
soft MMU, the same Permission fault as any other sanitizer-caught bug in this crate.
```

`parse_system_map(text: &str) -> HashMap<String, u32>` parses the standard `HEXADDR TYPE SYMBOL`
line shape (both symbol-type cases accepted, e.g. `T`/`t`), skipping malformed/noise lines rather
than erroring, since a real `System.map` is thousands of lines and only a handful are allocator
entry points we go looking for.

`register_kernel_allocator_hooks(hooks: &mut PcHooks, syms: &HashMap<String, u32>)` tries each of
a fixed list of known allocator/deallocator symbol names against `syms` and registers a hook for
every one present:

| Symbol | Shape | Arg convention |
|---|---|---|
| `kmalloc`, `__kmalloc`, `__kmalloc_noprof`, `kmalloc_noprof`, `__kmalloc_node` | alloc | size is the first argument -> `a0` |
| `kmalloc_trace` | alloc | `(cache, flags, size)` -> size is the **third** argument -> `a2`, not `a0` |
| `kmem_cache_alloc`, `kmem_cache_alloc_noprof` | alloc | **not hooked** — size is not an argument at all, it's `cachep->object_size` (a guest-memory read of a version-specific struct offset, which `hooks.rs`'s register-only design deliberately does not do); documented in `linux.rs` and skipped rather than guessed |
| `kfree`, `kfree_sensitive` | free | pointer is the only argument -> `a0` |
| `kmem_cache_free` | free | `(cache, objp)` -> pointer is the **second** argument -> `a1`, not `a0` |

Names vary by kernel version (e.g. the `_noprof` variants only exist under
`CONFIG_MEM_ALLOC_PROFILING`) — this is why registration tries every known name independently
rather than assuming a fixed set exists, mirroring the closed-source case where you likewise don't
get to assume which symbols a given binary happens to export.

**The one open gap:** `kmem_cache_alloc` allocations are currently invisible to the sanitizer
(no hook is registered for them), since their size lives in the cache object, not the call's
arguments. In practice a large share of kernel heap traffic still goes through the plain
`kmalloc`/`kfree` family covered above; closing the `kmem_cache_alloc` gap would mean extending
`AllocHook` with an optional "read size from guest memory at this struct offset instead of a
register" mode — a small, mechanical extension, but real guest-memory access rather than pure
register-file inspection, so it's called out here rather than silently faked with a guessed size.

**Worked example config (RISC-V, matches `hooks.rs` test coverage):**

```rust
use fs_san::{AllocHook, FreeHook, PcHooks, Sanitizer};

let mut hooks = PcHooks::new();
hooks.hook_alloc(AllocHook { entry_pc: KMALLOC_ENTRY, size_reg: 10 }); // a0 = size
hooks.hook_free(FreeHook { entry_pc: KFREE_ENTRY, ptr_reg: 10 });      // a0 = ptr

let mut san = Sanitizer::new(fs_san::DEFAULT_REDZONE);

// In the interpreter's step loop, once per retired instruction:
if let Some(event) = hooks.on_pc(cpu.pc, &cpu.regs) {
    match event {
        fs_san::HookEvent::Alloc { addr, size } => { let _ = san.alloc(&mut mmu, addr, size); }
        fs_san::HookEvent::Free { addr } => { let _ = san.free(&mut mmu, addr); }
    }
}
```

## 4. Contrast with compiler sanitizers (ASAN/KASAN)

| | ASAN / KASAN | `fs-san` |
|---|---|---|
| Where the check lives | Instrumentation the compiler inserts at every load/store in the *guest's own compiled code* | The soft MMU's permission check, which every guest instruction already goes through to reach memory — nothing guest-side is inserted |
| Requires guest source? | Yes — must recompile with `-fsanitize=address` / `CONFIG_KASAN` | No — works on a binary with zero source access, only its execution trace and (for the hook path) a symbol table |
| Requires guest cooperation? | Yes — the sanitizer runtime is linked into the guest | Hypercall path: a few lines of glue code. PC-hook path: **none at all** |
| Shadow memory | A separate compressed shadow region the instrumentation consults (1 shadow byte / 8 guest bytes on ASAN) | The permission plane fs-mmu already maintains 1:1 with guest bytes, for every VM, as an M0 feature — not sanitizer-specific overhead |
| Portable to closed-source targets (e.g. Windows kernel pool)? | **No** — there is no source to instrument | **Yes, in principle** — PC-hook the platform's known allocator entry points (`ExAllocatePoolWithTag`/`ExFreePool`) by address, same mechanism as §3b, modulo an x86 backend existing |
| Cost model | Guest-code size/speed penalty from inserted checks, paid on every VM regardless of whether it's the one being fuzzed | Host-side permission-byte check the interpreter already pays per access (architecture.md §3); zero *additional* guest overhead since nothing is inserted into guest code |

The line that matters for fuzzsoft specifically: architecture.md §3 already commits to the
byte-granular soft MMU as "the bug oracle" and calls the RAW/RWX enforcement something that "falls
out for free." `fs-san` is the layer that makes that free oracle *aware of allocator lifetimes*
(so it can also catch OOB and UAF, not just raw RWX violations and uninitialized reads on
already-mapped memory) — while preserving the "no guest instrumentation" property that is the
entire reason the soft-MMU approach is worth building in the first place, and the reason it is the
*only* approach among the two that can ever reach a closed-source Windows target.

## 5. Testing

Unit tests in `alloc.rs`, `hooks.rs`, `hypercall.rs` construct an `Mmu` directly and drive
`Sanitizer`/`PcHooks`/`hypercall::dispatch` against it: in-bounds access, OOB read/write hitting a
guard, uninitialized read, free -> UAF read/write, double-free, double-alloc, quarantine-then-legit
realloc, and quarantine-cap eviction (tracking is bounded, poisoning never regresses). `fs-mmu`
gained matching unit tests for the new `poison`/`in_bounds`/`perm_at` primitives.

`linux.rs`'s tests parse a small embedded `System.map`-shaped snippet (real symbol lines plus
assorted noise: blank lines, other symbol types, a malformed line) and assert: noise is ignored
and only real symbol lines are captured; `register_kernel_allocator_hooks` wires up a found
`kmalloc`/`kfree` pair end-to-end through `PcHooks::on_pc` exactly as `hooks.rs`'s own tests drive
`PcHooks` directly; `kmem_cache_alloc`/`kmem_cache_alloc_noprof` are confirmed to never fire a
hook (the documented size-unavailable gap) while `kmem_cache_free`'s pointer is confirmed to come
from `a1`, not `a0`, unlike every other free-shaped hook; and registering against an empty symbol
table is a safe no-op.
