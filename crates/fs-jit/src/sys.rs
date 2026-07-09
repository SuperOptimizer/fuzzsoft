//! Isolated unsafe surface for `fs-jit`'s native chain codegen (Phase 1 of
//! `docs/jit-scalar-design.md`): the W^X executable arena and the raw fn-pointer call into it.
//! Mirrors `fs-hostmem`'s `sys.rs` convention: `unsafe` is confined to exactly this file. Every
//! other module in this crate (`lib.rs`, `emit.rs`, `chain.rs`) keeps `#![forbid(unsafe_code)]`.
//!
//! **Dual-mapping W^X (Stage: parallel-scaling fix).** An earlier version used a single `mmap`
//! toggled `RW`->`R-X` via `mprotect` on every write. That's correct single-threaded, but
//! `mprotect` takes the kernel's process-wide `mmap_lock` (a single lock shared by every thread
//! in the process, regardless of which thread's arena is being touched) — under `--jobs N`, N
//! worker threads each compiling their own chains into their own *separate* arenas nonetheless
//! all serialize on that one lock. Measured via `strace -c -f --jit-chain --jobs 16`: `mprotect`
//! was 89.6% of syscall time (~994k calls), and `--jit-chain --jobs 32` ran at 0.08x the plain
//! interpreter — the JIT made parallel fuzzing drastically SLOWER, not faster.
//!
//! The fix: back the arena with a `memfd_create` file and `mmap` it **twice** (`MAP_SHARED`, same
//! fd, same offsets) into two disjoint virtual mappings of the SAME physical pages:
//!   - `write_ptr`: `PROT_READ | PROT_WRITE` — never executable. All codegen writes go here.
//!   - `exec_ptr`: `PROT_READ | PROT_EXEC` — never writable. Every [`JitFn`] returned by
//!     [`Arena::write`]/dispatched by [`Arena::call`] is `exec_ptr + offset`.
//!
//! Both mappings are established ONCE, at [`Arena::new`] (i.e. once per worker thread, at
//! per-thread arena construction) — never again for the rest of that thread's run. A compile
//! ([`Arena::write`]) is then a plain `memcpy` into `write_ptr`, with NO syscall at all, let alone
//! one that touches `mmap_lock`. This is the actual fix: the hot per-compile path no longer makes
//! ANY syscall, so N worker threads compiling concurrently make zero contended kernel calls
//! between them.
//!
//! **W^X invariant, restated for dual-mapping.** No single mapping is ever both writable and
//! executable: `write_ptr`'s mapping is permanently `rw-` (never gains `x`) and `exec_ptr`'s
//! mapping is permanently `r-x` (never gains `w`) for the entire lifetime of the `Arena` — neither
//! mapping's protection ever changes after `new()` returns. This is strictly stronger than Phase
//! 1's invariant (which allowed a page to be transiently `rw-` mid-write): here NEITHER view is
//! ever writable-and-executable, and in fact the execute view is never writable at all, at any
//! point in its existence. See `adversarial_wx_arena_is_never_simultaneously_writable_and_executable`
//! below for a test that reads `/proc/self/maps` and asserts both mappings' permissions.
//!
//! **Why this is safe without an explicit icache flush (x86-64 only).** x86-64's instruction
//! cache is coherent with the data cache via physical-address snooping: a store through
//! `write_ptr` and a subsequent fetch through `exec_ptr` both resolve to the same physical page,
//! and the CPU's cache-coherence protocol guarantees the fetch observes the store — no
//! `clflush`/serializing instruction is required for correctness on this architecture. (An
//! AArch64 port would NOT get this for free — ARM's icache is not automatically coherent with the
//! dcache, so a port would need `__builtin___clear_cache`/`cacheflush` after every write, before
//! the first call into freshly-written bytes.) [`Arena::write`] still issues a
//! `compiler_fence(Ordering::SeqCst)` after the `copy_nonoverlapping` and before returning the
//! offset, so the compiler itself cannot reorder the memcpy past the point callers (`chain.rs`)
//! treat the offset as callable — the "finish writing the whole chain, then call it" sequencing
//! the design doc calls for.

use fs_mmu::Bus;
use fs_riscv::{Cpu, LoadOp, StoreOp};
use std::io;
use std::sync::atomic::{compiler_fence, Ordering};

/// SAFETY-relevant ABI. `rdi`=cpu ptr, `rsi`/`rdx` = the two words of a decomposed `&mut dyn Bus`
/// fat pointer (unused by any Phase 1-emitted code — Phase 1 compiles no `Load`/`Store` and never
/// reads or writes `rsi`/`rdx` — but fixed in the signature now, per the design doc, so Phase 2
/// needs no ABI break). Every `JitFn` value in this crate points into `Arena`'s R-X mapping and was
/// produced by exactly [`crate::chain`]'s emitter — never interpreted as anything else.
pub type JitFn = unsafe extern "C" fn(cpu: *mut fs_riscv::Cpu, bus_data: *mut (), bus_vtable: *const ()) -> u64;

// -------------------------------------------------------------------------------------------
// Phase 2 (`docs/jit-scalar-design.md`): the packed `u64` tag scheme shared by (a) a chain's own
// `JitFn`-level return value and (b) every Load/Store call-out's return value — deliberately the
// SAME bit layout for both, so a Load/Store's rare (trap/halt/repoll) path can simply `ret` with
// the call-out's return value untouched and have it mean the right thing one level up, with zero
// repacking (see `chain.rs`'s `emit_step` Load/Store arms). Bits are checked in this priority
// order (a value only ever has at most one of these three high bits set):
//   - bit 63 (`TAG_TRAP`): a `Trap` is pending in `Cpu::jit_pending_trap`; every other bit ignored.
//   - bit 62 (`TAG_HALT`): HTIF `tohost` halt; bits 0..32 carry the halt code.
//   - bit 61 (`TAG_REPOLL`): a CLINT-range store retired but the chain must stop immediately so
//     the driver's per-instruction CLINT resync runs before anything else executes; no payload.
//   - none of the above: plain continue; bits 0..32 carry a Load's result value (0 for Store,
//     which has nothing to return).
// A `u32` payload (a loaded value or a halt code) can never collide with these bits.
// -------------------------------------------------------------------------------------------
pub(crate) const TAG_TRAP: u64 = 1 << 63;
pub(crate) const TAG_HALT: u64 = 1 << 62;
pub(crate) const TAG_REPOLL: u64 = 1 << 61;

/// Decompose `bus` into its two raw words (data pointer, vtable pointer) for passing across the
/// `JitFn`/shim ABI boundary (`rsi`/`rdx` — see `JitFn`'s doc above). [`recompose_bus`] is the
/// exact inverse; every call site in this crate uses this pair, never hand-rolling the fat-pointer
/// layout itself.
pub(crate) fn decompose_bus(bus: &mut dyn Bus) -> (*mut (), *const ()) {
    // SAFETY: on this (x86-64) target, `*mut dyn Bus` and `(*mut (), *const ())` are both exactly
    // two machine words wide with no niche/metadata beyond those two words (a trait object raw
    // pointer is a plain (data, vtable) pair) — `transmute` between them only ever produces the
    // two raw words here; they are never dereferenced directly, only fed back through
    // `recompose_bus`'s exact inverse. See `bus_fat_pointer_roundtrip` below for a real call
    // through the reconstructed reference.
    unsafe { std::mem::transmute::<*mut dyn Bus, (*mut (), *const ())>(bus as *mut dyn Bus) }
}

/// Reconstitute the `&mut dyn Bus` that [`decompose_bus`] produced `(data, vtable)` from.
///
/// SAFETY: caller must pass back exactly a `(data, vtable)` pair `decompose_bus` produced from a
/// `&mut dyn Bus` that is still live (not moved, not dropped, not aliased elsewhere) for the
/// duration of the returned reference's use. Every call site in this crate satisfies this: the
/// JIT shims (below) run synchronously inside one [`Arena::call`], itself inside one
/// `ChainCache::run_block` call, which holds the real `&mut dyn Bus` on its own stack frame for
/// the whole call and never touches it itself while a chain is running.
unsafe fn recompose_bus<'a>(data: *mut (), vtable: *const ()) -> &'a mut dyn Bus {
    // SAFETY: see this function's doc comment; `transmute`'s size/alignment precondition is the
    // same one `decompose_bus` relies on, in reverse.
    unsafe { std::mem::transmute::<(*mut (), *const ()), &mut dyn Bus>((data, vtable)) }
}

// -------------------------------------------------------------------------------------------
// Load/Store call-out shims. Each is a fixed, process-lifetime-stable `extern "C" fn` address
// `chain.rs`'s codegen `movabs`+`call`s. Monomorphized per `LoadOp`/`StoreOp` variant (rather than
// taking `size`/`signed`/`op` as runtime arguments) so the call site never needs a register beyond
// `va` (Load) / `va`+`val` (Store) on top of the already-live `cpu`/`bus_data`/`bus_vtable` — see
// `chain.rs`'s module doc for the full SysV-argument-register accounting (this is also why there
// is no 5th-arg register-pressure conflict with `R8`, the chain's pinned entry-pc register: it is
// saved via `push`/`pop` around every call, which conveniently also doubles as the argument
// register `val` needs when present — see `chain.rs`'s Store codegen).
// -------------------------------------------------------------------------------------------

/// Common Load body: call the byte-for-byte-shared [`fs_riscv::load_impl`], stash any `Trap` and
/// return `TAG_TRAP`, else return the loaded value directly (never collides with the tag bits —
/// see the tag doc above).
///
/// SAFETY: `cpu`/`bus_data`/`bus_vtable` are exactly what `ChainCache::run_block` passed into
/// [`Arena::call`], which forwards them unchanged as this function's own `rdi`/`rsi`/`rdx` — all
/// three are guaranteed valid, non-null, and not aliased elsewhere for the duration of this call
/// (the call is synchronous; `run_block` holds the real `&mut Cpu`/`&mut dyn Bus` on its own stack
/// frame and does not touch them again until the compiled chain returns).
unsafe fn load_common(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32, size: u8, signed: bool) -> u64 {
    // SAFETY: see this function's doc comment.
    let cpu = unsafe { &mut *cpu };
    // SAFETY: see this function's doc comment / `recompose_bus`'s.
    let bus = unsafe { recompose_bus(bus_data, bus_vtable) };
    match fs_riscv::load_impl(cpu, bus, va, size, signed) {
        Ok(v) => v as u64,
        Err(trap) => {
            cpu.jit_pending_trap = Some(trap);
            TAG_TRAP
        }
    }
}

/// Common Store body: mirrors `fs_riscv::Cpu::exec_one`'s `Store` arm exactly (byte-for-byte
/// reused soft-MMU write via [`fs_riscv::store_impl`], plus the identical HTIF-`tohost`-intercept
/// check for `Sw` — `is_sw` selects it at compile time, matching which shim function called this),
/// then additionally (Phase 2's new behavior, absent from the interpreter because the interpreter
/// re-syncs the CLINT before every single instruction anyway) tags a CLINT-range store so the
/// compiled chain stops immediately instead of continuing to run with a stale interrupt-pending
/// view. SAFETY: see [`load_common`]'s doc comment (identical argument).
unsafe fn store_common(
    cpu: *mut Cpu,
    bus_data: *mut (),
    bus_vtable: *const (),
    va: u32,
    val: u32,
    size: u8,
    is_sw: bool,
) -> u64 {
    // SAFETY: see `load_common`'s doc comment.
    let cpu = unsafe { &mut *cpu };
    // SAFETY: see `load_common`'s doc comment / `recompose_bus`'s.
    let bus = unsafe { recompose_bus(bus_data, bus_vtable) };
    if is_sw && cpu.htif_tohost == Some(va) {
        match fs_riscv::store_impl(cpu, bus, va, size, val) {
            Ok(()) => {
                if val & 1 != 0 {
                    return TAG_HALT | ((val >> 1) as u64);
                }
            }
            Err(trap) => {
                cpu.jit_pending_trap = Some(trap);
                return TAG_TRAP;
            }
        }
    } else if let Err(trap) = fs_riscv::store_impl(cpu, bus, va, size, val) {
        cpu.jit_pending_trap = Some(trap);
        return TAG_TRAP;
    }
    if bus.store_may_assert_interrupt(va, size) { TAG_REPOLL } else { 0 }
}

unsafe extern "C" fn jit_load_lb(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32) -> u64 {
    unsafe { load_common(cpu, bus_data, bus_vtable, va, 1, true) }
}
unsafe extern "C" fn jit_load_lbu(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32) -> u64 {
    unsafe { load_common(cpu, bus_data, bus_vtable, va, 1, false) }
}
unsafe extern "C" fn jit_load_lh(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32) -> u64 {
    unsafe { load_common(cpu, bus_data, bus_vtable, va, 2, true) }
}
unsafe extern "C" fn jit_load_lhu(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32) -> u64 {
    unsafe { load_common(cpu, bus_data, bus_vtable, va, 2, false) }
}
unsafe extern "C" fn jit_load_lw(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32) -> u64 {
    unsafe { load_common(cpu, bus_data, bus_vtable, va, 4, false) }
}

unsafe extern "C" fn jit_store_sb(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32, val: u32) -> u64 {
    unsafe { store_common(cpu, bus_data, bus_vtable, va, val, 1, false) }
}
unsafe extern "C" fn jit_store_sh(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32, val: u32) -> u64 {
    unsafe { store_common(cpu, bus_data, bus_vtable, va, val, 2, false) }
}
unsafe extern "C" fn jit_store_sw(cpu: *mut Cpu, bus_data: *mut (), bus_vtable: *const (), va: u32, val: u32) -> u64 {
    unsafe { store_common(cpu, bus_data, bus_vtable, va, val, 4, true) }
}

/// The fixed process-lifetime address `chain.rs`'s codegen `movabs`+`call`s for a given `LoadOp`
/// (Rust function addresses don't move — see `JitFn`'s doc — so this is safe to bake into
/// compiled-once native code and reuse across every dispatch of that chain).
pub(crate) fn load_shim_addr(op: LoadOp) -> u64 {
    let f: unsafe extern "C" fn(*mut Cpu, *mut (), *const (), u32) -> u64 = match op {
        LoadOp::Lb => jit_load_lb,
        LoadOp::Lbu => jit_load_lbu,
        LoadOp::Lh => jit_load_lh,
        LoadOp::Lhu => jit_load_lhu,
        LoadOp::Lw => jit_load_lw,
    };
    f as usize as u64
}

/// Same as [`load_shim_addr`], for `StoreOp`.
pub(crate) fn store_shim_addr(op: StoreOp) -> u64 {
    let f: unsafe extern "C" fn(*mut Cpu, *mut (), *const (), u32, u32) -> u64 = match op {
        StoreOp::Sb => jit_store_sb,
        StoreOp::Sh => jit_store_sh,
        StoreOp::Sw => jit_store_sw,
    };
    f as usize as u64
}

/// Total arena size: generously large for a fuzz campaign's ALU/branch working set while staying a
/// small, fixed, single `mmap` (no growth/relocation machinery in Phase 1 — see [`Arena::write`]).
/// Sized from a real measurement, not a guess: an initial 16 MiB arena filled up within seconds on
/// the boot+syscall-fuzz benchmark workload (a real kernel touches a very large number of distinct
/// physical chain-head addresses across a diverse corpus), degrading every subsequent miss at an
/// arena-full pc into the fallback path — correct, but far slower than the interpreter it's
/// supposed to beat. 256 MiB (still a single `mmap`, and `MAP_ANONYMOUS` pages are lazily
/// committed — an empty arena costs no resident memory beyond what's actually written) comfortably
/// covers this workload; see `docs/jit-scalar-design.md`'s benchmark writeup for the measured
/// utilization.
const ARENA_CAPACITY: usize = 256 * 1024 * 1024;

/// `MFD_CLOEXEC` (memfd flag). Not exposed by the `libc` crate for the glibc/Linux target as of
/// `libc = "0.2"` (only defined for its Android/FreeBSD/L4Re/Fuchsia targets) — the numeric value
/// is stable uapi (`include/uapi/linux/memfd.h`) so it's hardcoded here rather than pulled in via
/// a raw `libc::syscall`. Not load-bearing for correctness (this process never `exec`s, and the fd
/// is closed immediately after both mappings are established — see [`Arena::new`]); set anyway as
/// routine hygiene against some future code path that forks+execs while an `Arena` is alive.
const MFD_CLOEXEC: libc::c_uint = 1;

/// A single fixed-capacity, bump-allocated, dual-mapped W^X executable arena. Backed by one
/// `memfd_create` file, mapped twice (see the module doc): `write_ptr` (`RW`, never executable)
/// is where codegen writes; `exec_ptr` (`R-X`, never writable) is where compiled chains are
/// called from. Both mappings alias the same physical pages, established once in [`Arena::new`]
/// and never re-protected afterward — there is no per-compile `mprotect` (the entire point of
/// this design; see the module doc's parallel-scaling rationale).
pub struct Arena {
    write_ptr: *mut u8,
    exec_ptr: *mut u8,
    len: usize,
}

// SAFETY: `Arena` owns two exclusively-held mappings of a memfd it created itself; nothing
// aliases `write_ptr`/`exec_ptr` outside this type, and `ChainCache` (the sole owner) is used
// from a single thread in every caller in this codebase (one `Arena` per worker thread under
// `--jobs`, never shared). Not `Sync`; `Send` is fine (moving the mappings across threads, not
// sharing them, is safe).
unsafe impl Send for Arena {}

impl Arena {
    /// Create the arena's backing store (`memfd_create` + `ftruncate` to `ARENA_CAPACITY`) and
    /// map it twice: once `PROT_READ | PROT_WRITE` (`write_ptr`, for codegen), once
    /// `PROT_READ | PROT_EXEC` (`exec_ptr`, for calling compiled chains). Both `MAP_SHARED` over
    /// the same fd/offset so they alias the same physical pages. The fd itself is closed right
    /// after both mappings succeed — once mapped, a mapping keeps the underlying file object
    /// alive on its own; the fd is not needed again (this arena never grows or re-maps).
    pub fn new() -> io::Result<Self> {
        // SAFETY: `name` is a valid NUL-terminated C string literal; `flags` is a valid
        // `memfd_create` flags value. Return value is checked for the `-1` error sentinel before
        // use.
        let fd = unsafe { libc::memfd_create(c"fs-jit-chain-arena".as_ptr(), MFD_CLOEXEC) };
        if fd == -1 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `fd` was just created above and is a valid, open file descriptor; sizing it to
        // `ARENA_CAPACITY` before mmap'ing that many bytes from it is required (a `memfd`, like
        // any regular file, starts at length 0 — mmap'ing bytes past the file's length is valid
        // to *map* but faults on first access without this). memfd pages are allocated lazily on
        // write, same as `MAP_ANONYMOUS`, so this costs no resident memory up front.
        let rc = unsafe { libc::ftruncate(fd, ARENA_CAPACITY as libc::off_t) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            // SAFETY: `fd` is open and owned exclusively by this function at this point.
            unsafe { libc::close(fd) };
            return Err(err);
        }
        // SAFETY: standard shared-file mmap; `fd` is open and sized to at least `ARENA_CAPACITY`
        // bytes (just `ftruncate`'d above), `length`/`offset` are within that size, `addr` hint is
        // null (kernel chooses), flags/prot are a valid combination. Return value is checked
        // against `MAP_FAILED` before use.
        let write_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ARENA_CAPACITY,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if write_ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            // SAFETY: `fd` is still open and owned exclusively by this function.
            unsafe { libc::close(fd) };
            return Err(err);
        }
        // SAFETY: same fd, same offset/length as the mapping above, just a different `prot` — a
        // second, independent mapping of the same underlying pages. `write_ptr` (checked above)
        // proves the fd/size are valid; mapping it again with different permissions is exactly
        // what dual-mapping W^X requires.
        let exec_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), ARENA_CAPACITY, libc::PROT_READ | libc::PROT_EXEC, libc::MAP_SHARED, fd, 0)
        };
        if exec_ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            // SAFETY: unmapping exactly the mapping just established above; `fd` is still open
            // and owned exclusively by this function.
            unsafe {
                libc::munmap(write_ptr, ARENA_CAPACITY);
                libc::close(fd);
            }
            return Err(err);
        }
        // SAFETY: both mappings above now hold their own reference to the underlying file object;
        // closing the fd does not unmap them and this arena never needs the fd again (fixed
        // capacity, no growth/re-mapping).
        unsafe { libc::close(fd) };
        Ok(Arena { write_ptr: write_ptr as *mut u8, exec_ptr: exec_ptr as *mut u8, len: 0 })
    }

    /// Bytes already committed (used to size-check a would-be chain before writing it).
    pub fn used(&self) -> usize {
        self.len
    }

    /// Total fixed arena capacity (see [`ARENA_CAPACITY`]'s doc comment) — diagnostic only.
    pub fn capacity(&self) -> usize {
        ARENA_CAPACITY
    }

    /// Remaining capacity.
    fn remaining(&self) -> usize {
        ARENA_CAPACITY - self.len
    }

    /// Append `code` to the arena — a plain `memcpy` into the `RW` `write_ptr` view, no syscall
    /// at all (see the module doc: this is the entire point of dual-mapping — no per-compile
    /// `mprotect`, hence no `mmap_lock` contention across parallel worker threads) — returning the
    /// byte offset it now lives at (callable via [`Arena::call`], which reads it back through
    /// `exec_ptr`, the same physical bytes just written). `None` if the arena is full (Phase 1
    /// has no growth/eviction: the caller should just stop compiling new chains for the rest of
    /// the run, which is always still correct — merely un-cached — since every fallback path
    /// re-derives its result from the plain interpreter).
    pub fn write(&mut self, code: &[u8]) -> io::Result<Option<u32>> {
        if code.len() > self.remaining() {
            return Ok(None);
        }
        let off = self.len;
        // SAFETY: `write_ptr..write_ptr+ARENA_CAPACITY` is this `Arena`'s own `RW` mapping (never
        // executable — see the module doc); `[off, off+code.len())` is within it (`off +
        // code.len() <= ARENA_CAPACITY`, checked via `remaining()` above).
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.write_ptr.add(off), code.len());
        }
        // Ensure the compiler has not reordered the copy past this point before the offset is
        // handed back as "callable" — see the module doc's note on x86-64 not needing an icache
        // flush but still wanting the natural "finish writing, then call" sequencing enforced.
        compiler_fence(Ordering::SeqCst);
        self.len += code.len();
        Ok(Some(off as u32))
    }

    /// Call the chain compiled at byte offset `off` (must be a value previously returned by
    /// [`Arena::write`] on `self` — never one from a different `Arena`, and never after the arena
    /// has been dropped). `cpu` is forwarded unchanged as `rdi`; `bus_data`/`bus_vtable` (Phase 2:
    /// [`decompose_bus`]'s output) are forwarded unchanged as `rsi`/`rdx`, kept live across the
    /// whole chain and forwarded again into every Load/Store call-out — see the `JitFn` doc
    /// comment. Safe to call at any time: `exec_ptr` is permanently `R-X` for the whole lifetime
    /// of the arena (by construction — see the module doc), and the physical bytes at `off` were
    /// written through `write_ptr` before this offset was ever handed out, so they're already
    /// present by the time any call reaches them (x86-64 icache/dcache coherence — see the module
    /// doc). Relies on `off` addressing bytes this `Arena` itself emitted via [`crate::chain`]'s
    /// emitter, which is the actual unsafety this function packages up: a caller could in
    /// principle pass a bogus offset. `ChainCache` (the only caller) always passes back exactly
    /// what `write` returned, immediately followed here.
    pub fn call(&self, off: u32, cpu: *mut fs_riscv::Cpu, bus_data: *mut (), bus_vtable: *const ()) -> u64 {
        // SAFETY: `self.exec_ptr + off` lies within this `Arena`'s `R-X` mapping (permanently so —
        // see the module doc), and the bytes there were emitted by `chain`'s codegen to exactly
        // match `JitFn`'s calling convention (rdi=cpu, rsi/rdx=bus fat pointer, ret=u64, no other
        // register/stack preconditions on entry — see `chain.rs`'s codegen doc). Transmuting a
        // data pointer to a function pointer and calling it is exactly what an executable-arena
        // JIT is for.
        unsafe {
            let f: JitFn = std::mem::transmute(self.exec_ptr.add(off as usize));
            f(cpu, bus_data, bus_vtable)
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly the two regions `new` mapped, once, on drop. Each is an
        // independent mapping (established via two separate `mmap` calls over the same fd); both
        // must be unmapped since `munmap` only tears down the mapping named by its own address
        // range, not other mappings that happen to alias the same underlying pages.
        unsafe {
            libc::munmap(self.write_ptr as *mut libc::c_void, ARENA_CAPACITY);
            libc::munmap(self.exec_ptr as *mut libc::c_void, ARENA_CAPACITY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neither of the arena's two mappings is ever simultaneously `PROT_WRITE` and `PROT_EXEC`
    /// (nor, for dual-mapping, is either mapping EVER both at once, or ever changes protection at
    /// all): parse `/proc/self/maps` for the mapping containing a given pointer and return its
    /// permission string. This is the adversarial W^X test the design doc's validation plan
    /// requires, updated for dual-mapping to check BOTH views explicitly.
    fn perms_of_mapping_containing(addr: usize) -> String {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        for line in maps.lines() {
            let mut parts = line.splitn(6, ' ');
            let Some(range) = parts.next() else { continue };
            let Some((lo, hi)) = range.split_once('-') else { continue };
            let (lo, hi) = (
                usize::from_str_radix(lo, 16).unwrap(),
                usize::from_str_radix(hi, 16).unwrap(),
            );
            if addr >= lo && addr < hi {
                return parts.next().unwrap_or("").to_string();
            }
        }
        panic!("no /proc/self/maps entry contains {addr:#x}");
    }

    #[test]
    fn adversarial_wx_arena_is_never_simultaneously_writable_and_executable() {
        let mut arena = Arena::new().unwrap();

        // Fresh arena: write view is RW-not-X, execute view is R-X-not-W. Check both, before any
        // write has happened.
        let write_perms = perms_of_mapping_containing(arena.write_ptr as usize);
        assert!(write_perms.starts_with("rw-"), "write view should be RW: {write_perms}");
        assert!(!write_perms.contains('x'), "write view must never be executable: {write_perms}");
        let exec_perms = perms_of_mapping_containing(arena.exec_ptr as usize);
        assert!(exec_perms.starts_with("r-x"), "execute view should be R-X: {exec_perms}");
        assert!(!exec_perms.contains('w'), "execute view must never be writable: {exec_perms}");

        let off = arena.write(&[0x31, 0xC0, 0xC3]).unwrap().unwrap(); // `xor eax,eax; ret`

        // After a write (a plain memcpy, no mprotect at all in this design): both views' *
        // protections must be byte-for-byte unchanged from before — dual-mapping's whole premise
        // is that neither mapping's protection ever moves after `Arena::new`.
        let write_perms = perms_of_mapping_containing(arena.write_ptr as usize);
        assert!(write_perms.starts_with("rw-"), "write view should still be RW post-write: {write_perms}");
        assert!(!write_perms.contains('x'), "write view must still not be executable post-write: {write_perms}");
        let exec_perms = perms_of_mapping_containing(arena.exec_ptr as usize);
        assert!(exec_perms.starts_with("r-x"), "execute view should still be R-X post-write: {exec_perms}");
        assert!(!exec_perms.contains('w'), "execute view must still not be writable post-write: {exec_perms}");

        // Calling it (through the execute view) must be sound, and both views' protections must
        // remain exactly as above afterward.
        let mut cpu = fs_riscv::Cpu::new(0);
        let tag = arena.call(off, &mut cpu as *mut _, std::ptr::null_mut(), std::ptr::null());
        assert_eq!(tag, 0, "xor eax,eax; ret deterministically returns 0");
        let write_perms = perms_of_mapping_containing(arena.write_ptr as usize);
        assert!(!write_perms.contains('x'), "write view must not be executable after a call: {write_perms}");
        let exec_perms = perms_of_mapping_containing(arena.exec_ptr as usize);
        assert!(!exec_perms.contains('w'), "execute view must not be writable after a call: {exec_perms}");
    }

    #[test]
    fn write_then_call_round_trip_executes_real_code() {
        // `xor eax,eax; ret` — a minimal but non-trivial (not just `ret`) round trip.
        let mut arena = Arena::new().unwrap();
        let off = arena.write(&[0x31, 0xC0, 0xC3]).unwrap().unwrap();
        let mut cpu = fs_riscv::Cpu::new(0);
        let tag = arena.call(off, &mut cpu as *mut _, std::ptr::null_mut(), std::ptr::null());
        assert_eq!(tag, 0);
    }

    #[test]
    fn arena_full_returns_none_instead_of_panicking() {
        let mut arena = Arena::new().unwrap();
        let big = vec![0xC3u8; ARENA_CAPACITY + 1];
        assert!(arena.write(&big).unwrap().is_none());
    }

    /// `decompose_bus`/`recompose_bus` round-trip: reconstitute a real `&mut dyn Bus` from the two
    /// raw words and call a genuine method through it, proving the fat-pointer layout assumption
    /// (`JitFn`'s doc comment) actually holds on this target rather than merely "looking right".
    #[test]
    fn bus_fat_pointer_roundtrip() {
        use fs_mmu::{Mmu, PERM_READ, PERM_WRITE};
        let mut mmu = Mmu::new(0x1000, 0x1000);
        mmu.protect(0x1000, 0x1000, PERM_READ | PERM_WRITE).unwrap();
        let bus: &mut dyn Bus = &mut mmu;
        let (data, vtable) = decompose_bus(bus);
        // SAFETY: `mmu` (the `&mut dyn Bus` `decompose_bus` was just called on) is still alive and
        // untouched for the whole of this reconstructed reference's use, matching
        // `recompose_bus`'s documented precondition.
        let recomposed = unsafe { recompose_bus(data, vtable) };
        recomposed.store(0x1000, 4, 0xdead_beef).unwrap();
        assert_eq!(mmu.load(0x1000, 4).unwrap(), 0xdead_beef);
    }
}
