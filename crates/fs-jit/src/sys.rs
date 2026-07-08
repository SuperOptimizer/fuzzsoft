//! Isolated unsafe surface for `fs-jit`'s native chain codegen (Phase 1 of
//! `docs/jit-scalar-design.md`): the W^X executable arena (`mmap(RW)` -> write machine code ->
//! `mprotect(R-X)` before any execution) and the raw fn-pointer call into it. Mirrors
//! `fs-hostmem`'s `sys.rs` convention: `unsafe` is confined to exactly this file. Every other
//! module in this crate (`lib.rs`, `emit.rs`, `chain.rs`) keeps `#![forbid(unsafe_code)]`.
//!
//! **W^X invariant.** No byte of the arena is ever simultaneously writable and executable: each
//! [`Arena::write`] call `mprotect`s only the (page-aligned) range spanning the bytes it is about
//! to append to `RW`, copies, then immediately `mprotect`s that SAME range back to `R-X` before
//! returning — a page containing a previously-compiled, already-cached chain is never touched
//! again (append-only bump allocator; Phase 1 never evicts or rewrites), so once a page goes `R-X`
//! it stays `R-X` for the rest of the run. Nothing calls into the arena during a write's brief `RW`
//! window (compilation and execution never interleave — `fs-jit` is single-threaded and
//! `ChainCache::run_block` never calls a chain while mid-compile), so at every observable instant
//! every byte of the mapping is either `r-x` (compiled, callable) or `rw-` (not yet compiled, or
//! the one page currently being written), never `rwx` and never callable while writable. See
//! `adversarial_wx_arena_is_never_simultaneously_writable_and_executable` below for a test that
//! reads `/proc/self/maps` and asserts this.
//!
//! **Why per-write, not whole-arena, `mprotect`.** An earlier version `mprotect`'d the *entire*
//! arena on every write. That measured as a severe, worsening-over-time cost on the boot+
//! syscall-fuzz benchmark (`docs/jit-scalar-design.md`'s benchmark section): `mprotect`'s cost
//! scales with how much of the target range has actually been faulted in, so re-protecting the
//! whole (growing) already-populated prefix on every single new chain compile got progressively
//! more expensive as the run went on, at one point costing more than the interpreter work it was
//! supposed to save. `mprotect`ing only the newly-written page(s) bounds each call's cost to a
//! small, constant number of pages regardless of how full the arena already is.

use fs_mmu::Bus;
use fs_riscv::{Cpu, LoadOp, StoreOp};
use std::io;

/// Standard x86-64 Linux page size. Hardcoded rather than queried via `sysconf` (this crate is
/// already Linux/x86-64-specific — `sys.rs`'s raw codegen and `mmap`/`mprotect` usage don't
/// pretend otherwise) — bounds every `mprotect` call to the smallest range that covers a write.
const PAGE_SIZE: usize = 4096;

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

/// A single fixed-capacity, bump-allocated, W^X executable arena.
pub struct Arena {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `Arena` owns an exclusively-mapped memory region; nothing aliases `ptr` outside this
// type, and `ChainCache` (the sole owner) is used from a single thread in every caller in this
// codebase. Not `Sync`; `Send` is fine (moving the mapping across threads, not sharing it, is safe).
unsafe impl Send for Arena {}

impl Arena {
    /// `mmap` a fresh `ARENA_CAPACITY`-byte region, initially `PROT_READ | PROT_WRITE` (so the
    /// first chain can be written before anything ever executes from it).
    pub fn new() -> io::Result<Self> {
        // SAFETY: standard anonymous-mapping mmap; args are all valid (null hint, positive
        // length, private+anonymous flags, no fd/offset). The returned pointer is checked against
        // MAP_FAILED before use.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                ARENA_CAPACITY,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Arena { ptr: ptr as *mut u8, len: 0 })
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

    /// `mprotect` exactly the page-aligned range covering `[off, off+len)` to `prot`. Every call
    /// site passes a range that is either about to be written (`RW`) or was just written (`R-X`),
    /// so this never touches a page outside the write currently in progress.
    fn mprotect_range(&self, off: usize, len: usize, prot: libc::c_int) -> io::Result<()> {
        let page_lo = off & !(PAGE_SIZE - 1);
        let page_hi = (off + len).div_ceil(PAGE_SIZE) * PAGE_SIZE;
        // SAFETY: `page_lo..page_hi` is a page-aligned sub-range of `self.ptr..self.ptr+
        // ARENA_CAPACITY` (checked by `write`'s size guard before this is ever called), so this
        // `mprotect` targets only already-mapped memory this `Arena` owns.
        let rc = unsafe {
            libc::mprotect(self.ptr.add(page_lo) as *mut libc::c_void, page_hi - page_lo, prot)
        };
        if rc != 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }

    /// Append `code` to the arena (toggling just the newly-written page range to `RW`, copying,
    /// then immediately back to `R-X` — see the module doc's W^X invariant and its note on why
    /// this is scoped to the write's own range, not the whole arena), returning the byte offset it
    /// now lives at. `None` if the arena is full (Phase 1 has no growth/eviction: the caller should
    /// just stop compiling new chains for the rest of the run, which is always still correct —
    /// merely un-cached — since every fallback path re-derives its result from the plain
    /// interpreter).
    pub fn write(&mut self, code: &[u8]) -> io::Result<Option<u32>> {
        if code.len() > self.remaining() {
            return Ok(None);
        }
        let off = self.len;
        self.mprotect_range(off, code.len(), libc::PROT_READ | libc::PROT_WRITE)?;
        // SAFETY: the range just made `RW` covers exactly `[off, off+code.len())`, which is within
        // the mapped region (`off + code.len() <= len == ARENA_CAPACITY`, checked above).
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.ptr.add(off), code.len());
        }
        self.mprotect_range(off, code.len(), libc::PROT_READ | libc::PROT_EXEC)?;
        self.len += code.len();
        Ok(Some(off as u32))
    }

    /// Call the chain compiled at byte offset `off` (must be a value previously returned by
    /// [`Arena::write`] on `self` — never one from a different `Arena`, and never after the arena
    /// has been dropped). `cpu` is forwarded unchanged as `rdi`; `bus_data`/`bus_vtable` (Phase 2:
    /// [`decompose_bus`]'s output) are forwarded unchanged as `rsi`/`rdx`, kept live across the
    /// whole chain and forwarded again into every Load/Store call-out — see the `JitFn` doc
    /// comment. Safe to call at any time (the arena is always `R-X` whenever this runs, by
    /// construction — see the module doc) but relies on `off` addressing bytes this `Arena` itself
    /// emitted via [`crate::chain`]'s emitter, which is the actual unsafety this function packages
    /// up: a caller could in principle pass a bogus offset. `ChainCache` (the only caller) always
    /// passes back exactly what `write` returned, immediately followed here.
    pub fn call(&self, off: u32, cpu: *mut fs_riscv::Cpu, bus_data: *mut (), bus_vtable: *const ()) -> u64 {
        // SAFETY: `self.ptr + off` lies within a range `write` already `mprotect`'d to `R-X` (and
        // never touches again — see the module doc), so it is currently executable; the bytes
        // there were emitted by `chain`'s codegen to exactly
        // match `JitFn`'s calling convention (rdi=cpu, rsi/rdx=bus fat pointer, ret=u64, no other
        // register/stack preconditions on entry — see `chain.rs`'s codegen doc). Transmuting a
        // data pointer to a function pointer and calling it is exactly what an executable-arena
        // JIT is for.
        unsafe {
            let f: JitFn = std::mem::transmute(self.ptr.add(off as usize));
            f(cpu, bus_data, bus_vtable)
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly the region `new` mapped, once, on drop.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, ARENA_CAPACITY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arena's own mapping is never simultaneously `PROT_WRITE` and `PROT_EXEC`: parse
    /// `/proc/self/maps` for the mapping containing `arena`'s pointer and check its permission
    /// string, both right after construction (RW, not yet executable) and after writing a chain
    /// (R-X). This is the adversarial W^X test the design doc's validation plan requires.
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
        let perms = perms_of_mapping_containing(arena.ptr as usize);
        assert!(perms.starts_with("rw-"), "fresh arena should be RW, not executable: {perms}");
        assert!(!perms.contains('x'), "fresh arena must not be executable: {perms}");

        let off = arena.write(&[0x31, 0xC0, 0xC3]).unwrap().unwrap(); // `xor eax,eax; ret`
        let perms = perms_of_mapping_containing(arena.ptr as usize);
        assert!(perms.starts_with("r-x"), "post-write arena should be R-X: {perms}");
        assert!(!perms.contains('w'), "post-write arena must not be writable: {perms}");

        // Calling it must be sound and leaves it R-X afterward.
        let mut cpu = fs_riscv::Cpu::new(0);
        let tag = arena.call(off, &mut cpu as *mut _, std::ptr::null_mut(), std::ptr::null());
        assert_eq!(tag, 0, "xor eax,eax; ret deterministically returns 0");
        let perms = perms_of_mapping_containing(arena.ptr as usize);
        assert!(!perms.contains('w'), "arena must still not be writable after a call: {perms}");
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
