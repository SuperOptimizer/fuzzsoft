//! Soft MMU with byte-level permissions and read-after-write (uninitialized) detection.
//!
//! This follows Brandon Falk's (gamozolabs) Vectorized Emulation MMU design: a shadow
//! permission byte sits parallel to every guest byte, so RWX enforcement and uninitialized-read
//! detection fall out for free at single-byte granularity. It is the fuzzing bug oracle.
//!
//! M0 is a flat, scalar MMU over a single guest address window `[base, base+size)`. The
//! interleaved 16-lane vectorized layout (memory+perms at u32 granularity) is an M4 concern;
//! the byte-level permission model here is deliberately identical so it ports upward.

use std::fmt;
use std::sync::Arc;

/// Load-permitted.
pub const PERM_READ: u8 = 1 << 0;
/// Store-permitted.
pub const PERM_WRITE: u8 = 1 << 1;
/// Fetch-permitted.
pub const PERM_EXEC: u8 = 1 << 2;
/// Read-after-write: set on allocation (no READ). The first store upgrades the byte to READ
/// and clears RAW, so reads of never-written bytes fault — an uninitialized-memory oracle.
pub const PERM_RAW: u8 = 1 << 3;
/// Access/coverage bit (reserved; used by the emulator-native coverage layer later).
pub const PERM_ACC: u8 = 1 << 4;
/// KMSAN Stage 2 (`docs/kmsan.md`) value-taint shadow: a reused spare bit in the same perms byte,
/// NOT a separate `Vec<u8>` plane. Decisive reasoning (from the design doc): `fs-loader::load_into`
/// sets `perms[off] = perm` directly (bypassing `Mmu::write`, the only RAW-clearing path), so a
/// *separate* shadow array would default to all-tainted for the whole loaded kernel image — a
/// day-one false-positive storm. A reused bit is safe by construction: every existing caller
/// (loader/`protect`/`poison`/normal `write`) never sets bit 5, so every byte's taint starts at 0
/// for free, and is only ever set by the new KMSAN-gated paths below (`Bus::write_shadow`'s
/// store-scatter, and `fs-cli`'s `--kmsan` allocator-hook seeding via `Mmu::set_vtaint`). Distinct
/// from `PERM_RAW`: RAW is ASAN-strict (fault on first read of a never-written byte); VTAINT
/// *permits* the read and only matters when `Bus::read_raw_state` (Stage 1/2's load-taint gather)
/// or a KMSAN consumption checkpoint (`Branch`) inspects it. Never touched by the normal
/// `read`/`write`/`read_bytewise`/`write_bytewise` paths — those remain byte-for-byte unchanged.
pub const PERM_VTAINT: u8 = 1 << 5;

/// Phase 3 (`docs/jit-scalar-design.md`) fast-path "danger" mask: any byte carrying one of these
/// bits is, by definition, NOT the trivial case — [`Bus::fast_ptr`] bails on it regardless of
/// whether the caller's required bits (`PERM_READ`/`PERM_WRITE`) are also present. `PERM_RAW`
/// (uninitialized-read oracle) is the one that exists today; `PERM_ACC` is included pre-emptively
/// since its doc comment reserves it for "the emulator-native coverage layer later" — a future
/// sanitizer/coverage consumer of that bit gets the same "the fast path never silently races ahead
/// of me" guarantee RAW gets today, with zero changes needed here when that bit starts being set.
const FAST_PTR_FORBIDDEN: u8 = PERM_RAW | PERM_ACC;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    Exec,
}

impl fmt::Display for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Access::Read => "read",
            Access::Write => "write",
            Access::Exec => "exec",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    /// Address outside the mapped guest window.
    Unmapped,
    /// Mapped, but the required permission bit was not set.
    Permission,
    /// Access not naturally aligned (M0 does not emulate misaligned access support).
    Unaligned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub addr: u32,
    pub len: u32,
    pub access: Access,
    pub kind: FaultKind,
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            FaultKind::Unmapped => "unmapped",
            FaultKind::Permission => "permission",
            FaultKind::Unaligned => "unaligned",
        };
        write!(
            f,
            "{} fault on {} of {} byte(s) @ {:#010x}",
            kind, self.access, self.len, self.addr
        )
    }
}

/// A physical bus: RAM plus (in full-system) MMIO devices, addressed by physical address.
/// The interpreter routes every translated access through this so a `Machine` can dispatch to
/// devices (CLINT, later PLIC/UART/virtio) while plain RAM goes to the soft-MMU.
pub trait Bus {
    fn load(&mut self, addr: u32, size: u8) -> Result<u32, Fault>;
    fn store(&mut self, addr: u32, size: u8, val: u32) -> Result<(), Fault>;
    fn ifetch16(&mut self, addr: u32) -> Result<u16, Fault>;

    /// KMSAN load-taint source (`docs/kmsan.md`): gather byte-granular taint state for up to 4
    /// bytes at physical `addr`, bit `8*i` set iff byte `i` of the span carries EITHER `PERM_RAW`
    /// (Stage 1: never-written/uninitialized) OR `PERM_VTAINT` (Stage 2: explicitly seeded/
    /// propagated value-taint) — a `Load` gathers both sources with one call, which is what makes
    /// this *value* taint rather than just "never written" taint. Purely additive and read-only —
    /// deliberately separate from `load`/`read`/`read_bytewise` so it does NOT touch their
    /// RAW-clearing-on-write semantics or ASAN-strict fault-on-first-read behavior (`docs/kmsan.md`'s
    /// KMSAN *permits* reading uninitialized memory and reports only at consumption). Default (for
    /// `Bus` impls not backed by an [`Mmu`]) conservatively reports every byte clean (`0`) — KMSAN
    /// load-taint is a silent no-op there rather than a false positive until wired through.
    fn read_raw_state(&self, _addr: u32, _len: u8) -> u32 {
        0
    }

    /// KMSAN Stage 2 store-taint sink (`docs/kmsan.md`): scatter a byte-taint mask (same `8*i`
    /// encoding as [`Bus::read_raw_state`]) into up to 4 bytes' `PERM_VTAINT` bit at physical
    /// `addr` — an EXACT overwrite per byte (tainted bit set -> `PERM_VTAINT` set; clear ->
    /// `PERM_VTAINT` cleared), not an OR. This mirrors real KMSAN's shadow-copy-on-store semantics:
    /// storing a fully-known value must un-taint the destination bytes (or a byte tainted by a
    /// prior allocation could spuriously stay "uninitialized" forever after a legitimate write),
    /// while storing a tainted register value must taint them. Only ever called when
    /// [`crate`]-external KMSAN tracking is enabled (`fs_riscv::Cpu::kmsan_enabled`) — never
    /// touches `PERM_READ`/`PERM_WRITE`/`PERM_EXEC`/`PERM_RAW`/`PERM_ACC`. Default (for `Bus` impls
    /// with no taint shadow) is a silent no-op, matching `read_raw_state`'s default-clean stance.
    fn write_shadow(&mut self, _addr: u32, _len: u8, _taint_mask: u32) {}

    /// Does a (just-completed, successful) store to physical `addr` of `size` bytes potentially
    /// newly assert an interrupt that a per-instruction driver loop must observe before continuing
    /// (`docs/jit-scalar-design.md`'s Phase 2 "CLINT store early-exit" discussion)? Default `false`
    /// (plain [`Mmu`], with no MMIO devices, never does). Full-system `Bus` impls (`fs-platform`'s
    /// `Machine`/`CowMachine`) override this for their CLINT MMIO window: a native chain-JIT
    /// compiled run (`fs-jit`) can retire many instructions — including a `Store` — without
    /// returning control to the driver loop that normally resyncs the CLINT
    /// (`mtime`/`mtimecmp`/`msip` -> CSRs) before every single instruction; a `Store` into CLINT's
    /// window can make an interrupt newly deliverable, so the compiled chain must stop and return
    /// immediately after such a store (its own retirement/pc-advance already happened) rather than
    /// continuing to execute further chained instructions with a stale interrupt-pending view.
    /// Named generically (not `in_clint`) since any future MMIO device with the same "a store here
    /// can change interrupt-pending state" property would need the identical treatment.
    fn store_may_assert_interrupt(&self, _addr: u32, _size: u8) -> bool {
        false
    }

    /// Phase 3 inlined memory fast path (`docs/jit-scalar-design.md`): the ONE place the
    /// byte-granular perm/RAW oracle is consulted for a chain-JIT's inline load/store. Returns a
    /// raw host pointer to `len` (1/2/4) bytes at physical `addr` — safe for a direct load (when
    /// `need == PERM_READ`) or store (when `need == PERM_READ | PERM_WRITE`) of exactly `len`
    /// bytes — IFF every touched byte's permission byte has ALL of `need`'s bits set AND NONE of
    /// [`FAST_PTR_FORBIDDEN`]'s (`PERM_RAW`/`PERM_ACC` — any of those means "not the trivial case,
    /// a real sanitizer/oracle cares about this byte", so decline unconditionally regardless of
    /// what else is set) and the whole span is backed by directly host-addressable RAM (never
    /// MMIO). Note this is an "at least `need`, and none of the forbidden bits" check, NOT bitwise
    /// equality to `need` — a byte legitimately carrying extra permitted bits alongside `need`
    /// (e.g. `PERM_READ | PERM_WRITE | PERM_EXEC` on an RWX page, for a `need == PERM_READ` load)
    /// is still the ordinary, safe, common case and must still fast-path; only `PERM_RAW`/
    /// `PERM_ACC` are treated as disqualifying "something special is watching this byte" signals.
    /// `addr`/`len` are assumed already checked naturally aligned by the caller — a `len` that
    /// divides 4096 (RAM's page granularity) and an `addr` that is a multiple of `len` can never
    /// straddle a page, so this function does not itself re-check page-crossing.
    ///
    /// Declining (`None`) is ALWAYS correct: it just means "no fast path here, use the ordinary
    /// checked `load`/`store`", which remains the sole source of truth for every non-trivial case
    /// (RAW-poisoned bytes, missing perms, MMIO, out of bounds — anything at all). Default impl
    /// always declines, which is correct for any `Bus` with no direct host-addressable backing
    /// (or one that simply chooses not to support this).
    ///
    /// For a store (`need` includes `PERM_WRITE`), a `Some` return has ALREADY performed every
    /// side effect the ordinary checked `write` path would perform other than the byte content
    /// write itself (dirty-block tracking; copy-on-write page materialization for a COW-backed
    /// implementor) — the forbidden-bit check guarantees no RAW-clear/READ-upgrade is needed (no
    /// `PERM_RAW`, and `need` for a store already requires `PERM_READ` present too), so the
    /// caller's own direct store through the returned pointer is the ONLY remaining step, and is
    /// bit-for-bit what `write()` would have done.
    fn fast_ptr(&mut self, addr: u32, len: u8, need: u8) -> Option<*mut u8> {
        let _ = (addr, len, need);
        None
    }
}

/// Reset granularity for snapshot fuzzing: one cache line (decision #11).
pub const DIRTY_BLOCK: usize = 64;

/// A flat guest memory with a parallel permission plane.
#[derive(Clone)]
pub struct Mmu {
    base: u32,
    mem: Vec<u8>,
    perms: Vec<u8>,
    /// Dirty-block reset state (only active after `enable_dirty_tracking`).
    track_dirty: bool,
    dirty: Vec<usize>,
    dirty_bitmap: Vec<u64>,
}

impl Mmu {
    /// Create a zeroed guest window of `size` bytes starting at `base`. All permissions start
    /// clear, i.e. the whole window is "unmapped" until a loader/allocator grants access.
    pub fn new(base: u32, size: usize) -> Self {
        Self {
            base,
            mem: vec![0u8; size],
            perms: vec![0u8; size],
            track_dirty: false,
            dirty: Vec::new(),
            dirty_bitmap: Vec::new(),
        }
    }

    /// Begin tracking dirtied 64-byte blocks (for O(dirty) snapshot reset). Starts all-clean.
    pub fn enable_dirty_tracking(&mut self) {
        self.track_dirty = true;
        let blocks = self.mem.len().div_ceil(DIRTY_BLOCK);
        self.dirty_bitmap = vec![0u64; blocks.div_ceil(64)];
        self.dirty.clear();
    }

    #[inline]
    fn mark_dirty(&mut self, off: usize) {
        let blk = off / DIRTY_BLOCK;
        let (w, bit) = (blk / 64, blk % 64);
        if self.dirty_bitmap[w] & (1 << bit) == 0 {
            self.dirty_bitmap[w] |= 1 << bit;
            self.dirty.push(blk);
        }
    }

    /// Blocks dirtied since the last `clear_dirty` (indices into 64-byte blocks).
    pub fn dirty_blocks(&self) -> &[usize] {
        &self.dirty
    }

    /// Forget the dirty set (call after restoring, to start a fresh case all-clean). O(dirty):
    /// only the bitmap words that hold a dirtied block are cleared.
    pub fn clear_dirty(&mut self) {
        for i in 0..self.dirty.len() {
            self.dirty_bitmap[self.dirty[i] / 64] = 0;
        }
        self.dirty.clear();
    }

    /// Raw contents + permission planes (for capturing a golden snapshot).
    pub fn planes(&self) -> (&[u8], &[u8]) {
        (&self.mem, &self.perms)
    }

    /// Restore all dirtied blocks' contents+permissions from a golden snapshot, then start clean.
    /// Cost is O(bytes dirtied), never O(total RAM).
    pub fn reset_dirty(&mut self, gmem: &[u8], gperms: &[u8]) {
        for i in 0..self.dirty.len() {
            let blk = self.dirty[i];
            let start = blk * DIRTY_BLOCK;
            let end = (start + DIRTY_BLOCK).min(self.mem.len());
            self.mem[start..end].copy_from_slice(&gmem[start..end]);
            self.perms[start..end].copy_from_slice(&gperms[start..end]);
        }
        self.clear_dirty();
    }

    pub fn base(&self) -> u32 {
        self.base
    }
    pub fn size(&self) -> usize {
        self.mem.len()
    }
    pub fn end(&self) -> u32 {
        self.base.wrapping_add(self.mem.len() as u32)
    }

    #[inline]
    fn offset(&self, addr: u32) -> Option<usize> {
        if addr < self.base {
            return None;
        }
        let off = (addr - self.base) as usize;
        (off < self.mem.len()).then_some(off)
    }

    #[inline]
    fn fault(addr: u32, len: u32, access: Access, kind: FaultKind) -> Fault {
        Fault {
            addr,
            len,
            access,
            kind,
        }
    }

    /// Loader primitive: write `data` at `addr` and stamp `perm`, bypassing permission checks.
    pub fn map(&mut self, addr: u32, data: &[u8], perm: u8) -> Result<(), Fault> {
        for (i, b) in data.iter().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, 1, Access::Write, FaultKind::Unmapped))?;
            self.mem[off] = *b;
            self.perms[off] = perm;
        }
        Ok(())
    }

    /// Stamp permissions over a region without touching contents (e.g. .bss beyond filesz, or a stack).
    pub fn protect(&mut self, addr: u32, len: u32, perm: u8) -> Result<(), Fault> {
        for i in 0..len {
            let a = addr.wrapping_add(i);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, len, Access::Write, FaultKind::Unmapped))?;
            self.perms[off] = perm;
        }
        Ok(())
    }

    /// Poison a region: clear every permission bit (READ/WRITE/EXEC/RAW/ACC) so *any* access to
    /// these bytes faults with `FaultKind::Permission`, even though the address remains inside
    /// the mapped window (distinguishing "used to be valid, now forbidden" from `Unmapped`).
    ///
    /// This is the primitive a sanitizer layer builds on: redzones around an allocation and
    /// quarantined (freed) memory are both just poisoned bytes. Equivalent to
    /// `self.protect(addr, len, 0)`, named separately because the *intent* (deny-all guard) is
    /// distinct from `protect`'s general "stamp arbitrary perm" use.
    pub fn poison(&mut self, addr: u32, len: u32) -> Result<(), Fault> {
        self.protect(addr, len, 0)
    }

    /// KMSAN Stage 2 (`docs/kmsan.md`) bulk taint-seeding primitive: OR (`tainted = true`) or clear
    /// (`tainted = false`) `PERM_VTAINT` across `[addr, addr+len)`, leaving every other permission
    /// bit (READ/WRITE/EXEC/RAW/ACC) exactly as it was. Unlike [`Mmu::protect`] (which overwrites
    /// the whole perm byte), this flips only the one taint bit — the allocator hook seeding path
    /// (`fs-cli`'s `--kmsan` context) calls this on a freshly-returned kmalloc/kmem_cache_alloc
    /// payload whose R/W/X state was already established by the kernel's own memory map (or, under
    /// `--sanitize`'s own alloc hooks — mutually exclusive with `--kmsan`, never combined — by
    /// `Sanitizer`), and must not disturb it. Analogous to `Bus::write_shadow`, but unbounded in
    /// length (an allocation can be far larger than 4 bytes) and called directly on `&mut Mmu`
    /// rather than through the `Bus` trait, mirroring how `Sanitizer`/`PageSanitizer` already call
    /// `protect`/`poison` directly rather than through `Bus`.
    pub fn set_vtaint(&mut self, addr: u32, len: u32, tainted: bool) -> Result<(), Fault> {
        for i in 0..len {
            let a = addr.wrapping_add(i);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, len, Access::Write, FaultKind::Unmapped))?;
            if tainted {
                self.perms[off] |= PERM_VTAINT;
            } else {
                self.perms[off] &= !PERM_VTAINT;
            }
        }
        Ok(())
    }

    /// True if `[addr, addr+len)` lies entirely inside the mapped guest window, without
    /// touching contents or permissions. Lets callers (e.g. a redzone allocator) probe bounds
    /// before deciding how much of a guard region actually fits, instead of relying on a
    /// `protect`/`poison` call failing partway through.
    pub fn in_bounds(&self, addr: u32, len: u32) -> bool {
        if len == 0 {
            return addr >= self.base && addr <= self.end();
        }
        let Some(start) = self.offset(addr) else {
            return false;
        };
        let Some(last_addr) = addr.checked_add(len - 1) else {
            return false;
        };
        match self.offset(last_addr) {
            Some(last) => last >= start,
            None => false,
        }
    }

    /// Read-only permission byte at `addr` (e.g. for a sanitizer to inspect current state
    /// without performing a checked access). `None` if `addr` is outside the mapped window.
    pub fn perm_at(&self, addr: u32) -> Option<u8> {
        self.offset(addr).map(|off| self.perms[off])
    }

    /// Checked read: every byte must carry PERM_READ.
    ///
    /// Fast path (the overwhelmingly common case): the whole span is one contiguous, in-bounds,
    /// fully-readable slice — one bounds check, one tight (auto-vectorizable) permission scan, one
    /// `copy_from_slice`, instead of per-byte address recomputation + bounds/perm checks. Any miss
    /// (out of bounds, or a byte lacking `PERM_READ` — including RAW/uninitialized bytes, which
    /// carry RAW but not READ) falls through to the byte-wise path, which reports the exact
    /// faulting byte address, so fault semantics are unchanged.
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Fault> {
        let len = buf.len();
        if let Some(off) = self.offset(addr)
            && off + len <= self.mem.len()
            && self.perms[off..off + len].iter().all(|&p| p & PERM_READ != 0)
        {
            buf.copy_from_slice(&self.mem[off..off + len]);
            return Ok(());
        }
        self.read_bytewise(addr, buf)
    }

    #[cold]
    fn read_bytewise(&self, addr: u32, buf: &mut [u8]) -> Result<(), Fault> {
        let len = buf.len() as u32;
        for (i, out) in buf.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, len, Access::Read, FaultKind::Unmapped))?;
            if self.perms[off] & PERM_READ == 0 {
                return Err(Self::fault(a, len, Access::Read, FaultKind::Permission));
            }
            *out = self.mem[off];
        }
        Ok(())
    }

    /// Checked write: every byte must carry PERM_WRITE. Writing upgrades RAW bytes to READ
    /// (and clears RAW), unlocking reads of exactly the bytes that were written.
    ///
    /// Fast path (the common case): the whole span is one contiguous, in-bounds, fully-writable
    /// slice — one bounds check, one permission scan, one `copy_from_slice`, one perm-upgrade pass,
    /// and (under dirty tracking) marks the 1–2 blocks the span spans directly, instead of per-byte
    /// work. Any miss falls through to the byte-wise path, preserving exact fault semantics.
    pub fn write(&mut self, addr: u32, data: &[u8]) -> Result<(), Fault> {
        let len = data.len();
        if let Some(off) = self.offset(addr)
            && off + len <= self.mem.len()
            && self.perms[off..off + len].iter().all(|&p| p & PERM_WRITE != 0)
        {
            self.mem[off..off + len].copy_from_slice(data);
            for p in &mut self.perms[off..off + len] {
                *p = (*p | PERM_READ) & !PERM_RAW;
            }
            if self.track_dirty && len > 0 {
                for blk in (off / DIRTY_BLOCK)..=((off + len - 1) / DIRTY_BLOCK) {
                    let (w, bit) = (blk / 64, blk % 64);
                    if self.dirty_bitmap[w] & (1 << bit) == 0 {
                        self.dirty_bitmap[w] |= 1 << bit;
                        self.dirty.push(blk);
                    }
                }
            }
            return Ok(());
        }
        self.write_bytewise(addr, data)
    }

    #[cold]
    fn write_bytewise(&mut self, addr: u32, data: &[u8]) -> Result<(), Fault> {
        let len = data.len() as u32;
        for (i, b) in data.iter().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, len, Access::Write, FaultKind::Unmapped))?;
            if self.perms[off] & PERM_WRITE == 0 {
                return Err(Self::fault(a, len, Access::Write, FaultKind::Permission));
            }
            self.mem[off] = *b;
            self.perms[off] = (self.perms[off] | PERM_READ) & !PERM_RAW;
            if self.track_dirty {
                self.mark_dirty(off);
            }
        }
        Ok(())
    }

    #[inline]
    fn check_align(addr: u32, n: u32, access: Access) -> Result<(), Fault> {
        if addr.is_multiple_of(n) {
            Ok(())
        } else {
            Err(Self::fault(addr, n, access, FaultKind::Unaligned))
        }
    }

    pub fn read_u8(&self, addr: u32) -> Result<u8, Fault> {
        let mut b = [0u8; 1];
        self.read(addr, &mut b)?;
        Ok(b[0])
    }
    pub fn read_u16(&self, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Read)?;
        let mut b = [0u8; 2];
        self.read(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    pub fn read_u32(&self, addr: u32) -> Result<u32, Fault> {
        Self::check_align(addr, 4, Access::Read)?;
        let mut b = [0u8; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn write_u8(&mut self, addr: u32, v: u8) -> Result<(), Fault> {
        self.write(addr, &[v])
    }
    pub fn write_u16(&mut self, addr: u32, v: u16) -> Result<(), Fault> {
        Self::check_align(addr, 2, Access::Write)?;
        self.write(addr, &v.to_le_bytes())
    }
    pub fn write_u32(&mut self, addr: u32, v: u32) -> Result<(), Fault> {
        Self::check_align(addr, 4, Access::Write)?;
        self.write(addr, &v.to_le_bytes())
    }

    /// Read `N` contiguous executable bytes at `addr` into a buffer. Fast path: in-bounds and every
    /// byte carries `PERM_EXEC` → one bounds check + one perm scan + one `copy_from_slice`; else
    /// byte-wise with exact fault reporting. `addr` alignment is checked by the callers below.
    #[inline]
    fn fetch_bytes<const N: usize>(&self, addr: u32) -> Result<[u8; N], Fault> {
        let mut b = [0u8; N];
        if let Some(off) = self.offset(addr)
            && off + N <= self.mem.len()
            && self.perms[off..off + N].iter().all(|&p| p & PERM_EXEC != 0)
        {
            b.copy_from_slice(&self.mem[off..off + N]);
            return Ok(b);
        }
        for (i, out) in b.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, N as u32, Access::Exec, FaultKind::Unmapped))?;
            if self.perms[off] & PERM_EXEC == 0 {
                return Err(Self::fault(a, N as u32, Access::Exec, FaultKind::Permission));
            }
            *out = self.mem[off];
        }
        Ok(b)
    }

    /// Instruction half-word fetch: 2-byte aligned (IALIGN=16 with the C extension), every byte
    /// must carry PERM_EXEC (READ not required).
    pub fn fetch_u16(&self, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Exec)?;
        Ok(u16::from_le_bytes(self.fetch_bytes::<2>(addr)?))
    }

    /// Instruction fetch: 4-byte aligned, every byte must carry PERM_EXEC (READ not required).
    pub fn fetch_u32(&self, addr: u32) -> Result<u32, Fault> {
        Self::check_align(addr, 4, Access::Exec)?;
        Ok(u32::from_le_bytes(self.fetch_bytes::<4>(addr)?))
    }
}

/// Plain RAM is itself a bus (used directly by bare M0/M1 execution and the differential tester).
impl Bus for Mmu {
    fn load(&mut self, addr: u32, size: u8) -> Result<u32, Fault> {
        match size {
            1 => self.read_u8(addr).map(u32::from),
            2 => self.read_u16(addr).map(u32::from),
            _ => self.read_u32(addr),
        }
    }
    fn store(&mut self, addr: u32, size: u8, val: u32) -> Result<(), Fault> {
        match size {
            1 => self.write_u8(addr, val as u8),
            2 => self.write_u16(addr, val as u16),
            _ => self.write_u32(addr, val),
        }
    }
    fn ifetch16(&mut self, addr: u32) -> Result<u16, Fault> {
        self.fetch_u16(addr)
    }
    fn read_raw_state(&self, addr: u32, len: u8) -> u32 {
        let mut mask = 0u32;
        for i in 0..(len.min(4) as u32) {
            if let Some(off) = self.offset(addr.wrapping_add(i))
                && self.perms[off] & (PERM_RAW | PERM_VTAINT) != 0
            {
                mask |= 1 << (8 * i);
            }
        }
        mask
    }

    fn write_shadow(&mut self, addr: u32, len: u8, taint_mask: u32) {
        for i in 0..(len.min(4) as u32) {
            if let Some(off) = self.offset(addr.wrapping_add(i)) {
                if (taint_mask >> (8 * i)) & 1 != 0 {
                    self.perms[off] |= PERM_VTAINT;
                } else {
                    self.perms[off] &= !PERM_VTAINT;
                }
            }
        }
    }

    fn fast_ptr(&mut self, addr: u32, len: u8, need: u8) -> Option<*mut u8> {
        let off = self.offset(addr)?;
        let end = off.checked_add(len as usize)?;
        if end > self.mem.len() {
            return None;
        }
        if !self.perms[off..end].iter().all(|&p| (p & need) == need && (p & FAST_PTR_FORBIDDEN) == 0) {
            return None;
        }
        if need & PERM_WRITE != 0 && self.track_dirty {
            // Mirror `write`'s dirty-block bookkeeping exactly (the caller performs the actual
            // content write through the returned pointer immediately after this returns) — one
            // `mark_dirty` per touched byte, same as `write_bytewise`'s per-byte loop (`len` is at
            // most 4, so this is cheap).
            for i in off..end {
                self.mark_dirty(i);
            }
        }
        // Safe: slicing (bounds-checked above) then taking the slice's own pointer — no raw
        // pointer arithmetic, so this needs no `unsafe` (this crate is `forbid(unsafe_code)`).
        // The pointer is only ever *dereferenced* by `fs-jit`'s isolated unsafe surface, which
        // does so from freshly emitted native code, not from Rust — see `fs-jit`'s `sys.rs`.
        Some(self.mem[off..end].as_mut_ptr())
    }
}

// ---------------------------------------------------------------------------------------------
// PR1 of the software COW shared-guest-RAM design (see `docs/cow-shared-ram.md`): an immutable
// golden snapshot of an `Mmu`'s two planes, shared read-only via `Arc<Golden>` across every lane
// / core, plus a per-lane `CowRam` that copy-on-writes individual 4 KiB pages on first divergence.
// Everything below is purely additive — no existing `Mmu` method signature or behavior changes.
// ---------------------------------------------------------------------------------------------

/// Page granularity for the copy-on-write overlay: the guest's own 4 KiB paging unit (matches
/// sv32), so a shared read-only sv32 walk (PR4) can consult the same directory.
pub const PAGE_SIZE: usize = 4096;

/// `dir[pn]` sentinel meaning "page `pn` has no overlay yet — still golden".
const SENTINEL: u32 = u32::MAX;

/// An immutable golden guest-RAM snapshot, in the *exact* [`Mmu`] byte encoding (`mem` + `perms`
/// planes, byte-granular RWX/RAW preserved). Captured once (typically post-boot) via
/// [`Golden::from_mmu`]; from then on every lane/core shares the same image read-only. `Golden`
/// holds only `Vec<u8>`/`u32` fields (no interior mutability, no raw pointers), so it is `Send +
/// Sync` for free and nothing prevents wrapping it in `Arc<Golden>` for cheap, safe sharing.
pub struct Golden {
    base: u32,
    mem: Vec<u8>,
    perms: Vec<u8>,
}

impl Golden {
    /// Snapshot `m`'s current planes. This is the moment "golden" is defined: whatever `m` looks
    /// like right now is what every `CowRam::reset()` restores back to.
    pub fn from_mmu(m: &Mmu) -> Golden {
        let (mem, perms) = m.planes();
        Golden {
            base: m.base(),
            mem: mem.to_vec(),
            perms: perms.to_vec(),
        }
    }

    pub fn base(&self) -> u32 {
        self.base
    }
    pub fn size(&self) -> usize {
        self.mem.len()
    }
    /// Number of 4 KiB pages covering the window (the final page rounds up if `size()` isn't
    /// page-aligned — see `page_mem`/`page_perms` for how that partial tail is handled).
    pub fn num_pages(&self) -> usize {
        self.mem.len().div_ceil(PAGE_SIZE)
    }

    /// `[pn*4096 .. pn*4096+4096)`, clamped to `size()` for the final page.
    ///
    /// Tail handling: we do *not* pad `Golden`'s planes out to a whole number of pages. When
    /// `size()` isn't a multiple of 4096 the final page's slice is simply shorter than 4096
    /// bytes. `CowRam::offset()` mirrors `Mmu::offset` exactly (bounds-checked against `size()`,
    /// not against a page-rounded size), so no in-page index ever reaches past this clamped
    /// length — the missing tail bytes are never observable, and there is no golden content to
    /// invent for them.
    fn page_mem(&self, pn: usize) -> &[u8] {
        let start = pn * PAGE_SIZE;
        let end = (start + PAGE_SIZE).min(self.mem.len());
        &self.mem[start..end]
    }
    fn page_perms(&self, pn: usize) -> &[u8] {
        let start = pn * PAGE_SIZE;
        let end = (start + PAGE_SIZE).min(self.perms.len());
        &self.perms[start..end]
    }

    // -----------------------------------------------------------------------------------------
    // PR4 of the software COW design (`docs/cow-shared-ram.md`): direct, read-only physical
    // access into the golden image itself — no `Bus`, no `CowRam`, no mutation. Consumed by
    // `fs_riscv::Cpu::xlate_golden_readonly` (PTE reads) and `fs_vec::VecSystem`'s converged
    // shared-fetch/shared-load fast path (instruction/data reads). Permission gating mirrors
    // `Mmu` exactly (`read_*` requires `PERM_READ`, matching `Bus::load`'s gate that the mutating
    // `xlate`'s PTE reads go through; `fetch_u16` requires `PERM_EXEC`, matching `Mmu::fetch_u16`/
    // `Bus::ifetch16`) so a `None` here is exactly the set of cases the mutating path would fault
    // or need to touch state for — the caller's only correct response is to decline to the
    // per-lane path, never to synthesize a value.
    // -----------------------------------------------------------------------------------------

    #[inline]
    fn offset(&self, addr: u32) -> Option<usize> {
        if addr < self.base {
            return None;
        }
        let off = (addr - self.base) as usize;
        (off < self.mem.len()).then_some(off)
    }

    /// Physical byte read requiring `PERM_READ`. Mirrors `Mmu::read_u8`'s permission gate.
    pub fn read_u8(&self, addr: u32) -> Option<u8> {
        let off = self.offset(addr)?;
        if self.perms[off] & PERM_READ == 0 {
            return None;
        }
        Some(self.mem[off])
    }

    /// Physical 2-byte read (2-byte aligned), requiring `PERM_READ` on both bytes. Mirrors
    /// `Mmu::read_u16`.
    pub fn read_u16(&self, addr: u32) -> Option<u16> {
        if !addr.is_multiple_of(2) {
            return None;
        }
        let off = self.offset(addr)?;
        if off + 2 > self.mem.len() {
            return None;
        }
        if !self.perms[off..off + 2].iter().all(|&p| p & PERM_READ != 0) {
            return None;
        }
        Some(u16::from_le_bytes(self.mem[off..off + 2].try_into().unwrap()))
    }

    /// Physical 4-byte read (4-byte aligned), requiring `PERM_READ` on all four bytes. Mirrors
    /// `Mmu::read_u32` — this is what a read-only sv32 walk uses to fetch a PTE straight from the
    /// golden image instead of through a mutating `Bus::load`.
    pub fn read_u32(&self, addr: u32) -> Option<u32> {
        if !addr.is_multiple_of(4) {
            return None;
        }
        let off = self.offset(addr)?;
        if off + 4 > self.mem.len() {
            return None;
        }
        if !self.perms[off..off + 4].iter().all(|&p| p & PERM_READ != 0) {
            return None;
        }
        Some(u32::from_le_bytes(self.mem[off..off + 4].try_into().unwrap()))
    }

    /// Sized (1/2/4-byte) physical read requiring `PERM_READ`, matching `Mmu::load`'s dispatch —
    /// used by `VecSystem`'s converged same-address LOAD fast path to broadcast one golden read
    /// instead of `LANES` independent per-lane loads.
    pub fn read_sized(&self, addr: u32, size: u8) -> Option<u32> {
        match size {
            1 => self.read_u8(addr).map(u32::from),
            2 => self.read_u16(addr).map(u32::from),
            _ => self.read_u32(addr),
        }
    }

    /// Instruction half-word fetch (2-byte aligned), requiring `PERM_EXEC` (not `PERM_READ`) on
    /// both bytes. Mirrors `Mmu::fetch_u16` — used by `VecSystem`'s converged shared-fetch fast
    /// path instead of a per-lane `Bus::ifetch16`.
    pub fn fetch_u16(&self, addr: u32) -> Option<u16> {
        if !addr.is_multiple_of(2) {
            return None;
        }
        let off = self.offset(addr)?;
        if off + 2 > self.mem.len() {
            return None;
        }
        if !self.perms[off..off + 2].iter().all(|&p| p & PERM_EXEC != 0) {
            return None;
        }
        Some(u16::from_le_bytes(self.mem[off..off + 2].try_into().unwrap()))
    }
}

/// One page's private copy-on-write overlay: 4 KiB of guest memory plus its parallel permission
/// plane, always copied together from the same golden page on first divergence (never mixed).
struct CowPage {
    mem: [u8; PAGE_SIZE],
    perms: [u8; PAGE_SIZE],
}

/// A per-lane (or per-thread) working view over a shared [`Golden`] image. Unmodified pages read
/// straight through to golden (zero copy — safe to call speculatively); a write (or any other
/// perm mutation) copy-on-writes the containing 4 KiB page into a private overlay first. Access
/// semantics (fast-path / `#[cold]` bytewise fallback / RAW-upgrade-on-write / exact [`Fault`]
/// reporting) mirror `Mmu` byte-for-byte — see `tests/cow_ram.rs` for the differential proof.
///
/// A single call into `read`/`write`/`fetch_u16`/`fetch_u32` is expected to stay within one 4 KiB
/// page: the emulator (`fs-riscv`) already splits misaligned/page-crossing Bus accesses into
/// per-byte accesses upstream, and a naturally-aligned 1/2/4-byte access never crosses a 4 KiB
/// boundary. The bytewise fallback below resolves each byte independently by its own page number,
/// so nothing panics or misbehaves even if that assumption were ever violated — it is simply the
/// only case that matters for the fast path's single-page slice arithmetic.
pub struct CowRam {
    golden: Arc<Golden>,
    /// `dir[pn]` = index into `pages`, or `SENTINEL` if page `pn` is still golden.
    dir: Vec<u32>,
    pages: Vec<Box<CowPage>>,
    /// Overlaid page numbers, in COW order — lets `reset()` be O(dirty) instead of O(all pages).
    dirty: Vec<u32>,
}

impl CowRam {
    /// A fresh view over `golden`: every page starts golden (no overlay allocated yet).
    pub fn new(golden: Arc<Golden>) -> Self {
        let n = golden.num_pages();
        CowRam {
            golden,
            dir: vec![SENTINEL; n],
            pages: Vec::new(),
            dirty: Vec::new(),
        }
    }

    #[inline]
    fn offset(&self, addr: u32) -> Option<usize> {
        let base = self.golden.base();
        if addr < base {
            return None;
        }
        let off = (addr - base) as usize;
        (off < self.golden.size()).then_some(off)
    }

    #[inline]
    fn fault(addr: u32, len: u32, access: Access, kind: FaultKind) -> Fault {
        Fault { addr, len, access, kind }
    }

    /// Resolve page `pn` to its current (mem, perms) 4 KiB slices — from the overlay if this page
    /// was COW'd, else straight from golden. Both planes always come from the same source, so
    /// content and perms can never desync.
    #[inline]
    fn resolve(&self, pn: usize) -> (&[u8], &[u8]) {
        let slot = self.dir[pn];
        if slot == SENTINEL {
            (self.golden.page_mem(pn), self.golden.page_perms(pn))
        } else {
            let page = &self.pages[slot as usize];
            (&page.mem[..], &page.perms[..])
        }
    }

    /// Copy-on-write page `pn` (a no-op if already overlaid): clone golden's 4096 mem + 4096 perm
    /// bytes into a fresh private page, and record it in `dir`/`dirty`. Returns the slot in
    /// `pages`. If `pn` is the final, partial golden page, only the valid (clamped) bytes are
    /// copied into the start of the 4096-byte overlay; the tail is left zeroed and is never
    /// addressable (see `Golden::page_mem`).
    fn ensure_page(&mut self, pn: usize) -> usize {
        let slot = self.dir[pn];
        if slot != SENTINEL {
            return slot as usize;
        }
        let mut page = Box::new(CowPage {
            mem: [0u8; PAGE_SIZE],
            perms: [0u8; PAGE_SIZE],
        });
        let gmem = self.golden.page_mem(pn);
        let gperms = self.golden.page_perms(pn);
        page.mem[..gmem.len()].copy_from_slice(gmem);
        page.perms[..gperms.len()].copy_from_slice(gperms);
        let idx = self.pages.len();
        self.pages.push(page);
        self.dir[pn] = idx as u32;
        self.dirty.push(pn as u32);
        idx
    }

    /// Read-only permission byte at `addr` (mirrors `Mmu::perm_at`). `None` outside the window.
    pub fn perm_at(&self, addr: u32) -> Option<u8> {
        self.offset(addr).map(|off| {
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            self.resolve(pn).1[po]
        })
    }

    /// Checked read: every byte must carry `PERM_READ`. Never allocates an overlay.
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Fault> {
        let len = buf.len();
        if let Some(off) = self.offset(addr) {
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            if off + len <= self.golden.size() && po + len <= PAGE_SIZE {
                let (mem, perms) = self.resolve(pn);
                if perms[po..po + len].iter().all(|&p| p & PERM_READ != 0) {
                    buf.copy_from_slice(&mem[po..po + len]);
                    return Ok(());
                }
            }
        }
        self.read_bytewise(addr, buf)
    }

    #[cold]
    fn read_bytewise(&self, addr: u32, buf: &mut [u8]) -> Result<(), Fault> {
        let len = buf.len() as u32;
        for (i, out) in buf.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, len, Access::Read, FaultKind::Unmapped))?;
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            let (mem, perms) = self.resolve(pn);
            if perms[po] & PERM_READ == 0 {
                return Err(Self::fault(a, len, Access::Read, FaultKind::Permission));
            }
            *out = mem[po];
        }
        Ok(())
    }

    /// Checked write: every byte must carry `PERM_WRITE`. Copy-on-writes the containing page
    /// (only once permission is confirmed), then upgrades written bytes RAW -> READ. Mirrors
    /// `Mmu::write` exactly, including the bytewise fallback's partial-write-then-fault behavior
    /// on a mid-span permission miss (bytes before the faulting one are already committed).
    pub fn write(&mut self, addr: u32, data: &[u8]) -> Result<(), Fault> {
        let len = data.len();
        if let Some(off) = self.offset(addr) {
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            if off + len <= self.golden.size() && po + len <= PAGE_SIZE {
                let writable = {
                    let (_, perms) = self.resolve(pn);
                    perms[po..po + len].iter().all(|&p| p & PERM_WRITE != 0)
                };
                if writable {
                    let idx = self.ensure_page(pn);
                    let page = &mut self.pages[idx];
                    page.mem[po..po + len].copy_from_slice(data);
                    for p in &mut page.perms[po..po + len] {
                        *p = (*p | PERM_READ) & !PERM_RAW;
                    }
                    return Ok(());
                }
            }
        }
        self.write_bytewise(addr, data)
    }

    #[cold]
    fn write_bytewise(&mut self, addr: u32, data: &[u8]) -> Result<(), Fault> {
        let len = data.len() as u32;
        for (i, b) in data.iter().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, len, Access::Write, FaultKind::Unmapped))?;
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            let writable = self.resolve(pn).1[po] & PERM_WRITE != 0;
            if !writable {
                return Err(Self::fault(a, len, Access::Write, FaultKind::Permission));
            }
            let idx = self.ensure_page(pn);
            let page = &mut self.pages[idx];
            page.mem[po] = *b;
            page.perms[po] = (page.perms[po] | PERM_READ) & !PERM_RAW;
        }
        Ok(())
    }

    #[inline]
    fn check_align(addr: u32, n: u32, access: Access) -> Result<(), Fault> {
        if addr.is_multiple_of(n) {
            Ok(())
        } else {
            Err(Self::fault(addr, n, access, FaultKind::Unaligned))
        }
    }

    pub fn read_u8(&self, addr: u32) -> Result<u8, Fault> {
        let mut b = [0u8; 1];
        self.read(addr, &mut b)?;
        Ok(b[0])
    }
    pub fn read_u16(&self, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Read)?;
        let mut b = [0u8; 2];
        self.read(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    pub fn read_u32(&self, addr: u32) -> Result<u32, Fault> {
        Self::check_align(addr, 4, Access::Read)?;
        let mut b = [0u8; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn write_u8(&mut self, addr: u32, v: u8) -> Result<(), Fault> {
        self.write(addr, &[v])
    }
    pub fn write_u16(&mut self, addr: u32, v: u16) -> Result<(), Fault> {
        Self::check_align(addr, 2, Access::Write)?;
        self.write(addr, &v.to_le_bytes())
    }
    pub fn write_u32(&mut self, addr: u32, v: u32) -> Result<(), Fault> {
        Self::check_align(addr, 4, Access::Write)?;
        self.write(addr, &v.to_le_bytes())
    }

    #[inline]
    fn fetch_bytes<const N: usize>(&self, addr: u32) -> Result<[u8; N], Fault> {
        let mut b = [0u8; N];
        if let Some(off) = self.offset(addr) {
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            if off + N <= self.golden.size() && po + N <= PAGE_SIZE {
                let (mem, perms) = self.resolve(pn);
                if perms[po..po + N].iter().all(|&p| p & PERM_EXEC != 0) {
                    b.copy_from_slice(&mem[po..po + N]);
                    return Ok(b);
                }
            }
        }
        for (i, out) in b.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, N as u32, Access::Exec, FaultKind::Unmapped))?;
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            let (mem, perms) = self.resolve(pn);
            if perms[po] & PERM_EXEC == 0 {
                return Err(Self::fault(a, N as u32, Access::Exec, FaultKind::Permission));
            }
            *out = mem[po];
        }
        Ok(b)
    }

    /// Instruction half-word fetch: 2-byte aligned, every byte must carry PERM_EXEC.
    pub fn fetch_u16(&self, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Exec)?;
        Ok(u16::from_le_bytes(self.fetch_bytes::<2>(addr)?))
    }
    /// Instruction fetch: 4-byte aligned, every byte must carry PERM_EXEC.
    pub fn fetch_u32(&self, addr: u32) -> Result<u32, Fault> {
        Self::check_align(addr, 4, Access::Exec)?;
        Ok(u32::from_le_bytes(self.fetch_bytes::<4>(addr)?))
    }

    /// `fs_mmu::Bus::fast_ptr`'s twin for `CowRam` (`CowRam` itself doesn't implement `Bus` — it's
    /// wrapped by `fs_platform::CowMachine`, which forwards here). See that trait method's doc for
    /// the full contract; the only `CowRam`-specific wrinkle is on a store (`need & PERM_WRITE !=
    /// 0`): the containing page is copy-on-write materialized via [`CowRam::ensure_page`] — the
    /// exact same call [`CowRam::write`] itself makes — so the dirty/overlay bookkeeping this
    /// fast path must perform (mirroring the trait doc's "already performed every side effect
    /// except the content write" contract) is identical to the checked path's, not a second copy
    /// of it.
    pub fn fast_ptr(&mut self, addr: u32, len: u8, need: u8) -> Option<*mut u8> {
        let off = self.offset(addr)?;
        let pn = off / PAGE_SIZE;
        let po = off % PAGE_SIZE;
        let end = off.checked_add(len as usize)?;
        if end > self.golden.size() || po + len as usize > PAGE_SIZE {
            return None;
        }
        {
            let (_, perms) = self.resolve(pn);
            if !perms[po..po + len as usize]
                .iter()
                .all(|&p| (p & need) == need && (p & FAST_PTR_FORBIDDEN) == 0)
            {
                return None;
            }
        }
        if need & PERM_WRITE != 0 {
            let idx = self.ensure_page(pn);
            let page = &mut self.pages[idx];
            Some(page.mem[po..po + len as usize].as_mut_ptr())
        } else {
            let (mem, _) = self.resolve(pn);
            // Safe pointer-type cast (not a dereference) — turning the shared, read-only `&[u8]`
            // this load will only ever be read through into a `*mut u8` so it has the same type
            // as the write branch above; the caller (`fs-jit`) never writes through a pointer this
            // function returned for a `need` that lacked `PERM_WRITE`.
            Some(mem[po..po + len as usize].as_ptr() as *mut u8)
        }
    }

    /// `fs_mmu::Bus::read_raw_state`'s twin for `CowRam` (mirrors [`CowRam::fast_ptr`]'s "`CowRam`
    /// doesn't implement `Bus` itself, `fs_platform::CowMachine` forwards here" shape). Never
    /// allocates an overlay — reading VTAINT/RAW state, like reading content, is safe straight
    /// through golden for a page that has never diverged.
    pub fn read_raw_state(&self, addr: u32, len: u8) -> u32 {
        let mut mask = 0u32;
        for i in 0..(len.min(4) as u32) {
            let Some(off) = self.offset(addr.wrapping_add(i)) else { continue };
            let pn = off / PAGE_SIZE;
            let po = off % PAGE_SIZE;
            if po >= PAGE_SIZE {
                continue;
            }
            if self.resolve(pn).1[po] & (PERM_RAW | PERM_VTAINT) != 0 {
                mask |= 1 << (8 * i);
            }
        }
        mask
    }

    /// `fs_mmu::Bus::write_shadow`'s twin for `CowRam`. Copy-on-writes the containing page (exactly
    /// like [`CowRam::write`]) before mutating `PERM_VTAINT` — mutating a still-golden page in
    /// place would corrupt every other lane sharing the same `Arc<Golden>`.
    pub fn write_shadow(&mut self, addr: u32, len: u8, taint_mask: u32) {
        let Some(off) = self.offset(addr) else { return };
        let pn = off / PAGE_SIZE;
        let po = off % PAGE_SIZE;
        let n = len.min(4) as usize;
        if po + n > PAGE_SIZE {
            return; // caller's alignment guarantee (see `Bus::write_shadow`'s doc) never crosses a page
        }
        let idx = self.ensure_page(pn);
        let page = &mut self.pages[idx];
        for i in 0..n {
            if (taint_mask >> (8 * i)) & 1 != 0 {
                page.perms[po + i] |= PERM_VTAINT;
            } else {
                page.perms[po + i] &= !PERM_VTAINT;
            }
        }
    }

    /// Bus-shaped helpers, matching how `fs_platform::Machine` calls its `Mmu` (see PR2's
    /// `CowMachine`). `load`/`ifetch16` take `&self` (reads never allocate, safe to call
    /// speculatively); `store` takes `&mut self` since a write may copy-on-write a page.
    pub fn load(&self, addr: u32, size: u8) -> Result<u32, Fault> {
        match size {
            1 => self.read_u8(addr).map(u32::from),
            2 => self.read_u16(addr).map(u32::from),
            _ => self.read_u32(addr),
        }
    }
    pub fn store(&mut self, addr: u32, size: u8, val: u32) -> Result<(), Fault> {
        match size {
            1 => self.write_u8(addr, val as u8),
            2 => self.write_u16(addr, val as u16),
            _ => self.write_u32(addr, val),
        }
    }
    pub fn ifetch16(&self, addr: u32) -> Result<u16, Fault> {
        self.fetch_u16(addr)
    }

    /// O(dirty) reset: drop every overlaid page back to golden. Zero byte copy-back — golden is
    /// never mutated, so there is nothing to copy, just overlays to forget.
    pub fn reset(&mut self) {
        for &pn in &self.dirty {
            self.dir[pn as usize] = SENTINEL;
        }
        self.pages.clear();
        self.dirty.clear();
    }

    /// Overlaid page numbers since the last `reset()` (diagnostics / memory-drop reporting).
    pub fn dirty_pages(&self) -> &[u32] {
        &self.dirty
    }

    /// Whether page `pn` currently has a private overlay (`true`) or is still golden (`false`).
    /// Out-of-range `pn` reports `false` (there is nothing to overlay). Consumed by PR4's shared
    /// translate/fetch fast path (`docs/cow-shared-ram.md`): a code/data page any lane has
    /// privately COW'd (self-modified) must never have that divergence painted over by a
    /// golden-broadcast read.
    pub fn is_overlaid(&self, pn: usize) -> bool {
        self.dir.get(pn).is_some_and(|&d| d != SENTINEL)
    }
}

/// Cross-lane "this page was privately COW'd by someone" bitmap (1 bit / 4 KiB page). Consumed by
/// PR4's shared translate/fetch fast path to decide whether a golden-broadcast result is still
/// safe for every lane, or a page must fall back to per-lane resolution. Defined now so the
/// bit-twiddling is exercised in isolation; `CowRam::ensure_page` does not yet set it (that wiring
/// is PR4's job, once there's an actual cross-lane consumer to keep in sync).
pub struct SharedDirty(Vec<u64>);

impl SharedDirty {
    pub fn new(num_pages: usize) -> Self {
        SharedDirty(vec![0u64; num_pages.div_ceil(64)])
    }
    pub fn set(&mut self, pn: usize) {
        let (w, b) = (pn / 64, pn % 64);
        self.0[w] |= 1u64 << b;
    }
    pub fn bit(&self, pn: usize) -> bool {
        let (w, b) = (pn / 64, pn % 64);
        self.0[w] & (1u64 << b) != 0
    }
    pub fn clear_pages(&mut self, pns: &[u32]) {
        for &pn in pns {
            let (w, b) = (pn as usize / 64, pn as usize % 64);
            self.0[w] &= !(1u64 << b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_uninit_read_faults_until_written() {
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        // Allocate 4 bytes as RAW|WRITE (no READ).
        m.protect(0x8000_0000, 4, PERM_RAW | PERM_WRITE).unwrap();
        // Reading uninitialized bytes faults.
        assert_eq!(
            m.read_u8(0x8000_0000).unwrap_err().kind,
            FaultKind::Permission
        );
        // Writing one byte unlocks reads of exactly that byte.
        m.write_u8(0x8000_0000, 0xAB).unwrap();
        assert_eq!(m.read_u8(0x8000_0000).unwrap(), 0xAB);
        // The next (still-uninitialized) byte still faults.
        assert_eq!(
            m.read_u8(0x8000_0001).unwrap_err().kind,
            FaultKind::Permission
        );
    }

    #[test]
    fn unmapped_and_unaligned() {
        let m = Mmu::new(0x8000_0000, 0x1000);
        assert_eq!(m.read_u8(0x1234).unwrap_err().kind, FaultKind::Unmapped);
        assert_eq!(
            m.read_u32(0x8000_0001).unwrap_err().kind,
            FaultKind::Unaligned
        );
    }

    #[test]
    fn poison_faults_read_and_write_but_stays_mapped() {
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        m.protect(0x8000_0000, 8, PERM_READ | PERM_WRITE).unwrap();
        m.write_u8(0x8000_0000, 0x41).unwrap();
        m.poison(0x8000_0000, 8).unwrap();
        // Address is still inside the mapped window (Permission, not Unmapped).
        assert_eq!(
            m.read_u8(0x8000_0000).unwrap_err().kind,
            FaultKind::Permission
        );
        assert_eq!(
            m.write_u8(0x8000_0000, 0x42).unwrap_err().kind,
            FaultKind::Permission
        );
        assert_eq!(m.perm_at(0x8000_0000), Some(0));
    }

    #[test]
    fn in_bounds_checks_window_membership() {
        let m = Mmu::new(0x8000_0000, 0x1000);
        assert!(m.in_bounds(0x8000_0000, 0x1000));
        assert!(!m.in_bounds(0x8000_0000, 0x1001));
        assert!(!m.in_bounds(0x7fff_fffc, 8)); // starts before base
        assert!(!m.in_bounds(0xffff_fff8, 16)); // overflows u32 on addr+len-1
        assert!(m.in_bounds(0x8000_1000, 0)); // empty range at the exclusive end is fine
        assert!(!m.in_bounds(0x8000_1000, 1)); // one past the end is not
    }

    #[test]
    fn perm_at_reports_none_outside_window() {
        let m = Mmu::new(0x8000_0000, 0x1000);
        assert_eq!(m.perm_at(0x1234), None);
        assert_eq!(m.perm_at(0x8000_0000), Some(0));
    }

    // -- CowRam / Golden / SharedDirty targeted tests (the differential property test lives in
    // -- `tests/cow_ram.rs`, which needs the crate as an external dependency) --

    #[test]
    fn cow_reset_drops_overlays_and_reverts_memory() {
        let mut m = Mmu::new(0x8000_0000, 0x4000); // 4 pages
        m.map(0x8000_0000, &[0xAA; 16], PERM_READ | PERM_WRITE)
            .unwrap();
        // Page 2 (0x8000_2000) is mapped RW in golden too, but starts as all-zero content.
        m.protect(0x8000_2000, 4, PERM_READ | PERM_WRITE).unwrap();
        let golden = Arc::new(Golden::from_mmu(&m));
        let mut cow = CowRam::new(golden);

        // Untouched: reads straight through, no overlay allocated.
        assert_eq!(cow.read_u32(0x8000_0000).unwrap(), u32::from_le_bytes([0xAA; 4]));
        assert!(cow.dirty_pages().is_empty());

        // Diverge page 0 and page 2.
        cow.write_u32(0x8000_0000, 0x1111_1111).unwrap();
        cow.write_u32(0x8000_2000, 0x2222_2222).unwrap();
        assert_eq!(cow.read_u32(0x8000_0000).unwrap(), 0x1111_1111);
        assert_eq!(cow.read_u32(0x8000_2000).unwrap(), 0x2222_2222);
        assert_eq!(cow.dirty_pages().len(), 2);

        cow.reset();
        assert!(cow.dirty_pages().is_empty());
        // Both pages revert to golden content (0xAA-filled page 0, zeroed page 2), not to the
        // values written before reset.
        assert_eq!(cow.read_u32(0x8000_0000).unwrap(), u32::from_le_bytes([0xAA; 4]));
        assert_eq!(cow.read_u32(0x8000_2000).unwrap(), 0);

        // Re-diverge proves the page can be COW'd again after reset.
        cow.write_u32(0x8000_0000, 0x3333_3333).unwrap();
        assert_eq!(cow.read_u32(0x8000_0000).unwrap(), 0x3333_3333);
        assert_eq!(cow.dirty_pages(), &[0]);
    }

    #[test]
    fn cow_content_and_perms_come_from_same_source() {
        // Regression guard for "never mix golden mem with overlay perms": write unlocks READ on
        // exactly the overlay, and a subsequent poison stays consistent with the overlaid content.
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        m.protect(0x8000_0000, 4, PERM_RAW | PERM_WRITE).unwrap();
        let golden = Arc::new(Golden::from_mmu(&m));
        let mut cow = CowRam::new(golden);

        assert_eq!(
            cow.read_u8(0x8000_0000).unwrap_err().kind,
            FaultKind::Permission
        );
        cow.write_u8(0x8000_0000, 0xAB).unwrap();
        assert_eq!(cow.read_u8(0x8000_0000).unwrap(), 0xAB);
        assert_eq!(cow.perm_at(0x8000_0000), Some(PERM_READ | PERM_WRITE));
    }

    #[test]
    fn cow_is_overlaid_tracks_dirty_pages() {
        let mut m = Mmu::new(0x8000_0000, 0x4000); // 4 pages
        m.protect(0x8000_0000, 0x4000, PERM_READ | PERM_WRITE).unwrap();
        let golden = Arc::new(Golden::from_mmu(&m));
        let mut cow = CowRam::new(golden);

        assert!(!cow.is_overlaid(0));
        assert!(!cow.is_overlaid(2));
        // Out-of-range page number reports false rather than panicking.
        assert!(!cow.is_overlaid(999));

        cow.write_u32(0x8000_2000, 0x1234_5678).unwrap(); // dirties page 2 only
        assert!(!cow.is_overlaid(0));
        assert!(cow.is_overlaid(2));
        assert!(!cow.is_overlaid(3));

        cow.reset();
        assert!(!cow.is_overlaid(2));
    }

    #[test]
    fn golden_read_and_fetch_require_the_right_permission_bit() {
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        m.map(0x8000_0000, &0x1122_3344u32.to_le_bytes(), PERM_READ | PERM_WRITE)
            .unwrap();
        m.map(0x8000_0010, &0xBEEFu16.to_le_bytes(), PERM_EXEC).unwrap(); // exec-only, no READ
        let golden = Golden::from_mmu(&m);

        // read_* requires PERM_READ.
        assert_eq!(golden.read_u32(0x8000_0000), Some(0x1122_3344));
        assert_eq!(golden.read_u16(0x8000_0000), Some(0x3344));
        assert_eq!(golden.read_u8(0x8000_0000), Some(0x44));
        assert_eq!(golden.read_sized(0x8000_0000, 4), Some(0x1122_3344));
        // Exec-only bytes lack PERM_READ, so a data read of them declines.
        assert_eq!(golden.read_u16(0x8000_0010), None);

        // fetch_u16 requires PERM_EXEC, not PERM_READ.
        assert_eq!(golden.fetch_u16(0x8000_0010), Some(0xBEEF));
        // The RW (non-exec) word has no PERM_EXEC, so fetching it declines.
        assert_eq!(golden.fetch_u16(0x8000_0000), None);

        // Misaligned / out-of-bounds declines rather than panicking.
        assert_eq!(golden.read_u32(0x8000_0001), None);
        assert_eq!(golden.fetch_u16(0x1234), None);
    }

    #[test]
    fn shared_dirty_set_bit_clear() {
        let mut sd = SharedDirty::new(200); // spans 4 u64 words
        assert!(!sd.bit(0));
        assert!(!sd.bit(63));
        assert!(!sd.bit(64));
        assert!(!sd.bit(199));

        sd.set(0);
        sd.set(63);
        sd.set(64);
        sd.set(199);
        assert!(sd.bit(0));
        assert!(sd.bit(63));
        assert!(sd.bit(64));
        assert!(sd.bit(199));
        assert!(!sd.bit(1));

        sd.clear_pages(&[63, 199]);
        assert!(sd.bit(0));
        assert!(!sd.bit(63));
        assert!(sd.bit(64));
        assert!(!sd.bit(199));
    }

    // -- KMSAN Stage 2 (`docs/kmsan.md`): PERM_VTAINT shadow read/write --

    #[test]
    fn set_vtaint_bulk_seeding_preserves_other_perm_bits() {
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        m.protect(0x8000_0000, 8, PERM_READ | PERM_WRITE).unwrap();
        m.set_vtaint(0x8000_0000, 4, true).unwrap();
        // Tainted bytes: VTAINT set, R/W untouched.
        assert_eq!(
            m.perm_at(0x8000_0000),
            Some(PERM_READ | PERM_WRITE | PERM_VTAINT)
        );
        // Untouched tail (outside the seeded range) carries no VTAINT.
        assert_eq!(m.perm_at(0x8000_0004), Some(PERM_READ | PERM_WRITE));
        // Gather (the `Bus::read_raw_state` load-taint source) sees the seeded bytes as tainted,
        // even though they carry no PERM_RAW (KMSAN *permits* the read, unlike RAW's fault).
        assert_eq!(m.read_raw_state(0x8000_0000, 4), 0x0101_0101);
        assert_eq!(m.read_u8(0x8000_0000).unwrap(), 0); // VTAINT alone never faults a read
        // Clearing un-seeds without disturbing R/W.
        m.set_vtaint(0x8000_0000, 4, false).unwrap();
        assert_eq!(m.perm_at(0x8000_0000), Some(PERM_READ | PERM_WRITE));
        assert_eq!(m.read_raw_state(0x8000_0000, 4), 0);
    }

    #[test]
    fn read_raw_state_ors_raw_and_vtaint() {
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        // Byte 0: RAW only. Byte 1: VTAINT only. Byte 2: both. Byte 3: neither.
        m.protect(0x8000_0000, 1, PERM_READ | PERM_WRITE | PERM_RAW).unwrap();
        m.protect(0x8000_0001, 1, PERM_READ | PERM_WRITE | PERM_VTAINT).unwrap();
        m.protect(0x8000_0002, 1, PERM_READ | PERM_WRITE | PERM_RAW | PERM_VTAINT).unwrap();
        m.protect(0x8000_0003, 1, PERM_READ | PERM_WRITE).unwrap();
        assert_eq!(m.read_raw_state(0x8000_0000, 4), 0x0001_0101);
    }

    #[test]
    fn write_shadow_is_an_exact_overwrite_not_an_or() {
        let mut m = Mmu::new(0x8000_0000, 0x1000);
        m.protect(0x8000_0000, 4, PERM_READ | PERM_WRITE | PERM_VTAINT).unwrap();
        // Scatter taint=clean into bytes 0/1/3 (bits 0/8/24 clear), tainted into byte 2 (bit16 set).
        m.write_shadow(0x8000_0000, 4, 0x0001_0000);
        assert_eq!(m.perm_at(0x8000_0000), Some(PERM_READ | PERM_WRITE));
        assert_eq!(m.perm_at(0x8000_0001), Some(PERM_READ | PERM_WRITE));
        assert_eq!(m.perm_at(0x8000_0002), Some(PERM_READ | PERM_WRITE | PERM_VTAINT));
        assert_eq!(m.perm_at(0x8000_0003), Some(PERM_READ | PERM_WRITE));
        // Overwriting the whole span with all-tainted works too.
        m.write_shadow(0x8000_0000, 4, 0xffff_ffff);
        assert_eq!(m.read_raw_state(0x8000_0000, 4), 0x0101_0101);
    }

    #[test]
    fn cow_ram_write_shadow_taints_overlay_not_golden() {
        let mut m = Mmu::new(0x8000_0000, 0x4000);
        m.protect(0x8000_0000, 0x4000, PERM_READ | PERM_WRITE).unwrap();
        let golden = Arc::new(Golden::from_mmu(&m));
        let mut cow = CowRam::new(golden);

        assert_eq!(cow.read_raw_state(0x8000_0000, 4), 0);
        assert!(!cow.is_overlaid(0));

        cow.write_shadow(0x8000_0000, 4, 0x0000_0001); // taint byte 0 only
        assert!(cow.is_overlaid(0)); // mutating VTAINT COWs the page
        assert_eq!(cow.read_raw_state(0x8000_0000, 4), 0x0000_0001);
        assert_eq!(cow.perm_at(0x8000_0000), Some(PERM_READ | PERM_WRITE | PERM_VTAINT));

        // A second, independent `CowRam` over the same golden never observes the first's taint.
        let cow2 = CowRam::new(cow_golden_of(&cow));
        assert_eq!(cow2.read_raw_state(0x8000_0000, 4), 0);

        cow.reset();
        assert!(!cow.is_overlaid(0));
        assert_eq!(cow.read_raw_state(0x8000_0000, 4), 0);
    }

    /// Test-only helper: hand back the same `Arc<Golden>` a `CowRam` was built over, so a second
    /// independent lane over that identical golden image can be constructed for the isolation
    /// check above (mirrors how multiple lanes/cores share one `Arc<Golden>` in the real COW design).
    fn cow_golden_of(cow: &CowRam) -> Arc<Golden> {
        Arc::clone(&cow.golden)
    }
}
