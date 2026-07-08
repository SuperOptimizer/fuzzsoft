//! Prototype: host-mmap-backed guest RAM for the scalar / thread-parallel (`--jobs`) fuzzing path.
//!
//! `fs_mmu::Mmu` (the audited soft-MMU) is a *software* flat `Vec<u8>` + parallel byte-permission
//! plane with O(dirty) 64-byte-block reset tracking. It is correct and byte-granular, but every
//! `--jobs` thread pays a full private 128 MB copy and a from-scratch reset walk. This crate
//! explores the gamozolabs-style alternative for that specific path: back guest RAM with a Linux
//! **memfd** (the golden image, written once) mapped `MAP_PRIVATE` per thread, so the kernel
//! copy-on-write-shares untouched golden pages across threads and `MADV_DONTNEED` resets a
//! thread's divergence in one syscall instead of a software copy-back.
//!
//! Byte-granular RWX/RAW permissions — the actual bug-finding oracle — are preserved by keeping a
//! *second* memfd+`MAP_PRIVATE` mapping for the permission plane, reset the same way. Access
//! semantics (fast-path / bytewise-fallback / RAW-upgrade-on-write / exact [`Fault`] on miss)
//! deliberately mirror `fs_mmu::Mmu` byte-for-byte — see `tests/differential.rs`, which asserts
//! this crate is a drop-in-shaped, byte-exact stand-in.
//!
//! See `docs/cow-shared-ram.md` for the wider design context. This is exploratory measurement
//! (see `examples/bench.rs`), not yet wired into the fuzzer.
//!
//! This is the one crate in the workspace allowed `unsafe` (mmap/madvise/memfd live in [`sys`]);
//! every other crate keeps `forbid(unsafe_code)`.

mod sys;

use std::io;
use std::os::unix::io::RawFd;

pub use fs_mmu::{Access, Fault, FaultKind, PERM_ACC, PERM_EXEC, PERM_RAW, PERM_READ, PERM_WRITE};

/// A golden guest-RAM snapshot backing memory, held once in two anonymous memfds (mem + perms
/// planes). Cheap to spawn any number of independent [`HostMem`] working views from — each view
/// gets its own `MAP_PRIVATE` copy-on-write mapping over the *same* underlying pages, so untouched
/// pages are physically shared (page-cache-resident once) across every view.
pub struct Golden {
    base: u32,
    size: usize,
    mem_fd: RawFd,
    perm_fd: RawFd,
}

// SAFETY: `Golden` holds only plain fds (`RawFd` is just an `i32`) and value types; no raw
// pointers, so the auto-derived `Send`/`Sync` for those fields is genuinely sound — `new_view`
// only issues a fresh `mmap` syscall against the fd, which the kernel serializes safely.
impl Golden {
    /// Snapshot `mmu`'s current `mem`+`perms` planes into a fresh golden image. This is the
    /// moment "golden" is defined: whatever `mmu` looks like right now becomes what every
    /// `HostMem::reset()` restores back to.
    pub fn from_mmu(mmu: &fs_mmu::Mmu) -> io::Result<Self> {
        let (mem, perms) = mmu.planes();
        Self::from_planes(mmu.base(), mem, perms)
    }

    /// Snapshot raw `mem`+`perms` planes (must be equal length) into a fresh golden image.
    pub fn from_planes(base: u32, mem: &[u8], perms: &[u8]) -> io::Result<Self> {
        assert_eq!(mem.len(), perms.len(), "mem/perms planes must be the same length");
        let size = mem.len();
        let mem_fd = sys::memfd_create("fs-hostmem-mem", size)?;
        if let Err(e) = sys::pwrite_all(mem_fd, mem) {
            // SAFETY: `mem_fd` was just created by us above and is not used anywhere else yet.
            unsafe { sys::close(mem_fd) };
            return Err(e);
        }
        let perm_fd = match sys::memfd_create("fs-hostmem-perm", size) {
            Ok(fd) => fd,
            Err(e) => {
                // SAFETY: as above.
                unsafe { sys::close(mem_fd) };
                return Err(e);
            }
        };
        if let Err(e) = sys::pwrite_all(perm_fd, perms) {
            // SAFETY: both fds were just created by us above and are not used anywhere else yet.
            unsafe {
                sys::close(mem_fd);
                sys::close(perm_fd);
            }
            return Err(e);
        }
        Ok(Self { base, size, mem_fd, perm_fd })
    }

    /// Spawn a new independent working view: a fresh `MAP_PRIVATE` (copy-on-write) mapping of
    /// both planes over this golden image. Cheap — no bytes are copied; pages are faulted in
    /// (shared, read-only) lazily on first touch.
    pub fn new_view(&self) -> io::Result<HostMem> {
        let mem_ptr = sys::mmap_private(self.mem_fd, self.size)?;
        let perm_ptr = match sys::mmap_private(self.perm_fd, self.size) {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: `mem_ptr` was just returned by `mmap_private` above with this `size`
                // and nothing else references it yet.
                unsafe { sys::munmap(mem_ptr, self.size) };
                return Err(e);
            }
        };
        Ok(HostMem { base: self.base, size: self.size, mem_ptr, perm_ptr })
    }

    pub fn base(&self) -> u32 {
        self.base
    }
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Golden {
    fn drop(&mut self) {
        // SAFETY: these fds were created by us in `from_planes` and are not shared with any other
        // owner (views hold only the fd *value*, used transiently inside `mmap`, not a live
        // reference requiring the fd to stay open past their own mapping's lifetime).
        unsafe {
            sys::close(self.mem_fd);
            sys::close(self.perm_fd);
        }
    }
}

/// A host-mmap-backed guest RAM working view: one thread's copy-on-write divergence from a
/// [`Golden`] image. Access semantics (fast path, bytewise fallback, RAW-upgrade-on-write, exact
/// `Fault` reporting) mirror `fs_mmu::Mmu` field-for-field so it is a byte-exact stand-in.
pub struct HostMem {
    base: u32,
    size: usize,
    mem_ptr: *mut u8,
    perm_ptr: *mut u8,
}

impl HostMem {
    #[inline]
    fn mem_slice(&self) -> &[u8] {
        // SAFETY: `mem_ptr` is a live `mmap_private` mapping of `size` bytes for the lifetime of
        // `self` (dropped only in `Drop`); `&self` guarantees no concurrent mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.mem_ptr, self.size) }
    }
    #[inline]
    fn mem_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, plus `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.mem_ptr, self.size) }
    }
    #[inline]
    fn perm_slice(&self) -> &[u8] {
        // SAFETY: `perm_ptr` is a live `mmap_private` mapping of `size` bytes for the lifetime of
        // `self`; `&self` guarantees no concurrent mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.perm_ptr, self.size) }
    }
    #[inline]
    fn perm_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, plus `&mut self` guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.perm_ptr, self.size) }
    }

    pub fn base(&self) -> u32 {
        self.base
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn end(&self) -> u32 {
        self.base.wrapping_add(self.size as u32)
    }

    /// Raw contents + permission planes (mirrors `Mmu::planes`; used by the differential test and
    /// benchmarks to verify golden-restoration without going through the permission-checked API).
    pub fn planes(&self) -> (&[u8], &[u8]) {
        (self.mem_slice(), self.perm_slice())
    }

    #[inline]
    fn offset(&self, addr: u32) -> Option<usize> {
        if addr < self.base {
            return None;
        }
        let off = (addr - self.base) as usize;
        (off < self.size).then_some(off)
    }

    #[inline]
    fn fault(addr: u32, len: u32, access: Access, kind: FaultKind) -> Fault {
        Fault { addr, len, access, kind }
    }

    /// Read-only permission byte at `addr`. `None` if outside the mapped window.
    pub fn perm_at(&self, addr: u32) -> Option<u8> {
        self.offset(addr).map(|off| self.perm_slice()[off])
    }

    /// Checked read: every byte must carry `PERM_READ`. Fast path mirrors `Mmu::read` exactly.
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Fault> {
        let len = buf.len();
        if let Some(off) = self.offset(addr)
            && off + len <= self.size
            && self.perm_slice()[off..off + len].iter().all(|&p| p & PERM_READ != 0)
        {
            buf.copy_from_slice(&self.mem_slice()[off..off + len]);
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
            if self.perm_slice()[off] & PERM_READ == 0 {
                return Err(Self::fault(a, len, Access::Read, FaultKind::Permission));
            }
            *out = self.mem_slice()[off];
        }
        Ok(())
    }

    /// Checked write: every byte must carry `PERM_WRITE`. Writing upgrades RAW bytes to READ
    /// (clearing RAW). Fast path mirrors `Mmu::write` exactly.
    pub fn write(&mut self, addr: u32, data: &[u8]) -> Result<(), Fault> {
        let len = data.len();
        if let Some(off) = self.offset(addr)
            && off + len <= self.size
            && self.perm_slice()[off..off + len].iter().all(|&p| p & PERM_WRITE != 0)
        {
            self.mem_slice_mut()[off..off + len].copy_from_slice(data);
            for p in &mut self.perm_slice_mut()[off..off + len] {
                *p = (*p | PERM_READ) & !PERM_RAW;
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
            if self.perm_slice()[off] & PERM_WRITE == 0 {
                return Err(Self::fault(a, len, Access::Write, FaultKind::Permission));
            }
            self.mem_slice_mut()[off] = *b;
            let p = &mut self.perm_slice_mut()[off];
            *p = (*p | PERM_READ) & !PERM_RAW;
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
        if let Some(off) = self.offset(addr)
            && off + N <= self.size
            && self.perm_slice()[off..off + N].iter().all(|&p| p & PERM_EXEC != 0)
        {
            b.copy_from_slice(&self.mem_slice()[off..off + N]);
            return Ok(b);
        }
        for (i, out) in b.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, N as u32, Access::Exec, FaultKind::Unmapped))?;
            if self.perm_slice()[off] & PERM_EXEC == 0 {
                return Err(Self::fault(a, N as u32, Access::Exec, FaultKind::Permission));
            }
            *out = self.mem_slice()[off];
        }
        Ok(b)
    }

    pub fn fetch_u16(&self, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Exec)?;
        Ok(u16::from_le_bytes(self.fetch_bytes::<2>(addr)?))
    }
    pub fn fetch_u32(&self, addr: u32) -> Result<u32, Fault> {
        Self::check_align(addr, 4, Access::Exec)?;
        Ok(u32::from_le_bytes(self.fetch_bytes::<4>(addr)?))
    }

    /// O(1) reset: drop every COW-private (dirtied) page in both planes back to the golden
    /// memfd's pages via `MADV_DONTNEED`. The next access to a dropped page transparently
    /// refaults against the shared golden content — no software copy-back, unlike
    /// `Mmu::reset_dirty`.
    pub fn reset(&mut self) -> io::Result<()> {
        // SAFETY: `mem_ptr`/`perm_ptr` are live mappings owned exclusively by `self` (`&mut
        // self`), each of exactly `self.size` bytes as passed to the original `mmap_private`.
        unsafe {
            sys::madvise_dontneed(self.mem_ptr, self.size)?;
            sys::madvise_dontneed(self.perm_ptr, self.size)?;
        }
        Ok(())
    }
}

impl Drop for HostMem {
    fn drop(&mut self) {
        // SAFETY: `mem_ptr`/`perm_ptr` are mapping bases returned by `mmap_private` in
        // `Golden::new_view` with exactly `self.size` bytes each, not yet unmapped (this is the
        // only place that unmaps them), and `&mut self` in `drop` means nothing else can hold a
        // live slice into them.
        unsafe {
            sys::munmap(self.mem_ptr, self.size);
            sys::munmap(self.perm_ptr, self.size);
        }
    }
}

/// Plain RAM is itself a bus, matching `Mmu`'s `Bus` impl (useful if `HostMem` is later wired
/// into the interpreter's fast path).
impl fs_mmu::Bus for HostMem {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::Mmu;

    #[test]
    fn basic_read_write_roundtrip() {
        let mut mmu = Mmu::new(0x8000_0000, 0x1000);
        mmu.map(0x8000_0000, &[1, 2, 3, 4], PERM_READ | PERM_WRITE)
            .unwrap();
        let golden = Golden::from_mmu(&mmu).unwrap();
        let mut hm = golden.new_view().unwrap();
        assert_eq!(hm.read_u32(0x8000_0000).unwrap(), u32::from_le_bytes([1, 2, 3, 4]));
        hm.write_u32(0x8000_0000, 0xdead_beef).unwrap();
        assert_eq!(hm.read_u32(0x8000_0000).unwrap(), 0xdead_beef);
        // Golden itself is untouched by the view's writes.
        let hm2 = golden.new_view().unwrap();
        assert_eq!(hm2.read_u32(0x8000_0000).unwrap(), u32::from_le_bytes([1, 2, 3, 4]));
    }

    #[test]
    fn reset_restores_golden() {
        let mut mmu = Mmu::new(0x8000_0000, 0x2000);
        mmu.map(0x8000_0000, &[0xAA; 16], PERM_READ | PERM_WRITE)
            .unwrap();
        let golden = Golden::from_mmu(&mmu).unwrap();
        let mut hm = golden.new_view().unwrap();
        hm.write_u32(0x8000_0000, 0x1111_1111).unwrap();
        assert_ne!(hm.read_u32(0x8000_0000).unwrap(), 0xAAAA_AAAA);
        hm.reset().unwrap();
        assert_eq!(hm.read_u32(0x8000_0000).unwrap(), 0xAAAA_AAAA);
    }

    #[test]
    fn unmapped_and_unaligned_and_permission_faults() {
        let mut mmu = Mmu::new(0x8000_0000, 0x1000);
        mmu.protect(0x8000_0000, 4, PERM_RAW | PERM_WRITE).unwrap();
        let golden = Golden::from_mmu(&mmu).unwrap();
        let hm = golden.new_view().unwrap();
        assert_eq!(hm.read_u8(0x1234).unwrap_err().kind, FaultKind::Unmapped);
        assert_eq!(hm.read_u32(0x8000_0001).unwrap_err().kind, FaultKind::Unaligned);
        assert_eq!(hm.read_u8(0x8000_0000).unwrap_err().kind, FaultKind::Permission);
    }
}
