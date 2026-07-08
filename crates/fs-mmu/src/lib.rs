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
}

/// Reset granularity for snapshot fuzzing: one cache line (decision #11).
pub const DIRTY_BLOCK: usize = 64;

/// A flat guest memory with a parallel permission plane.
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

    /// Checked read: every byte must carry PERM_READ.
    pub fn read(&self, addr: u32, buf: &mut [u8]) -> Result<(), Fault> {
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
    pub fn write(&mut self, addr: u32, data: &[u8]) -> Result<(), Fault> {
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

    /// Instruction half-word fetch: 2-byte aligned (IALIGN=16 with the C extension), every byte
    /// must carry PERM_EXEC (READ not required).
    pub fn fetch_u16(&self, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Exec)?;
        let mut b = [0u8; 2];
        for (i, out) in b.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, 2, Access::Exec, FaultKind::Unmapped))?;
            if self.perms[off] & PERM_EXEC == 0 {
                return Err(Self::fault(a, 2, Access::Exec, FaultKind::Permission));
            }
            *out = self.mem[off];
        }
        Ok(u16::from_le_bytes(b))
    }

    /// Instruction fetch: 4-byte aligned, every byte must carry PERM_EXEC (READ not required).
    pub fn fetch_u32(&self, addr: u32) -> Result<u32, Fault> {
        Self::check_align(addr, 4, Access::Exec)?;
        let mut b = [0u8; 4];
        for (i, out) in b.iter_mut().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let off = self
                .offset(a)
                .ok_or_else(|| Self::fault(a, 4, Access::Exec, FaultKind::Unmapped))?;
            if self.perms[off] & PERM_EXEC == 0 {
                return Err(Self::fault(a, 4, Access::Exec, FaultKind::Permission));
            }
            *out = self.mem[off];
        }
        Ok(u32::from_le_bytes(b))
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
        assert_eq!(m.read_u32(0x8000_0001).unwrap_err().kind, FaultKind::Unaligned);
    }
}
