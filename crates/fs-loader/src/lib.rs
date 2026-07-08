//! ELF32 loader for static RISC-V executables.
//!
//! M0 scope: parse `ET_EXEC`/`ET_DYN` RV32 ELFs, collect `PT_LOAD` segments with permissions
//! from the program-header flags, and (best-effort) locate the HTIF `tohost` symbol so
//! riscv-tests-style images can signal exit. No dynamic linking, no relocations.

use fs_mmu::{Fault, Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};

const ELFCLASS32: u8 = 1;
const ELFDATA2LSB: u8 = 1;
const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const EM_RISCV: u16 = 243;
const PT_LOAD: u32 = 1;
const SHT_SYMTAB: u32 = 2;

#[derive(Debug, Clone)]
pub struct Segment {
    pub vaddr: u32,
    /// File-backed bytes (length p_filesz).
    pub data: Vec<u8>,
    /// In-memory size (p_memsz); bytes beyond `data.len()` are zero-filled .bss.
    pub memsz: u32,
    /// Soft-MMU permission byte derived from p_flags.
    pub perm: u8,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub entry: u32,
    pub tohost: Option<u32>,
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    Truncated,
    BadMagic,
    Not32BitLe,
    NotRiscv,
    NotExecutable,
    NoLoadSegments,
    Map(Fault),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            LoadError::Truncated => "ELF is truncated",
            LoadError::BadMagic => "not an ELF (bad magic)",
            LoadError::Not32BitLe => "not a 32-bit little-endian ELF",
            LoadError::NotRiscv => "not a RISC-V ELF",
            LoadError::NotExecutable => "not an executable (ET_EXEC/ET_DYN)",
            LoadError::NoLoadSegments => "no PT_LOAD segments",
            LoadError::Map(fault) => return write!(f, "segment does not fit guest memory: {fault}"),
        };
        f.write_str(s)
    }
}

/// Build a minimal, self-contained ELF32 RISC-V executable with a single RWX `PT_LOAD` segment
/// at `load_vaddr` (entry = `load_vaddr`). Includes a tiny section header table (NULL + .shstrtab)
/// so that stricter loaders (e.g. Spike, which asserts `e_shstrndx < e_shnum`) accept it.
pub fn build_flat_elf(load_vaddr: u32, code: &[u8]) -> Vec<u8> {
    const EHSIZE: u32 = 52;
    const PHENTSIZE: u16 = 32;
    const SHENTSIZE: u16 = 40;
    const CODE_OFFSET: u32 = 0x1000;

    fn u16(b: &mut Vec<u8>, v: u16) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }

    let shstrtab: &[u8] = b"\0.shstrtab\0";
    let shstrtab_off = CODE_OFFSET + code.len() as u32;
    let shoff = (shstrtab_off + shstrtab.len() as u32 + 3) & !3;

    let mut b = Vec::new();
    // e_ident
    b.extend_from_slice(&[0x7f, b'E', b'L', b'F', 1, 1, 1, 0]);
    b.extend_from_slice(&[0u8; 8]);
    u16(&mut b, 2); // ET_EXEC
    u16(&mut b, 243); // EM_RISCV
    u32(&mut b, 1); // e_version
    u32(&mut b, load_vaddr); // e_entry
    u32(&mut b, EHSIZE); // e_phoff
    u32(&mut b, shoff); // e_shoff
    u32(&mut b, 0); // e_flags
    u16(&mut b, EHSIZE as u16); // e_ehsize
    u16(&mut b, PHENTSIZE); // e_phentsize
    u16(&mut b, 1); // e_phnum
    u16(&mut b, SHENTSIZE); // e_shentsize
    u16(&mut b, 2); // e_shnum (NULL + .shstrtab)
    u16(&mut b, 1); // e_shstrndx

    // Program header: one RWX PT_LOAD.
    u32(&mut b, 1); // PT_LOAD
    u32(&mut b, CODE_OFFSET); // p_offset
    u32(&mut b, load_vaddr); // p_vaddr
    u32(&mut b, load_vaddr); // p_paddr
    u32(&mut b, code.len() as u32); // p_filesz
    u32(&mut b, code.len() as u32); // p_memsz
    u32(&mut b, 0x7); // p_flags = R|W|X
    u32(&mut b, 0x1000); // p_align

    b.resize(CODE_OFFSET as usize, 0);
    b.extend_from_slice(code);
    b.extend_from_slice(shstrtab);
    b.resize(shoff as usize, 0);

    // Section [0]: NULL.
    b.extend_from_slice(&[0u8; 40]);
    // Section [1]: .shstrtab (STRTAB).
    u32(&mut b, 1); // sh_name -> ".shstrtab"
    u32(&mut b, 3); // SHT_STRTAB
    u32(&mut b, 0); // sh_flags
    u32(&mut b, 0); // sh_addr
    u32(&mut b, shstrtab_off); // sh_offset
    u32(&mut b, shstrtab.len() as u32); // sh_size
    u32(&mut b, 0); // sh_link
    u32(&mut b, 0); // sh_info
    u32(&mut b, 1); // sh_addralign
    u32(&mut b, 0); // sh_entsize
    b
}

/// Build a differential-testing ELF: a single RWX `PT_LOAD` at `entry` holding `segment`, plus a
/// symbol table exposing HTIF `tohost`/`fromhost` at `tohost_vaddr`/`tohost_vaddr+8` so Spike
/// terminates cleanly on a store to `tohost` (and we get a full `--log-commits` trace). The caller
/// is responsible for `segment` actually reserving 16 zero bytes at `tohost_vaddr`.
pub fn build_diff_elf(entry: u32, segment: &[u8], tohost_vaddr: u32) -> Vec<u8> {
    const EHSIZE: u32 = 52;
    const SHENTSIZE: u16 = 40;
    const CODE_OFFSET: u32 = 0x1000;

    fn u16(b: &mut Vec<u8>, v: u16) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    let align4 = |x: u32| (x + 3) & !3;

    // String/symbol tables laid out after the loaded segment.
    let strtab: &[u8] = b"\0tohost\0fromhost\0"; // tohost@1, fromhost@8
    let shstrtab: &[u8] = b"\0.symtab\0.strtab\0.shstrtab\0"; // names @1, @9, @17

    let seg_end = CODE_OFFSET + segment.len() as u32;
    let symtab_off = align4(seg_end);
    let symtab_size = 3 * 16u32; // null + tohost + fromhost
    let strtab_off = symtab_off + symtab_size;
    let shstrtab_off = strtab_off + strtab.len() as u32;
    let shoff = align4(shstrtab_off + shstrtab.len() as u32);

    let mut b = Vec::new();
    // ELF header.
    b.extend_from_slice(&[0x7f, b'E', b'L', b'F', 1, 1, 1, 0]);
    b.extend_from_slice(&[0u8; 8]);
    u16(&mut b, 2); // ET_EXEC
    u16(&mut b, 243); // EM_RISCV
    u32(&mut b, 1); // version
    u32(&mut b, entry); // e_entry
    u32(&mut b, EHSIZE); // e_phoff
    u32(&mut b, shoff); // e_shoff
    u32(&mut b, 0); // e_flags
    u16(&mut b, EHSIZE as u16);
    u16(&mut b, 32); // e_phentsize
    u16(&mut b, 1); // e_phnum
    u16(&mut b, SHENTSIZE);
    u16(&mut b, 4); // e_shnum (NULL, .symtab, .strtab, .shstrtab)
    u16(&mut b, 3); // e_shstrndx

    // Program header: RWX PT_LOAD.
    u32(&mut b, 1);
    u32(&mut b, CODE_OFFSET);
    u32(&mut b, entry);
    u32(&mut b, entry);
    u32(&mut b, segment.len() as u32);
    u32(&mut b, segment.len() as u32);
    u32(&mut b, 0x7);
    u32(&mut b, 0x1000);

    b.resize(CODE_OFFSET as usize, 0);
    b.extend_from_slice(segment);

    // .symtab
    b.resize(symtab_off as usize, 0);
    b.extend_from_slice(&[0u8; 16]); // null symbol
    let sym = |b: &mut Vec<u8>, name: u32, value: u32| {
        u32(b, name);
        u32(b, value);
        u32(b, 8); // st_size
        b.push(0x11); // st_info = GLOBAL|OBJECT
        b.push(0); // st_other
        u16(b, 0xfff1); // st_shndx = SHN_ABS
    };
    sym(&mut b, 1, tohost_vaddr); // tohost
    sym(&mut b, 8, tohost_vaddr + 8); // fromhost

    // .strtab, .shstrtab
    b.extend_from_slice(strtab);
    b.extend_from_slice(shstrtab);

    // Section headers.
    b.resize(shoff as usize, 0);
    let shdr = |b: &mut Vec<u8>, name: u32, typ: u32, off: u32, size: u32, link: u32, info: u32, align: u32, entsize: u32| {
        u32(b, name);
        u32(b, typ);
        u32(b, 0); // flags
        u32(b, 0); // addr
        u32(b, off);
        u32(b, size);
        u32(b, link);
        u32(b, info);
        u32(b, align);
        u32(b, entsize);
    };
    b.extend_from_slice(&[0u8; 40]); // [0] NULL
    shdr(&mut b, 1, 2, symtab_off, symtab_size, 2, 1, 4, 16); // [1] .symtab (link -> strtab idx 2)
    shdr(&mut b, 9, 3, strtab_off, strtab.len() as u32, 0, 0, 1, 0); // [2] .strtab
    shdr(&mut b, 17, 3, shstrtab_off, shstrtab.len() as u32, 0, 0, 1, 0); // [3] .shstrtab
    b
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn u8(&self, off: usize) -> Result<u8, LoadError> {
        self.0.get(off).copied().ok_or(LoadError::Truncated)
    }
    fn u16(&self, off: usize) -> Result<u16, LoadError> {
        let b = self.0.get(off..off + 2).ok_or(LoadError::Truncated)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&self, off: usize) -> Result<u32, LoadError> {
        let b = self.0.get(off..off + 4).ok_or(LoadError::Truncated)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn slice(&self, off: usize, len: usize) -> Result<&'a [u8], LoadError> {
        self.0.get(off..off + len).ok_or(LoadError::Truncated)
    }
    fn cstr(&self, off: usize) -> &'a [u8] {
        let bytes = &self.0[off.min(self.0.len())..];
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        &bytes[..end]
    }
}

/// Parse an ELF32 RISC-V image into loadable segments + metadata.
pub fn parse(bytes: &[u8]) -> Result<Program, LoadError> {
    let r = Reader(bytes);
    if r.slice(0, 4)? != [0x7f, b'E', b'L', b'F'] {
        return Err(LoadError::BadMagic);
    }
    if r.u8(4)? != ELFCLASS32 || r.u8(5)? != ELFDATA2LSB {
        return Err(LoadError::Not32BitLe);
    }
    if r.u16(18)? != EM_RISCV {
        return Err(LoadError::NotRiscv);
    }
    let e_type = r.u16(16)?;
    if e_type != ET_EXEC && e_type != ET_DYN {
        return Err(LoadError::NotExecutable);
    }

    let entry = r.u32(24)?;
    let phoff = r.u32(28)? as usize;
    let phentsize = r.u16(42)? as usize;
    let phnum = r.u16(44)? as usize;

    let mut segments = Vec::new();
    for i in 0..phnum {
        let ph = phoff + i * phentsize;
        if r.u32(ph)? != PT_LOAD {
            continue;
        }
        let p_offset = r.u32(ph + 4)? as usize;
        let p_vaddr = r.u32(ph + 8)?;
        let p_filesz = r.u32(ph + 16)? as usize;
        let p_memsz = r.u32(ph + 20)?;
        let p_flags = r.u32(ph + 24)?;

        let mut perm = 0u8;
        if p_flags & 0x4 != 0 {
            perm |= PERM_READ;
        }
        if p_flags & 0x2 != 0 {
            perm |= PERM_WRITE;
        }
        if p_flags & 0x1 != 0 {
            perm |= PERM_EXEC;
        }

        let data = r.slice(p_offset, p_filesz)?.to_vec();
        segments.push(Segment { vaddr: p_vaddr, data, memsz: p_memsz, perm });
    }

    if segments.is_empty() {
        return Err(LoadError::NoLoadSegments);
    }

    Ok(Program {
        entry,
        tohost: find_tohost(&r, bytes),
        segments,
    })
}

/// Best-effort scan of the section headers / symtab for a `tohost` symbol (riscv-tests HTIF).
fn find_tohost(r: &Reader, bytes: &[u8]) -> Option<u32> {
    let shoff = r.u32(32).ok()? as usize;
    let shentsize = r.u16(46).ok()? as usize;
    let shnum = r.u16(48).ok()? as usize;
    if shoff == 0 || shnum == 0 {
        return None;
    }
    for i in 0..shnum {
        let sh = shoff + i * shentsize;
        if r.u32(sh + 4).ok()? != SHT_SYMTAB {
            continue;
        }
        let sym_off = r.u32(sh + 16).ok()? as usize;
        let sym_size = r.u32(sh + 20).ok()? as usize;
        let strtab_idx = r.u32(sh + 24).ok()? as usize;
        let entsize = r.u32(sh + 36).ok()?.max(16) as usize;

        let str_sh = shoff + strtab_idx * shentsize;
        let str_off = r.u32(str_sh + 16).ok()? as usize;

        let count = sym_size / entsize;
        for s in 0..count {
            let sym = sym_off + s * entsize;
            let st_name = r.u32(sym).ok()? as usize;
            let st_value = r.u32(sym + 4).ok()?;
            if str_off + st_name < bytes.len() && r.cstr(str_off + st_name) == b"tohost" {
                return Some(st_value);
            }
        }
    }
    None
}

impl Program {
    /// The `[lo, hi)` guest address span covered by all loadable segments.
    pub fn image_bounds(&self) -> (u32, u32) {
        let lo = self.segments.iter().map(|s| s.vaddr).min().unwrap_or(0);
        let hi = self
            .segments
            .iter()
            .map(|s| s.vaddr.wrapping_add(s.memsz))
            .max()
            .unwrap_or(0);
        (lo, hi)
    }

    /// Map every segment into `mmu` (contents + permissions; .bss beyond filesz is zero-filled).
    pub fn load_into(&self, mmu: &mut Mmu) -> Result<(), LoadError> {
        for seg in &self.segments {
            mmu.map(seg.vaddr, &seg.data, seg.perm).map_err(LoadError::Map)?;
            let filled = seg.data.len() as u32;
            if seg.memsz > filled {
                mmu.protect(
                    seg.vaddr.wrapping_add(filled),
                    seg.memsz - filled,
                    seg.perm | PERM_READ | PERM_WRITE,
                )
                .map_err(LoadError::Map)?;
            }
        }
        Ok(())
    }
}
