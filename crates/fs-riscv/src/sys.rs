//! Privileged architecture state: M/S/U modes, a CSR file, and trap-cause definitions.
//!
//! This is the M2 full-system layer. The decision doc puts privilege in a separate `fs-arch`
//! crate, but the executor must stay singular (one validated decoder/ALU — the digest's key seam),
//! so the privileged state lives alongside the core and the interpreter grows privilege-aware arms.
//! sv32 translation is added on top of this; OpenSBI itself runs with paging off (satp=0).

/// Privilege mode. Numeric values match the architectural encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priv {
    U = 0,
    S = 1,
    M = 3,
}

impl Priv {
    pub fn level(self) -> u32 {
        self as u32
    }
    pub fn from_bits(b: u32) -> Priv {
        match b & 0x3 {
            0 => Priv::U,
            1 => Priv::S,
            _ => Priv::M, // 3 = M; 2 is reserved, treat as M
        }
    }
}

// ---- CSR addresses ----
pub const MSTATUS: u16 = 0x300;
/// RV32 high half of mstatus (SBE/MBE endianness bits) — 0 for our little-endian machine.
pub const MSTATUSH: u16 = 0x310;
pub const MEDELEGH: u16 = 0x312;
pub const MISA: u16 = 0x301;
pub const MEDELEG: u16 = 0x302;
pub const MIDELEG: u16 = 0x303;
pub const MIE: u16 = 0x304;
pub const MTVEC: u16 = 0x305;
pub const MSCRATCH: u16 = 0x340;
pub const MEPC: u16 = 0x341;
pub const MCAUSE: u16 = 0x342;
pub const MTVAL: u16 = 0x343;
pub const MIP: u16 = 0x344;
pub const MHARTID: u16 = 0xf14;
pub const SSTATUS: u16 = 0x100;
pub const SIE: u16 = 0x104;
pub const STVEC: u16 = 0x105;
pub const SSCRATCH: u16 = 0x140;
pub const SEPC: u16 = 0x141;
pub const SCAUSE: u16 = 0x142;
pub const STVAL: u16 = 0x143;
pub const SIP: u16 = 0x144;
pub const SATP: u16 = 0x180;
/// Sstc supervisor timer compare (low/high halves on RV32).
pub const STIMECMP: u16 = 0x14d;
pub const STIMECMPH: u16 = 0x15d;
// Read-only M identification CSRs (all read as 0 here).
pub const MVENDORID: u16 = 0xf11;
pub const MARCHID: u16 = 0xf12;
pub const MIMPID: u16 = 0xf13;
pub const MCONFIGPTR: u16 = 0xf15;
// Counter-enable / env / inhibit (stubbed as plain WARL words).
pub const MCOUNTEREN: u16 = 0x306;
pub const SCOUNTEREN: u16 = 0x106;
pub const MCOUNTINHIBIT: u16 = 0x320;
pub const MENVCFG: u16 = 0x30a;
pub const MENVCFGH: u16 = 0x31a;
pub const SENVCFG: u16 = 0x10a;
// User-readable counters (handled in the interpreter — they need retired-instruction count).
pub const CYCLE: u16 = 0xc00;
pub const TIME: u16 = 0xc01;
pub const INSTRET: u16 = 0xc02;
pub const CYCLEH: u16 = 0xc80;
pub const TIMEH: u16 = 0xc81;
pub const INSTRETH: u16 = 0xc82;

// ---- mstatus fields ----
pub const MSTATUS_SIE: u32 = 1 << 1;
pub const MSTATUS_MIE: u32 = 1 << 3;
pub const MSTATUS_SPIE: u32 = 1 << 5;
pub const MSTATUS_MPIE: u32 = 1 << 7;
pub const MSTATUS_SPP: u32 = 1 << 8;
pub const MSTATUS_MPP: u32 = 3 << 11;
pub const MSTATUS_MPRV: u32 = 1 << 17;
pub const MSTATUS_SUM: u32 = 1 << 18;
pub const MSTATUS_MXR: u32 = 1 << 19;

const MSTATUS_WMASK: u32 = MSTATUS_SIE
    | MSTATUS_MIE
    | MSTATUS_SPIE
    | MSTATUS_MPIE
    | MSTATUS_SPP
    | MSTATUS_MPP
    | MSTATUS_MPRV
    | MSTATUS_SUM
    | MSTATUS_MXR;
const SSTATUS_MASK: u32 = MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP | MSTATUS_SUM | MSTATUS_MXR;
/// Supervisor-visible interrupt bits (SSIP/STIP/SEIP at 1/5/9).
const S_INT_MASK: u32 = (1 << 1) | (1 << 5) | (1 << 9);

// ---- trap causes (exceptions) ----
pub const E_INSTR_MISALIGNED: u32 = 0;
pub const E_INSTR_ACCESS: u32 = 1;
pub const E_ILLEGAL: u32 = 2;
pub const E_BREAKPOINT: u32 = 3;
pub const E_LOAD_MISALIGNED: u32 = 4;
pub const E_LOAD_ACCESS: u32 = 5;
pub const E_STORE_MISALIGNED: u32 = 6;
pub const E_STORE_ACCESS: u32 = 7;
pub const E_ECALL_U: u32 = 8;
pub const E_ECALL_S: u32 = 9;
pub const E_ECALL_M: u32 = 11;
pub const E_INSTR_PAGE_FAULT: u32 = 12;
pub const E_LOAD_PAGE_FAULT: u32 = 13;
pub const E_STORE_PAGE_FAULT: u32 = 15;

/// Marker: a CSR access that must raise an illegal-instruction trap (privilege or unknown CSR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsrIllegal;

/// The privileged CSR file (RV32 subset needed for OpenSBI + a booting kernel).
#[derive(Debug, Clone)]
pub struct Csr {
    pub mstatus: u32,
    pub mtvec: u32,
    pub mepc: u32,
    pub mcause: u32,
    pub mtval: u32,
    pub mscratch: u32,
    pub mie: u32,
    pub mip: u32,
    pub medeleg: u32,
    pub mideleg: u32,
    pub misa: u32,
    pub mhartid: u32,
    pub stvec: u32,
    pub sepc: u32,
    pub scause: u32,
    pub stval: u32,
    pub sscratch: u32,
    pub satp: u32,
    /// Sstc supervisor timer compare (64-bit); MTIP compare for the M-timer.
    pub stimecmp: u64,
    pub mtimecmp: u64,
    // OpenSBI/kernel stubs (WARL storage; PMP is permissive by construction — no enforcement yet).
    pub pmpcfg: [u32; 4],
    pub pmpaddr: [u32; 16],
    pub mcounteren: u32,
    pub scounteren: u32,
    pub mcountinhibit: u32,
    pub menvcfg: u32,
    pub menvcfgh: u32,
    pub senvcfg: u32,
}

impl Default for Csr {
    fn default() -> Self {
        // misa: RV32 (MXL=1) with extensions I, M, A, C, S, U.
        let ext = (1 << 8) | (1 << 12) | (1 << 0) | (1 << 2) | (1 << 18) | (1 << 20);
        Csr {
            mstatus: 0,
            mtvec: 0,
            mepc: 0,
            mcause: 0,
            mtval: 0,
            mscratch: 0,
            mie: 0,
            mip: 0,
            medeleg: 0,
            mideleg: 0,
            misa: (1 << 30) | ext,
            mhartid: 0,
            stvec: 0,
            sepc: 0,
            scause: 0,
            stval: 0,
            sscratch: 0,
            satp: 0,
            stimecmp: u64::MAX, // disabled until programmed
            mtimecmp: u64::MAX,
            pmpcfg: [0; 4],
            pmpaddr: [0; 16],
            mcounteren: 0,
            scounteren: 0,
            mcountinhibit: 0,
            menvcfg: 0,
            menvcfgh: 0,
            senvcfg: 0,
        }
    }
}

impl Csr {
    /// Read a CSR. `Err` means an illegal-instruction trap should be raised (privilege).
    pub fn read(&self, addr: u16, priv_: Priv) -> Result<u32, CsrIllegal> {
        if priv_.level() < ((addr >> 8) & 0x3) as u32 {
            return Err(CsrIllegal);
        }
        Ok(match addr {
            MSTATUS => self.mstatus,
            MISA => self.misa,
            MEDELEG => self.medeleg,
            MIDELEG => self.mideleg,
            MIE => self.mie,
            MTVEC => self.mtvec,
            MSCRATCH => self.mscratch,
            MEPC => self.mepc,
            MCAUSE => self.mcause,
            MTVAL => self.mtval,
            MIP => self.mip,
            MHARTID => self.mhartid,
            SSTATUS => self.mstatus & SSTATUS_MASK,
            SIE => self.mie & S_INT_MASK,
            STVEC => self.stvec,
            SSCRATCH => self.sscratch,
            SEPC => self.sepc,
            SCAUSE => self.scause,
            STVAL => self.stval,
            SIP => self.mip & S_INT_MASK,
            SATP => self.satp,
            STIMECMP => self.stimecmp as u32,
            STIMECMPH => (self.stimecmp >> 32) as u32,
            MVENDORID | MARCHID | MIMPID | MCONFIGPTR | MSTATUSH | MEDELEGH => 0,
            MCOUNTEREN => self.mcounteren,
            SCOUNTEREN => self.scounteren,
            MCOUNTINHIBIT => self.mcountinhibit,
            MENVCFG => self.menvcfg,
            MENVCFGH => self.menvcfgh,
            SENVCFG => self.senvcfg,
            a if (0x3a0..=0x3a3).contains(&a) => self.pmpcfg[(a - 0x3a0) as usize],
            a if (0x3b0..=0x3bf).contains(&a) => self.pmpaddr[(a - 0x3b0) as usize],
            _ => return Err(CsrIllegal), // unknown CSR
        })
    }

    /// Write a CSR (already legalized/masked). `Err(())` -> illegal-instruction trap.
    pub fn write(&mut self, addr: u16, val: u32, priv_: Priv) -> Result<(), CsrIllegal> {
        if priv_.level() < ((addr >> 8) & 0x3) as u32 {
            return Err(CsrIllegal);
        }
        if (addr >> 10) & 0x3 == 0x3 {
            return Err(CsrIllegal); // read-only CSR
        }
        match addr {
            MSTATUS => self.mstatus = val & MSTATUS_WMASK,
            MISA => {} // WARL; keep fixed
            MEDELEG => self.medeleg = val,
            MIDELEG => self.mideleg = val,
            MIE => self.mie = val,
            MTVEC => self.mtvec = val & !0x2, // modes 0 (direct) / 1 (vectored)
            MSCRATCH => self.mscratch = val,
            MEPC => self.mepc = val & !0x1,
            MCAUSE => self.mcause = val,
            MTVAL => self.mtval = val,
            MIP => self.mip = (self.mip & !S_INT_MASK) | (val & S_INT_MASK),
            SSTATUS => self.mstatus = (self.mstatus & !SSTATUS_MASK) | (val & SSTATUS_MASK),
            SIE => self.mie = (self.mie & !S_INT_MASK) | (val & S_INT_MASK),
            STVEC => self.stvec = val & !0x2,
            SSCRATCH => self.sscratch = val,
            SEPC => self.sepc = val & !0x1,
            SCAUSE => self.scause = val,
            STVAL => self.stval = val,
            SIP => self.mip = (self.mip & !S_INT_MASK) | (val & S_INT_MASK),
            SATP => self.satp = val,
            STIMECMP => self.stimecmp = (self.stimecmp & 0xffff_ffff_0000_0000) | val as u64,
            STIMECMPH => self.stimecmp = (self.stimecmp & 0xffff_ffff) | ((val as u64) << 32),
            MSTATUSH | MEDELEGH => {} // little-endian: SBE/MBE fixed 0; medelegh unused on our machine
            MCOUNTEREN => self.mcounteren = val,
            SCOUNTEREN => self.scounteren = val,
            MCOUNTINHIBIT => self.mcountinhibit = val,
            MENVCFG => self.menvcfg = val,
            MENVCFGH => self.menvcfgh = val,
            SENVCFG => self.senvcfg = val,
            a if (0x3a0..=0x3a3).contains(&a) => self.pmpcfg[(a - 0x3a0) as usize] = val,
            a if (0x3b0..=0x3bf).contains(&a) => self.pmpaddr[(a - 0x3b0) as usize] = val,
            _ => return Err(CsrIllegal),
        }
        Ok(())
    }
}
