//! RV32IM decoder (decode-to-IL) + scalar interpreter — the M0 golden model.
//!
//! Per decision #8, `decode()` lowers each 32-bit instruction into a typed [`Inst`] IL that BOTH
//! this scalar interpreter and (later) the hybrid AVX-512 executor consume, so there is exactly
//! one validated decoder. M0 covers rv32im only; the A and C extensions and the full privileged
//! surface (CSRs, traps, sv32) arrive in M1/M2.

#![forbid(unsafe_code)]

use fs_mmu::{Access, Bus, Fault, FaultKind, Golden};

pub mod asm;
pub mod sys;

use sys::{Csr, Priv};

// Register ABI aliases used in tests / codegen.
pub const X0: u8 = 0;
pub const SP: u8 = 2;
pub const T0: u8 = 5;
pub const T1: u8 = 6;
pub const A0: u8 = 10;
pub const A7: u8 = 17;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Sub,
    Sll,
    Slt,
    Sltu,
    Xor,
    Srl,
    Sra,
    Or,
    And,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulOp {
    Mul,
    Mulh,
    Mulhsu,
    Mulhu,
    Div,
    Divu,
    Rem,
    Remu,
}

/// A-extension read-modify-write operations (AMO*.W).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmoOp {
    Swap,
    Add,
    Xor,
    And,
    Or,
    Min,
    Max,
    Minu,
    Maxu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchOp {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    Lb,
    Lh,
    Lw,
    Lbu,
    Lhu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Sb,
    Sh,
    Sw,
}

/// CSR access operation. `*I` variants take a 5-bit zero-extended immediate instead of rs1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrOp {
    Rw,
    Rs,
    Rc,
    Rwi,
    Rsi,
    Rci,
}

/// Decoded instruction IL. `rd`/`rs1`/`rs2` are 5-bit register indices; `imm` is sign-extended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    Lui { rd: u8, imm: u32 },
    Auipc { rd: u8, imm: u32 },
    Jal { rd: u8, imm: i32 },
    Jalr { rd: u8, rs1: u8, imm: i32 },
    Branch { op: BranchOp, rs1: u8, rs2: u8, imm: i32 },
    Load { op: LoadOp, rd: u8, rs1: u8, imm: i32 },
    Store { op: StoreOp, rs1: u8, rs2: u8, imm: i32 },
    /// Register-immediate ALU. Shifts carry the shamt in `imm`.
    OpImm { op: AluOp, rd: u8, rs1: u8, imm: i32 },
    /// Register-register ALU.
    Op { op: AluOp, rd: u8, rs1: u8, rs2: u8 },
    /// M-extension register-register.
    Mul { op: MulOp, rd: u8, rs1: u8, rs2: u8 },
    /// A-extension load-reserved.
    LrW { rd: u8, rs1: u8, aq: bool, rl: bool },
    /// A-extension store-conditional.
    ScW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    /// A-extension atomic memory operation.
    AmoW { op: AmoOp, rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    Fence,
    Ecall,
    Ebreak,
    /// CSR read/modify/write. `src` is rs1 (register) or a 5-bit immediate per `op`.
    Csr { op: CsrOp, rd: u8, src: u8, csr: u16 },
    /// Machine/Supervisor trap return.
    Mret,
    Sret,
    /// Wait-for-interrupt (treated as a no-op hint in a deterministic core).
    Wfi,
    /// Supervisor fence (TLB flush).
    SfenceVma,
    Illegal(u32),
}

#[inline]
fn sext(v: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((v << shift) as i32) >> shift
}

/// Decode a 32-bit little-endian RV32IM instruction word into the IL.
pub fn decode(raw: u32) -> Inst {
    let opcode = raw & 0x7f;
    let rd = ((raw >> 7) & 0x1f) as u8;
    let funct3 = (raw >> 12) & 0x7;
    let rs1 = ((raw >> 15) & 0x1f) as u8;
    let rs2 = ((raw >> 20) & 0x1f) as u8;
    let funct7 = (raw >> 25) & 0x7f;

    let i_imm = (raw as i32) >> 20;
    let s_imm = (((raw & 0xfe00_0000) as i32) >> 20) | (((raw >> 7) & 0x1f) as i32);
    let b_imm = sext(
        ((raw >> 31) & 1) << 12
            | ((raw >> 7) & 1) << 11
            | ((raw >> 25) & 0x3f) << 5
            | ((raw >> 8) & 0xf) << 1,
        13,
    );
    let u_imm = raw & 0xffff_f000;
    let j_imm = sext(
        ((raw >> 31) & 1) << 20
            | ((raw >> 21) & 0x3ff) << 1
            | ((raw >> 20) & 1) << 11
            | ((raw >> 12) & 0xff) << 12,
        21,
    );

    match opcode {
        0x37 => Inst::Lui { rd, imm: u_imm },
        0x17 => Inst::Auipc { rd, imm: u_imm },
        0x6f => Inst::Jal { rd, imm: j_imm },
        0x67 if funct3 == 0 => Inst::Jalr { rd, rs1, imm: i_imm },
        0x63 => {
            let op = match funct3 {
                0 => BranchOp::Eq,
                1 => BranchOp::Ne,
                4 => BranchOp::Lt,
                5 => BranchOp::Ge,
                6 => BranchOp::Ltu,
                7 => BranchOp::Geu,
                _ => return Inst::Illegal(raw),
            };
            Inst::Branch { op, rs1, rs2, imm: b_imm }
        }
        0x03 => {
            let op = match funct3 {
                0 => LoadOp::Lb,
                1 => LoadOp::Lh,
                2 => LoadOp::Lw,
                4 => LoadOp::Lbu,
                5 => LoadOp::Lhu,
                _ => return Inst::Illegal(raw),
            };
            Inst::Load { op, rd, rs1, imm: i_imm }
        }
        0x23 => {
            let op = match funct3 {
                0 => StoreOp::Sb,
                1 => StoreOp::Sh,
                2 => StoreOp::Sw,
                _ => return Inst::Illegal(raw),
            };
            Inst::Store { op, rs1, rs2, imm: s_imm }
        }
        0x13 => {
            let shamt = (rs2 as i32) & 0x1f; // shift amount lives in the rs2 field
            match funct3 {
                0 => Inst::OpImm { op: AluOp::Add, rd, rs1, imm: i_imm },
                2 => Inst::OpImm { op: AluOp::Slt, rd, rs1, imm: i_imm },
                3 => Inst::OpImm { op: AluOp::Sltu, rd, rs1, imm: i_imm },
                4 => Inst::OpImm { op: AluOp::Xor, rd, rs1, imm: i_imm },
                6 => Inst::OpImm { op: AluOp::Or, rd, rs1, imm: i_imm },
                7 => Inst::OpImm { op: AluOp::And, rd, rs1, imm: i_imm },
                1 if funct7 == 0x00 => Inst::OpImm { op: AluOp::Sll, rd, rs1, imm: shamt },
                5 if funct7 == 0x00 => Inst::OpImm { op: AluOp::Srl, rd, rs1, imm: shamt },
                5 if funct7 == 0x20 => Inst::OpImm { op: AluOp::Sra, rd, rs1, imm: shamt },
                _ => Inst::Illegal(raw),
            }
        }
        0x33 if funct7 == 0x01 => {
            let op = match funct3 {
                0 => MulOp::Mul,
                1 => MulOp::Mulh,
                2 => MulOp::Mulhsu,
                3 => MulOp::Mulhu,
                4 => MulOp::Div,
                5 => MulOp::Divu,
                6 => MulOp::Rem,
                7 => MulOp::Remu,
                _ => return Inst::Illegal(raw),
            };
            Inst::Mul { op, rd, rs1, rs2 }
        }
        0x33 => {
            let op = match (funct3, funct7) {
                (0, 0x00) => AluOp::Add,
                (0, 0x20) => AluOp::Sub,
                (1, 0x00) => AluOp::Sll,
                (2, 0x00) => AluOp::Slt,
                (3, 0x00) => AluOp::Sltu,
                (4, 0x00) => AluOp::Xor,
                (5, 0x00) => AluOp::Srl,
                (5, 0x20) => AluOp::Sra,
                (6, 0x00) => AluOp::Or,
                (7, 0x00) => AluOp::And,
                _ => return Inst::Illegal(raw),
            };
            Inst::Op { op, rd, rs1, rs2 }
        }
        0x2f if funct3 == 2 => {
            // A-extension, .W (RV32 has no .D). funct5 in [31:27]; aq/rl in [26:25].
            let funct5 = (raw >> 27) & 0x1f;
            let aq = (raw >> 26) & 1 != 0;
            let rl = (raw >> 25) & 1 != 0;
            match funct5 {
                0x02 => Inst::LrW { rd, rs1, aq, rl },
                0x03 => Inst::ScW { rd, rs1, rs2, aq, rl },
                _ => {
                    let op = match funct5 {
                        0x00 => AmoOp::Add,
                        0x01 => AmoOp::Swap,
                        0x04 => AmoOp::Xor,
                        0x08 => AmoOp::Or,
                        0x0c => AmoOp::And,
                        0x10 => AmoOp::Min,
                        0x14 => AmoOp::Max,
                        0x18 => AmoOp::Minu,
                        0x1c => AmoOp::Maxu,
                        _ => return Inst::Illegal(raw),
                    };
                    Inst::AmoW { op, rd, rs1, rs2, aq, rl }
                }
            }
        }
        0x0f => Inst::Fence, // FENCE / FENCE.I are no-ops in a deterministic single-hart core
        0x73 => {
            let csr = ((raw >> 20) & 0xfff) as u16;
            match funct3 {
                0 => match raw {
                    0x0000_0073 => Inst::Ecall,
                    0x0010_0073 => Inst::Ebreak,
                    0x3020_0073 => Inst::Mret,
                    0x1020_0073 => Inst::Sret,
                    0x1050_0073 => Inst::Wfi,
                    _ if funct7 == 0x09 => Inst::SfenceVma, // SFENCE.VMA (rs1/rs2 ignored here)
                    _ => Inst::Illegal(raw),
                },
                1 => Inst::Csr { op: CsrOp::Rw, rd, src: rs1, csr },
                2 => Inst::Csr { op: CsrOp::Rs, rd, src: rs1, csr },
                3 => Inst::Csr { op: CsrOp::Rc, rd, src: rs1, csr },
                5 => Inst::Csr { op: CsrOp::Rwi, rd, src: rs1, csr },
                6 => Inst::Csr { op: CsrOp::Rsi, rd, src: rs1, csr },
                7 => Inst::Csr { op: CsrOp::Rci, rd, src: rs1, csr },
                _ => Inst::Illegal(raw),
            }
        }
        _ => Inst::Illegal(raw),
    }
}

// C.J / C.JAL jump offset — the notoriously scrambled 11-bit immediate.
#[inline]
fn cj_imm(h: u32) -> i32 {
    let v = ((h >> 12) & 1) << 11
        | ((h >> 11) & 1) << 4
        | ((h >> 9) & 0x3) << 8
        | ((h >> 8) & 1) << 10
        | ((h >> 7) & 1) << 6
        | ((h >> 6) & 1) << 7
        | ((h >> 3) & 0x7) << 1
        | ((h >> 2) & 1) << 5;
    sext(v, 12)
}

// C.BEQZ / C.BNEZ branch offset.
#[inline]
fn cb_imm(h: u32) -> i32 {
    let v = ((h >> 12) & 1) << 8
        | ((h >> 10) & 0x3) << 3
        | ((h >> 5) & 0x3) << 6
        | ((h >> 3) & 0x3) << 1
        | ((h >> 2) & 1) << 5;
    sext(v, 9)
}

/// Decode a 16-bit RVC (compressed) instruction, expanding it into the same [`Inst`] IL as its
/// 32-bit equivalent. Covers the RV32 integer subset (no FP, no RV64-only forms). The immediate
/// bit-scatter here is the highest-bug-density part of the decoder and is validated against Spike.
pub fn decode_compressed(half: u16) -> Inst {
    let h = half as u32;
    let op = h & 0x3;
    let funct3 = (h >> 13) & 0x7;
    // 3-bit compressed register fields map to x8..x15.
    let rd_c = (8 + ((h >> 2) & 0x7)) as u8; // field at [4:2]  (rd'/rs2')
    let rs1_c = (8 + ((h >> 7) & 0x7)) as u8; // field at [9:7] (rd'/rs1')

    match (op, funct3) {
        // ---- Quadrant 0 ----
        (0, 0) => {
            // C.ADDI4SPN: nzuimm[5:4]=h[12:11], [9:6]=h[10:7], [2]=h[6], [3]=h[5]
            let imm = ((h >> 11) & 0x3) << 4
                | ((h >> 7) & 0xf) << 6
                | ((h >> 6) & 0x1) << 2
                | ((h >> 5) & 0x1) << 3;
            if imm == 0 {
                return Inst::Illegal(h); // reserved
            }
            Inst::OpImm { op: AluOp::Add, rd: rd_c, rs1: 2, imm: imm as i32 }
        }
        (0, 2) => {
            // C.LW: off[5:3]=h[12:10], [2]=h[6], [6]=h[5]
            let off = ((h >> 10) & 0x7) << 3 | ((h >> 6) & 0x1) << 2 | ((h >> 5) & 0x1) << 6;
            Inst::Load { op: LoadOp::Lw, rd: rd_c, rs1: rs1_c, imm: off as i32 }
        }
        (0, 6) => {
            // C.SW: same offset layout; rs2' at [4:2].
            let off = ((h >> 10) & 0x7) << 3 | ((h >> 6) & 0x1) << 2 | ((h >> 5) & 0x1) << 6;
            Inst::Store { op: StoreOp::Sw, rs1: rs1_c, rs2: rd_c, imm: off as i32 }
        }
        // ---- Quadrant 1 ----
        (1, 0) => {
            // C.ADDI / C.NOP (rd==0, imm==0)
            let rd = ((h >> 7) & 0x1f) as u8;
            let imm = sext(((h >> 12) & 1) << 5 | ((h >> 2) & 0x1f), 6);
            Inst::OpImm { op: AluOp::Add, rd, rs1: rd, imm }
        }
        (1, 1) => Inst::Jal { rd: 1, imm: cj_imm(h) }, // C.JAL (RV32-only)
        (1, 2) => {
            // C.LI: addi rd, x0, imm
            let rd = ((h >> 7) & 0x1f) as u8;
            let imm = sext(((h >> 12) & 1) << 5 | ((h >> 2) & 0x1f), 6);
            Inst::OpImm { op: AluOp::Add, rd, rs1: 0, imm }
        }
        (1, 3) => {
            let rd = ((h >> 7) & 0x1f) as u8;
            if rd == 2 {
                // C.ADDI16SP: nzimm[9]=h[12],[4]=h[6],[6]=h[5],[8:7]=h[4:3],[5]=h[2]
                let imm = sext(
                    ((h >> 12) & 1) << 9
                        | ((h >> 6) & 1) << 4
                        | ((h >> 5) & 1) << 6
                        | ((h >> 3) & 0x3) << 7
                        | ((h >> 2) & 1) << 5,
                    10,
                );
                if imm == 0 {
                    return Inst::Illegal(h);
                }
                Inst::OpImm { op: AluOp::Add, rd: 2, rs1: 2, imm }
            } else if rd != 0 {
                // C.LUI: nzimm[17]=h[12], [16:12]=h[6:2]
                let imm = sext(((h >> 12) & 1) << 17 | ((h >> 2) & 0x1f) << 12, 18);
                if imm == 0 {
                    return Inst::Illegal(h);
                }
                Inst::Lui { rd, imm: imm as u32 }
            } else {
                Inst::Illegal(h)
            }
        }
        (1, 4) => {
            let funct2 = (h >> 10) & 0x3;
            let shamt = (((h >> 12) & 1) << 5 | ((h >> 2) & 0x1f)) as i32;
            match funct2 {
                0 => Inst::OpImm { op: AluOp::Srl, rd: rs1_c, rs1: rs1_c, imm: shamt }, // C.SRLI
                1 => Inst::OpImm { op: AluOp::Sra, rd: rs1_c, rs1: rs1_c, imm: shamt }, // C.SRAI
                2 => {
                    // C.ANDI
                    let imm = sext(((h >> 12) & 1) << 5 | ((h >> 2) & 0x1f), 6);
                    Inst::OpImm { op: AluOp::And, rd: rs1_c, rs1: rs1_c, imm }
                }
                _ => {
                    // register-register: [12] and [6:5] select the op; rs2' at [4:2].
                    let op = match ((h >> 12) & 1, (h >> 5) & 0x3) {
                        (0, 0) => AluOp::Sub,
                        (0, 1) => AluOp::Xor,
                        (0, 2) => AluOp::Or,
                        (0, 3) => AluOp::And,
                        _ => return Inst::Illegal(h), // (1,_) are RV64-only C.SUBW/C.ADDW
                    };
                    Inst::Op { op, rd: rs1_c, rs1: rs1_c, rs2: rd_c }
                }
            }
        }
        (1, 5) => Inst::Jal { rd: 0, imm: cj_imm(h) }, // C.J
        (1, 6) => Inst::Branch { op: BranchOp::Eq, rs1: rs1_c, rs2: 0, imm: cb_imm(h) }, // C.BEQZ
        (1, 7) => Inst::Branch { op: BranchOp::Ne, rs1: rs1_c, rs2: 0, imm: cb_imm(h) }, // C.BNEZ
        // ---- Quadrant 2 ----
        (2, 0) => {
            // C.SLLI
            let rd = ((h >> 7) & 0x1f) as u8;
            let shamt = (((h >> 12) & 1) << 5 | ((h >> 2) & 0x1f)) as i32;
            Inst::OpImm { op: AluOp::Sll, rd, rs1: rd, imm: shamt }
        }
        (2, 2) => {
            // C.LWSP: off[5]=h[12], [4:2]=h[6:4], [7:6]=h[3:2]
            let rd = ((h >> 7) & 0x1f) as u8;
            if rd == 0 {
                return Inst::Illegal(h); // reserved
            }
            let off = ((h >> 12) & 1) << 5 | ((h >> 4) & 0x7) << 2 | ((h >> 2) & 0x3) << 6;
            Inst::Load { op: LoadOp::Lw, rd, rs1: 2, imm: off as i32 }
        }
        (2, 4) => {
            let rd = ((h >> 7) & 0x1f) as u8;
            let rs2 = ((h >> 2) & 0x1f) as u8;
            match ((h >> 12) & 1, rd, rs2) {
                (0, 0, _) => Inst::Illegal(h),                          // reserved
                (0, _, 0) => Inst::Jalr { rd: 0, rs1: rd, imm: 0 },     // C.JR
                (0, _, _) => Inst::Op { op: AluOp::Add, rd, rs1: 0, rs2 }, // C.MV
                (_, 0, 0) => Inst::Ebreak,                              // C.EBREAK
                (_, _, 0) => Inst::Jalr { rd: 1, rs1: rd, imm: 0 },     // C.JALR
                (_, _, _) => Inst::Op { op: AluOp::Add, rd, rs1: rd, rs2 }, // C.ADD
            }
        }
        (2, 6) => {
            // C.SWSP: off[5:2]=h[12:9], [7:6]=h[8:7]
            let rs2 = ((h >> 2) & 0x1f) as u8;
            let off = ((h >> 9) & 0xf) << 2 | ((h >> 7) & 0x3) << 6;
            Inst::Store { op: StoreOp::Sw, rs1: 2, rs2, imm: off as i32 }
        }
        _ => Inst::Illegal(h),
    }
}

#[inline]
fn alu(op: AluOp, a: u32, b: u32) -> u32 {
    match op {
        AluOp::Add => a.wrapping_add(b),
        AluOp::Sub => a.wrapping_sub(b),
        AluOp::Sll => a.wrapping_shl(b & 31),
        AluOp::Slt => ((a as i32) < (b as i32)) as u32,
        AluOp::Sltu => (a < b) as u32,
        AluOp::Xor => a ^ b,
        AluOp::Srl => a.wrapping_shr(b & 31),
        AluOp::Sra => ((a as i32).wrapping_shr(b & 31)) as u32,
        AluOp::Or => a | b,
        AluOp::And => a & b,
    }
}

#[inline]
fn muldiv(op: MulOp, a: u32, b: u32) -> u32 {
    match op {
        MulOp::Mul => a.wrapping_mul(b),
        MulOp::Mulh => (((a as i32 as i64) * (b as i32 as i64)) >> 32) as u32,
        MulOp::Mulhsu => (((a as i32 as i64) * (b as i64)) >> 32) as u32,
        MulOp::Mulhu => (((a as u64) * (b as u64)) >> 32) as u32,
        MulOp::Div => {
            if b == 0 {
                0xffff_ffff
            } else if a == 0x8000_0000 && b == 0xffff_ffff {
                0x8000_0000
            } else {
                ((a as i32).wrapping_div(b as i32)) as u32
            }
        }
        MulOp::Divu => a.checked_div(b).unwrap_or(0xffff_ffff),
        MulOp::Rem => {
            if b == 0 {
                a
            } else if a == 0x8000_0000 && b == 0xffff_ffff {
                0
            } else {
                ((a as i32).wrapping_rem(b as i32)) as u32
            }
        }
        MulOp::Remu => a.checked_rem(b).unwrap_or(a),
    }
}

/// Outcome of a single [`Cpu::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Instruction retired normally; keep stepping.
    Continue,
    /// Environment call (`ecall`) — the runner inspects a7/a0.
    Ecall,
    /// Breakpoint (`ebreak`).
    Ebreak,
    /// HTIF `tohost` exit with the decoded exit code.
    Halt(u32),
}

/// A trap that aborts a step: a physical memory fault, an illegal instruction, or an
/// architectural exception with an explicit cause (e.g. an sv32 page fault).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    Mem(Fault),
    Illegal { pc: u32, raw: u32 },
    Exception { cause: u32, tval: u32 },
}

impl std::fmt::Display for Trap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trap::Mem(fault) => write!(f, "{fault}"),
            Trap::Illegal { pc, raw } => {
                write!(f, "illegal instruction {raw:#010x} @ {pc:#010x}")
            }
            Trap::Exception { cause, tval } => {
                write!(f, "exception cause {cause} tval {tval:#010x}")
            }
        }
    }
}

/// Why [`Cpu::xlate_golden_readonly`] declined: on ANY doubt it hands back this unit marker
/// instead of a `Trap`, since there is nothing to recover from a decline — the only correct
/// response is to fall back to the mutating, per-lane [`Cpu::xlate`] (which will independently
/// compute the real translation or the real fault).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XlateDeclined;

/// Number of direct-mapped software-TLB slots. 256 entries cover a 1 MiB VA working set — ample
/// for a kernel/syscall burst — at ~4 KiB per `Cpu` (cheap to clone on snapshot reset).
const TLB_SIZE: usize = 256;

/// One cached sv32 translation: a 4 KiB page's VA→PA plus its leaf permission bits.
#[derive(Clone, Copy)]
struct TlbEntry {
    /// Virtual page number (`va >> 12`). `u32::MAX` (an impossible 20-bit VPN) marks an empty slot.
    vpn: u32,
    /// Physical page number (`pa >> 12`) for this page.
    ppn: u32,
    r: bool,
    w: bool,
    x: bool,
    u: bool,
    /// PTE dirty bit already set, so a write hit needn't re-walk just to set D.
    d_set: bool,
}

/// Decoded leaf permission bits (R/W/X/U) — the same shape whether they came from a freshly
/// decoded PTE word or a cached [`TlbEntry`], so [`leaf_perm_priv_ok`] can take one bundle instead
/// of four bare bools (also keeps it under clippy's argument-count lint).
#[derive(Clone, Copy)]
struct LeafBits {
    r: bool,
    w: bool,
    x: bool,
    u: bool,
}

/// ONE source of truth for the sv32 leaf permission/privilege decision, shared by `Cpu::xlate`'s
/// TLB-hit path, its full walk's leaf check, and [`Cpu::xlate_golden_readonly`]'s read-only walk
/// (PR4 of the software-COW design, `docs/cow-shared-ram.md`).
#[inline]
fn leaf_perm_priv_ok(access: Access, p: Priv, sum: bool, mxr: bool, leaf: LeafBits) -> bool {
    let perm_ok = match access {
        Access::Exec => leaf.x,
        Access::Read => leaf.r || (mxr && leaf.x),
        Access::Write => leaf.w,
    };
    let priv_ok = match p {
        Priv::U => leaf.u,
        Priv::S => !(leaf.u && (access == Access::Exec || !sum)),
        Priv::M => true,
    };
    perm_ok && priv_ok
}

impl TlbEntry {
    const EMPTY: TlbEntry = TlbEntry {
        vpn: u32::MAX,
        ppn: 0,
        r: false,
        w: false,
        x: false,
        u: false,
        d_set: false,
    };
}

/// Software TLB: a direct-mapped VA→PA translation cache over the sv32 walk. A hit skips the two
/// PTE memory reads (and the A/D writeback) the walk performs — the dominant per-access cost once
/// paging is on. Flushed wholesale on `SFENCE.VMA` and on any write to `satp` (address-space
/// switch) — the only points the guest is architecturally required to fence after editing page
/// tables, so a stale entry can never be architecturally observed. Permission/privilege (U / SUM /
/// MXR) and the dirty bit are re-checked per access against the cached raw PTE bits, so privilege
/// transitions and read→write upgrades need no flush.
#[derive(Clone)]
struct Tlb {
    entries: [TlbEntry; TLB_SIZE],
}

impl Tlb {
    fn new() -> Self {
        Tlb { entries: [TlbEntry::EMPTY; TLB_SIZE] }
    }
    #[inline]
    fn flush(&mut self) {
        self.entries = [TlbEntry::EMPTY; TLB_SIZE];
    }
    #[inline]
    fn slot(vpn: u32) -> usize {
        (vpn as usize) & (TLB_SIZE - 1)
    }
    #[inline]
    fn get(&self, vpn: u32) -> Option<TlbEntry> {
        let e = self.entries[Self::slot(vpn)];
        (e.vpn == vpn).then_some(e)
    }
    #[inline]
    fn insert(&mut self, e: TlbEntry) {
        self.entries[Self::slot(e.vpn)] = e;
    }
}

/// Comparison-coverage (CMPLOG/RedQueen) log capacity — bounds the per-case recording cost so a
/// pathological tight compare loop can't grow the log unboundedly. Recording simply stops once
/// full (no ring eviction): the first `CMPLOG_CAP` comparisons of a case are ample signal for the
/// mutation-time value-matching pass in `fs-prog`, and an early-case magic-value compare (the
/// common case: an argument validated near the top of a syscall handler) is captured either way.
const CMPLOG_CAP: usize = 8192;

/// UBSAN div-by-zero log capacity — same rationale as [`CMPLOG_CAP`]: bounds recording cost so a
/// tight div-by-zero loop can't grow the log unboundedly. The fs-cli caller dedups by PC anyway,
/// so the first `UBSAN_CAP` hits of a case are ample signal.
const UBSAN_CAP: usize = 8192;

/// The scalar RV32IM core state.
#[derive(Clone)]
pub struct Cpu {
    pub regs: [u32; 32],
    pub pc: u32,
    /// HTIF `tohost` word address, if the loaded image exposes one (riscv-tests style).
    pub htif_tohost: Option<u32>,
    pub insns_retired: u64,
    /// LR/SC reservation: the address a valid load-reserved is watching.
    reservation: Option<u32>,
    /// Current privilege mode (bare M0/M1 execution stays in M).
    pub privilege: Priv,
    /// Privileged CSR file.
    pub csr: Csr,
    /// If set, an `ecall` with `a7 == eid` is intercepted by the harness (a fuzzing hypercall)
    /// instead of being delivered to the guest kernel.
    pub hypercall_eid: Option<u32>,
    /// Software TLB over the sv32 walk (flushed on SFENCE.VMA / satp write).
    tlb: Tlb,
    /// Comparison-coverage (CMPLOG) log: `Some(log)` while recording is enabled via
    /// [`Cpu::set_cmplog`], `None` (the default) otherwise. Every retired `Branch` and every
    /// `Op::{Sub,Xor}` (the common equality-compare lowering) appends its `(rs1_val, rs2_val)`
    /// pair, capped at [`CMPLOG_CAP`]. `None` costs exactly one pointer-sized discriminant check
    /// per instruction and no allocation — the hot fuzzing loop pays nothing extra unless a case
    /// explicitly opts in via `--cmplog`.
    cmplog: Option<Vec<(u32, u32)>>,
    /// UBSAN div-by-zero log: `Some(log)` while recording is enabled via [`Cpu::set_ubsan`],
    /// `None` (the default) otherwise. RISC-V *defines* DIV/0 (result all-ones) and REM/0 (result
    /// the dividend) — no trap — so this is the only way to observe it
    /// (`docs/emulator-sanitizers.md`'s UBSAN section). Every retired `Mul`-shaped instruction
    /// whose op is `Div`/`Divu`/`Rem`/`Remu` and whose divisor is `0` appends the faulting `pc`,
    /// capped at [`UBSAN_CAP`]. `None` costs exactly one discriminant check per such instruction
    /// and no allocation — zero cost unless a case explicitly opts in via `--ubsan`.
    ubsan: Option<Vec<u32>>,
}

impl Cpu {
    pub fn new(entry: u32) -> Self {
        Self {
            regs: [0; 32],
            pc: entry,
            htif_tohost: None,
            insns_retired: 0,
            reservation: None,
            privilege: Priv::M,
            csr: Csr::default(),
            hypercall_eid: None,
            tlb: Tlb::new(),
            cmplog: None,
            ubsan: None,
        }
    }

    /// Enable or disable comparison-coverage recording. Enabling (re)starts from an empty log;
    /// disabling drops any log content and reverts `step` to zero-cost. Purely observational —
    /// toggling it never changes what a program computes, only what side-channel is recorded.
    pub fn set_cmplog(&mut self, on: bool) {
        self.cmplog = if on { Some(Vec::new()) } else { None };
    }

    /// True if comparison-coverage recording is currently enabled.
    pub fn cmplog_enabled(&self) -> bool {
        self.cmplog.is_some()
    }

    /// Drain the recorded comparison operand pairs (`(rs1_val, rs2_val)` from every executed
    /// `Branch` and `Sub`/`Xor`), leaving recording enabled with a freshly emptied log. Returns an
    /// empty vec if recording was never enabled.
    pub fn cmplog_take(&mut self) -> Vec<(u32, u32)> {
        match &mut self.cmplog {
            Some(log) => std::mem::take(log),
            None => Vec::new(),
        }
    }

    /// Enable or disable UBSAN div-by-zero recording (`docs/emulator-sanitizers.md`'s UBSAN
    /// section). Enabling (re)starts from an empty log; disabling drops any log content and
    /// reverts execution to zero extra cost — mirrors [`Cpu::set_cmplog`]'s cost model exactly.
    pub fn set_ubsan(&mut self, on: bool) {
        self.ubsan = if on { Some(Vec::new()) } else { None };
    }

    /// True if UBSAN div-by-zero recording is currently enabled.
    pub fn ubsan_enabled(&self) -> bool {
        self.ubsan.is_some()
    }

    /// Drain the recorded div-by-zero PCs, leaving recording enabled with a freshly emptied log.
    /// Returns an empty vec if recording was never enabled.
    pub fn ubsan_take(&mut self) -> Vec<u32> {
        match &mut self.ubsan {
            Some(log) => std::mem::take(log),
            None => Vec::new(),
        }
    }

    #[inline]
    fn rd_reg(&self, i: u8) -> u32 {
        if i == 0 { 0 } else { self.regs[i as usize] }
    }

    #[inline]
    fn wr_reg(&mut self, i: u8, v: u32) {
        if i != 0 {
            self.regs[i as usize] = v;
        }
    }

    /// Effective privilege for a data access (loads/stores honour MPRV/MPP; fetches do not).
    fn effective_priv(&self, access: Access) -> Priv {
        if access != Access::Exec && self.csr.mstatus & sys::MSTATUS_MPRV != 0 {
            Priv::from_bits((self.csr.mstatus & sys::MSTATUS_MPP) >> 11)
        } else {
            self.privilege
        }
    }

    /// Translate a virtual address to physical via sv32, or identity in M-mode / paging-off.
    /// Enforces architectural R/W/X/U permissions and updates the PTE A/D bits.
    pub fn xlate(&mut self, bus: &mut dyn Bus, va: u32, access: Access) -> Result<u32, Trap> {
        let p = self.effective_priv(access);
        // satp.MODE: bit 31 (0 = Bare, 1 = Sv32). M-mode data/fetch is untranslated.
        if p == Priv::M || (self.csr.satp >> 31) == 0 {
            return Ok(va);
        }
        let fault = |access| Trap::Exception {
            cause: match access {
                Access::Exec => sys::E_INSTR_PAGE_FAULT,
                Access::Read => sys::E_LOAD_PAGE_FAULT,
                Access::Write => sys::E_STORE_PAGE_FAULT,
            },
            tval: va,
        };
        let sum = self.csr.mstatus & sys::MSTATUS_SUM != 0;
        let mxr = self.csr.mstatus & sys::MSTATUS_MXR != 0;

        // TLB consult: a hit reproduces the walk's leaf permission/privilege check against the
        // cached raw PTE bits and returns the translation without touching the page tables. A write
        // to a page whose dirty bit isn't known-set falls through to the walk (which sets D).
        let page_vpn = va >> 12;
        if let Some(e) = self.tlb.get(page_vpn) {
            if !leaf_perm_priv_ok(access, p, sum, mxr, LeafBits { r: e.r, w: e.w, x: e.x, u: e.u }) {
                return Err(fault(access));
            }
            if !(access == Access::Write && !e.d_set) {
                return Ok((e.ppn << 12) | (va & 0xfff));
            }
            // else: write to a not-yet-dirty page — fall through to the walk to set D and refill.
        }

        let vpn = [(va >> 12) & 0x3ff, (va >> 22) & 0x3ff];
        let mut a = (self.csr.satp & 0x3f_ffff) << 12; // root page-table PA
        for level in (0..2usize).rev() {
            let pte_addr = a.wrapping_add(vpn[level] * 4);
            let pte = bus.load(pte_addr, 4).map_err(|_| fault(access))?;
            let (v, r, w, x, u) = (
                pte & 1,
                (pte >> 1) & 1,
                (pte >> 2) & 1,
                (pte >> 3) & 1,
                (pte >> 4) & 1,
            );
            if v == 0 || (r == 0 && w == 1) {
                return Err(fault(access));
            }
            if r == 1 || x == 1 {
                // Leaf PTE — check permissions.
                if !leaf_perm_priv_ok(access, p, sum, mxr, LeafBits { r: r == 1, w: w == 1, x: x == 1, u: u == 1 }) {
                    return Err(fault(access));
                }
                let ppn1 = (pte >> 20) & 0xfff;
                let ppn0 = (pte >> 10) & 0x3ff;
                if level == 1 && ppn0 != 0 {
                    return Err(fault(access)); // misaligned superpage
                }
                // Set A (and D on write) in the PTE.
                let need_d = access == Access::Write;
                if (pte >> 6) & 1 == 0 || (need_d && (pte >> 7) & 1 == 0) {
                    let new = pte | (1 << 6) | if need_d { 1 << 7 } else { 0 };
                    bus.store(pte_addr, 4, new).map_err(|_| fault(access))?;
                }
                let pa = if level == 1 {
                    (ppn1 << 22) | (((va >> 12) & 0x3ff) << 12) | (va & 0xfff)
                } else {
                    (((pte >> 10) & 0x3f_ffff) << 12) | (va & 0xfff)
                };
                // Refill the TLB with this leaf's translation + permission bits. `d_set` reflects
                // whether the PTE's D bit is set now (originally, or just set by the write above).
                self.tlb.insert(TlbEntry {
                    vpn: page_vpn,
                    ppn: pa >> 12,
                    r: r == 1,
                    w: w == 1,
                    x: x == 1,
                    u: u == 1,
                    d_set: need_d || (pte >> 7) & 1 == 1,
                });
                return Ok(pa);
            }
            // Non-leaf: descend.
            a = ((pte >> 10) & 0x3f_ffff) << 12;
        }
        Err(fault(access))
    }

    /// Read-only, non-mutating sv32 walk against a [`Golden`] image instead of a live `Bus` — PR4
    /// of the software-COW design (`docs/cow-shared-ram.md`). Uses *this* CPU's `csr.satp`/
    /// `privilege`/`mstatus` (SUM/MXR/effective-priv) exactly like [`Cpu::xlate`], and performs the
    /// identical two-level walk + leaf permission/privilege check (via the same
    /// [`leaf_perm_priv_ok`] helper `xlate` calls) — but reads PTEs straight from `golden`
    /// (bypassing any `Bus`, mutating nothing) and **declines** (`Err(XlateDeclined)`) instead of
    /// succeeding whenever the mutating `xlate` would need to touch anything, or whenever `golden`
    /// cannot be trusted for a page-table read: a PTE read outside golden's bounds or lacking
    /// `PERM_READ`; `page_overlaid` reporting that a page-table page has been privately COW'd in
    /// some lane since `golden` was captured (its *current* content then lives in a per-lane
    /// overlay, not in `golden` — the common case once a kernel constructs its own page tables at
    /// runtime, since `golden` is typically captured once, well before that); a misaligned
    /// superpage; an architectural translation fault (unmapped/permission); or — the
    /// correctness-critical case — a leaf whose A bit (or D bit, for a write access) isn't
    /// *already* set, since setting it is a mutation only the real per-lane `xlate` may perform. A
    /// clean `Ok(pa)` therefore guarantees the mutating `xlate` would compute the exact same `pa`
    /// without touching a single byte — exactly what a converged `VecSystem` group speculatively
    /// translating against a shared golden image needs.
    ///
    /// `page_overlaid(addr)` is called once per page-table level, before trusting `golden`'s
    /// content at that physical address, so the caller (which owns the per-lane `CowRam`s and thus
    /// knows what's actually been privately dirtied) can veto a stale golden read. Note this only
    /// guards the page-*table* reads this walk itself performs; the caller is still responsible for
    /// checking overlay status of the *leaf* physical page the walk resolves to before trusting its
    /// content — that page is data/code, not a page table, and this function never reads it. See
    /// `xlate_golden_readonly_matches_xlate_and_declines_on_unset_ad` below for the differential
    /// proof.
    pub fn xlate_golden_readonly(
        &self,
        golden: &Golden,
        va: u32,
        access: Access,
        mut page_overlaid: impl FnMut(u32) -> bool,
    ) -> Result<u32, XlateDeclined> {
        let p = self.effective_priv(access);
        if p == Priv::M || (self.csr.satp >> 31) == 0 {
            return Ok(va);
        }
        let sum = self.csr.mstatus & sys::MSTATUS_SUM != 0;
        let mxr = self.csr.mstatus & sys::MSTATUS_MXR != 0;

        let vpn = [(va >> 12) & 0x3ff, (va >> 22) & 0x3ff];
        let mut a = (self.csr.satp & 0x3f_ffff) << 12; // root page-table PA
        for level in (0..2usize).rev() {
            let pte_addr = a.wrapping_add(vpn[level] * 4);
            if page_overlaid(pte_addr) {
                return Err(XlateDeclined); // this page-table page's real content is per-lane
            }
            let pte = golden.read_u32(pte_addr).ok_or(XlateDeclined)?;
            let (v, r, w, x, u) = (
                pte & 1,
                (pte >> 1) & 1,
                (pte >> 2) & 1,
                (pte >> 3) & 1,
                (pte >> 4) & 1,
            );
            if v == 0 || (r == 0 && w == 1) {
                return Err(XlateDeclined);
            }
            if r == 1 || x == 1 {
                if !leaf_perm_priv_ok(access, p, sum, mxr, LeafBits { r: r == 1, w: w == 1, x: x == 1, u: u == 1 }) {
                    return Err(XlateDeclined);
                }
                let ppn1 = (pte >> 20) & 0xfff;
                let ppn0 = (pte >> 10) & 0x3ff;
                if level == 1 && ppn0 != 0 {
                    return Err(XlateDeclined); // misaligned superpage — let the real walk fault
                }
                // A (and D, for a write) must already be set: setting it is a mutation only the
                // real per-lane `xlate` may perform.
                let need_d = access == Access::Write;
                let a_set = (pte >> 6) & 1 == 1;
                let d_set = (pte >> 7) & 1 == 1;
                if !a_set || (need_d && !d_set) {
                    return Err(XlateDeclined);
                }
                let pa = if level == 1 {
                    (ppn1 << 22) | (((va >> 12) & 0x3ff) << 12) | (va & 0xfff)
                } else {
                    (((pte >> 10) & 0x3f_ffff) << 12) | (va & 0xfff)
                };
                return Ok(pa);
            }
            // Non-leaf: descend.
            a = ((pte >> 10) & 0x3f_ffff) << 12;
        }
        Err(XlateDeclined)
    }

    fn fetch16(&mut self, bus: &mut dyn Bus, va: u32) -> Result<u16, Trap> {
        let pa = self.xlate(bus, va, Access::Exec)?;
        bus.ifetch16(pa).map_err(Trap::Mem)
    }

    #[inline]
    fn crosses_page(va: u32, size: u8) -> bool {
        (va & 0xfff) + size as u32 > 0x1000
    }
    #[inline]
    fn misaligned(va: u32, size: u8) -> bool {
        va & (size as u32 - 1) != 0
    }

    /// Translated load. `size` in bytes; `signed` sign-extends sub-word loads. Unaligned and
    /// page-crossing accesses are serviced byte-wise (native unaligned support — the kernel's
    /// `check_unaligned_access_emulated` probe expects this to just work).
    fn load(&mut self, bus: &mut dyn Bus, va: u32, size: u8, signed: bool) -> Result<u32, Trap> {
        let v = if Self::misaligned(va, size) || Self::crosses_page(va, size) {
            let mut acc = 0u32;
            for i in 0..size as u32 {
                let pa = self.xlate(bus, va.wrapping_add(i), Access::Read)?;
                let b = bus.load(pa, 1).map_err(Trap::Mem)?;
                acc |= (b & 0xff) << (8 * i);
            }
            acc
        } else {
            let pa = self.xlate(bus, va, Access::Read)?;
            bus.load(pa, size).map_err(Trap::Mem)?
        };
        Ok(if signed {
            match size {
                1 => v as u8 as i8 as i32 as u32,
                2 => v as u16 as i16 as i32 as u32,
                _ => v,
            }
        } else {
            v
        })
    }

    /// Translated store. `size` in bytes. Unaligned/page-crossing stores go byte-wise.
    fn store(&mut self, bus: &mut dyn Bus, va: u32, size: u8, val: u32) -> Result<(), Trap> {
        if Self::misaligned(va, size) || Self::crosses_page(va, size) {
            for i in 0..size as u32 {
                let pa = self.xlate(bus, va.wrapping_add(i), Access::Write)?;
                bus.store(pa, 1, (val >> (8 * i)) & 0xff).map_err(Trap::Mem)?;
            }
            Ok(())
        } else {
            let pa = self.xlate(bus, va, Access::Write)?;
            bus.store(pa, size, val).map_err(Trap::Mem)
        }
    }

    /// Execute one instruction. Advances `pc` and `insns_retired`.
    pub fn step(&mut self, bus: &mut dyn Bus) -> Result<Exit, Trap> {
        let pc = self.pc;
        // Variable-length fetch: a half-word whose low 2 bits != 0b11 is a 16-bit compressed
        // instruction (IALIGN=16); otherwise it is a 32-bit instruction.
        let lo = self.fetch16(bus, pc)?;
        let (inst, ilen, iword) = if lo & 0x3 != 0x3 {
            (decode_compressed(lo), 2u32, lo as u32)
        } else {
            let hi = self.fetch16(bus, pc.wrapping_add(2))?;
            let w = (lo as u32) | ((hi as u32) << 16);
            (decode(w), 4u32, w)
        };
        self.exec_one(bus, inst, pc, ilen, iword)
    }

    /// Execute one already-decoded instruction — the pure execute stage of [`Cpu::step`], factored
    /// out so a block cache (`fs-jit`) can replay a previously fetched+decoded instruction without
    /// repeating the fetch (translate + permission-checked `Bus::ifetch16`) or the decode. `pc` is
    /// this instruction's address (used for pc-relative `Auipc`/`Jal`/`Branch` targets, and as the
    /// `next`-pc for `Ecall`/`Ebreak`, which do not advance `pc`); it must be the CURRENT virtual
    /// address the instruction is being executed at (not necessarily the address it was originally
    /// decoded from — the decoded [`Inst`] IL carries no absolute address, only offsets read
    /// straight from the instruction encoding, so replaying it at a different but byte-identical VA
    /// — e.g. shared kernel `.text` reached via a different `satp` — computes the correct target).
    /// `ilen` is the encoded length (2 for RVC, 4 otherwise), used for the sequential fall-through
    /// pc. `iword` is the raw encoding, consulted only by the Csr/Mret/Sret illegal-instruction
    /// paths (never RVC-encoded, so always the full 32-bit word there). Advances `self.pc`/
    /// `self.insns_retired` exactly as `step` did — this is a pure refactor, zero behavior change.
    pub fn exec_one(
        &mut self,
        bus: &mut dyn Bus,
        inst: Inst,
        pc: u32,
        ilen: u32,
        iword: u32,
    ) -> Result<Exit, Trap> {
        let mut next = pc.wrapping_add(ilen);
        let mut exit = Exit::Continue;

        match inst {
            Inst::Lui { rd, imm } => self.wr_reg(rd, imm),
            Inst::Auipc { rd, imm } => self.wr_reg(rd, pc.wrapping_add(imm)),
            Inst::Jal { rd, imm } => {
                self.wr_reg(rd, next);
                next = pc.wrapping_add(imm as u32);
            }
            Inst::Jalr { rd, rs1, imm } => {
                let target = self.rd_reg(rs1).wrapping_add(imm as u32) & !1;
                self.wr_reg(rd, next);
                next = target;
            }
            Inst::Branch { op, rs1, rs2, imm } => {
                let a = self.rd_reg(rs1);
                let b = self.rd_reg(rs2);
                if let Some(log) = self.cmplog.as_mut()
                    && log.len() < CMPLOG_CAP
                {
                    log.push((a, b));
                }
                let taken = match op {
                    BranchOp::Eq => a == b,
                    BranchOp::Ne => a != b,
                    BranchOp::Lt => (a as i32) < (b as i32),
                    BranchOp::Ge => (a as i32) >= (b as i32),
                    BranchOp::Ltu => a < b,
                    BranchOp::Geu => a >= b,
                };
                if taken {
                    next = pc.wrapping_add(imm as u32);
                }
            }
            Inst::Load { op, rd, rs1, imm } => {
                let addr = self.rd_reg(rs1).wrapping_add(imm as u32);
                let (size, signed) = match op {
                    LoadOp::Lb => (1, true),
                    LoadOp::Lbu => (1, false),
                    LoadOp::Lh => (2, true),
                    LoadOp::Lhu => (2, false),
                    LoadOp::Lw => (4, false),
                };
                let v = self.load(bus, addr, size, signed)?;
                self.wr_reg(rd, v);
            }
            Inst::Store { op, rs1, rs2, imm } => {
                let addr = self.rd_reg(rs1).wrapping_add(imm as u32);
                let val = self.rd_reg(rs2);
                let size = match op {
                    StoreOp::Sb => 1,
                    StoreOp::Sh => 2,
                    StoreOp::Sw => 4,
                };
                // HTIF: a word store to `tohost` with bit0 set is an exit request.
                if op == StoreOp::Sw && self.htif_tohost == Some(addr) {
                    self.store(bus, addr, 4, val)?;
                    if val & 1 != 0 {
                        exit = Exit::Halt(val >> 1);
                    }
                } else {
                    self.store(bus, addr, size, val)?;
                }
            }
            Inst::OpImm { op, rd, rs1, imm } => {
                let v = alu(op, self.rd_reg(rs1), imm as u32);
                self.wr_reg(rd, v);
            }
            Inst::Op { op, rd, rs1, rs2 } => {
                let ra = self.rd_reg(rs1);
                let rb = self.rd_reg(rs2);
                // Sub/Xor are the common compiler lowering for an equality compare (`a - b == 0`
                // / `a ^ b == 0`) that a later branch tests — log their operands too, same as a
                // direct Branch, so CMPLOG catches `if (x == y)` patterns the compiler didn't
                // lower straight to a Branch on x/y.
                if matches!(op, AluOp::Sub | AluOp::Xor)
                    && let Some(log) = self.cmplog.as_mut()
                    && log.len() < CMPLOG_CAP
                {
                    log.push((ra, rb));
                }
                let v = alu(op, ra, rb);
                self.wr_reg(rd, v);
            }
            Inst::Mul { op, rd, rs1, rs2 } => {
                let a = self.rd_reg(rs1);
                let b = self.rd_reg(rs2);
                // UBSAN div-by-zero: RISC-V defines DIV/0 (all-ones) and REM/0 (dividend) rather
                // than trapping, so this is the only way to catch it — record the faulting `pc`
                // when recording is enabled (see `Cpu::set_ubsan`'s doc comment).
                if matches!(op, MulOp::Div | MulOp::Divu | MulOp::Rem | MulOp::Remu)
                    && b == 0
                    && let Some(log) = self.ubsan.as_mut()
                    && log.len() < UBSAN_CAP
                {
                    log.push(pc);
                }
                let v = muldiv(op, a, b);
                self.wr_reg(rd, v);
            }
            Inst::LrW { rd, rs1, .. } => {
                let addr = self.rd_reg(rs1);
                let v = self.load(bus, addr, 4, false)?;
                self.reservation = Some(addr);
                self.wr_reg(rd, v);
            }
            Inst::ScW { rd, rs1, rs2, .. } => {
                let addr = self.rd_reg(rs1);
                let success = self.reservation == Some(addr);
                if success {
                    let v = self.rd_reg(rs2);
                    self.store(bus, addr, 4, v)?;
                }
                // A reservation is single-use, and any trap/context-switch would clear it too.
                self.reservation = None;
                self.wr_reg(rd, if success { 0 } else { 1 });
            }
            Inst::AmoW { op, rd, rs1, rs2, .. } => {
                let addr = self.rd_reg(rs1);
                let pa = self.xlate(bus, addr, Access::Write)?;
                let old = bus.load(pa, 4).map_err(Trap::Mem)?;
                let src = self.rd_reg(rs2);
                let result = match op {
                    AmoOp::Swap => src,
                    AmoOp::Add => old.wrapping_add(src),
                    AmoOp::Xor => old ^ src,
                    AmoOp::And => old & src,
                    AmoOp::Or => old | src,
                    AmoOp::Min => (old as i32).min(src as i32) as u32,
                    AmoOp::Max => (old as i32).max(src as i32) as u32,
                    AmoOp::Minu => old.min(src),
                    AmoOp::Maxu => old.max(src),
                };
                bus.store(pa, 4, result).map_err(Trap::Mem)?;
                self.reservation = None;
                self.wr_reg(rd, old);
            }
            Inst::Fence => {}
            // Ecall/Ebreak do not advance pc: in full-system mode the trap's epc must be the
            // instruction address; bare M0/M1 uses these as an exit and ignores pc.
            Inst::Ecall => {
                exit = Exit::Ecall;
                next = pc;
            }
            Inst::Ebreak => {
                exit = Exit::Ebreak;
                next = pc;
            }
            Inst::Csr { op, rd, src, csr } => {
                // Counters need the retired-instruction count / virtual time, which live on the hart.
                let old = match csr {
                    sys::CYCLE | sys::INSTRET => self.insns_retired as u32,
                    sys::CYCLEH | sys::INSTRETH => (self.insns_retired >> 32) as u32,
                    sys::TIME => self.virtual_time() as u32,
                    sys::TIMEH => (self.virtual_time() >> 32) as u32,
                    _ => self
                        .csr
                        .read(csr, self.privilege)
                        .map_err(|_| Trap::Illegal { pc, raw: iword })?,
                };
                let src_val = match op {
                    CsrOp::Rwi | CsrOp::Rsi | CsrOp::Rci => src as u32,
                    _ => self.rd_reg(src),
                };
                let do_write = match op {
                    CsrOp::Rw | CsrOp::Rwi => true,
                    _ => src != 0, // set/clear with x0/uimm==0 performs no write
                };
                let new = match op {
                    CsrOp::Rw | CsrOp::Rwi => src_val,
                    CsrOp::Rs | CsrOp::Rsi => old | src_val,
                    CsrOp::Rc | CsrOp::Rci => old & !src_val,
                };
                if do_write {
                    self.csr
                        .write(csr, new, self.privilege)
                        .map_err(|_| Trap::Illegal { pc, raw: iword })?;
                    // A satp write switches the address space — every cached translation is now
                    // for the wrong page tables. Flush (we key the TLB on VPN only, no ASID).
                    if csr == sys::SATP {
                        self.tlb.flush();
                    }
                }
                self.wr_reg(rd, old);
            }
            Inst::Mret => {
                if self.privilege.level() < Priv::M.level() {
                    return Err(Trap::Illegal { pc, raw: iword });
                }
                let mpp = (self.csr.mstatus & sys::MSTATUS_MPP) >> 11;
                let mpie = self.csr.mstatus & sys::MSTATUS_MPIE != 0;
                let mut s = self.csr.mstatus;
                s = (s & !sys::MSTATUS_MIE) | if mpie { sys::MSTATUS_MIE } else { 0 };
                s |= sys::MSTATUS_MPIE;
                s &= !sys::MSTATUS_MPP;
                let newp = Priv::from_bits(mpp);
                if newp != Priv::M {
                    s &= !sys::MSTATUS_MPRV;
                }
                self.csr.mstatus = s;
                self.privilege = newp;
                next = self.csr.mepc;
            }
            Inst::Sret => {
                if self.privilege.level() < Priv::S.level() {
                    return Err(Trap::Illegal { pc, raw: iword });
                }
                let spp = (self.csr.mstatus & sys::MSTATUS_SPP) >> 8;
                let spie = self.csr.mstatus & sys::MSTATUS_SPIE != 0;
                let mut s = self.csr.mstatus;
                s = (s & !sys::MSTATUS_SIE) | if spie { sys::MSTATUS_SIE } else { 0 };
                s |= sys::MSTATUS_SPIE;
                s &= !sys::MSTATUS_SPP;
                let newp = Priv::from_bits(spp);
                if newp != Priv::M {
                    s &= !sys::MSTATUS_MPRV;
                }
                self.csr.mstatus = s;
                self.privilege = newp;
                next = self.csr.sepc;
            }
            Inst::Wfi => {} // no-op hint in a deterministic core
            Inst::SfenceVma => self.tlb.flush(), // conservative: flush the whole software TLB
            Inst::Illegal(raw) => return Err(Trap::Illegal { pc, raw }),
        }

        self.pc = next;
        self.insns_retired += 1;
        Ok(exit)
    }

    /// Deliver a trap (exception or interrupt), honouring medeleg/mideleg delegation to S-mode.
    /// `self.pc` must be the trapping instruction's address on entry.
    pub fn take_trap(&mut self, cause: u32, tval: u32, interrupt: bool) {
        let deleg = if interrupt {
            self.csr.mideleg
        } else {
            self.csr.medeleg
        };
        let to_s = self.privilege.level() <= Priv::S.level() && (deleg >> cause) & 1 != 0;
        let epc = self.pc;
        let code = if interrupt { 1 << 31 } else { 0 } | cause;

        if to_s {
            self.csr.sepc = epc;
            self.csr.scause = code;
            self.csr.stval = tval;
            let sie = self.csr.mstatus & sys::MSTATUS_SIE != 0;
            let mut s = self.csr.mstatus;
            s = (s & !sys::MSTATUS_SPIE) | if sie { sys::MSTATUS_SPIE } else { 0 };
            s &= !sys::MSTATUS_SIE;
            s = (s & !sys::MSTATUS_SPP) | ((self.privilege.level() & 1) << 8);
            self.csr.mstatus = s;
            self.privilege = Priv::S;
            self.pc = trap_vector(self.csr.stvec, cause, interrupt);
        } else {
            self.csr.mepc = epc;
            self.csr.mcause = code;
            self.csr.mtval = tval;
            let mie = self.csr.mstatus & sys::MSTATUS_MIE != 0;
            let mut s = self.csr.mstatus;
            s = (s & !sys::MSTATUS_MPIE) | if mie { sys::MSTATUS_MPIE } else { 0 };
            s &= !sys::MSTATUS_MIE;
            s = (s & !sys::MSTATUS_MPP) | (self.privilege.level() << 11);
            self.csr.mstatus = s;
            self.privilege = Priv::M;
            self.pc = trap_vector(self.csr.mtvec, cause, interrupt);
        }
    }

    /// The highest-priority pending+enabled interrupt cause, if any (standard priority order).
    fn pending_interrupt(&self) -> Option<u32> {
        let pending = self.csr.mip & self.csr.mie;
        if pending == 0 {
            return None;
        }
        // M interrupts (not delegated) then S interrupts (delegated), each gated by the global
        // enable for the current mode.
        for &code in &[11u32, 3, 7, 9, 1, 5] {
            let bit = 1 << code;
            if pending & bit == 0 {
                continue;
            }
            let to_s = (self.csr.mideleg >> code) & 1 != 0;
            let enabled = if to_s {
                self.privilege == Priv::U
                    || (self.privilege == Priv::S && self.csr.mstatus & sys::MSTATUS_SIE != 0)
            } else {
                self.privilege != Priv::M || self.csr.mstatus & sys::MSTATUS_MIE != 0
            };
            if enabled {
                return Some(code);
            }
        }
        None
    }

    /// Virtual time = retired instruction count (deterministic; decision #7).
    pub fn virtual_time(&self) -> u64 {
        self.insns_retired
    }

    /// Refresh timer-interrupt-pending bits from virtual time (Sstc STIP and the M-timer MTIP).
    fn update_timers(&mut self) {
        let t = self.virtual_time();
        let stip = 1 << 5;
        if t >= self.csr.stimecmp {
            self.csr.mip |= stip;
        } else {
            self.csr.mip &= !stip;
        }
        let mtip = 1 << 7;
        if t >= self.csr.mtimecmp {
            self.csr.mip |= mtip;
        } else {
            self.csr.mip &= !mtip;
        }
    }

    /// Refresh timers and, if an interrupt is now pending and enabled, take it (redirecting `pc`
    /// to the trap vector). Returns `true` if a trap was taken. Factored out of `step_system`'s
    /// opening so a cached-block runner (`fs-jit`) can reproduce the exact same per-instruction
    /// interrupt-polling semantics between two cached (fetch/decode-free) instructions, without
    /// pulling in the fetch+decode `step` does. Pure refactor — `step_system` calling this first is
    /// byte-identical to its previous inlined check.
    pub fn poll_interrupt(&mut self) -> bool {
        self.update_timers();
        if let Some(code) = self.pending_interrupt() {
            self.take_trap(code, 0, true);
            true
        } else {
            false
        }
    }

    /// Turn a [`Cpu::step`]/[`Cpu::exec_one`] result into a [`SysExit`]: vector any trap, and
    /// intercept a registered fuzzing hypercall before it would be delivered to the guest kernel.
    /// This is exactly the match `step_system` used to run inline after calling `step`, factored
    /// out so `fs-jit` can apply the identical post-execution handling after calling `exec_one`
    /// directly (no behavior change — same arms, same order).
    pub fn finish_exit(&mut self, r: Result<Exit, Trap>) -> SysExit {
        match r {
            Ok(Exit::Continue) => SysExit::Continue,
            Ok(Exit::Halt(c)) => SysExit::Halt(c),
            Ok(Exit::Ecall) => {
                // Intercept fuzzing hypercalls (reserved a7) before delivering to the kernel.
                if let Some(eid) = self.hypercall_eid
                    && self.regs[17] == eid
                {
                    self.pc = self.pc.wrapping_add(4); // resume past the ecall on return
                    return SysExit::Hypercall(self.regs[10]);
                }
                let cause = match self.privilege {
                    Priv::U => sys::E_ECALL_U,
                    Priv::S => sys::E_ECALL_S,
                    Priv::M => sys::E_ECALL_M,
                };
                self.take_trap(cause, 0, false);
                SysExit::Continue
            }
            Ok(Exit::Ebreak) => {
                self.take_trap(sys::E_BREAKPOINT, self.pc, false);
                SysExit::Continue
            }
            Err(Trap::Illegal { raw, .. }) => {
                self.take_trap(sys::E_ILLEGAL, raw, false);
                SysExit::Continue
            }
            Err(Trap::Mem(f)) => {
                self.take_trap(mem_cause(f), f.addr, false);
                SysExit::Continue
            }
            Err(Trap::Exception { cause, tval }) => {
                self.take_trap(cause, tval, false);
                SysExit::Continue
            }
        }
    }

    /// One full-system step: refresh timers, take any pending interrupt, else execute one
    /// instruction and vector any resulting trap. Returns `Halt` on an HTIF `tohost` exit.
    pub fn step_system(&mut self, bus: &mut dyn Bus) -> SysExit {
        if self.poll_interrupt() {
            return SysExit::Continue;
        }
        let r = self.step(bus);
        self.finish_exit(r)
    }
}

/// Full-system step outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysExit {
    Continue,
    Halt(u32),
    /// A fuzzing hypercall (`ecall` with the reserved a7); carries a0 (the command).
    Hypercall(u32),
}

fn trap_vector(tvec: u32, cause: u32, interrupt: bool) -> u32 {
    let base = tvec & !0x3;
    if tvec & 0x3 == 1 && interrupt {
        base + 4 * cause
    } else {
        base
    }
}

fn mem_cause(f: Fault) -> u32 {
    match (f.access, f.kind) {
        (Access::Exec, FaultKind::Unaligned) => sys::E_INSTR_MISALIGNED,
        (Access::Exec, _) => sys::E_INSTR_ACCESS,
        (Access::Read, FaultKind::Unaligned) => sys::E_LOAD_MISALIGNED,
        (Access::Read, _) => sys::E_LOAD_ACCESS,
        (Access::Write, FaultKind::Unaligned) => sys::E_STORE_MISALIGNED,
        (Access::Write, _) => sys::E_STORE_ACCESS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};

    /// Assemble a program, run it in a fresh MMU, return (exit_code, cpu).
    fn run(program: &[u32]) -> (u32, Cpu) {
        let base = 0x8000_0000u32;
        let mut mmu = Mmu::new(base, 0x1_0000);
        // Whole window RW (scratch for loads/stores/atomics); code region also gets EXEC below.
        mmu.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        let mut bytes = Vec::new();
        for w in program {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mmu.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cpu = Cpu::new(base);
        for _ in 0..10_000 {
            match cpu.step(&mut mmu).unwrap() {
                Exit::Continue => {}
                Exit::Ecall => return (cpu.regs[A0 as usize], cpu),
                Exit::Ebreak => return (0, cpu),
                Exit::Halt(c) => return (c, cpu),
            }
        }
        panic!("program did not terminate");
    }

    #[test]
    fn sum_1_to_10_via_loop() {
        use asm::*;
        // a0 = sum(1..=10) = 55, then ecall exit.
        let prog = [
            addi(A0, X0, 0),   // 0: sum = 0
            addi(T0, X0, 1),   // 1: i = 1
            addi(T1, X0, 11),  // 2: limit = 11
            bge(T0, T1, 16),   // 3: if i >= 11 -> done (idx 7)
            add(A0, A0, T0),   // 4: sum += i
            addi(T0, T0, 1),   // 5: i++
            jal(X0, -12),      // 6: -> idx 3
            addi(A7, X0, 93),  // 7: done: a7 = exit
            ecall(),           // 8:
        ];
        let (code, cpu) = run(&prog);
        assert_eq!(code, 55);
        assert_eq!(cpu.regs[A7 as usize], 93);
    }

    #[test]
    fn m_extension_div_by_zero_and_overflow() {
        assert_eq!(muldiv(MulOp::Div, 10, 0), 0xffff_ffff);
        assert_eq!(muldiv(MulOp::Rem, 10, 0), 10);
        assert_eq!(muldiv(MulOp::Div, 0x8000_0000, 0xffff_ffff), 0x8000_0000);
        assert_eq!(muldiv(MulOp::Rem, 0x8000_0000, 0xffff_ffff), 0);
        assert_eq!(muldiv(MulOp::Mul, 6, 7), 42);
    }

    #[test]
    fn ubsan_records_div_by_zero_pc_only_when_enabled() {
        use asm::*;
        // a0 = 0 (divisor), t0 = 10 (dividend); divu t1, t0, a0 -> b == 0, then ecall exit.
        let prog = [
            addi(A0, X0, 0),   // 0: divisor = 0
            addi(T0, X0, 10),  // 1: t0 = 10 (dividend)
            divu(T1, T0, A0),  // 2: t1 = t0 / a0 (div-by-zero, saturating result, no trap)
            addi(A7, X0, 93),  // 3: exit
            ecall(),           // 4:
        ];
        let base = 0x8000_0000u32;
        let mut mmu = Mmu::new(base, 0x1_0000);
        mmu.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mmu.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

        // Disabled (default): zero cost, nothing recorded, execution unaffected.
        let mut cpu = Cpu::new(base);
        assert!(!cpu.ubsan_enabled());
        loop {
            match cpu.step(&mut mmu).unwrap() {
                Exit::Ecall => break,
                Exit::Continue => {}
                other => panic!("unexpected exit {other:?}"),
            }
        }
        assert_eq!(cpu.ubsan_take(), Vec::<u32>::new());

        // Enabled: the div-by-zero instruction's pc is recorded exactly once, and the saturating
        // (non-trapping) RISC-V semantics are unchanged.
        let mut mmu2 = Mmu::new(base, 0x1_0000);
        mmu2.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        mmu2.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cpu2 = Cpu::new(base);
        cpu2.set_ubsan(true);
        assert!(cpu2.ubsan_enabled());
        let div_pc = base + 2 * 4; // the divu instruction's address
        loop {
            match cpu2.step(&mut mmu2).unwrap() {
                Exit::Ecall => break,
                Exit::Continue => {}
                other => panic!("unexpected exit {other:?}"),
            }
        }
        assert_eq!(cpu2.regs[T1 as usize], 0xffff_ffff); // Divu/0 -> all-ones, no trap
        assert_eq!(cpu2.ubsan_take(), vec![div_pc]);
        // Draining clears the log but leaves recording enabled.
        assert_eq!(cpu2.ubsan_take(), Vec::<u32>::new());
        assert!(cpu2.ubsan_enabled());
    }

    #[test]
    fn a_extension_lr_sc_amo() {
        use asm::*;
        let scratch = 0x8000_1000u32; // within run()'s RW window, initialised to 0

        // t1 = scratch. amoadd += 7 (old 0 -> t3). lr/sc swaps 100 in (t6 = 0 success). Then
        // a0 = [t1] + t3 + t4 + t6 = 100 + 0 + 7 + 0 = 107.
        let prog = [
            lui(6, scratch),    // 0: t1 = scratch (low 12 bits zero)
            addi(7, X0, 7),     // 1: t2 = 7
            amoadd_w(28, 6, 7), // 2: t3 = [t1] (0); [t1] += 7 => [t1] = 7
            lr_w(29, 6),        // 3: t4 = [t1] (7); reserve
            addi(30, X0, 100),  // 4: t5 = 100
            sc_w(31, 6, 30),    // 5: [t1] = 100 (reserved); t6 = 0
            lw(A0, 6, 0),       // 6: a0 = [t1] (100)
            add(A0, A0, 28),    // 7: a0 += t3 (0)  => 100
            add(A0, A0, 29),    // 8: a0 += t4 (7)  => 107
            add(A0, A0, 31),    // 9: a0 += t6 (0)  => 107
            addi(A7, X0, 93),   // 10
            ecall(),            // 11
        ];
        let (code, _cpu) = run(&prog);
        assert_eq!(code, 107);
    }

    #[test]
    fn supervisor_timer_interrupt() {
        use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
        use sys::Priv;

        let base = 0x8000_0000u32;
        let tohost = 0x8000_2000u32;
        let handler = base + 0x100;
        let mut mmu = Mmu::new(base, 0x1_0000);
        mmu.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        // Loop forever in S-mode until the timer fires.
        mmu.map(base, &0x0000_006fu32.to_le_bytes(), PERM_READ | PERM_WRITE | PERM_EXEC)
            .unwrap();
        // Handler: a0 = 99, then HTIF exit.
        let mut code = Vec::new();
        for w in [asm::addi(A0, X0, 99), asm::lui(T0, tohost), asm::addi(T1, X0, 1), asm::sw(T0, T1, 0)] {
            code.extend_from_slice(&w.to_le_bytes());
        }
        mmu.map(handler, &code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

        let mut cpu = Cpu::new(base);
        cpu.privilege = Priv::S;
        cpu.htif_tohost = Some(tohost);
        cpu.csr.stvec = handler;
        cpu.csr.mideleg |= 1 << 5; // delegate the supervisor timer to S
        cpu.csr.mie |= 1 << 5; // STIE
        cpu.csr.mstatus |= sys::MSTATUS_SIE;
        cpu.csr.stimecmp = 5; // fire after 5 retired instructions

        let mut halted = false;
        for _ in 0..1000 {
            if let SysExit::Halt(_) = cpu.step_system(&mut mmu) {
                halted = true;
                break;
            }
        }
        assert!(halted, "timer interrupt never delivered");
        assert_eq!(cpu.regs[A0 as usize], 99); // handler ran
        assert_eq!(cpu.csr.scause, 0x8000_0005); // interrupt | supervisor-timer(5)
    }

    #[test]
    fn sv32_translation() {
        use fs_mmu::{Access, PERM_READ, PERM_WRITE};
        use sys::Priv;

        let base = 0x8000_0000u32;
        let mut mmu = Mmu::new(base, 0x0080_0000);
        mmu.protect(base, 0x0080_0000, PERM_READ | PERM_WRITE).unwrap();

        // Page tables: root @0x8000_1000, level-0 @0x8000_2000.
        let root = 0x8000_1000u32;
        let l0 = 0x8000_2000u32;
        let leaf = |pa: u32, flags: u32| ((pa >> 12) << 10) | flags; // A|D preset to skip writeback
        const V: u32 = 1;
        const R: u32 = 2;
        const W: u32 = 4;
        const X: u32 = 8;
        const U: u32 = 16;
        const AD: u32 = (1 << 6) | (1 << 7);

        // root[0] -> level-0 table (non-leaf, V only).
        mmu.write_u32(root, (l0 >> 12) << 10 | V).unwrap();
        // level-0[4] -> 0x8000_3000, RWX kernel page (U=0).  VA 0x0000_4000.
        mmu.write_u32(l0 + 4 * 4, leaf(0x8000_3000, V | R | W | X | AD)).unwrap();
        // level-0[8] -> 0x8000_4000, RW *user* page (U=1).   VA 0x0000_8000.
        mmu.write_u32(l0 + 8 * 4, leaf(0x8000_4000, V | R | W | U | AD)).unwrap();
        // root[1] -> 4 MiB superpage @0x8040_0000 (ppn0==0), RWX.  VA 0x0040_0000.
        mmu.write_u32(root + 4, leaf(0x8040_0000, V | R | W | X | AD)).unwrap();

        let mut cpu = Cpu::new(0);
        cpu.privilege = Priv::S;
        cpu.csr.satp = (1 << 31) | (root >> 12); // Sv32, root PPN

        // 4 KiB page: VA 0x4000 (+ offset) -> 0x8000_3000.
        assert_eq!(cpu.xlate(&mut mmu, 0x0000_4000, Access::Read).unwrap(), 0x8000_3000);
        assert_eq!(cpu.xlate(&mut mmu, 0x0000_402a, Access::Read).unwrap(), 0x8000_302a);
        // 4 MiB superpage: VA 0x0040_0123 -> 0x8040_0123.
        assert_eq!(cpu.xlate(&mut mmu, 0x0040_0123, Access::Exec).unwrap(), 0x8040_0123);
        // Unmapped VA faults.
        assert!(matches!(
            cpu.xlate(&mut mmu, 0x0001_0000, Access::Read),
            Err(Trap::Exception { cause, .. }) if cause == sys::E_LOAD_PAGE_FAULT
        ));
        // S-mode read of a U page faults unless SUM is set.
        assert!(cpu.xlate(&mut mmu, 0x0000_8000, Access::Read).is_err());
        cpu.csr.mstatus |= sys::MSTATUS_SUM;
        assert_eq!(cpu.xlate(&mut mmu, 0x0000_8000, Access::Read).unwrap(), 0x8000_4000);
        // But S-mode may never *execute* a U page, even with SUM.
        assert!(cpu.xlate(&mut mmu, 0x0000_8000, Access::Exec).is_err());
    }

    #[test]
    fn xlate_golden_readonly_matches_xlate_and_declines_on_unset_ad() {
        use fs_mmu::{Access, Golden, PERM_READ, PERM_WRITE};
        use sys::Priv;

        let base = 0x8000_0000u32;
        let mut mmu = Mmu::new(base, 0x0080_0000);
        mmu.protect(base, 0x0080_0000, PERM_READ | PERM_WRITE).unwrap();

        let root = 0x8000_1000u32;
        let l0 = 0x8000_2000u32;
        let leaf = |pa: u32, flags: u32| ((pa >> 12) << 10) | flags;
        const V: u32 = 1;
        const R: u32 = 2;
        const W: u32 = 4;
        const X: u32 = 8;
        const AD: u32 = (1 << 6) | (1 << 7);
        const A_ONLY: u32 = 1 << 6;

        // root[0] -> level-0 table (non-leaf).
        mmu.write_u32(root, (l0 >> 12) << 10 | V).unwrap();
        // VA 0x0000_4000: RWX kernel leaf, A+D already set ("already-accessed").
        mmu.write_u32(l0 + 4 * 4, leaf(0x8000_3000, V | R | W | X | AD)).unwrap();
        // VA 0x0000_8000: RW leaf, neither A nor D set ("never accessed").
        mmu.write_u32(l0 + 8 * 4, leaf(0x8000_4000, V | R | W)).unwrap();
        // VA 0x0000_c000: RW leaf, A set but D unset ("read but never written").
        mmu.write_u32(l0 + 12 * 4, leaf(0x8000_5000, V | R | W | A_ONLY)).unwrap();

        let mut cpu = Cpu::new(0);
        cpu.privilege = Priv::S;
        cpu.csr.satp = (1 << 31) | (root >> 12);

        let golden = Golden::from_mmu(&mmu);

        // Already-accessed page: the read-only walk agrees byte-for-byte with the real (mutating)
        // xlate, for both a read and an exec access (A/D already set, so xlate performs no
        // writeback either — the two walks must return identically).
        let va = 0x0000_4000u32;
        assert_eq!(
            cpu.xlate(&mut mmu, va, Access::Read).unwrap(),
            cpu.xlate_golden_readonly(&golden, va, Access::Read, |_| false).unwrap(),
        );
        assert_eq!(
            cpu.xlate(&mut mmu, va, Access::Exec).unwrap(),
            cpu.xlate_golden_readonly(&golden, va, Access::Exec, |_| false).unwrap(),
        );

        // Never-accessed page (A unset): the real xlate succeeds (and sets A as a side effect),
        // but the read-only walk must decline rather than silently succeeding without setting it.
        let va_unset = 0x0000_8000u32;
        assert!(cpu.xlate_golden_readonly(&golden, va_unset, Access::Read, |_| false).is_err());

        // A-only page: the read-only walk succeeds for a Read (D isn't required for a read), but
        // declines for a Write (D would need to be set, a mutation only the real xlate may do).
        let va_a_only = 0x0000_c000u32;
        assert_eq!(
            cpu.xlate_golden_readonly(&golden, va_a_only, Access::Read, |_| false).unwrap(),
            0x8000_5000,
        );
        assert!(cpu.xlate_golden_readonly(&golden, va_a_only, Access::Write, |_| false).is_err());

        // `page_overlaid` returning `true` for a page-table page must decline the walk even
        // though the leaf itself is otherwise a clean, already-accessed success — the caller's
        // signal that golden's PTE bytes there may be stale (some lane privately COW'd that page
        // since golden was captured, e.g. a kernel constructing its own page tables at runtime).
        assert!(cpu.xlate_golden_readonly(&golden, va, Access::Read, |_| true).is_err());
    }

    #[test]
    fn decode_roundtrip_examples() {
        assert_eq!(
            decode(asm::addi(A0, X0, 93)),
            Inst::OpImm { op: AluOp::Add, rd: A0, rs1: X0, imm: 93 }
        );
        assert_eq!(decode(asm::ecall()), Inst::Ecall);
        assert_eq!(
            decode(asm::add(A0, A0, T0)),
            Inst::Op { op: AluOp::Add, rd: A0, rs1: A0, rs2: T0 }
        );
    }
}
