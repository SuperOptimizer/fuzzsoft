//! Minimal RV32 instruction encoders — used to hand-assemble hermetic test programs
//! (decision #26) and to build the `gen-elf` sample. These mirror `decode()` and are the
//! reference the M1 differential tests will also lean on.

#[inline]
fn i_type(op: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
    ((imm as u32 & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
}

#[inline]
fn r_type(op: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | op
}

#[inline]
fn b_type(op: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let i = imm as u32;
    ((i >> 12) & 1) << 31
        | ((i >> 5) & 0x3f) << 25
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((i >> 1) & 0xf) << 8
        | ((i >> 11) & 1) << 7
        | op
}

#[inline]
fn j_type(op: u32, rd: u8, imm: i32) -> u32 {
    let i = imm as u32;
    ((i >> 20) & 1) << 31
        | ((i >> 1) & 0x3ff) << 21
        | ((i >> 11) & 1) << 20
        | ((i >> 12) & 0xff) << 12
        | ((rd as u32) << 7)
        | op
}

pub fn lui(rd: u8, imm: u32) -> u32 {
    (imm & 0xffff_f000) | ((rd as u32) << 7) | 0x37
}
pub fn addi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 0, rd, rs1, imm)
}
pub fn add(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 0, 0x00, rd, rs1, rs2)
}
pub fn sub(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 0, 0x20, rd, rs1, rs2)
}
pub fn mul(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 0, 0x01, rd, rs1, rs2)
}
pub fn beq(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(0x63, 0, rs1, rs2, imm)
}
pub fn bne(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(0x63, 1, rs1, rs2, imm)
}
pub fn bge(rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(0x63, 5, rs1, rs2, imm)
}
pub fn jal(rd: u8, imm: i32) -> u32 {
    j_type(0x6f, rd, imm)
}
pub fn ecall() -> u32 {
    0x0000_0073
}

pub fn lw(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 2, rd, rs1, imm)
}
pub fn sw(rs1: u8, rs2: u8, imm: i32) -> u32 {
    // S-type: base is rs1, source is rs2.
    let i = imm as u32;
    ((i >> 5) & 0x7f) << 25
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (2 << 12)
        | ((i & 0x1f) << 7)
        | 0x23
}

fn amo(funct5: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct5 << 27) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (0b010 << 12) | ((rd as u32) << 7) | 0x2f
}
pub fn lr_w(rd: u8, rs1: u8) -> u32 {
    amo(0x02, rd, rs1, 0)
}
pub fn sc_w(rd: u8, rs1: u8, rs2: u8) -> u32 {
    amo(0x03, rd, rs1, rs2)
}
pub fn amoadd_w(rd: u8, rs1: u8, rs2: u8) -> u32 {
    amo(0x00, rd, rs1, rs2)
}
pub fn amoswap_w(rd: u8, rs1: u8, rs2: u8) -> u32 {
    amo(0x01, rd, rs1, rs2)
}
