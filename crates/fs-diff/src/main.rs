//! M1 differential tester: generate random RV32IMAC programs, run them in fuzzsoft AND Spike, and
//! compare the final register state. Determinism is guaranteed by a prologue that zeroes x1..x30
//! (overriding Spike's boot-rom register state) and points x31 at a scratch page, so both machines
//! start from a known state and any divergence is a real bug in our core.
//!
//! Modes:
//!   --mode full  (default): OP-IMM, OP, the full M extension, aligned loads/stores against the
//!                 scratch page (x31 base), and forward-only branches/jumps.
//!   --mode c    : straight-line compressed (RVC) ALU/register ops — validates the C decoder.
//!   --mode priv : CSR + ecall -> M-mode handler -> mret round-trip.
//!   --mode sv32 : Sv32 paging — M-mode builds two page tables in physical RAM, `mret`s into
//!                 S-mode, and exercises the software TLB's fill/hit paths, a write to a
//!                 not-yet-dirty PTE (A/D auto-update), SFENCE.VMA, a `satp` switch that remaps the
//!                 same VA to a different PA (TLB-flush-on-satp), and a deliberate page fault back
//!                 to M-mode. See `build_sv32`'s doc comment for the full scenario and why Spike is
//!                 run with `_svadu` in its ISA string for this mode.
//!   --mode a    : LR.W/SC.W (A-extension) success + mismatched-reservation failure sequences.
//!   --mode all  : runs every mode above in sequence.
//!
//!   fs-diff --seed N [--count K] [--insns M] [--mode full|c|priv|sv32|a|all] [--spike PATH] [--dump]
//!
//! Exit status is non-zero if any mismatch is found.

use std::process::{Command, ExitCode};

use fs_loader::build_diff_elf;
use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_riscv::{Cpu, Exit, SysExit};

const ENTRY: u32 = 0x8000_0000;
const SCRATCH_SIZE: u32 = 0x1000;
const SCRATCH_BASE_REG: u32 = 31; // x31 holds the scratch base and is never clobbered

/// Deterministic xorshift32 PRNG (no host entropy — decision #7).
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

// ---- 32-bit encoders ----
fn i_type(op: u32, f3: u32, rd: u32, rs1: u32, imm: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
fn r_type(op: u32, f3: u32, f7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
fn s_type(op: u32, f3: u32, rs1: u32, rs2: u32, imm: u32) -> u32 {
    ((imm >> 5) & 0x7f) << 25 | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1f) << 7) | op
}
fn b_type(f3: u32, rs1: u32, rs2: u32, imm: u32) -> u32 {
    ((imm >> 12) & 1) << 31
        | ((imm >> 5) & 0x3f) << 25
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm >> 1) & 0xf) << 8
        | ((imm >> 11) & 1) << 7
        | 0x63
}
fn j_type(rd: u32, imm: u32) -> u32 {
    ((imm >> 20) & 1) << 31
        | ((imm >> 1) & 0x3ff) << 21
        | ((imm >> 11) & 1) << 20
        | ((imm >> 12) & 0xff) << 12
        | (rd << 7)
        | 0x6f
}
fn lui(rd: u32, imm: u32) -> u32 {
    (imm & 0xffff_f000) | (rd << 7) | 0x37
}
/// AMO*.W with aq=rl=0 (address in rs1).
fn amo(funct5: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    (funct5 << 27) | (rs2 << 20) | (rs1 << 15) | (0b010 << 12) | (rd << 7) | 0x2f
}

/// A random ALU / M-extension instruction (rd in 1..=30, never x31).
fn gen_alu(rng: &mut Rng) -> u32 {
    let rd = 1 + rng.below(30);
    let rs1 = rng.below(31);
    let rs2 = rng.below(31);
    match rng.below(3) {
        0 => {
            let imm = rng.next() & 0xfff;
            let shamt = rng.below(32);
            match rng.below(9) {
                0 => i_type(0x13, 0, rd, rs1, imm),
                1 => i_type(0x13, 2, rd, rs1, imm),
                2 => i_type(0x13, 3, rd, rs1, imm),
                3 => i_type(0x13, 4, rd, rs1, imm),
                4 => i_type(0x13, 6, rd, rs1, imm),
                5 => i_type(0x13, 7, rd, rs1, imm),
                6 => i_type(0x13, 1, rd, rs1, shamt),
                7 => i_type(0x13, 5, rd, rs1, shamt),
                _ => i_type(0x13, 5, rd, rs1, 0x400 | shamt),
            }
        }
        1 => match rng.below(10) {
            0 => r_type(0x33, 0, 0x00, rd, rs1, rs2),
            1 => r_type(0x33, 0, 0x20, rd, rs1, rs2),
            2 => r_type(0x33, 1, 0x00, rd, rs1, rs2),
            3 => r_type(0x33, 2, 0x00, rd, rs1, rs2),
            4 => r_type(0x33, 3, 0x00, rd, rs1, rs2),
            5 => r_type(0x33, 4, 0x00, rd, rs1, rs2),
            6 => r_type(0x33, 5, 0x00, rd, rs1, rs2),
            7 => r_type(0x33, 5, 0x20, rd, rs1, rs2),
            8 => r_type(0x33, 6, 0x00, rd, rs1, rs2),
            _ => r_type(0x33, 7, 0x00, rd, rs1, rs2),
        },
        _ => r_type(0x33, rng.below(8), 0x01, rd, rs1, rs2),
    }
}

/// One body instruction at word index `wi`; `last_wi` is the final body index (forward-only
/// control flow). Loads/stores use x31 (the scratch base) so they stay in bounds.
fn gen_body(rng: &mut Rng, wi: u32, last_wi: u32) -> u32 {
    let span = last_wi.saturating_sub(wi);
    match rng.below(80) {
        40..=55 => {
            let rd = 1 + rng.below(30);
            let rs2 = rng.below(31);
            match rng.below(11) {
                0 => i_type(0x03, 0, rd, SCRATCH_BASE_REG, rng.below(SCRATCH_SIZE)),
                1 => i_type(0x03, 4, rd, SCRATCH_BASE_REG, rng.below(SCRATCH_SIZE)),
                2 => i_type(0x03, 1, rd, SCRATCH_BASE_REG, rng.below(SCRATCH_SIZE / 2) * 2),
                3 => i_type(0x03, 5, rd, SCRATCH_BASE_REG, rng.below(SCRATCH_SIZE / 2) * 2),
                4 => i_type(0x03, 2, rd, SCRATCH_BASE_REG, rng.below(SCRATCH_SIZE / 4) * 4),
                5 => s_type(0x23, 0, SCRATCH_BASE_REG, rs2, rng.below(SCRATCH_SIZE)),
                6 => s_type(0x23, 1, SCRATCH_BASE_REG, rs2, rng.below(SCRATCH_SIZE / 2) * 2),
                7 => s_type(0x23, 2, SCRATCH_BASE_REG, rs2, rng.below(SCRATCH_SIZE / 4) * 4),
                // A-extension AMO*.W on the (aligned) scratch base x31.
                _ => {
                    let funct5 = [0x00, 0x01, 0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c][rng.below(9) as usize];
                    amo(funct5, rd, SCRATCH_BASE_REG, rs2)
                }
            }
        }
        56..=70 if span >= 1 => {
            let off = (1 + rng.below(span.min(3))) * 4;
            let f3 = [0, 1, 4, 5, 6, 7][rng.below(6) as usize];
            b_type(f3, rng.below(31), rng.below(31), off)
        }
        71..=76 if span >= 1 => j_type(1 + rng.below(30), (1 + rng.below(span.min(3))) * 4),
        _ => gen_alu(rng),
    }
}

// ---- compressed (RVC) encoders ----
fn c_li(rd: u32, imm6: u32) -> u16 {
    ((2 << 13) | ((imm6 >> 5) & 1) << 12 | (rd << 7) | (imm6 & 0x1f) << 2 | 0b01) as u16
}
fn c_addi(rd: u32, imm6: u32) -> u16 {
    (((imm6 >> 5) & 1) << 12 | (rd << 7) | (imm6 & 0x1f) << 2 | 0b01) as u16
}
fn c_slli(rd: u32, shamt: u32) -> u16 {
    (((shamt >> 5) & 1) << 12 | (rd << 7) | (shamt & 0x1f) << 2 | 0b10) as u16
}
fn c_mv(rd: u32, rs2: u32) -> u16 {
    ((4 << 13) | (rd << 7) | (rs2 << 2) | 0b10) as u16
}
fn c_add(rd: u32, rs2: u32) -> u16 {
    ((4 << 13) | (1 << 12) | (rd << 7) | (rs2 << 2) | 0b10) as u16
}
/// funct2=11 register op on x8..15: sel 0=sub,1=xor,2=or,3=and.
fn c_regop(rd3: u32, rs3: u32, sel: u32) -> u16 {
    ((4 << 13) | (0b11 << 10) | (rd3 << 7) | (sel << 5) | (rs3 << 2) | 0b01) as u16
}
fn c_andi(rd3: u32, imm6: u32) -> u16 {
    ((4 << 13) | ((imm6 >> 5) & 1) << 12 | (0b10 << 10) | (rd3 << 7) | (imm6 & 0x1f) << 2 | 0b01) as u16
}
/// funct2 00=srli, 01=srai on x8..15.
fn c_shift(rd3: u32, shamt: u32, funct2: u32) -> u16 {
    ((4 << 13) | ((shamt >> 5) & 1) << 12 | (funct2 << 10) | (rd3 << 7) | (shamt & 0x1f) << 2 | 0b01) as u16
}

fn gen_compressed(rng: &mut Rng) -> u16 {
    let rd = 1 + rng.below(31);
    let rs2 = 1 + rng.below(31);
    let rd3 = rng.below(8);
    let rs3 = rng.below(8);
    let imm6 = rng.next() & 0x3f;
    let shamt = 1 + rng.below(31);
    match rng.below(11) {
        0 => c_li(rd, imm6),
        1 => c_addi(rd, imm6),
        2 => c_slli(rd, shamt),
        3 => c_mv(rd, rs2),
        4 => c_add(rd, rs2),
        5 => c_regop(rd3, rs3, 0), // c.sub
        6 => c_regop(rd3, rs3, 1), // c.xor
        7 => c_regop(rd3, rs3, 2), // c.or
        8 => c_regop(rd3, rs3, 3), // c.and
        9 => c_andi(rd3, imm6),
        _ => c_shift(rd3, shamt, rng.below(2)), // c.srli / c.srai
    }
}

fn push_u32(b: &mut Vec<u8>, w: u32) {
    b.extend_from_slice(&w.to_le_bytes());
}

fn to_bytes(words: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(words.len() * 4);
    for w in words {
        push_u32(&mut b, *w);
    }
    b
}

/// x1..x30 = 0, x31 = scratch_base (32-bit prologue; 31 words).
fn prologue(scratch_base: u32) -> Vec<u8> {
    let mut b = Vec::new();
    for r in 1..=30u32 {
        push_u32(&mut b, i_type(0x13, 0, r, 0, 0));
    }
    push_u32(&mut b, lui(SCRATCH_BASE_REG, scratch_base));
    b
}

/// HTIF exit (32-bit; 4 words): store 1 to tohost, then a self-loop.
fn exit_seq(b: &mut Vec<u8>, tohost: u32) {
    push_u32(b, lui(5, tohost));
    push_u32(b, i_type(0x13, 0, 6, 0, 1));
    push_u32(b, s_type(0x23, 2, 5, 6, 0));
    push_u32(b, 0x0000_006f);
}

/// Word-level twin of [`exit_seq`], for builders that assemble a `Vec<u32>` directly.
fn exit_seq_words(w: &mut Vec<u32>, tohost: u32) {
    w.push(lui(5, tohost));
    w.push(i_type(0x13, 0, 6, 0, 1));
    w.push(s_type(0x23, 2, 5, 6, 0));
    w.push(0x0000_006f);
}

const PROLOGUE_BYTES: u32 = 31 * 4;
const EXIT_BYTES: u32 = 4 * 4;

/// Compute the tohost + scratch layout for a body of `body_bytes` bytes.
fn layout(body_bytes: u32) -> (u32, u32, u32) {
    let code_len = PROLOGUE_BYTES + body_bytes + EXIT_BYTES;
    let tohost = (ENTRY + code_len + 0xfff) & !0xfff;
    let scratch_base = tohost + 0x1000;
    let seg_len = (scratch_base + SCRATCH_SIZE) - ENTRY;
    (tohost, scratch_base, seg_len)
}

/// Full-mode program: ALU/M/memory/branch/jal. Returns (code_bytes, tohost, seg_len).
fn build_full(rng: &mut Rng, n: u32) -> (Vec<u8>, u32, u32) {
    let (tohost, scratch_base, seg_len) = layout(n * 4);
    let mut code = prologue(scratch_base);
    let prologue_words = 31;
    let last_wi = prologue_words + n - 1;
    for j in 0..n {
        push_u32(&mut code, gen_body(rng, prologue_words + j, last_wi));
    }
    exit_seq(&mut code, tohost);
    (code, tohost, seg_len)
}

/// Compressed-mode program: straight-line RVC ALU ops (2 bytes each).
fn build_compressed(rng: &mut Rng, n: u32) -> (Vec<u8>, u32, u32) {
    let (tohost, scratch_base, seg_len) = layout(n * 2);
    let mut code = prologue(scratch_base);
    for _ in 0..n {
        code.extend_from_slice(&gen_compressed(rng).to_le_bytes());
    }
    exit_seq(&mut code, tohost);
    (code, tohost, seg_len)
}

fn csrrw(rd: u32, csr: u32, rs1: u32) -> u32 {
    (csr << 20) | (rs1 << 15) | (1 << 12) | (rd << 7) | 0x73
}
fn csrrs(rd: u32, csr: u32, rs1: u32) -> u32 {
    (csr << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x73
}
const MRET: u32 = 0x3020_0073;
/// SFENCE.VMA x0, x0 (flush the whole TLB; rs1=rs2=0 = "all addresses, all ASIDs").
const SFENCE_VMA: u32 = (0x09 << 25) | 0x73;

/// Split a 32-bit constant into (hi20, lo12-signed) the way a compiler's `li` pseudo-op would:
/// `lo` is the sign-extended low 12 bits (so a subsequent ADDI reproduces them exactly), and `hi`
/// is `val - lo` with the low 12 bits masked off (so LUI supplies the rest).
fn split32(val: u32) -> (u32, i32) {
    let lo = ((val as i32) << 20) >> 20;
    let hi = (val as i64 - lo as i64) as u32 & 0xffff_f000;
    (hi, lo)
}

/// Load a 32-bit immediate into `rd`, emitting the minimal LUI(+ADDI) pair. Used wherever the
/// value is known up front (addresses/PTE words computed as Rust constants).
fn li32(b: &mut Vec<u32>, rd: u32, val: u32) {
    let (hi, lo) = split32(val);
    if hi != 0 {
        b.push(lui(rd, hi));
        if lo != 0 {
            b.push(i_type(0x13, 0, rd, rd, (lo as u32) & 0xfff));
        }
    } else {
        b.push(i_type(0x13, 0, rd, 0, (lo as u32) & 0xfff));
    }
}

/// Like [`li32`] but always emits exactly LUI+ADDI (2 words) — for a forward reference whose
/// real value (a handler/body address that hasn't been laid out yet) is patched in later via
/// [`patch_li32_2`].
fn li32_2(b: &mut Vec<u32>, rd: u32, val: u32) {
    let (hi, lo) = split32(val);
    b.push(lui(rd, hi));
    b.push(i_type(0x13, 0, rd, rd, (lo as u32) & 0xfff));
}

/// Overwrite a [`li32_2`] site (recorded word index) with the now-known real value.
fn patch_li32_2(words: &mut [u32], idx: usize, rd: u32, val: u32) {
    let (hi, lo) = split32(val);
    words[idx] = lui(rd, hi);
    words[idx + 1] = i_type(0x13, 0, rd, rd, (lo as u32) & 0xfff);
}

/// A fixed CSR + M-mode trap round-trip program: set mtvec, `ecall`, a handler that reads
/// mcause/mepc and returns via `mret`. Validates the privileged machinery against Spike.
/// Expected final regs: a0=42, a1=7, s0=11 (EcallM), s1=mepc+4.
fn build_priv() -> (Vec<u8>, u32, u32) {
    const N: u32 = 47;
    let tohost = (ENTRY + N * 4 + 0xfff) & !0xfff;
    let seg_len = (tohost + 0x1000) - ENTRY;
    let handler = ENTRY + 41 * 4; // word index 41

    let mut w: Vec<u32> = Vec::new();
    for r in 1..=31u32 {
        w.push(i_type(0x13, 0, r, 0, 0)); // zero x1..x31
    }
    w.push(lui(5, handler & 0xffff_f000)); // 31: lui t0, %hi(handler)
    w.push(i_type(0x13, 0, 5, 5, handler & 0xfff)); // 32: addi t0, t0, %lo
    w.push(csrrw(0, 0x305, 5)); // 33: csrw mtvec, t0
    w.push(i_type(0x13, 0, 10, 0, 0)); // 34: li a0, 0
    w.push(0x0000_0073); // 35: ecall   (mepc = ENTRY+35*4)
    w.push(i_type(0x13, 0, 11, 0, 7)); // 36: li a1, 7   (return point)
    w.push(lui(5, tohost)); // 37: lui t0, tohost
    w.push(i_type(0x13, 0, 6, 0, 1)); // 38: li t1, 1
    w.push(s_type(0x23, 2, 5, 6, 0)); // 39: sw t1, 0(t0)
    w.push(0x0000_006f); // 40: jal x0, 0
    w.push(csrrs(8, 0x342, 0)); // 41: csrr s0, mcause
    w.push(csrrs(9, 0x341, 0)); // 42: csrr s1, mepc
    w.push(i_type(0x13, 0, 9, 9, 4)); // 43: addi s1, s1, 4
    w.push(csrrw(0, 0x341, 9)); // 44: csrw mepc, s1
    w.push(i_type(0x13, 0, 10, 10, 42)); // 45: addi a0, a0, 42
    w.push(MRET); // 46: mret

    (to_bytes(&w), tohost, seg_len)
}

// ---- sv32 paging differential ----
//
// Physical layout (offsets from ENTRY, one 4 KiB page each): the whole 8-page window is a single
// RWX PT_LOAD segment (like every other mode), so both Spike and our `Mmu` start from identical
// zeroed memory and the M-mode setup code below is what actually builds the page tables — the
// only way to get two independent processes into byte-identical page-table state.
const SV32_CODE: u32 = 0x0000; // M-mode setup + S-mode body + M-mode fault handler + exit
const SV32_ROOT1: u32 = 0x1000; // address space 1's root table
const SV32_L01: u32 = 0x2000; // address space 1's level-0 table
const SV32_DATA_A: u32 = 0x3000; // data page mapped by address space 1
const SV32_TOHOST: u32 = 0x4000; // plain physical scratch word (M-mode only, never translated)
const SV32_ROOT2: u32 = 0x5000; // address space 2's root table (post satp-switch)
const SV32_L02: u32 = 0x6000; // address space 2's level-0 table
const SV32_DATA_B: u32 = 0x7000; // data page mapped by address space 2
const SV32_SEG_LEN: u32 = 0x8000;

// Leaf/non-leaf PTE bit positions.
const PTE_V: u32 = 1;
const PTE_R: u32 = 1 << 1;
const PTE_W: u32 = 1 << 2;
const PTE_X: u32 = 1 << 3;
const PTE_A: u32 = 1 << 6;

fn pte(pa: u32, flags: u32) -> u32 {
    ((pa >> 12) << 10) | flags
}

/// VPN[1] shared by every VA this scenario uses, so one level-0 table serves all of them:
/// `ENTRY >> 22`.
fn vpn1(va: u32) -> u32 {
    (va >> 22) & 0x3ff
}
fn vpn0(va: u32) -> u32 {
    (va >> 12) & 0x3ff
}

/// sv32 paging differential: in M-mode, build two independent Sv32 address spaces in physical
/// RAM (each a root + one level-0 table), `mret` into S-mode under the first, then exercise the
/// software TLB's fill path (first touch of a page), hit path (repeat touch), the "write to a
/// not-yet-dirty page" walk-and-refill path, an explicit SFENCE.VMA, and a `satp` write that
/// retargets the very same virtual address at a different physical page (the TLB-flush-on-satp
/// case) — then deliberately touches an unmapped VA to take a page fault back to M-mode, where a
/// handler reads back mcause/mtval/mepc plus the raw PTE bytes (proving A/D got set) and the
/// translated data pages' physical contents (proving the walk computed the right PA), before the
/// usual HTIF exit. `rng` only perturbs the two stored payload values and the in-page byte
/// offset touched, so the address-space/page-table skeleton is identical every run.
fn build_sv32(rng: &mut Rng) -> (Vec<u8>, u32, u32) {
    let code_pa = ENTRY + SV32_CODE;
    let root1_pa = ENTRY + SV32_ROOT1;
    let l01_pa = ENTRY + SV32_L01;
    let data_a_pa = ENTRY + SV32_DATA_A;
    let tohost_pa = ENTRY + SV32_TOHOST;
    let root2_pa = ENTRY + SV32_ROOT2;
    let l02_pa = ENTRY + SV32_L02;
    let data_b_pa = ENTRY + SV32_DATA_B;

    // VA_DATA shares CODE's 4 MiB region (same VPN[1]) but is *not* identity-mapped (VPN[0]=147),
    // so translation is a real remap, not a trivial identity fold. VA_FAULT's VPN[0] (16) is never
    // populated in either level-0 table, so it always page-faults.
    let va_code = ENTRY; // identity: VPN[0] = 0
    let va_data = ENTRY + 147 * 0x1000;
    let va_fault = ENTRY + 16 * 0x1000;
    assert_eq!(vpn1(va_code), vpn1(va_data));
    assert_eq!(vpn1(va_code), vpn1(va_fault));
    let byte_off = rng.below(4) * 4; // varies the exact word touched within the data page
    let val_a = rng.next() | 1; // never 0, so "still zero" vs "written" is unambiguous
    let val_b = rng.next() | 1;

    let root1 = pte(l01_pa, PTE_V);
    let l01_code = pte(code_pa, PTE_V | PTE_X); // A/D unset: fetch must set A
    let l01_data_a = pte(data_a_pa, PTE_V | PTE_R | PTE_W | PTE_A); // A set, D unset: write must set D
    let root2 = pte(l02_pa, PTE_V);
    let l02_code = pte(code_pa, PTE_V | PTE_X);
    let l02_data_b = pte(data_b_pa, PTE_V | PTE_R | PTE_W); // A and D both unset

    let satp1 = (1u32 << 31) | (root1_pa >> 12);
    let satp2 = (1u32 << 31) | (root2_pa >> 12);

    // Registers: x5..x7 scratch (t0..t2), x28..x30 more scratch (t3..t5), x31 untouched.
    // x8/x9 (s0/s1) + x18..x21 (s2..s5) hold the M-mode handler's physical read-back. x10..x17
    // (a0..a7) hold the S-mode body's translated results + the fault CSRs.
    const T0: u32 = 5;
    const T1: u32 = 6;
    const T2: u32 = 7;
    const T3: u32 = 28;

    let mut w: Vec<u32> = Vec::new();
    for r in 1..=31u32 {
        w.push(i_type(0x13, 0, r, 0, 0)); // zero x1..x31
    }

    // Build address space 1: L01[0]=CODE, L01[147]=DATA_A, ROOT1[vpn1]=L01.
    li32(&mut w, T0, l01_code);
    li32(&mut w, T1, l01_pa + vpn0(va_code) * 4);
    w.push(s_type(0x23, 2, T1, T0, 0));
    li32(&mut w, T0, l01_data_a);
    li32(&mut w, T1, l01_pa + vpn0(va_data) * 4);
    w.push(s_type(0x23, 2, T1, T0, 0));
    li32(&mut w, T0, root1);
    li32(&mut w, T1, root1_pa + vpn1(va_code) * 4);
    w.push(s_type(0x23, 2, T1, T0, 0));

    // Build address space 2: L02[0]=CODE, L02[147]=DATA_B, ROOT2[vpn1]=L02.
    li32(&mut w, T0, l02_code);
    li32(&mut w, T1, l02_pa + vpn0(va_code) * 4);
    w.push(s_type(0x23, 2, T1, T0, 0));
    li32(&mut w, T0, l02_data_b);
    li32(&mut w, T1, l02_pa + vpn0(va_data) * 4);
    w.push(s_type(0x23, 2, T1, T0, 0));
    li32(&mut w, T0, root2);
    li32(&mut w, T1, root2_pa + vpn1(va_code) * 4);
    w.push(s_type(0x23, 2, T1, T0, 0));

    // menvcfg.ADUE (bit 61; menvcfgh bit 29 on RV32) — see `isa_string`'s doc comment: without
    // this (and Spike's isa string carrying `_svadu`), Spike raises a page fault instead of
    // auto-updating A/D, unlike our own `Cpu::xlate` which always auto-updates.
    li32(&mut w, T0, 1u32 << 29);
    w.push(csrrw(0, 0x31a, T0)); // csrw menvcfgh, t0
    // satp <- address space 1; mstatus.MPP <- S; mtvec/mepc <- forward refs (patched below); mret.
    li32(&mut w, T0, satp1);
    w.push(csrrw(0, 0x180, T0)); // csrw satp, t0
    li32(&mut w, T0, 1 << 11); // MPP = 01 (S)
    w.push(csrrw(0, 0x300, T0)); // csrw mstatus, t0
    let mtvec_idx = w.len();
    li32_2(&mut w, T0, 0); // placeholder: %hi/%lo(handler)
    w.push(csrrw(0, 0x305, T0)); // csrw mtvec, t0
    let mepc_idx = w.len();
    li32_2(&mut w, T0, 0); // placeholder: %hi/%lo(s_body)
    w.push(csrrw(0, 0x341, T0)); // csrw mepc, t0
    w.push(MRET);

    // ---- S-mode body (address space 1 initially) ----
    let s_body_idx = w.len();
    li32(&mut w, T0, va_data + byte_off);
    w.push(i_type(0x03, 2, 10, T0, 0)); // a0 = lw [va_data]   (TLB fill; expect 0)
    w.push(i_type(0x03, 2, T2, T0, 0)); // (throwaway) lw again (TLB hit)
    li32(&mut w, T1, val_a);
    w.push(s_type(0x23, 2, T0, T1, 0)); // sw val_a, [va_data]  (write to not-yet-dirty PTE: sets D)
    w.push(i_type(0x03, 2, 11, T0, 0)); // a1 = lw [va_data]   (TLB hit; expect val_a)
    w.push(SFENCE_VMA);
    w.push(i_type(0x03, 2, 12, T0, 0)); // a2 = lw [va_data]   (forced refill post-flush; expect val_a)
    li32(&mut w, T2, satp2);
    w.push(csrrw(0, 0x180, T2)); // csrw satp, t2 -> address space 2 (flushes the TLB)
    w.push(i_type(0x03, 2, 13, T0, 0)); // a3 = lw [va_data]   (fresh page under AS2; expect 0)
    li32(&mut w, T3, val_b);
    w.push(s_type(0x23, 2, T0, T3, 0)); // sw val_b, [va_data]  (AS2: sets A+D)
    w.push(i_type(0x03, 2, 14, T0, 0)); // a4 = lw [va_data]   (expect val_b)
    li32(&mut w, T1, va_fault);
    w.push(i_type(0x03, 2, T2, T1, 0)); // lw (unmapped VPN[0]=16 under AS2) -> page fault -> traps to M

    // ---- M-mode fault handler (falls straight through into the exit sequence) ----
    let handler_idx = w.len();
    w.push(csrrs(15, 0x342, 0)); // a5 = mcause  (expect 13, E_LOAD_PAGE_FAULT)
    w.push(csrrs(16, 0x343, 0)); // a6 = mtval   (expect va_fault)
    w.push(csrrs(17, 0x341, 0)); // a7 = mepc    (expect the faulting lw's address)
    li32(&mut w, T0, data_a_pa);
    w.push(i_type(0x03, 2, 8, T0, 0)); // s0 = phys[data_a]  (expect val_a)
    li32(&mut w, T0, data_b_pa);
    w.push(i_type(0x03, 2, 9, T0, 0)); // s1 = phys[data_b]  (expect val_b)
    li32(&mut w, T0, l01_pa + vpn0(va_code) * 4);
    w.push(i_type(0x03, 2, T1, T0, 0));
    w.push(i_type(0x13, 7, 18, T1, PTE_A as i32 as u32 & 0xfff)); // s2 = L01[CODE] & A  (expect A)
    li32(&mut w, T0, l01_pa + vpn0(va_data) * 4);
    w.push(i_type(0x03, 2, T1, T0, 0));
    w.push(i_type(0x13, 7, 19, T1, 0xc0)); // s3 = L01[DATA_A] & (A|D)  (expect both)
    li32(&mut w, T0, l02_pa + vpn0(va_data) * 4);
    w.push(i_type(0x03, 2, T1, T0, 0));
    w.push(i_type(0x13, 7, 20, T1, 0xc0)); // s4 = L02[DATA_B] & (A|D)  (expect both)
    li32(&mut w, T0, l02_pa + vpn0(va_code) * 4);
    w.push(i_type(0x03, 2, T1, T0, 0));
    w.push(i_type(0x13, 7, 21, T1, PTE_A & 0xfff)); // s5 = L02[CODE] & A  (expect A: post-swap fetch)

    patch_li32_2(&mut w, mtvec_idx, T0, code_pa + handler_idx as u32 * 4);
    patch_li32_2(&mut w, mepc_idx, T0, code_pa + s_body_idx as u32 * 4);

    let tohost = tohost_pa;
    exit_seq_words(&mut w, tohost);

    assert!((w.len() as u32) * 4 < SV32_ROOT1, "sv32 code overran the page-table region");
    (to_bytes(&w), tohost, SV32_SEG_LEN)
}

// ---- LR/SC (A-extension) differential ----
//
// `n` groups of {compute address, LR.W, one safe ALU noise op, SC.W, read back}, alternating a
// same-address (success) and a mismatched-address (failure) reservation per group. x31 is the
// primary scratch address, x30 a fixed "other" address (scratch+half-page) for the mismatch case;
// noise never targets x30/x31 so both stay valid addresses throughout. Each group gets its own
// (LR dest, SC dest, read-back dest) register triple (x1..x29, wrapping if `n` > 9), so a bug
// anywhere shows up in the final register file, not just the last group.
fn gen_noise(rng: &mut Rng) -> u32 {
    let rd = 1 + rng.below(29); // never x30/x31 — this mode's fixed address registers
    let rs1 = rng.below(31);
    let rs2 = rng.below(31);
    match rng.below(3) {
        0 => i_type(0x13, 0, rd, rs1, rng.next() & 0xfff), // addi
        1 => r_type(0x33, 0, 0x00, rd, rs1, rs2),          // add
        _ => r_type(0x33, 4, 0x00, rd, rs1, rs2),          // xor
    }
}

fn build_a(rng: &mut Rng, n: u32) -> (Vec<u8>, u32, u32) {
    let n = n.clamp(1, 9); // 9 groups * 3 regs = 27, fits in x1..x29
    let body_words = 1 + n * 4; // x30 setup, then (lr + noise + sc + lw) per group
    let (tohost, scratch_base, seg_len) = layout(body_words * 4);

    let mut code = prologue(scratch_base);
    // x30 = scratch_base + SCRATCH_SIZE/2 : a fixed, always-valid, always-mismatched address.
    push_u32(&mut code, i_type(0x13, 0, 30, SCRATCH_BASE_REG, SCRATCH_SIZE / 2));

    for i in 0..n {
        let (rd1, rd2, rd3) = (1 + 3 * i, 2 + 3 * i, 3 + 3 * i);
        let success = rng.below(2) == 0;
        let addr_reg = if success { SCRATCH_BASE_REG } else { 30 };
        push_u32(&mut code, amo(0x02, rd1, SCRATCH_BASE_REG, 0)); // LR.W rd1, (x31) — always reserves here
        push_u32(&mut code, gen_noise(rng));
        push_u32(&mut code, amo(0x03, rd2, addr_reg, rd1)); // SC.W rd2, (addr_reg), rd1
        push_u32(&mut code, i_type(0x03, 2, rd3, SCRATCH_BASE_REG, 0)); // rd3 = lw [x31] (post-SC memory)
    }
    exit_seq(&mut code, tohost);
    (code, tohost, seg_len)
}

/// Run a program in fuzzsoft full-system mode (traps vectored via step_system).
fn run_ours_system(code: &[u8], tohost: u32, seg_len: u32) -> [u32; 32] {
    let mut mmu = Mmu::new(ENTRY, seg_len as usize);
    mmu.protect(ENTRY, seg_len, PERM_READ | PERM_WRITE).unwrap();
    mmu.map(ENTRY, code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    let mut cpu = Cpu::new(ENTRY);
    cpu.htif_tohost = Some(tohost);
    for _ in 0..1_000_000u64 {
        if let SysExit::Halt(_) = cpu.step_system(&mut mmu) {
            break;
        }
    }
    cpu.regs
}

/// Run the program in fuzzsoft; return the final register file.
fn run_ours(code: &[u8], tohost: u32, seg_len: u32) -> [u32; 32] {
    let mut mmu = Mmu::new(ENTRY, seg_len as usize);
    mmu.protect(ENTRY, seg_len, PERM_READ | PERM_WRITE).unwrap();
    mmu.map(ENTRY, code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    let mut cpu = Cpu::new(ENTRY);
    cpu.htif_tohost = Some(tohost);
    for _ in 0..50_000_000u64 {
        match cpu.step(&mut mmu).expect("our core trapped") {
            Exit::Continue => {}
            _ => break,
        }
    }
    cpu.regs
}

/// `sv32` sets `menvcfg.ADUE` so Spike auto-updates PTE A/D bits instead of raising a page fault
/// (Spike's default, matching real hardware that implements Svade rather than Svadu — see
/// `riscv/mmu.cc`'s `walk()`: `hade` is gated on `MENVCFG_ADUE`, which is only writable at all if
/// `Sv32`'s ISA string carries `_svadu`). Our own `Cpu::xlate` always auto-updates A/D
/// unconditionally (it doesn't model Svade), so this is the ISA configuration under which the two
/// engines' A/D behavior actually agrees.
fn isa_string(mode: &str) -> &'static str {
    if mode == "sv32" {
        "RV32IMAC_svadu"
    } else {
        "RV32IMAC"
    }
}

fn run_spike(spike: &str, elf_path: &str, isa: &str) -> Result<[u32; 32], String> {
    let out = Command::new(spike)
        .args(["-l", "--log-commits", &format!("--isa={isa}"), elf_path])
        .output()
        .map_err(|e| format!("failed to launch spike: {e}"))?;
    Ok(parse_spike_regs(&String::from_utf8_lossy(&out.stderr)))
}

/// Parse a Spike `-l --log-commits` trace, applying each `x<rd> 0x<val>` write in order.
fn parse_spike_regs(log: &str) -> [u32; 32] {
    let mut regs = [0u32; 32];
    for line in log.lines() {
        if let Some(pos) = line.find(" x") {
            let rest = &line[pos + 2..];
            let mut it = rest.split_whitespace();
            if let (Some(rd_s), Some(val_s)) = (it.next(), it.next())
                && let Ok(rd) = rd_s.parse::<usize>()
                && let Some(hex) = val_s.strip_prefix("0x")
                && rd < 32
                && let Ok(v) = u64::from_str_radix(hex, 16)
            {
                regs[rd] = v as u32;
            }
        }
    }
    regs
}

/// All individually-selectable modes, in the order `--mode all` runs them.
const ALL_MODES: &[&str] = &["full", "c", "priv", "sv32", "a"];

/// Run `count` cases of `mode` against `spike`; prints one line per case. Returns the number of
/// mismatches, or `Err` if Spike/the ELF writer itself failed (a harness error, not a divergence).
fn run_cases(mode: &str, seed: u32, count: u32, insns: u32, spike: &str, dump: bool, tmp: &str) -> Result<u32, ()> {
    let mut failures = 0u32;
    for k in 0..count {
        let mut rng = Rng(seed.wrapping_add(k).wrapping_mul(2654435761).max(1));
        let priv_mode = matches!(mode, "priv" | "p" | "sv32");
        let (code, tohost, seg_len) = match mode {
            "c" | "compressed" => build_compressed(&mut rng, insns),
            "priv" | "p" => build_priv(),
            "sv32" => build_sv32(&mut rng),
            "a" | "lrsc" => build_a(&mut rng, insns),
            _ => build_full(&mut rng, insns),
        };

        let mut seg = code.clone();
        seg.resize(seg_len as usize, 0);
        let elf = build_diff_elf(ENTRY, &seg, tohost);
        if let Err(e) = std::fs::write(tmp, &elf) {
            eprintln!("write elf: {e}");
            return Err(());
        }

        let ours = if priv_mode {
            run_ours_system(&code, tohost, seg_len)
        } else {
            run_ours(&code, tohost, seg_len)
        };
        let isa = isa_string(mode);
        let theirs = match run_spike(spike, tmp, isa) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                return Err(());
            }
        };

        if dump {
            let out = Command::new(spike)
                .args(["-l", "--log-commits", &format!("--isa={isa}"), tmp])
                .output()
                .unwrap();
            let s = String::from_utf8_lossy(&out.stderr);
            eprintln!("--- spike log (tail) ---");
            for l in s.lines().rev().take(12).collect::<Vec<_>>().iter().rev() {
                eprintln!("{l}");
            }
        }

        let mism: Vec<_> = (1..32)
            .filter(|&r| ours[r] != theirs[r])
            .map(|r| (r, ours[r], theirs[r]))
            .collect();
        if mism.is_empty() {
            println!("seed {seed} case {k} [{mode}]: OK ({insns} insns, 31 regs match)");
        } else {
            failures += 1;
            println!("seed {seed} case {k} [{mode}]: MISMATCH");
            for (r, o, t) in mism {
                println!("  x{r:<2}: ours={o:#010x} spike={t:#010x}");
            }
        }
    }
    Ok(failures)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 1u32;
    let mut count = 1u32;
    let mut insns = 64u32;
    let mut spike = "spike".to_string();
    let mut mode = "full".to_string();
    let mut dump = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(1);
            }
            "--count" => {
                i += 1;
                count = args[i].parse().unwrap_or(1);
            }
            "--insns" => {
                i += 1;
                insns = args[i].parse().unwrap_or(64);
            }
            "--spike" => {
                i += 1;
                spike = args[i].clone();
            }
            "--mode" => {
                i += 1;
                mode = args[i].clone();
            }
            "--dump" => dump = true,
            other => {
                eprintln!("unknown arg {other:?}");
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let tmp = std::env::temp_dir().join("fs-diff.elf");
    let tmp = tmp.to_str().unwrap();

    let single = [mode.as_str()];
    let modes: &[&str] = if mode == "all" { ALL_MODES } else { &single };
    let mut failures = 0u32;
    let mut total = 0u32;
    for &m in modes {
        match run_cases(m, seed, count, insns, &spike, dump, tmp) {
            Ok(f) => {
                failures += f;
                total += count;
            }
            Err(()) => return ExitCode::FAILURE,
        }
    }

    if failures == 0 {
        println!("all {total} case(s) matched Spike");
        ExitCode::SUCCESS
    } else {
        println!("{failures}/{total} case(s) diverged");
        ExitCode::FAILURE
    }
}
