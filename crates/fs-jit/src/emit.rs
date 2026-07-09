//! Minimal x86-64 byte-level emitter for Phase 1 of `docs/jit-scalar-design.md`.
//!
//! Deliberately tiny: only the handful of instruction forms Phase 1 needs (see the design doc's
//! "x86-64 emitter scope for Phase 1" section), each hand-encoded against the Intel SDM's opcode
//! tables. No unsafe here — this module only pushes bytes into a `Vec<u8>`; the arena/mmap/call
//! machinery that turns those bytes into running code lives in `sys.rs`, the crate's one isolated
//! unsafe surface.
//!
//! Every memory operand emitted by this module addresses `[base + disp32]` with `base` either
//! `RDI` (the cpu pointer, register index 7) or `R8` (the chain's entry pc, register index 8) —
//! both indices are `!= 4` (RSP) and the `(mod=10, rm=101)` special case only applies when
//! `mod=00`, so no SIB byte is ever needed and disp32 is always emitted verbatim (mod=0b10).

#![forbid(unsafe_code)]

/// A general-purpose x86-64 register, by its 4-bit encoding (0..=15). Only the ones Phase 1's
/// codegen actually uses are named; `raw()` exposes the underlying index for the rare case a
/// caller needs it (none currently do outside this module).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Reg(pub u8);

impl Reg {
    pub const RAX: Reg = Reg(0);
    pub const RCX: Reg = Reg(1);
    pub const RDX: Reg = Reg(2);
    pub const RSI: Reg = Reg(6);
    pub const RDI: Reg = Reg(7);
    pub const R8: Reg = Reg(8);
    pub const R9: Reg = Reg(9);
    pub const R10: Reg = Reg(10);
    pub const R11: Reg = Reg(11);

    #[inline]
    fn low3(self) -> u8 {
        self.0 & 7
    }
    #[inline]
    fn needs_ext(self) -> bool {
        self.0 >= 8
    }
}

/// One of the 8 "Group 1" arithmetic/compare ops sharing x86's regular opcode pattern (Intel SDM
/// vol 2, opcode maps for ADD/OR/ADC/SBB/AND/SUB/XOR/CMP): `base+1` is the `op r/m32, r32` opcode
/// byte, `base>>3` is the `/digit` used by the `81 /digit id` (`op r/m32, imm32`) form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alu2 {
    Add,
    Or,
    And,
    Sub,
    Xor,
    Cmp,
}

impl Alu2 {
    fn base(self) -> u8 {
        match self {
            Alu2::Add => 0x00,
            Alu2::Or => 0x08,
            Alu2::And => 0x20,
            Alu2::Sub => 0x28,
            Alu2::Xor => 0x30,
            Alu2::Cmp => 0x38,
        }
    }
    fn rm_r_opcode(self) -> u8 {
        self.base() + 1
    }
    fn imm_digit(self) -> u8 {
        self.base() >> 3
    }
}

/// A branch/set condition, shared by `Jcc` (Phase 2: the one-target forward "skip a rare block"
/// pattern — see `chain.rs`'s `emit_skip`; Phase 1 emitted none), `SETcc`, and `CMOVcc`. `code()`
/// is the low nibble of the `0F 8x`/`0F 9x`/`0F 4x` opcode. `Ns` (sign flag clear) is a Phase 2
/// addition: after `test rax,rax` on a JIT call-out's packed `u64` return, `SF` is exactly bit 63
/// (the `TAG_TRAP` bit, so `Ns` means "no trap") and `ZF` is exactly "the whole value is zero" —
/// see `sys.rs`'s tag doc and `chain.rs`'s Load/Store codegen for how both flags are consumed from
/// that single `test`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cc {
    E,
    Ne,
    L,
    Ge,
    B,
    Ae,
    Ns,
}

impl Cc {
    fn code(self) -> u8 {
        match self {
            Cc::E => 0x4,
            Cc::Ne => 0x5,
            Cc::B => 0x2,
            Cc::Ae => 0x3,
            Cc::L => 0xc,
            Cc::Ge => 0xd,
            Cc::Ns => 0x9,
        }
    }
}

/// A tiny append-only byte assembler.
#[derive(Default, Clone)]
pub struct Asm {
    pub buf: Vec<u8>,
}

#[inline]
fn rex_byte(w: bool, r: bool, b: bool) -> Option<u8> {
    // X (SIB index extension) is never needed — this emitter never uses a SIB byte.
    if !w && !r && !b { None } else { Some(0x40 | ((w as u8) << 3) | ((r as u8) << 2) | (b as u8)) }
}

#[inline]
fn modrm(m: u8, reg_field: u8, rm_field: u8) -> u8 {
    (m << 6) | ((reg_field & 7) << 3) | (rm_field & 7)
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn push_rex(&mut self, w: bool, r: bool, b: bool) {
        if let Some(rex) = rex_byte(w, r, b) {
            self.buf.push(rex);
        }
    }

    /// `mov r32, imm32` — opcode `B8+r id` (register embedded in the opcode byte; no ModRM).
    pub fn mov_r32_imm32(&mut self, dst: Reg, imm: u32) {
        self.push_rex(false, false, dst.needs_ext());
        self.buf.push(0xB8 + dst.low3());
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    /// `mov r32, [base+disp32]` — opcode `8B /r`, mod=10 (disp32), no SIB (base is never RSP/R12).
    pub fn mov_r32_mem(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.push_rex(false, dst.needs_ext(), base.needs_ext());
        self.buf.push(0x8B);
        self.buf.push(modrm(0b10, dst.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov [base+disp32], r32` — opcode `89 /r`.
    pub fn mov_mem_r32(&mut self, base: Reg, disp: i32, src: Reg) {
        self.push_rex(false, src.needs_ext(), base.needs_ext());
        self.buf.push(0x89);
        self.buf.push(modrm(0b10, src.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `lea r32, [base+disp32]` — opcode `8D /r`.
    pub fn lea_r32_mem(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.push_rex(false, dst.needs_ext(), base.needs_ext());
        self.buf.push(0x8D);
        self.buf.push(modrm(0b10, dst.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `<op> dst, src` (both plain registers) — `op r/m32, r32` form, dst=r/m, src=reg.
    pub fn alu_r32_r32(&mut self, op: Alu2, dst: Reg, src: Reg) {
        self.push_rex(false, src.needs_ext(), dst.needs_ext());
        self.buf.push(op.rm_r_opcode());
        self.buf.push(modrm(0b11, src.low3(), dst.low3()));
    }

    /// `<op> dst, imm32` — `81 /digit id` form.
    pub fn alu_r32_imm32(&mut self, op: Alu2, dst: Reg, imm: u32) {
        self.push_rex(false, false, dst.needs_ext());
        self.buf.push(0x81);
        self.buf.push(modrm(0b11, op.imm_digit(), dst.low3()));
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    /// `shl/shr/sar dst, cl` — opcode `D3 /digit` (digit: shl=4, shr=5, sar=7).
    pub fn shl_cl(&mut self, dst: Reg) {
        self.shift_cl(4, dst);
    }
    pub fn shr_cl(&mut self, dst: Reg) {
        self.shift_cl(5, dst);
    }
    pub fn sar_cl(&mut self, dst: Reg) {
        self.shift_cl(7, dst);
    }
    fn shift_cl(&mut self, digit: u8, dst: Reg) {
        self.push_rex(false, false, dst.needs_ext());
        self.buf.push(0xD3);
        self.buf.push(modrm(0b11, digit, dst.low3()));
    }

    /// `shl/shr/sar dst, imm8` — opcode `C1 /digit ib`.
    pub fn shl_imm8(&mut self, dst: Reg, imm: u8) {
        self.shift_imm8(4, dst, imm);
    }
    pub fn shr_imm8(&mut self, dst: Reg, imm: u8) {
        self.shift_imm8(5, dst, imm);
    }
    pub fn sar_imm8(&mut self, dst: Reg, imm: u8) {
        self.shift_imm8(7, dst, imm);
    }
    fn shift_imm8(&mut self, digit: u8, dst: Reg, imm: u8) {
        self.push_rex(false, false, dst.needs_ext());
        self.buf.push(0xC1);
        self.buf.push(modrm(0b11, digit, dst.low3()));
        self.buf.push(imm);
    }

    /// `setcc dst8` — opcode `0F 90+cc /r` (reg field unused/0).
    pub fn setcc(&mut self, cc: Cc, dst: Reg) {
        // Any of our destination byte registers (r9b here) needs REX just to be addressed at all
        // as an 8-bit register via ModRM without the legacy AH/CH/DH/BH aliasing; `needs_ext`
        // additionally covers the r8-r15 case correctly since both conditions want a REX prefix.
        self.push_rex(false, false, dst.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0x90 + cc.code());
        self.buf.push(modrm(0b11, 0, dst.low3()));
    }

    /// `cmovcc dst32, src32` — opcode `0F 40+cc /r`, dst=reg, src=r/m.
    pub fn cmovcc(&mut self, cc: Cc, dst: Reg, src: Reg) {
        self.push_rex(false, dst.needs_ext(), src.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0x40 + cc.code());
        self.buf.push(modrm(0b11, dst.low3(), src.low3()));
    }

    /// `movzx dst32, src8` — opcode `0F B6 /r`, dst=reg, src=r/m.
    pub fn movzx_r32_r8(&mut self, dst: Reg, src: Reg) {
        self.push_rex(false, dst.needs_ext(), src.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0xB6);
        self.buf.push(modrm(0b11, dst.low3(), src.low3()));
    }

    /// `add qword [base+disp32], imm8` — REX.W + `83 /0 ib` (the `insns_retired` bump: a 64-bit
    /// field, so this is the one place REX.W is used).
    pub fn add_qword_mem_imm8(&mut self, base: Reg, disp: i32, imm: i8) {
        self.push_rex(true, false, base.needs_ext());
        self.buf.push(0x83);
        self.buf.push(modrm(0b10, 0 /* ADD */, base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
        self.buf.push(imm as u8);
    }

    /// `xor dst, dst` — zeroes `dst` (used both for materializing the constant 0 that an `x0`
    /// read becomes, and for the final `xor eax,eax` before `ret` that sets the `Continue` tag).
    pub fn zero(&mut self, dst: Reg) {
        self.alu_r32_r32(Alu2::Xor, dst, dst);
    }

    /// `ret` — opcode `C3`.
    pub fn ret(&mut self) {
        self.buf.push(0xC3);
    }

    // -------------------------------------------------------------------------------------------
    // Phase 2 additions (`docs/jit-scalar-design.md`): the handful of extra forms needed for the
    // Load/Store call-out (`push`/`pop` to protect the pinned entry-pc register `R8` — caller-saved
    // per SysV, and thus not guaranteed to survive a real `call` — across the shim call; `movabs`+
    // indirect `call` to reach the shim's fixed process-lifetime address; `test`+`Jcc` for the
    // packed-tag branch). See `chain.rs`'s module doc and `emit_skip` for how `Jcc` is used without
    // any general label table or backpatching machinery (exactly one forward target per emission,
    // whose length is measured by building it into a temporary buffer first).
    // -------------------------------------------------------------------------------------------

    /// `push r64` — opcode `50+rd` (no REX.W: push/pop already default to 64-bit operand size in
    /// long mode; only REX.B is ever needed, for r8-r15).
    pub fn push_r64(&mut self, r: Reg) {
        self.push_rex(false, false, r.needs_ext());
        self.buf.push(0x50 + r.low3());
    }

    /// `pop r64` — opcode `58+rd`.
    pub fn pop_r64(&mut self, r: Reg) {
        self.push_rex(false, false, r.needs_ext());
        self.buf.push(0x58 + r.low3());
    }

    /// `test a, b` (64-bit) — opcode `85 /r` with REX.W. Used as `test rax, rax` to read a JIT
    /// call-out's packed `u64` return into `SF`(=bit 63)/`ZF`(=is it all-zero) in one instruction.
    pub fn test_r64_r64(&mut self, a: Reg, b: Reg) {
        self.push_rex(true, b.needs_ext(), a.needs_ext());
        self.buf.push(0x85);
        self.buf.push(modrm(0b11, b.low3(), a.low3()));
    }

    /// `movabs dst, imm64` — opcode `B8+rd` with REX.W (the imm32 form's REX.W-set 64-bit-immediate
    /// sibling), for loading a shim function's absolute, process-lifetime-stable address (never a
    /// `rel32` direct call — the mmap'd arena can be arbitrarily far from it in the address space).
    pub fn mov_r64_imm64(&mut self, dst: Reg, imm: u64) {
        self.push_rex(true, false, dst.needs_ext());
        self.buf.push(0xB8 + dst.low3());
        self.buf.extend_from_slice(&imm.to_le_bytes());
    }

    /// `call r64` (indirect) — opcode `FF /2`, mod=11 reg=2(digit) rm=dst. No REX.W needed (call's
    /// operand size already defaults to 64-bit in long mode); REX.B if `dst` is r8-r15.
    pub fn call_r64(&mut self, dst: Reg) {
        self.push_rex(false, false, dst.needs_ext());
        self.buf.push(0xFF);
        self.buf.push(modrm(0b11, 2, dst.low3()));
    }

    /// `Jcc rel32` (near conditional jump) — opcode `0F 80+cc id`. No REX (no register operand).
    pub fn jcc_rel32(&mut self, cc: Cc, rel: i32) {
        self.buf.push(0x0F);
        self.buf.push(0x80 + cc.code());
        self.buf.extend_from_slice(&rel.to_le_bytes());
    }

    // -------------------------------------------------------------------------------------------
    // Phase 3 additions (`docs/jit-scalar-design.md`): the inlined memory fast path needs an
    // unconditional jump (the "skip the slow block entirely" arm of `chain.rs`'s `emit_if_else`)
    // and sized/signed loads straight from a host pointer (`fs-jit`'s Load fast path never calls
    // out at all on a hit, so the emitted code itself must do the correctly-sized, correctly-signed
    // load/store that `load_impl`/`store_impl` would otherwise have done).
    // -------------------------------------------------------------------------------------------

    /// `jmp rel32` (near unconditional jump) — opcode `E9 id`. No REX (no register operand).
    pub fn jmp_rel32(&mut self, rel: i32) {
        self.buf.push(0xE9);
        self.buf.extend_from_slice(&rel.to_le_bytes());
    }

    /// `movzx r32, byte [base+disp32]` — opcode `0F B6 /r`, mod=10 (disp32 memory form).
    pub fn movzx_r32_mem8(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.push_rex(false, dst.needs_ext(), base.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0xB6);
        self.buf.push(modrm(0b10, dst.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `movsx r32, byte [base+disp32]` — opcode `0F BE /r`.
    pub fn movsx_r32_mem8(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.push_rex(false, dst.needs_ext(), base.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0xBE);
        self.buf.push(modrm(0b10, dst.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `movzx r32, word [base+disp32]` — opcode `0F B7 /r`.
    pub fn movzx_r32_mem16(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.push_rex(false, dst.needs_ext(), base.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0xB7);
        self.buf.push(modrm(0b10, dst.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `movsx r32, word [base+disp32]` — opcode `0F BF /r`.
    pub fn movsx_r32_mem16(&mut self, dst: Reg, base: Reg, disp: i32) {
        self.push_rex(false, dst.needs_ext(), base.needs_ext());
        self.buf.push(0x0F);
        self.buf.push(0xBF);
        self.buf.push(modrm(0b10, dst.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov byte [base+disp32], src8` — opcode `88 /r`. `src`'s low byte is stored; any REX-needing
    /// source (`needs_ext()`, or a register whose byte form would otherwise alias
    /// AH/CH/DH/BH — see `push_rex`'s doc) gets a REX prefix so the correct byte register is
    /// addressed. Every call site in this crate uses `RCX`/`RAX` as `src` (never RSI/RDI/RBP/RSP,
    /// which would need a REX purely to avoid the legacy high-byte aliasing) — see `chain.rs`'s
    /// Store fast-path codegen.
    pub fn mov_mem8_r8(&mut self, base: Reg, disp: i32, src: Reg) {
        self.push_rex(false, src.needs_ext(), base.needs_ext());
        self.buf.push(0x88);
        self.buf.push(modrm(0b10, src.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov word [base+disp32], src16` — opcode `66 89 /r` (the operand-size override prefix
    /// selects the 16-bit form of the same `MOV r/m, r` opcode `mov_mem_r32` uses for 32-bit).
    pub fn mov_mem16_r16(&mut self, base: Reg, disp: i32, src: Reg) {
        self.buf.push(0x66); // operand-size override: 16-bit
        self.push_rex(false, src.needs_ext(), base.needs_ext());
        self.buf.push(0x89);
        self.buf.push(modrm(0b10, src.low3(), base.low3()));
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    /// Cross-check an encoded sequence against `ndisasm`'s reading of the same bytes: parse its
    /// `-b64` output back into a normalized (no addresses/bytes, just mnemonics) string and assert
    /// it's non-empty and contains no `db 0x` (ndisasm's marker for bytes it couldn't decode as a
    /// valid instruction) — i.e. every byte we emitted forms *some* valid x86-64 instruction
    /// stream, which is meaningful cross-validation for a hand-written encoder even without
    /// asserting the exact mnemonic text (kept as a `println!` for manual inspection).
    fn assert_disassembles_cleanly(bytes: &[u8]) -> String {
        let mut child = Command::new("ndisasm")
            .args(["-b64", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("ndisasm not found — install nasm's ndisasm for this cross-check");
        child.stdin.take().unwrap().write_all(bytes).unwrap();
        let out = child.wait_with_output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(!text.contains("db 0x"), "ndisasm could not decode some bytes:\n{text}");
        assert!(!text.trim().is_empty(), "ndisasm produced no output for {bytes:02x?}");
        text
    }

    /// Same cross-check idea as [`assert_disassembles_cleanly`], but via `objdump` instead of
    /// `ndisasm`. Needed for `FF /2` (indirect `CALL r/m64`): the `ndisasm` build available in this
    /// environment (NDISASM 3.01) mis-decodes that whole opcode group in 64-bit mode (confirmed:
    /// it fails identically on `call rax`/`jmp rax`, bytes `ff d0`/`ff e0`, which `objdump` reads
    /// correctly) — a real tool limitation here, not an encoding bug, so `call_r64`'s test uses
    /// this instead.
    fn assert_disassembles_via_objdump(bytes: &[u8]) -> String {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("fs-jit-emit-test-{:x}.bin", std::process::id()));
        std::fs::write(&tmp, bytes).unwrap();
        let out = Command::new("objdump")
            .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel"])
            .arg(&tmp)
            .output()
            .expect("objdump not found for this cross-check");
        let _ = std::fs::remove_file(&tmp);
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(!text.contains("(bad)"), "objdump could not decode some bytes:\n{text}");
        text
    }

    #[test]
    fn mov_r32_imm32_encoding() {
        let mut a = Asm::new();
        a.mov_r32_imm32(Reg::RAX, 0x1234_5678);
        // B8 + 0 (rax), imm32 LE.
        assert_eq!(a.buf, vec![0xB8, 0x78, 0x56, 0x34, 0x12]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov eax,0x12345678"), "{text}");
    }

    #[test]
    fn mov_r32_imm32_extended_reg_needs_rex() {
        let mut a = Asm::new();
        a.mov_r32_imm32(Reg::R9, 42);
        // REX.B (0x41) + B8+1 (r9's low3=1), imm32.
        assert_eq!(a.buf, vec![0x41, 0xB9, 42, 0, 0, 0]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov r9d,0x2a"), "{text}");
    }

    #[test]
    fn mov_r32_mem_load_from_rdi() {
        let mut a = Asm::new();
        a.mov_r32_mem(Reg::RAX, Reg::RDI, 0x10);
        // 8B /r, mod=10 reg=000(rax) rm=111(rdi) => ModRM 0x87, disp32=0x10.
        assert_eq!(a.buf, vec![0x8B, 0x87, 0x10, 0x00, 0x00, 0x00]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov eax,[rdi+0x10]"), "{text}");
    }

    #[test]
    fn mov_r32_mem_negative_disp() {
        let mut a = Asm::new();
        a.mov_r32_mem(Reg::RCX, Reg::RDI, -8);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov ecx,[rdi-0x8]"), "{text}");
    }

    #[test]
    fn mov_mem_r32_store_to_rdi() {
        let mut a = Asm::new();
        a.mov_mem_r32(Reg::RDI, 4, Reg::RCX);
        // 89 /r, mod=10 reg=001(rcx) rm=111(rdi) => ModRM 0x8F.
        assert_eq!(a.buf, vec![0x89, 0x8F, 0x04, 0x00, 0x00, 0x00]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov [rdi+0x4],ecx"), "{text}");
    }

    #[test]
    fn lea_from_r8_base_needs_rex_b() {
        let mut a = Asm::new();
        a.lea_r32_mem(Reg::RAX, Reg::R8, 0x100);
        // REX.B (0x41), 8D /r mod=10 reg=000(rax) rm=000(r8) => ModRM 0x80, disp32.
        assert_eq!(a.buf, vec![0x41, 0x8D, 0x80, 0x00, 0x01, 0x00, 0x00]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("lea eax,[r8+0x100]"), "{text}");
    }

    #[test]
    fn alu_r32_r32_all_ops() {
        let cases = [
            (Alu2::Add, 0x01),
            (Alu2::Or, 0x09),
            (Alu2::And, 0x21),
            (Alu2::Sub, 0x29),
            (Alu2::Xor, 0x31),
            (Alu2::Cmp, 0x39),
        ];
        for (op, opcode) in cases {
            let mut a = Asm::new();
            a.alu_r32_r32(op, Reg::RAX, Reg::RCX);
            // mod=11 reg=001(rcx,src) rm=000(rax,dst) => 0xC8.
            assert_eq!(a.buf, vec![opcode, 0xC8], "{op:?}");
            assert_disassembles_cleanly(&a.buf);
        }
    }

    #[test]
    fn alu_r32_imm32_all_ops() {
        let cases = [
            (Alu2::Add, 0),
            (Alu2::Or, 1),
            (Alu2::And, 4),
            (Alu2::Sub, 5),
            (Alu2::Xor, 6),
            (Alu2::Cmp, 7),
        ];
        for (op, digit) in cases {
            let mut a = Asm::new();
            a.alu_r32_imm32(op, Reg::RAX, 0xffff_fffe);
            assert_eq!(a.buf[0], 0x81);
            assert_eq!(a.buf[1], modrm(0b11, digit, 0));
            assert_eq!(&a.buf[2..6], &0xffff_fffeu32.to_le_bytes());
            assert_disassembles_cleanly(&a.buf);
        }
    }

    #[test]
    fn jalr_mask_is_and_eax_fffffffe() {
        let mut a = Asm::new();
        a.alu_r32_imm32(Alu2::And, Reg::RCX, 0xffff_fffe);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("and ecx,0xfffffffe"), "{text}");
    }

    #[test]
    fn shift_by_cl_all_ops() {
        let mut a = Asm::new();
        a.shl_cl(Reg::RAX);
        a.shr_cl(Reg::RCX);
        a.sar_cl(Reg::R9);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("shl eax,cl"), "{text}");
        assert!(text.contains("shr ecx,cl"), "{text}");
        assert!(text.contains("sar r9d,cl"), "{text}");
    }

    #[test]
    fn shift_by_imm8_shamt_0_and_31() {
        for shamt in [0u8, 1, 31] {
            let mut a = Asm::new();
            a.shl_imm8(Reg::RAX, shamt);
            let text = assert_disassembles_cleanly(&a.buf);
            assert!(text.contains(&format!("shl eax,{shamt:#x}")) || text.contains("shl eax,byte "), "{text}");
        }
    }

    #[test]
    fn setcc_and_movzx_roundtrip_all_conditions() {
        for cc in [Cc::E, Cc::Ne, Cc::L, Cc::Ge, Cc::B, Cc::Ae] {
            let mut a = Asm::new();
            a.setcc(cc, Reg::R9);
            a.movzx_r32_r8(Reg::RAX, Reg::R9);
            let text = assert_disassembles_cleanly(&a.buf);
            // ndisasm prints some of these condition codes under their "Z"/"NB" alias spelling
            // (setz==sete, setnb==setae) rather than the mnemonic we chose — same opcode either
            // way, so accept either spelling.
            let mnemonics: &[&str] = match cc {
                Cc::E => &["sete", "setz"],
                Cc::Ne => &["setne", "setnz"],
                Cc::L => &["setl"],
                Cc::Ge => &["setge", "setnl"],
                Cc::B => &["setb", "setc"],
                Cc::Ae => &["setae", "setnb", "setnc"],
                Cc::Ns => unreachable!("not exercised by this SETcc sweep"),
            };
            assert!(mnemonics.iter().any(|m| text.contains(m)), "{cc:?}: {text}");
            assert!(text.contains("movzx eax,r9b"), "{cc:?}: {text}");
        }
    }

    #[test]
    fn cmovcc_all_conditions() {
        for cc in [Cc::E, Cc::Ne, Cc::L, Cc::Ge, Cc::B, Cc::Ae] {
            let mut a = Asm::new();
            a.cmovcc(cc, Reg::R9, Reg::R10);
            let text = assert_disassembles_cleanly(&a.buf);
            let mnemonics: &[&str] = match cc {
                Cc::E => &["cmove", "cmovz"],
                Cc::Ne => &["cmovne", "cmovnz"],
                Cc::L => &["cmovl"],
                Cc::Ge => &["cmovge", "cmovnl"],
                Cc::B => &["cmovb", "cmovc"],
                Cc::Ae => &["cmovae", "cmovnb", "cmovnc"],
                Cc::Ns => unreachable!("not exercised by this CMOVcc sweep"),
            };
            assert!(mnemonics.iter().any(|m| text.contains(m)), "{cc:?}: {text}");
            assert!(text.contains("r9d,r10d"), "{cc:?}: {text}");
        }
    }

    #[test]
    fn add_qword_mem_imm8_insns_retired_bump() {
        let mut a = Asm::new();
        a.add_qword_mem_imm8(Reg::RDI, 0x20, 1);
        // REX.W(0x48), 83 /0 ib, mod=10 reg=000 rm=111(rdi) => 0x87, disp32, imm8.
        assert_eq!(a.buf, vec![0x48, 0x83, 0x87, 0x20, 0x00, 0x00, 0x00, 0x01]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("add qword [rdi+0x20],byte") || text.contains("add qword [rdi+0x20],0x1"), "{text}");
    }

    #[test]
    fn zero_is_xor_self() {
        let mut a = Asm::new();
        a.zero(Reg::RAX);
        assert_eq!(a.buf, vec![0x31, 0xC0]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("xor eax,eax"), "{text}");
    }

    #[test]
    fn ret_is_c3() {
        let mut a = Asm::new();
        a.ret();
        assert_eq!(a.buf, vec![0xC3]);
        assert_disassembles_cleanly(&a.buf);
    }

    /// A whole tiny hand-assembled sequence — `mov eax,[rdi+0]; add eax,5; mov [rdi+0],eax; ret`
    /// — disassembles as a clean, sensible instruction stream (a coarser but very legible
    /// end-to-end cross-check of the emitter beyond single-instruction unit tests).
    #[test]
    fn small_sequence_disassembles_as_expected() {
        let mut a = Asm::new();
        a.mov_r32_mem(Reg::RAX, Reg::RDI, 0);
        a.alu_r32_imm32(Alu2::Add, Reg::RAX, 5);
        a.mov_mem_r32(Reg::RDI, 0, Reg::RAX);
        a.ret();
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov eax,[rdi+0x0]"), "{text}");
        assert!(text.contains("add eax,0x5"), "{text}");
        assert!(text.contains("mov [rdi+0x0],eax"), "{text}");
        assert!(text.contains("ret"), "{text}");
    }

    // ---------------------------------------------------------------------------------------
    // Phase 2 additions.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn push_pop_r64_low_and_extended_regs() {
        let mut a = Asm::new();
        a.push_r64(Reg::RDI);
        a.pop_r64(Reg::RDI);
        assert_eq!(a.buf, vec![0x57, 0x5F]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("push rdi"), "{text}");
        assert!(text.contains("pop rdi"), "{text}");

        let mut a = Asm::new();
        a.push_r64(Reg::R8);
        a.pop_r64(Reg::R8);
        // REX.B(0x41) + 50+0(r8's low3) ; REX.B(0x41) + 58+0.
        assert_eq!(a.buf, vec![0x41, 0x50, 0x41, 0x58]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("push r8"), "{text}");
        assert!(text.contains("pop r8"), "{text}");
    }

    #[test]
    fn test_r64_r64_encoding() {
        let mut a = Asm::new();
        a.test_r64_r64(Reg::RAX, Reg::RAX);
        // REX.W(0x48), 85 /r, mod=11 reg=000(rax) rm=000(rax) => 0xC0.
        assert_eq!(a.buf, vec![0x48, 0x85, 0xC0]);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("test rax,rax"), "{text}");
    }

    #[test]
    fn mov_r64_imm64_movabs() {
        let mut a = Asm::new();
        a.mov_r64_imm64(Reg::R11, 0x1122_3344_5566_7788);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov r11,0x1122334455667788"), "{text}");
    }

    #[test]
    fn call_r64_indirect() {
        let mut a = Asm::new();
        a.call_r64(Reg::R11);
        // REX.B(0x41), FF /2, mod=11 reg=010(digit 2) rm=011(r11's low3) => 0xD3.
        assert_eq!(a.buf, vec![0x41, 0xFF, 0xD3]);
        let text = assert_disassembles_via_objdump(&a.buf);
        assert!(text.contains("call") && text.contains("r11"), "{text}");
    }

    #[test]
    fn jcc_rel32_sign_and_not_sign() {
        // `test rax,rax; jns +5; ret` (the 5 skips exactly one 1-byte `ret` plus... just checking
        // the jcc's own encoding + a plausible disassembly here; `chain.rs`'s `emit_skip` is what
        // exercises real skip-distance arithmetic end-to-end).
        let mut a = Asm::new();
        a.test_r64_r64(Reg::RAX, Reg::RAX);
        a.jcc_rel32(Cc::Ns, 1);
        a.ret();
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("jns"), "{text}");

        let mut a = Asm::new();
        a.test_r64_r64(Reg::RAX, Reg::RAX);
        a.jcc_rel32(Cc::E, 1);
        a.ret();
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("je") || text.contains("jz"), "{text}");
    }

    // ---------------------------------------------------------------------------------------
    // Phase 3 additions (`docs/jit-scalar-design.md`): inlined memory fast-path support.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn jmp_rel32_is_e9() {
        let mut a = Asm::new();
        a.jmp_rel32(1);
        a.ret();
        assert_eq!(a.buf[0], 0xE9);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("jmp"), "{text}");
    }

    #[test]
    fn movzx_and_movsx_mem8_load_from_pointer() {
        let mut a = Asm::new();
        a.movzx_r32_mem8(Reg::RCX, Reg::RAX, 0);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("movzx ecx,byte [rax") || text.contains("movzx ecx, byte [rax"), "{text}");

        let mut a = Asm::new();
        a.movsx_r32_mem8(Reg::RCX, Reg::RAX, 0);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("movsx ecx,byte [rax") || text.contains("movsx ecx, byte [rax"), "{text}");
    }

    #[test]
    fn movzx_and_movsx_mem16_load_from_pointer() {
        let mut a = Asm::new();
        a.movzx_r32_mem16(Reg::RCX, Reg::RAX, 0);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("movzx ecx,word [rax") || text.contains("movzx ecx, word [rax"), "{text}");

        let mut a = Asm::new();
        a.movsx_r32_mem16(Reg::RCX, Reg::RAX, 0);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("movsx ecx,word [rax") || text.contains("movsx ecx, word [rax"), "{text}");
    }

    #[test]
    fn mov_mem8_r8_and_mem16_r16_store_to_pointer() {
        let mut a = Asm::new();
        a.mov_mem8_r8(Reg::RAX, 0, Reg::RCX);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov [rax") && text.contains("cl"), "{text}");

        let mut a = Asm::new();
        a.mov_mem16_r16(Reg::RAX, 0, Reg::RCX);
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("mov [rax") && text.contains("cx"), "{text}");
    }

    /// A whole fast-path-shaped sequence: `movzx ecx, byte [rax]; ret` — plausible end-to-end
    /// disassembly beyond the single-instruction unit tests above.
    #[test]
    fn fast_load_sequence_disassembles_as_expected() {
        let mut a = Asm::new();
        a.movzx_r32_mem8(Reg::RCX, Reg::RAX, 0);
        a.ret();
        let text = assert_disassembles_cleanly(&a.buf);
        assert!(text.contains("movzx ecx,byte [rax") || text.contains("movzx ecx, byte [rax"), "{text}");
        assert!(text.contains("ret"), "{text}");
    }
}
