//! M1 differential tester: generate random RV32IMAC programs, run them in fuzzsoft AND Spike, and
//! compare the final register state. Determinism is guaranteed by a prologue that zeroes x1..x30
//! (overriding Spike's boot-rom register state) and points x31 at a scratch page, so both machines
//! start from a known state and any divergence is a real bug in our core.
//!
//! Two modes:
//!   --mode full  (default): OP-IMM, OP, the full M extension, aligned loads/stores against the
//!                 scratch page (x31 base), and forward-only branches/jumps.
//!   --mode c    : straight-line compressed (RVC) ALU/register ops — validates the C decoder.
//!
//!   fs-diff --seed N [--count K] [--insns M] [--mode full|c] [--spike PATH] [--dump]
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

fn run_spike(spike: &str, elf_path: &str) -> Result<[u32; 32], String> {
    let out = Command::new(spike)
        .args(["-l", "--log-commits", "--isa=RV32IMAC", elf_path])
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
    let mut failures = 0u32;

    for k in 0..count {
        let mut rng = Rng(seed.wrapping_add(k).wrapping_mul(2654435761).max(1));
        let priv_mode = matches!(mode.as_str(), "priv" | "p");
        let (code, tohost, seg_len) = match mode.as_str() {
            "c" | "compressed" => build_compressed(&mut rng, insns),
            "priv" | "p" => build_priv(),
            _ => build_full(&mut rng, insns),
        };

        let mut seg = code.clone();
        seg.resize(seg_len as usize, 0);
        let elf = build_diff_elf(ENTRY, &seg, tohost);
        if let Err(e) = std::fs::write(tmp, &elf) {
            eprintln!("write elf: {e}");
            return ExitCode::FAILURE;
        }

        let ours = if priv_mode {
            run_ours_system(&code, tohost, seg_len)
        } else {
            run_ours(&code, tohost, seg_len)
        };
        let theirs = match run_spike(&spike, tmp) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };

        if dump {
            let out = Command::new(&spike)
                .args(["-l", "--log-commits", "--isa=RV32IMAC", tmp])
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

    if failures == 0 {
        println!("all {count} case(s) matched Spike");
        ExitCode::SUCCESS
    } else {
        println!("{failures}/{count} case(s) diverged");
        ExitCode::FAILURE
    }
}
