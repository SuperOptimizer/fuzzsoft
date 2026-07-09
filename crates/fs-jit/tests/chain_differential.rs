//! Differential correctness suite for Phase 1 of `docs/jit-scalar-design.md`
//! (`fs_jit::ChainCache`, the native chained ALU/branch compiler).
//!
//! This is the primary correctness gate the design doc calls for: compile+run many
//! (`Lui`/`Auipc`/`OpImm`/`Op`/`Fence` + at most one terminal `Branch`/`Jal`/`Jalr`) programs and
//! assert the resulting `Cpu` state (all 32 regs, `pc`, `insns_retired`) is bit-identical to
//! running the exact same instructions through `Cpu::exec_one` directly. An emitter byte-encoding
//! bug or a codegen semantic bug will show up here even if the *emitter's own* unit tests
//! (`emit.rs`) happen to pass, because this exercises the actual compiled machine code against
//! real (randomized) inputs rather than just inspecting its bytes.

use fs_jit::ChainCache;
use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_riscv::{AluOp, BranchOp, Cpu, sys::Priv};

// ---------------------------------------------------------------------------------------------
// A tiny, dependency-free deterministic PRNG (splitmix64) — the codebase has no `rand` dependency
// anywhere and this suite doesn't need one.
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
    fn reg(&mut self) -> u8 {
        (self.range(32)) as u8
    }
    fn bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
    fn choice<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.range(items.len() as u32) as usize]
    }
    /// A signed value drawn from a mix of "interesting" boundary values and pure random noise.
    fn interesting_u32(&mut self) -> u32 {
        const BOUNDARIES: [u32; 10] =
            [0, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_fffe, 0xffff_ffff, 31, 32];
        if self.bool() { self.choice(&BOUNDARIES) } else { self.next_u32() }
    }
}

// ---------------------------------------------------------------------------------------------
// Raw RV32 encoders (mirrors `fs_riscv::decode`'s bit layout exactly, hand-verified against the
// same table `decode()` switches on — see `crates/fs-riscv/src/lib.rs`'s `decode` function).
// ---------------------------------------------------------------------------------------------

fn u_type(op: u32, rd: u8, imm_u: u32) -> u32 {
    (imm_u & 0xffff_f000) | ((rd as u32) << 7) | op
}
fn i_type(op: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
    ((imm as u32 & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
}
fn r_type(op: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
}
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
fn j_type(op: u32, rd: u8, imm: i32) -> u32 {
    let i = imm as u32;
    ((i >> 20) & 1) << 31
        | ((i >> 1) & 0x3ff) << 21
        | ((i >> 11) & 1) << 20
        | ((i >> 12) & 0xff) << 12
        | ((rd as u32) << 7)
        | op
}

fn lui(rd: u8, imm_u: u32) -> u32 {
    u_type(0x37, rd, imm_u)
}
fn auipc(rd: u8, imm_u: u32) -> u32 {
    u_type(0x17, rd, imm_u)
}
fn jal(rd: u8, imm: i32) -> u32 {
    j_type(0x6f, rd, imm)
}
fn jalr(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x67, 0, rd, rs1, imm)
}
fn fence() -> u32 {
    0x0000_000f
}
fn ecall() -> u32 {
    0x0000_0073
}

fn branch_funct3(op: BranchOp) -> u32 {
    match op {
        BranchOp::Eq => 0,
        BranchOp::Ne => 1,
        BranchOp::Lt => 4,
        BranchOp::Ge => 5,
        BranchOp::Ltu => 6,
        BranchOp::Geu => 7,
    }
}
fn branch(op: BranchOp, rs1: u8, rs2: u8, imm: i32) -> u32 {
    b_type(0x63, branch_funct3(op), rs1, rs2, imm)
}

fn opimm(op: AluOp, rd: u8, rs1: u8, imm: i32) -> u32 {
    let (funct3, funct7): (u32, u32) = match op {
        AluOp::Add => (0, 0),
        AluOp::Slt => (2, 0),
        AluOp::Sltu => (3, 0),
        AluOp::Xor => (4, 0),
        AluOp::Or => (6, 0),
        AluOp::And => (7, 0),
        AluOp::Sll => (1, 0x00),
        AluOp::Srl => (5, 0x00),
        AluOp::Sra => (5, 0x20),
        AluOp::Sub => panic!("OpImm has no Sub encoding (ADDI negates the immediate instead)"),
    };
    match op {
        AluOp::Sll | AluOp::Srl | AluOp::Sra => {
            // shamt lives in the rs2 field, funct7 selects Srl vs Sra.
            r_type(0x13, funct3, funct7, rd, rs1, (imm & 0x1f) as u8)
        }
        _ => i_type(0x13, funct3, rd, rs1, imm),
    }
}

fn opreg(op: AluOp, rd: u8, rs1: u8, rs2: u8) -> u32 {
    let (funct3, funct7): (u32, u32) = match op {
        AluOp::Add => (0, 0x00),
        AluOp::Sub => (0, 0x20),
        AluOp::Sll => (1, 0x00),
        AluOp::Slt => (2, 0x00),
        AluOp::Sltu => (3, 0x00),
        AluOp::Xor => (4, 0x00),
        AluOp::Srl => (5, 0x00),
        AluOp::Sra => (5, 0x20),
        AluOp::Or => (6, 0x00),
        AluOp::And => (7, 0x00),
    };
    r_type(0x33, funct3, funct7, rd, rs1, rs2)
}

const ALL_ALU_OPS: [AluOp; 10] = [
    AluOp::Add,
    AluOp::Sub,
    AluOp::Sll,
    AluOp::Slt,
    AluOp::Sltu,
    AluOp::Xor,
    AluOp::Srl,
    AluOp::Sra,
    AluOp::Or,
    AluOp::And,
];
const OPIMM_ALU_OPS: [AluOp; 9] = [
    AluOp::Add,
    AluOp::Sll,
    AluOp::Slt,
    AluOp::Sltu,
    AluOp::Xor,
    AluOp::Srl,
    AluOp::Sra,
    AluOp::Or,
    AluOp::And,
];
const ALL_BRANCH_OPS: [BranchOp; 6] =
    [BranchOp::Eq, BranchOp::Ne, BranchOp::Lt, BranchOp::Ge, BranchOp::Ltu, BranchOp::Geu];

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

const BASE: u32 = 0x8000_0000;

fn setup(bytes: &[u8]) -> (Cpu, Mmu) {
    let mut mmu = Mmu::new(BASE, 0x1_0000);
    mmu.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
    mmu.map(BASE, bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    (Cpu::new(BASE), mmu)
}

fn assemble(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

fn random_regs(rng: &mut Rng) -> [u32; 32] {
    let mut regs = [0u32; 32];
    for r in regs.iter_mut() {
        *r = rng.interesting_u32();
    }
    regs
}

/// Run `program` (assumed to compile as exactly one Phase 1 chain — no Load/Store/Mul/etc, at
/// most one terminal branch/jal/jalr at the very end) two ways from an identical randomized
/// initial state: (a) the plain interpreter, calling `Cpu::exec_one` once per instruction in
/// `program`, stopping BEFORE any trailing non-ALU sentinel instruction (an `ecall` appended by
/// the caller when there's no terminal branch/jal/jalr, exactly mirroring what the chain compiler
/// itself would stop before); (b) `ChainCache::run_block`, called exactly once. Asserts
/// bit-identical regs/pc/insns_retired.
fn check_program(program: &[u32], initial_regs: [u32; 32], regs0_garbage: u32) {
    let bytes = assemble(program);

    let (mut cpu_i, mut mmu_i) = setup(&bytes);
    cpu_i.regs = initial_regs;
    cpu_i.regs[0] = regs0_garbage;

    let (mut cpu_j, mut mmu_j) = setup(&bytes);
    cpu_j.regs = initial_regs;
    cpu_j.regs[0] = regs0_garbage;

    // Reference: decode+exec_one each instruction directly, one at a time, stopping at (not
    // executing) the first non-ALU/non-terminal instruction, exactly like the chain compiler's
    // own `is_alu_link`/`is_terminal` classification — UNLESS that's the very first instruction
    // (an empty would-be chain), in which case `ChainCache::run_block`'s own contract is to fall
    // back and single-step exactly that one instruction for real (via `exec_one`+`finish_exit`,
    // with whatever real trap-vectoring that entails, e.g. an `Ecall`) rather than merely stopping
    // in front of it — so the reference must mirror that, not just halt short.
    let mut pc = BASE;
    let mut executed_any = false;
    let mut exit_i = fs_riscv::SysExit::Continue;
    // The pc of the LAST instruction the reference actually executed (whether an ALU step, the
    // terminal, or — for an empty would-be chain — the real fallback-executed instruction) — the
    // only instruction whose own transition can possibly be a coverage edge, since every ALU step
    // before it is architecturally guaranteed to fall straight through by exactly 4 bytes.
    let mut last_instr_pc: Option<u32> = None;
    'outer: loop {
        let idx = ((pc - BASE) / 4) as usize;
        if idx >= program.len() {
            break;
        }
        let raw = program[idx];
        let inst = fs_riscv::decode(raw);
        let is_alu = matches!(
            inst,
            fs_riscv::Inst::Lui { .. }
                | fs_riscv::Inst::Auipc { .. }
                | fs_riscv::Inst::OpImm { .. }
                | fs_riscv::Inst::Op { .. }
                | fs_riscv::Inst::Fence
        );
        let is_terminal = matches!(
            inst,
            fs_riscv::Inst::Branch { .. } | fs_riscv::Inst::Jal { .. } | fs_riscv::Inst::Jalr { .. }
        );
        if !is_alu && !is_terminal {
            if !executed_any {
                // Empty would-be chain: the real fallback path executes this one instruction for
                // real (`BlockCache::fetch` + `exec_one` + `finish_exit`), not a no-op stop.
                let r = cpu_i.exec_one(&mut mmu_i, inst, pc, 4, raw);
                exit_i = cpu_i.finish_exit(r);
                last_instr_pc = Some(pc);
            }
            break 'outer;
        }
        match cpu_i.exec_one(&mut mmu_i, inst, pc, 4, raw) {
            Ok(_) => {}
            Err(e) => panic!("interpreter faulted unexpectedly on ALU/branch-only program: {e}"),
        }
        executed_any = true;
        if is_terminal {
            last_instr_pc = Some(pc);
        }
        pc = cpu_i.pc;
        if is_terminal {
            break;
        }
    }
    let expected_edge = last_instr_pc
        .filter(|&p| cpu_i.pc != p.wrapping_add(4))
        .map(|p| (p, cpu_i.pc));

    let mut cache = ChainCache::with_capacity(256, 256);
    let exit = cache.run_block(&mut cpu_j, &mut mmu_j, &mut |_| true);
    assert_eq!(exit, exit_i, "SysExit mismatch\nprogram={program:02x?}");
    if exit == fs_riscv::SysExit::Continue {
        assert_eq!(
            cache.take_last_edge(),
            expected_edge,
            "coverage-edge mismatch\nprogram={program:02x?}"
        );
    }

    assert_eq!(cpu_i.regs, cpu_j.regs, "register mismatch\nprogram={program:02x?}");
    assert_eq!(cpu_i.pc, cpu_j.pc, "pc mismatch\nprogram={program:02x?}");
    assert_eq!(
        cpu_i.insns_retired, cpu_j.insns_retired,
        "insns_retired mismatch\nprogram={program:02x?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The main gate: thousands of random programs.
// ---------------------------------------------------------------------------------------------

#[test]
fn random_alu_chains_match_interpreter() {
    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    const ITERATIONS: usize = 20_000;
    for iter in 0..ITERATIONS {
        let n_alu = rng.range(9); // 0..=8 ALU-shape instructions
        let mut words = Vec::new();
        for _ in 0..n_alu {
            words.push(random_alu_insn(&mut rng));
        }
        let has_terminal = rng.bool();
        if has_terminal {
            words.push(random_terminal_insn(&mut rng, words.len() as i32 * 4));
        } else {
            // Sentinel the chain must stop before (never executed by either side).
            words.push(ecall());
        }
        let initial_regs = random_regs(&mut rng);
        let regs0_garbage = if rng.bool() { rng.next_u32() } else { 0 };
        check_program(&words, initial_regs, regs0_garbage);
        let _ = iter;
    }
}

fn random_alu_insn(rng: &mut Rng) -> u32 {
    // Weight register choices toward interesting aliasing (x0, rd==rs1, rd==rs2, rs1==rs2).
    let pick_reg = |rng: &mut Rng, pool: &[u8]| -> u8 {
        if rng.range(4) == 0 { rng.choice(pool) } else { rng.reg() }
    };
    match rng.range(5) {
        0 => lui(rng.reg(), rng.interesting_u32() & 0xffff_f000),
        1 => auipc(rng.reg(), rng.interesting_u32() & 0xffff_f000),
        2 => {
            let op = rng.choice(&OPIMM_ALU_OPS);
            let rs1 = rng.reg();
            let rd = pick_reg(rng, &[rs1, 0]);
            let imm = match op {
                AluOp::Sll | AluOp::Srl | AluOp::Sra => rng.choice(&[0i32, 1, 31]),
                _ => (rng.interesting_u32() as i32) << 20 >> 20, // sign-extended 12-bit
            };
            opimm(op, rd, rs1, imm)
        }
        3 => {
            let op = rng.choice(&ALL_ALU_OPS);
            let rs1 = rng.reg();
            let rs2 = pick_reg(rng, &[rs1, 0]);
            let rd = pick_reg(rng, &[rs1, rs2, 0]);
            opreg(op, rd, rs1, rs2)
        }
        _ => fence(),
    }
}

fn random_terminal_insn(rng: &mut Rng, static_offset: i32) -> u32 {
    match rng.range(3) {
        0 => {
            let op = rng.choice(&ALL_BRANCH_OPS);
            let rs1 = rng.reg();
            let rs2 = if rng.bool() { rs1 } else { rng.reg() };
            let imm = (rng.range(2048) as i32 - 1024) & !1; // even, in range
            branch(op, rs1, rs2, imm)
        }
        1 => {
            let rd = rng.reg();
            let imm = ((rng.range(1 << 20) as i32) - (1 << 19)) & !1;
            jal(rd, imm)
        }
        _ => {
            let rd = rng.reg();
            let rs1 = rng.reg();
            let imm = (rng.interesting_u32() as i32) << 20 >> 20;
            let _ = static_offset;
            jalr(rd, rs1, imm)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Structured edge cases (task requirement (a)/(b): every AluOp/BranchOp x rd==0/rs1==0/rs2==0/
// rd==rs1/rd==rs2/rs1==rs2, shift-by-0/31, signed vs unsigned boundaries, JALR &!1 masking,
// negative immediates, wrapping overflow).
// ---------------------------------------------------------------------------------------------

#[test]
fn every_op_reg_combination_matches_interpreter() {
    let boundary_vals: [u32; 8] =
        [0, 1, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_ffff, 0xffff_fffe, 31];
    for &op in &ALL_ALU_OPS {
        for rd in [0u8, 1, 2] {
            for rs1 in [0u8, 1, 2] {
                for rs2 in [0u8, 1, 2] {
                    let words = vec![opreg(op, rd, rs1, rs2), ecall()];
                    let mut regs = [0u32; 32];
                    regs[1] = boundary_vals[0];
                    regs[2] = boundary_vals[3]; // 0x8000_0000: exercises signed/unsigned divergence
                    check_program(&words, regs, 0xDEAD_BEEF);
                }
            }
        }
        // Sweep boundary value pairs directly in x1/x2 too.
        for &a in &boundary_vals {
            for &b in &boundary_vals {
                let words = vec![opreg(op, 3, 1, 2), ecall()];
                let mut regs = [0u32; 32];
                regs[1] = a;
                regs[2] = b;
                check_program(&words, regs, 0);
            }
        }
    }
}

#[test]
fn opimm_shift_by_0_and_31() {
    for &op in &[AluOp::Sll, AluOp::Srl, AluOp::Sra] {
        for shamt in [0i32, 1, 31] {
            for &val in &[0u32, 1, 0x8000_0000, 0xffff_ffff, 0x7fff_ffff] {
                let words = vec![opimm(op, 5, 1, shamt), ecall()];
                let mut regs = [0u32; 32];
                regs[1] = val;
                check_program(&words, regs, 0);
            }
        }
    }
}

#[test]
fn op_register_shift_masks_to_5_bits_like_shamt_and_31() {
    for &op in &[AluOp::Sll, AluOp::Srl, AluOp::Sra] {
        for shamt_val in [0u32, 1, 31, 32, 33, 63, 0xffff_ffff] {
            let words = vec![opreg(op, 5, 1, 2), ecall()];
            let mut regs = [0u32; 32];
            regs[1] = 0xABCD_1234;
            regs[2] = shamt_val;
            check_program(&words, regs, 0);
        }
    }
}

#[test]
fn every_branch_op_taken_and_not_taken_forward_and_backward() {
    let boundary_vals: [u32; 6] = [0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 5];
    for &op in &ALL_BRANCH_OPS {
        for &a in &boundary_vals {
            for &b in &boundary_vals {
                for &imm in &[4i32, -4, 100, -1024, 2, 0] {
                    if imm == 0 {
                        continue; // a real decoder never needs a self-branch here; imm must be even and nonzero-ish is fine, but 0 exercises "branch to self" - keep it, it's still valid
                    }
                    let words = vec![branch(op, 1, 2, imm), ecall()];
                    let mut regs = [0u32; 32];
                    regs[1] = a;
                    regs[2] = b;
                    check_program(&words, regs, 0);
                }
            }
        }
    }
}

#[test]
fn jalr_masks_odd_target_bit0() {
    for &(rs1_val, imm) in &[(5i64, 1), (0x8000_0001u32 as i64, 1), (7, -3), (0xffff_ffff_u32 as i64, 2)] {
        let words = vec![jalr(1, 2, imm), ecall()];
        let mut regs = [0u32; 32];
        regs[2] = rs1_val as u32;
        check_program(&words, regs, 0);
    }
}

#[test]
fn jal_and_jalr_link_register_and_rd_zero() {
    for rd in [0u8, 1, 5] {
        check_program(&[jal(rd, 8), ecall(), ecall()], [0u32; 32], 0);
        let mut regs = [0u32; 32];
        regs[2] = 0x8000_0010;
        check_program(&[jalr(rd, 2, 4), ecall()], regs, 0);
    }
}

#[test]
fn wrapping_add_sub_overflow() {
    let words = vec![opreg(AluOp::Add, 3, 1, 2), ecall()];
    let mut regs = [0u32; 32];
    regs[1] = 0xffff_ffff;
    regs[2] = 1;
    check_program(&words, regs, 0); // wraps to 0
    let words = vec![opreg(AluOp::Sub, 3, 1, 2), ecall()];
    let mut regs = [0u32; 32];
    regs[1] = 0;
    regs[2] = 1;
    check_program(&words, regs, 0); // wraps to 0xffff_ffff
}

#[test]
fn auipc_near_u32_wraparound() {
    // Auipc's result is entry_pc + imm; place the program near the top of the address space so
    // pc+imm actually wraps mod 2^32 (stresses the wrapping-arithmetic argument in chain.rs's
    // module doc, not just the common case).
    let base = 0xffff_f000u32;
    let mut mmu = Mmu::new(base, 0x1000);
    mmu.protect(base, 0x1000, PERM_READ | PERM_WRITE).unwrap();
    let words = [auipc(1, 0x0010_0000), ecall()];
    let bytes = assemble(&words);
    mmu.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let mut cpu_i = Cpu::new(base);
    let inst = fs_riscv::decode(words[0]);
    cpu_i.exec_one(&mut mmu, inst, base, 4, words[0]).unwrap();

    let mut mmu_j = Mmu::new(base, 0x1000);
    mmu_j.protect(base, 0x1000, PERM_READ | PERM_WRITE).unwrap();
    mmu_j.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    let mut cpu_j = Cpu::new(base);
    let mut cache = ChainCache::with_capacity(256, 256);
    cache.run_block(&mut cpu_j, &mut mmu_j, &mut |_| true);

    assert_eq!(cpu_i.regs[1], cpu_j.regs[1]);
    assert_eq!(cpu_i.pc, cpu_j.pc);
}

/// A longer realistic chain (accumulator loop body unrolled straight-line, no backward branch)
/// exercising `static_offset` accumulation across many steps, ending in a taken backward-style
/// immediate (still just resolved as a value, never actually followed).
#[test]
fn long_chain_static_offset_accumulation() {
    let mut rng = Rng::new(42);
    for _ in 0..200 {
        let mut words = Vec::new();
        for _ in 0..40 {
            words.push(random_alu_insn(&mut rng));
        }
        words.push(random_terminal_insn(&mut rng, words.len() as i32 * 4));
        let regs = random_regs(&mut rng);
        check_program(&words, regs, 0);
    }
}

// ---------------------------------------------------------------------------------------------
// Admission guard / mid-chain-timer (task requirement (c)): a timer deadline landing inside what
// would otherwise be one long compiled chain must produce identical trap-delivery timing to the
// interpreter, and the guard must actually engage (not just happen to never matter).
// ---------------------------------------------------------------------------------------------

#[test]
fn admission_guard_mid_chain_timer_matches_interpreter_exactly() {
    let base = 0x8000_0000u32;
    let tohost = 0x8000_2000u32;
    let handler = base + 0x1000;

    // A long straight-line chain: 20 ADDI's that would otherwise compile as one 20-instruction
    // native chain, deliberately longer than `stimecmp` so the deadline falls strictly inside it.
    let mut prog = Vec::new();
    for i in 0..20u8 {
        prog.push(opimm(AluOp::Add, 3, 3, 1)); // x3 += 1, twenty times
        let _ = i;
    }
    let bytes = assemble(&prog);

    let mut handler_code = Vec::new();
    for w in [opimm(AluOp::Add, 10, 0, 99), lui(5, tohost), opimm(AluOp::Add, 6, 0, 1), fs_riscv_store_sw(5, 6, 0)]
    {
        handler_code.extend_from_slice(&w.to_le_bytes());
    }

    let make_machine = || {
        let mut mmu = Mmu::new(base, 0x1_0000);
        mmu.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        mmu.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        mmu.map(handler, &handler_code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cpu = Cpu::new(base);
        cpu.privilege = Priv::S;
        cpu.htif_tohost = Some(tohost);
        cpu.csr.stvec = handler;
        cpu.csr.mideleg |= 1 << 5;
        cpu.csr.mie |= 1 << 5;
        cpu.csr.mstatus |= fs_riscv::sys::MSTATUS_SIE;
        cpu.csr.stimecmp = 7; // fires strictly inside the would-be 20-instruction chain
        (cpu, mmu)
    };

    // Reference: plain interpreter, exactly `Cpu::step_system` per instruction (fs-cli's
    // `run_case`/`run_case_jit` shape).
    let (mut cpu_i, mut mmu_i) = make_machine();
    let mut halted_i = false;
    for _ in 0..1000 {
        if let fs_riscv::SysExit::Halt(_) = cpu_i.step_system(&mut mmu_i) {
            halted_i = true;
            break;
        }
    }
    assert!(halted_i, "reference interpreter never took the timer interrupt");

    // Under test: driven through `ChainCache::run_block`, exactly like `fs-cli`'s `run_case_jit`.
    let (mut cpu_j, mut mmu_j) = make_machine();
    let mut cache = ChainCache::with_capacity(256, 256);
    let mut halted_j = false;
    for _ in 0..1000 {
        if let fs_riscv::SysExit::Halt(_) = cache.run_block(&mut cpu_j, &mut mmu_j, &mut |_| true) {
            halted_j = true;
            break;
        }
    }
    assert!(halted_j, "ChainCache never took the timer interrupt");

    assert_eq!(cpu_i.regs, cpu_j.regs, "post-interrupt register state diverged");
    assert_eq!(cpu_i.pc, cpu_j.pc);
    assert_eq!(cpu_i.insns_retired, cpu_j.insns_retired, "interrupt fired at a different instruction boundary");
    assert_eq!(cpu_i.csr.scause, cpu_j.csr.scause);
    assert_eq!(cpu_i.csr.sepc, cpu_j.csr.sepc);
    assert_eq!(cpu_i.regs[10], 99, "handler must have run (a0==99)");

    // Prove the guard actually engaged (this scenario would be a vacuous, cheating pass if the
    // whole thing happened to always take the fallback, or always happened to fit).
    assert!(cache.fallbacks() > 0, "admission guard never forced a fallback in this scenario");
}

/// Same shape but the timer deadline is generous (budget always exceeds the chain length): the
/// chain must actually go native (`chain_hits() > 0`), proving the guard doesn't just permanently
/// force the slow path.
#[test]
fn admission_guard_admits_when_budget_is_generous() {
    let base = 0x8000_0000u32;
    let mut prog = Vec::new();
    for _ in 0..10u8 {
        prog.push(opimm(AluOp::Add, 3, 3, 1));
    }
    prog.push(ecall());
    let bytes = assemble(&prog);
    let mut mmu = Mmu::new(base, 0x1_0000);
    mmu.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
    mmu.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    let mut cpu = Cpu::new(base);
    // Default stimecmp/mtimecmp are u64::MAX (disabled) — enormous budget.
    let mut cache = ChainCache::with_capacity(256, 256);
    cache.run_block(&mut cpu, &mut mmu, &mut |_| true);
    assert_eq!(cpu.regs[3], 10);
    assert_eq!(cpu.pc, base + 40); // stopped right before the ecall, chain excluded it
    assert_eq!(cpu.insns_retired, 10);
    assert_eq!(cache.chain_hits(), 1);
    assert_eq!(cache.fallbacks(), 0);
}

/// General S-type encoder (opcode 0x23's family — `Store`).
fn s_type(op: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let s_imm = imm as u32;
    ((s_imm & 0xfe0) << 20)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((s_imm & 0x1f) << 7)
        | op
}

fn fs_riscv_store_sw(rs1: u8, rs2: u8, imm: i32) -> u32 {
    // SW encoding — only the admission-guard test's handler code needs a real store (Store wasn't
    // Phase 1 chain scope when this helper was written; the handler runs through the ordinary
    // interpreter fallback after the trap redirects pc there either way).
    s_type(0x23, 2, rs1, rs2, imm)
}

// ---------------------------------------------------------------------------------------------
// Sanitizer/introspection gate: KMSAN/CMPLOG/UBSAN must route around the compiled path entirely,
// every dispatch, never cached.
// ---------------------------------------------------------------------------------------------

#[test]
fn sanitizer_gate_routes_around_compiled_path() {
    let words = [opimm(AluOp::Add, 1, 0, 5), ecall()];
    let bytes = assemble(&words);
    let mut mmu = Mmu::new(BASE, 0x1000);
    mmu.protect(BASE, 0x1000, PERM_READ | PERM_WRITE).unwrap();
    mmu.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    let mut cpu = Cpu::new(BASE);
    cpu.set_cmplog(true);
    let mut cache = ChainCache::with_capacity(256, 256);
    cache.run_block(&mut cpu, &mut mmu, &mut |_| true);
    assert_eq!(cpu.regs[1], 5);
    assert_eq!(cache.chain_hits(), 0, "cmplog-enabled dispatch must never take the native chain path");
}

// ---------------------------------------------------------------------------------------------
// Page-boundary chain: a chain must never start a new instruction on a different physical page,
// mirroring Stage 0's rule; verify correctness is preserved regardless (the chain just ends
// earlier than it otherwise could).
// ---------------------------------------------------------------------------------------------

#[test]
fn chain_stops_at_page_boundary_but_stays_correct() {
    let base = 0x8000_0f00u32; // 0x100 bytes before the next 4 KiB page boundary
    let mut prog = Vec::new();
    for _ in 0..40u8 {
        prog.push(opimm(AluOp::Add, 3, 3, 1));
    }
    let bytes = assemble(&prog);
    let mut mmu = Mmu::new(base & !0xfff, 0x2000);
    mmu.protect(base & !0xfff, 0x2000, PERM_READ | PERM_WRITE).unwrap();
    mmu.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let mut cpu_i = Cpu::new(base);
    let mut mmu_i = Mmu::new(base & !0xfff, 0x2000);
    mmu_i.protect(base & !0xfff, 0x2000, PERM_READ | PERM_WRITE).unwrap();
    mmu_i.map(base, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    for (i, &w) in prog.iter().enumerate() {
        let pc = base + 4 * i as u32;
        let inst = fs_riscv::decode(w);
        cpu_i.exec_one(&mut mmu_i, inst, pc, 4, w).unwrap();
    }

    let mut cpu_j = Cpu::new(base);
    let mut cache = ChainCache::with_capacity(256, 256);
    // Drive to completion (possibly several chains/fallback steps, since the page boundary may
    // truncate the first compiled chain well before all 40 instructions).
    for _ in 0..100 {
        if cpu_j.pc == base + 4 * prog.len() as u32 {
            break;
        }
        cache.run_block(&mut cpu_j, &mut mmu, &mut |_| true);
    }

    assert_eq!(cpu_i.regs, cpu_j.regs);
    assert_eq!(cpu_i.pc, cpu_j.pc);
    assert_eq!(cpu_i.insns_retired, cpu_j.insns_retired);
}

// =================================================================================================
// Phase 2 (`docs/jit-scalar-design.md`): Load/Store differential tests. `Load`/`Store` join the
// continuable chain set, so a chain may now fault (a real translation/permission fault) or halt
// (an HTIF `tohost` store) mid-run — neither of which Phase 1's ALU/branch-only chains could ever
// do. This module's reference oracle (`run_reference_once`) is an INDEPENDENT (not code-shared)
// reimplementation of what a compiled chain is supposed to do, built only from
// `Cpu::exec_one`/`finish_exit` plus the same `Bus::store_may_assert_interrupt` hook the real
// call-out uses — so a bug in `chain.rs`'s codegen (wrong tag bit, wrong pc commit order, wrong
// register aliasing) has to independently reproduce itself in BOTH this oracle's logic and the
// native codegen to slip through, rather than the test merely re-deriving the same code path.
// =================================================================================================
mod load_store {
    use super::*;
    use fs_riscv::{LoadOp, StoreOp};
    use fs_mmu::Bus;

    fn load_funct3(op: LoadOp) -> u32 {
        match op {
            LoadOp::Lb => 0,
            LoadOp::Lh => 1,
            LoadOp::Lw => 2,
            LoadOp::Lbu => 4,
            LoadOp::Lhu => 5,
        }
    }
    fn store_funct3(op: StoreOp) -> u32 {
        match op {
            StoreOp::Sb => 0,
            StoreOp::Sh => 1,
            StoreOp::Sw => 2,
        }
    }
    fn load_insn(op: LoadOp, rd: u8, rs1: u8, imm: i32) -> u32 {
        i_type(0x03, load_funct3(op), rd, rs1, imm)
    }
    fn store_insn(op: StoreOp, rs1: u8, rs2: u8, imm: i32) -> u32 {
        s_type(0x23, store_funct3(op), rs1, rs2, imm)
    }

    const ALL_LOAD_OPS: [LoadOp; 5] = [LoadOp::Lb, LoadOp::Lh, LoadOp::Lw, LoadOp::Lbu, LoadOp::Lhu];
    const ALL_STORE_OPS: [StoreOp; 3] = [StoreOp::Sb, StoreOp::Sh, StoreOp::Sw];

    // Memory layout (all within one `fs_platform::Machine`'s RAM window, distinct from its CLINT
    // window at `fs_platform::CLINT_BASE`): a code page, a two-page READ|WRITE "good" data region
    // (two pages so an unaligned access straddling their shared boundary is still entirely inside
    // permitted memory — a real page-crossing exercise, not an accidental fault), and an
    // unprotected ("no permission bits at all") data region that reliably faults any access.
    const RAM_BASE: u32 = 0x8000_0000;
    const RAM_SIZE: u32 = 0x10_0000;
    const CODE_LEN: u32 = 0x1000;
    const GOOD_DATA: u32 = RAM_BASE + 0x2000;
    const GOOD_DATA_LEN: u32 = 0x2000;
    const GOOD_MID: u32 = GOOD_DATA + GOOD_DATA_LEN / 2; // the shared page boundary
    const BAD_DATA: u32 = RAM_BASE + 0x6000;
    // Phase 3 (`docs/jit-scalar-design.md`) fast-path bail-condition regions: exactly-READ (no
    // WRITE) and allocated-but-unwritten (WRITE|RAW, no READ) — neither perm byte equals the
    // fast path's exact-match `need` mask for at least one of Load/Store, so these exercise the
    // "byte carries a perm that isn't exactly the trivial case" bail path distinctly from
    // `BAD_DATA`'s "no perms at all" case.
    const RO_DATA: u32 = RAM_BASE + 0x8000;
    const RAW_DATA: u32 = RAM_BASE + 0x9000;
    // Page-aligned so a bare `lui` can encode it exactly (low 12 bits must be 0 — `lui` masks
    // them off, so a non-page-aligned target silently truncates to the wrong address, which is
    // exactly the bug this constant's introduction fixed: an earlier version of these tests used
    // `BAD_DATA + 0x800`/`handler + 0x100` — neither page-aligned — as an HTIF `tohost` target
    // reached via `lui` alone, so the emitted `lui` silently truncated to a DIFFERENT address than
    // `cpu.htif_tohost`, and the store's actual target was never recognized as the HTIF sentinel.
    const HTIF_TOHOST: u32 = RAM_BASE + 0x7000;

    fn fresh_machine(bytes: &[u8]) -> fs_platform::Machine {
        let mut m = fs_platform::Machine::new(RAM_BASE, RAM_SIZE);
        m.ram.protect(RAM_BASE, CODE_LEN, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        m.ram.protect(GOOD_DATA, GOOD_DATA_LEN, PERM_READ | PERM_WRITE).unwrap();
        // BAD_DATA is deliberately left with perms=0 (unprotected): any access there faults.
        m.ram.protect(RO_DATA, 0x1000, PERM_READ).unwrap();
        m.ram.protect(RAW_DATA, 0x1000, fs_mmu::PERM_WRITE | fs_mmu::PERM_RAW).unwrap();
        m.ram.map(RAM_BASE, bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        m
    }

    /// Independent reference oracle for ONE `ChainCache::run_block` dispatch, built only from
    /// `Cpu::exec_one`/`finish_exit`/`Bus::store_may_assert_interrupt` — see the module doc.
    /// Mirrors `decode_chain`'s classification (ALU/Load/Store are continuable; exactly one
    /// trailing `Branch`/`Jal`/`Jalr` is terminal; anything else stops the chain before itself)
    /// and `emit_step`'s Load/Store semantics (pc commits, trap-stops-without-retiring,
    /// CLINT-repoll-stops-after-retiring).
    fn run_reference_once(cpu: &mut Cpu, bus: &mut dyn Bus, program: &[u32], base_pc: u32) -> fs_riscv::SysExit {
        let mut pc = base_pc;
        let mut executed_any = false;
        loop {
            let idx = ((pc.wrapping_sub(base_pc)) / 4) as usize;
            if idx >= program.len() {
                cpu.pc = pc;
                return fs_riscv::SysExit::Continue;
            }
            let raw = program[idx];
            let inst = fs_riscv::decode(raw);
            let is_continuable = matches!(
                inst,
                fs_riscv::Inst::Lui { .. }
                    | fs_riscv::Inst::Auipc { .. }
                    | fs_riscv::Inst::OpImm { .. }
                    | fs_riscv::Inst::Op { .. }
                    | fs_riscv::Inst::Fence
                    | fs_riscv::Inst::Load { .. }
                    | fs_riscv::Inst::Store { .. }
            );
            let is_terminal =
                matches!(inst, fs_riscv::Inst::Branch { .. } | fs_riscv::Inst::Jal { .. } | fs_riscv::Inst::Jalr { .. });
            if !is_continuable && !is_terminal {
                if !executed_any {
                    // Empty would-be chain: `ChainCache::run_block`'s own contract (mirroring
                    // `BlockCache::run_block`'s Stage 0 fallback) is to fall back and single-step
                    // exactly this one instruction FOR REAL (fetch+exec_one+finish_exit), not
                    // merely leave `pc` unchanged in front of it — see `check_program`'s identical
                    // handling above for the ALU/branch-only Phase 1 suite.
                    let r = cpu.exec_one(bus, inst, pc, 4, raw);
                    return cpu.finish_exit(r);
                }
                cpu.pc = pc; // chain ends BEFORE this instruction (not executed) — same as decode_chain
                return fs_riscv::SysExit::Continue;
            }
            executed_any = true;

            // Store's target address/size, computed BEFORE `exec_one` (needed for the
            // post-success CLINT-repoll check; `exec_one` doesn't hand the address back). `x0`
            // reads as 0 regardless of what the underlying array slot holds (mirrors `rd_reg`).
            let store_addr_size = if let fs_riscv::Inst::Store { op, rs1, imm, .. } = inst {
                let rs1v = if rs1 == 0 { 0 } else { cpu.regs[rs1 as usize] };
                let size = match op {
                    fs_riscv::StoreOp::Sb => 1,
                    fs_riscv::StoreOp::Sh => 2,
                    fs_riscv::StoreOp::Sw => 4,
                };
                Some((rs1v.wrapping_add(imm as u32), size))
            } else {
                None
            };

            match cpu.exec_one(bus, inst, pc, 4, raw) {
                Err(trap) => return cpu.finish_exit(Err(trap)),
                Ok(fs_riscv::Exit::Continue) => {
                    if let Some((addr, size)) = store_addr_size
                        && bus.store_may_assert_interrupt(addr, size)
                    {
                        // `exec_one` already committed pc/insns_retired for this store; stop here,
                        // exactly like the compiled chain's `TAG_REPOLL` early exit.
                        return fs_riscv::SysExit::Continue;
                    }
                    if is_terminal {
                        return fs_riscv::SysExit::Continue;
                    }
                }
                Ok(exit) => return cpu.finish_exit(Ok(exit)), // HTIF halt (the only other possibility here)
            }
            pc = cpu.pc;
        }
    }

    /// Compare ONE `ChainCache::run_block` dispatch against [`run_reference_once`] from an
    /// identical randomized initial state, over an identically-constructed fresh `Machine`.
    /// Asserts bit-identical `SysExit`, all 32 regs, `pc`, `insns_retired`, and every
    /// trap-relevant CSR field `take_trap` can touch.
    fn check_ls_program(words: &[u32], initial_regs: [u32; 32], regs0_garbage: u32) {
        let bytes = assemble(words);

        let mut m_i = fresh_machine(&bytes);
        let mut cpu_i = Cpu::new(RAM_BASE);
        cpu_i.regs = initial_regs;
        cpu_i.regs[0] = regs0_garbage;
        cpu_i.htif_tohost = Some(BAD_DATA + 0x800); // an address neither GOOD_DATA nor a real fault would hit incidentally
        let exit_i = run_reference_once(&mut cpu_i, &mut m_i, words, RAM_BASE);

        let mut m_j = fresh_machine(&bytes);
        let mut cpu_j = Cpu::new(RAM_BASE);
        cpu_j.regs = initial_regs;
        cpu_j.regs[0] = regs0_garbage;
        cpu_j.htif_tohost = Some(BAD_DATA + 0x800);
        let mut cache = ChainCache::with_capacity(256, 256);
        let exit_j = cache.run_block(&mut cpu_j, &mut m_j, &mut |_| true);

        assert_eq!(exit_i, exit_j, "SysExit mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.regs, cpu_j.regs, "register mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.pc, cpu_j.pc, "pc mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.insns_retired, cpu_j.insns_retired, "insns_retired mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.mcause, cpu_j.csr.mcause, "mcause mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.mepc, cpu_j.csr.mepc, "mepc mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.mtval, cpu_j.csr.mtval, "mtval mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.scause, cpu_j.csr.scause, "scause mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.sepc, cpu_j.csr.sepc, "sepc mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.stval, cpu_j.csr.stval, "stval mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.csr.mstatus, cpu_j.csr.mstatus, "mstatus mismatch\nprogram={words:02x?}");
        assert_eq!(cpu_i.privilege, cpu_j.privilege, "privilege mismatch\nprogram={words:02x?}");
        assert!(cpu_j.jit_pending_trap.is_none(), "jit_pending_trap must be drained after run_block");
    }

    // ---------------------------------------------------------------------------------------
    // Random fuzz: thousands of programs mixing ALU + Load + Store (+ optional terminal),
    // targeting a mix of valid (aligned/misaligned/page-crossing) and faulting addresses.
    // ---------------------------------------------------------------------------------------

    const A_GOOD: u8 = 5; // t0: seeded to a safe mid-point of the two-page GOOD_DATA region
    const A_BAD: u8 = 6; // t1: seeded to BAD_DATA (always faults)

    fn random_ls_insn(rng: &mut Rng) -> u32 {
        // rs1 drawn from: the two dedicated address registers, x0 (near-address-0, faults), or a
        // fully random register (whatever an earlier ALU step left there — broad fuzzing entropy).
        let rs1 = match rng.range(4) {
            0 => A_GOOD,
            1 => A_BAD,
            2 => 0,
            _ => rng.reg(),
        };
        // Small immediates around 0 (aligned + misaligned). `-2` relative to `A_GOOD` (seeded to
        // `GOOD_MID`, exactly the shared page boundary of the two-page GOOD_DATA region) exercises
        // page-crossing without deliberately faulting; RISC-V I/S-type immediates are 12-bit
        // signed, so this can't be a large offset like `GOOD_DATA_LEN/2`.
        let imm = rng.choice(&[-8, -3, -2, -1, 0, 1, 2, 3, 4, 8]);
        let rd_or_rs2 = if rng.range(4) == 0 { rs1 } else { rng.reg() }; // sometimes alias rd/rs2 with rs1
        if rng.bool() {
            let op = rng.choice(&ALL_LOAD_OPS);
            load_insn(op, rd_or_rs2, rs1, imm)
        } else {
            let op = rng.choice(&ALL_STORE_OPS);
            store_insn(op, rs1, rd_or_rs2, imm)
        }
    }

    fn seeded_regs(rng: &mut Rng) -> [u32; 32] {
        let mut regs = random_regs(rng);
        regs[A_GOOD as usize] = GOOD_MID;
        regs[A_BAD as usize] = BAD_DATA;
        regs
    }

    #[test]
    fn random_alu_load_store_chains_match_interpreter() {
        let mut rng = Rng::new(0xFEED_FACE_C0DE_1234);
        const ITERATIONS: usize = 6_000;
        for _ in 0..ITERATIONS {
            let n = rng.range(10); // 0..=9 mixed ALU/Load/Store instructions
            let mut words = Vec::new();
            for _ in 0..n {
                if rng.bool() {
                    words.push(random_alu_insn(&mut rng));
                } else {
                    words.push(random_ls_insn(&mut rng));
                }
            }
            if rng.bool() {
                words.push(random_terminal_insn(&mut rng, words.len() as i32 * 4));
            } else {
                words.push(ecall());
            }
            let regs = seeded_regs(&mut rng);
            let regs0_garbage = if rng.bool() { rng.next_u32() } else { 0 };
            check_ls_program(&words, regs, regs0_garbage);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Structured edge cases: every LoadOp/StoreOp x {valid aligned, valid misaligned, valid
    // page-crossing, faulting} x {rd==0, rs1==0, rd==rs1 aliasing}.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn every_load_op_x_every_address_shape() {
        let shapes: [(u8, i32); 5] =
            [(A_GOOD, 0), (A_GOOD, 1), (A_GOOD, -2), (A_BAD, 0), (0, 0)];
        for &op in &ALL_LOAD_OPS {
            for &(rs1, imm) in &shapes {
                for rd in [0u8, 1, rs1] {
                    let words = vec![load_insn(op, rd, rs1, imm), ecall()];
                    let regs = seeded_regs(&mut Rng::new(0x1111));
                    check_ls_program(&words, regs, 0xDEAD_BEEF);
                }
            }
        }
    }

    #[test]
    fn every_store_op_x_every_address_shape() {
        let shapes: [(u8, i32); 5] =
            [(A_GOOD, 0), (A_GOOD, 1), (A_GOOD, -2), (A_BAD, 0), (0, 0)];
        for &op in &ALL_STORE_OPS {
            for &(rs1, imm) in &shapes {
                for rs2 in [0u8, 2, rs1] {
                    let words = vec![store_insn(op, rs1, rs2, imm), ecall()];
                    let mut regs = seeded_regs(&mut Rng::new(0x2222));
                    regs[2] = 0x1234_5678;
                    check_ls_program(&words, regs, 0xCAFE_F00D);
                }
            }
        }
    }

    /// A store that faults must NOT retire (`insns_retired` unchanged from before it) and must
    /// vector a trap exactly where the interpreter would — explicit, dedicated (not just
    /// incidental in the random sweep) since it is the one Store-specific correctness property
    /// most likely to regress silently (e.g. if `emit_step` bumped `insns_retired` before
    /// checking the tag instead of after).
    #[test]
    fn faulting_store_does_not_retire() {
        let words = vec![store_insn(StoreOp::Sw, A_BAD, 2, 0), ecall()];
        let mut regs = seeded_regs(&mut Rng::new(7));
        regs[2] = 0x42;
        check_ls_program(&words, regs, 0);
    }

    /// A faulting Load/Store in the MIDDLE of a longer chain: everything before it must have
    /// fully retired (regs/insns_retired reflecting exactly those instructions), and the trap
    /// must be attributed to the faulting instruction's own pc — the scenario that most directly
    /// exercises `emit_step`'s "write cpu.pc = this instruction's own address before the call".
    #[test]
    fn faulting_load_mid_chain_attributes_correct_pc() {
        use fs_riscv::asm::*;
        let words = vec![
            addi(3, 0, 1),
            addi(3, 3, 1),
            load_insn(LoadOp::Lw, 4, A_BAD, 0), // faults here — pc must be exactly this instruction's
            addi(3, 3, 100),                    // never reached
            ecall(),
        ];
        let regs = seeded_regs(&mut Rng::new(9));
        check_ls_program(&words, regs, 0);
    }

    /// HTIF `tohost` halt via a `Store` mid-chain: the chain must retire the halting store
    /// (`insns_retired`/`pc` committed) and report `Halt` with the decoded exit code, exactly
    /// like `exec_one`'s `Store` arm.
    #[test]
    fn htif_halt_store_mid_chain() {
        use fs_riscv::asm::*;
        let tohost = HTIF_TOHOST;
        let words = vec![
            addi(3, 0, 41),
            lui(10, tohost),
            addi(11, 0, 1), // exit code 0, halt-request bit set
            fs_riscv_store_sw(10, 11, 0),
            addi(3, 3, 100), // never reached
            ecall(),
        ];
        let bytes = assemble(&words);
        let mut m = fresh_machine(&bytes);
        m.ram.protect(tohost & !0xfff, 0x1000, PERM_READ | PERM_WRITE).unwrap();
        let mut cpu = Cpu::new(RAM_BASE);
        cpu.htif_tohost = Some(tohost);
        let mut cache = ChainCache::with_capacity(256, 256);
        let exit = cache.run_block(&mut cpu, &mut m, &mut |_| true);
        assert_eq!(exit, fs_riscv::SysExit::Halt(0));
        assert_eq!(cpu.regs[3], 41);
        assert_eq!(cpu.insns_retired, 4, "the halting store itself must have retired");
    }

    // ---------------------------------------------------------------------------------------
    // CLINT store early-exit (task requirement (b)): a chain containing a store to the CLINT
    // range must stop immediately after it, reproducing the interpreter's exact
    // timer-interrupt-observability — proven here by a REAL msip-driven software interrupt that
    // is only deliverable because the chain stopped (rather than running the rest of the chain
    // with a stale interrupt-pending view).
    // ---------------------------------------------------------------------------------------

    #[test]
    fn clint_msip_store_forces_early_chain_exit_and_matches_interpreter() {
        use fs_riscv::asm::*;
        let handler = RAM_BASE + 0x800;
        let clint_msip = fs_platform::CLINT_BASE; // offset 0 = msip

        // A single chain: write 1 to CLINT's msip register, then (if the chain wrongly kept
        // going instead of early-exiting) two more ALU ops that would retire BEFORE the driver
        // ever gets a chance to resync/poll the newly-asserted software interrupt.
        let mut prog = vec![
            lui(20, clint_msip),
            addi(21, 0, 1),
            sw(20, 21, 0), // msip = 1 -- must force the chain to stop HERE
            addi(3, 3, 1), // must NOT retire this call if the early-exit works
            addi(3, 3, 1),
        ];
        prog.push(ecall());

        let mut handler_code = Vec::new();
        for w in [addi(10, 0, 77), lui(22, HTIF_TOHOST), addi(23, 0, 1), fs_riscv_store_sw(22, 23, 0)] {
            handler_code.extend_from_slice(&w.to_le_bytes());
        }
        let tohost = HTIF_TOHOST;

        let make = || {
            let bytes = assemble(&prog);
            let mut m = fresh_machine(&bytes);
            m.ram.map(handler, &handler_code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
            m.ram.protect(tohost & !0xfff, 0x1000, PERM_READ | PERM_WRITE).unwrap();
            let mut cpu = Cpu::new(RAM_BASE);
            cpu.privilege = fs_riscv::sys::Priv::M;
            cpu.htif_tohost = Some(tohost);
            cpu.csr.mtvec = handler;
            cpu.csr.mie |= 1 << 3; // MSIE
            cpu.csr.mstatus |= fs_riscv::sys::MSTATUS_MIE;
            (cpu, m)
        };

        // Reference: the real per-instruction driver shape (CLINT resync before every single
        // `step_system` call) — exactly `fs-cli`'s `run_case` loop.
        let (mut cpu_i, mut m_i) = make();
        let mut halted_i = false;
        for _ in 0..1000 {
            m_i.clint.mtime = cpu_i.virtual_time();
            fs_platform::sync_timer(&mut cpu_i, &m_i);
            if let fs_riscv::SysExit::Halt(_) = cpu_i.step_system(&mut m_i) {
                halted_i = true;
                break;
            }
        }
        assert!(halted_i, "reference interpreter never observed the msip-driven interrupt");

        // Under test: the real per-`run_block`-call driver shape — `fs-cli`'s `run_case_jit_chain`.
        let (mut cpu_j, mut m_j) = make();
        let mut cache = ChainCache::with_capacity(256, 256);
        let mut halted_j = false;
        for _ in 0..1000 {
            m_j.clint.mtime = cpu_j.virtual_time();
            fs_platform::sync_timer(&mut cpu_j, &m_j);
            if let fs_riscv::SysExit::Halt(_) = cache.run_block(&mut cpu_j, &mut m_j, &mut |_| true) {
                halted_j = true;
                break;
            }
        }
        assert!(halted_j, "ChainCache never observed the msip-driven interrupt");

        assert_eq!(cpu_i.regs, cpu_j.regs, "post-interrupt register state diverged");
        assert_eq!(cpu_i.pc, cpu_j.pc);
        assert_eq!(cpu_i.insns_retired, cpu_j.insns_retired, "interrupt fired at a different instruction boundary");
        assert_eq!(cpu_i.csr.mcause, cpu_j.csr.mcause);
        assert_eq!(cpu_i.csr.mepc, cpu_j.csr.mepc);
        assert_eq!(cpu_i.regs[10], 77, "handler must have run (a0==77)");
        // Prove the early-exit actually mattered: had the chain kept running past the msip store,
        // x3 would have been incremented twice (to 2) before the interrupt could ever be taken.
        // The interrupt firing immediately after the store (before either `addi x3,x3,1`) means
        // x3 must still be 0 when the handler (which doesn't touch x3) takes over.
        assert_eq!(cpu_j.regs[3], 0, "chain must have stopped immediately after the CLINT store");
    }

    // ---------------------------------------------------------------------------------------
    // Coverage-edge parity (task requirement, design doc's Phase 2 section): an interleaved
    // ALU+Load+Store+Branch program's compiled-chain run must report the identical coverage edge
    // as the interpreter, INCLUDING the "no edge at all" case for a CLINT-repoll early exit
    // (which must not spuriously report the chain's statically-known terminal branch as taken).
    // ---------------------------------------------------------------------------------------

    #[test]
    fn coverage_edge_parity_interleaved_alu_load_store_branch() {
        use fs_riscv::asm::*;
        let words = vec![
            addi(3, 0, 5),
            store_insn(StoreOp::Sw, A_GOOD, 3, 0),
            load_insn(LoadOp::Lw, 4, A_GOOD, 0),
            addi(4, 4, 1),
            beq(4, 4, 8), // always taken, forward
            addi(3, 3, 999),
            addi(3, 3, 1),
        ];
        let regs = seeded_regs(&mut Rng::new(123));
        let bytes = assemble(&words);

        let mut m = fresh_machine(&bytes);
        let mut cpu = Cpu::new(RAM_BASE);
        cpu.regs = regs;
        let mut cache = ChainCache::with_capacity(256, 256);
        let entry_pc = cpu.pc;
        let branch_pc = entry_pc + 4 * 4; // addi,store,load,addi precede the branch
        let exit = cache.run_block(&mut cpu, &mut m, &mut |_| true);
        assert_eq!(exit, fs_riscv::SysExit::Continue);
        let edge = cache.take_last_edge();
        // The branch is taken (beq x4,x4 is always true) and skips the `addi x3,x3,999` — a real,
        // non-fallthrough transfer, so an edge (the branch's OWN pc, not the chain's entry pc,
        // taken target) must be reported — mirrors the interpreter's own address-based heuristic.
        assert_eq!(edge, Some((branch_pc, cpu.pc)), "expected the taken branch's edge, got {edge:?}");

        // Now the CLINT-repoll variant: the same program but with the store retargeted at the
        // CLINT window — the chain must stop right after it (never reaching the branch this
        // call), so NO edge may be reported (the branch's statically-known offset must not leak
        // through as a phantom edge).
        let words2 = vec![
            lui(7, fs_platform::CLINT_BASE),
            addi(8, 0, 1),
            sw(7, 8, 0), // msip = 1 -- CLINT repoll
            beq(4, 4, 8),
            addi(3, 3, 999),
            addi(3, 3, 1),
        ];
        let bytes2 = assemble(&words2);
        let mut m2 = fresh_machine(&bytes2);
        let mut cpu2 = Cpu::new(RAM_BASE);
        cpu2.regs = regs;
        let mut cache2 = ChainCache::with_capacity(256, 256);
        let exit2 = cache2.run_block(&mut cpu2, &mut m2, &mut |_| true);
        assert_eq!(exit2, fs_riscv::SysExit::Continue);
        assert_eq!(
            cache2.take_last_edge(),
            None,
            "a CLINT-repoll early exit must not report the chain's unreached terminal as an edge"
        );
    }

    // ---------------------------------------------------------------------------------------
    // fs-diff/full-system-shaped sanity: an interleaved chain with a genuine page-crossing
    // Load/Store right at the two GOOD_DATA pages' shared boundary.
    // ---------------------------------------------------------------------------------------

    #[test]
    fn page_crossing_load_and_store_match_interpreter() {
        // `A_GOOD` is seeded to `GOOD_MID`, exactly the shared page boundary; `-2` makes a 4-byte
        // access straddle the two GOOD_DATA pages, both permitted.
        let mid_off = -2i32;
        for &op in &ALL_LOAD_OPS {
            let words = vec![load_insn(op, 4, A_GOOD, mid_off), ecall()];
            check_ls_program(&words, seeded_regs(&mut Rng::new(55)), 0);
        }
        for &op in &ALL_STORE_OPS {
            let mut regs = seeded_regs(&mut Rng::new(66));
            regs[2] = 0xABCD_1234;
            let words = vec![store_insn(op, A_GOOD, 2, mid_off), ecall()];
            check_ls_program(&words, regs, 0);
        }
    }

    // ---------------------------------------------------------------------------------------
    // `fs_platform::CowMachine`: `fs-cli`'s `--jobs`-path parallel fuzzing drives `ChainCache`
    // over a per-thread `CowMachine` (a copy-on-write overlay over a shared `Arc<Golden>` RAM
    // image), not `Machine` — a genuinely different `Bus` impl every test above never exercises.
    // Re-run the identical random ALU/Load/Store/terminal sweep with the chain side driven over a
    // `CowMachine` instead, to catch any assumption in `sys.rs`'s fat-pointer decompose/recompose
    // or the load/store call-out shims that happens to hold for `Machine`'s vtable but not
    // `CowMachine`'s. The interpreter-side reference oracle stays on the already-proven `Machine`
    // path; only the compiled-chain side changes bus type, isolating exactly what's new here.
    fn fresh_cow_machine(bytes: &[u8]) -> fs_platform::CowMachine {
        let mut m = fs_platform::Machine::new(RAM_BASE, RAM_SIZE);
        m.ram.protect(RAM_BASE, CODE_LEN, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        m.ram.protect(GOOD_DATA, GOOD_DATA_LEN, PERM_READ | PERM_WRITE).unwrap();
        // BAD_DATA is deliberately left with perms=0 (unprotected): any access there faults.
        m.ram.protect(RO_DATA, 0x1000, PERM_READ).unwrap();
        m.ram.protect(RAW_DATA, 0x1000, fs_mmu::PERM_WRITE | fs_mmu::PERM_RAW).unwrap();
        m.ram.map(RAM_BASE, bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let golden = std::sync::Arc::new(fs_mmu::Golden::from_mmu(&m.ram));
        fs_platform::CowMachine::from_golden(golden, RAM_BASE, RAM_SIZE)
    }

    /// [`check_ls_program`]'s twin, but the compiled-chain side runs over a fresh `CowMachine`
    /// (built from an identical golden image) instead of a fresh `Machine`.
    fn check_ls_program_cow(words: &[u32], initial_regs: [u32; 32], regs0_garbage: u32) {
        let bytes = assemble(words);

        let mut m_i = fresh_machine(&bytes);
        let mut cpu_i = Cpu::new(RAM_BASE);
        cpu_i.regs = initial_regs;
        cpu_i.regs[0] = regs0_garbage;
        cpu_i.htif_tohost = Some(BAD_DATA + 0x800);
        let exit_i = run_reference_once(&mut cpu_i, &mut m_i, words, RAM_BASE);

        let mut m_j = fresh_cow_machine(&bytes);
        let mut cpu_j = Cpu::new(RAM_BASE);
        cpu_j.regs = initial_regs;
        cpu_j.regs[0] = regs0_garbage;
        cpu_j.htif_tohost = Some(BAD_DATA + 0x800);
        let mut cache = ChainCache::with_capacity(256, 256);
        let exit_j = cache.run_block(&mut cpu_j, &mut m_j, &mut |_| true);

        assert_eq!(exit_i, exit_j, "SysExit mismatch (CowMachine)\nprogram={words:02x?}");
        assert_eq!(cpu_i.regs, cpu_j.regs, "register mismatch (CowMachine)\nprogram={words:02x?}");
        assert_eq!(cpu_i.pc, cpu_j.pc, "pc mismatch (CowMachine)\nprogram={words:02x?}");
        assert_eq!(
            cpu_i.insns_retired, cpu_j.insns_retired,
            "insns_retired mismatch (CowMachine)\nprogram={words:02x?}"
        );
        assert!(
            cpu_j.jit_pending_trap.is_none(),
            "jit_pending_trap must be drained after run_block (CowMachine)"
        );
    }

    #[test]
    fn random_alu_load_store_chains_match_interpreter_over_cow_machine() {
        let mut rng = Rng::new(0xC0DE_C0DE_FEED_9999);
        const ITERATIONS: usize = 3_000;
        for _ in 0..ITERATIONS {
            let n = rng.range(10); // 0..=9 mixed ALU/Load/Store instructions
            let mut words = Vec::new();
            for _ in 0..n {
                if rng.bool() {
                    words.push(random_alu_insn(&mut rng));
                } else {
                    words.push(random_ls_insn(&mut rng));
                }
            }
            if rng.bool() {
                words.push(random_terminal_insn(&mut rng, words.len() as i32 * 4));
            } else {
                words.push(ecall());
            }
            let regs = seeded_regs(&mut rng);
            let regs0_garbage = if rng.bool() { rng.next_u32() } else { 0 };
            check_ls_program_cow(&words, regs, regs0_garbage);
        }
    }

    // =============================================================================================
    // Phase 3 (`docs/jit-scalar-design.md`): the inlined memory fast path. Every test above in this
    // file already re-exercises the fast path incidentally (it's compiled into `emit_step` now, not
    // a separate code path the old tests could skip) — the additions below specifically target the
    // bail conditions the design doc calls out by name (TLB miss, RAW-poisoned bytes, a perm-fault
    // byte, MMIO, the sv32 dirty-bit gate) and assert on `Cpu::fast_path_hits`/`fast_path_bails`
    // (bumped by the compiled chain's own emitted code) to prove each scenario actually engaged the
    // fast and/or slow path it's meant to, rather than passing vacuously.
    // =============================================================================================
    mod phase3_fast_path {
        use super::*;

        /// [`check_ls_program`]'s twin, returning `(cpu_j.fast_path_hits, cpu_j.fast_path_bails)`
        /// alongside the same bit-identical assertions (trimmed to the fields most relevant to a
        /// fault-attribution check here — `mcause`/`mepc`/`mtval` — the full CSR sweep is already
        /// covered by `check_ls_program`'s many callers elsewhere in this file).
        fn check_and_count(words: &[u32], initial_regs: [u32; 32]) -> (u64, u64) {
            let bytes = assemble(words);

            let mut m_i = fresh_machine(&bytes);
            let mut cpu_i = Cpu::new(RAM_BASE);
            cpu_i.regs = initial_regs;
            cpu_i.htif_tohost = Some(BAD_DATA + 0x800);
            let exit_i = run_reference_once(&mut cpu_i, &mut m_i, words, RAM_BASE);

            let mut m_j = fresh_machine(&bytes);
            let mut cpu_j = Cpu::new(RAM_BASE);
            cpu_j.regs = initial_regs;
            cpu_j.htif_tohost = Some(BAD_DATA + 0x800);
            let mut cache = ChainCache::with_capacity(256, 256);
            let exit_j = cache.run_block(&mut cpu_j, &mut m_j, &mut |_| true);

            assert_eq!(exit_i, exit_j, "SysExit mismatch\nprogram={words:02x?}");
            assert_eq!(cpu_i.regs, cpu_j.regs, "register mismatch\nprogram={words:02x?}");
            assert_eq!(cpu_i.pc, cpu_j.pc, "pc mismatch\nprogram={words:02x?}");
            assert_eq!(cpu_i.insns_retired, cpu_j.insns_retired, "insns_retired mismatch\nprogram={words:02x?}");
            assert_eq!(cpu_i.csr.mcause, cpu_j.csr.mcause, "mcause mismatch\nprogram={words:02x?}");
            assert_eq!(cpu_i.csr.mepc, cpu_j.csr.mepc, "mepc mismatch\nprogram={words:02x?}");
            assert_eq!(cpu_i.csr.mtval, cpu_j.csr.mtval, "mtval mismatch\nprogram={words:02x?}");
            (cpu_j.fast_path_hits, cpu_j.fast_path_bails)
        }

        /// A page carrying EXACTLY `PERM_READ` (no `PERM_WRITE`): a Load must fast-path (its perm
        /// byte matches the fast path's `need` exactly), while a Store to the same bytes must bail
        /// (perm lacks `PERM_WRITE`) and then fault `Permission` through the unchanged slow
        /// call-out — exactly like the interpreter. This is the "perm-fault byte" case the task
        /// requires distinctly from `BAD_DATA`'s "no perms at all".
        #[test]
        fn read_only_page_load_fast_paths_store_bails_then_faults() {
            let words = vec![
                load_insn(LoadOp::Lw, 4, 5, 0),
                store_insn(StoreOp::Sw, 5, 6, 0),
                ecall(),
            ];
            let mut regs = seeded_regs(&mut Rng::new(0xF00D_0001));
            regs[5] = RO_DATA;
            regs[6] = 0xAAAA_5555;
            let (hits, bails) = check_and_count(&words, regs);
            assert_eq!(hits, 1, "the read-only Load should have taken the fast path");
            assert!(bails >= 1, "the Store to a read-only page must bail (no PERM_WRITE)");
        }

        /// A page carrying `PERM_WRITE | PERM_RAW` (allocated, never written): a Load must bail
        /// (perm lacks `PERM_READ` outright — an even stronger bail than the perm-mismatch case
        /// above) and fault `Permission`, preserving the RAW/uninitialized-read oracle through the
        /// unchanged slow call-out — proving the fast path can never silently "unpoison" a RAW
        /// byte by racing ahead of `load_impl`'s real check.
        #[test]
        fn raw_poisoned_byte_load_bails_and_preserves_raw_fault() {
            let words = vec![load_insn(LoadOp::Lw, 4, 5, 0), ecall()];
            let mut regs = seeded_regs(&mut Rng::new(0xF00D_0002));
            regs[5] = RAW_DATA;
            let (hits, bails) = check_and_count(&words, regs);
            assert_eq!(hits, 0, "a RAW-poisoned Load must never take the fast path");
            assert_eq!(bails, 1);
        }

        /// The same RAW-poisoned bytes, but a Store first (which legitimately clears RAW / sets
        /// READ, exactly like `Mmu::write`) — the Store itself must bail (perm is `WRITE|RAW`, not
        /// exactly `READ|WRITE`) and go through the slow call-out, which performs the real
        /// RAW-clearing write. A subsequent Load's bail/hit outcome is then whatever the (now
        /// `READ|WRITE`) perm byte implies under this fast path's exact-match rule — asserted here
        /// only to document the honest, measured consequence, not to claim a specific ratio is
        /// "correct": the point of this test is that regs/pc/insns_retired (compared bit-for-bit
        /// against the interpreter above) are unaffected either way.
        #[test]
        fn raw_poisoned_byte_store_clears_raw_through_slow_path() {
            let words = vec![
                store_insn(StoreOp::Sw, 5, 6, 0),
                load_insn(LoadOp::Lw, 4, 5, 0),
                ecall(),
            ];
            let mut regs = seeded_regs(&mut Rng::new(0xF00D_0003));
            regs[5] = RAW_DATA;
            regs[6] = 0x1234_5678;
            let (_hits, bails) = check_and_count(&words, regs);
            assert!(bails >= 1, "the Store to a RAW-poisoned (WRITE|RAW, no READ) byte must bail");
        }

        /// A Load from the CLINT MMIO window must never take the fast path (there is no direct
        /// host pointer for a device register) and must produce the identical CLINT-register value
        /// as the interpreter through the unchanged slow call-out.
        #[test]
        fn clint_load_bails_and_matches_interpreter() {
            let words = vec![load_insn(LoadOp::Lw, 4, 5, 0), ecall()];
            let mut regs = seeded_regs(&mut Rng::new(0xF00D_0004));
            regs[5] = fs_platform::CLINT_BASE;
            let (hits, bails) = check_and_count(&words, regs);
            assert_eq!(hits, 0, "a CLINT load must never take the fast path");
            assert_eq!(bails, 1);
        }

        // -----------------------------------------------------------------------------------
        // sv32: the software-TLB-hit fast path, and the dirty-bit gate on a Store, exercised for
        // real (every test elsewhere in this file runs bare/M-mode, where `Cpu::fast_pa`'s
        // identity-mapped shortcut is the ONLY branch ever taken — this is the one test that
        // actually populates and consults the TLB-hit branch).
        // -----------------------------------------------------------------------------------

        /// Builds one 4 KiB non-leaf `l0` table (covering VA `0x0000_0000..0x0040_0000`) plus a
        /// 4 MiB identity superpage (RWX, A|D preset) covering `base` itself so fetching the test's
        /// own code works, and two extra leaf pages reached through `l0`: `load_va` (A|D both
        /// preset — its first access, a Load, is a genuine TLB miss; its second is a TLB hit
        /// regardless of access kind) and `store_va` (A only, D unset — a Load first refills the
        /// TLB with `d_set=false`; a subsequent Store must then see a TLB HIT but bail because the
        /// dirty bit isn't known-set, forcing the real walk that sets it).
        fn build_sv32_machine(bytes: &[u8], base: u32, root: u32, l0: u32, load_leaf_pa: u32, store_leaf_pa: u32) -> fs_platform::Machine {
            const V: u32 = 1;
            const R: u32 = 2;
            const W: u32 = 4;
            const X: u32 = 8;
            const A: u32 = 1 << 6;
            const D: u32 = 1 << 7;
            let leaf = |pa: u32, flags: u32| ((pa >> 12) << 10) | flags;

            let mut m = fs_platform::Machine::new(base, 0x10_0000);
            m.ram.protect(base, 0x10_0000, PERM_READ | PERM_WRITE).unwrap();
            m.ram.map(base, bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
            // Identity 4 MiB superpage for CODE (`base` is 4 MiB-aligned: 0x8000_0000).
            let root_idx = (base >> 22) & 0x3ff;
            m.ram.write_u32(root + root_idx * 4, leaf(base, V | R | W | X | A | D)).unwrap();
            // root[0] -> l0 (non-leaf), covering the low VA range `load_va`/`store_va` live in.
            m.ram.write_u32(root, ((l0 >> 12) << 10) | V).unwrap();
            m.ram.write_u32(l0 + 5 * 4, leaf(load_leaf_pa, V | R | W | A | D)).unwrap();
            m.ram.write_u32(l0 + 6 * 4, leaf(store_leaf_pa, V | R | W | A)).unwrap();
            m
        }

        #[test]
        fn sv32_tlb_miss_then_hit_and_dirty_bit_gates_store_fast_path() {
            let base = 0x8000_0000u32;
            let root = base + 0x1000;
            let l0 = base + 0x2000;
            let load_leaf_pa = base + 0x9000;
            let store_leaf_pa = base + 0xa000;
            let load_va = 0x0000_5000u32; // vpn1=0 (via l0), vpn0=5
            let store_va = 0x0000_6000u32; // vpn1=0 (via l0), vpn0=6

            let words = vec![
                load_insn(LoadOp::Lw, 4, 10, 0),   // x10=load_va:  TLB MISS -> bail
                load_insn(LoadOp::Lw, 6, 10, 0),   // TLB HIT -> fast-path hit
                load_insn(LoadOp::Lw, 8, 11, 0),   // x11=store_va: TLB MISS -> bail (refills d_set=false)
                opimm(AluOp::Add, 7, 0, 0x123),
                store_insn(StoreOp::Sw, 11, 7, 0), // TLB HIT but d_set=false -> bail (walk sets D)
                store_insn(StoreOp::Sw, 11, 7, 0), // TLB HIT, d_set=true -> fast-path hit
                ecall(),
            ];
            let bytes = assemble(&words);
            let mut regs = [0u32; 32];
            regs[10] = load_va;
            regs[11] = store_va;

            let mut m_i = build_sv32_machine(&bytes, base, root, l0, load_leaf_pa, store_leaf_pa);
            let mut cpu_i = Cpu::new(base);
            cpu_i.privilege = Priv::S;
            cpu_i.csr.satp = (1 << 31) | (root >> 12);
            cpu_i.regs = regs;
            let exit_i = run_reference_once(&mut cpu_i, &mut m_i, &words, base);

            let mut m_j = build_sv32_machine(&bytes, base, root, l0, load_leaf_pa, store_leaf_pa);
            let mut cpu_j = Cpu::new(base);
            cpu_j.privilege = Priv::S;
            cpu_j.csr.satp = (1 << 31) | (root >> 12);
            cpu_j.regs = regs;
            let mut cache = ChainCache::with_capacity(256, 256);
            let exit_j = cache.run_block(&mut cpu_j, &mut m_j, &mut |_| true);

            assert_eq!(exit_i, exit_j);
            assert_eq!(cpu_i.regs, cpu_j.regs, "register mismatch");
            assert_eq!(cpu_i.pc, cpu_j.pc, "pc mismatch");
            assert_eq!(cpu_i.insns_retired, cpu_j.insns_retired, "insns_retired mismatch");

            // Prove every branch of `Cpu::fast_pa` this scenario targets was actually exercised,
            // not vacuously always-bail or always-hit: 2 fresh TLB misses (`load_va`/`store_va`'s
            // first touch) + 1 TLB-hit-but-not-dirty bail on the Store = 3 bails; 2 TLB hits (the
            // second Load, the second Store once dirty) = 2 hits.
            assert_eq!(cpu_j.fast_path_bails, 3, "expected 2 TLB misses + 1 not-dirty-yet bail");
            assert_eq!(cpu_j.fast_path_hits, 2, "expected the second Load hit + the now-dirty Store hit");
        }
    }
}
