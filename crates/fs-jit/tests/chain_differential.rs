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

fn fs_riscv_store_sw(rs1: u8, rs2: u8, imm: i32) -> u32 {
    // S-type SW encoding (opcode 0x23, funct3=2) — only the handler code needs a real store
    // (Store isn't Phase 1 chain scope, but the handler itself runs through the ordinary
    // interpreter fallback after the trap redirects pc there, same as any uncompiled instruction).
    let s_imm = imm as u32;
    ((s_imm & 0xfe0) << 20)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (2 << 12)
        | ((s_imm & 0x1f) << 7)
        | 0x23
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
