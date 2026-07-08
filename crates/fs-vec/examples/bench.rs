//! Throughput micro-benchmark: converged-lane SIMD ALU fast path vs. scalar-over-lanes.
//!
//! Run with `cargo run --release --example bench -p fs-vec` (release matters — this is a raw
//! interpreter loop and debug builds are dominated by bounds-check/panic-machinery noise, not the
//! thing being measured). No external crates: timing is plain `std::time::Instant`.
//!
//! Both engines run the *exact same* RV32IM program: a fixed-iteration-count counting loop (same
//! trip count on every lane, so lanes stay `pc`-converged for the whole run — architecture.md
//! §2's "lockstep" precondition) wrapped around a straight-line ALU body that folds a per-lane
//! seed (`T4`) into a per-lane accumulator (`T3`), the same instruction mix
//! `converged_straight_line_alu_program_uses_the_simd_fast_path` in `src/lib.rs` exercises unit
//! -tested, just looped instead of unrolled once.
//!
//! - **SIMD path**: [`fs_vec::VecCpu::step`] — because every lane's `pc` stays converged for the
//!   loop's entire run, `try_simd_alu_step` decodes each ALU instruction once and executes it as
//!   one packed `Simd<u32, LANES>` op across all 16 lanes; only the loop's `bge`/`jal` fall back
//!   to the scalar path (once each per iteration).
//! - **Scalar-over-lanes path**: `LANES` independent [`fs_riscv::Cpu`]s, each stepped one
//!   instruction at a time. This is the exact decode/execute logic `VecCpu`'s own scalar fallback
//!   calls, just run `LANES` times over instead of once as a packed vector op — i.e. what
//!   `VecCpu::step` did before the SIMD fast path existed (see `DESIGN.md`).

use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_riscv::{asm, Exit, A0, A7, T0, T1, X0};
use fs_vec::{VecCpu, LANES};
use std::time::Instant;

const BASE: u32 = 0x8000_0000;
const T3: u8 = 28; // per-lane ALU accumulator
const T4: u8 = 29; // per-lane seed, folded into the accumulator every iteration
// `addi`'s I-type immediate is a signed 12-bit field (-2048..=2047, `fs_riscv::asm::addi`), so the
// loop trip count encoded directly into the guest program is capped there; `OUTER_REPEATS` below
// re-runs the whole program at the Rust level to reach a stable measurement instead.
const ITERS: i32 = 2000;
const OUTER_REPEATS: u32 = 40;

// --- Minimal R-type/I-type encoders for the ALU ops `fs_riscv::asm` doesn't expose helpers for
// (mirrors `decode`'s funct3/funct7 tables — see `src/lib.rs`'s test module for the same pattern).
fn i_type(op: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
    ((imm as u32 & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
}
fn r_type(op: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
}
fn and_(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 7, 0x00, rd, rs1, rs2)
}
fn or_(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 6, 0x00, rd, rs1, rs2)
}
fn xor_(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 4, 0x00, rd, rs1, rs2)
}
fn slli(rd: u8, rs1: u8, shamt: u8) -> u32 {
    r_type(0x13, 1, 0x00, rd, rs1, shamt)
}
fn srli(rd: u8, rs1: u8, shamt: u8) -> u32 {
    r_type(0x13, 5, 0x00, rd, rs1, shamt)
}
fn andi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 7, rd, rs1, imm)
}

/// Indices (each instruction is 4 bytes):
/// ```text
/// 0  addi T0, X0, 0        i = 0
/// 1  addi T1, X0, ITERS    limit
/// 2  bge  T0, T1, +48      if i >= limit -> 14 (done)
/// 3  addi T3, T3, 5
/// 4  and_ T3, T3, T4
/// 5  or_  T3, T3, T4
/// 6  xor_ T3, T3, T4
/// 7  slli T3, T3, 1
/// 8  srli T3, T3, 1
/// 9  add  T3, T3, T4
/// 10 sub  T3, T3, T4
/// 11 andi T3, T3, 0x7fff
/// 12 addi T0, T0, 1        i++
/// 13 jal  X0, -44          -> 2
/// 14 add  A0, T3, X0       done: a0 = t3
/// 15 addi A7, X0, 93       a7 = exit
/// 16 ecall
/// ```
fn counting_alu_loop_program() -> Vec<u32> {
    use asm::*;
    vec![
        addi(T0, X0, 0),
        addi(T1, X0, ITERS),
        bge(T0, T1, 48),
        addi(T3, T3, 5),
        and_(T3, T3, T4),
        or_(T3, T3, T4),
        xor_(T3, T3, T4),
        slli(T3, T3, 1),
        srli(T3, T3, 1),
        add(T3, T3, T4),
        sub(T3, T3, T4),
        andi(T3, T3, 0x7fff),
        addi(T0, T0, 1),
        jal(X0, -44),
        add(A0, T3, X0),
        addi(A7, X0, 93),
        ecall(),
    ]
}

fn make_mmu(prog: &[u32]) -> Mmu {
    let mut mmu = Mmu::new(BASE, 0x1_0000);
    mmu.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
    let mut bytes = Vec::new();
    for w in prog {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mmu.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    mmu
}

fn seed(lane: usize) -> u32 {
    (lane as u32) * 7 + 1
}

/// Runs `prog` on `VecCpu`, which takes the SIMD fast path for every converged ALU instruction.
/// Returns `(total lane-instructions retired, step() calls that used the SIMD fast path)`.
fn run_simd(prog: &[u32]) -> (u64, u64) {
    let mut buses: Vec<Mmu> = (0..LANES).map(|_| make_mmu(prog)).collect();
    let mut vcpu = VecCpu::new(BASE);
    for lane in 0..LANES {
        vcpu.set_reg(lane, T4, seed(lane));
    }
    while vcpu.any_active() {
        vcpu.step(&mut buses);
    }
    (vcpu.insns_retired.iter().sum(), vcpu.simd_alu_steps)
}

/// Runs `prog` on `LANES` independent scalar `fs_riscv::Cpu`s, one instruction at a time — the
/// scalar-over-lanes baseline `VecCpu::step` itself falls back to, run without any packing.
fn run_scalar_over_lanes(prog: &[u32]) -> u64 {
    let mut total = 0u64;
    for lane in 0..LANES {
        let mut mmu = make_mmu(prog);
        let mut cpu = fs_riscv::Cpu::new(BASE);
        cpu.regs[T4 as usize] = seed(lane);
        loop {
            match cpu.step(&mut mmu).unwrap() {
                Exit::Continue => total += 1,
                Exit::Ecall => break,
                other => panic!("unexpected scalar exit: {other:?}"),
            }
        }
    }
    total
}

fn main() {
    let prog = counting_alu_loop_program();

    let simd_start = Instant::now();
    let mut simd_total_insns = 0u64;
    let mut simd_alu_steps = 0u64;
    for _ in 0..OUTER_REPEATS {
        let (insns, steps) = run_simd(&prog);
        simd_total_insns += insns;
        simd_alu_steps += steps;
    }
    let simd_elapsed = simd_start.elapsed();

    let scalar_start = Instant::now();
    let mut scalar_total_insns = 0u64;
    for _ in 0..OUTER_REPEATS {
        scalar_total_insns += run_scalar_over_lanes(&prog);
    }
    let scalar_elapsed = scalar_start.elapsed();

    assert_eq!(
        simd_total_insns, scalar_total_insns,
        "the two engines must retire the same number of instructions (correctness precondition \
         for a meaningful throughput comparison)"
    );

    let simd_rate = simd_total_insns as f64 / simd_elapsed.as_secs_f64();
    let scalar_rate = scalar_total_insns as f64 / scalar_elapsed.as_secs_f64();

    println!(
        "fs-vec throughput micro-benchmark ({LANES} lanes x {ITERS} loop iterations x \
         {OUTER_REPEATS} repeats)"
    );
    println!(
        "  SIMD fast path:    {simd_total_insns} lane-instructions in {simd_elapsed:?}  =  \
         {simd_rate:.0} lane-instr/sec  ({simd_alu_steps} step() calls took the SIMD ALU path)"
    );
    println!(
        "  scalar-over-lanes: {scalar_total_insns} lane-instructions in {scalar_elapsed:?}  =  \
         {scalar_rate:.0} lane-instr/sec"
    );
    println!("  speedup:           {:.2}x", simd_rate / scalar_rate);
}
