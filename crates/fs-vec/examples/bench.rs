//! Throughput micro-benchmark: converged-lane SIMD ALU fast path vs. scalar-over-lanes.
//!
//! Run with `cargo run --release --example bench -p fs-vec` (release matters — this is a raw
//! interpreter loop and debug builds are dominated by bounds-check/panic-machinery noise, not the
//! thing being measured). No external crates: timing is plain `std::time::Instant`.
//!
//! Two RV32IM programs are benchmarked, both fixed-iteration-count counting loops (same trip
//! count on every lane, so lanes stay `pc`-converged for the whole run — architecture.md §2's
//! "lockstep" precondition), each run on two engines:
//!
//! - **SIMD path**: [`fs_vec::VecCpu::step`] against a single shared [`fs_vec::VecMmu`] — because
//!   every lane's `pc` stays converged for the loop's entire run, the group fetches its
//!   instruction **once** per step from the shared interleaved store (`VecMmu::ifetch16_same`,
//!   not `LANES` per-lane fetches as in the original cut of this executor); `try_simd_alu` decodes
//!   each ALU instruction once and executes it as one packed `Simd<u32, LANES>` op across all 16
//!   lanes, and the loop's `bge`/`jal` are now *also* packed (`try_simd_branch`/`try_simd_jal`) —
//!   every lane computes its own branch outcome/jump target from its own operands as one masked
//!   vector op, so this loop body has no remaining always-scalar instruction at all (see
//!   `DESIGN.md`, which used to note `bge`/`jal` as exactly the two instructions per iteration
//!   that could never be vectorized).
//! - **Scalar-over-lanes path**: `LANES` independent [`fs_riscv::Cpu`]s, each with its own
//!   `fs_mmu::Mmu` and each stepped one instruction at a time. This is the exact decode/execute
//!   logic `VecCpu`'s own scalar fallback calls, just run `LANES` times over instead of once as a
//!   packed vector op — i.e. what `VecCpu::step` did before the SIMD fast path existed, fetch
//!   included (see `DESIGN.md`).
//!
//! `counting_alu_loop_program` folds a per-lane seed (`T4`) into a per-lane accumulator (`T3`)
//! kept in a register — the same instruction mix
//! `converged_straight_line_alu_program_uses_the_simd_fast_path` in `src/lib.rs` exercises
//! unit-tested, just looped instead of unrolled once; it exercises the fetch + ALU fast paths
//! only (no memory ops in its loop body). `counting_mem_loop_program` is the same shape but routes
//! the accumulator through ONE shared guest address every iteration instead, exercising `VecMmu`'s
//! same-address `load_same`/`store_same` fast path too. `counting_mul_loop_program` folds in one
//! `mul` per iteration on top of the plain ALU mix, exercising `try_simd_mul`'s masked-scalarize
//! fast path (`src/lib.rs`'s module docs): before that payload existed, `Inst::Mul` fell straight
//! through `try_simd_fast_step` (returning `false`), so the ONE step per iteration that decoded the
//! `mul` dropped the *entire converged group* to `step_lane`'s per-lane fallback for that step
//! alone — every other instruction in the same loop body still took its own fast path. `bench_one`
//! reports each scenario's fast-path *fraction* (the share of `step()` calls that took ANY SIMD
//! payload rather than falling all the way to scalar-over-lanes) precisely to make that "one
//! instruction, one step, not the whole loop" accounting visible for this scenario.

use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_riscv::{asm, Exit, A0, A7, T0, T1, X0};
use fs_vec::{VecCpu, VecMmu, LANES};
use std::time::Instant;

const BASE: u32 = 0x8000_0000;
const T3: u8 = 28; // per-lane ALU accumulator
const T4: u8 = 29; // per-lane seed, folded into the accumulator every iteration
const T5: u8 = 30; // shared (same-address-across-lanes) data pointer, memory-loop benchmark only
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

/// A shared-address memory loop: same trip count and shape as `counting_alu_loop_program`, but
/// the per-lane accumulator now round-trips through ONE shared guest address (`t5`, identical
/// across every lane) every iteration instead of staying in a register — exercising `VecCpu`'s
/// *other* new fast path, `VecMmu`'s same-address `load_same`/`store_same` (DESIGN.md "Memory:
/// same-address fast path"), which the ALU-only loop above never touches. Each lane still reads
/// back only its own value (the interleaved layout keeps every lane's copy of that shared address
/// independent), so this is still a valid throughput comparison, not a data race.
///
/// Indices (each instruction is 4 bytes):
/// ```text
///  0  addi T0, X0, 0        i = 0
///  1  addi T1, X0, ITERS    limit
///  2  lui  T5, BASE         t5 = BASE
///  3  addi T5, T5, 0x100    t5 += 0x100 (shared data pointer, same for every lane)
///  4  bge  T0, T1, +24      if i >= limit -> 10 (done)
///  5  lw   T3, T5, 0        t3 = mem[t5]  (this lane's own copy)
///  6  add  T3, T3, T4       t3 += seed
///  7  sw   T5, T3, 0        mem[t5] = t3
///  8  addi T0, T0, 1        i++
///  9  jal  X0, -20          -> 4
/// 10 add  A0, T3, X0        done: a0 = t3
/// 11 addi A7, X0, 93        a7 = exit
/// 12 ecall
/// ```
fn counting_mem_loop_program() -> Vec<u32> {
    use asm::*;
    vec![
        addi(T0, X0, 0),
        addi(T1, X0, ITERS),
        lui(T5, BASE),
        addi(T5, T5, 0x100),
        bge(T0, T1, 24),
        lw(T3, T5, 0),
        add(T3, T3, T4),
        sw(T5, T3, 0),
        addi(T0, T0, 1),
        jal(X0, -20),
        add(A0, T3, X0),
        addi(A7, X0, 93),
        ecall(),
    ]
}

/// Same shape as `counting_alu_loop_program`, but with one `mul` folded into the loop body — the
/// scenario `try_simd_mul`'s masked-scalarize fast path (src/lib.rs) exists for: MUL/DIV/REM have
/// no clean packed SIMD form, so before that payload existed, `try_simd_fast_step` declined
/// outright on `Inst::Mul` and the WHOLE converged group fell to `step_lane`'s per-lane fallback
/// for that one step (re-fetching per lane and losing the shared decode) — even though every other
/// instruction in the same loop body stayed converged and packed. This is exactly the
/// "reduce scalar fallback" case the fast path targets: MUL is common in real code (address math,
/// hashing), so a loop that touches it even once per iteration used to give back a chunk of the
/// throughput win the surrounding ALU instructions otherwise get.
///
/// Indices (each instruction is 4 bytes):
/// ```text
///  0  addi T0, X0, 0        i = 0
///  1  addi T1, X0, ITERS    limit
///  2  bge  T0, T1, +32      if i >= limit -> 10 (done)
///  3  addi T3, T3, 5
///  4  mul  T3, T3, T4       t3 *= seed   (the one non-packed-ALU instruction per iteration)
///  5  add  T3, T3, T4
///  6  sub  T3, T3, T4
///  7  andi T3, T3, 0x7fff
///  8  addi T0, T0, 1        i++
///  9  jal  X0, -28          -> 2
/// 10 add  A0, T3, X0        done: a0 = t3
/// 11 addi A7, X0, 93        a7 = exit
/// 12 ecall
/// ```
fn counting_mul_loop_program() -> Vec<u32> {
    use asm::*;
    vec![
        addi(T0, X0, 0),
        addi(T1, X0, ITERS),
        bge(T0, T1, 32),
        addi(T3, T3, 5),
        mul(T3, T3, T4),
        add(T3, T3, T4),
        sub(T3, T3, T4),
        andi(T3, T3, 0x7fff),
        addi(T0, T0, 1),
        jal(X0, -28),
        add(A0, T3, X0),
        addi(A7, X0, 93),
        ecall(),
    ]
}

/// A divergent-address ("gather/scatter") memory loop: same trip count and shape as
/// `counting_mem_loop_program`, but `t5` is seeded to a DISTINCT address per lane (`gather_addr`,
/// set once before the run, never recomputed by the program itself) instead of one shared address
/// — exercising `VecCpu`'s gather/scatter SIMD fast path (`VecMmu::load_gather_fast`/
/// `store_scatter_fast`, Goal 2) instead of the same-address path `counting_mem_loop_program`
/// isolates. Every lane's `t5` stays within the mapped window and never overlaps another lane's,
/// so this is still race-free, just genuinely per-lane-scattered.
///
/// Indices (each instruction is 4 bytes):
/// ```text
///  0  addi T0, X0, 0        i = 0
///  1  addi T1, X0, ITERS    limit
///  2  bge  T0, T1, +24      if i >= limit -> 8 (done)
///  3  lw   T3, T5, 0        t3 = mem[t5]  (t5 preset per-lane, distinct across lanes)
///  4  add  T3, T3, T4       t3 += seed
///  5  sw   T5, T3, 0        mem[t5] = t3
///  6  addi T0, T0, 1        i++
///  7  jal  X0, -20          -> 2
///  8  add  A0, T3, X0       done: a0 = t3
///  9  addi A7, X0, 93       a7 = exit
/// 10 ecall
/// ```
fn counting_gather_loop_program() -> Vec<u32> {
    use asm::*;
    vec![
        addi(T0, X0, 0),
        addi(T1, X0, ITERS),
        bge(T0, T1, 24),
        lw(T3, T5, 0),
        add(T3, T3, T4),
        sw(T5, T3, 0),
        addi(T0, T0, 1),
        jal(X0, -20),
        add(A0, T3, X0),
        addi(A7, X0, 93),
        ecall(),
    ]
}

/// Per-lane gather/scatter data address for `counting_gather_loop_program`: spread `LANES` words
/// apart so no two lanes ever touch the same guest word, well within the mapped `0x1_0000`-byte
/// window.
fn gather_addr(lane: usize) -> u32 {
    BASE + 0x400 + (lane as u32) * 4
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

/// Same program, laid out once in the shared interleaved [`VecMmu`] (`map`/`protect` broadcast
/// identically to every lane) instead of `LANES` separate `fs_mmu::Mmu`s — this single shared
/// store is what lets a converged group's fetch happen once instead of `LANES` times.
fn make_vec_mmu(prog: &[u32]) -> VecMmu {
    let mut mmu = VecMmu::new(BASE, 0x1_0000);
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

/// One `run_simd` call's counters: lane-instructions retired, each `simd_*_steps` counter, and the
/// total number of `step()` calls made (`group_steps`) — the denominator for the fast-path
/// fraction `bench_one` reports (what share of `step()` calls took ANY SIMD payload rather than
/// falling all the way to `step_lane`'s scalar-over-lanes loop).
struct SimdRunStats {
    insns: u64,
    alu_steps: u64,
    branch_steps: u64,
    mem_steps: u64,
    gather_steps: u64,
    muldiv_steps: u64,
    group_steps: u64,
}

/// Runs `prog` on `VecCpu`, which takes the SIMD fast path for every converged ALU/branch/jump/
/// memory/MUL instruction. `t5_seed(lane)`, if given, presets each lane's `T5` before the run (used
/// only by `counting_gather_loop_program`, which relies on a preset per-lane address rather than
/// computing one in-program).
fn run_simd(prog: &[u32], t5_seed: Option<fn(usize) -> u32>) -> SimdRunStats {
    let mut mmu = make_vec_mmu(prog);
    let mut vcpu = VecCpu::new(BASE);
    for lane in 0..LANES {
        vcpu.set_reg(lane, T4, seed(lane));
        if let Some(f) = t5_seed {
            vcpu.set_reg(lane, T5, f(lane));
        }
    }
    let mut group_steps = 0u64;
    while vcpu.any_active() {
        vcpu.step(&mut mmu);
        group_steps += 1;
    }
    SimdRunStats {
        insns: vcpu.insns_retired.iter().sum(),
        alu_steps: vcpu.simd_alu_steps,
        branch_steps: vcpu.simd_branch_steps,
        mem_steps: vcpu.simd_mem_steps,
        gather_steps: vcpu.simd_gather_steps,
        muldiv_steps: vcpu.simd_muldiv_steps,
        group_steps,
    }
}

/// Runs `prog` on `LANES` independent scalar `fs_riscv::Cpu`s, one instruction at a time — the
/// scalar-over-lanes baseline `VecCpu::step` itself falls back to, run without any packing.
fn run_scalar_over_lanes(prog: &[u32], t5_seed: Option<fn(usize) -> u32>) -> u64 {
    let mut total = 0u64;
    for lane in 0..LANES {
        let mut mmu = make_mmu(prog);
        let mut cpu = fs_riscv::Cpu::new(BASE);
        cpu.regs[T4 as usize] = seed(lane);
        if let Some(f) = t5_seed {
            cpu.regs[T5 as usize] = f(lane);
        }
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

/// Times both engines on `prog` and prints the comparison, labeled `name`.
fn bench_one(name: &str, prog: &[u32], t5_seed: Option<fn(usize) -> u32>) {
    let simd_start = Instant::now();
    let mut simd_total_insns = 0u64;
    let mut simd_alu_steps = 0u64;
    let mut simd_branch_steps = 0u64;
    let mut simd_mem_steps = 0u64;
    let mut simd_gather_steps = 0u64;
    let mut simd_muldiv_steps = 0u64;
    let mut simd_group_steps = 0u64;
    for _ in 0..OUTER_REPEATS {
        let stats = run_simd(prog, t5_seed);
        simd_total_insns += stats.insns;
        simd_alu_steps += stats.alu_steps;
        simd_branch_steps += stats.branch_steps;
        simd_mem_steps += stats.mem_steps;
        simd_gather_steps += stats.gather_steps;
        simd_muldiv_steps += stats.muldiv_steps;
        simd_group_steps += stats.group_steps;
    }
    let simd_elapsed = simd_start.elapsed();

    let scalar_start = Instant::now();
    let mut scalar_total_insns = 0u64;
    for _ in 0..OUTER_REPEATS {
        scalar_total_insns += run_scalar_over_lanes(prog, t5_seed);
    }
    let scalar_elapsed = scalar_start.elapsed();

    assert_eq!(
        simd_total_insns, scalar_total_insns,
        "the two engines must retire the same number of instructions (correctness precondition \
         for a meaningful throughput comparison)"
    );

    let simd_rate = simd_total_insns as f64 / simd_elapsed.as_secs_f64();
    let scalar_rate = scalar_total_insns as f64 / scalar_elapsed.as_secs_f64();
    let fast_path_steps =
        simd_alu_steps + simd_branch_steps + simd_mem_steps + simd_gather_steps + simd_muldiv_steps;
    let fast_path_fraction = fast_path_steps as f64 / simd_group_steps as f64;

    println!("{name} ({LANES} lanes x {ITERS} loop iterations x {OUTER_REPEATS} repeats)");
    println!(
        "  SIMD fast path:    {simd_total_insns} lane-instructions in {simd_elapsed:?}  =  \
         {simd_rate:.0} lane-instr/sec  ({simd_alu_steps} ALU-path + {simd_branch_steps} \
         branch/jump-path + {simd_mem_steps} same-address-mem-path + {simd_gather_steps} \
         gather/scatter-mem-path + {simd_muldiv_steps} masked-scalarize-MUL-path step() calls)"
    );
    println!(
        "  fast-path share:   {fast_path_steps}/{simd_group_steps} step() calls took a SIMD \
         payload  =  {:.2}% (rest fell to the scalar-over-lanes step_lane loop)",
        fast_path_fraction * 100.0
    );
    println!(
        "  scalar-over-lanes: {scalar_total_insns} lane-instructions in {scalar_elapsed:?}  =  \
         {scalar_rate:.0} lane-instr/sec"
    );
    println!("  speedup:           {:.2}x\n", simd_rate / scalar_rate);
}

fn main() {
    bench_one(
        "fs-vec throughput micro-benchmark: ALU+fetch-bound loop",
        &counting_alu_loop_program(),
        None,
    );
    bench_one(
        "fs-vec throughput micro-benchmark: same-address memory loop",
        &counting_mem_loop_program(),
        None,
    );
    bench_one(
        "fs-vec throughput micro-benchmark: divergent-address (gather/scatter) memory loop",
        &counting_gather_loop_program(),
        Some(gather_addr),
    );
    bench_one(
        "fs-vec throughput micro-benchmark: MUL-in-loop (masked-scalarize fast path)",
        &counting_mul_loop_program(),
        None,
    );
}
