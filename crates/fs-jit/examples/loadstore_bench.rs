//! Load/Store-heavy micro-benchmark for Phase 3 of `docs/jit-scalar-design.md` (the inlined
//! memory fast path). Deliberately the OPPOSITE mix from the real boot+syscall-fuzz workload
//! (`fs-cli fuzz --jit-chain`, which is ALU/branch-dominated with only occasional memory ops):
//! a tight loop over a large array where every iteration retires one Load, one Store, and only
//! two ALU ops plus the loop branch — so Phase 3's inlined fast path (vs. Phase 2's always-call-out
//! Load/Store) gets the largest possible share of retired-instruction time to win back.
//!
//! Not wired into any differential harness (this is a raw throughput measurement, not a
//! correctness gate — the correctness gate is `tests/chain_differential.rs`) — run with:
//! `cargo run -p fs-jit --release --example loadstore_bench`.

use fs_jit::ChainCache;
use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_riscv::{asm, Cpu, SysExit, A0, T0};

/// `a1` (x11) — not exported as a named const by `fs-riscv` (only a handful of registers get
/// aliases), so named locally.
const A1: u8 = 11;

const BASE: u32 = 0x8000_0000;
/// Array size: 1 MiB of `u32` words — big enough that the loop body's compiled chain is reused
/// across many iterations (one compile, millions of native dispatches), matching the real
/// fuzzing workload's "compile once, run the same physical code very many times" shape.
const ARRAY_WORDS: u32 = 256 * 1024;
const ARRAY_BASE: u32 = BASE + 0x10_0000;
const ARRAY_END: u32 = ARRAY_BASE + ARRAY_WORDS * 4;
/// How many full sweeps over the array to run — sized for a multi-second measurement window.
const SWEEPS: u32 = 40;

fn build_program() -> Vec<u32> {
    use asm::*;
    // a0 = ARRAY_BASE (cursor), a1 = ARRAY_END (limit), t0 = scratch.
    // loop: lw t0,0(a0); addi t0,t0,1; sw t0,0(a0); addi a0,a0,4; bne a0,a1,loop
    vec![
        lw(T0, A0, 0),
        addi(T0, T0, 1),
        sw(A0, T0, 0),
        addi(A0, A0, 4),
        bne(A0, A1, -16),
        // Sentinel the chain must stop before / the fallback path executes for real: an ecall
        // exit once the outer sweep count is reached (checked by the driver loop below, which
        // just re-primes a0/a1 and re-enters — the `ecall` here is never actually reached because
        // the driver loop always resets `cpu.pc` back to the loop head instead).
        ecall(),
    ]
}

fn setup() -> (Cpu, Mmu) {
    let mut mmu = Mmu::new(BASE, 0x20_0000);
    mmu.protect(BASE, 0x20_0000, PERM_READ | PERM_WRITE).unwrap();
    let prog = build_program();
    let mut bytes = Vec::new();
    for w in &prog {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mmu.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    mmu.protect(ARRAY_BASE, ARRAY_WORDS * 4, PERM_READ | PERM_WRITE).unwrap();
    let mut cpu = Cpu::new(BASE);
    cpu.regs[A0 as usize] = ARRAY_BASE;
    cpu.regs[A1 as usize] = ARRAY_END;
    (cpu, mmu)
}

fn main() {
    let (mut cpu, mut mmu) = setup();
    let mut cache = ChainCache::new();

    let t0 = std::time::Instant::now();
    let start_insns = cpu.insns_retired;
    for _ in 0..SWEEPS {
        cpu.pc = BASE;
        cpu.regs[A0 as usize] = ARRAY_BASE;
        cpu.regs[A1 as usize] = ARRAY_END;
        loop {
            match cache.run_block(&mut cpu, &mut mmu, &mut |_| true) {
                SysExit::Continue => {
                    if cpu.regs[A0 as usize] == ARRAY_END {
                        break;
                    }
                }
                other => panic!("unexpected exit {other:?}"),
            }
        }
    }
    let elapsed = t0.elapsed();
    let retired = cpu.insns_retired - start_insns;
    let mips = retired as f64 / elapsed.as_secs_f64() / 1e6;

    println!("== fs-jit load/store-heavy micro-benchmark ==");
    println!("  array         : {ARRAY_WORDS} words ({} sweeps)", SWEEPS);
    println!("  insns retired : {retired}");
    println!("  elapsed       : {:.3}s", elapsed.as_secs_f64());
    println!("  guest speed   : {mips:.0} MIPS");
    println!(
        "  chain-jit     : {} native chains, {} compiles, {} fallback single-steps",
        cache.chain_hits(),
        cache.chain_misses(),
        cache.fallbacks()
    );
    let fast_total = cpu.fast_path_hits + cpu.fast_path_bails;
    let fast_pct = if fast_total > 0 { cpu.fast_path_hits as f64 / fast_total as f64 * 100.0 } else { 0.0 };
    println!(
        "  fast mem path : {} hits, {} bails ({fast_pct:.1}% hit rate)  [Phase 3, docs/jit-scalar-design.md]",
        cpu.fast_path_hits, cpu.fast_path_bails
    );
}
