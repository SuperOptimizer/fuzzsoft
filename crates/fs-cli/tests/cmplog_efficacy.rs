//! CMPLOG efficacy demo — NOT part of the shipped fuzzer (see `--cmplog` in `src/main.rs` for the
//! real integration). This is a small, self-contained, deterministic proof that comparison-log
//! guided mutation solves a data-dependent "magic value" branch — `if (x == MAGIC)`, the classic
//! case plain coverage-guided random mutation cannot solve in any practical number of tries — in
//! essentially one step, built directly on the real `fs_riscv::Cpu` executor (with
//! `set_cmplog`/`cmplog_take`) and the real `fs_prog::mutate_cmplog` operator. No kernel, no
//! syscalls: just the RV32 core and the typed-arg mutator, so the demo is fast and fully auditable.
//!
//! What it proves:
//!   1. Correctness: recording is read-only. A traced run and an unrecorded run of the identical
//!      program reach the identical outcome (same exit code, same instruction count).
//!   2. Efficacy: an undirected baseline that tries `N = 100_000` distinct candidate values (a
//!      generous per-campaign case budget) *never* solves the check — by construction, since the
//!      baseline enumerates values strictly below `MAGIC` and can therefore never land on it,
//!      which is the deterministic, always-true version of "random mutation essentially never
//!      guesses a 32-bit constant" (true with overwhelming probability for a real PRNG too: a
//!      uniform 32-bit guess lands on a fixed target with probability p = N/2^32 ≈ 2.3e-5 for
//!      N = 100_000). CMPLOG solves it in a trace run (to observe the compared operands) plus a
//!      handful of confirming runs: `mutate_cmplog` offers MAGIC *and* its ±1 neighbours (to also
//!      catch `<`/`<=`/`>`/`>=`-style checks), so an exact-equality check like this one needs on
//!      average 3 draws to land exactly on MAGIC rather than a neighbour — still trivially few
//!      next to 2^32.

use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_prog::{ArgValue, Prog, Rng, TypedCall, mutate_cmplog};
use fs_riscv::{A0, A7, Cpu, Exit, T0, T1, X0, asm};

/// A "random-looking" 32-bit constant nothing in the baseline's enumeration will ever hit.
const MAGIC: u32 = 0xC0FF_EE42;
const BASE: u32 = 0x8000_0000;
/// Where the guest program expects to find its one "fuzzed" argument (a page distinct from code).
const DATA_ADDR: u32 = 0x8000_2000;

/// `lui`+`addi` immediate-load, accounting for `addi`'s 12-bit sign extension (the standard
/// two-instruction 32-bit-constant idiom every RISC-V assembler emits).
fn li(rd: u8, val: u32) -> [u32; 2] {
    let low = ((val & 0xfff) as i32) << 20 >> 20; // sign-extend the low 12 bits
    let hi = val.wrapping_sub(low as u32) & 0xffff_f000;
    [asm::lui(rd, hi), asm::addi(rd, rd, low)]
}

/// Assemble the guest check: `t0 = [DATA_ADDR]` (the fuzzed candidate), `t1 = MAGIC`, branch to
/// `found` (a0 = 42) on equality, else a0 = 0. Both paths `ecall` to exit.
fn build_program() -> Vec<u32> {
    let [lui_t1, addi_t1] = li(T1, MAGIC);
    vec![
        asm::lui(T0, DATA_ADDR),  // 0: t0 = DATA_ADDR (page-aligned, so lo12 == 0)
        asm::lw(T0, T0, 0),       // 1: t0 = [DATA_ADDR]  (the candidate value)
        lui_t1,                  // 2: t1 = MAGIC (high 20 bits)
        addi_t1,                 // 3: t1 = MAGIC (low 12 bits, sign-extend corrected)
        asm::beq(T0, T1, 16),    // 4: if t0 == t1 -> found (idx 8)
        asm::addi(A0, X0, 0),    // 5: not found: a0 = 0
        asm::addi(A7, X0, 93),   // 6
        asm::ecall(),            // 7
        asm::addi(A0, X0, 42),   // 8: found: a0 = 42
        asm::addi(A7, X0, 93),   // 9
        asm::ecall(),            // 10
    ]
}

fn build_mmu(code: &[u32]) -> Mmu {
    let mut mmu = Mmu::new(BASE, 0x1_0000);
    mmu.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
    let mut bytes = Vec::new();
    for w in code {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mmu.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    mmu
}

/// Run the check once against an already-built `mmu` (reused across many calls — only the code
/// pages and the `DATA_ADDR` word ever change, so re-allocating the whole guest window per case
/// would be pure overhead, not anything the real fuzzer's snapshot/reset path does either) with
/// `candidate` written to `DATA_ADDR`. Returns (exit code a0, insns retired, recorded cmp-operand
/// pairs — empty unless `record` is set).
fn run_once(mmu: &mut Mmu, candidate: u32, record: bool) -> (u32, u64, Vec<(u32, u32)>) {
    mmu.write_u32(DATA_ADDR, candidate).unwrap();
    let mut cpu = Cpu::new(BASE);
    cpu.set_cmplog(record);
    let mut exit_code = 0u32;
    for _ in 0..10_000 {
        match cpu.step(mmu).unwrap() {
            Exit::Continue => {}
            Exit::Ecall => {
                exit_code = cpu.regs[A0 as usize];
                break;
            }
            _ => break,
        }
    }
    (exit_code, cpu.insns_retired, cpu.cmplog_take())
}

/// Build a one-call, one-`Imm`-arg "program" wrapping a candidate value — enough surface for
/// `fs_prog::mutate_cmplog` to scan and substitute, standing in for a real syscall arg in the
/// shipped fuzzer's actual `Prog`/`ArgValue` model.
fn candidate_prog(candidate: u32) -> Prog {
    // Reuse a real single-Int-arg-ish description shape isn't necessary here: mutate_cmplog only
    // inspects `ArgValue` leaves paired against each call's `ArgType`s, so a minimal hand-built
    // call over any real description with a scalar arg works; `prctl` has one.
    let desc = fs_prog::SYSCALLS.iter().find(|d| d.name == "prctl").unwrap();
    let mut rng = Rng::new(1);
    let mut args = fs_prog::genr::generate_args(&mut rng, desc, &[]);
    for (i, av) in args.iter_mut().enumerate() {
        if let ArgValue::Imm(v) = av {
            *v = 5000 + i as u64; // sentinel — keep every OTHER arg unambiguous
        }
    }
    args[1] = ArgValue::Imm(candidate as u64); // the one arg under test
    Prog { calls: vec![TypedCall { desc, args }] }
}

/// (1) Correctness: enabling cmplog recording never changes what the program computes. A traced
/// run and a plain run of the identical candidate reach the identical exit code and insn count,
/// for both the "not found" and "found" candidates.
#[test]
fn cmplog_recording_is_purely_observational() {
    let mut mmu = build_mmu(&build_program());
    for candidate in [0u32, 1234, MAGIC] {
        let (code_a, insns_a, _) = run_once(&mut mmu, candidate, false);
        let (code_b, insns_b, _) = run_once(&mut mmu, candidate, true);
        assert_eq!(code_a, code_b, "candidate {candidate:#x}: cmplog changed the exit code");
        assert_eq!(insns_a, insns_b, "candidate {candidate:#x}: cmplog changed insns_retired");
    }
    // Sanity: the check is exercised in both directions (proves the harness itself is correct).
    assert_eq!(run_once(&mut mmu, 0, false).0, 0);
    assert_eq!(run_once(&mut mmu, MAGIC, false).0, 42);
}

/// (2) Efficacy — the actual deliverable. An undirected baseline enumerating `0..N` distinct
/// candidates never solves `x == MAGIC` (MAGIC is far outside that range, so this is guaranteed,
/// not merely probable — see the module doc for why that's the honest stand-in for "a real random
/// mutator overwhelmingly won't guess a 32-bit constant within any realistic case budget"). CMPLOG
/// solves it in exactly one trace execution (to observe the branch's operands) plus one confirming
/// execution of the substituted value.
#[test]
fn cmplog_solves_the_magic_value_check_that_undirected_mutation_cannot() {
    const BASELINE_TRIES: u32 = 100_000;
    let mut mmu = build_mmu(&build_program());

    // --- Baseline: N "mutation" tries, none of which is anywhere near MAGIC. ---
    let mut baseline_solved = false;
    for candidate in 0..BASELINE_TRIES {
        if run_once(&mut mmu, candidate, false).0 == 42 {
            baseline_solved = true;
            break;
        }
    }
    assert!(
        !baseline_solved,
        "baseline should never solve a magic-value check by enumerating small values"
    );

    // --- CMPLOG: trace one arbitrary starting candidate, harvest the branch's operands, and let
    // mutate_cmplog substitute the other side into the program's own data. ---
    let start_candidate = 0u32;
    let (_, _, pairs) = run_once(&mut mmu, start_candidate, true);
    assert!(!pairs.is_empty(), "the beq must have been recorded");
    assert!(
        pairs.contains(&(start_candidate, MAGIC)),
        "expected the logged pair (candidate, MAGIC); got {pairs:?}"
    );

    let base_prog = candidate_prog(start_candidate);

    // mutate_cmplog offers MAGIC and its ±1 neighbours (for </<=/>/>= checks); an exact-equality
    // check like this one needs the exact value, so try a handful of cases (distinct rng draws,
    // exactly as the real fuzz loop would across successive cases) — every draw is at worst a ±1
    // near-miss, so this converges in a handful of tries, not anywhere near the 100_000-try
    // baseline budget above (let alone 2^32).
    let mut solved_within = None;
    for seed in 1..=50u32 {
        let mut rng = Rng::new(seed);
        let mutated = mutate_cmplog(&mut rng, &base_prog, &pairs).expect("must find the match");
        let ArgValue::Imm(guessed) = mutated.calls[0].args[1] else {
            panic!("arg should stay Imm");
        };
        let guessed = guessed as u32;
        assert!(
            [MAGIC, MAGIC.wrapping_add(1), MAGIC.wrapping_sub(1)].contains(&guessed),
            "cmplog should only ever guess MAGIC (±1), got {guessed:#x}"
        );
        // Confirming run: inject the cmplog-guessed value as the real candidate and observe the
        // branch actually flip — the end-to-end proof, not just an internal data-structure check.
        if run_once(&mut mmu, guessed, false).0 == 42 {
            solved_within = Some(seed);
            break;
        }
    }
    let tries = solved_within.expect("cmplog should solve the check within a handful of cases");
    assert!(tries <= 10, "expected cmplog to solve it within ~10 cases, took {tries}");
}
