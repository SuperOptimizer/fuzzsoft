//! SMP mechanical core proof (`docs/smp-design.md`, T5.1a): hand-written 2-hart bare-metal RV32
//! programs exercising shared memory + cross-hart LR/SC and AMO over `fs_platform::run_smp`.
//!
//! Three properties are asserted, matching the roadmap chunk's validation gate:
//!  1. A concurrent LR/SC compare-and-swap increment loop on a shared counter, driven by the
//!     round-robin scheduler, produces the SAME final counter value as a genuinely sequential
//!     reference (hart 0's whole loop, then hart 1's whole loop, back to back, no interleaving —
//!     built from the pre-existing, untouched single-hart `run_until`). If cross-hart LR/SC
//!     invalidation were missing or wrong, concurrent increments would lose updates (spurious SC
//!     successes overwriting each other) and this would NOT match.
//!  2. Running the same interleaved program twice from the same starting state produces
//!     byte-identical final state (full RAM contents + every hart's registers/pc/insns_retired/
//!     reservation) — the determinism/reproducibility guarantee the whole SMP epic depends on.
//!  3. A direct, deterministic (quantum = 1, so execution strictly alternates hart0/hart1
//!     one-instruction-at-a-time) proof that `sc.w` FAILS when the other hart writes the reserved
//!     word between this hart's `lr.w` and `sc.w` — the cross-hart invalidation mechanism itself,
//!     isolated from the statistical CAS-loop test above.

use fs_mmu::{Bus, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_platform::{Machine, Stop};
use fs_riscv::{asm::*, Cpu, A0, A7, T1, X0};

const BASE: u32 = 0x8000_0000;
const RAM_SIZE: u32 = 0x1_0000;
const COUNTER: u32 = BASE + 0x1000; // low 12 bits zero -> `lui` loads it directly
const ENTRY0: u32 = BASE;
const ENTRY1: u32 = BASE + 0x0200;

/// t1=x6 (counter addr, constant), t2=x7 (remaining-iterations loop counter), t3=x28 (old value
/// from `lr.w`), t4=x29 (candidate new value), t5=x30 (`sc.w` result). Retries the `lr.w`/`sc.w`
/// pair on failure, so this is a correct atomic increment under ANY interleaving, provided
/// cross-hart reservation invalidation works. Ends by reading the counter fresh into `a0` and
/// raising the fuzzing hypercall `eid` so the caller can tell the two harts apart on halt.
fn cas_increment_program(iters: i32, eid: i32) -> Vec<u8> {
    let prog = [
        lui(T1, COUNTER),        // 0
        addi(7, X0, iters),      // 1: t2 = iters
        beq(7, X0, 28),          // 2: if t2==0 -> idx 9 (done)
        lr_w(28, T1),            // 3: t3 = [t1]; reserve
        addi(29, 28, 1),         // 4: t4 = t3 + 1
        sc_w(30, T1, 29),        // 5: attempt [t1] = t4; t5 = 0 (ok) / 1 (fail)
        bne(30, X0, -12),        // 6: if t5 != 0 -> idx 3 (retry)
        addi(7, 7, -1),          // 7: t2 -= 1
        jal(X0, -24),            // 8: -> idx 2
        lw(A0, T1, 0),           // 9: a0 = [t1] (fresh read of shared counter)
        addi(A7, X0, eid),       // 10
        ecall(),                 // 11
    ];
    let mut bytes = Vec::new();
    for w in prog {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// Builds a fresh 2-hart `Machine` with both harts' CAS-increment programs mapped in and the
/// shared counter initialised to 0. Deterministic in its arguments alone.
fn setup(iters0: i32, iters1: i32, eid: i32) -> (Vec<Cpu>, Machine) {
    let mut m = Machine::new(BASE, RAM_SIZE);
    m.ram.protect(BASE, RAM_SIZE, PERM_READ | PERM_WRITE).unwrap();
    m.ram.map(ENTRY0, &cas_increment_program(iters0, eid), PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    m.ram.map(ENTRY1, &cas_increment_program(iters1, eid), PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let mut cpu0 = Cpu::new_hart(ENTRY0, 0);
    cpu0.hypercall_eid = Some(eid as u32);
    let mut cpu1 = Cpu::new_hart(ENTRY1, 1);
    cpu1.hypercall_eid = Some(eid as u32);
    assert_eq!(cpu0.csr.mhartid, 0);
    assert_eq!(cpu1.csr.mhartid, 1);
    (vec![cpu0, cpu1], m)
}

// `addi`'s immediate is a 12-bit signed field (-2048..2047, see `fs_riscv::asm::i_type`) — the
// hypercall eid is loaded via a plain `addi`, so it must fit in that range (unlike `COUNTER`,
// which is loaded via `lui` and can be a full upper-20-bit address).
const EID: i32 = 42;

#[test]
fn smp_cas_increment_matches_sequential_reference() {
    let iters0 = 137i32;
    let iters1 = 211i32;

    // Interleaved: both harts run concurrently over one shared `Machine` via the round-robin
    // scheduler, quantum small enough to guarantee heavy interleaving.
    let (mut cpus, mut m) = setup(iters0, iters1, EID);
    let stops = fs_platform::run_smp(&mut cpus, &mut m, 7, 2_000_000, false);
    assert!(matches!(stops[0], Stop::Hypercall(_)), "hart0 did not finish: {:?}", stops[0]);
    assert!(matches!(stops[1], Stop::Hypercall(_)), "hart1 did not finish: {:?}", stops[1]);
    let interleaved_final = m.load(COUNTER, 4).unwrap();
    assert_eq!(interleaved_final, (iters0 + iters1) as u32);

    // Sequential reference: hart 0's ENTIRE loop, then hart 1's ENTIRE loop, back to back, no
    // interleaving at all — built from the pre-existing, untouched single-hart `run_until`. This
    // is a genuine differential check, not just an arithmetic restatement of the expected total.
    let mut m_seq = Machine::new(BASE, RAM_SIZE);
    m_seq.ram.protect(BASE, RAM_SIZE, PERM_READ | PERM_WRITE).unwrap();
    m_seq
        .ram
        .map(ENTRY0, &cas_increment_program(iters0, EID), PERM_READ | PERM_WRITE | PERM_EXEC)
        .unwrap();
    m_seq
        .ram
        .map(ENTRY1, &cas_increment_program(iters1, EID), PERM_READ | PERM_WRITE | PERM_EXEC)
        .unwrap();
    let mut cpu0 = Cpu::new(ENTRY0); // plain single-hart Cpu::new — untouched constructor
    cpu0.hypercall_eid = Some(EID as u32);
    match fs_platform::run_until(&mut cpu0, &mut m_seq, 1_000_000) {
        Stop::Hypercall(_) => {}
        other => panic!("hart0 sequential run did not finish: {other:?}"),
    }
    let mut cpu1 = Cpu::new(ENTRY1);
    cpu1.hypercall_eid = Some(EID as u32);
    match fs_platform::run_until(&mut cpu1, &mut m_seq, 1_000_000) {
        Stop::Hypercall(_) => {}
        other => panic!("hart1 sequential run did not finish: {other:?}"),
    }
    let sequential_final = m_seq.load(COUNTER, 4).unwrap();
    assert_eq!(sequential_final, (iters0 + iters1) as u32);
    assert_eq!(
        interleaved_final, sequential_final,
        "concurrent CAS-increment result diverged from the sequential reference \
         (cross-hart LR/SC invalidation likely lost an update)"
    );
}

/// Extracts the observable, comparable subset of a hart's state for the determinism check below:
/// registers, pc, retired-instruction count, and outstanding LR/SC reservation. `Cpu` doesn't
/// derive `PartialEq` (it carries `Option<Vec<_>>`/`Option<Box<_>>` diagnostic fields that don't
/// need to participate), so this is the deliberate, minimal comparable projection.
fn cpu_fingerprint(cpu: &Cpu) -> (Vec<u32>, u32, u64, Option<u32>, u32, u32) {
    (cpu.regs.to_vec(), cpu.pc, cpu.insns_retired, cpu.reservation(), cpu.csr.mhartid, cpu.csr.mcause)
}

#[test]
fn smp_replay_is_byte_identical() {
    let iters0 = 137i32;
    let iters1 = 211i32;

    let (mut cpus_a, mut m_a) = setup(iters0, iters1, EID);
    let stops_a = fs_platform::run_smp(&mut cpus_a, &mut m_a, 7, 2_000_000, false);

    let (mut cpus_b, mut m_b) = setup(iters0, iters1, EID);
    let stops_b = fs_platform::run_smp(&mut cpus_b, &mut m_b, 7, 2_000_000, false);

    assert_eq!(stops_a, stops_b, "replay produced different stop reasons");
    assert_eq!(
        cpus_a.iter().map(cpu_fingerprint).collect::<Vec<_>>(),
        cpus_b.iter().map(cpu_fingerprint).collect::<Vec<_>>(),
        "replay produced different hart state"
    );
    let (mem_a, perms_a) = m_a.ram.planes();
    let (mem_b, perms_b) = m_b.ram.planes();
    assert_eq!(mem_a, mem_b, "replay produced different RAM contents");
    assert_eq!(perms_a, perms_b, "replay produced different RAM permissions");
}

/// t1=x6 (counter addr). Reserves the counter, waits one instruction (so the OTHER hart's write
/// lands strictly between this hart's `lr.w` and `sc.w` under quantum=1 round-robin), then
/// attempts `sc.w` with a candidate value of 555. Reports the SC result (0=success, 1=fail) in
/// `a0` via the hypercall.
fn lr_sc_race_hart0(eid: i32) -> Vec<u8> {
    let prog = [
        lui(T1, COUNTER),   // 0
        lr_w(28, T1),       // 1: t3 = [t1]; reserve
        addi(29, X0, 555),  // 2: t4 = 555 (candidate new value) -- no memory effect
        sc_w(30, T1, 29),   // 3: attempt [t1] = t4; t5 = SC result
        addi(A7, X0, eid),  // 4
        add(A0, X0, 30),    // 5: a0 = t5 (SC result)
        ecall(),            // 6
    ];
    let mut bytes = Vec::new();
    for w in prog {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// t1=x6 (counter addr), t2=x7 (value to store). Writes 777 straight to the shared counter with a
/// plain `sw` — no LR/SC of its own — timed (via quantum=1 round-robin) to land between hart 0's
/// `lr.w` and `sc.w`.
fn lr_sc_race_hart1(eid: i32) -> Vec<u8> {
    let prog = [
        lui(T1, COUNTER),  // 0
        addi(7, X0, 777),  // 1: t2 = 777 -- no memory effect
        sw(T1, 7, 0),      // 2: [t1] = 777 -- THE cross-hart write
        addi(A7, X0, eid), // 3
        addi(A0, X0, 999), // 4: a0 = marker
        ecall(),           // 5
    ];
    let mut bytes = Vec::new();
    for w in prog {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

#[test]
fn cross_hart_store_invalidates_reservation_sc_fails() {
    let eid = 99i32; // must fit `addi`'s 12-bit signed immediate, see the `EID` comment above
    let mut m = Machine::new(BASE, RAM_SIZE);
    m.ram.protect(BASE, RAM_SIZE, PERM_READ | PERM_WRITE).unwrap();
    m.ram.map(ENTRY0, &lr_sc_race_hart0(eid), PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    m.ram.map(ENTRY1, &lr_sc_race_hart1(eid), PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

    let mut cpu0 = Cpu::new_hart(ENTRY0, 0);
    cpu0.hypercall_eid = Some(eid as u32);
    let mut cpu1 = Cpu::new_hart(ENTRY1, 1);
    cpu1.hypercall_eid = Some(eid as u32);
    let mut cpus = vec![cpu0, cpu1];

    // quantum = 1: strict one-instruction-at-a-time alternation (hart0, hart1, hart0, hart1, ...)
    // — the exact, deterministic interleaving worked out above, not a statistical hope.
    let stops = fs_platform::run_smp(&mut cpus, &mut m, 1, 1_000, false);

    assert_eq!(stops[0], Stop::Hypercall(1), "hart0's sc.w should have FAILED (a0=1)");
    assert_eq!(stops[1], Stop::Hypercall(999), "hart1 should have completed with marker 999");
    // Memory holds hart1's write (777); hart0's SC must NOT have overwritten it with 555.
    assert_eq!(m.load(COUNTER, 4).unwrap(), 777);
}
