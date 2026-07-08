//! Differential test: `CowMachine` (page-COW `CowRam` over a shared `Golden` image, PR2 of the
//! software-COW design, see `docs/cow-shared-ram.md`) must route an identical RAM/CLINT/UART
//! access stream to identical results as `Machine` (owned `Mmu`) — both driven purely through
//! `&mut dyn Bus`, since that's the whole point: the audited scalar `fs_riscv::Cpu::step_system`
//! sees no difference between the two. Style mirrors `fs-mmu/tests/cow_ram.rs`, which proves the
//! analogous `CowRam` vs `Mmu` invariant one layer down.

use fs_mmu::{Bus, FaultKind, Golden, PERM_EXEC, PERM_RAW, PERM_READ, PERM_WRITE};
use fs_platform::{CowMachine, Machine, CLINT_BASE, UART_BASE};
use std::sync::Arc;

/// Deterministic xorshift32 (same construction as `fs-mmu/tests/cow_ram.rs`'s `Rng`).
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed.max(1))
    }
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

const BASE: u32 = 0x8000_0000;
// Deliberately not page-aligned, to exercise `Golden`'s clamped final-page tail handling too.
const SIZE: u32 = 96 * 1024 - 7;

/// Build a random RAM layout on a fresh `Machine` (mirrors `fs-mmu/tests/cow_ram.rs`'s
/// `random_layout`, at the `Machine`/`Mmu` level instead of bare `Mmu`). Deterministic in `seed`
/// alone, so calling this twice with the same seed reproduces byte-identical RAM.
fn random_layout(seed: u32) -> Machine {
    let mut rng = Rng::new(seed);
    let mut m = Machine::new(BASE, SIZE);
    for _ in 0..300 {
        let addr = BASE + rng.below(SIZE.saturating_sub(96));
        let len = 1 + rng.below(96);
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        match rng.below(6) {
            0 => m
                .ram
                .map(addr, &data, PERM_READ | PERM_WRITE | PERM_EXEC)
                .unwrap(),
            1 => m.ram.protect(addr, len, PERM_RAW | PERM_WRITE).unwrap(),
            2 => m.ram.map(addr, &data, PERM_READ).unwrap(),
            3 => {
                m.ram.map(addr, &data, PERM_READ | PERM_WRITE).unwrap();
                m.ram.poison(addr, len).unwrap();
            }
            4 => m.ram.map(addr, &data, PERM_EXEC).unwrap(),
            _ => m.ram.map(addr, &data, PERM_READ | PERM_WRITE).unwrap(),
        }
    }
    m
}

/// One random `Bus` op, applied identically to both machines (as `&mut dyn Bus`, exactly how
/// `fs_riscv::Cpu::step_system` would see them), asserting identical `Result` on every access.
/// Addresses cover RAM (in- and out-of-bounds), CLINT (0x0200_xxxx), and UART (0x1000_0000).
fn run_op(rng: &mut Rng, a: &mut dyn Bus, b: &mut dyn Bus) {
    // Bias address selection across the three regions plus some genuinely unmapped space.
    let addr = match rng.below(4) {
        0 => BASE.wrapping_sub(64).wrapping_add(rng.below(SIZE + 128)), // RAM window +- slop
        1 => CLINT_BASE + rng.below(0x1_0000),                          // CLINT window
        2 => UART_BASE + rng.below(0x100),                              // UART window
        _ => rng.next(),                                                // anything at all
    };
    match rng.below(7) {
        0 => assert_eq!(a.load(addr, 1), b.load(addr, 1), "load1 @ {addr:#x}"),
        1 => assert_eq!(a.load(addr, 2), b.load(addr, 2), "load2 @ {addr:#x}"),
        2 => assert_eq!(a.load(addr, 4), b.load(addr, 4), "load4 @ {addr:#x}"),
        3 => {
            let v = rng.next();
            assert_eq!(a.store(addr, 1, v), b.store(addr, 1, v), "store1 @ {addr:#x}");
        }
        4 => {
            let v = rng.next();
            assert_eq!(a.store(addr, 2, v), b.store(addr, 2, v), "store2 @ {addr:#x}");
        }
        5 => {
            let v = rng.next();
            assert_eq!(a.store(addr, 4, v), b.store(addr, 4, v), "store4 @ {addr:#x}");
        }
        _ => assert_eq!(a.ifetch16(addr), b.ifetch16(addr), "ifetch16 @ {addr:#x}"),
    }
}

/// Full differential run for one seed: identical random layout -> capture golden -> N-op random
/// `Bus` stream against both `&mut dyn Bus` -> CLINT/UART parity -> reset parity.
fn run_seed(seed: u32, ops: u32) {
    let mut rng = Rng::new(seed ^ 0x5bd1_e995);
    let mut machine = random_layout(seed);
    let golden = Arc::new(Golden::from_mmu(&machine.ram));
    let mut cow = CowMachine::from_golden(golden, BASE, SIZE);

    for _ in 0..ops {
        run_op(&mut rng, &mut machine, &mut cow);
    }

    // CLINT/UART state must match bit-for-bit after an identical op stream (RAM parity was
    // already asserted op-by-op above, via identical `load`/`store`/`ifetch16` results).
    assert_eq!(machine.clint.msip, cow.clint.msip, "seed {seed}: clint.msip diverged");
    assert_eq!(machine.clint.mtimecmp, cow.clint.mtimecmp, "seed {seed}: clint.mtimecmp diverged");
    assert_eq!(machine.uart.out, cow.uart.out, "seed {seed}: uart output diverged");

    // Reset parity: `cow.reset_case()` must revert every RAM byte+perm back to golden, i.e. back
    // to the *original* (pre-op-stream) layout — reconstructed here (deterministically, from the
    // same seed) as an independent reference, since `Machine` has no built-in "revert to golden"
    // (that's `Snapshot`'s job, one layer up).
    cow.reset_case();
    assert!(cow.ram.dirty_pages().is_empty(), "seed {seed}: reset left dirty pages");
    let original = random_layout(seed);
    for pn in 0..(SIZE as usize).div_ceil(fs_mmu::PAGE_SIZE) {
        let page_addr = BASE.wrapping_add((pn * fs_mmu::PAGE_SIZE) as u32);
        for off in 0..fs_mmu::PAGE_SIZE as u32 {
            let a = page_addr.wrapping_add(off);
            if !original.ram.in_bounds(a, 1) {
                continue;
            }
            assert_eq!(
                original.ram.perm_at(a),
                cow.ram.perm_at(a),
                "seed {seed}: perm mismatch after reset @ {a:#x}"
            );
            // Compare actual content wherever the byte is readable in the original layout
            // (unreadable bytes have no defined content in either backing).
            if original.ram.perm_at(a).unwrap_or(0) & PERM_READ != 0 {
                assert_eq!(
                    original.ram.read_u8(a).ok(),
                    cow.ram.read_u8(a).ok(),
                    "seed {seed}: content mismatch after reset @ {a:#x}"
                );
            }
        }
    }
}

#[test]
fn cow_machine_matches_machine() {
    for seed in [1u32, 2, 42, 1337, 0xdead_beef, 0x1234_5678] {
        run_seed(seed, 20_000);
    }
}

#[test]
fn cow_machine_unmapped_and_unaligned_faults_match() {
    let mut machine = Machine::new(BASE, 0x1000);
    let golden = Arc::new(Golden::from_mmu(&machine.ram));
    let mut cow = CowMachine::from_golden(golden, BASE, 0x1000);

    // Out-of-bounds RAM.
    assert_eq!(machine.load(0x1234, 1).unwrap_err().kind, FaultKind::Unmapped);
    assert_eq!(cow.load(0x1234, 1).unwrap_err().kind, FaultKind::Unmapped);

    // Unaligned in-bounds RAM.
    machine.ram.protect(BASE, 0x10, PERM_READ | PERM_WRITE).unwrap();
    assert_eq!(machine.load(BASE + 1, 4).unwrap_err().kind, FaultKind::Unaligned);
    assert_eq!(cow.load(BASE + 1, 4).unwrap_err().kind, FaultKind::Unaligned);

    // CLINT roundtrip identical on both.
    machine.store(CLINT_BASE + 0x4000, 4, 0xdead_beef).unwrap();
    cow.store(CLINT_BASE + 0x4000, 4, 0xdead_beef).unwrap();
    assert_eq!(machine.load(CLINT_BASE + 0x4000, 4), cow.load(CLINT_BASE + 0x4000, 4));

    // UART writes identical on both.
    machine.store(UART_BASE, 1, b'A' as u32).unwrap();
    cow.store(UART_BASE, 1, b'A' as u32).unwrap();
    assert_eq!(machine.uart.out, cow.uart.out);

    // Exec fault (no PERM_EXEC on the mapped-but-RW-only region) must match too.
    assert_eq!(machine.ifetch16(BASE).unwrap_err().kind, FaultKind::Permission);
    assert_eq!(cow.ifetch16(BASE).unwrap_err().kind, FaultKind::Permission);
}

#[test]
fn cow_machine_from_machine_matches_from_golden() {
    let machine = random_layout(7);
    let mut a = CowMachine::from_machine(&machine);
    let mut b = CowMachine::from_golden(Arc::new(Golden::from_mmu(&machine.ram)), BASE, SIZE);

    let mut rng = Rng::new(99);
    for _ in 0..5_000 {
        let addr = BASE.wrapping_sub(32).wrapping_add(rng.below(SIZE + 64));
        match rng.below(2) {
            0 => assert_eq!(a.load(addr, 1), b.load(addr, 1), "load @ {addr:#x}"),
            _ => {
                let v = rng.next();
                assert_eq!(a.store(addr, 1, v), b.store(addr, 1, v), "store @ {addr:#x}");
            }
        }
    }
}
