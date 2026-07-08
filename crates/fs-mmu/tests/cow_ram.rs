//! Differential test: `CowRam` (page-COW over a shared `Golden` image) must be byte-exact and
//! fault-exact vs `Mmu` over an identical random layout, a random access stream, and reset
//! semantics. This is PR1 of the software-COW design (see `docs/cow-shared-ram.md`) — the
//! load-bearing invariant every later step (`CowMachine`, VecSystem substrate swap, shared
//! translate/fetch) depends on. Style mirrors `fs-hostmem/tests/differential.rs`, which proves
//! the same invariant for that crate's host-mmap-backed prototype.

use fs_mmu::{
    Access, CowRam, FaultKind, Golden, Mmu, PERM_EXEC, PERM_RAW, PERM_READ, PERM_WRITE,
};
use std::sync::Arc;

/// Deterministic xorshift32 (same construction as `fs-prog`'s `Rng` / `fs-hostmem`'s test PRNG;
/// not worth an extra workspace-wide dependency for a test file).
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
// Deliberately *not* a multiple of 4096 (128 KiB - 7), so the differential stream exercises
// `Golden`'s clamped final-page tail handling, not just whole pages.
const SIZE: usize = 128 * 1024 - 7;

/// Build a random golden layout: a mix of RWX-mapped data, RAW (uninitialized) allocations,
/// read-only data, exec-only data, and poisoned (freed/redzone) regions, plus untouched gaps
/// (which are indistinguishable from poisoned — both are perm 0 inside the mapped window).
fn random_layout(rng: &mut Rng) -> Mmu {
    let mut mmu = Mmu::new(BASE, SIZE);
    for _ in 0..400 {
        let addr = BASE + rng.below((SIZE as u32).saturating_sub(96));
        let len = 1 + rng.below(96);
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        match rng.below(6) {
            0 => mmu
                .map(addr, &data, PERM_READ | PERM_WRITE | PERM_EXEC)
                .unwrap(),
            1 => mmu.protect(addr, len, PERM_RAW | PERM_WRITE).unwrap(),
            2 => mmu.map(addr, &data, PERM_READ).unwrap(),
            3 => {
                mmu.map(addr, &data, PERM_READ | PERM_WRITE).unwrap();
                mmu.poison(addr, len).unwrap();
            }
            4 => mmu.map(addr, &data, PERM_EXEC).unwrap(),
            _ => mmu.map(addr, &data, PERM_READ | PERM_WRITE).unwrap(),
        }
    }
    mmu
}

/// Run one random access against both backings and assert identical `Result` (value or exact
/// `Fault`). Addresses range a bit wider than the mapped window so `Unmapped` faults get
/// exercised too, and sizes/alignment are randomized so `Unaligned` faults get exercised.
fn run_op(rng: &mut Rng, mmu: &mut Mmu, cow: &mut CowRam) {
    let addr = BASE
        .wrapping_sub(64)
        .wrapping_add(rng.below(SIZE as u32 + 128));
    match rng.below(8) {
        0 => assert_eq!(mmu.read_u8(addr), cow.read_u8(addr), "read_u8 @ {addr:#x}"),
        1 => assert_eq!(
            mmu.read_u16(addr),
            cow.read_u16(addr),
            "read_u16 @ {addr:#x}"
        ),
        2 => assert_eq!(
            mmu.read_u32(addr),
            cow.read_u32(addr),
            "read_u32 @ {addr:#x}"
        ),
        3 => {
            let v = rng.next() as u8;
            assert_eq!(
                mmu.write_u8(addr, v),
                cow.write_u8(addr, v),
                "write_u8 @ {addr:#x}"
            );
        }
        4 => {
            let v = rng.next() as u16;
            assert_eq!(
                mmu.write_u16(addr, v),
                cow.write_u16(addr, v),
                "write_u16 @ {addr:#x}"
            );
        }
        5 => {
            let v = rng.next();
            assert_eq!(
                mmu.write_u32(addr, v),
                cow.write_u32(addr, v),
                "write_u32 @ {addr:#x}"
            );
        }
        6 => assert_eq!(
            mmu.fetch_u16(addr),
            cow.fetch_u16(addr),
            "fetch_u16 @ {addr:#x}"
        ),
        _ => assert_eq!(
            mmu.fetch_u32(addr),
            cow.fetch_u32(addr),
            "fetch_u32 @ {addr:#x}"
        ),
    }
    // Perm-plane parity too (catches a content/perm desync a value-only comparison would miss).
    assert_eq!(mmu.perm_at(addr), cow.perm_at(addr), "perm_at @ {addr:#x}");
}

/// Full differential run for one seed: random layout -> capture golden -> 20k-op stream ->
/// reset + byte-exact golden restoration -> re-diverge -> 20k more ops -> reset again.
fn run_seed(seed: u32, ops_per_phase: u32) {
    let mut rng = Rng::new(seed);
    let mut mmu = random_layout(&mut rng);

    // Capture the golden planes *before* dirty tracking starts diverging `mmu`, so we have an
    // independent reference to assert exact restoration against later.
    let (gmem, gperms) = mmu.planes();
    let (gmem, gperms) = (gmem.to_vec(), gperms.to_vec());

    let golden = Arc::new(Golden::from_mmu(&mmu));
    assert_eq!(golden.base(), BASE);
    assert_eq!(golden.size(), SIZE);
    assert_eq!(golden.num_pages(), SIZE.div_ceil(4096));

    let mut cow = CowRam::new(golden);

    mmu.enable_dirty_tracking();

    for _ in 0..ops_per_phase {
        run_op(&mut rng, &mut mmu, &mut cow);
    }

    // Reset both and assert byte-exact golden restoration.
    mmu.reset_dirty(&gmem, &gperms);
    cow.reset();
    assert!(cow.dirty_pages().is_empty(), "seed {seed}: reset left dirty pages");

    let (m, p) = mmu.planes();
    assert_eq!(m, gmem.as_slice(), "seed {seed}: mmu mem plane not restored to golden");
    assert_eq!(p, gperms.as_slice(), "seed {seed}: mmu perms plane not restored to golden");
    assert_golden_bytewise(&cow, &gmem, &gperms, seed, "after first reset");

    // Re-diverge after reset and keep comparing — proves reset leaves both backings in an
    // identical, fully-usable state, not just byte-equal-but-broken (re-COW works).
    for _ in 0..ops_per_phase {
        run_op(&mut rng, &mut mmu, &mut cow);
    }

    mmu.reset_dirty(&gmem, &gperms);
    cow.reset();
    assert!(cow.dirty_pages().is_empty(), "seed {seed}: second reset left dirty pages");
    let (m, p) = mmu.planes();
    assert_eq!(m, gmem.as_slice());
    assert_eq!(p, gperms.as_slice());
    assert_golden_bytewise(&cow, &gmem, &gperms, seed, "after second reset");
}

/// Walk every byte of the window through `CowRam`'s checked API and assert it matches the golden
/// planes exactly (content where readable, and the perm byte always). Bypasses `read`'s own
/// permission gate for content by only comparing content when golden itself carries PERM_READ
/// (otherwise `cow.read_u8` would legitimately fault, same as `Mmu` would).
fn assert_golden_bytewise(cow: &CowRam, gmem: &[u8], gperms: &[u8], seed: u32, when: &str) {
    for i in 0..SIZE {
        let addr = BASE.wrapping_add(i as u32);
        assert_eq!(
            cow.perm_at(addr),
            Some(gperms[i]),
            "seed {seed} {when}: perm mismatch @ {addr:#x}"
        );
        if gperms[i] & PERM_READ != 0 {
            assert_eq!(
                cow.read_u8(addr),
                Ok(gmem[i]),
                "seed {seed} {when}: content mismatch @ {addr:#x}"
            );
        }
    }
}

#[test]
fn cow_ram_is_byte_exact_vs_mmu() {
    // Several independent seeds, 20k ops per phase (40k total per seed across the two phases) —
    // matches `fs-hostmem`'s differential test scale for the analogous prototype.
    for seed in [0xC0FF_EE42u32, 1, 2, 42, 0xDEAD_BEEF, 0x1234_5678, 7] {
        run_seed(seed, 20_000);
    }
}

#[test]
fn raw_upgrade_and_poison_parity() {
    let mut mmu = Mmu::new(BASE, 0x1000);
    mmu.protect(BASE, 4, PERM_RAW | PERM_WRITE).unwrap();
    mmu.protect(BASE + 4, 4, PERM_READ | PERM_WRITE).unwrap();
    mmu.write_u8(BASE + 4, 0xAA).unwrap();
    mmu.poison(BASE + 4, 4).unwrap();

    let golden = Arc::new(Golden::from_mmu(&mmu));
    let mut cow = CowRam::new(golden);

    // RAW byte: unread until written, on both backings, then unlocked by exactly the write.
    assert_eq!(mmu.read_u8(BASE).unwrap_err(), cow.read_u8(BASE).unwrap_err());
    assert_eq!(mmu.write_u8(BASE, 0x41), cow.write_u8(BASE, 0x41));
    assert_eq!(mmu.read_u8(BASE), cow.read_u8(BASE));
    assert_eq!(mmu.read_u8(BASE).unwrap(), 0x41);
    assert_eq!(cow.read_u8(BASE).unwrap(), 0x41);

    // Poisoned region: both fault Permission (not Unmapped — still inside the window).
    assert_eq!(
        mmu.read_u8(BASE + 4).unwrap_err(),
        cow.read_u8(BASE + 4).unwrap_err()
    );
    assert_eq!(
        mmu.write_u8(BASE + 4, 1).unwrap_err(),
        cow.write_u8(BASE + 4, 1).unwrap_err()
    );
    assert_eq!(mmu.read_u8(BASE + 4).unwrap_err().kind, FaultKind::Permission);
}

#[test]
fn unmapped_and_unaligned_parity() {
    let mmu = Mmu::new(BASE, 0x1000);
    let golden = Arc::new(Golden::from_mmu(&mmu));
    let cow = CowRam::new(golden);

    assert_eq!(mmu.read_u8(0x1234).unwrap_err(), cow.read_u8(0x1234).unwrap_err());
    assert_eq!(mmu.read_u8(0x1234).unwrap_err().kind, FaultKind::Unmapped);
    assert_eq!(
        mmu.read_u32(BASE + 1).unwrap_err(),
        cow.read_u32(BASE + 1).unwrap_err()
    );
    assert_eq!(mmu.read_u32(BASE + 1).unwrap_err().kind, FaultKind::Unaligned);
    assert_eq!(
        mmu.fetch_u16(BASE + 1).unwrap_err(),
        cow.fetch_u16(BASE + 1).unwrap_err()
    );
}

/// Bus-shaped `load`/`store`/`ifetch16` helpers (added for PR2's `CowMachine`) must agree with
/// `Mmu`'s `Bus` impl for the same sizes/addresses — they're thin wrappers over the same
/// checked read/write/fetch API, but this pins the exact call shape down.
#[test]
fn bus_shaped_helpers_match_mmu_bus_impl() {
    use fs_mmu::Bus;

    let mut mmu = Mmu::new(BASE, 0x1000);
    mmu.map(BASE, &[1, 2, 3, 4], PERM_READ | PERM_WRITE | PERM_EXEC)
        .unwrap();
    let golden = Arc::new(Golden::from_mmu(&mmu));
    let mut cow = CowRam::new(golden);

    for &size in &[1u8, 2, 4] {
        assert_eq!(
            Bus::load(&mut mmu, BASE, size),
            cow.load(BASE, size),
            "load size {size}"
        );
    }
    assert_eq!(Bus::ifetch16(&mut mmu, BASE), cow.ifetch16(BASE));

    assert_eq!(
        Bus::store(&mut mmu, BASE, 4, 0xDEAD_BEEF),
        cow.store(BASE, 4, 0xDEAD_BEEF)
    );
    assert_eq!(Bus::load(&mut mmu, BASE, 4), cow.load(BASE, 4));
    assert_eq!(Bus::load(&mut mmu, BASE, 4).unwrap(), 0xDEAD_BEEF);

    // Out-of-bounds / unmapped parity too.
    assert_eq!(Bus::load(&mut mmu, 0x1234, 4), cow.load(0x1234, 4));
    assert_eq!(
        Bus::load(&mut mmu, 0x1234, 4).unwrap_err().access,
        Access::Read
    );
}
