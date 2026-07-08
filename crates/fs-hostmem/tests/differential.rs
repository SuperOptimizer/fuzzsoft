//! Differential test: `HostMem` (mmap+memfd) must be byte-exact and fault-exact vs `fs_mmu::Mmu`
//! over an identical random layout, a random access stream, and reset semantics. This is the
//! correctness backbone the whole prototype depends on (see docs/cow-shared-ram.md).

use fs_hostmem::{Golden, HostMem, PERM_EXEC, PERM_RAW, PERM_READ, PERM_WRITE};
use fs_mmu::Mmu;

/// Deterministic xorshift32 (same construction as `fs-prog`'s `Rng`; not worth an extra
/// workspace-wide dependency for a test file).
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
const SIZE: usize = 128 * 1024;

/// Build a random golden layout: a mix of RWX-mapped data, RAW (uninitialized) allocations,
/// read-only data, poisoned (freed/redzone) regions, and exec-only regions.
fn random_layout(rng: &mut Rng) -> Mmu {
    let mut mmu = Mmu::new(BASE, SIZE);
    for _ in 0..200 {
        let addr = BASE + rng.below(SIZE as u32 - 64);
        let len = 1 + rng.below(64);
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        match rng.below(5) {
            0 => mmu.map(addr, &data, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap(),
            1 => mmu.protect(addr, len, PERM_RAW | PERM_WRITE).unwrap(),
            2 => mmu.map(addr, &data, PERM_READ).unwrap(),
            3 => {
                mmu.map(addr, &data, PERM_READ | PERM_WRITE).unwrap();
                mmu.poison(addr, len).unwrap();
            }
            _ => mmu.map(addr, &data, PERM_READ | PERM_WRITE).unwrap(),
        }
    }
    mmu
}

/// Run one random access against both backings and assert identical `Result` (value or exact
/// `Fault`). Addresses range a bit wider than the mapped window so `Unmapped` faults get exercised
/// too, and sizes/alignment are randomized so `Unaligned` faults get exercised.
fn run_op(rng: &mut Rng, mmu: &mut Mmu, hm: &mut HostMem) {
    let addr = BASE.wrapping_sub(64).wrapping_add(rng.below(SIZE as u32 + 128));
    match rng.below(8) {
        0 => assert_eq!(mmu.read_u8(addr), hm.read_u8(addr), "read_u8 @ {addr:#x}"),
        1 => assert_eq!(mmu.read_u16(addr), hm.read_u16(addr), "read_u16 @ {addr:#x}"),
        2 => assert_eq!(mmu.read_u32(addr), hm.read_u32(addr), "read_u32 @ {addr:#x}"),
        3 => {
            let v = rng.next() as u8;
            assert_eq!(mmu.write_u8(addr, v), hm.write_u8(addr, v), "write_u8 @ {addr:#x}");
        }
        4 => {
            let v = rng.next() as u16;
            assert_eq!(mmu.write_u16(addr, v), hm.write_u16(addr, v), "write_u16 @ {addr:#x}");
        }
        5 => {
            let v = rng.next();
            assert_eq!(mmu.write_u32(addr, v), hm.write_u32(addr, v), "write_u32 @ {addr:#x}");
        }
        6 => assert_eq!(mmu.fetch_u16(addr), hm.fetch_u16(addr), "fetch_u16 @ {addr:#x}"),
        _ => assert_eq!(mmu.fetch_u32(addr), hm.fetch_u32(addr), "fetch_u32 @ {addr:#x}"),
    }
}

#[test]
fn hostmem_matches_mmu() {
    let mut rng = Rng::new(0xC0FF_EE42);
    let mut mmu = random_layout(&mut rng);

    // Capture the golden planes *before* dirty tracking starts diverging `mmu`, so we have a
    // reference to assert exact restoration against later.
    let (gmem, gperms) = mmu.planes();
    let (gmem, gperms) = (gmem.to_vec(), gperms.to_vec());

    let golden = Golden::from_mmu(&mmu).expect("golden image creation failed");
    let mut hm = golden.new_view().expect("hostmem view creation failed");

    mmu.enable_dirty_tracking();

    for _ in 0..20_000 {
        run_op(&mut rng, &mut mmu, &mut hm);
    }

    // Reset both and assert byte-exact golden restoration.
    mmu.reset_dirty(&gmem, &gperms);
    hm.reset().expect("hostmem reset failed");

    let (m, p) = mmu.planes();
    assert_eq!(m, gmem.as_slice(), "mmu mem plane not restored to golden");
    assert_eq!(p, gperms.as_slice(), "mmu perms plane not restored to golden");
    let (m, p) = hm.planes();
    assert_eq!(m, gmem.as_slice(), "hostmem mem plane not restored to golden");
    assert_eq!(p, gperms.as_slice(), "hostmem perms plane not restored to golden");

    // Re-diverge after reset and keep comparing — proves reset leaves both backings in an
    // identical, fully-usable state, not just byte-equal-but-broken.
    for _ in 0..20_000 {
        run_op(&mut rng, &mut mmu, &mut hm);
    }

    mmu.reset_dirty(&gmem, &gperms);
    hm.reset().expect("hostmem reset (2nd) failed");
    let (m, p) = mmu.planes();
    assert_eq!(m, gmem.as_slice());
    assert_eq!(p, gperms.as_slice());
    let (m, p) = hm.planes();
    assert_eq!(m, gmem.as_slice());
    assert_eq!(p, gperms.as_slice());
}

#[test]
fn raw_upgrade_and_poison_parity() {
    let mut mmu = Mmu::new(BASE, 0x1000);
    mmu.protect(BASE, 4, PERM_RAW | PERM_WRITE).unwrap();
    mmu.protect(BASE + 4, 4, PERM_READ | PERM_WRITE).unwrap();
    mmu.write_u8(BASE + 4, 0xAA).unwrap();
    mmu.poison(BASE + 4, 4).unwrap();

    let golden = Golden::from_mmu(&mmu).unwrap();
    let mut hm = golden.new_view().unwrap();

    // RAW byte: unread until written, on both backings, then unlocked by exactly the write.
    assert_eq!(mmu.read_u8(BASE).unwrap_err(), hm.read_u8(BASE).unwrap_err());
    assert_eq!(mmu.write_u8(BASE, 0x41), hm.write_u8(BASE, 0x41));
    assert_eq!(mmu.read_u8(BASE), hm.read_u8(BASE));
    assert_eq!(mmu.read_u8(BASE).unwrap(), 0x41);

    // Poisoned region: both fault Permission (not Unmapped — still inside the window).
    assert_eq!(mmu.read_u8(BASE + 4).unwrap_err(), hm.read_u8(BASE + 4).unwrap_err());
    assert_eq!(mmu.write_u8(BASE + 4, 1).unwrap_err(), hm.write_u8(BASE + 4, 1).unwrap_err());
}
