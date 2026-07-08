//! Redzone allocator model + free-quarantine, built entirely out of soft-MMU permission bits.
//!
//! There is no separate "shadow state" here beyond what [`fs_mmu::Mmu`] already offers: a live
//! allocation is a payload region stamped `WRITE|RAW` (so the existing uninitialized-read oracle
//! applies for free) flanked by poisoned guard bytes; a freed allocation is the same region fully
//! poisoned. All three bug classes fall out of faults the MMU already produces:
//!
//! - **OOB read/write** -> the access lands on a poisoned guard byte -> `FaultKind::Permission`.
//! - **Use-after-free** -> the access lands on a poisoned (quarantined) payload byte -> same fault.
//! - **Uninitialized read** -> the byte is still `RAW` (never written) -> same fault, pre-existing.
//!
//! The sanitizer's own job is bookkeeping: remembering which addresses are live (so `free` can be
//! validated and `alloc` can size the guard region), and remembering which addresses are
//! quarantined (so a UAF is distinguishable, in principle, from a wild pointer — see
//! `DESIGN.md`).

use std::collections::{HashMap, VecDeque};

use fs_mmu::{Fault, Mmu, PERM_RAW, PERM_WRITE};

/// Default guard width on each side of a payload, in bytes. Falk's bump allocator (architecture
/// §3) uses single-byte guard holes; we default wider (still tiny) since a soft-MMU guard is just
/// permission bits, not backing memory, so there is no cost to a few extra poisoned bytes and it
/// catches more than 1-byte overflows.
pub const DEFAULT_REDZONE: u32 = 16;

/// How many freed allocations to remember before we stop tracking the oldest ones. This bounds
/// the sanitizer's own metadata, mirroring the "bounded dirty list" discipline in decision #11 —
/// unbounded fuzzing must not have unbounded host-side bookkeeping. Eviction only forgets our
/// *tracking* of an address; the bytes stay poisoned (quarantined) either way, so UAF detection
/// is never weakened by eviction — only the "was this specifically a freed pointer" classification
/// is lost for the oldest entries, falling back to the plain "wild/unmapped-perm access" fault.
pub const DEFAULT_QUARANTINE_CAP: usize = 4096;

/// Bookkeeping for one live allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LiveAlloc {
    /// How many bytes starting at the allocation's address `free()` poisons. For a classic
    /// [`Sanitizer::alloc`] this is the payload size; for [`Sanitizer::alloc_with_slack`] this is
    /// the *whole bucket* (payload + slack), so freeing a slack-style allocation poisons the
    /// entire slot, including any slack a `ksize()` re-open had opened back up.
    size: u32,
    /// Cross-object guard width used by [`Sanitizer::alloc`]. Always `0` for
    /// [`Sanitizer::alloc_with_slack`] allocations — see that method's doc comment for why a
    /// cross-object guard is never placed there.
    redzone: u32,
    /// `Some((req_size, bucket_size))` for a [`Sanitizer::alloc_with_slack`] allocation, so a
    /// later `ksize()`/`krealloc()` hook can find and re-open exactly `[addr+req_size,
    /// addr+bucket_size)`. `None` for a classic [`Sanitizer::alloc`] allocation, which has no
    /// slack concept.
    slack: Option<(u32, u32)>,
}

/// Something went wrong at the sanitizer policy level (as opposed to a plain MMU [`Fault`], which
/// is passed through unchanged since it's already a perfectly good bug signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SanError {
    /// `alloc()` was called with an address that is already live. Either the guest allocator
    /// double-allocated (a guest bug) or the sanitizer's hook/hypercall configuration is wrong
    /// (e.g. hooked the wrong return-value register). Reported rather than silently overwritten
    /// so both cases are visible.
    DoubleAlloc { addr: u32 },
    /// `free()` was called on an address that is not currently live: a double-free, a free of a
    /// pointer that was never allocated (or already reported to us), or a freed-then-freed-again
    /// path. This is itself a real bug class the sanitizer should report to the fuzzer.
    InvalidFree { addr: u32 },
    /// [`Sanitizer::reopen_slack`] (the `ksize()`/`krealloc()` hook primitive) was called on an
    /// address that is not currently live. Like `InvalidFree`, this means either a wild/unknown
    /// pointer reached the hook, or the hook/register wiring upstream is wrong — surfaced rather
    /// than silently ignored.
    UnknownPointer { addr: u32 },
    /// The MMU rejected a poison/stamp operation, almost always because the reported
    /// `(addr, size)` falls outside the mapped guest window — a strong signal that whatever fed
    /// us this address/size pair (hypercall args or a PC-hooked register) is wrong.
    Mmu(Fault),
}

impl From<Fault> for SanError {
    fn from(f: Fault) -> Self {
        SanError::Mmu(f)
    }
}

impl std::fmt::Display for SanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SanError::DoubleAlloc { addr } => {
                write!(f, "sanitizer: alloc() on already-live address {addr:#010x}")
            }
            SanError::InvalidFree { addr } => write!(
                f,
                "sanitizer: free() on non-live address {addr:#010x} (double-free or wild pointer)"
            ),
            SanError::UnknownPointer { addr } => write!(
                f,
                "sanitizer: reopen_slack() on non-live address {addr:#010x} (wild pointer or bad hook wiring)"
            ),
            SanError::Mmu(fault) => write!(f, "sanitizer: {fault}"),
        }
    }
}

impl std::error::Error for SanError {}

/// The sanitizer policy layer: redzone allocation + free-quarantine over a [`fs_mmu::Mmu`].
///
/// Carries no reference to the `Mmu` itself — every method takes `&mut Mmu` explicitly, so the
/// sanitizer's bookkeeping (which addresses are live/quarantined) lives independently of, and can
/// be snapshotted/reset alongside or separately from, the guest memory it annotates.
pub struct Sanitizer {
    redzone: u32,
    quarantine_cap: usize,
    live: HashMap<u32, LiveAlloc>,
    /// FIFO of quarantined addresses, oldest first, so we know which to evict once
    /// `quarantine_cap` is exceeded. `quarantined` is the lookup side of the same set.
    quarantine_order: VecDeque<u32>,
    quarantined: HashMap<u32, LiveAlloc>,
}

impl Default for Sanitizer {
    fn default() -> Self {
        Self::new(DEFAULT_REDZONE)
    }
}

impl Sanitizer {
    /// A sanitizer with `redzone` guard bytes on each side of every allocation and the default
    /// quarantine bound.
    pub fn new(redzone: u32) -> Self {
        Self::with_quarantine_cap(redzone, DEFAULT_QUARANTINE_CAP)
    }

    /// Full control over both knobs (mainly for tests that want a tiny `quarantine_cap`).
    pub fn with_quarantine_cap(redzone: u32, quarantine_cap: usize) -> Self {
        Self {
            redzone,
            quarantine_cap,
            live: HashMap::new(),
            quarantine_order: VecDeque::new(),
            quarantined: HashMap::new(),
        }
    }

    /// Guard width in bytes on each side of a payload.
    pub fn redzone_len(&self) -> u32 {
        self.redzone
    }

    /// True if `addr` is a currently-live allocation's base address.
    pub fn is_live(&self, addr: u32) -> bool {
        self.live.contains_key(&addr)
    }

    /// Payload size of a live allocation at `addr`, if any.
    pub fn live_size(&self, addr: u32) -> Option<u32> {
        self.live.get(&addr).map(|a| a.size)
    }

    /// True if `addr` is a freed allocation still sitting in quarantine (poisoned, not yet
    /// reused). Note that quarantine bytes stay poisoned even after eviction from *tracking*
    /// (see `DEFAULT_QUARANTINE_CAP`); this only reflects whether we still recognize `addr`
    /// specifically as a freed pointer.
    pub fn is_quarantined(&self, addr: u32) -> bool {
        self.quarantined.contains_key(&addr)
    }

    /// Record a new allocation of `size` payload bytes at `addr` (the guest allocator's own
    /// bump/slab cursor decided the address; we never allocate address space ourselves — see
    /// `DESIGN.md` for why that split is the right one for an uninstrumented guest).
    ///
    /// Effects:
    /// - Guard bytes `[addr-redzone, addr)` and `[addr+size, addr+size+redzone)` are poisoned
    ///   (best-effort: guard bytes that would fall outside the mapped window are skipped rather
    ///   than erroring, since a payload placed near the edge of guest RAM is legal).
    /// - Payload bytes `[addr, addr+size)` are stamped `WRITE|RAW` (no `READ`): the existing
    ///   RAW oracle now applies to this allocation, so a read before the first write to a given
    ///   byte faults `Permission` — the uninitialized-read case.
    /// - If `addr` was in quarantine, it is evicted from quarantine (this models a real
    ///   allocator's slab/bump cursor reusing a freed address after some churn).
    pub fn alloc(&mut self, mmu: &mut Mmu, addr: u32, size: u32) -> Result<(), SanError> {
        if self.live.contains_key(&addr) {
            return Err(SanError::DoubleAlloc { addr });
        }

        self.poison_guard_before(mmu, addr)?;
        self.poison_guard_after(mmu, addr, size)?;
        mmu.protect(addr, size, PERM_WRITE | PERM_RAW)?;

        self.evict_quarantine(&addr);
        self.live.insert(
            addr,
            LiveAlloc {
                size,
                redzone: self.redzone,
                slack: None,
            },
        );
        Ok(())
    }

    /// Record a new **slack-only** allocation: `req_size` live payload bytes at `addr`, rounded
    /// up by the allocator to a `bucket_size`-byte slot (e.g. SLUB's kmalloc bucket rounding via
    /// `linux::kmalloc_bucket`). This is the correct, zero-false-positive alternative to
    /// [`Sanitizer::alloc`]'s cross-object redzone for a *packed* allocator (see
    /// `docs/emulator-sanitizers.md`'s KASAN section): stock SLUB packs objects with zero gap, so
    /// any guard byte placed past `addr+bucket_size` is the first byte of a live neighbor, not
    /// slack — poisoning it is a guaranteed false positive on real kernel heap traffic.
    ///
    /// Effects:
    /// - `[addr, addr+req_size)` (the live object) is stamped `WRITE | RAW`, exactly like
    ///   [`Sanitizer::alloc`]'s payload handling — the existing uninitialized-read oracle applies
    ///   unchanged.
    /// - `[addr+req_size, addr+bucket_size)` — the object's own rounding slack, provably *inside
    ///   the same slot* and therefore never a neighbor — is poisoned no-access. When
    ///   `req_size == bucket_size` (an exact-fit allocation) this range is empty and nothing is
    ///   poisoned.
    /// - **No cross-object guard is placed past `addr+bucket_size`.** This is the load-bearing
    ///   difference from `alloc()`: that guard is exactly the false-positive mechanism
    ///   `docs/kernel-san.md`'s experiment measured at ~40% on stock SLUB. The honest
    ///   consequence: a write that overruns `addr+bucket_size` into a packed neighbor is *not*
    ///   caught by this sanitizer — only a kernel-cooperative oracle (`SLUB_DEBUG_ON`) can catch
    ///   that class. See `docs/emulator-sanitizers.md`.
    /// - As with `alloc()`, `addr` is evicted from quarantine if it was there (a real allocator
    ///   handing the same freed address back out is routine, not suspicious).
    pub fn alloc_with_slack(
        &mut self,
        mmu: &mut Mmu,
        addr: u32,
        req_size: u32,
        bucket_size: u32,
    ) -> Result<(), SanError> {
        if self.live.contains_key(&addr) {
            return Err(SanError::DoubleAlloc { addr });
        }
        // Defensive: a bucket size can never be smaller than what was actually requested. If it
        // somehow is (a caller passing raw/unrounded sizes), fall back to "no slack" rather than
        // poisoning bytes inside the live object.
        let bucket_size = bucket_size.max(req_size);

        mmu.protect(addr, req_size, PERM_WRITE | PERM_RAW)?;
        Self::poison_slack(mmu, addr, req_size, bucket_size)?;

        self.evict_quarantine(&addr);
        self.live.insert(
            addr,
            LiveAlloc {
                size: bucket_size,
                redzone: 0,
                slack: Some((req_size, bucket_size)),
            },
        );
        Ok(())
    }

    /// Re-open (un-poison back to `WRITE | RAW`) the rounding slack of a live
    /// [`Sanitizer::alloc_with_slack`] allocation at `addr` — the `ksize()`/`krealloc()` hook
    /// primitive from `docs/emulator-sanitizers.md`. Call this when the guest kernel legitimately
    /// queries or grows into the usable size of an allocation (`ksize()`, `krealloc()` in place,
    /// `kmalloc_size_roundup()`-then-populate), so that access doesn't fault against slack that
    /// was poisoned purely as a not-yet-declared-usable placeholder.
    ///
    /// A no-op (`Ok(())`) if `addr` is live but was allocated via plain [`Sanitizer::alloc`] (no
    /// slack concept applies) or if it has no slack to reopen (an exact-fit allocation). An error
    /// if `addr` is not currently live at all — almost always a wild pointer or a hook wired to
    /// the wrong register, exactly as `free()`'s `InvalidFree` reasoning.
    ///
    /// Re-opened slack is re-poisoned on the next `free()` of this allocation (see `LiveAlloc`'s
    /// `size` field doc comment) — a `ksize()`-driven re-open never outlives the allocation it
    /// belongs to.
    pub fn reopen_slack(&mut self, mmu: &mut Mmu, addr: u32) -> Result<(), SanError> {
        let Some(a) = self.live.get(&addr) else {
            return Err(SanError::UnknownPointer { addr });
        };
        let Some((req_size, bucket_size)) = a.slack else {
            return Ok(());
        };
        let slack_len = bucket_size - req_size;
        if slack_len == 0 {
            return Ok(());
        }
        let Some(slack_start) = addr.checked_add(req_size) else {
            return Ok(());
        };
        mmu.protect(slack_start, slack_len, PERM_WRITE | PERM_RAW)?;
        Ok(())
    }

    /// Poison exactly `[addr+req_size, addr+bucket_size)` (best-effort: skipped if it would fall
    /// outside the mapped window, mirroring `poison_guard_before`/`after`'s discipline). Shared by
    /// `alloc_with_slack` and left as a free function on `Self` (no `&self`/`&mut self` state
    /// needed) so its bounds logic is exercised identically wherever slack needs poisoning.
    fn poison_slack(mmu: &mut Mmu, addr: u32, req_size: u32, bucket_size: u32) -> Result<(), SanError> {
        let slack_len = bucket_size - req_size;
        if slack_len == 0 {
            return Ok(());
        }
        let Some(slack_start) = addr.checked_add(req_size) else {
            return Ok(());
        };
        if mmu.in_bounds(slack_start, slack_len) {
            mmu.poison(slack_start, slack_len)?;
        }
        Ok(())
    }

    /// Free the live allocation at `addr`: poison the payload (guards were already poisoned and
    /// stay that way) and move it into quarantine so a subsequent access — before the address is
    /// ever reused by `alloc` — faults as use-after-free instead of silently succeeding.
    pub fn free(&mut self, mmu: &mut Mmu, addr: u32) -> Result<(), SanError> {
        let Some(a) = self.live.remove(&addr) else {
            return Err(SanError::InvalidFree { addr });
        };

        mmu.poison(addr, a.size)?;

        if self.quarantine_order.len() >= self.quarantine_cap
            && let Some(oldest) = self.quarantine_order.pop_front()
        {
            self.quarantined.remove(&oldest);
        }
        self.quarantine_order.push_back(addr);
        self.quarantined.insert(addr, a);
        Ok(())
    }

    fn evict_quarantine(&mut self, addr: &u32) {
        if self.quarantined.remove(addr).is_some() {
            self.quarantine_order.retain(|a| a != addr);
        }
    }

    fn poison_guard_before(&self, mmu: &mut Mmu, addr: u32) -> Result<(), SanError> {
        if self.redzone == 0 {
            return Ok(());
        }
        let start = addr.wrapping_sub(self.redzone);
        // Best-effort: skip the guard if it would run off the front of the mapped window
        // (e.g. an allocation placed at the very base of guest RAM).
        if start <= addr && mmu.in_bounds(start, self.redzone) {
            mmu.poison(start, self.redzone)?;
        }
        Ok(())
    }

    fn poison_guard_after(&self, mmu: &mut Mmu, addr: u32, size: u32) -> Result<(), SanError> {
        if self.redzone == 0 {
            return Ok(());
        }
        let Some(start) = addr.checked_add(size) else {
            return Ok(());
        };
        if mmu.in_bounds(start, self.redzone) {
            mmu.poison(start, self.redzone)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::FaultKind;

    fn mmu() -> Mmu {
        Mmu::new(0x8000_0000, 0x10000)
    }

    #[test]
    fn in_bounds_access_ok() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_1000;
        san.alloc(&mut mmu, base, 32).unwrap();

        mmu.write(base, &[1, 2, 3, 4]).unwrap();
        let mut buf = [0u8; 4];
        mmu.read(base, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn oob_write_hits_guard() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_1000;
        san.alloc(&mut mmu, base, 32).unwrap();

        // One byte past the payload lands in the trailing redzone.
        let err = mmu.write_u8(base + 32, 0xAA).unwrap_err();
        assert_eq!(err.kind, FaultKind::Permission);

        // One byte before the payload lands in the leading redzone.
        let err = mmu.write_u8(base - 1, 0xAA).unwrap_err();
        assert_eq!(err.kind, FaultKind::Permission);
    }

    #[test]
    fn oob_read_hits_guard() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_2000;
        san.alloc(&mut mmu, base, 8).unwrap();
        mmu.write(base, &[0; 8]).unwrap();

        let err = mmu.read_u8(base + 8).unwrap_err();
        assert_eq!(err.kind, FaultKind::Permission);
    }

    #[test]
    fn uninitialized_read_faults() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_3000;
        san.alloc(&mut mmu, base, 8).unwrap();

        // Nothing written yet: RAW still set, so even an in-bounds read faults.
        assert_eq!(mmu.read_u8(base).unwrap_err().kind, FaultKind::Permission);

        mmu.write_u8(base, 0x7).unwrap();
        assert_eq!(mmu.read_u8(base).unwrap(), 0x7);
        // The rest of the payload is still uninitialized.
        assert_eq!(
            mmu.read_u8(base + 1).unwrap_err().kind,
            FaultKind::Permission
        );
    }

    #[test]
    fn free_then_use_after_free_faults() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_4000;
        san.alloc(&mut mmu, base, 16).unwrap();
        mmu.write(base, &[9; 16]).unwrap();

        san.free(&mut mmu, base).unwrap();
        assert!(san.is_quarantined(base));
        assert!(!san.is_live(base));

        assert_eq!(mmu.read_u8(base).unwrap_err().kind, FaultKind::Permission);
        assert_eq!(
            mmu.write_u8(base, 1).unwrap_err().kind,
            FaultKind::Permission
        );
    }

    #[test]
    fn double_free_is_reported() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_5000;
        san.alloc(&mut mmu, base, 16).unwrap();
        san.free(&mut mmu, base).unwrap();
        assert_eq!(
            san.free(&mut mmu, base).unwrap_err(),
            SanError::InvalidFree { addr: base }
        );
    }

    #[test]
    fn double_alloc_is_reported() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_6000;
        san.alloc(&mut mmu, base, 16).unwrap();
        assert_eq!(
            san.alloc(&mut mmu, base, 16).unwrap_err(),
            SanError::DoubleAlloc { addr: base }
        );
    }

    #[test]
    fn realloc_after_quarantine_works() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_7000;

        san.alloc(&mut mmu, base, 16).unwrap();
        mmu.write(base, &[1; 16]).unwrap();
        san.free(&mut mmu, base).unwrap();
        assert!(san.is_quarantined(base));

        // The guest allocator hands the same address back out (realistic for a slab/bump
        // allocator under fuzzing-scale churn). A fresh alloc() must fully un-poison it.
        san.alloc(&mut mmu, base, 16).unwrap();
        assert!(!san.is_quarantined(base));
        assert!(san.is_live(base));

        // Freshly allocated: uninitialized again, not somehow still readable from the old data.
        assert_eq!(mmu.read_u8(base).unwrap_err().kind, FaultKind::Permission);
        mmu.write(base, &[2; 16]).unwrap();
        let mut buf = [0u8; 16];
        mmu.read(base, &mut buf).unwrap();
        assert_eq!(buf, [2; 16]);
    }

    #[test]
    fn quarantine_cap_bounds_tracking_but_memory_stays_poisoned() {
        let mut mmu = Mmu::new(0x8000_0000, 0x2_0000);
        let mut san = Sanitizer::with_quarantine_cap(4, 2);

        let a = 0x8000_0000u32;
        let b = 0x8000_0100u32;
        let c = 0x8000_0200u32;
        for addr in [a, b, c] {
            san.alloc(&mut mmu, addr, 8).unwrap();
            san.free(&mut mmu, addr).unwrap();
        }

        // Cap is 2: the oldest (`a`) was evicted from *tracking* once `c` was freed...
        assert!(!san.is_quarantined(a));
        assert!(san.is_quarantined(b));
        assert!(san.is_quarantined(c));

        // ...but the bytes are still poisoned regardless of tracking (safety never regresses).
        assert_eq!(mmu.read_u8(a).unwrap_err().kind, FaultKind::Permission);
    }

    // -- alloc_with_slack: slack-only OOB, the zero-cross-object-false-positive KASAN core --

    #[test]
    fn slack_only_pokes_exactly_the_rounding_slack_kmalloc_30_in_bucket_32() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16); // redzone width must be irrelevant to this path
        let base = 0x8000_8000;
        let req_size = 30u32;
        let bucket_size = 32u32;

        san.alloc_with_slack(&mut mmu, base, req_size, bucket_size)
            .unwrap();

        // The live object [addr, addr+req_size) is WRITE|RAW: unwritten bytes still fault as
        // the uninitialized-read oracle, and after a write, in-bounds access succeeds.
        assert_eq!(
            mmu.read_u8(base).unwrap_err().kind,
            FaultKind::Permission,
            "unwritten live byte must still fault (RAW oracle)"
        );
        mmu.write(base, &[0xAA; 30]).unwrap();
        let mut buf = [0u8; 30];
        mmu.read(base, &mut buf).unwrap();
        assert_eq!(buf, [0xAA; 30]);

        // The 2 slack bytes [addr+30, addr+32) are poisoned no-access.
        assert_eq!(
            mmu.read_u8(base + 30).unwrap_err().kind,
            FaultKind::Permission
        );
        assert_eq!(
            mmu.write_u8(base + 30, 0x41).unwrap_err().kind,
            FaultKind::Permission
        );
        assert_eq!(
            mmu.read_u8(base + 31).unwrap_err().kind,
            FaultKind::Permission
        );
        assert_eq!(mmu.perm_at(base + 30), Some(0));
        assert_eq!(mmu.perm_at(base + 31), Some(0));

        // The byte at the bucket boundary (the first byte of what would be a packed neighbor
        // object in real SLUB) is UNTOUCHED — no cross-object guard was placed. A fresh Mmu
        // starts with perm 0 everywhere, so "untouched" here means still exactly that: not
        // poisoned *by this call* (poison() also produces perm 0, so the load-bearing proof is
        // that alloc_with_slack never called into the Mmu at this address at all — verified
        // structurally by the method's own code path, and behaviorally here by confirming nobody
        // else touched it either, i.e. it is still in its pristine pre-alloc state).
        assert_eq!(mmu.perm_at(base + 32), Some(0));
        // If the neighbor object had legitimately been allocated (as it would be in a packed
        // SLUB page), a write to it must succeed untouched by this allocation's bookkeeping.
        mmu.protect(base + 32, 4, PERM_WRITE | PERM_RAW).unwrap();
        mmu.write(base + 32, &[1, 2, 3, 4]).unwrap();
        let mut nbuf = [0u8; 4];
        mmu.read(base + 32, &mut nbuf).unwrap();
        assert_eq!(nbuf, [1, 2, 3, 4]);
    }

    #[test]
    fn slack_only_exact_fit_kmalloc_32_in_bucket_32_poisons_nothing() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_9000;
        let req_size = 32u32;
        let bucket_size = 32u32; // exact fit: no slack at all

        san.alloc_with_slack(&mut mmu, base, req_size, bucket_size)
            .unwrap();

        // The whole 32-byte object is live and usable, uninitialized-read oracle still applies.
        assert_eq!(mmu.read_u8(base).unwrap_err().kind, FaultKind::Permission);
        mmu.write(base, &[0x11; 32]).unwrap();
        let mut buf = [0u8; 32];
        mmu.read(base, &mut buf).unwrap();
        assert_eq!(buf, [0x11; 32]);

        // Nothing at all was poisoned by this call: the byte exactly at the bucket boundary
        // (would-be neighbor) is untouched, exactly as the slack case above. This is the
        // documented, honest gap: an exact-fit allocation (like Image.buggy's kmalloc(32) with
        // a planted write at offset 32) gets zero OOB coverage from this sanitizer, by
        // construction — only a kernel-cooperative oracle (SLUB_DEBUG_ON) catches that class.
        assert_eq!(mmu.perm_at(base + 32), Some(0));
        mmu.protect(base + 32, 1, PERM_WRITE | PERM_RAW).unwrap();
        mmu.write_u8(base + 32, 0x41).unwrap(); // succeeds: this sanitizer cannot see this write
        assert_eq!(mmu.read_u8(base + 32).unwrap(), 0x41);
    }

    #[test]
    fn slack_alloc_double_alloc_is_reported() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_a000;
        san.alloc_with_slack(&mut mmu, base, 30, 32).unwrap();
        assert_eq!(
            san.alloc_with_slack(&mut mmu, base, 30, 32).unwrap_err(),
            SanError::DoubleAlloc { addr: base }
        );
    }

    #[test]
    fn slack_alloc_free_poisons_the_whole_bucket_including_reopened_slack() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_b000;
        san.alloc_with_slack(&mut mmu, base, 30, 32).unwrap();
        mmu.write(base, &[1; 30]).unwrap();

        // ksize() legitimately re-opens the slack mid-lifetime...
        san.reopen_slack(&mut mmu, base).unwrap();
        mmu.write(base + 30, &[2; 2]).unwrap();

        // ...but free() poisons the entire bucket, including the now-reopened slack, since the
        // whole slot is dead.
        san.free(&mut mmu, base).unwrap();
        for off in 0..32u32 {
            assert_eq!(
                mmu.read_u8(base + off).unwrap_err().kind,
                FaultKind::Permission,
                "byte at offset {off} should be poisoned after free"
            );
        }
        // The would-be neighbor byte is still untouched by any of this.
        assert_eq!(mmu.perm_at(base + 32), Some(0));
    }

    // -- ksize()/krealloc() slack re-open primitive --

    #[test]
    fn ksize_reopen_unfaults_the_slack_then_it_faults_again_after_free() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_c000;
        san.alloc_with_slack(&mut mmu, base, 30, 32).unwrap();

        // Before ksize(): slack faults.
        assert_eq!(
            mmu.read_u8(base + 30).unwrap_err().kind,
            FaultKind::Permission
        );
        assert_eq!(
            mmu.write_u8(base + 30, 0xAA).unwrap_err().kind,
            FaultKind::Permission
        );

        san.reopen_slack(&mut mmu, base).unwrap();

        // After ksize(): slack behaves like ordinary uninitialized live memory (WRITE|RAW) —
        // still faults on read-before-write (the uninit oracle still applies), but a write now
        // succeeds and unlocks the read, instead of hard-faulting on write like poisoned memory.
        assert_eq!(
            mmu.read_u8(base + 30).unwrap_err().kind,
            FaultKind::Permission,
            "reopened slack is uninitialized, not pre-readable"
        );
        mmu.write(base, &[9; 30]).unwrap();
        mmu.write(base + 30, &[7, 7]).unwrap();
        let mut buf = [0u8; 32];
        mmu.read(base, &mut buf).unwrap();
        let mut expected = [9u8; 32];
        expected[30] = 7;
        expected[31] = 7;
        assert_eq!(buf, expected);
    }

    #[test]
    fn ksize_reopen_on_exact_fit_alloc_is_a_harmless_no_op() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_d000;
        san.alloc_with_slack(&mut mmu, base, 32, 32).unwrap();
        // No slack to reopen; must not error and must not touch the (nonexistent) neighbor byte.
        san.reopen_slack(&mut mmu, base).unwrap();
        assert_eq!(mmu.perm_at(base + 32), Some(0));
    }

    #[test]
    fn ksize_reopen_on_classic_redzone_alloc_is_a_harmless_no_op() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let base = 0x8000_e000;
        san.alloc(&mut mmu, base, 16).unwrap();
        // Classic alloc() has no slack concept at all; reopen_slack must not error or panic.
        san.reopen_slack(&mut mmu, base).unwrap();
        // The trailing redzone is untouched (still poisoned as alloc() left it).
        assert_eq!(
            mmu.read_u8(base + 16).unwrap_err().kind,
            FaultKind::Permission
        );
    }

    #[test]
    fn ksize_reopen_on_unknown_pointer_is_reported() {
        let mut mmu = mmu();
        let mut san = Sanitizer::new(16);
        let addr = 0x8000_f000;
        assert_eq!(
            san.reopen_slack(&mut mmu, addr).unwrap_err(),
            SanError::UnknownPointer { addr }
        );
    }
}
