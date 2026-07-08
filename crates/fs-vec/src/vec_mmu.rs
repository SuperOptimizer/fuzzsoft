//! `VecMmu` — the interleaved shared-memory model (`DESIGN.md` §3, `docs/architecture.md` §2/§3).
//!
//! Replaces "one `fs_mmu::Mmu` per lane" with **one shared store for all [`LANES`] lanes**, laid
//! out exactly as architecture.md §2/§3 specifies: for guest word `W` (four bytes, `addr & !3`),
//! lane `L`'s copy of that word lives at `content[W * LANES + L]` and its parallel byte-permission
//! plane (RWX+RAW, identical bit encoding to [`fs_mmu`]: `PERM_READ`/`PERM_WRITE`/`PERM_EXEC`/
//! `PERM_RAW`, one packed byte per guest byte) lives at `perms[W * LANES + L]`. All `LANES` lanes'
//! copies of one guest word are therefore contiguous in memory — the mechanical precondition for
//! a single `vmovdqa32` (content) / `vmovdqa32` (perms) to touch every lane's copy of that word in
//! one instruction, instead of `LANES` separate reads into `LANES` separate `Mmu`s.
//!
//! Three access shapes, matching DESIGN.md:
//! 1. **Same-address vectorized access** ([`VecMmu::load_same`]/[`VecMmu::store_same`]/
//!    [`VecMmu::ifetch16_same`]): the lockstep common case — every *active* lane targets the
//!    *same* guest address (true for a converged fetch by construction, and for a load/store
//!    whose effective address happens to agree across lanes). One aligned read/write of the
//!    16-lane-wide interleaved line services every lane; the 16-lane permission line is checked
//!    in one masked compare. Declines (returns `None`/`false`, mutating nothing) on misalignment,
//!    out-of-bounds, or any *active* lane failing the permission check — callers fall back to the
//!    per-lane path below for exactly that instruction.
//! 2. **Divergent-address path** ([`VecMmu::load_lane`]/[`VecMmu::store_lane`] plus the batch
//!    wrappers [`VecMmu::load_gather`]/[`VecMmu::store_scatter`]): lanes disagree on the effective
//!    address (or the same-address fast path declined). A scalar loop over active lanes today
//!    (decision #45 — correctness first, safe Rust); structured so that swapping the loop body for
//!    `vpgatherdd`/`vpscatterdd` later is mechanical: `load_gather` already takes one address *per
//!    lane* and returns one result *per lane*, exactly `vpgatherdd`'s calling convention, with the
//!    per-lane permission check taking the place of the mask register a real gather instruction
//!    would use to suppress faulting lanes.
//! 3. **Vectorized instruction fetch** ([`VecMmu::ifetch16_same`]): a converged group's fetch is
//!    just case 1 specialized to `PERM_EXEC`-only checking — this is what removes the
//!    per-lane `ifetch` that capped `fs-vec`'s original SIMD fast path at ~2x (DESIGN.md).

use fs_mmu::{Access, Fault, FaultKind, PERM_EXEC, PERM_RAW, PERM_READ, PERM_WRITE};
use std::simd::prelude::*;

use crate::LANES;

/// Shared interleaved guest memory for all `LANES` lanes (`docs/architecture.md` §2/§3).
///
/// `content`/`perms` are both `Vec<u32>` of length `num_words * LANES`; index `word * LANES +
/// lane` is lane `lane`'s copy of guest word `word` (guest address `base + word*4`). Each `u32` in
/// both planes packs its four guest bytes little-endian (byte `i`'s value/perm sits at bits
/// `[i*8..i*8+8)`) — the same byte order `fs_mmu::Mmu` uses (`u32::from_le_bytes`), so `map`'s
/// broadcast below reproduces `fs_mmu::Mmu::map`'s contents byte-for-byte in every lane.
pub struct VecMmu {
    base: u32,
    num_words: usize,
    content: Vec<u32>,
    perms: Vec<u32>,
}

impl VecMmu {
    /// A zeroed, all-lanes-identical guest window of `size` bytes starting at `base` (all
    /// permissions clear, i.e. unmapped, until `map`/`protect` grant access — same contract as
    /// `fs_mmu::Mmu::new`). `base` and `size` must be word (4-byte) multiples: every real base in
    /// this codebase already is (`0x8000_0000`), and word granularity is what makes the
    /// interleaved layout below well-defined without a cross-word-boundary special case.
    pub fn new(base: u32, size: usize) -> Self {
        assert_eq!(base % 4, 0, "VecMmu requires a word-aligned base");
        assert_eq!(size % 4, 0, "VecMmu requires a word-multiple size");
        let num_words = size / 4;
        Self { base, num_words, content: vec![0u32; num_words * LANES], perms: vec![0u32; num_words * LANES] }
    }

    pub fn base(&self) -> u32 {
        self.base
    }
    pub fn size(&self) -> usize {
        self.num_words * 4
    }

    /// Guest address -> (word index, byte offset within that word), or `None` if out of the
    /// mapped window. Because `base` is word-aligned (asserted in `new`), `addr % 4` and the
    /// in-word byte offset coincide, so alignment checks below can test `addr` directly.
    #[inline]
    fn word_off(&self, addr: u32) -> Option<(usize, u32)> {
        if addr < self.base {
            return None;
        }
        let rel = (addr - self.base) as usize;
        let word = rel / 4;
        (word < self.num_words).then_some((word, (rel % 4) as u32))
    }

    #[inline]
    fn fault(addr: u32, len: u32, access: Access, kind: FaultKind) -> Fault {
        Fault { addr, len, access, kind }
    }

    #[inline]
    fn check_align(addr: u32, n: u32, access: Access) -> Result<(), Fault> {
        if addr.is_multiple_of(n) {
            Ok(())
        } else {
            Err(Self::fault(addr, n, access, FaultKind::Unaligned))
        }
    }

    /// A little-endian byte-region mask covering `size` (1/2/4) bytes starting at byte offset
    /// `off` inside one packed u32 word (`off + size <= 4`, guaranteed by the alignment rules
    /// `fs_mmu`/`fs_riscv` already enforce: `size == 4` implies `off == 0`, `size == 2` implies
    /// `off` is `0` or `2`, `size == 1` allows any `off` — none of these ever spans two words).
    #[inline]
    fn region_mask(off: u32, size: u8) -> u32 {
        let bits = size as u32 * 8;
        let unshifted: u32 = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        unshifted << (off * 8)
    }

    /// One permission bit replicated into every one of a u32's four byte lanes (e.g.
    /// `PERM_READ` (`0x01`) -> `0x0101_0101`) — ANDed with [`Self::region_mask`] to build "this
    /// bit must be set on every touched byte" in one word-wide compare.
    #[inline]
    fn splat_byte(bit: u8) -> u32 {
        (bit as u32) * 0x0101_0101
    }

    // ---------------------------------------------------------------------------------------
    // Loader primitives: map/protect stamp every lane identically. This is the "byte-identical
    // starting snapshot" every fuzz case begins from (architecture.md §4/§7) — divergence between
    // lanes only ever arises from lanes *executing* differently afterwards, never from a
    // different starting image.
    // ---------------------------------------------------------------------------------------

    /// Write `data` at `addr` and stamp `perm`, identically in every lane, bypassing permission
    /// checks (mirrors `fs_mmu::Mmu::map`).
    pub fn map(&mut self, addr: u32, data: &[u8], perm: u8) -> Result<(), Fault> {
        for (i, &b) in data.iter().enumerate() {
            let a = addr.wrapping_add(i as u32);
            let (word, off) =
                self.word_off(a).ok_or_else(|| Self::fault(a, 1, Access::Write, FaultKind::Unmapped))?;
            let shift = off * 8;
            let byte_clear = !(0xFFu32 << shift);
            for lane in 0..LANES {
                let idx = word * LANES + lane;
                self.content[idx] = (self.content[idx] & byte_clear) | ((b as u32) << shift);
                self.perms[idx] = (self.perms[idx] & byte_clear) | ((perm as u32) << shift);
            }
        }
        Ok(())
    }

    /// Stamp permissions over `[addr, addr+len)` without touching contents, identically in every
    /// lane (mirrors `fs_mmu::Mmu::protect`).
    pub fn protect(&mut self, addr: u32, len: u32, perm: u8) -> Result<(), Fault> {
        for i in 0..len {
            let a = addr.wrapping_add(i);
            let (word, off) =
                self.word_off(a).ok_or_else(|| Self::fault(a, len, Access::Write, FaultKind::Unmapped))?;
            let shift = off * 8;
            let byte_clear = !(0xFFu32 << shift);
            for lane in 0..LANES {
                let idx = word * LANES + lane;
                self.perms[idx] = (self.perms[idx] & byte_clear) | ((perm as u32) << shift);
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // Per-lane checked access — the divergent-address path's scalar body (see module docs).
    // ---------------------------------------------------------------------------------------

    /// Checked load of `size` (1/2/4) bytes at `addr` for a single `lane`. This is the scalar
    /// body [`Self::load_gather`] batches over all lanes; the real AVX-512 executor's
    /// `vpgatherdd` replaces the *batching*, not this per-lane permission-check-then-extract
    /// logic, which stays scalar either way (a gather instruction still needs a per-lane fault
    /// mask derived from exactly this check).
    pub fn load_lane(&self, lane: usize, addr: u32, size: u8) -> Result<u32, Fault> {
        if size == 2 {
            Self::check_align(addr, 2, Access::Read)?;
        } else if size == 4 {
            Self::check_align(addr, 4, Access::Read)?;
        }
        let (word, off) = self
            .word_off(addr)
            .ok_or_else(|| Self::fault(addr, size as u32, Access::Read, FaultKind::Unmapped))?;
        let region = Self::region_mask(off, size);
        let need = Self::splat_byte(PERM_READ) & region;
        let idx = word * LANES + lane;
        if self.perms[idx] & need != need {
            return Err(Self::fault(addr, size as u32, Access::Read, FaultKind::Permission));
        }
        Ok((self.content[idx] & region) >> (off * 8))
    }

    /// Checked store of `size` (1/2/4) bytes at `addr` for a single `lane`, including the RAW
    /// tracking `fs_mmu::Mmu::write` performs: writing a byte upgrades it to `PERM_READ` and
    /// clears `PERM_RAW`, unlocking reads of exactly the bytes just written.
    pub fn store_lane(&mut self, lane: usize, addr: u32, size: u8, val: u32) -> Result<(), Fault> {
        if size == 2 {
            Self::check_align(addr, 2, Access::Write)?;
        } else if size == 4 {
            Self::check_align(addr, 4, Access::Write)?;
        }
        let (word, off) = self
            .word_off(addr)
            .ok_or_else(|| Self::fault(addr, size as u32, Access::Write, FaultKind::Unmapped))?;
        let region = Self::region_mask(off, size);
        let need = Self::splat_byte(PERM_WRITE) & region;
        let idx = word * LANES + lane;
        if self.perms[idx] & need != need {
            return Err(Self::fault(addr, size as u32, Access::Write, FaultKind::Permission));
        }
        let shift = off * 8;
        self.content[idx] = (self.content[idx] & !region) | ((val << shift) & region);
        let read_bits = Self::splat_byte(PERM_READ) & region;
        let raw_bits = Self::splat_byte(PERM_RAW) & region;
        self.perms[idx] = (self.perms[idx] | read_bits) & !raw_bits;
        Ok(())
    }

    /// Checked instruction half-word fetch for a single `lane` (2-byte aligned, `PERM_EXEC`
    /// only — `fs_mmu::Mmu::fetch_u16`'s contract).
    pub fn ifetch16_lane(&self, lane: usize, addr: u32) -> Result<u16, Fault> {
        Self::check_align(addr, 2, Access::Exec)?;
        let (word, off) = self
            .word_off(addr)
            .ok_or_else(|| Self::fault(addr, 2, Access::Exec, FaultKind::Unmapped))?;
        let region = Self::region_mask(off, 2);
        let need = Self::splat_byte(PERM_EXEC) & region;
        let idx = word * LANES + lane;
        if self.perms[idx] & need != need {
            return Err(Self::fault(addr, 2, Access::Exec, FaultKind::Permission));
        }
        Ok((((self.content[idx] & region) >> (off * 8)) & 0xFFFF) as u16)
    }

    /// Divergent-address load: one guest address *per active lane* (`addrs[lane]` is meaningless
    /// for inactive lanes), one `Result` per lane. A scalar loop over [`Self::load_lane`] today;
    /// the real AVX-512 version replaces this loop with `vpgatherdd` against a word-index vector
    /// (`(addr-base)/4*LANES + lane_iota`, itself a `vpaddd`/`vpslld`) plus a `vpcmpd`-derived
    /// fault k-mask — never a single vector fault, since faults are inherently per-lane/divergent.
    pub fn load_gather(
        &self,
        addrs: [u32; LANES],
        size: u8,
        active: [bool; LANES],
    ) -> [Result<u32, Fault>; LANES] {
        std::array::from_fn(|lane| {
            if active[lane] { self.load_lane(lane, addrs[lane], size) } else { Ok(0) }
        })
    }

    /// Divergent-address store: one guest address and value *per active lane*, one `Result` per
    /// lane. The scatter twin of [`Self::load_gather`] (`vpscatterdd` eventually).
    pub fn store_scatter(
        &mut self,
        addrs: [u32; LANES],
        size: u8,
        vals: [u32; LANES],
        active: [bool; LANES],
    ) -> [Result<(), Fault>; LANES] {
        std::array::from_fn(|lane| {
            if active[lane] { self.store_lane(lane, addrs[lane], size, vals[lane]) } else { Ok(()) }
        })
    }

    // ---------------------------------------------------------------------------------------
    // Vectorized gather/scatter fast path (DESIGN.md "Remaining gap" #3, closed for the common
    // case): the divergent-address loop above is `LANES` independent per-lane `load_lane`/
    // `store_lane` calls; the two functions below instead compute every active lane's host index
    // (`word*LANES+lane`, [`Self::load_lane`]'s own indexing, vectorized) as one `Simd<usize,
    // LANES>` and issue ONE `Simd::gather_select`/`scatter_select` against `content`/`perms` —
    // `portable_simd`'s safe (bounds-checked internally, per its own doc contract), no-`unsafe`
    // stand-in for `vpgatherdd`/`vpscatterdd` — for the common case where every active lane's
    // address is in-bounds, correctly `size`-aligned, and permission-checked. Only a lane that
    // actually fails (out-of-bounds/misaligned/missing permission — the rare case) needs the
    // exact per-lane [`Self::load_lane`]/[`Self::store_lane`] to classify its precise [`Fault`];
    // callers (`VecCpu::try_simd_gather_load`/`try_simd_gather_store`) re-service exactly those
    // lanes individually instead of falling back to a fully scalar loop over every active lane.
    // ---------------------------------------------------------------------------------------

    /// Shared address-plan helper for [`Self::load_gather_fast`]/[`Self::store_scatter_fast`]:
    /// from per-lane `addrs` and a `size` (1/2/4) shared across the whole group (one decoded
    /// instruction), computes:
    /// - `host_idx`: `word*LANES + lane` per lane — [`Self::load_lane`]'s own indexing,
    ///   vectorized. Invalid lanes' `word` is clamped to `0` first so this `usize` multiply/add
    ///   can never overflow, even for a wildly out-of-range `addr` (`addr < base` wraps `rel` to
    ///   a huge value) — `valid` (and `Simd::gather_select`/`scatter_select`'s own bounds check
    ///   against the slice length) is what actually excludes such a lane, not the arithmetic.
    /// - `region`/`shift`: the same little-endian byte-region mask [`Self::region_mask`] computes
    ///   per lane, except `shift` now varies *per lane* (a variable per-lane shift, `vpsllvd`/
    ///   `vpsrlvd`-shaped — same idiom [`simd_alu`](crate::VecCpu)'s `Sll`/`Srl`/`Sra` already
    ///   use) since addresses diverge here, unlike `load_same`/`store_same`'s single shared shift.
    /// - `valid`: `active` lanes whose address is in-bounds and `size`-aligned. Permission is
    ///   deliberately NOT checked here (load needs `PERM_READ`, store needs `PERM_WRITE` — the
    ///   caller checks the bit that applies to it).
    fn gather_addr_plan(
        &self,
        addrs: Simd<u32, LANES>,
        size: u8,
        active: Mask<i32, LANES>,
    ) -> (Simd<usize, LANES>, Simd<u32, LANES>, Simd<u32, LANES>, Mask<i32, LANES>) {
        let base_v = Simd::splat(self.base);
        let in_range_lo = addrs.simd_ge(base_v);
        // Wrapping sub, matching hardware/`word_off`'s own `(addr - base)` — an out-of-range
        // `addr < base` wraps to a huge `rel`, which `in_bounds` below then excludes via `valid`.
        let rel = addrs - base_v;
        let word = rel >> Simd::splat(2u32); // rel / 4 (base/size are word multiples, see `new`)
        let in_bounds = in_range_lo & word.simd_lt(Simd::splat(self.num_words as u32));

        let align_ok: Mask<i32, LANES> = if size == 1 {
            Mask::splat(true)
        } else {
            (addrs & Simd::splat((size as u32) - 1)).simd_eq(Simd::splat(0))
        };

        let valid = active & in_bounds & align_ok;

        let word_clamped = valid.select(word, Simd::splat(0));
        let word_usize: Simd<usize, LANES> = word_clamped.cast();
        let lane_iota: Simd<usize, LANES> = Simd::from_array(std::array::from_fn(|i| i));
        let host_idx = word_usize * Simd::splat(LANES) + lane_iota;

        let off = addrs & Simd::splat(3u32);
        let shift = off * Simd::splat(8u32);
        let bits = (size as u32) * 8;
        let unshifted: u32 = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let region = Simd::splat(unshifted) << shift;

        (host_idx, region, shift, valid)
    }

    /// Vectorized divergent-address load fast path: see the module section comment above. On
    /// return, `ok[lane]` (meaningful only where `active[lane]` is set) is `true` iff that lane's
    /// load succeeded via this fast path, in which case `values[lane]` holds its raw (not yet
    /// sign-extended) loaded value; `false` means the caller must re-service that one lane via
    /// [`Self::load_lane`] to get its precise [`Fault`]. Read-only — safe to call speculatively.
    pub fn load_gather_fast(
        &self,
        addrs: Simd<u32, LANES>,
        size: u8,
        active: Mask<i32, LANES>,
    ) -> (Simd<u32, LANES>, Mask<i32, LANES>) {
        let (host_idx, region, shift, valid) = self.gather_addr_plan(addrs, size, active);

        let need = Simd::splat(Self::splat_byte(PERM_READ)) & region;
        let perm_line =
            Simd::<u32, LANES>::gather_select(&self.perms, valid.cast(), host_idx, Simd::splat(0));
        let ok = valid & (perm_line & need).simd_eq(need);

        let content_line =
            Simd::<u32, LANES>::gather_select(&self.content, ok.cast(), host_idx, Simd::splat(0));
        let values = (content_line & region) >> shift;
        (values, ok)
    }

    /// Vectorized divergent-address store fast path, the scatter twin of
    /// [`Self::load_gather_fast`] (see the module section comment above): for lanes that pass
    /// (in-bounds, aligned, `PERM_WRITE`), performs the same read-modify-write
    /// [`Self::store_lane`] does (content merge + RAW-bit clearing) via one more gather (existing
    /// content) plus two `Simd::scatter_select` calls (content, perms) — fully committed before
    /// returning. Distinct active lanes always map to distinct `host_idx` (the lane index is
    /// folded into its low bits), so `scatter_select`'s "last write wins on index collision"
    /// caveat can never actually trigger here.
    ///
    /// Returns `ok`: `true` for lanes (within `active`) already fully committed by this fast path;
    /// `false` (left completely untouched) for any lane the caller must re-service via
    /// [`Self::store_lane`] to get its precise [`Fault`] and commit it.
    pub fn store_scatter_fast(
        &mut self,
        addrs: Simd<u32, LANES>,
        size: u8,
        vals: Simd<u32, LANES>,
        active: Mask<i32, LANES>,
    ) -> Mask<i32, LANES> {
        let (host_idx, region, shift, valid) = self.gather_addr_plan(addrs, size, active);

        let need = Simd::splat(Self::splat_byte(PERM_WRITE)) & region;
        let perm_line =
            Simd::<u32, LANES>::gather_select(&self.perms, valid.cast(), host_idx, Simd::splat(0));
        let ok = valid & (perm_line & need).simd_eq(need);
        let ok_isize = ok.cast();

        let content_line =
            Simd::<u32, LANES>::gather_select(&self.content, ok_isize, host_idx, Simd::splat(0));
        let shifted = (vals << shift) & region;
        let new_content = (content_line & !region) | shifted;
        new_content.scatter_select(&mut self.content, ok_isize, host_idx);

        let read_bits = Simd::splat(Self::splat_byte(PERM_READ)) & region;
        let raw_bits = Simd::splat(Self::splat_byte(PERM_RAW)) & region;
        let new_perm = (perm_line | read_bits) & !raw_bits;
        new_perm.scatter_select(&mut self.perms, ok_isize, host_idx);

        ok
    }

    // ---------------------------------------------------------------------------------------
    // Same-address vectorized access — the lockstep common case (see module docs).
    // ---------------------------------------------------------------------------------------

    /// Same-address vectorized load: every lane where `active[lane]` is set must be reading
    /// `addr`. On success, returns all `LANES` lanes' loaded (zero-extended) values packed into
    /// one `Simd<u32, LANES>` — inactive lanes' entries are unspecified garbage, callers must not
    /// read them. Returns `None`, having mutated nothing, if `addr` is misaligned/out-of-bounds
    /// or if *any* active lane's permission line is missing `PERM_READ` on a touched byte; callers
    /// fall back to [`Self::load_lane`]/[`Self::load_gather`] in that case.
    pub fn load_same(&self, addr: u32, size: u8, active: [bool; LANES]) -> Option<Simd<u32, LANES>> {
        if size == 2 && !addr.is_multiple_of(2) {
            return None;
        }
        if size == 4 && !addr.is_multiple_of(4) {
            return None;
        }
        let (word, off) = self.word_off(addr)?;
        let region = Self::region_mask(off, size);
        let need = Simd::splat(Self::splat_byte(PERM_READ) & region);
        let base_idx = word * LANES;
        let perm_line = Simd::<u32, LANES>::from_slice(&self.perms[base_idx..base_idx + LANES]);
        let ok = (perm_line & need).simd_eq(need);
        let active_mask: Mask<i32, LANES> = Mask::from_array(active);
        if (active_mask & !ok).any() {
            return None;
        }
        let content_line = Simd::<u32, LANES>::from_slice(&self.content[base_idx..base_idx + LANES]);
        Some((content_line & Simd::splat(region)) >> Simd::splat(off * 8))
    }

    /// Same-address vectorized store: every lane where `active[lane]` is set must be writing
    /// `addr`, taking its stored value from the corresponding lane of `vals` (`size` low bytes of
    /// each lane, per RV32's SB/SH/SW). On success, updates content *and* the RAW-tracking
    /// permission bits for all `LANES` lanes in one masked pass and returns `true`. Returns
    /// `false`, having mutated nothing, on the same declines as [`Self::load_same`].
    pub fn store_same(
        &mut self,
        addr: u32,
        size: u8,
        active: [bool; LANES],
        vals: Simd<u32, LANES>,
    ) -> bool {
        if size == 2 && !addr.is_multiple_of(2) {
            return false;
        }
        if size == 4 && !addr.is_multiple_of(4) {
            return false;
        }
        let Some((word, off)) = self.word_off(addr) else {
            return false;
        };
        let region = Self::region_mask(off, size);
        let need = Simd::splat(Self::splat_byte(PERM_WRITE) & region);
        let base_idx = word * LANES;
        let perm_line = Simd::<u32, LANES>::from_slice(&self.perms[base_idx..base_idx + LANES]);
        let ok = (perm_line & need).simd_eq(need);
        let active_mask: Mask<i32, LANES> = Mask::from_array(active);
        if (active_mask & !ok).any() {
            return false;
        }
        let content_line = Simd::<u32, LANES>::from_slice(&self.content[base_idx..base_idx + LANES]);
        let region_v = Simd::splat(region);
        let shifted = (vals << Simd::splat(off * 8)) & region_v;
        let new_content = active_mask.select((content_line & !region_v) | shifted, content_line);
        new_content.copy_to_slice(&mut self.content[base_idx..base_idx + LANES]);

        let read_bits = Simd::splat(Self::splat_byte(PERM_READ) & region);
        let raw_bits = Simd::splat(Self::splat_byte(PERM_RAW) & region);
        let new_perm = active_mask.select((perm_line | read_bits) & !raw_bits, perm_line);
        new_perm.copy_to_slice(&mut self.perms[base_idx..base_idx + LANES]);
        true
    }

    /// Same-address vectorized instruction-fetch half-word: [`Self::load_same`] specialized to
    /// `PERM_EXEC`-only checking (no `PERM_READ` needed to execute, matching `fs_mmu`). This is
    /// the fetch every converged `VecCpu::step` now issues once per group instead of once per
    /// lane (DESIGN.md).
    pub fn ifetch16_same(&self, addr: u32, active: [bool; LANES]) -> Option<Simd<u32, LANES>> {
        if !addr.is_multiple_of(2) {
            return None;
        }
        let (word, off) = self.word_off(addr)?;
        let region = Self::region_mask(off, 2);
        let need = Simd::splat(Self::splat_byte(PERM_EXEC) & region);
        let base_idx = word * LANES;
        let perm_line = Simd::<u32, LANES>::from_slice(&self.perms[base_idx..base_idx + LANES]);
        let ok = (perm_line & need).simd_eq(need);
        let active_mask: Mask<i32, LANES> = Mask::from_array(active);
        if (active_mask & !ok).any() {
            return None;
        }
        let content_line = Simd::<u32, LANES>::from_slice(&self.content[base_idx..base_idx + LANES]);
        Some((content_line & Simd::splat(region)) >> Simd::splat(off * 8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::FaultKind;

    const BASE: u32 = 0x8000_0000;

    #[test]
    fn map_broadcasts_identically_to_every_lane() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.map(BASE, &[0xAA, 0xBB, 0xCC, 0xDD], PERM_READ | PERM_WRITE).unwrap();
        for lane in 0..LANES {
            assert_eq!(mmu.load_lane(lane, BASE, 4).unwrap(), 0xDDCC_BBAA);
        }
    }

    #[test]
    fn store_lane_only_touches_its_own_lane() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 4, PERM_READ | PERM_WRITE).unwrap();
        mmu.store_lane(3, BASE, 4, 0x1234_5678).unwrap();
        assert_eq!(mmu.load_lane(3, BASE, 4).unwrap(), 0x1234_5678);
        // every other lane is untouched (still zero, and still readable since protect granted
        // PERM_READ up front, unlike a fresh allocation).
        for lane in 0..LANES {
            if lane != 3 {
                assert_eq!(mmu.load_lane(lane, BASE, 4).unwrap(), 0);
            }
        }
    }

    #[test]
    fn raw_uninit_read_faults_until_written_per_lane() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 4, PERM_RAW | PERM_WRITE).unwrap();
        assert_eq!(mmu.load_lane(0, BASE, 1).unwrap_err().kind, FaultKind::Permission);
        mmu.store_lane(0, BASE, 1, 0xAB).unwrap();
        assert_eq!(mmu.load_lane(0, BASE, 1).unwrap(), 0xAB);
        // Byte 1 is still uninitialized for lane 0 ...
        assert_eq!(mmu.load_lane(0, BASE + 1, 1).unwrap_err().kind, FaultKind::Permission);
        // ... and lane 1 never had byte 0 written, so it still faults independently of lane 0.
        assert_eq!(mmu.load_lane(1, BASE, 1).unwrap_err().kind, FaultKind::Permission);
    }

    #[test]
    fn unmapped_and_unaligned_faults() {
        let mmu = VecMmu::new(BASE, 0x1000);
        assert_eq!(mmu.load_lane(0, 0x1234, 1).unwrap_err().kind, FaultKind::Unmapped);
        assert_eq!(mmu.load_lane(0, BASE + 1, 4).unwrap_err().kind, FaultKind::Unaligned);
    }

    #[test]
    fn load_same_matches_load_lane_when_all_lanes_agree() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.map(BASE, &0x1122_3344u32.to_le_bytes(), PERM_READ | PERM_WRITE).unwrap();
        let active = [true; LANES];
        let vec = mmu.load_same(BASE, 4, active).expect("same-address fast path should engage");
        for lane in 0..LANES {
            assert_eq!(vec.to_array()[lane], mmu.load_lane(lane, BASE, 4).unwrap());
        }
    }

    #[test]
    fn load_same_declines_on_unmapped_address() {
        let mmu = VecMmu::new(BASE, 0x1000);
        assert!(mmu.load_same(0x1234, 4, [true; LANES]).is_none());
    }

    #[test]
    fn load_same_declines_when_any_active_lane_lacks_read_permission() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 4, PERM_WRITE).unwrap(); // no PERM_READ anywhere: every lane fails
        assert!(mmu.load_same(BASE, 4, [true; LANES]).is_none());
        // But the same fast path succeeds once every active lane is granted PERM_READ.
        mmu.protect(BASE, 4, PERM_READ | PERM_WRITE).unwrap();
        assert!(mmu.load_same(BASE, 4, [true; LANES]).is_some());
    }

    #[test]
    fn store_same_updates_every_active_lane_and_tracks_raw() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 4, PERM_RAW | PERM_WRITE).unwrap();
        let vals = Simd::from_array(std::array::from_fn(|lane| lane as u32));
        assert!(mmu.store_same(BASE, 4, [true; LANES], vals));
        for lane in 0..LANES {
            assert_eq!(mmu.load_lane(lane, BASE, 4).unwrap(), lane as u32);
        }
    }

    #[test]
    fn store_same_respects_inactive_lanes() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 4, PERM_READ | PERM_WRITE).unwrap();
        let vals = Simd::splat(0xFFFF_FFFFu32);
        let mut active = [true; LANES];
        active[2] = false;
        assert!(mmu.store_same(BASE, 4, active, vals));
        assert_eq!(mmu.load_lane(2, BASE, 4).unwrap(), 0, "inactive lane must be untouched");
        assert_eq!(mmu.load_lane(3, BASE, 4).unwrap(), 0xFFFF_FFFF);
    }

    #[test]
    fn gather_scatter_matches_per_lane_access() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 0x40, PERM_READ | PERM_WRITE).unwrap();
        let addrs: [u32; LANES] = std::array::from_fn(|lane| BASE + (lane as u32) * 4);
        let vals: [u32; LANES] = std::array::from_fn(|lane| lane as u32 * 100);
        let active = [true; LANES];
        let store_results = mmu.store_scatter(addrs, 4, vals, active);
        for r in &store_results {
            assert!(r.is_ok());
        }
        let load_results = mmu.load_gather(addrs, 4, active);
        for (lane, r) in load_results.iter().enumerate() {
            assert_eq!(*r, Ok(vals[lane]));
        }
    }

    #[test]
    fn gather_scatter_fast_matches_per_lane_access_on_the_common_case() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.protect(BASE, 0x40, PERM_READ | PERM_WRITE).unwrap();
        let addrs: [u32; LANES] = std::array::from_fn(|lane| BASE + (lane as u32) * 4);
        let vals: [u32; LANES] = std::array::from_fn(|lane| lane as u32 * 100);
        let active = [true; LANES];
        let active_mask: Mask<i32, LANES> = Mask::from_array(active);

        let store_ok = mmu.store_scatter_fast(Simd::from_array(addrs), 4, Simd::from_array(vals), active_mask);
        assert_eq!(store_ok, active_mask, "every lane is in-bounds/aligned/writable");
        for (lane, &addr) in addrs.iter().enumerate() {
            assert_eq!(mmu.load_lane(lane, addr, 4).unwrap(), vals[lane]);
        }

        let (loaded, load_ok) = mmu.load_gather_fast(Simd::from_array(addrs), 4, active_mask);
        assert_eq!(load_ok, active_mask, "every lane is in-bounds/aligned/readable");
        assert_eq!(loaded.to_array(), vals, "fast gather must match what was just fast-scattered");
    }

    /// A tiny xorshift PRNG (mirrors `lib.rs`'s own, no external `rand` dependency).
    struct XorShift64(u64);
    impl XorShift64 {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as u32
        }
        fn next_bool(&mut self) -> bool {
            self.next_u32() & 1 == 1
        }
    }

    /// Property test (Goal 2): [`VecMmu::load_gather_fast`]/[`VecMmu::store_scatter_fast`] must
    /// never disagree with the scalar per-lane [`VecMmu::load_lane`]/[`VecMmu::store_lane`] they
    /// vectorize — across random per-lane addresses that are a mix of in-bounds/out-of-bounds,
    /// aligned/misaligned, and permitted/unpermitted, exercising both the common all-succeed case
    /// and the rare per-lane-fault case in the same run. This is what actually proves the
    /// vectorized fast path is a sound, byte-exact stand-in for the scalar loop it replaces, not
    /// just "fast on the happy path".
    #[test]
    fn gather_fast_and_scatter_fast_never_disagree_with_the_scalar_per_lane_reference() {
        let mut rng = XorShift64(0xd1b5_4a32_d192_ed03);
        const SIZE_MEM: usize = 0x200;

        for _iter in 0..200 {
            // Fresh MMU each iteration, random permission holes so some lanes fault.
            let mut mmu = VecMmu::new(BASE, SIZE_MEM * 4);
            for word in 0..SIZE_MEM {
                let addr = BASE + (word as u32) * 4;
                // Randomly grant a mix of permissions per word (broadcast to every lane, matching
                // `protect`'s contract) so lanes sharing a word can still diverge on read/write
                // permission only via the address itself (different bytes / different words).
                let perm = if rng.next_bool() { PERM_READ | PERM_WRITE } else { 0 };
                mmu.protect(addr, 4, perm).unwrap();
            }

            // Per-lane sizes must agree (one decoded instruction), but addresses/values are free
            // to diverge arbitrarily, including out-of-bounds and misaligned.
            let size = *[1u8, 2, 4].get((rng.next_u32() as usize) % 3).unwrap();
            let mut addrs = [0u32; LANES];
            let mut vals = [0u32; LANES];
            let mut active = [true; LANES];
            for lane in 0..LANES {
                active[lane] = rng.next_bool() || lane == 0; // keep at least lane 0 active
                vals[lane] = rng.next_u32();
                // Bias toward in-bounds addresses (most of the value space) but let a chunk of the
                // range spill out-of-bounds / negative-offset to exercise the rare fault path too.
                let raw = rng.next_u32() % (SIZE_MEM as u32 * 4 + 64);
                addrs[lane] = BASE.wrapping_add(raw).wrapping_sub(16);
            }
            let active_mask: Mask<i32, LANES> = Mask::from_array(active);
            let addrs_v: Simd<u32, LANES> = Simd::from_array(addrs);

            // --- store: fast path vs. scalar reference on an independent clone of the same state.
            let mut mmu_ref = VecMmu {
                base: mmu.base(),
                num_words: mmu.size() / 4,
                content: mmu.content.clone(),
                perms: mmu.perms.clone(),
            };
            let store_ok = mmu.store_scatter_fast(addrs_v, size, Simd::from_array(vals), active_mask);
            let store_ok_arr = store_ok.to_array();
            let scalar_store_results = mmu_ref.store_scatter(addrs, size, vals, active);

            for lane in 0..LANES {
                if !active[lane] {
                    continue;
                }
                assert_eq!(
                    store_ok_arr[lane],
                    scalar_store_results[lane].is_ok(),
                    "iter {_iter} lane {lane}: fast/scalar store success disagreed (addr {:#x}, size {size})",
                    addrs[lane]
                );
            }
            // Re-run the scalar store on `mmu` too (idempotent for lanes the fast path already
            // committed, and this is the only way the halted/faulting lanes actually get written
            // in `mmu` for the assertion below — mirrors exactly what `VecCpu::try_simd_gather_store`
            // now does for the rare per-lane fallback).
            for lane in 0..LANES {
                if active[lane] && !store_ok_arr[lane] {
                    let _ = mmu.store_lane(lane, addrs[lane], size, vals[lane]);
                }
            }
            // After reconciling, `mmu` and `mmu_ref` (which took the full scalar path throughout)
            // must be byte-for-byte identical.
            assert_eq!(mmu.content, mmu_ref.content, "iter {_iter}: content diverged from scalar reference");
            assert_eq!(mmu.perms, mmu_ref.perms, "iter {_iter}: perms diverged from scalar reference");

            // --- load: fast path vs. scalar reference, against the now-identical `mmu`.
            let (loaded, load_ok) = mmu.load_gather_fast(addrs_v, size, active_mask);
            let loaded_arr = loaded.to_array();
            let load_ok_arr = load_ok.to_array();
            let scalar_load_results = mmu.load_gather(addrs, size, active);

            for lane in 0..LANES {
                if !active[lane] {
                    continue;
                }
                assert_eq!(
                    load_ok_arr[lane],
                    scalar_load_results[lane].is_ok(),
                    "iter {_iter} lane {lane}: fast/scalar load success disagreed (addr {:#x}, size {size})",
                    addrs[lane]
                );
                if load_ok_arr[lane] {
                    assert_eq!(
                        loaded_arr[lane],
                        scalar_load_results[lane].unwrap(),
                        "iter {_iter} lane {lane}: fast/scalar load VALUE disagreed (addr {:#x}, size {size})",
                        addrs[lane]
                    );
                }
            }
        }
    }

    #[test]
    fn ifetch16_same_matches_ifetch16_lane() {
        let mut mmu = VecMmu::new(BASE, 0x1000);
        mmu.map(BASE, &0xBEEFu16.to_le_bytes(), PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let vec = mmu.ifetch16_same(BASE, [true; LANES]).unwrap();
        for lane in 0..LANES {
            assert_eq!(vec.to_array()[lane] as u16, mmu.ifetch16_lane(lane, BASE).unwrap());
        }
    }
}
