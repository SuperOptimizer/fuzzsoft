//! Coverage.
//!
//! M0 uses an EXACT edge/block set (decision #16): no hashing, no collisions, trivially
//! inspectable while we bring the interpreter up. Before we vectorize (M4) this is swapped for a
//! fixed-size AFL-style `hash(prev_pc, pc)` bitmap that is cheap per SIMD lane. The recording API
//! (`record_edge`) is intentionally the same shape so callers don't change.

use std::collections::BTreeSet;

#[derive(Default)]
pub struct Coverage {
    /// Directed control-flow edges (from_pc -> to_pc) observed.
    pub edges: BTreeSet<(u32, u32)>,
    /// Distinct basic-block / instruction addresses reached.
    pub blocks: BTreeSet<u32>,
}

impl Coverage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a control-flow transition. `to` is also counted as a reached block.
    #[inline]
    pub fn record_edge(&mut self, from: u32, to: u32) {
        self.edges.insert((from, to));
        self.blocks.insert(to);
    }

    /// Seed the initial block (the entry point) so it counts even before the first edge.
    pub fn seed_block(&mut self, pc: u32) {
        self.blocks.insert(pc);
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

// ---------------------------------------------------------------------------------------------
// AFL-style bitmap (decision #16, pre-M4 swap-in).
//
// `Coverage` above stays exact/BTreeSet-based and untouched so nothing depending on it breaks.
// `CovBitmap` is the fixed-size, allocation-free-per-edge alternative that `fs-cli`'s fuzzer can
// switch to for the hot per-lane path: a single `Vec`/boxed-array index + saturating increment,
// no tree insert, no collision-free guarantee needed. `VirginMap` is the accumulated/"virgin" map
// that turns a per-run `CovBitmap` into the coverage-guided "did this input find anything new"
// signal (decision #33 notes compare-coverage is a separate, later fast-follow on top of this).
// ---------------------------------------------------------------------------------------------

/// Fixed bitmap size, in buckets. Must be a power of two (the hash mask relies on it).
pub const MAP_SIZE: usize = 65536;

/// AFL-style edge hash: mix the previous PC (shifted right one bit, so A->B and B->A don't
/// collide as trivially as a plain XOR would) with the current PC, then mask down to
/// `MAP_SIZE` buckets. `MAP_SIZE` is a power of two so `& (MAP_SIZE - 1)` is exactly `% MAP_SIZE`.
#[inline]
pub fn hash_edge(prev_pc: u32, cur_pc: u32) -> usize {
    (((prev_pc >> 1) ^ cur_pc) as usize) & (MAP_SIZE - 1)
}

/// Classify a raw hit count into one of AFL's canonical hit-count buckets. Two runs that hit the
/// same edge a "similar" number of times land in the same bucket; a run that hits it enough more
/// times to cross a bucket boundary counts as new/interesting coverage.
///
/// Buckets (0 = never hit): 0, 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+ -- returned as the class
/// index `0..=8`.
#[inline]
pub fn classify_count(count: u8) -> u8 {
    match count {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4..=7 => 4,
        8..=15 => 5,
        16..=31 => 6,
        32..=127 => 7,
        _ => 8,
    }
}

/// A fixed `MAP_SIZE`-bucket AFL-style coverage map for a single run (or a single accumulation
/// target -- see [`VirginMap`]). Each bucket is a saturating hit counter (max 255).
pub struct CovBitmap {
    hits: Box<[u8]>,
}

impl CovBitmap {
    /// A fresh, all-zero map.
    pub fn new() -> Self {
        Self {
            hits: vec![0u8; MAP_SIZE].into_boxed_slice(),
        }
    }

    /// Record a control-flow transition `prev_pc -> cur_pc`: hash to a bucket and saturating-
    /// increment its hit count.
    #[inline]
    pub fn record_edge(&mut self, prev_pc: u32, cur_pc: u32) {
        let idx = hash_edge(prev_pc, cur_pc);
        self.hits[idx] = self.hits[idx].saturating_add(1);
    }

    /// Raw hit count for a bucket.
    #[inline]
    pub fn hit(&self, idx: usize) -> u8 {
        self.hits[idx]
    }

    /// Read-only view of the whole map, e.g. for a caller wanting to iterate all buckets.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.hits
    }

    /// Number of buckets with at least one hit in this run.
    pub fn covered_buckets(&self) -> usize {
        self.hits.iter().filter(|&&h| h != 0).count()
    }

    /// Zero every bucket, for reuse across runs without reallocating.
    pub fn clear(&mut self) {
        self.hits.iter_mut().for_each(|h| *h = 0);
    }
}

impl Default for CovBitmap {
    fn default() -> Self {
        Self::new()
    }
}

/// The global accumulated map ("virgin map" in AFL terminology): for every bucket, the highest
/// hit-count *class* (see [`classify_count`]) ever observed across all runs so far. Feeding a
/// per-run [`CovBitmap`] through [`VirginMap::has_new_bits`] is the coverage-guided feedback
/// signal: true means this run reached a bucket that was never hit before, or hit an
/// already-seen bucket enough more times to cross into a new class -- i.e. "add this input to
/// the corpus".
pub struct VirginMap {
    /// Highest class (0..=8, see `classify_count`) seen per bucket so far.
    classes: Box<[u8]>,
}

impl VirginMap {
    /// A fresh map that has seen nothing yet.
    pub fn new() -> Self {
        Self {
            classes: vec![0u8; MAP_SIZE].into_boxed_slice(),
        }
    }

    /// Fold `run_map` into the accumulated map. For every bucket, classify the run's hit count
    /// and compare against the highest class already recorded for that bucket; if the run's
    /// class is higher, update it and record that new coverage was found.
    ///
    /// Returns `true` iff at least one bucket's class advanced -- the "this input is
    /// interesting, keep it" signal for the fuzzer's corpus-selection logic.
    pub fn has_new_bits(&mut self, run_map: &CovBitmap) -> bool {
        let mut found_new = false;
        for i in 0..MAP_SIZE {
            let run_class = classify_count(run_map.hit(i));
            if run_class > self.classes[i] {
                self.classes[i] = run_class;
                found_new = true;
            }
        }
        found_new
    }

    /// Number of buckets that have ever been hit (non-zero class) across all folded runs.
    pub fn covered_buckets(&self) -> usize {
        self.classes.iter().filter(|&&c| c != 0).count()
    }
}

impl Default for VirginMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_edge_sets_expected_bucket() {
        let mut map = CovBitmap::new();
        let idx = hash_edge(0x1000, 0x1004);
        assert_eq!(map.hit(idx), 0);
        map.record_edge(0x1000, 0x1004);
        assert_eq!(map.hit(idx), 1);
        // Recording again bumps the same bucket.
        map.record_edge(0x1000, 0x1004);
        assert_eq!(map.hit(idx), 2);
        // Everything else stays untouched.
        assert_eq!(map.covered_buckets(), 1);
    }

    #[test]
    fn record_edge_saturates_at_255() {
        let mut map = CovBitmap::new();
        for _ in 0..300 {
            map.record_edge(1, 2);
        }
        let idx = hash_edge(1, 2);
        assert_eq!(map.hit(idx), 255);
    }

    #[test]
    fn hash_is_within_map_and_deterministic() {
        for (prev, cur) in [(0u32, 0u32), (0x8000_0000, 1), (u32::MAX, u32::MAX)] {
            let idx = hash_edge(prev, cur);
            assert!(idx < MAP_SIZE);
            assert_eq!(idx, hash_edge(prev, cur));
        }
    }

    #[test]
    fn bucketing_thresholds() {
        assert_eq!(classify_count(0), 0);
        assert_eq!(classify_count(1), 1);
        assert_eq!(classify_count(2), 2);
        assert_eq!(classify_count(3), 3);
        assert_eq!(classify_count(4), 4);
        assert_eq!(classify_count(7), 4);
        assert_eq!(classify_count(8), 5);
        assert_eq!(classify_count(15), 5);
        assert_eq!(classify_count(16), 6);
        assert_eq!(classify_count(31), 6);
        assert_eq!(classify_count(32), 7);
        assert_eq!(classify_count(127), 7);
        assert_eq!(classify_count(128), 8);
        assert_eq!(classify_count(255), 8);
    }

    #[test]
    fn bucketing_is_monotonic() {
        let mut prev = 0u8;
        for count in 0u16..=255 {
            let cls = classify_count(count as u8);
            assert!(cls >= prev, "class regressed at count {count}");
            prev = cls;
        }
    }

    #[test]
    fn has_new_bits_true_on_genuinely_new_coverage() {
        let mut virgin = VirginMap::new();
        let mut run1 = CovBitmap::new();
        run1.record_edge(0x1000, 0x1004);
        assert!(virgin.has_new_bits(&run1));
        assert_eq!(virgin.covered_buckets(), 1);

        // A second run that reaches a different edge is new coverage too.
        let mut run2 = CovBitmap::new();
        run2.record_edge(0x2000, 0x2004);
        assert!(virgin.has_new_bits(&run2));
        assert_eq!(virgin.covered_buckets(), 2);
    }

    #[test]
    fn has_new_bits_false_on_exact_repeat() {
        let mut virgin = VirginMap::new();
        let mut run = CovBitmap::new();
        run.record_edge(0x1000, 0x1004);
        assert!(virgin.has_new_bits(&run));

        // Folding the identical map again finds nothing new.
        assert!(!virgin.has_new_bits(&run));
    }

    #[test]
    fn has_new_bits_true_when_crossing_a_higher_bucket_class() {
        let mut virgin = VirginMap::new();
        let mut run = CovBitmap::new();
        run.record_edge(0x1000, 0x1004); // count 1, class 1
        assert!(virgin.has_new_bits(&run));
        assert!(!virgin.has_new_bits(&run));

        // Hit the same edge enough more times to cross into class 2 (count == 2).
        run.record_edge(0x1000, 0x1004); // count 2, class 2
        assert!(virgin.has_new_bits(&run));
        // Class already at 2 now -- another identical fold finds nothing new.
        assert!(!virgin.has_new_bits(&run));
    }
}
