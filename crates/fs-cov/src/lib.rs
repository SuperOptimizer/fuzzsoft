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
