//! Deterministic xorshift32 PRNG (matches `fs-cli`'s existing generator/mutator, decision #7),
//! plus a few small helpers used throughout generation/mutation.

#[derive(Clone, Debug)]
pub struct Rng(pub u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(seed.max(1))
    }

    // Matches `fs-cli`'s existing `Rng::next` naming (this is not `Iterator::next`; the PRNG
    // is not an iterator, just a bare xorshift step named to match the sibling crate).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    pub fn next_u64(&mut self) -> u64 {
        ((self.next() as u64) << 32) | self.next() as u64
    }

    /// Uniform in `[0, n)`; returns 0 if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() as usize) % n
        }
    }

    /// True with probability `pct/100`.
    pub fn chance(&mut self, pct: u32) -> bool {
        self.next() % 100 < pct
    }

    pub fn pick<'a, T>(&mut self, s: &'a [T]) -> &'a T {
        &s[self.below(s.len())]
    }

    pub fn bool(&mut self) -> bool {
        self.next().is_multiple_of(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn below_is_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            assert!(r.below(5) < 5);
        }
        assert_eq!(r.below(0), 0);
    }
}
