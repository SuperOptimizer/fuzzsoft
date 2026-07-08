//! MEASUREMENT experiment (not a feature build): quantifies how much PC-convergence a 16-lane
//! `VecSystem` batch actually achieves during real full-system syscall fuzzing, and how much
//! headroom a *future* dynamic-warp-formation scheduler (an oversubscribed pool of guest VMs,
//! regrouped by PC into full 16-lane warps) could plausibly recover. This directly gates whether
//! the AVX-512 vector JIT (Stage 2) is worth building: vectorization only pays off when lanes
//! share a PC, and a prior prototype measured only 7-27% convergence during real fuzzing (0.83x
//! net — a LOSS). See `crates/fs-vec/DESIGN.md` and `docs/architecture.md` §2/§8 for the strategic
//! context this number feeds into.
//!
//! Two independent questions, both measured here against the REAL full-system path (golden
//! snapshot of a real OpenSBI+Linux boot, real `fs_prog` syscall programs lowered into guest
//! memory, real `VecSystem::step` lockstep execution — not a synthetic proxy):
//!
//! 1. **Does same-skeleton batch mutation (`fs_prog::mutate_batch`) raise 16-lane convergence
//!    over independent/random per-lane programs?** (`--experiment lanes16`, the default slice of
//!    this binary's output.) For every lockstep step of every batch, records: whether all active
//!    lanes share one PC (`convergence`), the distinct-PC count among active lanes (histogram),
//!    and the fraction of active lanes sitting at the single most popular PC (`top-PC occupancy`).
//!
//! 2. **How much regroup headroom exists in an oversubscribed pool of K > 16 candidate VMs?**
//!    (`--experiment pool`.) Rather than analytically extrapolating from the 16-lane distribution
//!    alone (which can't tell you whether two *different* 16-lane groups' most-popular PCs would
//!    ever coincide), this runs `ceil(K/16)` independent `VecSystem` instances **concurrently, in
//!    lockstep tick-by-tick** (every instance's `step()` always advances every active lane by
//!    exactly one guest instruction, so tick index == per-lane instructions-retired-since-reset
//!    uniformly across every instance — a fair, aligned basis for pooling their PCs post hoc) and,
//!    at every tick, pools every active lane's PC across all instances to measure the REAL
//!    distinct-PC/top-PC-occupancy distribution over the full K-lane pool. This is still a
//!    measurement, not a scheduler: no lane state is ever migrated between instances, only their
//!    PCs are read out. Two pool compositions are measured, since a real oversubscribed pool could
//!    look like either:
//!    - `same-base`: all K lanes are siblings of ONE base program (`mutate_batch(base, K)`, split
//!      across instances) — the best case, directly extending lever (a).
//!    - `diverse-base`: each of the `ceil(K/16)` instances runs a DIFFERENT in-flight batch (its
//!      own independently generated base, 16 siblings each) — the more realistic case where the
//!      pool holds several concurrent corpus explorations, not just one replicated 16-wide.
//!    - `independent`: every one of the K lanes (across all instances) is its OWN independently
//!      generated program (no shared skeleton anywhere) — the worst-case floor.
//!
//! Honesty notes (read before trusting the numbers):
//! - This is the REAL full-system path: real OpenSBI + real Linux kernel boot to the guest agent's
//!   SNAPSHOT hypercall, real `fs_prog`-generated/mutated syscall programs lowered into guest
//!   memory exactly as `fs-cli`'s scalar fuzzer and `examples/fuzz_vec.rs` do, real
//!   `VecSystem::step` (the same convergence-gated ALU/shared-load fast path production code
//!   uses). Nothing here is a synthetic microbenchmark proxy for full-system fuzzing.
//! - The pool experiment is a measurement proxy for a scheduler, not a working scheduler: no lane
//!   is ever migrated between `VecSystem` instances mid-batch, so a pooled step's "top-PC
//!   occupancy >= 16" is a necessary-but-not-sufficient condition for a real regroup scheduler to
//!   form a warp there (it would also need the migrated lanes' *non-PC* state — privilege/satp/
//!   pending-trap/register-file layout — to be compatible, which this measurement does not check;
//!   see `VecSystem::try_converged_fast_path`'s full gate list). Treat the reported fractions as an
//!   optimistic upper bound on regroup headroom, not a guaranteed win.
//! - Convergence is computed over ALL active-lane steps of a batch (mirroring how
//!   `VecSystem::simd_steps`/`scalar_steps` already account for the existing ALU fast path), which
//!   means the tail of a batch (as lanes finish out of order and the active count shrinks) can
//!   inflate the raw convergence fraction: 1 active lane is trivially "converged". A second set of
//!   counters, restricted to steps where all `LANES` lanes are still active, is reported alongside
//!   as the less tail-biased number.
//!
//! Run with `cargo run --release --example convergence -p fs-vec -- [--experiment lanes16|pool|all]
//! [--batches N] [--case-insns N] [--boot-insns N] [--seed N] [--pool-batches N] [--pools 32,48,64]`
//! from the repo root (needs `firmware/{fw_jump.bin,Image,fuzzsoft.dtb}`; nightly toolchain,
//! `portable_simd`).

use fs_mmu::{Access, Golden, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_platform::{run_until, Clint, Machine, Stop};
use fs_riscv::Cpu;
use fs_vec::{VecSystem, LANES};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const HC_EID: u32 = 0x0A55_0000;
const HC_SNAPSHOT: u32 = 0;

const RAM_BASE: u32 = 0x8000_0000;
const KERNEL_ADDR: u32 = 0x8040_0000;

/// Translate `count` consecutive words starting at guest VA `va` to their guest physical addresses
/// (write access) — copied from `examples/fuzz_vec.rs` (same boot-to-snapshot prelude).
fn translate_words(cpu: &mut Cpu, m: &mut Machine, va: u32, count: usize) -> Vec<u32> {
    let mut pas = Vec::with_capacity(count);
    for k in 0..count as u32 {
        pas.push(cpu.xlate(m, va + k * 4, Access::Write).expect("translate guest buffer"));
    }
    pas
}

// -------------------------------------------------------------------------------------------
// Per-step PC-distribution accumulator: shared by both the 16-lane and pool experiments (the
// pool experiment just feeds it a wider `pcs` slice every tick).
// -------------------------------------------------------------------------------------------

#[derive(Default)]
struct Accum {
    total_steps: u64,
    converged_steps: u64, // distinct_pcs == 1
    /// distinct_pcs histogram, 1-indexed (index 0 unused) up to `cap` (16 for a single VecSystem,
    /// up to the pool size K for the pool experiment).
    distinct_hist: Vec<u64>,
    /// top-PC occupancy COUNT histogram (how many active lanes sat at the single most popular
    /// pc), same indexing as `distinct_hist`.
    top_hist: Vec<u64>,
    /// Same two counters, but restricted to steps where `active_count == full_width` (no lane has
    /// exited yet this batch) — the less tail-biased view (see module doc's honesty note).
    total_steps_full: u64,
    converged_steps_full: u64,
    /// Sum of top-occupancy FRACTION (top_count / active_count) over all steps, for a quick mean.
    top_frac_sum: f64,
}

impl Accum {
    fn new(cap: usize) -> Self {
        Self {
            distinct_hist: vec![0; cap + 1],
            top_hist: vec![0; cap + 1],
            ..Default::default()
        }
    }

    /// Record one lockstep tick given every currently-active lane's PC (any order), and the
    /// "full width" this batch nominally has (16 for one `VecSystem`, K for a pool of them) so the
    /// tail-bias-free counters know when a step is still "all lanes present".
    fn record(&mut self, pcs: &[u32], full_width: usize) {
        if pcs.is_empty() {
            return;
        }
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &pc in pcs {
            *counts.entry(pc).or_insert(0) += 1;
        }
        let distinct = counts.len();
        let top = *counts.values().max().unwrap();
        self.total_steps += 1;
        self.distinct_hist[distinct] += 1;
        self.top_hist[top as usize] += 1;
        self.top_frac_sum += top as f64 / pcs.len() as f64;
        if distinct == 1 {
            self.converged_steps += 1;
        }
        if pcs.len() == full_width {
            self.total_steps_full += 1;
            if distinct == 1 {
                self.converged_steps_full += 1;
            }
        }
    }

    fn convergence_frac(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.converged_steps as f64 / self.total_steps as f64
        }
    }

    fn convergence_frac_full(&self) -> f64 {
        if self.total_steps_full == 0 {
            0.0
        } else {
            self.converged_steps_full as f64 / self.total_steps_full as f64
        }
    }

    fn mean_distinct(&self) -> f64 {
        let mut num = 0.0;
        for (d, &c) in self.distinct_hist.iter().enumerate() {
            num += d as f64 * c as f64;
        }
        num / self.total_steps.max(1) as f64
    }

    fn median_distinct(&self) -> usize {
        let half = self.total_steps / 2;
        let mut running = 0u64;
        for (d, &c) in self.distinct_hist.iter().enumerate() {
            running += c;
            if running > half {
                return d;
            }
        }
        0
    }

    fn p90_distinct(&self) -> usize {
        let target = (self.total_steps as f64 * 0.90) as u64;
        let mut running = 0u64;
        for (d, &c) in self.distinct_hist.iter().enumerate() {
            running += c;
            if running >= target {
                return d;
            }
        }
        self.distinct_hist.len() - 1
    }

    fn mean_top_frac(&self) -> f64 {
        self.top_frac_sum / self.total_steps.max(1) as f64
    }

    /// Fraction of steps whose top-PC occupancy COUNT was >= `warp`. This is the direct
    /// regroup-headroom readout: "how often could a `warp`-wide full warp be formed from the
    /// single most-popular pc at this tick" — meaningful standalone for the 16-lane experiment
    /// (`warp == 16 == full_width`) and, for the pool experiment, for any `warp <= K`.
    fn frac_top_at_least(&self, warp: usize) -> f64 {
        let hit: u64 = self.top_hist.iter().skip(warp).sum();
        hit as f64 / self.total_steps.max(1) as f64
    }

    fn report(&self, label: &str, full_width: usize) {
        println!("  [{label}] {} steps recorded", self.total_steps);
        println!(
            "    convergence (all lanes @ same pc): {:.2}% (raw, all active-counts) / {:.2}% (only when {} lanes still active)",
            self.convergence_frac() * 100.0,
            self.convergence_frac_full() * 100.0,
            full_width
        );
        println!(
            "    distinct-pc per step: mean={:.2}  median={}  p90={}",
            self.mean_distinct(),
            self.median_distinct(),
            self.p90_distinct()
        );
        println!("    top-pc occupancy: mean fraction of active lanes at the top pc = {:.1}%", self.mean_top_frac() * 100.0);
        print!("    distinct-pc histogram (1..{full_width}): ");
        for d in 1..=full_width {
            let c = self.distinct_hist.get(d).copied().unwrap_or(0);
            if c > 0 {
                print!("{d}:{:.1}% ", 100.0 * c as f64 / self.total_steps.max(1) as f64);
            }
        }
        println!();
    }
}

// -------------------------------------------------------------------------------------------
// Shared boot-to-snapshot prelude (identical to `examples/fuzz_vec.rs`).
// -------------------------------------------------------------------------------------------

struct Snapshot {
    golden: Arc<Golden>,
    golden_cpu: Cpu,
    golden_clint: Clint,
    base_uart: usize,
    prog_pas: Vec<u32>,
    scratch_pas: Vec<u32>,
    scratch_va: u32,
    ram_base: u32,
    ram_size: u32,
}

fn boot_to_snapshot(boot_insns: u64, ram_mb: u32) -> Snapshot {
    let ram_size = ram_mb * 0x0010_0000;
    let mut m = Machine::new(RAM_BASE, ram_size);
    m.ram.protect(RAM_BASE, ram_size, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    for (path, addr) in [("firmware/fw_jump.bin", RAM_BASE), ("firmware/Image", KERNEL_ADDR)] {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        m.ram.map(addr, &b, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    }
    let dtb = std::fs::read("firmware/fuzzsoft.dtb").unwrap_or_default();
    let dtb_addr = RAM_BASE + ram_size - 0x0020_0000;
    m.ram.map(dtb_addr, &dtb, PERM_READ | PERM_WRITE).unwrap();

    let mut cpu = Cpu::new(RAM_BASE);
    cpu.hypercall_eid = Some(HC_EID);
    cpu.regs[10] = 0;
    cpu.regs[11] = dtb_addr;

    eprintln!("convergence: booting to snapshot (budget {boot_insns} insns)...");
    match run_until(&mut cpu, &mut m, boot_insns) {
        Stop::Hypercall(HC_SNAPSHOT) => {}
        other => panic!("convergence: expected snapshot hypercall, got {other:?} (pc={:#x})", cpu.pc),
    }
    eprintln!("convergence: snapshot captured at pc={:#010x} after {} insns", cpu.pc, cpu.insns_retired);

    let prog_va = cpu.regs[11];
    let scratch_va = cpu.regs[12];
    let prog_pas = translate_words(&mut cpu, &mut m, prog_va, fs_prog::WIRE_WORDS);
    let scratch_words = (fs_prog::DEFAULT_SCRATCH_CAP / 4) as usize;
    let scratch_pas = translate_words(&mut cpu, &mut m, scratch_va, scratch_words);
    let base_uart = m.uart.out.len();

    let golden = Arc::new(Golden::from_mmu(&m.ram));
    let golden_cpu = cpu.clone();
    let golden_clint = m.clint.clone();

    Snapshot {
        golden,
        golden_cpu,
        golden_clint,
        base_uart,
        prog_pas,
        scratch_pas,
        scratch_va,
        ram_base: RAM_BASE,
        ram_size,
    }
}

fn new_system(snap: &Snapshot) -> VecSystem {
    VecSystem::from_golden_cpu(Arc::clone(&snap.golden), &snap.golden_cpu, snap.ram_base, snap.ram_size)
}

fn inject(vs: &mut VecSystem, snap: &Snapshot, lane: usize, low: &fs_prog::Lowered) {
    vs.inject_lane(lane, &snap.prog_pas, &fs_prog::to_wire(low));
    vs.inject_scratch(lane, &snap.scratch_pas, &low.scratch);
}

// -------------------------------------------------------------------------------------------
// Experiment 1: 16-lane convergence, strategy A (independent) vs strategy B (same-skeleton).
// -------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum LaneStrategy {
    /// Worst case: 16 completely independent, freshly-generated programs per batch.
    Independent,
    /// The convergence-friendly lever: 16 siblings of ONE base via `fs_prog::mutate_batch`
    /// (mutate only leaf data, same call skeleton).
    SameSkeleton,
}

fn run_lanes16(snap: &Snapshot, strategy: LaneStrategy, batches: u32, case_insns: u64, seed: u32) -> Accum {
    let mut rng = fs_prog::Rng::new(seed);
    let mut vs = new_system(snap);
    let mut acc = Accum::new(LANES);

    for _ in 0..batches {
        vs.reset_batch(&snap.golden_cpu, &snap.golden_clint, snap.base_uart);
        let lane_progs: Vec<fs_prog::Prog> = match strategy {
            LaneStrategy::Independent => (0..LANES).map(|_| fs_prog::generate(&mut rng)).collect(),
            LaneStrategy::SameSkeleton => {
                let base = fs_prog::generate(&mut rng);
                fs_prog::mutate_batch(&mut rng, &base, LANES)
            }
        };
        let lowered: Vec<fs_prog::Lowered> = lane_progs.iter().map(|p| fs_prog::lower(p, snap.scratch_va)).collect();
        for (lane, low) in lowered.iter().enumerate() {
            inject(&mut vs, snap, lane, low);
        }

        // Mirror `VecSystem::run_batch`'s own loop, inserting the PC snapshot right before each
        // `step()` call (so recorded pcs are "about to execute this tick", matching what a real
        // regroup scheduler would consult).
        let start_insns: [u64; LANES] = std::array::from_fn(|l| vs.insns_retired(l));
        while vs.any_active() {
            for (lane, &start) in start_insns.iter().enumerate() {
                if vs.active[lane] && vs.insns_retired(lane) - start >= case_insns {
                    // Budget exhausted: mirror `run_batch`'s halt-on-budget by just marking inactive
                    // via a fresh run_batch-style call would double-step; instead replicate exactly:
                    vs.active[lane] = false;
                }
            }
            if !vs.any_active() {
                break;
            }
            let pcs: Vec<u32> = (0..LANES).filter(|&l| vs.active[l]).map(|l| vs.lane_cpu(l).pc).collect();
            acc.record(&pcs, LANES);
            vs.step();
        }
    }
    acc
}

// -------------------------------------------------------------------------------------------
// Experiment 2: oversubscribed-pool regroup headroom, K in {32,48,64}, three pool compositions.
// -------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolStrategy {
    /// Every lane across the whole K-pool is its own independently generated program.
    Independent,
    /// All K lanes are siblings of ONE base program (`mutate_batch(base, K)`, split across the
    /// `ceil(K/16)` `VecSystem` instances) — best case.
    SameBase,
    /// Each of the `ceil(K/16)` instances runs its OWN independently generated base (16 siblings
    /// each) — a more realistic mixed pool of several concurrent corpus explorations.
    DiverseBase,
}

fn run_pool(snap: &Snapshot, strategy: PoolStrategy, k: usize, batches: u32, case_insns: u64, seed: u32) -> Accum {
    assert_eq!(k % LANES, 0, "pool size must be a multiple of {LANES} for this measurement");
    let n_systems = k / LANES;
    let mut rng = fs_prog::Rng::new(seed);
    let mut systems: Vec<VecSystem> = (0..n_systems).map(|_| new_system(snap)).collect();
    let mut acc = Accum::new(k);

    for _ in 0..batches {
        for vs in systems.iter_mut() {
            vs.reset_batch(&snap.golden_cpu, &snap.golden_clint, snap.base_uart);
        }

        match strategy {
            PoolStrategy::Independent => {
                for vs in systems.iter_mut() {
                    let progs: Vec<fs_prog::Prog> = (0..LANES).map(|_| fs_prog::generate(&mut rng)).collect();
                    for (lane, p) in progs.iter().enumerate() {
                        let low = fs_prog::lower(p, snap.scratch_va);
                        inject(vs, snap, lane, &low);
                    }
                }
            }
            PoolStrategy::SameBase => {
                let base = fs_prog::generate(&mut rng);
                let siblings = fs_prog::mutate_batch(&mut rng, &base, k);
                for (sys_idx, vs) in systems.iter_mut().enumerate() {
                    for lane in 0..LANES {
                        let p = &siblings[sys_idx * LANES + lane];
                        let low = fs_prog::lower(p, snap.scratch_va);
                        inject(vs, snap, lane, &low);
                    }
                }
            }
            PoolStrategy::DiverseBase => {
                for vs in systems.iter_mut() {
                    let base = fs_prog::generate(&mut rng);
                    let siblings = fs_prog::mutate_batch(&mut rng, &base, LANES);
                    for (lane, p) in siblings.iter().enumerate() {
                        let low = fs_prog::lower(p, snap.scratch_va);
                        inject(vs, snap, lane, &low);
                    }
                }
            }
        }

        let start_insns: Vec<[u64; LANES]> =
            systems.iter().map(|vs| std::array::from_fn(|l| vs.insns_retired(l))).collect();

        loop {
            let mut any = false;
            for (si, vs) in systems.iter_mut().enumerate() {
                for (lane, &start) in start_insns[si].iter().enumerate() {
                    if vs.active[lane] && vs.insns_retired(lane) - start >= case_insns {
                        vs.active[lane] = false;
                    }
                }
                if vs.any_active() {
                    any = true;
                }
            }
            if !any {
                break;
            }
            let pcs: Vec<u32> = systems
                .iter()
                .flat_map(|vs| (0..LANES).filter(|&l| vs.active[l]).map(|l| vs.lane_cpu(l).pc).collect::<Vec<_>>())
                .collect();
            acc.record(&pcs, k);
            for vs in systems.iter_mut() {
                if vs.any_active() {
                    vs.step();
                }
            }
        }
    }
    acc
}

// -------------------------------------------------------------------------------------------

struct Args {
    experiment: String,
    batches: u32,
    case_insns: u64,
    boot_insns: u64,
    seed: u32,
    ram_mb: u32,
    pool_batches: u32,
    pools: Vec<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        experiment: "all".to_string(),
        batches: 300,
        case_insns: 2_000_000,
        boot_insns: 3_000_000_000,
        seed: 1,
        ram_mb: 128,
        pool_batches: 150,
        pools: vec![32, 48, 64],
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let val = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--experiment" => a.experiment = val(i),
            "--batches" => a.batches = val(i).parse().unwrap_or(a.batches),
            "--case-insns" => a.case_insns = val(i).parse().unwrap_or(a.case_insns),
            "--boot-insns" => a.boot_insns = val(i).parse().unwrap_or(a.boot_insns),
            "--seed" => a.seed = val(i).parse().unwrap_or(a.seed),
            "--ram-mb" => a.ram_mb = val(i).parse().unwrap_or(a.ram_mb),
            "--pool-batches" => a.pool_batches = val(i).parse().unwrap_or(a.pool_batches),
            "--pools" => {
                a.pools = val(i).split(',').filter_map(|s| s.trim().parse().ok()).collect();
            }
            other => {
                eprintln!("convergence: unexpected argument {other:?}");
                std::process::exit(1);
            }
        }
        i += 2;
    }
    a
}

fn main() {
    let args = parse_args();
    let snap = boot_to_snapshot(args.boot_insns, args.ram_mb);

    let run_lanes16_experiment = matches!(args.experiment.as_str(), "lanes16" | "all");
    let run_pool_experiment = matches!(args.experiment.as_str(), "pool" | "all");

    if run_lanes16_experiment {
        println!("\n== Experiment 1: 16-lane convergence (real full-system path, {} batches x {LANES} lanes, {} insns/lane budget) ==", args.batches, args.case_insns);
        let t0 = Instant::now();
        let acc_a = run_lanes16(&snap, LaneStrategy::Independent, args.batches, args.case_insns, args.seed);
        println!("Strategy A (independent random programs per lane):");
        acc_a.report("A/independent", LANES);
        println!("    P(full 16-lane warp available right now) = {:.2}%", acc_a.frac_top_at_least(16) * 100.0);

        let acc_b = run_lanes16(&snap, LaneStrategy::SameSkeleton, args.batches, args.case_insns, args.seed);
        println!("Strategy B (fs_prog::mutate_batch — 16 siblings of ONE base, leaf-data-only mutation):");
        acc_b.report("B/same-skeleton", LANES);
        println!("    P(full 16-lane warp available right now) = {:.2}%", acc_b.frac_top_at_least(16) * 100.0);
        println!("Experiment 1 took {:.1}s", t0.elapsed().as_secs_f64());
    }

    if run_pool_experiment {
        println!("\n== Experiment 2: oversubscribed-pool regroup headroom ({} batches/pool-size) ==", args.pool_batches);
        for &k in &args.pools {
            let t0 = Instant::now();
            println!("-- pool size K={k} ({} concurrent VecSystem instances) --", k / LANES);

            let acc_ind = run_pool(&snap, PoolStrategy::Independent, k, args.pool_batches, args.case_insns, args.seed);
            println!("  independent (no shared skeleton anywhere in the pool):");
            acc_ind.report("independent", k);
            println!("    P(a full 16-lane warp could be regroup-formed) = {:.2}%", acc_ind.frac_top_at_least(LANES) * 100.0);

            let acc_same = run_pool(&snap, PoolStrategy::SameBase, k, args.pool_batches, args.case_insns, args.seed);
            println!("  same-base (all K lanes are siblings of ONE base):");
            acc_same.report("same-base", k);
            println!("    P(a full 16-lane warp could be regroup-formed) = {:.2}%", acc_same.frac_top_at_least(LANES) * 100.0);
            println!(
                "    P(>=2 full warps), i.e. >=32 lanes at one pc = {:.2}%",
                acc_same.frac_top_at_least(32.min(k)) * 100.0
            );

            let acc_div = run_pool(&snap, PoolStrategy::DiverseBase, k, args.pool_batches, args.case_insns, args.seed);
            println!("  diverse-base ({} independent same-skeleton batches of 16 pooled together):", k / LANES);
            acc_div.report("diverse-base", k);
            println!("    P(a full 16-lane warp could be regroup-formed) = {:.2}%", acc_div.frac_top_at_least(LANES) * 100.0);

            println!("  (K={k} took {:.1}s)", t0.elapsed().as_secs_f64());
        }
    }
}
