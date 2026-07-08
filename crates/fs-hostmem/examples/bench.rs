//! Honest head-to-head numbers: `HostMem` (memfd + MAP_PRIVATE + MADV_DONTNEED) vs `fs_mmu::Mmu`
//! (software `Vec<u8>` + byte-perm-plane + O(dirty) block reset), all on 128 MiB guests. This is
//! the deliverable for the host-mmap prototype evaluation (see docs/cow-shared-ram.md): does
//! trading byte-granular software tracking for host-MMU-backed COW actually pay off for the
//! scalar `--jobs` fuzzing path?
//!
//! Sections:
//!   1. Reset cost vs pages dirtied (the headline: the fuzzer resets every case).
//!   2. Memory footprint/sharing across 16 threads (Pss/RSS), vs 16 independent Vec<u8> copies.
//!   3. Sequential + random 4-byte read/write access throughput (MIPS).
//!   4. A realistic "fuzz case" microbench: ~200k scattered 4-byte writes then reset, repeated.

use fs_hostmem::{Golden, PERM_EXEC, PERM_READ, PERM_WRITE};
use fs_mmu::Mmu;
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

const BASE: u32 = 0x8000_0000;
const SIZE: usize = 128 * 1024 * 1024;
/// x86-64 base page size. Not queried via `sysconf` — this prototype targets exactly the
/// environment it's benchmarked on (Linux/x86-64), and 4 KiB is that host's page size.
const PAGE: usize = 4096;

/// Deterministic xorshift32 (matches the differential test's; kept local rather than pulling in
/// a `rand` dependency for a benchmark).
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

/// A 128 MiB guest, fully mapped RWX, zeroed content — the "golden" layout every section starts
/// from. `protect` over the whole window is an O(size) one-time setup cost (excluded from every
/// timed measurement below).
fn make_golden_mmu() -> Mmu {
    let mut mmu = Mmu::new(BASE, SIZE);
    mmu.protect(BASE, SIZE as u32, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
    mmu
}

fn fmt_dur(d: Duration) -> String {
    if d.as_secs_f64() >= 1e-3 {
        format!("{:.3} ms", d.as_secs_f64() * 1e3)
    } else {
        format!("{:.1} us", d.as_secs_f64() * 1e6)
    }
}

// ---------------------------------------------------------------------------------------------
// Section 1: reset cost vs pages dirtied.
// ---------------------------------------------------------------------------------------------

fn spread_page_indices(count: usize) -> Vec<usize> {
    let total_pages = SIZE / PAGE;
    let count = count.min(total_pages);
    let stride = (total_pages / count).max(1);
    (0..count).map(|i| (i * stride) % total_pages).collect()
}

fn bench_reset_costs() {
    println!("\n== 1. Reset cost vs pages dirtied (128 MiB guest, avg over trials) ==");
    println!(
        "{:>8}  {:>16}  {:>16}  {:>10}",
        "pages", "Mmu reset_dirty", "HostMem MADVISE", "speedup"
    );

    let mut mmu = make_golden_mmu();
    let (gmem, gperms) = mmu.planes();
    let (gmem, gperms) = (gmem.to_vec(), gperms.to_vec());
    mmu.enable_dirty_tracking();

    let golden = Golden::from_planes(BASE, &gmem, &gperms).unwrap();
    let mut hm = golden.new_view().unwrap();

    for &pages in &[1usize, 64, 1024, 16384, 32768] {
        let idx = spread_page_indices(pages);
        let trials = if pages <= 1024 { 200 } else { 30 };

        let mut total = Duration::ZERO;
        for _ in 0..trials {
            for &pn in &idx {
                mmu.write_u32(BASE + (pn * PAGE) as u32, 0xDEAD_BEEF).unwrap();
            }
            let t0 = Instant::now();
            mmu.reset_dirty(&gmem, &gperms);
            total += t0.elapsed();
        }
        let mmu_avg = total / trials as u32;

        let mut total = Duration::ZERO;
        for _ in 0..trials {
            for &pn in &idx {
                hm.write_u32(BASE + (pn * PAGE) as u32, 0xDEAD_BEEF).unwrap();
            }
            let t0 = Instant::now();
            hm.reset().unwrap();
            total += t0.elapsed();
        }
        let host_avg = total / trials as u32;

        let speedup = mmu_avg.as_secs_f64() / host_avg.as_secs_f64();
        println!(
            "{:>8}  {:>16}  {:>16}  {:>9.2}x",
            pages,
            fmt_dur(mmu_avg),
            fmt_dur(host_avg),
            speedup
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Section 2: memory footprint / sharing across threads.
// ---------------------------------------------------------------------------------------------

fn read_status_kb(field: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn read_smaps_rollup_pss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("Pss:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn fmt_kb(v: Option<u64>) -> String {
    match v {
        Some(kb) => format!("{:.1} MiB", kb as f64 / 1024.0),
        None => "n/a".to_string(),
    }
}

const THREADS: usize = 16;
const TOUCH_PAGES: usize = 200;

fn bench_memory_hostmem() -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let mmu = make_golden_mmu();
    let golden = Arc::new(Golden::from_mmu(&mmu).unwrap());
    drop(mmu); // don't let the one-off software copy inflate the "before" baseline

    let before_pss = read_smaps_rollup_pss_kb();
    let before_rss = read_status_kb("VmRSS:");

    let barrier = Arc::new(Barrier::new(THREADS + 1));
    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let golden = Arc::clone(&golden);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut hm = golden.new_view().unwrap();
                let mut rng = Rng::new(0x1000 + tid as u32);
                for _ in 0..TOUCH_PAGES {
                    let pn = rng.below((SIZE / PAGE) as u32) as usize;
                    hm.write_u32(BASE + (pn * PAGE) as u32, 0xAAAA_AAAA).unwrap();
                }
                barrier.wait(); // hold the mapping live while main measures
                barrier.wait(); // wait for release
            })
        })
        .collect();
    barrier.wait();
    let live_pss = read_smaps_rollup_pss_kb();
    let live_rss = read_status_kb("VmRSS:");
    barrier.wait();
    for h in handles {
        h.join().unwrap();
    }
    (before_pss, before_rss, live_pss, live_rss)
}

fn bench_memory_private_copies() -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    // Stand-in for today's model: each `--jobs` thread owns an independent 128 MiB `Mmu`
    // (mem + perms planes), cloned from one golden buffer.
    let golden_mem = Arc::new(vec![0u8; SIZE]);
    let golden_perms = Arc::new(vec![PERM_READ | PERM_WRITE | PERM_EXEC; SIZE]);

    let before_pss = read_smaps_rollup_pss_kb();
    let before_rss = read_status_kb("VmRSS:");

    let barrier = Arc::new(Barrier::new(THREADS + 1));
    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let golden_mem = Arc::clone(&golden_mem);
            let golden_perms = Arc::clone(&golden_perms);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut mem = (*golden_mem).clone();
                let perms = (*golden_perms).clone();
                let mut rng = Rng::new(0x2000 + tid as u32);
                for _ in 0..TOUCH_PAGES {
                    let pn = rng.below((SIZE / PAGE) as u32) as usize;
                    mem[pn * PAGE] = 0xAA;
                }
                barrier.wait();
                barrier.wait();
                black_box((&mem, &perms));
            })
        })
        .collect();
    barrier.wait();
    let live_pss = read_smaps_rollup_pss_kb();
    let live_rss = read_status_kb("VmRSS:");
    barrier.wait();
    for h in handles {
        h.join().unwrap();
    }
    (before_pss, before_rss, live_pss, live_rss)
}

fn bench_memory_sharing() {
    println!(
        "\n== 2. Memory footprint: {THREADS} threads x 128 MiB guest, {TOUCH_PAGES} pages touched/thread =="
    );
    let (b_pss, b_rss, l_pss, l_rss) = bench_memory_hostmem();
    println!("HostMem (shared golden memfd, MAP_PRIVATE views):");
    println!("  before: Pss={}  VmRSS={}", fmt_kb(b_pss), fmt_kb(b_rss));
    println!("  live:   Pss={}  VmRSS={}", fmt_kb(l_pss), fmt_kb(l_rss));
    if let (Some(l), Some(b)) = (l_pss, b_pss) {
        println!("  delta Pss (live-before): {:.1} MiB", (l as f64 - b as f64) / 1024.0);
    }

    let (b_pss, b_rss, l_pss, l_rss) = bench_memory_private_copies();
    println!("Vec<u8> baseline ({THREADS} independent 128+128 MiB private copies):");
    println!("  before: Pss={}  VmRSS={}", fmt_kb(b_pss), fmt_kb(b_rss));
    println!("  live:   Pss={}  VmRSS={}", fmt_kb(l_pss), fmt_kb(l_rss));
    if let (Some(l), Some(b)) = (l_pss, b_pss) {
        println!("  delta Pss (live-before): {:.1} MiB", (l as f64 - b as f64) / 1024.0);
    }
}

// ---------------------------------------------------------------------------------------------
// Section 3: access throughput (MIPS).
// ---------------------------------------------------------------------------------------------

fn bench_throughput() {
    println!("\n== 3. Access throughput (4-byte read/write, MIPS) ==");
    let mut mmu = make_golden_mmu();
    let golden = Golden::from_mmu(&mmu).unwrap();
    let mut hm = golden.new_view().unwrap();

    let seq_iters: u64 = 30_000_000;
    let rand_iters: usize = 4_000_000;

    // Sequential addresses (wrap around the window), precomputed random addresses (aligned to 4).
    let seq_addr = |i: u64| BASE + ((i * 4) % (SIZE as u64 - 4)) as u32;
    let mut rng = Rng::new(0xBEEF_0001);
    let rand_addrs: Vec<u32> = (0..rand_iters)
        .map(|_| BASE + (rng.below((SIZE / 4) as u32) * 4))
        .collect();

    macro_rules! time_mips {
        ($iters:expr, $body:expr) => {{
            let t0 = Instant::now();
            $body;
            let secs = t0.elapsed().as_secs_f64();
            ($iters as f64 / secs) / 1e6
        }};
    }

    let mmu_seq_write = time_mips!(seq_iters, {
        for i in 0..seq_iters {
            mmu.write_u32(seq_addr(i), i as u32).unwrap();
        }
    });
    let host_seq_write = time_mips!(seq_iters, {
        for i in 0..seq_iters {
            hm.write_u32(seq_addr(i), i as u32).unwrap();
        }
    });
    let mmu_seq_read = time_mips!(seq_iters, {
        for i in 0..seq_iters {
            black_box(mmu.read_u32(seq_addr(i)).unwrap());
        }
    });
    let host_seq_read = time_mips!(seq_iters, {
        for i in 0..seq_iters {
            black_box(hm.read_u32(seq_addr(i)).unwrap());
        }
    });
    let mmu_rand_write = time_mips!(rand_addrs.len(), {
        for &a in &rand_addrs {
            mmu.write_u32(a, a).unwrap();
        }
    });
    let host_rand_write = time_mips!(rand_addrs.len(), {
        for &a in &rand_addrs {
            hm.write_u32(a, a).unwrap();
        }
    });
    let mmu_rand_read = time_mips!(rand_addrs.len(), {
        for &a in &rand_addrs {
            black_box(mmu.read_u32(a).unwrap());
        }
    });
    let host_rand_read = time_mips!(rand_addrs.len(), {
        for &a in &rand_addrs {
            black_box(hm.read_u32(a).unwrap());
        }
    });

    println!("{:>18}  {:>10}  {:>10}", "pattern", "Mmu", "HostMem");
    println!("{:>18}  {:>10.1}  {:>10.1}", "seq write", mmu_seq_write, host_seq_write);
    println!("{:>18}  {:>10.1}  {:>10.1}", "seq read", mmu_seq_read, host_seq_read);
    println!("{:>18}  {:>10.1}  {:>10.1}", "random write", mmu_rand_write, host_rand_write);
    println!("{:>18}  {:>10.1}  {:>10.1}", "random read", mmu_rand_read, host_rand_read);
}

// ---------------------------------------------------------------------------------------------
// Section 4: realistic "fuzz case" microbench.
// ---------------------------------------------------------------------------------------------

fn bench_fuzz_case() {
    println!("\n== 4. Fuzz-case microbench: ~200k scattered 4-byte writes + reset, per case ==");
    const WRITES_PER_CASE: usize = 200_000;
    const WORKING_PAGES: usize = 300;
    const CASES: usize = 100;

    let total_pages = SIZE / PAGE;
    let mut rng = Rng::new(0xF00D_0001);
    let page_start = rng.below((total_pages - WORKING_PAGES) as u32) as usize;
    let addrs: Vec<u32> = (0..WRITES_PER_CASE)
        .map(|_| {
            let pn = page_start + rng.below(WORKING_PAGES as u32) as usize;
            let word = rng.below((PAGE / 4) as u32) as usize;
            BASE + (pn * PAGE + word * 4) as u32
        })
        .collect();

    let mut mmu = make_golden_mmu();
    let (gmem, gperms) = mmu.planes();
    let (gmem, gperms) = (gmem.to_vec(), gperms.to_vec());
    mmu.enable_dirty_tracking();

    let t0 = Instant::now();
    for case in 0..CASES {
        for &a in &addrs {
            mmu.write_u32(a, case as u32).unwrap();
        }
        mmu.reset_dirty(&gmem, &gperms);
    }
    let mmu_elapsed = t0.elapsed();

    let golden = Golden::from_planes(BASE, &gmem, &gperms).unwrap();
    let mut hm = golden.new_view().unwrap();
    let t0 = Instant::now();
    for case in 0..CASES {
        for &a in &addrs {
            hm.write_u32(a, case as u32).unwrap();
        }
        hm.reset().unwrap();
    }
    let host_elapsed = t0.elapsed();

    let mmu_cps = CASES as f64 / mmu_elapsed.as_secs_f64();
    let host_cps = CASES as f64 / host_elapsed.as_secs_f64();
    println!(
        "{WRITES_PER_CASE} writes/case across {WORKING_PAGES} pages, {CASES} cases:"
    );
    println!("  Mmu:     {:>10.1} cases/sec  (total {})", mmu_cps, fmt_dur(mmu_elapsed));
    println!("  HostMem: {:>10.1} cases/sec  (total {})", host_cps, fmt_dur(host_elapsed));
    println!("  speedup: {:.2}x", host_cps / mmu_cps);
}

fn main() {
    println!("fs-hostmem prototype benchmark — HostMem (memfd+mmap+madvise) vs fs_mmu::Mmu");
    println!("guest size = {} MiB, host page size assumed = {} B", SIZE / 1024 / 1024, PAGE);
    bench_reset_costs();
    bench_memory_sharing();
    bench_throughput();
    bench_fuzz_case();
}
