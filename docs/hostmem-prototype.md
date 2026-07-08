# Host-mmap guest RAM — prototype & decision (do NOT adopt; go software COW)

`crates/fs-hostmem` is a measured prototype (not production) of host-MMU-backed guest RAM for the
scalar `--jobs` path: golden bytes in a **memfd**, per-thread `mmap(MAP_PRIVATE)` copy-on-write view,
reset via `madvise(MADV_DONTNEED)`. Software byte-permission plane retained (second memfd) so the
byte-granular RAW/redzone sanitizer survives. `hostmem_matches_mmu` differential test passes
(byte-exact + fault-exact vs `fs_mmu::Mmu`, golden restored on reset). Uses `libc`; the crate is the
one place `unsafe` lives.

## Measured (128 MiB guest, release)

**Reset cost vs pages dirtied — the primitive that motivated the whole idea:**

| pages dirtied | `Mmu::reset_dirty` | `HostMem` MADV_DONTNEED | host speedup |
|---|---|---|---|
| 1 | 0.0 µs | 4.0 µs | 0.01× |
| 64 | 1.8 µs | 130 µs | 0.01× |
| 1024 | 24 µs | 1.6 ms | 0.01× |
| 16384 | 858 µs | 8.6 ms | 0.10× |
| 32768 (whole guest) | 2.6 ms | 13.9 ms | 0.19× |

**No crossover anywhere** — software `reset_dirty` (an O(dirty) 64 B in-process memcpy) is 5–100×
faster than `MADV_DONTNEED` (each COW-dirtied page = a real kernel page-table teardown). The reset
win host-mmap promised does not exist for our workload.

**Other axes:** random-write ~2.3× faster on HostMem (raw mmap vs Vec + per-write software dirty
mark); reads ~even. Fuzz-case microbench (200k scattered writes + reset): **1.46×** — but from access
throughput + lazy fault-in, *not* reset. Memory: HostMem faults in only its working set (~123 MiB
for 16 threads) vs 16 eager 256 MiB `Vec` copies (~4 GiB) — a real "lazy vs eager" ~33× difference,
though within one process `Pss` can't prove cross-thread physical dedup.

## Decision: implement the **software COW** (`docs/cow-shared-ram.md`) for *both* paths

The prototype's numbers argue against adopting host-mmap, because the safe software COW design
dominates it on every axis that matters:

- **Reset:** software COW resets by *dropping* per-lane overlays (set directory entries to SENTINEL,
  clear the page vec) — **zero copy-back**, since golden is never mutated. That's even faster than
  `reset_dirty` (no memcpy at all), and far faster than `MADV_DONTNEED`.
- **Memory:** `Arc<Golden>` shares the golden image across every thread *and* lane; per-lane overlays
  hold only dirtied pages — same footprint win as host-mmap's lazy fault-in, but real cross-thread
  sharing (`Arc`), in one image.
- **Safety/portability:** all safe Rust, no `unsafe`/`libc`/Linux-signal machinery, `forbid(unsafe_code)`
  intact.
- **Serves both paths:** the same `CowRam` over `Arc<Golden>` works for the scalar `--jobs` threads
  *and* the vectorized `VecSystem` lanes — one memory backend, not two.
- The host-mmap microbench's 1.46× is on a pure-memory workload; the real fuzzer's per-case time is
  dominated by *running the kernel* (millions of guest instructions), so a raw-memory speedup shrinks
  end-to-end, while its worse reset (run every case) would hurt.

`fs-hostmem` stays in-tree as a reference + reproducible benchmark (`cargo run -p fs-hostmem
--example bench --release`), but is not wired into the fuzzer. Next: build software COW (`CowRam` in
fs-mmu, PR1) per `docs/cow-shared-ram.md` — and it can retrofit the scalar `--jobs` path too, not just
`VecSystem`.
