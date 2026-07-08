# fuzzsoft — Design Decisions (locked)

Authoritative record of decisions made during the design phase. Companion to `docs/architecture.md`.
Status legend: **LOCKED** (user-confirmed) · **DEFAULT** (adopted from research digest, not overridden — change freely) · **OPEN** (still to decide, noted for the milestone it blocks).

## Core architecture

| # | Decision | Choice | Status | Milestone |
|---|----------|--------|--------|-----------|
| 1 | Language | Rust (nightly, edition 2024) | LOCKED | all |
| 2 | Target ISA | RISC-V **RV32IMAC** full-system | LOCKED | all |
| 3 | Execution model end-state | **Hybrid**: AVX-512 SIMD on hot common paths, scalar-execute divergent tails, re-converge at common PCs. Scalar core built & validated first (M0–M3); vectorize at M4. | LOCKED | M4 |
| 4 | Vectorization substrate | AVX-512, 16× u32 lanes/ZMM = 16 guests/thread; 32 cores → ~512 VMs baseline (16 lanes/thread) | LOCKED / DEFAULT topology | M4–M5 |
| 5 | Firmware / SBI boundary | **Run real OpenSBI in emulated M-mode** (full M-mode CSR/trap surface + M→S transition). *Override of the "emulate SBI on host" recommendation.* | LOCKED | M2 |
| 6 | Snapshot & input injection | **Post-boot snapshot** at a reserved hypercall from a tiny guest init; inject fuzz input by **patching syscall args into guest registers** | LOCKED | M3 |
| 7 | Determinism | Enforced from first boot: virtual time = f(retired instructions), fixed `timebase-frequency`, all host entropy stubbed/seeded, fixed interrupt schedule. Any leak = P0. | DEFAULT (mandatory) | M2+ |

## Emulator internals

| # | Decision | Choice | Status | Milestone |
|---|----------|--------|--------|-----------|
| 8 | Decoder seam | **Decode-to-IL enum from M0**: `fs-riscv` decodes to a typed Op/DecodedInst consumed by BOTH the scalar interpreter and (later) the hybrid AVX-512 executor. One validated decoder, no second hand-written one. | LOCKED | M0 |
| 9 | ISA rollout | **rv32im first** (M0), add **A + C** in M1 (C = table-driven, differential-tested before M2 boot). Soft-float `rv32ima` — no F/D. | LOCKED | M0–M1 |
| 10 | Soft MMU | Byte-level shadow perms `READ/WRITE/EXEC/RAW/ACC`; RAW uninit-read oracle; bump allocator w/ guard holes. Ported ~verbatim from Falk. | DEFAULT | M0+ |
| 11 | Reset | Dirty-block (64B) list + dedup bitmap; restore contents+perms+regfile+allocator together; O(bytes dirtied). Bounded dirty list. | DEFAULT | M3 |
| 12 | Guest RAM | Small per-lane (~64–128 MiB), flat interleaved, single shared union dirty list. **No CoW until M5.** | DEFAULT | M3+ |
| 13 | RAW/uninit policy | Off for kernel pages (const `DISABLE_UNINIT` on boot image/zeroed pages), on for fuzz-controlled buffers/heap. | DEFAULT | M3 |
| 14 | Timer | Prefer **Sstc** (`stimecmp`); SBI `set_timer`+CLINT `mip.STIP` fallback. Driven off retired-instruction count. | DEFAULT | M2 |
| 15 | Memory map | Match **QEMU `virt`** so stock kernels/DTBs align and we can cross-check. | DEFAULT | M2 |

## Coverage & fuzzing

| # | Decision | Choice | Status | Milestone |
|---|----------|--------|--------|-----------|
| 16 | Coverage representation | **Exact edge/block set (HashSet) in M0** for easy debugging; **swap to 64KB AFL-style `hash(prev_pc,pc)` bitmap before M4.** | LOCKED | M0 → M4 |
| 17 | Coverage source | Falls out of the emulator (instrument dispatch); no kcov. | DEFAULT | M0 |
| 18 | Coverage scope | **Whole guest (kernel + user)** — record all edges, not just S-mode. *Override of the kernel-only recommendation.* | LOCKED | M2 |
| 19 | Bug oracle | **Soft-MMU as primary sanitizer for the whole kernel** — hook guest allocator (kmalloc/kfree, buddy) to stamp guard/poison perms (M3) — **plus panic/oops/WARN/BUG detection** by watching guest PC hit those symbols. No in-kernel sanitizers by default. | LOCKED | M2–M3 |
| 20 | Fuzz input model | **Minimal syzkaller-style syscall descriptions** (typed args: ints, ptr-to-buffer, fd, flags); generate/mutate typed programs, lower to register injection. | LOCKED | M3 |
| 21 | Crash dedup | By `(pc, fault-type, address-class)`; reproduction free via determinism. | DEFAULT | M3 |

## Toolchain & validation

| # | Decision | Choice | Status | Milestone |
|---|----------|--------|--------|-----------|
| 22 | Guest toolchain | **Clang/LLVM for everything** incl. the kernel (`make LLVM=1`). No GNU riscv gcc. | LOCKED | M0+ |
| 23 | SIMD stack | **rustup + nightly**, `std::simd` (portable_simd) for `fs-vec`. Installed. | LOCKED | M4 |
| 24 | ISA differential oracle | **Spike** (`--enable-commitlog`) canonical + **qemu-riscv32** user-mode cross-check. Installed. | LOCKED | M1 |
| 25 | M1 diff harness | **Per-instruction commit-log lockstep vs Spike** + random-RV32-instruction fuzzing. riscv-tests (built with clang) as seed suite. | LOCKED | M1 |
| 26 | M0 test inputs | **Both**: in-repo hand-encoded RV32 ELF (hermetic) + a few clang-built bare-metal `.S`/`.c` samples. | LOCKED | M0 |
| 27 | Halt protocol | Support **both HTIF `tohost`/`fromhost`** (riscv-tests) **and ECALL `a7=93` exit(a0)** (our samples). | LOCKED | M0 |
| 28 | Kernel source | **Fresh shallow checkout of latest mainline stable** (confirm exact tag at M2), configured `rv32ima` soft-float **nosmp**; decoupled from the in-repo x86-built tree. *Override of the LTS recommendation — newest rv32 code over long-term stability.* | LOCKED | M2 |
| 29 | Full-system boot oracle | **Build `qemu-system-riscv32` from the in-repo `qemu/` tree** (configure `riscv32-softmmu`), since installed `qemu-system-misc` doesn't ship it. | LOCKED | M2 |
| 30 | OpenSBI firmware mode | **fw_jump** — OpenSBI at M-mode reset jumps to a fixed compile-time address where our loader placed the kernel. | LOCKED | M2 |
| 31 | Rootfs / userland | **M2:** tiny static `nostdlib` init (clang, raw `ecall` syscalls) in an initramfs (hermetic, proves S→U). **M3:** add busybox/buildroot userland for a real syscall surface. | LOCKED | M2–M3 |

## Fuzzing internals (M1 / M3)

| # | Decision | Choice | Status | Milestone |
|---|----------|--------|--------|-----------|
| 32 | Random-instruction generator | **Both**: validity-biased stream (well-formed opcodes/fields, stress C-ext scrambled immediates) + a fraction of pure-random 32/16-bit words for illegal-encoding paths. Diff each step vs Spike. | LOCKED | M1 |
| 33 | Compare-coverage (cmpcov) | **Edge-coverage first** in M3; add AFL++/Falk-style compare-coverage (branch-operand instrumentation, matching-byte progress → mutator) as a fast follow. | LOCKED | M3 |
| 34 | Corpus on-disk format | **Single append-only database file** (à la syzkaller `workdir/corpus.db`) — compact, few inodes. | LOCKED | M3 |
| 35 | First target surface | **Broad from the start**: auto-enumerate the rv32 syscall table and fuzz broadly (descriptions cover the table), rather than a focused smoke-target. | LOCKED | M3 |
| 36 | Corpus sync (multi-core) | **Per-worker corpora + periodic merge to a shared master** (AFL/Falk master-secondary); minimal lock contention at 32 threads. | LOCKED | M5 |
| 37 | Crash reproducer | **Full syzkaller-style standalone minimized C reproducer from the start**, in addition to deterministic raw-case replay via `fuzzsoft repro`. *Override of the "raw replay first" recommendation.* | LOCKED | M3 |
| 38 | Snapshot persistence | **Serialize the post-boot golden snapshot to disk, mmap on startup** (boot once, restore instantly per launch). | LOCKED | M3 |
| 39 | Run config format | **TOML config file** (target, cores, VM count, guest RAM, kernel/snapshot/corpus paths); CLI flags override. *Override of the JSON/manager.cfg recommendation.* | LOCKED | M1+ |
| 40 | DTB source | **Hand-author a `.dts`, compile with `dtc`** (minimal nodes: cpus+timebase, memory, chosen/bootargs, initrd). *Override of the "dump QEMU DTB" recommendation.* | LOCKED | M2 |
| 41 | Telemetry | **Terminal live stats** (exec/s, edges, corpus, crashes, VM utilization) now; syzkaller-style **HTTP dashboard** at M5. | LOCKED | M3 → M5 |
| 42 | Mutators | **Full syzkaller-style set**: typed arg havoc, insert/remove syscalls, splice/merge programs, dependency-aware (fd/resource) fixups. | LOCKED | M3 |
| 43 | Seed corpus | **Start fully empty** — pure coverage-driven growth. *Override of the hand-seeds recommendation.* | LOCKED | M3 |
| 44 | Kernel build | **Modular (loadable `.ko`)** so module-load paths are fuzzable. *Override of monolithic.* ⚠ Interacts with determinism (#7): runtime module loading during fuzzing must be deterministic; pre-snapshot loads are free. | LOCKED | M2 |
| 45 | Hot-path safety | **Correctness-first in safe Rust**, targeted `unsafe` in profiled hot spots (translate/dispatch/block-copy) only when justified. | LOCKED | all |
| 46 | Mutation dictionary | **Static token dictionary** (syscall nrs, flags, ioctl codes, magic constants) **+ auto-extract operands from comparisons** once cmpcov lands. | LOCKED | M3 |
| 47 | Exec budget / hang detection | **Deterministic per-case instruction-count budget** (exceed = timeout). Matches determinism (#7); no wall-clock. | LOCKED | M3 |
| 48 | Generate vs mutate | **Mostly mutate corpus, periodically generate fresh** programs from descriptions for diversity. | LOCKED | M3 |
| 49 | Fault injection | **Yes, M3 fast-follow**: systematic failure injection (fail_nth-style) to reach kernel error/cleanup paths, added just after the base loop works. | LOCKED | M3 |
| 50 | CI | **Deferred** — add hosted CI once the workspace stabilizes; local checks in the meantime. | LOCKED (deferred) | later |
| 51 | Kernel-heap oracle (near-term) | **Ship a second `slub_debug` kernel variant** (`firmware/Image.slubdebug`, built with `CONFIG_SLUB_DEBUG_ON=y` via `scripts/build-slubdebug-kernel.sh`): the *allocator* red-zones/poisons every slab object and oopses on corruption, which the existing `kernel_crash_sig` console oracle already catches — **zero emulator poisoning**. Chosen because emulator-side redzone poisoning false-positived ~40% on stock SLUB (adjacent live objects). Stock `firmware/Image` stays the default for throughput; run heap campaigns with `--kernel firmware/Image.slubdebug` (no `--sanitize`). Tradeoff: slower boot/runtime, but true positives. In-emulator KFENCE remains the future uninstrumented path. | LOCKED | M3 |
| 52 | Host codegen (`target-cpu=native`) | **Build the whole workspace with `-C target-cpu=native`** via `.cargo/config.toml`. rustc's default x86-64 target enables only sse/sse2, so `std::simd::Simd<u32,16>` silently compiles to 4× 128-bit SSE2 (verified by disassembly: 0 zmm) — the vectorized emulator gets *no* AVX-512 without this. With it, real `zmm`+k-mask code is emitted and the fs-vec bench goes ~3-4× → ~9-11× over scalar-per-lane; the scalar fuzzer also gains ~10%. Integer emulation is bit-identical regardless of SIMD width, so this is a pure perf/determinism-neutral change. Tradeoff: binaries become host-specific — fine for this pinned single-host research project (AMD 7945HX, Zen 4). | LOCKED | M4 |

## Minor defaults adopted (no need to decide; change anytime)
- Guest RAM: **128 MiB/lane** to start (fits rv32 kernel+initramfs; revisit vs host budget at M4/M5).
- Logging: `tracing` crate; errors: `thiserror` in libs, `anyhow` at binary boundaries.
- Snapshot serialization: custom little-endian binary blob + a small TOML/JSON manifest (versioned).
- Reproducer C: compiled static with clang (`--target=riscv32`), runnable under our emulator and qemu-riscv32.
- Endianness: RV32 little-endian only. Kernel cmdline via DTB `/chosen/bootargs`.

## Rust workspace (from architecture §8)

`fs-mmu`, `fs-riscv` (decoder+IL+scalar interp), `fs-arch` (CSR/trap/sv32/M-mode/OpenSBI host glue), `fs-sbi`, `fs-platform`, `fs-loader`, `fs-snapshot`, `fs-cov`, `fs-vec` (M4), `fs-fuzz`, `fs-cli`. M0 creates `fs-mmu`, `fs-riscv`, `fs-cov`, `fs-loader`, `fs-cli`; the rest are added as their milestone arrives.

## Open items to revisit at their milestone
- **#18** coverage scope (kernel-only masking) — confirm at M2.
- **#29** `qemu-system-riscv32` full-system oracle — resolve at M2.
- OpenSBI firmware mode (fw_jump / fw_dynamic / fw_payload) and building it with clang — M2.
- Rootfs = initramfs (busybox/musl via buildroot or hand-rolled) — M2/M3 (digest default: initramfs).
- Random-instruction generator constraints; corpus on-disk format; M5 multi-core corpus sync.
