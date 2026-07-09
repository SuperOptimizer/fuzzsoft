# fuzzsoft roadmap — more fuzzing, more coverage, more speed, better sanitizing

Living plan for advancing fuzzsoft after the scalar-JIT milestone. Organized as parallel **tracks**,
executed in **waves**, with each leaf task tagged `[executor · crate(s) · gate · depends-on]`. The
project's non-negotiable discipline runs through everything: **every change is differential-gated**
(bit-exact vs the interpreter and/or Spike; sanitizers get a planted-positive + clean-negative
control). Correctness is the deliverable; speed/coverage/bugs are the measurements.

## Where we are (baseline, 2026-07-08)
- **Emulator**: RV32IMAC full-system, Spike-validated, byte-granular soft-MMU, deterministic
  snapshot/dirty-reset, COW-shared golden RAM (flat memory at any `--jobs`).
- **Fuzzer**: coverage-guided, 90+ resource-threaded syscall descriptions, constant dictionary,
  CMPLOG/RedQueen, same-skeleton batch mutation, corpus persistence, `--jobs` parallel.
  ~5.5k execs/s @ jobs=32; **scalar chain-JIT (`--jit-chain`) = 2.9× single / ~1.8× @ jobs=16.**
- **Sanitizers (emulator-native, uninstrumented guests)**: KASAN (slack + page, 0-FP), UBSAN div0,
  KMSAN Stage 1 core (register taint — *not yet a live oracle*).
- **Bug-finding infra**: fault injection (`fail_nth`, `Image.failinj`, validated chain), fs-covmap
  coverage attribution, `--dump-edges`.
- **Known structural limits**: `bpf/io_uring/keyctl/mount/namespaces/splice` have **0** fs-prog
  descriptions (the coverage plateau's real cause); **nosmp** excludes the race/TOCTOU bug class;
  `MAX_CALLS=8` shallow-limits completion-loop subsystems; vectorized-JIT is a measured NO-GO.

---

## T0 — Measurement & attribution (cross-cutting, continuous)
Know where you are before optimizing. Feeds every other track's targeting.
- **T0.1** Per-case cost profiler: break down where time goes now (JIT native-chain vs fallback
  single-step vs snapshot reset vs mutex wait vs boot-amortized). `[agent · fs-cli/fs-jit · report ·
  —]`
- **T0.2** Coverage-frontier tool: run fs-covmap on live campaign corpora → rank **reachable but
  uncovered** kernel functions (the targeting signal for T2). `[agent · fs-covmap+scripts · report ·
  —]`
- **T0.3** Bug-yield / campaign dashboard: crashes, unique PCs, coverage growth curve, plateau
  detection, per-subsystem attribution over time. `[agent · scripts · artifact · T0.2]`

## T1 — Throughput (make each core faster)
- **T1.1 Shared-state contention fix (HIGHEST throughput lever).** At jobs=16 `futex` is ~75% of
  syscall time: workers serialize on the shared coverage-map + corpus mutex, hiding the JIT's
  per-case win. Shard coverage into per-thread maps with periodic lock-free-ish merge; per-thread
  corpus sampling with batched sync. `[agent · fs-cli · parallel results unchanged vs baseline +
  execs/s scales · —]`
- **T1.2 JIT Phase 3 — inline TLB/perm fast path.** The load/store call-out is the remaining
  single-thread cost. Inline the sv32 TLB lookup + byte-perm check in emitted code, call out only on
  miss/fault/MMIO. Highest codegen risk. `[agent(worktree) · fs-jit/fs-mmu · differential incl.
  perm-fault/RAW/sanitizer-bit exactness + Spike + bench · scalar JIT done]`
- **T1.3 JIT polish.** Cross-thread shared read-only compiled-code cache (cut redundant per-worker
  compiles), superblock/trace formation past single-branch chains, indirect-branch target caching
  for `Jalr`. `[agent(worktree) · fs-jit · differential + bench · T1.2]`
- **T1.4 Reset/snapshot tiering** if T0.1 shows reset cost material (already dirty-block; consider
  generational/region reset). `[agent · fs-platform/fs-mmu · byte-exact reset + bench · T0.1]`

## T2 — Coverage (reach more code)
The plateau is structural (missing descriptions), not a mutator weakness — fs-covmap proved it.
- **T2.1 Cheap subsystems (WORKFLOW fan-out, one agent per subsystem).** Each adds fs-prog
  descriptions + resource threading and proves a coverage lift via fs-covmap. Independent, highly
  parallel: `splice/vmsplice/tee`, `unshare/setns/namespaces`, `keyctl/add_key/request_key`,
  `timerfd/eventfd/signalfd/inotify/fanotify`, `prctl`, `process_vm_readv/writev`, `io_uring`-adjacent
  `eventfd`. `[workflow · fs-prog · per-subsystem: fs-covmap covered/total rises from ~0 + all
  descriptions lower & run clean · —]`
- **T2.2 Scaffolded subsystems.** Need guest-side setup: `mount/overlayfs/tmpfs/ext4` (initramfs +
  loop devices), deeper `netlink` families, richer socket-option trees. `[agents · fs-prog +
  boot/initramfs · coverage lift · T2.1]`
- **T2.3 High-density hard subsystems (epics).** `io_uring` (SQE/CQE grammar, submission→completion
  loops) and `bpf` (a mini in-guest BPF program generator). Highest bug density, architecturally
  strained; each needs T2.5 + T2.6. `[workflow+agents(worktree) · fs-prog · coverage + error-path
  reach · T2.5, T2.6]`
- **T2.4 Deeper chaining.** Raise `MAX_CALLS`, richer resource-type graph, and a **branch-on-guest-
  output** wire-format extension so programs can react to a syscall's return (retry-on-EAGAIN,
  fd-from-completion) — the shape io_uring/epoll bugs need. `[agent · fs-prog/fs-cli · deeper chains
  lower & run + coverage · —]`
- **T2.5 Grammar-aware generation framework.** A reusable structured-input layer (typed
  sub-grammars) for complex opaque syscall args (bpf insns, io_uring SQEs, netlink attrs). `[agent ·
  fs-prog · unit-tested generators · —]`
- **T2.6 Corpus distillation at scale.** Periodic coverage-minimization + cross-campaign corpus
  merge, seeded from T0.2's frontier. `[agent · fs-cli/scripts · minimized corpus preserves coverage
  · T0.2]`

## T3 — Sanitizing (detect more bug classes on uninstrumented guests)
- **T3.1 KMSAN Stage 1.5 — wire the register-taint as a LIVE ORACLE (HIGH value, turns shipped work
  into a detector). DONE, but the headline finding raises T3.2's priority.** `--kmsan` is wired
  (`Cpu::kmsan_hit` stashed by `finish_exit`, drained + minimized + reproduced like the crash
  oracle; synthetic positive/negative controls pass). The clean-kernel false-positive measurement
  (4000 cases) found **0 hits — but for a structural reason, not a precision one**: Stage 1's only
  taint entry point (`Load` gathering `PERM_RAW`) has no live producer reachable outside
  `--sanitize`'s allocator hooks (which `--kmsan` is, by design, forbidden to combine with), so a
  live `--kmsan` run currently has zero taint sources at all — confirmed dormant even with
  `--sanitize`'s hooks active in a one-off diagnostic. See `docs/kmsan.md`'s "T3.1 outcome" section.
  `[agent · fs-cli/fs-riscv · positive (planted uninit-branch fires) + negative (clean = 0) controls
  · KMSAN Stage 1 (done)]`
- **T3.2 KMSAN Stage 2 — memory shadow.** `PERM_VTAINT` spare perm bit; loads OR in RAW+VTAINT;
  store→reload→branch round-trip propagation. Makes it true *value* taint. `[agent(worktree) ·
  fs-mmu/fs-riscv · positive+negative + store-reload test · T3.1]`
- **T3.3 KMSAN Stage 3/4 — precision + sinks.** Carry-smear (sound Add/Sub), known-byte clearing,
  tainted-shift; `copy_to_user`/`put_user` sink hooks; syscall-return-taint checkpoint. `[agent ·
  fs-riscv/fs-san · per-refinement positive+negative · T3.2]`
- **T3.4 KASAN depth.** Quarantine/delayed-free (stronger UAF), `alloc_pages`/struct-page tracking,
  allocator-cooperation to catch packed-neighbor OOB (current hard ceiling). `[agent · fs-san ·
  planted-UAF/OOB positives + 0-FP · —]`
- **T3.5 UBSAN breadth.** Shift-past-width, signed-overflow, misalignment, null-deref, array-bounds
  where derivable. `[agent · fs-riscv · per-check positive+negative · —]`
- **T3.6 Sanitizer × `--jobs`.** `--sanitize` is serial-only (per-instruction PC-hooks). Make the
  hook state per-worker so sanitizer campaigns parallelize; or document the serial-throughput
  tradeoff. `[agent · fs-cli · parallel sanitizer results == serial · T1.1]`
- **T3.7 KCSAN** — parked; unblocked only by T5 (SMP). Data-race detection over shared memory.

## T4 — Bug-finding (actually hunt, triage, prove)
- **T4.1 Fault-injection campaigns at scale.** Now that the JIT + `--fail-inject` + `--jobs` compose,
  run sustained `Image.failinj` campaigns targeting error/cleanup paths (where UAF/double-free/leak
  concentrate). `[campaign · — · bugs or honest-clean-with-coverage-evidence · T1.1 (for speed)]`
- **T4.2 Real-CVE pipeline validation.** Build a kernel with a *reverted* known single-task bug
  (e.g. an epoll/integer-overflow, non-race), prove find→minimize→reproduce end-to-end organically.
  `[agent · boot/scripts/fs-prog · oracle fires + reproducer replays · —]`
- **T4.3 Crash triage upgrade.** Stronger dedup (type+PC+addr), auto-minimization (have),
  standalone C-reproducer emission, bucket clustering + report artifacts. `[agent · fs-cli · replay
  fidelity · —]`
- **T4.4 Multi-config kernel matrix.** Fuzz several `.config`s (SLUB_DEBUG, KASAN-config, debug knobs)
  in rotation; attribute bugs to configs. `[agent · scripts/boot · matrix runs · T0.3]`
- **T4.5 RV32/arch-specific surface (the unique edge).** Target what an x86 fleet structurally can't:
  AMO/LR-SC edge cases, unaligned access, RV32 syscall-marshalling/compat paths. `[agent · fs-prog/
  fs-diff · coverage of arch-specific paths · —]`
- **T4.6 Continuous campaign infra.** Long-running, resumable, monitored campaigns with alerting on
  new unique crashes. `[agent · scripts · uptime + resume · T0.3]`

## T5 — Concurrency / SMP (the biggest structural unlock — its own epic)
`nosmp` excludes the **entire race/UAF-via-race/TOCTOU/lock-ordering bug class** — a large share of
real syzbot findings. This is the single largest expansion of detectable bugs, and the largest
effort. Gate it behind a **feasibility spike** before committing.
- **T5.0 Feasibility spike.** Scope multi-hart cost against the current single-hart architecture
  (per-hart CLINT/`msip` IPI, shared memory coherence in the soft-MMU, SBI HSM, deterministic
  interleaving model). `[workflow(design panel) · report+blueprint · —]`
- **T5.1 Multi-hart emulator** (2+ harts, shared RAM, IPIs, per-hart timers). `[agents(worktree) ·
  fs-riscv/fs-platform/fs-mmu · boots SMP Linux + Spike-consistent per-hart · T5.0]`
- **T5.2 Deterministic interleaving exploration** (controlled preemption points, PCT-style schedule
  search) — reproducible races. `[agent · fs-platform/fs-cli · known race reproduced deterministically
  · T5.1]`
- **T5.3 KCSAN-style data-race detection** over the shared soft-MMU (happens-before / watchpoints).
  `[agent · fs-mmu/fs-san · planted race detected + 0-FP · T5.1]`

---

## Execution: waves

**Wave 1 (now — highest value, parallelizable, low dependency; these compound):**
- T1.1 shared-state contention fix `[fs-cli]` — unlocks the JIT win across all cores.
- T2.1 cheap-subsystem coverage `[workflow · fs-prog]` — most reachable new code per unit effort.
- T3.1 KMSAN live oracle `[fs-riscv/fs-cli]` — turns shipped taint-tracking into a bug detector.
- T4.1 fault-injection campaign `[background]` — hunt while the code work proceeds.
- T0.1/T0.2 measurement `[agent]` — targeting for everything after.
(Disjoint-crate scheduling: T1.1 and T3.1 both touch fs-cli → sequence or one combined agent; T2.1
is fs-prog, T4.1 is a campaign — both fully parallel.)

**Wave 2 (after Wave 1 lands):** T1.2 JIT Phase 3, T2.2 scaffolded subsystems, T2.4 deeper chaining,
T3.2 KMSAN memory shadow, T3.4 KASAN depth, T4.2 real-CVE validation, T0.3 dashboard.

**Wave 3 (epics, after their prereqs + a go/no-go):** T2.3 io_uring/bpf (needs T2.5/T2.6), T3.3 KMSAN
precision/sinks, T4.4 config matrix, T5.0→T5.1 SMP spike then build.

## Agentic model
- **Workflows** for fan-out with a shared shape (per-subsystem descriptions; multi-lens bug
  verification; design panels for epics like SMP).
- **Agents in worktrees** for deep single-crate work (JIT phases, sanitizer stages, SMP).
- **Background campaigns** for the actual hunting.
- **Main loop (lead)** integrates via cherry-pick, runs the correctness gate on every landing,
  serializes edits to shared files (fs-cli), and prunes worktrees.

## Honest ceilings (no overselling)
- **SMP** is the biggest payoff and the biggest cost — weeks-to-months, gate behind the spike.
- **io_uring/bpf** are the highest bug density but architecturally strained (opaque grammars,
  completion loops vs `MAX_CALLS`); expect real effort for real reach.
- **Sanitizer limits are inherent**: KASAN can't beat a packed allocator without cooperation; KMSAN
  v0 rules over-taint until Stage 3. State them; don't paper over them.
- The compounding near-term wins are **T1.1 (throughput) + T2.1 (coverage) + T3.1 (a new detector) +
  T4.1 (hunt)** — Wave 1 is deliberately the highest ratio of value to risk.
