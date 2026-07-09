# SMP feasibility & phased blueprint (T5.0 spike output)

Status: design doc, produced from five independent code-grounded feasibility assessments plus
direct verification against the current tree (2026-07-08, commit `711dd33`). This is the T5.0
deliverable referenced in `docs/roadmap.md`; it answers the go/no-go question and, since the
answer is GO-IF, lays out the phased build.

## 0. Verdict

**GO-IF.** Build the 2-hart mechanical spike and the deterministic-race-repro proof point now
(bounded, ~4-9 weeks, cheap to abandon); gate the larger, open-ended investment (coverage-guided
concurrent program generation, PCT-style schedule search, KCSAN-lite detection) behind that proof
actually reproducing a real race at a usable rate. The single biggest determining factor is **not**
whether SMP can be made deterministic — it can, cheaply, because fuzzsoft already runs everything
on one host thread with virtual time as a pure function of retired instructions, so N harts
cooperatively round-robined against one shared `Mmu` (never real OS threads) stays exactly as
reproducible as today's single-hart model. The real crux is **interleaving-schedule quality**: a
fixed round-robin is trivially deterministic but explores exactly one interleaving forever, and a
naively "random" schedule may simply never land on the narrow window a given TOCTOU/UAF-via-race
bug needs. Whether a modest, PCT-style seeded scheduler (preemption points biased toward AMO/LR-SC/
syscall boundaries) finds real races at a rate that justifies the engineering is an empirical
question the codebase cannot answer on paper — it has to be measured against a known, previously
un-reachable-under-`nosmp` race (docs/bug-finding.md already names one: the epoll UAF-via-race
`a6dc643c6931`, proven "provably unreachable" on the current single-hart build). That experiment is
cheap enough to run before committing to the full epic, and its result is the actual go/no-go gate.

Grounded reason to believe the mechanics are tractable, not just aspirational: `Cpu` (crates/
fs-riscv/src/lib.rs:817) is already a `#[derive(Clone)]`, fully self-contained per-hart value (regs,
pc, CSRs including a real but hardcoded-to-0 `mhartid` field at sys.rs:135/172, its own sv32 TLB,
its own LR/SC `reservation: Option<u32>`) with no global/static/thread-local state anywhere in
fs-riscv/fs-mmu/fs-platform. `fs-cli::run_parallel` (main.rs:2227) already clones a `Cpu` per worker
today, so `Vec<Cpu>` is not new plumbing. Real OpenSBI is already vendored and compiled in
(`third_party/opensbi`, decision #5 in docs/decisions.md — LOCKED override of "emulate SBI on
host"), and its HSM/IPI implementation (`lib/sbi/sbi_hsm.c`, `sbi_ipi.c`) is present and derives
`hart_count` purely by walking `cpu@N` nodes in the FDT at coldboot (`platform/generic/
platform.c:148-186`) — so hart bring-up is primarily a **DTB change** (today's `boot/fuzzsoft.dts`
has exactly one `cpu@0` and `bootargs = "... nosmp ..."`), not new firmware. This means fuzzsoft
does **not** need to write HSM/hart-start from scratch, which removes one of the biggest
theoretical unknowns.

## 1. Minimum viable SMP (the proof-of-value slice)

**Not general N-hart. Exactly 2 harts, one shared `Machine` (never `CowMachine`/`CowRam` — that
type exists specifically to give independent fuzzing *lanes* private, mutually invisible COW
overlays, which is the opposite of what SMP coherence needs within one guest), one host thread, a
fixed-quantum round-robin scheduler, and a hand-planted or hand-directed real race.**

Concrete slice:
1. `mhartid` plumbing: `Cpu::new` takes a hart index instead of hardcoding 0 (sys.rs:172).
2. `Vec<Cpu>` (size 2) driven by a new outer scheduler in fs-platform, replacing the flat
   `while insns_retired < deadline { step_system }` in `run`/`run_until` (lib.rs:378-403).
3. `Clint` (lib.rs:22-27, today one scalar `msip`/`mtimecmp` pair) becomes per-hart arrays indexed
   by `(offset − base)/stride`; `mtime` becomes a function of a scheduler-owned global tick, not any
   one hart's `Cpu::virtual_time()` (today `insns_retired`, lib.rs:1756 — a single-hart identity
   that does not generalize).
4. Cross-hart LR/SC invalidation: today `reservation` (lib.rs:824) is cleared only by its own
   hart's SC or trap (lib.rs:1544-1550) — a second hart's plain store to the same word does **not**
   invalidate hart 0's reservation. This is the single most correctness-critical addition: get it
   wrong and SC spuriously succeeds concurrently with another hart's write, which either fabricates
   impossible "races" (emulator artifacts, not kernel bugs) or masks real ones. Needs a small
   shared-reservation check threaded into every store/AMO regardless of issuing hart.
5. `Snapshot` (fs-platform:310-341, today `cpu: Cpu` singular) becomes `cpus: Vec<Cpu>` — mechanical,
   same all-or-nothing reset discipline, just iterate.
6. Real-OpenSBI HSM hart-start against the existing `HC_SNAPSHOT` hypercall path in fs-cli, OR (as
   an explicit scope-reduction for the spike only) a hand-stubbed "both harts running from reset" if
   HSM bring-up proves to eat unplanned time.
7. `boot/fuzzsoft.dts`: add `cpu@1` + intc + extend `clint@...`'s `interrupts-extended`; drop
   `nosmp` from `bootargs`.
8. Proof test: either (a) a bare-metal two-hart asm program doing LR/SC ping-pong / AMOADD on a
   shared counter, checked against a sequential reference and replayed for bit-identical output
   across repeated runs, or — preferably, since it's the thing that actually answers the go/no-go
   question — (b) the real, already-identified epoll UAF-via-race (`a6dc643c6931`) reverted into the
   kernel build, driven by a **hand-written, fixed** 2-thread call schedule (not yet a
   coverage-guided generator), confirming it crashes under some recorded interleaving, does not
   crash on sequential/1-hart execution (already true today), and reproduces 100% given the
   schedule seed.

**Effort:** 1-3 weeks for the mechanical slice (items 1-7; medium-high confidence, directly grounded
in code read), plus the bring-up/debugging tail against a real `-smp 2` OpenSBI/HSM path exercised
for the first time (unknown until run — no existing test coverage here). All five independent
assessments converge on this range; treat 1-3 weeks as the mechanical floor, not the full proof.

## 2. Phased build order

Each phase is independently testable and has its own determinism/reproducibility gate. Do not start
a phase until the previous phase's gate passes.

### Phase 0 — Design (this document)
Output: this doc, reviewed and locked into `docs/decisions.md` alongside the other LOCKED items
before Phase 1 starts (in particular: shared-`Machine`-not-`CowMachine` as the SMP substrate, and
"never real OS threads across harts" as a permanent constraint, not an optimization left open).

### Phase 1 — 2-hart mechanical boot to snapshot
Build items 1, 2, 3, 5, 6, 7 from §1 (everything except cross-hart LR/SC and the race proof).
Fixed-quantum round-robin (e.g. 64 or 10k instructions/hart-turn — arbitrary for this phase).
- **Gate:** two independent runs from the same snapshot with the same fixed schedule produce
  byte-identical guest RAM + all register/CSR state. This is the direct generalization of the
  project's existing determinism claim (virtual time = f(retired instructions), decision #7) from 1
  hart to N, and must hold before anything else is trusted.
- Also confirms end-to-end: DTB → OpenSBI HSM → Linux `secondary_start_kernel` on hart 1 → shared
  hypercall reachable from either hart.

### Phase 2 — Cross-hart coherence correctness
Add the shared reservation-invalidation table (item 4). Extend the project's existing
Spike-differential discipline (decisions #24/#25) to a 2-hart LR/SC/AMO interleaving cross-check.
- **Gate:** the hand-written spinlock/shared-counter test (§1 item 8a) matches a sequential
  reference (no lost updates, no spurious SC successes) **and** is byte-identical across repeated
  replays of the same schedule. Do this before combining with fs-jit or a real kernel — a wrong
  reservation model is silent (not a crash) and corrupts every downstream result.
- Explicitly decide and document the memory-model scope here: single-host-thread cooperative
  scheduling gives free sequential consistency (stronger than real RVWMO) — sound (no false
  positives) but unable to find pure store-reordering/missing-barrier bugs. This is an acceptable,
  intentional trade for the TOCTOU/lock-ordering/UAF-via-race class this project targets, but must
  be stated as a known scope limit, not discovered later as a surprise gap.
- Decide whether fs-jit stays gated off (interpreter-only) for this milestone. The existing
  `run_parallel` design (chains take fresh `cpu`/`bus` pointers per call, `last_edge` drained per
  call — chain.rs:175, 390) suggests the chain cache is bus-agnostic and safely shareable across
  harts, but this has never been exercised under true multi-hart interleaving; treat that as an
  assumption to stress-test in this phase, not a settled fact. Separately, fs-jit's block cache is
  documented as never invalidated on code writes even today (fs-jit/src/lib.rs:20) — under SMP this
  stops being a perf-only gap (module load / ftrace / kprobes patching code on one hart while
  another hart has it JIT-cached is a real soundness bug), so keep JIT disabled for this phase
  regardless.

### Phase 3 — Deterministic interleaving as a fuzzable dimension
Replace the fixed round-robin with a seeded scheduler: a per-case PRNG (drawn from the same seed
stream that already drives mutation) picks preemption points, biased toward synchronization
boundaries (AMO/LR-SC, FENCE, satp/CSR writes, trap entry/exit, syscall entry/exit) rather than
uniform per-instruction — this is the PCT-style (Burckhardt et al.) approach the roadmap's T5.2
already gestures at, and it's necessary because a naive fixed or uniform-random schedule is known
to rarely land on narrow racy windows.
- **Gate:** same `(input, schedule-seed)` pair reproduces bit-identically on replay; different
  seeds visibly explore different interleavings (verify by logging the preemption-point sequence
  per run and diffing across seeds on the same input).
- The schedule seed must be captured by `Snapshot` and stored in the corpus/reproducer exactly like
  existing per-case RNG seed, or a found race stops being replayable — this is the one place where
  a subtle mistake most directly reintroduces the "ghost crash" failure mode decision #7 was written
  to prevent.

### Phase 4 — Race detection proof point (the actual go/no-go gate)
Run the §1 item 8b experiment for real: revert the epoll UAF-via-race CVE, drive it with the Phase
3 seeded scheduler (not yet a coverage-guided generator — a hand-written 2-thread call schedule
targeting the known racy syscall pair is enough), and measure whether it crashes within a bounded
exec/seed budget.
- **Gate — this is the decision point for the rest of the epic:** does the hand-directed schedule
  crash under *some* seed within a reasonable budget (say, low thousands of seed variations), does
  it reproduce 100% given that seed, and does it *not* crash sequentially on 1 hart (confirming the
  bug is genuinely interleaving-dependent and was genuinely unreachable before)? A clean pass here
  licenses Phase 5+. A failure to find the known race within budget is a legitimate signal that
  schedule-quality is the bottleneck it's suspected to be, and is grounds to stop and reassess before
  investing further (see §4).

### Phase 5 — Multi-thread fs-prog programs (only if Phase 4 passes)
`Prog` today is single-thread sequential (crates/fs-prog/src/prog.rs); `boot/agent.c` runs one
`do_syscall` loop. This phase adds a genuinely new generator/mutator axis: concurrent call
sequences with the schedule as a co-mutated dimension, targeting shared-resource contention (two
threads racing the same fd/mmap/file/inode) rather than arbitrary parallel syscalls. This is
comparable in size to fs-prog's existing resource-threading model — new surface, not a small patch.
- **Gate:** the coverage-guided concurrent generator finds *a* race (not necessarily the same named
  CVE) within a campaign, with the same reproduction guarantee as Phase 4.

### Phase 6 — Stretch (parked pending Phase 5 results)
KCSAN-lite data-race detector (fs-san, watching byte-granular last-writer/hart-id on the shared soft
-MMU — `docs/roadmap.md` T5.3) to catch silent races that don't crash; general N-hart scale-out
beyond 2; PLIC — confirmed **not needed** for any of this (IPI/HSM/timer all route through CLINT
only; fuzzsoft's UART is polled with no `interrupts-extended`, and Linux SMP bring-up/IPIs go
through `EXT_IPI`→CLINT MSIP, never PLIC).

## 3. Cost vs. payoff

**Payoff is categorical, not incremental.** `docs/bug-finding.md` already documents that the epoll
UAF-via-race is *provably unreachable* under the current `nosmp` build — this is not "SMP finds more
bugs of a kind we already find," it's a bug class (race/UAF-via-race/TOCTOU/lock-ordering — a large
share of real syzbot findings, historically) that is structurally excluded today and that no amount
of further T1-T4 investment (more coverage, sanitizer depth, fault injection) can ever reach. That
argues strongly for eventually doing this.

**Cost is real and this is the single widest-blast-radius item on the roadmap** — it touches
fs-riscv, fs-mmu, fs-platform, fs-cli, and (for the self-sustaining version) fs-prog simultaneously,
more crates at once than any other roadmap item.

Honest estimate:
- Phases 1-2 (mechanical 2-hart boot + coherence correctness): **1-3 weeks**, medium-high confidence.
- Phases 3-4 (seeded scheduler + the actual known-race proof point — this is the real go/no-go
  artifact): **total 4-9 weeks** from a cold start, medium confidence — consistent with the
  roadmap's own framing ("weeks-to-months, gate behind the spike").
- Phases 5-6 (coverage-guided concurrent fs-prog generator, KCSAN-lite, general N-hart, PCT-quality
  tuning): **open-ended, additional weeks-to-a-quarter**, low-to-medium confidence — this is genuine
  research/tuning work (schedule-quality effort is comparable to what dedicated concurrency-fuzzing
  tools invest), not sizeable on paper until Phase 4's yield data exists.

**Opportunity cost** (what gets deferred if SMP is prioritized now): T2.4 (directed same-resource
cycle threading, already in progress), the six subsystems with **zero** fs-prog descriptions
(bpf/io_uring/keyctl/mount/namespaces/splice — the documented real cause of the coverage plateau),
further fault-injection mining (T4.1, called out in the roadmap as "the #1 lever, cheapest, reuses
~100% of infra" and not yet fully mined), and deeper sanitizer work (KMSAN beyond Stage 2, KASAN
refinements). These are each days-to-low-weeks with already-measured or already-proven ROI (T1.1
shipped a measured throughput lift; T4.2 already validated the real-CVE pipeline end-to-end on the
single-hart path) — cheaper and lower-risk than SMP, but capped at the bug classes `nosmp`
structurally already permits.

**One more permanent, structural cost, not just a scheduling cost:** fs-vec's AVX-512 vectorization
(LOCKED decisions #3/#4 — 16 lockstepped single-hart guests per ZMM, the project's main ~512-guest
throughput multiplier) is architecturally incompatible with scheduler-driven SMP guests. Each
vectorized lane's PC must stay in lockstep with the others; a scheduler-chosen "active hart" whose
PC depends on accumulated scheduling history diverges across 16 independent SMP guests almost
immediately, breaking the lockstep/masking model vectorization depends on. **SMP guests can only run
through the scalar (+`--jit-chain`) `--jobs` path, at scalar-fleet throughput (~5.5k execs/s @
jobs=32 baseline), never the 512-guest target.** This must be communicated up front as "a second,
scalar-only, additive race-hunting fleet serving a different bug class," not "SMP support for the
fuzzer" — otherwise it reads as a regression against the project's main throughput thesis when it
is actually orthogonal to it.

**Recommendation: NOW for Phases 1-4 only.** They are bounded (4-9 weeks), cheap to abandon if the
Phase 4 gate fails, and they produce the one artifact that actually retires the risk everyone should
care about: does a deterministic, interleaving-scheduled 2-hart emulator reproduce a real,
previously-unreachable kernel race at a usable rate. Treat Phases 5-6 (the open-ended, multi-month
full build) as **explicitly gated on Phase 4's result** — not pre-committed. If Phase 4 fails to
reproduce the known race within a reasonable seed/exec budget, that is a legitimate signal to
redirect effort back into T1-T4 (known, cheaper, already-partially-proven ROI) rather than push
further into schedule-quality research with no evidence it will pay off.

## 4. Correctness/determinism risks that could sink it

Ranked by how likely each is to actually derail the project, not just by severity:

1. **Interleaving-schedule yield risk (the dominant risk).** It is fully possible to build a
   correct, fully deterministic N-hart emulator that still rarely triggers the exact race window a
   given kernel bug needs — schedule-quality is genuine, open research/tuning work (comparable to
   what dedicated concurrency-fuzzing tools like CHESS/PCT invest), and no amount of mechanical
   correctness substitutes for it. This is why Phase 4's known-race proof point is the real gate,
   not Phase 1's mechanical boot.

2. **Cross-hart LR/SC invalidation implemented subtly wrong.** This is silent, not a crash: too
   permissive → spurious SC success silently breaks every spinlock and fabricates "races" that are
   pure emulator artifacts, not real kernel bugs — actively corrupting the exact bug class SMP was
   built to find, and potentially the most damaging failure mode because it looks like success.
   Requires a differential test against a real dual-hart reference (Spike or `qemu-system-riscv32
   -smp 2`) specifically for LR/SC/AMO interleavings, mirroring the project's existing
   decoder-differential discipline — not "looks right by inspection."

3. **New nondeterminism-leak surface in the timer/IPI redefinition.** Today `mtime` is elegantly
   `cpu.insns_retired` — a single-hart identity (lib.rs:1756). The multi-hart replacement (a global
   scheduler tick independent of any one hart's retired-instruction count) must stay a pure function
   of `(snapshot, schedule-seed)`; getting it wrong quietly reintroduces the "ghost crash"
   /non-reproducible-case failure mode that decision #7 was originally written to eliminate — now
   with more moving parts (per-hart clocks, IPI delivery timing, schedule-seed capture in
   `Snapshot`) than before. Needs its own explicit twice-run differential test before any SMP result
   is trusted, per phase.

4. **Real OpenSBI HSM/IPI exercised in a multi-hart configuration for the first time, with zero
   existing test coverage.** Decision #5 (real OpenSBI in M-mode) was validated single-hart only;
   subtle CLINT-offset or CSR-masking mismatches under real HSM/IPI traffic are plausible and
   unverified until Phase 1 actually boots — this assessment is grounded in static code reading, not
   an executed boot attempt.

5. **The temptation to "simplify" TLB/shootdown fidelity.** Eagerly flushing all harts' software TLB
   on any `SFENCE.VMA` for safety would silently delete the exact shootdown-race bug class that's
   much of the point of doing SMP at all. Needs deliberate, careful modeling in Phase 2, not the
   easy path.

6. **fs-jit chain-cache cross-hart sharing is asserted safe by code-shape (fresh `cpu`/`bus`
   pointers per call, `last_edge` drained per call) but has never been exercised under true
   multi-hart interleaving.** Stress-test this specifically in Phase 2 before trusting it at scale;
   keep JIT gated off until then regardless, both for this reason and for reason 7.

7. **Someone later "speeding up" SMP with real OS threads for performance.** This must be a
   permanently locked constraint (decisions.md, added at Phase 0), not an optimization left open —
   real concurrency across harts reintroduces host-timing-dependent nondeterminism and destroys the
   entire reproduction guarantee the fuzzer depends on. One host thread, fixed/seeded cooperative
   round-robin, always.

8. **Expectation-setting risk, not a technical one:** the fs-vec/AVX-512 incompatibility (§3) must
   be communicated up front as scope, or SMP will be perceived as a regression against the project's
   main 512-guest throughput thesis rather than a parallel, additive product line for a different
   bug class.
