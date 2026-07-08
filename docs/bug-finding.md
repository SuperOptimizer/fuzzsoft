# Bug-finding strategy

fuzzsoft is fast (32-thread, ~3500 exec/s, flat 430MB) and broad (90 resource-threaded syscalls +
dictionary + cmplog), yet finds **0 bugs** on clean released kernels (200k-case campaigns: ~35k
coverage buckets, 0 crashes). That's the *expected* result on a clean tree — the fix isn't "fuzz
harder," it's **fuzz where the bugs are.** From a 3-angle strategy workflow + synthesis (source-verified
against `build/linux-src` 7.2.0-rc2).

## Honest assessment (no hype)

The realistic outcome of this plan is **"prove the pipeline can find+reproduce real bugs, and add one
genuinely new bug-finding axis (fault-injected error paths) to the covered surface"** — *not* "find an
upstream 0-day syzkaller missed." Three by-design ceilings:
1. syzkaller's continuous fleet hammers the same *generic* code paths harder, with an enormous
   exec-history — a fresh find there after 200k more cases is low-probability by construction.
2. **nosmp / single-hart / single-task deterministic execution structurally excludes the entire
   concurrency/race bug class** (UAF-via-race, TOCTOU, lock-ordering) — a large share of what syzbot
   actually finds. (Proof: the epoll subsystem's `a6dc643c6931` race-UAF is provably unreachable here,
   sitting right next to a bug that *is* reachable.)
3. `MAX_CALLS=8` + a straight-line, no-branch-on-guest-output wire format make the highest-bug-density
   subsystems (io_uring completion loops, retry-on-EAGAIN) only shallowly expressible.

**A fine, honest deliverable at this scale:** prove find→minimize→reproduce against a *real* (not
planted) bug, add the fault-injection axis (cheap, orthogonal to the coverage plateau, ~how a real
fraction of syzbot findings are produced), and — the unique-edge bet — look at **RV32/arch-specific
surfaces** (the emulator's own AMO/vector/unaligned edge cases, this port's syscall-marshalling paths)
that an x86-centric fleet structurally cannot reach.

## Prioritized plan

**1. FAULT INJECTION FIRST (the #1 lever, cheapest, reuses ~100% of infra).** Random argument fuzzing
essentially never makes a real `kmalloc`/`alloc_pages`/`copy_from_user` legitimately fail in a small
guest — so every "the allocation failed, clean up" branch is dead code no campaign has executed, and
that's exactly where UAF/double-free/leak bugs concentrate. Mechanism (source-verified, boot-safe by
construction):
- **`fail_nth`**: `lib/fault-inject.c::should_fail_ex()` only fails if `current->fail_nth != 0`
  (decrement-to-0, fail once, self-disarm); with `fail_nth==0` it falls to the probabilistic path
  gated by `attr->probability` which `FAULT_ATTR_INITIALIZER` zero-inits → short-circuits. **Nothing
  fails unless a task writes a positive N to its *own* `/proc/<pid>/fail-nth`.** So it's safe to leave
  armed across a whole campaign without corrupting the clean-boot baseline.
- **Free reset**: `task_struct.fail_nth` + the failslab/fail_page_alloc knobs all live in **guest RAM**,
  so the existing dirty-block `Snapshot::reset` (decision #11) reverts them every case with **zero new
  plumbing** (unlike the host-side `Sanitizer`, which needed a manual clone/restore).
- **The load-bearing gotcha**: `failslab.ignore_gfp_reclaim = true` by default → ordinary `GFP_KERNEL`
  kmalloc (the common case) is *silently exempt*. Must flip `ignore-gfp-wait`=0 (and fail_page_alloc's
  `ignore-gfp-wait`/`ignore-gfp-highmem`/`min-order`=0) via **debugfs** (needs
  `CONFIG_FAULT_INJECTION_DEBUG_FS`; deps already `=y`). Skip this and injection is compiled-in but
  inert → a false "0 crashes, mechanism dead" read.
- **Build**: `Image.failinj` = FAULT_INJECTION + FAILSLAB + FAIL_PAGE_ALLOC + FAULT_INJECTION_DEBUG_FS +
  **SLUB_DEBUG_ON** (the combo oracle: injection drives cleanup paths, SLUB_DEBUG catches the resulting
  corruption via the *existing* `kernel_crash_sig`). `boot/agent.c` pre-snapshot: mount proc/sysfs/
  debugfs + write "0" to the knobs (baked into the golden image, costs nothing/case). Arm per-case via
  two ordinary guest syscalls (`openat("/proc/self/fail-nth")` + `write("<N>")`) using fs-prog's
  existing fd-fixup machinery — new `openat$fail_nth`/`write$fail_nth` descs + a `--fail-inject`
  generator bias that prepends the 2-call preamble. **Composes with `--jobs` parallel** (unlike
  `--sanitize`). Exclude `FAULT_INJECTION_USERCOPY` initially (drowns signal in -EFAULT).
- **Prove the chain works**: `Image.failinj.buggy` — a hand-planted double-free gated *on an
  allocation-failure branch*, confirming `kernel_crash_sig` fires when `--fail-inject` is ON and NOT
  when OFF. Without this, a "0 crashes on Image.failinj" result could just mean "never armed."

**2. REAL-CVE PIPELINE VALIDATION (proves real-bug-finding end-to-end).** `Image.cve-epoll-loop` =
hand-revert `fdcfce93073d` (EP_MAX_NESTS integer-overflow in `ep_loop_check_proc`, single-task/non-race,
within existing epoll_create1/epoll_ctl/close coverage). Two tiers: (a) a fixed 7-call epoll chain
proves the oracle detects it; (b) a real coverage-guided campaign measures whether genr.rs *organically*
builds a ≥7-deep same-kind fd chain — a previously-unmeasured signal about the mutator's chain-building
limit. Document the sibling race-UAF as the honest "unreachable here" ceiling.

**3. COVERAGE-SURFACE EXPANSION (last — most expensive, least certain).** Cheap first: unshare/setns,
splice/vmsplice/tee (hours each, shapes fs-prog already models). Then keyctl/add_key, mount$overlay/tmpfs
(needs initramfs scaffolding). Only then bpf/io_uring (days each, architecturally strained). Build a
System.map coverage-attribution tool early as a progress gauge (expect it to *confirm* the known gaps,
not reveal new ones).

## Coverage gaps (already answered, was "plateau, cause unknown")

`.config` has these compiled in but fs-prog has **zero descriptions** for them → structurally unreached:
**bpf** (`CONFIG_BPF_SYSCALL=y`), **io_uring** (`CONFIG_IO_URING=y`), **keyctl/add_key**
(`CONFIG_KEYS=y`), **mount/overlayfs/ext4/btrfs**, **unshare/setns namespaces** (`CONFIG_USER_NS=y`),
**splice/vmsplice/tee**. That's the plateau's cause — not a mutator weakness.

## Open questions (measure, don't assume)
Does genr.rs organically build ≥7-deep same-kind fd chains, or only via seeded corpus? (epoll tier-2
answers it.) Does the epoll overflow crash cleanly with VMAP_STACK off, or need a planted WARN_ON? Is
`fail_nth`'s single shared countdown precise enough in an 8-call program, or do most armed cases fail
uninterestingly early? **Is phase-3 (bpf/io_uring) worth multi-week effort vs RV32/arch-specific
surfaces syzkaller structurally can't reach?** — the highest-leverage strategic question, unresolved.

## Shipped: fault-injection axis (build + agent + descriptions)

The fault-injection mechanism above is built and boot-validated; only the fs-cli `--fail-inject`
generator wiring is left (a deliberate follow-up for the lead, so fs-cli stays owned by one agent).

- **Kernels** (`scripts/build-failinj-kernel.sh`, clean-worktree + `O=` pattern):
  `firmware/Image.failinj` = `FAULT_INJECTION + FAILSLAB + FAIL_PAGE_ALLOC + FAULT_INJECTION_DEBUG_FS
  + SLUB_DEBUG_ON` (verified stuck; `FAULT_INJECTION_USERCOPY` left OFF).
  `firmware/Image.failinj.buggy` = same config + `scripts/failinj-bug.patch` = a hand-planted
  double-free in `memfd_create()` gated on an *allocation-failure* branch (`p=kmalloc; if(p){
  q=kmalloc; if(!q){ kfree(p); kfree(p); } }`) — dead code unless fault injection fails the 2nd
  kmalloc, so it validates the whole arm→inject→crash→`kernel_crash_sig` chain (SLUB "Object already
  free" report) without polluting the clean baseline.
- **Agent** (`boot/agent.c`, pre-snapshot / baked into the golden image, all errors ignored):
  mounts proc/sysfs/debugfs, writes `0` to `failslab/ignore-gfp-wait`,
  `fail_page_alloc/{ignore-gfp-wait,ignore-gfp-highmem,min-order}` — flipping the load-bearing
  `ignore_gfp_reclaim` default so `GFP_KERNEL` allocations actually fail. Harmless on kernels
  without these paths (mounts/opens no-op). `boot/initramfs.spec` gains the mountpoint dirs.
  Boot-validated: `Image.failinj` (2.12B insns) AND a stock-config kernel with this same agent
  (1.74B insns) both reach the snapshot hypercall — no boot regression, no boot-insn bump needed.
- **Descriptions** (`crates/fs-prog`): `openat$fail_nth` (opens `/proc/self/fail-nth` `O_WRONLY`)
  and `write$fail_nth` (writes a small ASCII countdown), plus `prepend_fail_inject(rng, prog)` which
  prepends the fd-threaded 2-call arming preamble (truncating the tail to `MAX_CALLS`). Wire
  constants/public API otherwise unchanged.
- **Left for the lead:** an fs-cli `--fail-inject[=PCT]` flag that calls
  `fs_prog::prepend_fail_inject` on a fraction of generated programs (per-worker rng, composes with
  `--jobs`), then validate `fuzz --fail-inject --kernel firmware/Image.failinj.buggy` fires
  `[KERNEL CRASH]` while the same run on `firmware/Image.failinj` (and any run *without*
  `--fail-inject`) stays clean.
