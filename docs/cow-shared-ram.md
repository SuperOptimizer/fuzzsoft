# COW-shared guest RAM for VecSystem

Design chosen by a 3-way independent-design workflow + adversarial synthesis (2026-07-08). The
problem: `VecSystem` (the full-system vectorized emulator) currently holds 16 **independent** 128 MB
`fs_platform::Machine`s, so (a) memory is 16× (and 32 cores × 16 lanes × 256 MB ≈ 128 GB — infeasible),
and (b) the converged SIMD fast path is a *net loss* because it speculatively translates+fetches
**per lane** before it knows the op qualifies.

## Recommended architecture — page-COW over a shared immutable golden image

- **One immutable `Arc<Golden>`** byte-plane image (`mem: Vec<u8>` + `perms: Vec<u8>`, the *exact*
  `fs_mmu::Mmu` encoding — byte-granular RWX/RAW preserved), captured once post-boot via
  `Golden::from_mmu(&Mmu)`. Immutable ⇒ `Send + Sync` ⇒ shared read-only by **every lane and every
  core** — one 256 MB copy process-wide.
- **Per-lane 4 KB-page copy-on-write** via a direct-mapped page directory:
  `CowRam { golden: Arc<Golden>, dir: Vec<u32> /* len = size>>12; SENTINEL=u32::MAX = "still golden" */,
  pages: Vec<Box<CowPage>> /* CowPage = [u8;4096] mem + [u8;4096] perms */, dirty: Vec<u32> }`.
  Read: `pn=(pa-base)>>12; slot=dir[pn]` → golden slice or overlay slice (content **and** perms from
  the same source, so they can't desync). Write / any perm mutation: `ensure_page(pn)` copies the
  4 KB golden page into a fresh overlay, then mutates only the overlay. Reads never allocate (safe to
  call speculatively). Baseline per-lane cost = the 128 KB directory (all SENTINEL) + tiny Clint/Uart.
- **Correctness backbone: expose each lane's memory to the audited scalar core as a `Bus`.** `Cpu` is
  already generic over `Bus`, so a new `fs_platform::CowMachine { ram: CowRam, clint, uart }` (same
  CLINT/UART/RAM routing as `Machine`) gives the entire per-lane fallback via the unchanged
  `Cpu::step_system(&mut dyn Bus)` — **ZERO `fs-riscv` changes**. This is the load-bearing lever:
  faults are byte-identical to scalar by construction (same byte-plane semantics per page slice).
- **All safe Rust** (`Arc`/`Vec`/`Box`) — fs-mmu/fs-vec/fs-platform keep `forbid(unsafe_code)`.
- **Additive API**: `Mmu`'s public API is untouched (scalar core / boot / fs-cli fuzzer unaffected);
  only VecSystem's internal bus type changes and `from_template`'s signature is preserved (so
  `boot_vec` needs no edit).

## Two independent wins, sequenced

- **Phase 1 (memory):** swap VecSystem's per-lane `Machine` → `CowMachine`, keep today's fast path
  verbatim. Memory: 256 MB shared + ~2 MB directories + a few MB of overlays per 16-lane core vs
  ~4 GB today — ~60–400× less, enabling 32×16. Perf: ~neutral (one predictably-biased branch per RAM
  access: code/rodata stay golden, stack/heap stay overlay).
- **Phase 2 (perf):** add a **read-only sv32 walk against `Golden`** (a free fn returning
  `(pa, leaf_perms, ad_already_set)` that **declines** when A/D would need writing) + a cross-lane
  `SharedDirty` page bitmap (1 bit/4 KB, set on every COW). A converged group then translates+fetches
  **once** from golden and broadcasts, declining to per-lane only on an A/D writeback or a
  union-dirty page. Flips the documented net loss into a win: even the decline case is ~16× cheaper
  (speculation is 1× not 16×); converged straight-line ALU/branch/same-address-load bursts (boot,
  memset/memcpy/string loops, syscall prologues) run as one `Simd<u32,16>` op.

## Top correctness risks + mitigations

1. **A/D writeback divergence** (top risk): never use the *mutating* `Cpu::xlate` in the shared path;
   use the read-only `walk_golden` that declines when A (or D on store) is unset, then let per-lane
   `step_system` set A/D identically in all 16 overlays. Golden is post-boot so kernel .text/PTE A
   bits are pre-set → declines are rare / one-shot per satp epoch.
2. **A lane privately modified a shared code/PTE page**: set the `union` bit on *every* COW and consult
   it before every shared fetch/load; the only failure direction is a spurious per-lane fallback
   (slower, still correct). Back it with a release-off `debug_assert!` that the broadcast golden
   result equals the per-lane `xlate`+`ifetch16`.
3. **Content/perm desync**: both planes always sourced from the same page, copied together on COW.
4. **Page-crossing access**: `CowRam` resolves a single 4 KB page; fs-riscv already splits
   misaligned/page-crossing accesses to per-byte (each within one page). Assert no Bus access spans a page.
5. **Reset**: overlay-drop is O(dirty) with **zero byte copy-back** (golden never mutated) — walk
   `dirty`, set `dir[pn]=SENTINEL`, `pages.clear()`, clear the union bits. A test asserts post-reset
   byte-equality to golden.

## Validation (differential, byte-exact at every step)

- **PR1 (fs-mmu only):** `Golden`, `CowPage`, `CowRam` (+ Bus-shaped `load/store/ifetch16` + `reset`)
  and a property test `cow_ram_is_byte_exact_vs_mmu`: random layout + random 1/2/4-byte load/store/
  fetch stream against a reference `Mmu` and a `CowRam` over its golden, asserting identical values
  **and** identical `Fault{kind,addr,access}` on every access, RAW-upgrade + poison parity, then
  `reset()` + re-diverge proves reset restores golden exactly. Independently correct, zero executor
  risk, cannot affect boot — proves the load-bearing invariant every later step relies on.
- **PR2:** `CowMachine` vs `Machine` route an identical CLINT/UART/RAM sequence to identical results.
- **PR3:** VecSystem substrate swap — the existing `vec_system_matches_scalar_oracle…` divergent-seed
  test must still pass, then the `boot_vec` acid test (lane 0 UART == scalar, all 16 lanes identical);
  print overlay page counts to confirm the memory drop.
- **PR4:** shared translate/fetch perf, guarded by the `debug_assert` equality check + a test that
  COWs a code/PTE page in one lane and asserts the union guard forces per-lane fallback.

## Note

This is the **software** COW for the *vectorized* path (mandatory — 16 lanes share one host address
space, so the host MMU can't back divergent lane views, and byte-granular RAW/redzone perms need
software). The **scalar `--jobs` path** is separately a candidate for **host-mmap-backed** guest RAM
(kernel COW-share + `MADV_DONTNEED` O(1) reset) — see the queued mmap prototype; that trades byte
granularity for near-native speed and is being evaluated with real numbers before committing.
