# Emulator-native KMSAN — staged plan

Value-taint tracking (uninitialized-*value* use) on **uninstrumented** guests. From a 2-lens design
workflow + adversarial synthesis (2026-07-08), verified against the actual source. Mode-gated
(default runs pay nothing), deterministic, **byte-granular** (no finer ground-truth claim than
`PERM_RAW`). Distinct from `PERM_RAW`'s ASAN-strict fault-on-first-read: KMSAN *permits* reading
uninit and reports only at **consumption** — so the two are separate campaigns, never simultaneous on
one access.

## Load-bearing decisions (verified against the repo)

- **Register shadow = `Cpu.regs_taint: Option<Box<[u32;32]>>`** (byte-taint: bit at 0/8/16/24 = that
  byte of the value is uninit), mirroring the existing `cmplog: Option<Vec<..>>` idiom. NOT a bare
  `[u32;32]` — `Cpu` is `clone()`d **every fuzz case** (`main.rs` `*cpu = golden_cpu.clone()`), so a
  bare array is a 128-byte memcpy per case even with KMSAN off; the `Option` keeps the off-cost at one
  pointer. `VecCpu.regs_taint: [[u32;16];32]` (SoA twin; one packed OR in `try_simd_alu`).
- **Memory shadow = a spare bit in the existing `perms` byte, `PERM_VTAINT = 1<<5`** — NOT a separate
  `Vec<u8>` plane. Decisive: `fs-loader::load_into` sets `perms[off] = perm` **directly** (bypassing
  `Mmu::write`, the only RAW-clearing path), so a *separate* shadow array would default to all-tainted
  for the whole loaded kernel image → a day-one false-positive storm. A reused bit is safe by
  construction (existing callers never set bit 5; `perms`≥0 for any accessible byte leaves taint=0 for
  free). Zero extra memory, and CowRam's page-lazy-alloc already covers it. `Mmu::write`'s unconditional
  `(*p|READ)&!RAW` stays untouched; a KMSAN-gated `write_shadow` fires alongside the store. Replicate
  across `Mmu` / `Golden` (read side) / `CowRam` (both).
- **Report-and-stop** (`Trap::KmsanTainted{pc,rs1,rs2,taint_a,taint_b}`), NOT an accumulating log —
  matches the codebase's fault-shaped bug-oracle idiom (cmplog's drain is a coverage side-channel, not
  an oracle).
- **exec_one seam**: KMSAN wires taint arms next to the existing `wr_reg`/`load`/`store` in `step`'s
  match — the same lines the JIT Stage 0 `exec_one` extraction touches. Land KMSAN build **after** JIT
  Stage 0 so it rides `exec_one` (a rebase-coordination issue, not a design conflict).

## Propagation (Stage 0 = pure `alu_taint`/`muldiv_taint(op, a_val,a_taint,b_val,b_taint)`, unit-testable)

- v0 (sound-enough, ship first): Xor/shift/Slt/Add/Sub = OR of operand byte-masks; And/Or = OR (accept
  documented over-taint); Mul/Div/Rem = taint-all if either operand tainted; Lui/Auipc = 0; Load =
  gather byte-taint from `PERM_RAW` (Stage 1) then `|PERM_VTAINT` (Stage 2), sign/zero-extend the taint
  mirroring the value; Store = scatter the low-N reg byte-taint.
- **v1 refinements (Stage 3, not someday — both designs flag these as the dominant FP sources):**
  - **Add/Sub: byte-position carry-smear** — plain OR is UNSOUND (a tainted low byte's carry can flip a
    "clean" high byte). Smear the byte-mask upward (`m|=m<<1; m|=m<<2` over the 4 byte-positions) —
    Valgrind-Memcheck-style, sound, byte-granular.
  - **And/Or: known-byte clearing** — a byte forced by a known-0 (AND) / known-1 (OR) operand byte is
    untainted regardless of the other operand.
  - **Register-shift (`Op` form): if the shift-amount `rs2` is tainted → taint-all** (any of 32 amounts
    could apply); immediate-form shifts stay exact.
  - **Sra / sign-extending load**: propagate the shadow's top-byte taint into filled high bytes.

## Stages (each gated on a synthetic RV32 test with a POSITIVE and NEGATIVE control — asymmetric testing is how the loader-seeding FP storm would slip through)

- **Stage 0** — pure `alu_taint`/`muldiv_taint` functions + per-op unit tests. Zero Cpu/Mmu touch.
- **Stage 1 (FIRST INCREMENT)** — `regs_taint: Option<Box<[u32;32]>>` + `set_kmsan`/`kmsan_enabled`
  (mirror `set_cmplog`), wire v0 rules into `step`'s Op/OpImm/Mul arms, load-taint via a new additive
  `Mmu::read_raw_state` (does NOT touch `read`/`read_bytewise`), branch-on-tainted → `Trap::KmsanTainted`.
  NO memory shadow, NO refinements, NO checkpoints beyond Branch. Validate: positive (RAW byte →
  Lw → Add → Branch fires), negative (store-first → doesn't fire), clean-regression (off = zero behavior
  change + clone cost unchanged). Independently mergeable, touches no write path.
- **Stage 2** — `PERM_VTAINT` spare bit + `write_shadow`; loads OR in both RAW and VTAINT (this is what
  makes it *value* taint, not just byte taint). Extend to store-then-reload-then-branch round trip.
- **Stage 3** — the v1 precision refinements above (carry-smear, known-byte, tainted-shift, Sra).
- **Stage 4** — more checkpoints (independently landable once Stage 2 exists): Jalr-target-tainted,
  syscall-return (`a0`) tainted, and `copy_to_user`/`put_user` **sink hooks** via a new
  `fs-san::PcHooks` hook kind (OR-reduce the memory shadow over `[ptr,ptr+len)`).

Origin tracking (which alloc first introduced a taint) is out of scope (a 4th-plane stretch).

## Open questions (decide, don't assume)
Byte-taint packing (bits 0-3 vs byte-aligned 0/8/16/24 — recommend byte-aligned for Stage-3 shift math);
mode flag as runtime `Option` (start here, matches cmplog) vs const-generic (only if profiled);
coordinate the `exec_one` rebase with JIT Stage 0; Stage-2 `read_raw_state` interaction with the
software TLB (caches perms at translate time); Stage-4 syscall-return check keyed off `fs-san`'s
`Convention` table to avoid firing on void-returning syscalls.
