# fuzzsoft — Architecture Digest

*A vectorized RISC-V RV32IMAC full-system emulator that boots one Linux guest per AVX-512 lane and harvests coverage for free. Synthesis of six specialist research digests, following Brandon Falk's (gamozolabs) published guidance, adapted to RV32IMAC + full-system Linux boot on a Zen 4 / AVX-512 host.*

> Status: **design phase.** Open decisions are tracked in `docs/decisions.md`; no implementation has started. Two of six research inputs (vectorized-core, fuzz-pipeline) were degraded during generation and reconstructed by the synthesis pass — treat §2 and §7 as slightly lower-confidence and re-verify against primary sources during implementation.

## 1. Overview and thesis

fuzzsoft applies Falk's core insight — *a fuzz case is not a boot, it is a golden snapshot plus a tiny mutation, executed then reset in O(bytes-dirtied)* — to a full-system RV32 Linux target, and runs many guests in lockstep across the 16 u32 lanes of a 512-bit ZMM register. Three of Falk's ideas are load-bearing and port almost verbatim:

1. **A byte-granular soft MMU** carrying a shadow permission array parallel to guest memory; RWX + read-after-write (uninit) detection fall out for free ([MMU Design](https://gamozolabs.github.io/fuzzing/2018/11/19/vectorized_emulation_mmu.html)).
2. **Dirty-block reset** — restore only the 64-byte cache lines a case actually touched, from a pristine master ([MMU Design](https://gamozolabs.github.io/fuzzing/2018/11/19/vectorized_emulation_mmu.html); [fuzz_with_emus/src/mmu.rs](https://github.com/gamozolabs/fuzz_with_emus/blob/master/src/mmu.rs)).
3. **Determinism as an enabler**, not a nicety — all entropy comes from the snapshot, giving free crash replay and *differential coverage* (16 byte-identical lanes; any inter-lane divergence is itself a coverage signal) ([Vectorized Emulation](https://gamozolabs.github.io/fuzzing/2018/10/14/vectorized_emulation.html)).

The strategic bet that RV32 is *better* than Falk's RV64 for us: XLEN=32 means a ZMM holds **16 u32 lanes instead of 8 u64 lanes**, doubling VM density, and an RV32 word access is exactly one u32 lane — a clean map to `vmovdqa32` with none of the 8-byte awkwardness the Xeon Phi had.

**Two hard truths the digests agree on.** (a) The open-source `fuzz_with_emus` is a *scalar RV64I interpreter*; the "2 trillion inst/sec" AVX-512 JIT is a **separate, unpublished codebase**. We must write the vectorized executor ourselves — the single largest unknown. (b) Full-system kernel boot is an order-of-magnitude larger surface than Falk's userland harness (CSRs, S-mode traps, sv32, atomics, timer, SBI, DTB), and lockstep SIMD collides head-on with per-lane kernel control-flow divergence. These two facts shape the entire roadmap: **build a validated scalar full-system emulator first; vectorize last, from a converged post-boot snapshot.**

## 2. Vectorized execution model

- **Lane geometry.** 512-bit ZMM = 16 × u32 lanes. One lane = one RV32 guest VM. Baseline: 16 VMs/thread × 32 cores = **512 concurrent guests**; possibly multiple ZMM groups per thread later.
- **State as SoA.** The register file is `[u32; 33]` (x0–x31, PC=32). Across a 16-lane group, register *i* becomes a 16-wide u32 vector (`reg[i]` transposed into a ZMM column). Design the scalar core's state block so this transpose is mechanical.
- **Lockstep + masking.** Lanes advance together while their PCs agree; a k-mask disables lanes that have diverged, faulted, or halted. Converged lanes share PC ⇒ share instruction length, so variable-length C-extension fetch is safe under lockstep (divergence is handled by masking, never by per-lane length assumptions).
- **Divergence is *the* cost model.** On the Phi, non-divergent 8B access was ~35 cycles vs ~157 for a per-lane gather/scatter — **~4×** ([MMU Design](https://gamozolabs.github.io/fuzzing/2018/11/19/vectorized_emulation_mmu.html)). Detect cross-lane address equality up front; use the aligned `vmovdqa32` fast path when all lanes hit the same guest address, fall back to `vpgatherdd`/`vpscatterdd` only on divergence. **Minimizing divergence is the primary throughput lever.**
- **Instructions with no packed form** (M-extension DIV/REM, and MULH* which needs 64-bit widening) force a **masked scalarize-16-lanes** path. Budget for it; keep it *exact* (see §5).
- **Reality check for full-system.** Different syscall inputs drive kernels down different paths; timer preemption adds churn. Effective utilization will sit **well below 16×**. Two mitigations: (a) snapshot *post-boot* so all lanes start byte-identical and only the syscall-under-test is vectorized; (b) a hybrid executor — vectorize hot common paths, scalar-execute divergent tails, re-converge at common PCs. The fallback architecture if divergence dominates is *1 scalar VM/core with SIMD only on hot loops*.
- **Zen 4 caveats vs the Phi numbers.** The 35/157-cycle figures are Xeon Phi (8-byte pages, different cache hierarchy). AVX-512 gather/scatter on Zen 4 decodes to many uops and may be *worse* relatively. **Re-benchmark the divergent-vs-aligned gap on the 7945HX before committing the memory path.**

## 3. Soft MMU (the bug oracle)

Port Falk's design verbatim; it is XLEN-independent and the single most reusable piece.

- **Shadow permissions, 1:1 with bytes.** Parallel `memory: Vec<u8>` and `permissions: Vec<Perm>` where `Perm` is `#[repr(transparent)] Perm(u8)`. Encoding (exact): `PERM_READ=0x01`, `PERM_WRITE=0x02`, `PERM_EXEC=0x04`, `PERM_RAW=0x08` (read-after-write / uninitialized), `PERM_ACC=0x10` (access/coverage). Independent bits express execute-only and give W^X for free.
- **RAW uninitialized-read detection.** `allocate()` stamps `PERM_RAW|PERM_WRITE` with **no** read bit. A load requires `PERM_READ`. On store: `perm = (perm | PERM_READ) & !PERM_RAW`, unlocking reads for exactly the written byte. `malloc(8); write(1); read(8)` faults on the 7 untouched bytes at single-byte granularity. Gate behind a `const DISABLE_UNINIT` for a fast mode.
- **Bump allocator with guard holes.** `cur_alc: VirtAddr` monotonic cursor (starts 0x10000 in Falk, 32-byte aligned), `active_alcs: HashMap<VirtAddr, usize>` records live sizes; per-byte perms let it punch one-byte guard holes to catch 1–2 byte overflows.
- **Two permission *layers* under full-system.** Keep them distinct: (1) *architectural* sv32 PTE R/W/X/U bits for spec-correct translation; (2) *soft-MMU* R/W/X/RAW bits as the fuzzing bug oracle. The sv32 walk produces a guest-physical address; that GPA then indexes the flat soft MMU.
- **Address base.** Falk indexes guest-addr-as-host-offset from base 0. A real RV32 kernel uses high physical addresses (DRAM base `0x8000_0000`), so we need a base/offset mapping, not literal indexing.
- **Vectorized interleaving.** Interleave memory *and* permissions at u32 granularity so all lanes' copy of a guest word is contiguous: `host_word_index = guest_word * 16 + lane`; one `vmovdqa32` touches all 16 lanes. Layout two planes per interleaved word: 16 permission u32s (one 64-byte line) then 16 content u32s (next line). A single `vpcmp`/`vptest` on the permission line validates all 16 lanes at once; branch to a slow per-lane fault path only on failure.
- **Kernel noise caveat.** RAW/uninit on a real kernel is noisy (zeroed pages, struct padding, DMA regions). Need per-region opt-out (mark `.bss` pre-initialized) or `DISABLE_UNINIT` on kernel pages while keeping it live on fuzzed buffers.

## 4. Reset and snapshot (the keystone)

- **Two-structure dirty tracking.** `DirtyState { dirty: Vec<usize>, dirty_bitmap: Vec<u64> }`. `DIRTY_BLOCK_SIZE = 64` bytes (one cache line). First write into a block: if the bitmap bit is clear, set it and push the block index onto `dirty`. The **bitmap is only for dedup** (O(1) "already listed?"); the Vec is the reset worklist. **Never iterate the bitmap on reset — iterate the Vec.**
- **Reset restores contents *and* permissions together.** `reset(&mut self, master: &Mmu)`: for each block in `dirty`, `copy_from_slice` its 64 bytes of **both** `memory` and `permissions` from the pristine master, zero the containing bitmap word, then `dirty.clear()` and restore allocator state (`cur_alc`, `active_alcs`). Cost is O(bytes mutated), never O(total RAM). RAW/alloc state resets for free because perms ride along.
- **Reset is all-or-nothing across contents + perms + register file + allocator.** Forgetting any one yields non-reproducible "ghost" crashes. The `[u32;33]` register file (incl. PC and CSRs) **must** be part of the snapshot/reset — hold it in a struct you memcpy from master, or in pinned ZMMs you reload.
- **Snapshot is mid-execution, not a cold boot.** Boot once (millions–billions of instructions), run to a quiescent harness point — a tiny init issuing a reserved SBI/ecall *snapshot-me-here* hypercall — capture full machine state (all guest RAM + every CSR + MMU/TLB + timer/interrupt state), then loop cases. A fuzz case = patch mutated input into the snapshot, run, reset.
- **Vectorized reset is one wide copy per block.** Because lanes are interleaved into the same physical block, one dirty entry + one wide block-copy restores all 16 lanes at once; over-restoring matched lanes is free. **This couples layout and reset: keep a single shared union dirty list — do NOT keep per-lane lists, and do NOT store lanes in separate SoA arrays, or the shared list becomes wrong.**
- **Bound the dirty list.** Size it to `mem/block + 1`; on overflow, VM-exit that lane/batch rather than growing unbounded — keeps worst-case per-case reset cost capped (Falk's behavior).
- **Block-size knob.** 64B is Falk's scalar balance. The ×16 interleave multiplies real copy cost, so re-benchmark on Zen 4; 256B is a floated alternative that shortens the dirty list at the cost of coarser restore.
- **CoW aliasing is a *later* tier.** Master template + per-child pages with aliased/CoW/dirty bits in the leaf PTE (Falk fit 2048×4GiB in <200MiB). Needed only if VM count far exceeds SIMD width or flat per-lane RAM blows the host budget. For the 16-in-register design, flat interleaved array + dirty list is the hot path — skip CoW initially.

## 5. RV32IMAC core

**Golden model:** `fuzz_with_emus/src/emulator.rs` is a scalar RV64I interpreter — our starting point, minus RV64.

- **Register file.** `[u32; 33]`, `#[repr(usize)]` Register enum Zero(0)..T6(31), Pc(32). x0 hardwired 0 on read, writes dropped. Keep PC in-array so snapshot/reset covers it uniformly.
- **Decode.** `opcode = inst & 0x7f`, dispatch, then typed R/I/S/B/U/J bitfield extraction with arithmetic-shift sign extension (I-imm `= (inst as i32) >> 20`; S/B/J reshuffles as in the spec). Unchanged from Falk in RV32.
- **Delete all RV64-only forms:** OP-32 `0b0111011`, OP-IMM-32 `0b0011011`, and LD/LWU/SD load/store funct3 cases. Narrow shift amounts to 5 bits.
- **M extension** (OP `0b0110011`, funct7 `0b0000001`): funct3 0..7 = MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU. **No traps.** DIV/0 → `0xffffffff`; REM/0 → dividend; signed INT_MIN/−1 overflow → DIV=`0x80000000`, REM=0. MULH* need a 64-bit intermediate (`((a as i64 * b as i64) >> 32)`). *These break clean SIMD — masked scalar fallback, must be exact.*
- **A extension** (`0b0101111`, funct3 `010` for .W; RV32 has no .D): funct5 in `[31:27]` selects LR/SC/AMOSWAP/ADD/XOR/AND/OR/MIN/MAX/MINU/MAXU; aq/rl in `[26:25]`. AMOs are RMW requiring READ+WRITE and 4-byte alignment. **Per-hart (per-lane) reservation `{addr, valid}`.** LR sets it; SC stores iff valid+matching (rd=0) else rd=1 no store. **Invalidate the reservation on any trap/context-switch** or SC wrongly succeeds and silently breaks every kernel spinlock. aq/rl are no-ops under deterministic single-threaded execution but must still decode.
- **C extension** (any halfword with `[1:0] != 0b11`): variable-length fetch — read one halfword, if `(h&3)!=3` decode 16-bit and `pc+=2`, else read full word and `pc+=4`. Quadrant `[1:0]`, funct3 `[15:13]`; compressed reg fields are 3-bit → x8..x15. Immediates are **non-contiguously scrambled per encoding** — the highest-bug-density part of the decoder. C.JAL is **RV32-only**; `0x0000` is a defined **illegal** encoding. Relax EXEC-fetch alignment to 2 bytes (IALIGN=16) and make coverage-edge granularity 2 bytes. **Write it as an explicit (quadrant, funct3) table with hand-verified bit-scatter and differential-test random 16-bit patterns against Spike.**
- **Full-system privileged surface Falk never wrote:**
  - SYSTEM `0b1110011` CSR forms: CSRRW/S/C = funct3 001/010/011, CSRRWI/SI/CI = 101/110/111, csr = `[31:20]`; with legal-field masking on write.
  - Priv insts by full word: MRET `0x30200073`, SRET `0x10200073`, WFI `0x10500073` (nop/hint or fast-forward to next timer), SFENCE.VMA (funct7 `0b0001001`, TLB flush).
  - Modes M(3)/S(1)/U(0). CSRs: S-set (`sstatus, stvec, sepc, scause, stval, sie, sip, sscratch, satp`) plus (if any M-mode) the M-mirrors. If we never enter M-mode, medeleg/mideleg are moot.
  - Trap machinery: save pc→sepc, cause→scause, jump stvec, flip privilege; xRET restores.
  - **sv32** GVA→GPA: satp MODE bit[31], ASID[30:22], root PPN[21:0]; 2-level walk, 4 KiB pages + 4 MiB megapages; 32-bit PTE bits V/R/W/X/U/G/A/D at [0..7], PPN at [10:31]; A/D updates; permission + misaligned-superpage faults. Cache in a software TLB keyed on (satp, VPN), flush on SFENCE.VMA / satp write.
- **Real ELF/Image + DTB loading in-process** (Falk shells out to `readelf`/`nm` — unacceptable for a self-contained boot flow).
- **FP open question:** RV32IMAC alone may be insufficient if the kernel touches F/D in context-switch save/restore. Pin a soft-float `rv32ima` kernel config to avoid this; confirm early.

## 6. Full-system / boot requirements

The minimum viable RV32 machine is genuinely small if we make the right firmware decision.

- **Boot contract (RV32):** load the flat `arch/riscv/boot/Image` at a **4 MiB-aligned** physical address (RV32 maps early RAM with sv32 4 MiB megapages), enter its first instruction in **S-mode with MMU off (satp=0)**, `a0=hartid (0)`, `a1=physical DTB pointer`. Everything else is discovered from the DTB. No bootloader/ELF loader for the flat Image.
- **Skip OpenSBI — emulate the SBI ABI in host Rust, boot directly into S-mode.** This is Falk's "intercept the ABI boundary on the host" philosophy moved up from the Linux-syscall boundary to the SBI boundary. Deletes M-mode firmware, most M-mode CSRs, and an entire boot stage. Trap `ecall`-from-S and dispatch on `a7`=EID / `a6`=FID: legacy console_putchar (0x01), legacy set_timer (0x00) and/or TIME ext (0x54494D45), HSM (0x48534D, stub for single hart), SRST (0x53525354 → clean VM-exit), DBCN (0x4442434E) for bulk console. Answer `rdtime` directly from the cycle counter.
- **Devices to reach a shell: essentially none.** With an SBI console (`console=hvc0 earlycon=sbi`) and an **initramfs** rootfs (in guest RAM, no block device, no DMA, no external interrupts — ideal for snapshot/reset), the first boot needs **zero emulated MMIO**. Defer ns16550 UART, PLIC, and virtio-mmio until after a booting kernel.
- **The one required interrupt is the supervisor timer.** Prefer the **Sstc extension** (`stimecmp`/`stimecmph` CSRs) so the kernel arms the timer directly with no SBI round-trip; fall back to SBI set_timer driving `mip.STIP`→`sip.STIP`, delivered when `sstatus.SIE` and `sie.STIE` are set. If you keep CLINT: base `0x0200_0000`, `msip` +0x0000, `mtimecmp` +0x4000, `mtime` +0xBFF8 (64-bit, accessed as two 32-bit halves on RV32).
- **Device tree:** hand-craft a tiny DTB. Required: `/cpus` with fixed `timebase-frequency` (e.g. 10 MHz, constant for determinism) and `cpu@0` with `riscv,isa="rv32ima"`, `mmu-type="riscv,sv32"`; `/memory@80000000`; `/chosen` with `bootargs` and (for -initrd) `linux,initrd-start/-end`; an interrupt-controller stub. No UART node needed with the SBI console.
- **Compatibility target = QEMU `virt` memory map** so stock kernels/DTBs "just work" and we can cross-check boot **instruction-by-instruction against QEMU** during bring-up: DRAM `0x8000_0000` (put kernel at `0x8040_0000`), CLINT `0x0200_0000`, PLIC `0x0c00_0000`, UART `0x1000_0000`, virtio-mmio `0x1000_1000` (8 × 0x1000, IRQ 1–8).
- **Kernel config:** base on `arch/riscv/configs/rv32_defconfig`; **nosmp / single hart** (avoids IPI/HSM, simplest for a snapshot fuzzer); **soft-float** (`rv32ima`, no F/D); SBI console + initramfs. Note `-march=rv32ima_zicsr_zifencei` (Zicsr/Zifencei split out of base I in recent gcc). Userland via **musl** (buildroot rv32) — glibc rv32 needs 2.33+. The linux, qemu, and syzkaller trees already checked out in this repo are the reference oracles for exactly this.
- **Determinism at the boundary:** virtual time = a strict function of retired-instruction count, *not* host wall-clock; fix `timebase-frequency`; seed/stub all host-derived syscall returns (time, PIDs, RNG); fixed interrupt schedule. Without this, differential coverage is noisy, crash replay breaks, and SIMD lanes desync.

## 7. Fuzzing pipeline

- **Coverage falls out of the emulator — strictly better than kcov.** Instrument the dispatch loop: on each basic-block entry / taken branch, fold `hash(prev_pc, pc)` into a per-lane AFL-style edge bitmap (or use the `PERM_ACC` first-access bit). Zero kernel changes, no `CONFIG_KCOV`, no `/sys/kernel/debug/kcov` ioctl, deterministic, and it captures *all* code (interrupt handlers, softirqs, every task) that per-task kcov misses. Mask recording to kernel space (privilege==S, or effective PC ≥ PAGE_OFFSET) to fuzz the syscall surface.
- **Differential coverage (the vectorization payoff).** 16 lanes start byte-identical; after each step, horizontally compare pc / branch-taken / reg-diff / mem-diff across lanes with AVX-512 compares. Record state only when lanes *diverge* — meaning the input influenced behavior — which avoids combinatorial state explosion.
- **Fuzz loop.** Reset (dirty-block) → inject mutated input into the guest (write a syscall into a task's registers, or feed bytes to a guest agent — TBD) → run to next harness hypercall / fault / timeout → service SBI vmexits → on fault, dedup crash by `(pc, fault-type, address-class)` (Falk's key, XLEN-independent) → feed coverage delta to the scheduler/mutator.
- **Crash reproduction is free** thanks to determinism: replay the same input, single-step the exact path.

## 8. Proposed Rust workspace layout

A Cargo workspace of focused crates. The scalar core is fully usable before any AVX-512 crate exists; the vectorized executor is an additive layer over the same decoder and MMU traits.

```
fuzzsoft/                      (cargo workspace root)
├─ crates/
│  ├─ fs-mmu/          Soft MMU: Perm bits, memory+permissions planes,
│  │                   bump allocator, DirtyState, reset/fork. Bug oracle.
│  ├─ fs-riscv/        RV32IMAC decoder (R/I/S/B/U/J + M/A/C tables) and the
│  │                   scalar interpreter (golden model). No privilege here.
│  ├─ fs-arch/         Privileged arch: CSR file, M/S/U modes, trap delivery,
│  │                   sv32 walker + software TLB, LR/SC reservation, FENCE/SFENCE.
│  ├─ fs-sbi/          Host-side SBI ABI (TIME/HSM/SRST/DBCN/legacy), rdtime.
│  ├─ fs-platform/     Machine: memory map, CLINT/Sstc timer, deterministic
│  │                   virtual time, DTB builder, (later) PLIC/UART/virtio.
│  ├─ fs-loader/       In-process ELF + flat Image + initramfs/DTB placement.
│  ├─ fs-snapshot/     Whole-machine snapshot (RAM+CSR+TLB+timer+regs) and
│  │                   dirty-block restore driven by fs-mmu::DirtyState.
│  ├─ fs-cov/          Edge-coverage bitmaps, hash(prev_pc,pc), differential
│  │                   lane-divergence coverage.
│  ├─ fs-vec/          AVX-512 vectorized executor: SoA transpose, 16-lane
│  │                   lockstep, k-mask divergence handling, interleaved MMU,
│  │                   scalarize fallback for DIV/REM/MULH. (Milestone M4+.)
│  ├─ fs-fuzz/         Corpus, mutator, scheduler, crash dedup, input injection.
│  └─ fs-cli/          Binary: config, thread/affinity pinning, stats thread.
└─ Cargo.toml
```

Key seam: `fs-riscv` decodes into an IL/op form that *both* the scalar interpreter and `fs-vec` consume, so the AVX-512 executor reuses one validated decoder rather than a second hand-written one (the classic silent-miscompare trap).

## 9. Staged roadmap

**M0 — Minimal end-to-end skeleton (thinnest vertical slice; the required first milestone).** Scalar `rv32im` (no A/C/privilege yet) interpreter in `fs-riscv` over a flat `fs-mmu` with byte permissions and RAW. `fs-loader` loads a static bare-metal RV32 ELF (a `riscv-tests` binary), executes it to completion/ECALL, and `fs-cov` emits an edge-coverage bitmap to disk. **Deliverable: `fuzzsoft run test.elf` executes RISC-V and prints a coverage bitmap.** No kernel, no vectorization, no reset. This proves the decode→execute→coverage spine.

**M1 — Complete + validate RV32IMAC (scalar).** Add M (exact DIV/REM/MULH semantics), A (LR/SC + AMO, per-hart reservation), C (table-driven, 2-byte fetch). Differential-test the whole decoder against **Spike** on `riscv-tests` and random instruction fuzzing (esp. C immediates). This is the golden model everything else is measured against.

**M2 — Full-system boot to a shell (scalar).** Add `fs-arch` (CSRs, S-mode traps, sv32 + TLB), `fs-sbi` (host SBI, boot directly in S-mode, no OpenSBI), `fs-platform` (Sstc/CLINT timer, deterministic virtual time, DTB builder). Boot the pinned `rv32ima` soft-float `nosmp` kernel with an initramfs and SBI console, **zero MMIO devices**. Cross-check the boot trace against QEMU. **Deliverable: a single scalar guest reaches a shell.**

**M3 — Snapshot + reset + single-lane fuzzing.** Add `fs-snapshot` (whole-machine snapshot at a harness hypercall from a minimal init) and dirty-block restore. Add `fs-fuzz`: input injection, mutator, coverage-guided scheduler, crash dedup by `(pc, fault-type, addr-class)`. **Deliverable: one core fuzzes kernel syscalls from a post-boot snapshot, coverage-guided, with reproducible crashes.** This is a complete fuzzer — everything after is throughput.

**M4 — Vectorize.** `fs-vec`: SoA transpose of `[u32;33]`, 16-lane lockstep AVX-512 executor with k-mask divergence handling, interleaved MMU (memory+perms at u32 granularity, shared union dirty list, wide block reset), masked scalar fallback for DIV/REM/MULH. Optimize the same-address fast path; re-benchmark the divergent gap on Zen 4. **Deliverable: 16 guests/thread from one converged post-boot snapshot.**

**M5 — Scale + differential coverage.** 32-core thread pool with affinity pinning and a stats thread; differential coverage via horizontal lane compares; CoW page aliasing *only if* host RAM demands it. **Deliverable: ~512 concurrent guests with divergence-based coverage.**
