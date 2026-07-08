//! Full-system vectorized executor (M4): `LANES` independent full-system hearts — each an
//! [`fs_riscv::Cpu`] (regs/pc/CSR/privilege) paired with its own [`fs_platform::Machine`]
//! (per-lane RAM + CLINT + UART) — stepped together one instruction at a time.
//!
//! Unlike [`crate::VecCpu`] (the user-mode-only, SoA-register-file executor elsewhere in this
//! crate), `VecSystem` is deliberately **array-of-structures**: each lane owns a real
//! `fs_riscv::Cpu` and a real `fs_platform::Machine`, so the *exact* scalar full-system semantics
//! (CSR read/write, sv32 translation, trap delivery, CLINT timer interrupts, MMIO) are reused
//! verbatim via `Cpu::step_system` — there is no second privileged/trap implementation to keep in
//! sync with `fs_riscv`'s. That is the whole point: correctness first, by construction.
//!
//! On top of that scalar-per-lane core, [`VecSystem::step`] carries one narrow, convergence-gated
//! SIMD fast path: when every active lane shares `pc`/`privilege`/`csr.satp` and none has a
//! pending trap this step, and the fetched instruction is a plain ALU op (`OpImm`/`Op` — no
//! memory, no CSR, no divide, never faults), the whole group executes it as one packed
//! `Simd<u32, LANES>` op via [`crate::simd_alu`] instead of `LANES` independent
//! `Cpu::step_system` calls. Every other instruction (loads, stores, branches, CSR, `mret`/
//! `sret`, `ecall`, MUL/DIV, atomics) — and any divergence at all — falls straight through to the
//! per-lane scalar path. See `DESIGN.md`'s "Full-system (VecSystem)" section for the honest
//! accounting of how often that fast path actually fires on real kernel code.

use fs_cov::CovBitmap;
use fs_mmu::{Access, Bus, Golden, PAGE_SIZE};
use fs_platform::{Clint, CowMachine, Machine};
use fs_riscv::sys::{self, Priv};
use fs_riscv::{decode, decode_compressed, AluOp, Cpu, Inst, LoadOp, SysExit};
use std::simd::prelude::*;
use std::sync::Arc;

use crate::{simd_alu, LANES};

/// Why a lane stopped advancing (full-system semantics: an HTIF `tohost` halt, or a fuzzing
/// hypercall — the same two outcomes `fs_riscv::SysExit`/`fs_platform::Stop` carry). Named
/// distinctly from the crate-root [`crate::LaneExit`] (which is [`crate::VecCpu`]'s user-mode-only
/// exit reason) to avoid confusion between the two executors' very different trap surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneExit {
    /// HTIF `tohost` exit, carrying the decoded exit code.
    Halt(u32),
    /// A fuzzing hypercall (`ecall` with the harness's reserved a7), carrying a0.
    Hypercall(u32),
    /// [`VecSystem::run_batch`]'s per-lane instruction budget was exhausted before this lane
    /// reached the requested hypercall — mirrors `fs_platform::Stop::Budget` (the scalar fuzzer's
    /// "case timed out" outcome) instead of a genuine guest-driven exit.
    Budget,
}

/// `LANES` full-system lanes: each a real `fs_riscv::Cpu` + `fs_platform::CowMachine` pair,
/// stepped together. Per-lane RAM is a page-copy-on-write view (`fs_platform::CowMachine`'s
/// `CowRam`) over one shared, immutable `Arc<Golden>` image (`docs/cow-shared-ram.md`'s PR3): all
/// `LANES` lanes share a single golden RAM copy process-wide, only diverging pages (stack/heap/
/// per-lane-dirtied kernel state) pay for a private 4 KiB overlay.
pub struct VecSystem {
    pub lanes: Box<[Cpu; LANES]>,
    pub bus: Box<[CowMachine; LANES]>,
    /// The shared immutable golden RAM image every lane's `CowMachine` COWs pages out of. Kept
    /// alive here (not just inside each lane's `Arc` clone) so `VecSystem` can report its size /
    /// rebuild lanes later without threading it through separately.
    golden: Arc<Golden>,
    /// Per-lane active mask; a lane is masked off once it halts or hypercalls.
    pub active: [bool; LANES],
    /// Set exactly when a lane transitions from active to inactive.
    pub exit: [Option<LaneExit>; LANES],
    /// Per-lane AFL-style edge coverage bitmap, fed by every non-fall-through `pc` transition this
    /// lane retires (mirrors `fs-cli`'s `run_case`/`run_case_bus`: `record_edge(prev_pc, cur_pc)`
    /// exactly when `cur != prev+4 && cur != prev+2`, and never on the step a lane halts/
    /// hypercalls). The converged SIMD/shared-fetch paths only ever execute straight-line
    /// fall-through instructions (ALU / same-address LOAD, both always `pc += ilen`), so in
    /// practice every recorded edge comes from the per-lane scalar loop today — but the check is
    /// applied uniformly to both paths so it stays correct if a future fast path ever executes a
    /// branch/jump.
    pub cov: Box<[CovBitmap; LANES]>,
    /// `step()` calls that executed a converged SIMD payload (ALU, or a same-address LOAD —
    /// `docs/cow-shared-ram.md` PR4) instead of the per-lane scalar loop.
    pub simd_steps: u64,
    /// `step()` calls that fell back to the per-lane scalar `Cpu::step_system` loop (for at least
    /// one active lane — i.e. every `step()` call that did not take a SIMD payload).
    pub scalar_steps: u64,
    /// `step()` calls whose instruction fetch (both halfwords, if a 32-bit instruction) was
    /// serviced by the PR4 shared translate+fetch path — ONE golden-image translate+read instead
    /// of `LANES` independent per-lane ones — regardless of whether the decoded instruction then
    /// went on to execute via a SIMD payload or declined to the per-lane loop for its actual
    /// execution (a decline still only pays for the shared fetch once, not `LANES` times). This is
    /// the PR4 perf metric `boot_vec` reports as "shared-path fraction".
    pub shared_fetch_steps: u64,
}

impl VecSystem {
    /// Clone one post-load `Cpu`+`Machine` (already loaded with firmware/kernel/dtb, not yet run)
    /// into all `LANES` lanes — byte-identical starting state, mirroring `VecCpu::new`'s
    /// "identical lanes" contract. Seed per-lane divergent inputs afterwards with
    /// [`VecSystem::set_reg`].
    ///
    /// Captures `machine.ram` as ONE shared, immutable `Golden` image and builds all `LANES`
    /// lanes as `CowMachine::from_golden` over the same `Arc` — a single golden RAM copy shared
    /// process-wide instead of `LANES` independent full RAM clones (`docs/cow-shared-ram.md`'s
    /// PR3). CLINT/UART start at their defaults per lane, exactly as `Machine::new` does (the
    /// template's `machine.clint`/`machine.uart` are not carried over, matching today's
    /// pre-swap behavior since a freshly loaded template is always CLINT/UART-default at this
    /// point).
    pub fn from_template(cpu: &Cpu, machine: &Machine) -> Self {
        let ram_base = machine.ram.base();
        let ram_size = machine.ram.size() as u32;
        let golden = Arc::new(Golden::from_mmu(&machine.ram));
        Self::from_golden_cpu(golden, cpu, ram_base, ram_size)
    }

    /// Build all `LANES` lanes from an ALREADY-CAPTURED shared golden image (`golden`) and one
    /// post-boot `Cpu` snapshot (typically the state at the fuzzing harness's SNAPSHOT hypercall,
    /// captured once on a scalar `Cpu`+`Machine` — see `examples/fuzz_vec.rs`), instead of
    /// `from_template`'s "capture golden from a `Machine` right now" convenience. This is the
    /// entry point the vectorized fuzzer actually uses: `golden` is typically shared with other
    /// consumers too (e.g. a future multi-thread fuzz worker, mirroring `fs-cli`'s `--jobs` path,
    /// which threads the very same `Arc<Golden>` across worker threads), so it is taken by
    /// reference-counted ownership here rather than re-derived from a `Machine`.
    pub fn from_golden_cpu(golden: Arc<Golden>, cpu: &Cpu, ram_base: u32, ram_size: u32) -> Self {
        let lanes: Box<[Cpu; LANES]> = Box::new(std::array::from_fn(|_| cpu.clone()));
        let bus: Box<[CowMachine; LANES]> =
            Box::new(std::array::from_fn(|_| CowMachine::from_golden(Arc::clone(&golden), ram_base, ram_size)));
        Self {
            lanes,
            bus,
            golden,
            active: [true; LANES],
            exit: [None; LANES],
            cov: Box::new(std::array::from_fn(|_| CovBitmap::new())),
            simd_steps: 0,
            scalar_steps: 0,
            shared_fetch_steps: 0,
        }
    }

    /// Total number of per-lane overlay pages currently allocated across all `LANES` lanes (each
    /// 4 KiB) — the memory-drop measurement: shared golden (once) + this many private overlays vs.
    /// `LANES` independent full RAM copies.
    pub fn total_overlay_pages(&self) -> usize {
        self.bus.iter().map(|b| b.ram.dirty_pages().len()).sum()
    }

    /// The shared golden RAM image's size in bytes (allocated exactly once, regardless of `LANES`).
    pub fn golden_pages(&self) -> usize {
        self.golden.num_pages()
    }

    /// Seed one lane's register (writes to x0 are dropped, matching hardware).
    pub fn set_reg(&mut self, lane: usize, i: u8, v: u32) {
        if i != 0 {
            self.lanes[lane].regs[i as usize] = v;
        }
    }

    pub fn lane_cpu(&self, lane: usize) -> &Cpu {
        &self.lanes[lane]
    }

    /// This lane's UART console output so far.
    pub fn uart(&self, lane: usize) -> &[u8] {
        &self.bus[lane].uart.out
    }

    /// This lane's accumulated AFL-style edge coverage bitmap.
    pub fn lane_coverage(&self, lane: usize) -> &CovBitmap {
        &self.cov[lane]
    }

    pub fn any_active(&self) -> bool {
        self.active.iter().any(|&a| a)
    }

    pub fn insns_retired(&self, lane: usize) -> u64 {
        self.lanes[lane].insns_retired
    }

    fn halt(&mut self, lane: usize, exit: LaneExit) {
        self.active[lane] = false;
        self.exit[lane] = Some(exit);
    }

    /// Inject one lowered fuzz program into lane `lane`'s guest physical memory: `pas[k]` is the
    /// physical address of the k-th program word (precomputed once by translating the guest's
    /// program-buffer VA, exactly as `fs-cli`'s `write_words` does for the scalar/parallel
    /// fuzzers). Each lane's `CowMachine` privately copy-on-writes only the pages this touches —
    /// the other 15 lanes' views of those same physical pages are untouched.
    pub fn inject_lane(&mut self, lane: usize, prog_pas: &[u32], words: &[u32]) {
        for (&pa, &w) in prog_pas.iter().zip(words) {
            let _ = self.bus[lane].store(pa, 4, w);
        }
    }

    /// Inject one lane's lowered scratch image (pointer-argument pointee bytes) into guest
    /// physical memory, word-by-word via the scratch region's precomputed per-word physical
    /// addresses — the per-lane analogue of `fs-cli`'s `write_scratch_bytes`. Bytes beyond `bytes`
    /// up to `scratch_pas.len()` words are zero-padded (matching the scalar fuzzer's behavior:
    /// scratch is a fixed-size region, only the lowered image's prefix is meaningful).
    pub fn inject_scratch(&mut self, lane: usize, scratch_pas: &[u32], bytes: &[u8]) {
        for (i, &pa) in scratch_pas.iter().enumerate() {
            let off = i * 4;
            let mut word = [0u8; 4];
            for (b, wb) in word.iter_mut().enumerate() {
                if let Some(&v) = bytes.get(off + b) {
                    *wb = v;
                }
            }
            let _ = self.bus[lane].store(pa, 4, u32::from_le_bytes(word));
        }
    }

    /// Step every active lane until each one has either hit the fuzzing DONE hypercall
    /// (`SysExit::Hypercall(done_code)` — masked off with [`LaneExit::Hypercall`]) or exhausted
    /// its own per-lane instruction budget (masked off with [`LaneExit::Budget`], mirroring
    /// `fs_platform::Stop::Budget`). Lanes finish at different times — once a lane's exit
    /// condition is hit it stops being stepped (masked off), while the rest keep going; the
    /// SIMD/shared-fetch fast paths only fire while the still-active subset stays converged, so
    /// most of a batch runs per-lane scalar once lanes start diverging and dropping out. That
    /// divergence (not a bug) is exactly the honest cost `docs/DESIGN.md`'s performance section
    /// has to account for.
    pub fn run_batch(&mut self, done_code: u32, budget: u64) {
        // `step()` already masks a lane off (`LaneExit::Halt`/`LaneExit::Hypercall`) the instant
        // its `Cpu::step_system` reports one — for this harness's guest agent the only hypercall
        // code that can fire mid-batch is `done_code` (SNAPSHOT only ever fires once, before any
        // `VecSystem` batch exists). `done_code` is threaded through as a documented parameter
        // (and asserted below) rather than re-checked here, so callers get an immediate, precise
        // panic if that assumption is ever violated instead of a silently-wrong batch.
        let start_insns: [u64; LANES] = std::array::from_fn(|l| self.lanes[l].insns_retired);
        while self.any_active() {
            for (lane, &start) in start_insns.iter().enumerate() {
                if self.active[lane] && self.lanes[lane].insns_retired - start >= budget {
                    self.halt(lane, LaneExit::Budget);
                }
            }
            if !self.any_active() {
                break;
            }
            self.step();
        }
        for lane in 0..LANES {
            if let Some(LaneExit::Hypercall(c)) = self.exit[lane] {
                debug_assert_eq!(c, done_code, "lane {lane}: unexpected hypercall code {c} (expected DONE={done_code})");
            }
        }
    }

    /// Reset every lane back to the golden snapshot state for the next batch: drop this lane's
    /// RAM overlays (`CowRam::reset` — O(dirty) directory entries dropped, zero byte copy-back,
    /// mirroring `fs-cli`'s `reset_cow`), restore the hart to a fresh clone of `cpu`, restore
    /// CLINT to a fresh clone of `clint`, truncate UART output back to `base_uart`, clear the
    /// active mask/exit reason, and clear each lane's per-batch coverage bitmap (coverage is
    /// merged into the caller's global `VirginMap` between batches, exactly as the scalar fuzzer
    /// merges its per-case `CovBitmap` — see `examples/fuzz_vec.rs`).
    pub fn reset_batch(&mut self, cpu: &Cpu, clint: &Clint, base_uart: usize) {
        for lane in 0..LANES {
            self.lanes[lane] = cpu.clone();
            self.bus[lane].ram.reset();
            self.bus[lane].clint = clint.clone();
            self.bus[lane].uart.out.truncate(base_uart);
            self.active[lane] = true;
            self.exit[lane] = None;
            self.cov[lane].clear();
        }
    }

    /// Advance every active lane by exactly one instruction.
    ///
    /// Correctness contract: for any lane not handled by the SIMD fast path this step, the
    /// architectural effect is byte-identical to `fs_platform::run_until`'s per-step body —
    /// `bus.clint.mtime = cpu.virtual_time(); sync_timer(cpu, bus); cpu.step_system(bus)` — run on
    /// a standalone scalar `Cpu`+`Machine`. The SIMD fast path only ever engages for an
    /// instruction class (converged ALU) that provably produces the identical result, and declines
    /// (falling through to the scalar path) on the slightest doubt.
    pub fn step(&mut self) {
        // Universal per-lane CLINT/timer sync (mirrors `fs_platform::run_until`'s per-step prep):
        // needed unconditionally, whether a lane ends up on the SIMD fast path or the scalar one,
        // since even the fast path's convergence precondition (no pending trap) depends on it.
        for lane in 0..LANES {
            if !self.active[lane] {
                continue;
            }
            self.bus[lane].clint.mtime = self.lanes[lane].virtual_time();
            fs_platform::sync_timer_cow(&mut self.lanes[lane], &self.bus[lane]);
        }

        // Snapshot each active lane's pre-step `pc`, for edge-coverage recording below — captured
        // before either path mutates anything, exactly mirroring `fs-cli`'s `run_case`/
        // `run_case_bus` (`let prev = cpu.pc;` immediately before `cpu.step_system(m)`).
        let active_before = self.active;
        let prev_pc: [u32; LANES] = std::array::from_fn(|lane| self.lanes[lane].pc);

        if self.try_converged_fast_path() {
            self.simd_steps += 1;
            // Coverage (`docs/cow-shared-ram.md`-style honesty note): the converged SIMD/shared
            // paths only ever execute straight-line ALU/LOAD instructions, which always advance
            // `pc` by `ilen` (2 or 4) — i.e. always the "fall-through" case `fs-cli`'s coverage
            // gate excludes — so this loop does not currently record anything in practice. It is
            // still applied here (not skipped) so a future fast path that executes a branch/jump
            // stays correct by construction instead of silently under-reporting coverage.
            for lane in 0..LANES {
                if !active_before[lane] {
                    continue;
                }
                let (prev, cur) = (prev_pc[lane], self.lanes[lane].pc);
                if cur != prev.wrapping_add(4) && cur != prev.wrapping_add(2) {
                    self.cov[lane].record_edge(prev, cur);
                }
            }
            return;
        }
        self.scalar_steps += 1;
        for lane in 0..LANES {
            if !active_before[lane] {
                continue;
            }
            let prev = prev_pc[lane];
            match self.lanes[lane].step_system(&mut self.bus[lane]) {
                SysExit::Continue => {
                    let cur = self.lanes[lane].pc;
                    if cur != prev.wrapping_add(4) && cur != prev.wrapping_add(2) {
                        self.cov[lane].record_edge(prev, cur);
                    }
                }
                SysExit::Halt(c) => self.halt(lane, LaneExit::Halt(c)),
                SysExit::Hypercall(c) => self.halt(lane, LaneExit::Hypercall(c)),
            }
        }
    }

    /// The converged-lane fast path (`docs/cow-shared-ram.md` PR4). Returns `true` (having
    /// executed the instruction across every active lane) only when ALL of the following hold,
    /// checked *before* anything is mutated beyond the timer-pending-bit refresh every step
    /// already needs (see below):
    ///
    /// - every active lane shares `pc`, `privilege`, and `csr.satp` (so translation resolves the
    ///   same way for every lane, *pending* the shared-fetch/overlay checks below);
    /// - `update_timers`'s effect (replicated here byte-for-byte from `fs_riscv::Cpu`'s private
    ///   method, since the fast path bypasses `step_system`) leaves no active lane with a pending,
    ///   enabled interrupt — a trap this step would need full `step_system` trap-vectoring, which
    ///   this fast path does not implement;
    /// - the instruction is translated+fetched ONCE against the shared golden image
    ///   ([`Self::shared_fetch16`]) instead of once per lane — the PR4 perf win: a group that
    ///   turns out not to qualify below still only pays for this 1× speculative fetch, not `LANES`×;
    /// - the decoded instruction is `Inst::OpImm`/`Inst::Op` (packed ALU, no memory/CSR, cannot
    ///   fault on any operand value — dispatched to the existing [`Self::dispatch_simd_alu`]
    ///   payload), or `Inst::Load` whose effective address also happens to agree across every
    ///   active lane ([`Self::try_shared_load`] translates+reads that one address once too).
    ///
    /// On any doubt this declines, having mutated nothing except the same per-lane
    /// `mip`/CLINT-derived bits `Cpu::step_system` would unconditionally mutate anyway (see
    /// `update_timers` below) — falling through to the scalar per-lane loop reproduces the exact
    /// same state from there (that loop's `step_system` recomputes the identical, idempotent
    /// `update_timers` result and does its own translation).
    fn try_converged_fast_path(&mut self) -> bool {
        let active = self.active;
        let Some(first) = active.iter().position(|&a| a) else {
            return false;
        };
        let pc0 = self.lanes[first].pc;
        let priv0 = self.lanes[first].privilege;
        let satp0 = self.lanes[first].csr.satp;
        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            let cpu = &self.lanes[lane];
            if cpu.pc != pc0 || cpu.privilege != priv0 || cpu.csr.satp != satp0 {
                return false;
            }
        }

        // Refresh each active lane's timer-pending mip bits (STIP/MTIP) — the same computation
        // `Cpu::step_system` performs unconditionally at the top of every step via its private
        // `update_timers`. Decline (without having done anything else) if any lane now has a
        // pending, enabled interrupt: that lane needs `step_system`'s trap-vectoring, which this
        // fast path does not implement.
        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            update_timers(&mut self.lanes[lane]);
            if pending_interrupt(&self.lanes[lane]) {
                return false;
            }
        }

        // Shared translate+fetch (PR4): ONE golden-image translate+read instead of `LANES`
        // independent per-lane ones. Declines to the per-lane loop (which retranslates/refetches
        // each lane correctly, exactly as it always has) on any doubt at all.
        let Some(lo0) = self.shared_fetch16(first, pc0) else {
            return false;
        };
        let (inst, ilen) = if lo0 & 0x3 != 0x3 {
            (decode_compressed(lo0), 2u32)
        } else {
            let Some(hi0) = self.shared_fetch16(first, pc0.wrapping_add(2)) else {
                return false;
            };
            (decode((lo0 as u32) | ((hi0 as u32) << 16)), 4u32)
        };
        // The instruction fetch itself succeeded via the shared path, regardless of what the
        // dispatch below does with it — this is the metric `boot_vec` reports as "shared-path
        // fraction": how often the O(LANES) speculative fetch was replaced by an O(1) one.
        self.shared_fetch_steps += 1;

        match inst {
            Inst::OpImm { op, rd, rs1, imm } => {
                self.dispatch_simd_alu(active, AluDispatch { op, rd, rs1, rs2: None, imm }, ilen);
                true
            }
            Inst::Op { op, rd, rs1, rs2 } => {
                self.dispatch_simd_alu(active, AluDispatch { op, rd, rs1, rs2: Some(rs2), imm: 0 }, ilen);
                true
            }
            Inst::Load { op, rd, rs1, imm } => self.try_shared_load(active, op, rd, rs1, imm, ilen),
            // Stores (data is per-lane), branches/jumps, CSR/system/mul-div/atomics: the per-lane
            // scalar loop handles these exactly as before. The shared fetch above still saved
            // `LANES`-1 speculative translate+fetches for this one instruction.
            _ => false,
        }
    }

    /// Translate+fetch one instruction halfword at `va` for the WHOLE converged group in one
    /// shot, against the shared golden image, instead of the O(`LANES`) per-lane translate+fetch
    /// this used to require. Uses lane `first`'s CPU state (every active lane shares `pc`/
    /// `privilege`/`csr.satp` by the caller's convergence check, so translation resolves
    /// identically for all of them via [`fs_riscv::Cpu::xlate_golden_readonly`]).
    ///
    /// Declines (`None`), having mutated nothing, when: the read-only walk itself declines (an
    /// unmapped/faulting VA, an A/D bit that would need setting, a superpage edge, an
    /// out-of-golden-bounds PTE read, or a page-table page privately COW'd in any lane since
    /// golden was captured — the mutating per-lane `xlate` must run instead in every case); the
    /// resulting physical address falls outside golden's RAM window entirely; the resulting code
    /// page is privately COW'd (self-modified) in ANY lane, active or not — a lane that diverged
    /// its own code/PTE page must never have that divergence painted over by a golden broadcast;
    /// or the golden fetch itself declines (out of bounds / missing `PERM_EXEC`).
    ///
    /// In debug builds, on success this additionally performs lane `first`'s REAL `xlate`+
    /// `ifetch16` (idempotent here: a clean `xlate_golden_readonly` success already proves A/D
    /// need no writeback, so this cannot mutate any memory — it only fills lane `first`'s own TLB,
    /// exactly as a future real step would anyway) and asserts the two agree byte-for-byte — the
    /// runtime guard against any drift between the golden-readonly walk and the real one (compiled
    /// out entirely in release).
    fn shared_fetch16(&mut self, first: usize, va: u32) -> Option<u16> {
        let golden = &self.golden;
        let bus = &self.bus;
        let pa = self.lanes[first]
            .xlate_golden_readonly(golden, va, Access::Exec, |addr| any_lane_overlaid(golden, bus, addr))
            .ok()?;
        if any_lane_overlaid(golden, bus, pa) {
            return None; // some lane privately modified this code page: never broadcast golden
        }
        let hw = self.golden.fetch_u16(pa)?;

        #[cfg(debug_assertions)]
        {
            let real_pa = self.lanes[first].xlate(&mut self.bus[first], va, Access::Exec).unwrap_or_else(|e| {
                panic!("PR4 guard: shared fetch succeeded (va={va:#x} pa={pa:#x}) but lane {first}'s real xlate faulted: {e}")
            });
            debug_assert_eq!(
                real_pa, pa,
                "PR4 guard: shared golden translation ({pa:#x}) disagreed with lane {first}'s real xlate ({real_pa:#x}) at va={va:#x}"
            );
            let real_hw = self.bus[first].ifetch16(real_pa).unwrap_or_else(|f| {
                panic!("PR4 guard: shared fetch succeeded (va={va:#x} pa={pa:#x}) but lane {first}'s real ifetch16 faulted: {f}")
            });
            debug_assert_eq!(
                hw, real_hw,
                "PR4 guard: shared golden-broadcast fetch ({hw:#06x}) disagreed with lane {first}'s real xlate+ifetch16 ({real_hw:#06x}) at va={va:#x}"
            );
        }

        Some(hw)
    }

    /// Converged same-address LOAD fast path (PR4): computes each active lane's effective address
    /// from its own (independent) registers — this is address *computation*, not the shared
    /// fetch above, so addresses may legitimately diverge even though `pc`/`satp` agree — and only
    /// proceeds if every active lane's address is identical. On agreement, translates that one
    /// address read-only against golden (`Access::Read`), requires the containing page to be
    /// un-overlaid in every lane (the same self-modified-page guard `shared_fetch16` uses — a
    /// store to a page one lane reads from must never be masked by a golden broadcast), then reads
    /// the value ONCE from golden and broadcasts it (sign-extended per `LoadOp`) to every active
    /// lane. Declines (`false`, having mutated nothing) on address divergence, a misaligned/
    /// page-crossing access (serviced byte-wise by the scalar `Cpu::load` instead), a translation
    /// decline, an overlaid page, or a golden read miss; the per-lane scalar loop then re-services
    /// every active lane exactly as it always has. Stores are never handled here — store data is
    /// inherently per-lane.
    fn try_shared_load(&mut self, active: [bool; LANES], op: LoadOp, rd: u8, rs1: u8, imm: i32, ilen: u32) -> bool {
        let Some(first) = active.iter().position(|&a| a) else {
            return false;
        };
        let addr0 = rd_reg(&self.lanes[first], rs1).wrapping_add(imm as u32);
        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            if rd_reg(&self.lanes[lane], rs1).wrapping_add(imm as u32) != addr0 {
                return false; // divergent effective address: per-lane loop handles the gather
            }
        }
        let (size, signed) = match op {
            LoadOp::Lb => (1u8, true),
            LoadOp::Lbu => (1, false),
            LoadOp::Lh => (2, true),
            LoadOp::Lhu => (2, false),
            LoadOp::Lw => (4, false),
        };
        // Misaligned/page-crossing accesses are serviced byte-wise by the scalar `Cpu::load`; this
        // fast path only ever handles the common naturally-aligned, single-page case.
        if !addr0.is_multiple_of(size as u32) {
            return false;
        }
        let golden = &self.golden;
        let bus = &self.bus;
        let Ok(pa) =
            self.lanes[first].xlate_golden_readonly(golden, addr0, Access::Read, |addr| any_lane_overlaid(golden, bus, addr))
        else {
            return false;
        };
        if any_lane_overlaid(golden, bus, pa) {
            return false; // some lane privately wrote this page: never broadcast golden
        }
        let Some(raw) = self.golden.read_sized(pa, size) else {
            return false;
        };

        #[cfg(debug_assertions)]
        {
            let real_pa = self.lanes[first].xlate(&mut self.bus[first], addr0, Access::Read).unwrap_or_else(|e| {
                panic!("PR4 guard: shared load succeeded (va={addr0:#x} pa={pa:#x}) but lane {first}'s real xlate faulted: {e}")
            });
            debug_assert_eq!(
                real_pa, pa,
                "PR4 guard: shared golden load translation ({pa:#x}) disagreed with lane {first}'s real xlate ({real_pa:#x}) at va={addr0:#x}"
            );
            let real_val = self.bus[first].load(real_pa, size).unwrap_or_else(|f| {
                panic!("PR4 guard: shared load succeeded (va={addr0:#x} pa={pa:#x}) but lane {first}'s real bus.load faulted: {f}")
            });
            debug_assert_eq!(
                raw, real_val,
                "PR4 guard: shared golden-broadcast load ({raw:#x}) disagreed with lane {first}'s real load ({real_val:#x}) at va={addr0:#x}"
            );
        }

        let value = if signed {
            match size {
                1 => raw as u8 as i8 as i32 as u32,
                2 => raw as u16 as i16 as i32 as u32,
                _ => raw,
            }
        } else {
            raw
        };

        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            if rd != 0 {
                self.lanes[lane].regs[rd as usize] = value;
            }
            self.lanes[lane].pc = self.lanes[lane].pc.wrapping_add(ilen);
            self.lanes[lane].insns_retired += 1;
        }
        true
    }

    /// Shared packed-ALU payload: reads each active lane's own `rs1`/`rs2`-or-`imm` operands
    /// (independent per lane even though `pc`/`satp` agreed), executes `op` across all `LANES`
    /// lanes as one `Simd<u32, LANES>` op, and writes back `rd` + advances `pc`/`insns_retired`
    /// for every active lane. Never faults, never touches memory/CSR — the only reason `OpImm`/
    /// `Op` are safe to always dispatch here once decoded.
    fn dispatch_simd_alu(&mut self, active: [bool; LANES], d: AluDispatch, ilen: u32) {
        let mut a_arr = [0u32; LANES];
        let mut b_arr = [0u32; LANES];
        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            a_arr[lane] = rd_reg(&self.lanes[lane], d.rs1);
            b_arr[lane] = match d.rs2 {
                Some(rs2) => rd_reg(&self.lanes[lane], rs2),
                None => d.imm as u32,
            };
        }
        let result = simd_alu(d.op, Simd::from_array(a_arr), Simd::from_array(b_arr)).to_array();

        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            if d.rd != 0 {
                self.lanes[lane].regs[d.rd as usize] = result[lane];
            }
            self.lanes[lane].pc = self.lanes[lane].pc.wrapping_add(ilen);
            self.lanes[lane].insns_retired += 1;
        }
    }
}

/// Bundled operands for [`VecSystem::dispatch_simd_alu`] (`OpImm`'s immediate and `Op`'s `rs2`
/// collapse into one dispatch shape: `rs2: None` means "use `imm`" exactly as the scalar
/// interpreter's own `OpImm`/`Op` handling does) — keeps the method under clippy's argument-count
/// lint and reads as one decoded-instruction value instead of five loose parameters.
struct AluDispatch {
    op: AluOp,
    rd: u8,
    rs1: u8,
    rs2: Option<u8>,
    imm: i32,
}

/// Whether physical address `addr` cannot be trusted against `golden` — either it falls outside
/// golden's RAM window entirely, or the containing 4 KiB page has been privately COW'd in any
/// lane's `CowRam` since `golden` was captured (its *real* current content then lives in that
/// lane's overlay, not in `golden`). This is the ONE guard consumed in two places: threaded into
/// `Cpu::xlate_golden_readonly` as its `page_overlaid` callback (guards the page-table pages the
/// walk itself reads), and called directly on the final leaf physical address a walk resolves to
/// (guards the code/data page `shared_fetch16`/`try_shared_load` then read from golden) — both are
/// required, since the walk only ever reads page-table pages, never the leaf itself.
///
/// This is the correctness-critical guard for PR4's whole premise: `golden` is typically captured
/// once (e.g. `VecSystem::from_template`, well before any lane executes), so ANY physical page —
/// including one a kernel later builds its own page tables in — can have diverged from golden by
/// the time a lane actually consults it. A lane that hasn't touched that page still reads through
/// to golden correctly; the moment ANY lane's `CowRam` has privately overlaid it, golden can no
/// longer be trusted for it at all (even by lanes that never wrote it themselves), so this checks
/// every lane, not just the currently-translating one.
#[inline]
fn any_lane_overlaid(golden: &Golden, bus: &[CowMachine; LANES], addr: u32) -> bool {
    let base = golden.base();
    if addr < base {
        return true; // outside golden's window: nothing to trust
    }
    let pn = ((addr - base) as usize) / PAGE_SIZE;
    pn >= golden.num_pages() || bus.iter().any(|b| b.ram.is_overlaid(pn))
}

/// `rd_reg` (x0 hardwired zero), copied from `fs_riscv::Cpu`'s private helper — `Cpu::regs` is a
/// plain `[u32; 32]` array with no built-in x0 enforcement, so every reader must go through this.
#[inline]
fn rd_reg(cpu: &Cpu, i: u8) -> u32 {
    if i == 0 {
        0
    } else {
        cpu.regs[i as usize]
    }
}

/// Byte-for-byte copy of `fs_riscv::Cpu`'s private `update_timers`: refreshes the Sstc supervisor
/// timer (STIP) and M-timer (MTIP) pending bits in `mip` from virtual time vs. `stimecmp`/
/// `mtimecmp`. Needed here because the SIMD fast path bypasses `Cpu::step_system` (which calls
/// this unconditionally at the top of every step) for the lanes it handles.
#[inline]
fn update_timers(cpu: &mut Cpu) {
    let t = cpu.virtual_time();
    let stip = 1 << 5;
    if t >= cpu.csr.stimecmp {
        cpu.csr.mip |= stip;
    } else {
        cpu.csr.mip &= !stip;
    }
    let mtip = 1 << 7;
    if t >= cpu.csr.mtimecmp {
        cpu.csr.mip |= mtip;
    } else {
        cpu.csr.mip &= !mtip;
    }
}

/// Byte-for-byte copy of `fs_riscv::Cpu`'s private `pending_interrupt`: whether there is a
/// currently pending *and* enabled interrupt (standard RISC-V priority order), i.e. whether the
/// next `step_system` call would deliver a trap instead of executing an instruction. The fast path
/// must decline whenever this is true for any active lane — it has no trap-vectoring of its own.
#[inline]
fn pending_interrupt(cpu: &Cpu) -> bool {
    let pending = cpu.csr.mip & cpu.csr.mie;
    if pending == 0 {
        return false;
    }
    for &code in &[11u32, 3, 7, 9, 1, 5] {
        let bit = 1 << code;
        if pending & bit == 0 {
            continue;
        }
        let to_s = (cpu.csr.mideleg >> code) & 1 != 0;
        let enabled = if to_s {
            cpu.privilege == Priv::U
                || (cpu.privilege == Priv::S && cpu.csr.mstatus & sys::MSTATUS_SIE != 0)
        } else {
            cpu.privilege != Priv::M || cpu.csr.mstatus & sys::MSTATUS_MIE != 0
        };
        if enabled {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_platform::Machine;
    use fs_riscv::{asm, A0, T0, T1, X0};

    const BASE: u32 = 0x8000_0000;

    /// Build a fresh `Cpu`+`Machine` template with `prog` mapped at `BASE`, `handler_prog` mapped
    /// at `handler`, RAM otherwise RW, `mtvec = handler`, and `tohost` registered as the HTIF exit
    /// word — the exact memory layout both `VecSystem::from_template` and the standalone scalar
    /// oracle run from, so the only difference between the two is the executor itself.
    fn make_template(prog: &[u32], handler: u32, handler_prog: &[u32], tohost: u32) -> (Cpu, Machine) {
        let mut m = Machine::new(BASE, 0x1_0000);
        m.ram.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        m.ram.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut hbytes = Vec::new();
        for w in handler_prog {
            hbytes.extend_from_slice(&w.to_le_bytes());
        }
        m.ram.map(handler, &hbytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cpu = Cpu::new(BASE);
        cpu.htif_tohost = Some(tohost);
        cpu.csr.mtvec = handler;
        (cpu, m)
    }

    /// Run a standalone scalar `Cpu`+`Machine` (fully independent of `VecSystem`) to its HTIF
    /// halt, seeding one register first — the oracle every `VecSystem` lane is checked against.
    fn run_scalar_to_halt(
        prog: &[u32],
        handler: u32,
        handler_prog: &[u32],
        tohost: u32,
        seed: (u8, u32),
    ) -> (u32, Cpu) {
        let (mut cpu, mut m) = make_template(prog, handler, handler_prog, tohost);
        cpu.regs[seed.0 as usize] = seed.1;
        match fs_platform::run_until(&mut cpu, &mut m, 100_000) {
            fs_platform::Stop::Halt(c) => (c, cpu),
            other => panic!("scalar oracle run did not halt: {other:?}"),
        }
    }

    /// Minimal encoders `fs_riscv::asm` doesn't provide (funct3=4/7 I-type, B-type, csrrw, mret),
    /// mirroring `decode`'s tables exactly — used to hand-assemble the CSR/trap/mret/branch
    /// property-test program below.
    fn i_type(op: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
        ((imm as u32 & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
    }
    fn b_type(op: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
        let i = imm as u32;
        ((i >> 12) & 1) << 31
            | ((i >> 5) & 0x3f) << 25
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | (funct3 << 12)
            | ((i >> 1) & 0xf) << 8
            | ((i >> 11) & 1) << 7
            | op
    }
    fn andi(rd: u8, rs1: u8, imm: i32) -> u32 {
        i_type(0x13, 7, rd, rs1, imm)
    }
    fn slli(rd: u8, rs1: u8, shamt: i32) -> u32 {
        i_type(0x13, 1, rd, rs1, shamt)
    }
    fn ori(rd: u8, rs1: u8, imm: i32) -> u32 {
        i_type(0x13, 6, rd, rs1, imm)
    }
    fn blt(rs1: u8, rs2: u8, imm: i32) -> u32 {
        b_type(0x63, 4, rs1, rs2, imm)
    }
    fn csrrw(rd: u8, csr: u16, rs1: u8) -> u32 {
        ((csr as u32) << 20) | ((rs1 as u32) << 15) | (1 << 12) | ((rd as u32) << 7) | 0x73
    }
    fn csrrs(rd: u8, csr: u16, rs1: u8) -> u32 {
        ((csr as u32) << 20) | ((rs1 as u32) << 15) | (2 << 12) | ((rd as u32) << 7) | 0x73
    }
    fn mret() -> u32 {
        0x3020_0073
    }

    /// A hermetic full-system property test (no external firmware files, decision #26-style):
    /// every lane runs, from a byte-identical `Cpu`+`Machine` template, a program with per-lane
    /// DIVERGENT seed-driven ALU results (SIMD fast-path candidates), a genuine per-lane-divergent
    /// conditional branch (some lanes take it, some don't — forcing the fast path to decline and
    /// the two sub-groups to run scalar independently), a `csrrw` (MSCRATCH read/modify/write), an
    /// `ecall` that traps to an M-mode handler (`mtvec`) which advances `mepc` past the `ecall` and
    /// `mret`s back, and a final HTIF exit whose code threads the whole computation through. Every
    /// lane's final architectural state must match an independently-run standalone scalar
    /// `Cpu`+`Machine` seeded identically — the correctness contract `VecSystem::step` documents.
    #[test]
    fn vec_system_matches_scalar_oracle_across_divergent_seeds_with_branch_csr_and_trap() {
        const T2: u8 = 7; // per-lane seed register (x7)
        const T4: u8 = 29; // branch threshold
        const T5: u8 = 30; // HTIF pointer scratch
        const MSCRATCH: u16 = 0x340;
        const MEPC: u16 = 0x341;
        let handler = BASE + 0x100;
        let tohost = BASE + 0x2000;

        // idx  addr        instruction
        //  0   BASE+0x00   t0 = seed + 5                (ALU — SIMD fast-path candidate)
        //  1   BASE+0x04   t0 &= 0xff                    (ALU)
        //  2   BASE+0x08   t4 = 100                      (ALU: branch threshold)
        //  3   BASE+0x0c   if t0 < t4: pc += 8 (skip 4)  (Branch — always declines the fast path)
        //  4   BASE+0x10   nop (only lanes with t0>=100 execute this)
        //  5   BASE+0x14   t1 = mscratch; mscratch = t0  (csrrw — always scalar)
        //  6   BASE+0x18   ecall                         (traps to `handler` via mtvec)
        //  7   BASE+0x1c   a0 = t0 + t1                  (resumed here: mepc was bumped past ecall)
        //  8   BASE+0x20   a0 <<= 1
        //  9   BASE+0x24   t1 = a0 | 1
        // 10   BASE+0x28   t5 = tohost
        // 11   BASE+0x2c   [t5] = t1                     (HTIF exit, code = original t0)
        let prog = vec![
            asm::addi(T0, T2, 5),     // 0
            andi(T0, T0, 0xff),       // 1
            asm::addi(T4, X0, 100),   // 2
            blt(T0, T4, 8),           // 3
            asm::addi(X0, X0, 0),     // 4: nop
            csrrw(T1, MSCRATCH, T0),  // 5
            asm::ecall(),             // 6
            asm::add(A0, T0, T1),     // 7
            slli(A0, A0, 1),          // 8
            ori(T1, A0, 1),           // 9
            asm::lui(T5, tohost),     // 10
            asm::sw(T5, T1, 0),       // 11
        ];
        // Handler: mepc += 4 (skip past the `ecall`), then `mret`.
        let handler_prog = vec![
            csrrs(28, MEPC, X0), // t3 = mepc (no write: rs1=x0)
            asm::addi(28, 28, 4),
            csrrw(X0, MEPC, 28), // mepc = t3 (rd=x0: old value discarded)
            mret(),
        ];

        let (cpu, m) = make_template(&prog, handler, &handler_prog, tohost);
        let mut vs = VecSystem::from_template(&cpu, &m);
        // Per-lane divergent seeds: some land t0 < 100 (branch taken), some >= 100 (not taken).
        let seeds: [u32; LANES] = std::array::from_fn(|lane| (lane as u32) * 13);
        for (lane, &seed) in seeds.iter().enumerate() {
            vs.set_reg(lane, T2, seed);
        }

        let mut guard = 0;
        while vs.any_active() {
            vs.step();
            guard += 1;
            assert!(guard < 10_000, "VecSystem program did not converge to a halt");
        }

        assert!(vs.simd_steps > 0, "the leading straight-line ALU ops should hit the SIMD fast path");
        assert!(vs.scalar_steps > 0, "the branch/csrrw/ecall/mret ops must fall back to scalar");

        for (lane, &seed) in seeds.iter().enumerate() {
            let expected_t0 = seed.wrapping_add(5) & 0xff;
            let (oracle_code, oracle_cpu) =
                run_scalar_to_halt(&prog, handler, &handler_prog, tohost, (T2, seed));

            match vs.exit[lane] {
                Some(LaneExit::Halt(c)) => {
                    assert_eq!(c, expected_t0, "lane {lane} HTIF exit code");
                    assert_eq!(c, oracle_code, "lane {lane} vs scalar oracle exit code");
                }
                other => panic!("lane {lane}: expected a Halt exit, got {other:?}"),
            }
            for reg in 0..32 {
                assert_eq!(
                    vs.lanes[lane].regs[reg], oracle_cpu.regs[reg],
                    "lane {lane} register x{reg} vs scalar oracle"
                );
            }
            assert_eq!(vs.lanes[lane].pc, oracle_cpu.pc, "lane {lane} pc vs scalar oracle");
            assert_eq!(vs.lanes[lane].csr.mscratch, oracle_cpu.csr.mscratch, "lane {lane} mscratch");
            assert_eq!(vs.lanes[lane].csr.mcause, oracle_cpu.csr.mcause, "lane {lane} mcause");
            assert_eq!(vs.lanes[lane].csr.mepc, oracle_cpu.csr.mepc, "lane {lane} mepc");
            assert_eq!(vs.lanes[lane].privilege, oracle_cpu.privilege, "lane {lane} privilege");
        }

        // Sanity: seeds really did drive divergent branch outcomes (not all lanes on one side).
        let taken = seeds.iter().filter(|&&s| (s.wrapping_add(5) & 0xff) < 100).count();
        assert!(taken > 0 && taken < LANES, "seeds should split across both sides of the branch");
    }

    /// Register-register encoders `fs_riscv::asm` doesn't provide (AND/XOR), mirroring `decode`'s
    /// tables exactly (funct3=7/funct7=0 -> And, funct3=4/funct7=0 -> Xor) — used below to build a
    /// branchless per-lane address select without needing a control-flow-diverging branch.
    fn r_type(op: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
        (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
    }
    fn and_r(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 7, 0x00, rd, rs1, rs2)
    }
    fn xor_r(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 4, 0x00, rd, rs1, rs2)
    }
    /// Standard RISC-V "li" lowering, always emitted as exactly 2 instructions (`lui`+`addi`, even
    /// when the low 12 bits are zero) so every caller can rely on a fixed, statically-known length
    /// when laying out a program by hand.
    fn li2(rd: u8, imm: u32) -> [u32; 2] {
        let hi = imm.wrapping_add(0x800) & 0xffff_f000;
        let lo = imm.wrapping_sub(hi) as i32;
        [asm::lui(rd, hi), asm::addi(rd, rd, lo)]
    }

    /// PR4-specific correctness stress test: a converged lane group (identical `pc`/`privilege`/
    /// `csr.satp` throughout — no paging, no branching, so the group NEVER desyncs on `pc`) where,
    /// via a branchless per-lane address select, HALF the lanes self-modify their own code page
    /// (overwriting the very next instruction they are about to fetch) while the other half write
    /// to an unrelated scratch page instead. This is the "union/overlay decline" risk
    /// `docs/cow-shared-ram.md` calls out: at the fetch immediately after the store, the group is
    /// perfectly `pc`-converged, but the modifying lanes' code page is privately COW'd while the
    /// others' is still golden. If `VecSystem::shared_fetch16` ever failed to consult
    /// `CowRam::is_overlaid` (or checked only some lanes instead of every lane), it would broadcast
    /// golden's stale bytes to the modifying lanes too, silently corrupting their execution. This
    /// test proves each lane instead sees exactly its own bytes, matching an independently-run
    /// scalar oracle seeded identically.
    #[test]
    fn vec_system_declines_shared_fetch_when_a_lane_self_modifies_its_code_page() {
        const T2: u8 = 7; // per-lane seed register (x7): seed & 1 selects "self-modify or not"
        const BIT: u8 = 28;
        const MASK: u8 = 29;
        const SCRATCH: u8 = 18;
        const XORC: u8 = 19;
        const SELTMP: u8 = 20;
        const ADDR: u8 = 21;
        const VAL: u8 = 22;
        const T5: u8 = 23; // tohost pointer scratch
        let scratch_addr = BASE + 0x1000; // a different 4 KiB page than the code (page 0)
        let new_instr: u32 = asm::addi(A0, X0, 42); // what a "self-modified" lane will execute
        let tohost = BASE + 0x3000;

        // Every lane executes the IDENTICAL straight-line instruction sequence below (no branch
        // anywhere), so the group's `pc` never desyncs — only the STORE's target *address*
        // (computed branchlessly from the per-lane seed bit) diverges between lanes.
        let mut prog: Vec<u32> = Vec::new();
        prog.push(andi(BIT, T2, 1)); // 0: bit = seed & 1
        prog.push(asm::sub(MASK, X0, BIT)); // 1: mask = 0 - bit (all-ones if bit=1, else 0)
        prog.extend(li2(SCRATCH, scratch_addr)); // 2,3
        // target_addr = BASE + 11*4 (index 11 below, the "maybe modified" instruction slot).
        let target_addr = BASE + 11 * 4;
        let xorc = target_addr ^ scratch_addr;
        prog.extend(li2(XORC, xorc)); // 4,5
        prog.push(and_r(SELTMP, MASK, XORC)); // 6: seltmp = mask & xorc
        prog.push(xor_r(ADDR, SCRATCH, SELTMP)); // 7: addr = scratch ^ seltmp = target if bit=1 else scratch
        prog.extend(li2(VAL, new_instr)); // 8,9
        prog.push(asm::sw(ADDR, VAL, 0)); // 10: store new_instr to ADDR (code page or scratch page)
        assert_eq!(prog.len(), 11, "index 11 must land exactly at target_addr");
        prog.push(asm::addi(A0, X0, 1)); // 11: target_addr — default/unmodified: a0 = 1
        prog.extend(li2(T5, tohost)); // 12,13
        prog.push(slli(A0, A0, 1)); // 14
        prog.push(ori(A0, A0, 1)); // 15
        prog.push(asm::sw(T5, A0, 0)); // 16: HTIF exit, code = (a0 << 1) | 1

        let mut m = Machine::new(BASE, 0x1_0000);
        m.ram.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut bytes = Vec::new();
        for w in &prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        m.ram.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        let mut cpu = Cpu::new(BASE);
        cpu.htif_tohost = Some(tohost);

        let mut vs = VecSystem::from_template(&cpu, &m);
        let seeds: [u32; LANES] = std::array::from_fn(|lane| lane as u32); // mix of odd/even
        for (lane, &seed) in seeds.iter().enumerate() {
            vs.set_reg(lane, T2, seed);
        }

        let mut guard = 0;
        while vs.any_active() {
            vs.step();
            guard += 1;
            assert!(guard < 10_000, "VecSystem program did not converge to a halt");
        }

        assert!(vs.scalar_steps > 0, "the store must always decline to the per-lane loop");
        assert!(
            vs.total_overlay_pages() > 0,
            "at least one lane must have privately COW'd a page (code or scratch)"
        );

        let mut modified_count = 0;
        let mut unmodified_count = 0;
        for (lane, &seed) in seeds.iter().enumerate() {
            let (mut ocpu, mut om) = (Cpu::new(BASE), Machine::new(BASE, 0x1_0000));
            om.ram.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
            om.ram.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
            ocpu.htif_tohost = Some(tohost);
            ocpu.regs[T2 as usize] = seed;
            let oracle_code = match fs_platform::run_until(&mut ocpu, &mut om, 100_000) {
                fs_platform::Stop::Halt(c) => c,
                other => panic!("scalar oracle (seed {seed}) did not halt: {other:?}"),
            };

            // HTIF exit code is `val >> 1` (see `Cpu::step`'s Store handling); `val = (a0 << 1) | 1`,
            // so the reported code is just `a0` itself (1 = unmodified, 42 = self-modified).
            let expected = if seed & 1 == 1 {
                modified_count += 1;
                42u32
            } else {
                unmodified_count += 1;
                1u32
            };
            assert_eq!(oracle_code, expected, "scalar oracle sanity check for lane {lane} (seed {seed})");

            match vs.exit[lane] {
                Some(LaneExit::Halt(c)) => {
                    assert_eq!(c, expected, "lane {lane} (seed {seed}) HTIF exit code");
                    assert_eq!(c, oracle_code, "lane {lane} vs scalar oracle exit code");
                }
                other => panic!("lane {lane}: expected a Halt exit, got {other:?}"),
            }
        }

        assert!(modified_count > 0 && unmodified_count > 0, "seeds must split across both parities");
    }
}
