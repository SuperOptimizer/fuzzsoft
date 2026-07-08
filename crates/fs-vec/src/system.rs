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

use fs_mmu::{Access, Bus, Golden};
use fs_platform::{CowMachine, Machine};
use fs_riscv::sys::{self, Priv};
use fs_riscv::{decode, decode_compressed, AluOp, Cpu, Inst, SysExit};
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
    /// `step()` calls that took the converged-lane SIMD ALU fast path.
    pub simd_steps: u64,
    /// `step()` calls that fell back to the per-lane scalar `Cpu::step_system` loop (for at least
    /// one active lane — i.e. every `step()` call that did not take the SIMD fast path).
    pub scalar_steps: u64,
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
        let lanes: Box<[Cpu; LANES]> = Box::new(std::array::from_fn(|_| cpu.clone()));
        let ram_base = machine.ram.base();
        let ram_size = machine.ram.size() as u32;
        let golden = Arc::new(Golden::from_mmu(&machine.ram));
        let bus: Box<[CowMachine; LANES]> =
            Box::new(std::array::from_fn(|_| CowMachine::from_golden(Arc::clone(&golden), ram_base, ram_size)));
        Self {
            lanes,
            bus,
            golden,
            active: [true; LANES],
            exit: [None; LANES],
            simd_steps: 0,
            scalar_steps: 0,
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

        if self.try_simd_alu_step() {
            self.simd_steps += 1;
            return;
        }
        self.scalar_steps += 1;
        for lane in 0..LANES {
            if !self.active[lane] {
                continue;
            }
            match self.lanes[lane].step_system(&mut self.bus[lane]) {
                SysExit::Continue => {}
                SysExit::Halt(c) => self.halt(lane, LaneExit::Halt(c)),
                SysExit::Hypercall(c) => self.halt(lane, LaneExit::Hypercall(c)),
            }
        }
    }

    /// The converged-lane SIMD ALU fast path. Returns `true` (having executed the instruction
    /// across every active lane) only when ALL of the following hold, checked *before* anything
    /// is mutated beyond the timer-pending-bit refresh every step already needs (see below):
    ///
    /// - every active lane shares `pc`, `privilege`, and `csr.satp` (so translation — if any —
    ///   resolves the same way for every lane, *pending* the identical-bytes check below);
    /// - `update_timers`'s effect (replicated here byte-for-byte from `fs_riscv::Cpu`'s private
    ///   method, since the fast path bypasses `step_system`) leaves no active lane with a pending,
    ///   enabled interrupt — a trap this step would need full `step_system` trap-vectoring, which
    ///   this fast path does not implement;
    /// - the fetch (translated per-lane, byte-compared across lanes — never assumed identical
    ///   just because `pc`/`satp` agree, since each lane owns independent physical memory) decodes
    ///   to `Inst::OpImm`/`Inst::Op`: the only instruction classes `decode`/`decode_compressed`
    ///   produce that touch no memory, no CSR, and cannot fault on any operand value.
    ///
    /// On any doubt this declines, having mutated nothing except the same per-lane
    /// `mip`/CLINT-derived bits `Cpu::step_system` would unconditionally mutate anyway (see
    /// `update_timers` below) — falling through to the scalar per-lane loop reproduces the exact
    /// same state from there (that loop's `step_system` recomputes the identical, idempotent
    /// `update_timers` result and does its own translation).
    fn try_simd_alu_step(&mut self) -> bool {
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

        // Fetch: translate + read per lane (translation/permission is genuinely per-lane state,
        // even though satp/privilege agree), requiring every active lane's bytes to agree. The
        // low halfword alone carries the full opcode for a 32-bit instruction (bits [6:0]), so a
        // class that can never decode to `OpImm`/`Op` (load/store/branch/jal/jalr/system/amo/
        // fence — the majority of a real kernel's non-ALU instruction mix) is rejected right here,
        // *before* paying for a second per-lane translate+fetch that would only be thrown away.
        let Some(lo0) = self.fetch16_converged(active, pc0) else {
            return false;
        };
        let (inst, ilen) = if lo0 & 0x3 != 0x3 {
            (decode_compressed(lo0), 2u32)
        } else {
            let opcode = lo0 & 0x7f;
            if opcode != 0x13 && opcode != 0x33 {
                return false; // provably not OpImm/Op — decline without fetching the high half
            }
            let Some(hi0) = self.fetch16_converged(active, pc0.wrapping_add(2)) else {
                return false;
            };
            (decode((lo0 as u32) | ((hi0 as u32) << 16)), 4u32)
        };

        let (op, rd, rs1, rs2, imm): (AluOp, u8, u8, Option<u8>, i32) = match inst {
            Inst::OpImm { op, rd, rs1, imm } => (op, rd, rs1, None, imm),
            Inst::Op { op, rd, rs1, rs2 } => (op, rd, rs1, Some(rs2), 0),
            _ => return false, // anything else: scalar path handles it exactly as today
        };

        let mut a_arr = [0u32; LANES];
        let mut b_arr = [0u32; LANES];
        for lane in 0..LANES {
            if !active[lane] {
                continue;
            }
            a_arr[lane] = rd_reg(&self.lanes[lane], rs1);
            b_arr[lane] = match rs2 {
                Some(rs2) => rd_reg(&self.lanes[lane], rs2),
                None => imm as u32,
            };
        }
        let result = simd_alu(op, Simd::from_array(a_arr), Simd::from_array(b_arr)).to_array();

        for lane in 0..LANES {
            if !active[lane] {
                continue;
            }
            if rd != 0 {
                self.lanes[lane].regs[rd as usize] = result[lane];
            }
            self.lanes[lane].pc = self.lanes[lane].pc.wrapping_add(ilen);
            self.lanes[lane].insns_retired += 1;
        }
        true
    }

    /// Translate+fetch one halfword at `va` for every active lane, requiring byte-identical
    /// results across all of them (never assumed from `pc`/`satp` agreement alone — each lane's
    /// physical memory is genuinely independent). `None` on any lane's translation/access fault or
    /// on the slightest byte disagreement.
    fn fetch16_converged(&mut self, active: [bool; LANES], va: u32) -> Option<u16> {
        let mut first: Option<u16> = None;
        for (lane, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            let pa = self.lanes[lane].xlate(&mut self.bus[lane], va, Access::Exec).ok()?;
            let v = self.bus[lane].ifetch16(pa).ok()?;
            match first {
                None => first = Some(v),
                Some(f) if f != v => return None,
                _ => {}
            }
        }
        first
    }
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
}
