//! SoA lane-batched RV32IM(A) executor — the M4 vectorization foundation.
//!
//! Steps `LANES` guest lanes in lockstep, reusing `fs_riscv`'s decoder (`decode` /
//! `decode_compressed`) and its exact RV32M semantics, so there is exactly one validated decoder
//! shared by the scalar interpreter and this executor (decision #8, architecture.md §8).
//!
//! This first cut executes **scalar-over-lanes** — a plain loop over each lane's own state and
//! own [`fs_mmu::Mmu`] (decision #45: correctness-first, safe Rust; no AVX-512 intrinsics yet).
//! What *is* built now is the state layout the real AVX-512 executor will need mechanically:
//!
//! - **SoA register file**: `regs[reg][lane]` — register *i* is a 16-wide `u32` vector, exactly
//!   the shape of a ZMM transpose column (architecture.md §2: "register *i* becomes a 16-wide u32
//!   vector"). Swapping the inner loop body for `std::simd`/AVX-512 intrinsics later does not
//!   require touching this layout.
//! - **Per-lane `pc`** and a **per-lane active mask** (`active`): a lane is masked off once it
//!   faults, halts (`ecall`/`ebreak`), or decodes an instruction this rv32im(a)-only executor
//!   does not implement (CSR/privileged forms — out of scope until `fs-arch` lands). `step` skips
//!   masked-off lanes entirely, so a future k-mask-predicated AVX-512 step drops in over the same
//!   field.
//!
//! On top of that scalar core, [`VecCpu::step`] now also carries a **converged-lane SIMD fast
//! path** (`try_simd_fast_step`, in two flavors: `try_simd_alu`/`try_simd_load`/`try_simd_store`):
//! when every active lane shares the same `pc`, the whole group fetches its instruction **once**
//! from the shared [`VecMmu`] (`VecMmu::ifetch16_same` — no more per-lane fetch, see `vec_mmu.rs`)
//! and decodes it once. ALU-class instructions (OP-IMM / OP: add/sub/and/or/xor/sll/srl/sra/slt/
//! sltu — the ones with a clean packed form, architecture.md §2) execute across all `LANES` lanes
//! with a single `std::simd::Simd<u32, LANES>` operation; `Load`/`Store` whose effective address
//! also agrees across every active lane (the common case) go through `VecMmu`'s same-address
//! fast path in one more shared access. Anything that does not fit that shape (divergent pc,
//! divergent addresses, branches/jumps, DIV/REM/MULH*) falls straight through to the
//! scalar-over-lanes `step_lane` path below, now itself rewired onto the same shared `VecMmu`
//! (`load_lane`/`store_lane`/`ifetch16_lane`) instead of a private per-lane `fs_mmu::Mmu` — see
//! `DESIGN.md` for exactly what's vectorized today and what remains scalar on the road to real
//! AVX-512.
//!
//! See `DESIGN.md` in this crate for the full AVX-512 target (interleaved MMU — now implemented
//! in safe Rust in [`vec_mmu`] — `vmovdqa32` same-address fast path vs `vpgatherdd`/`vpscatterdd`,
//! masked scalarize-16 fallback for DIV/REM/MULH, and where `unsafe` will eventually live).

#![feature(portable_simd)]
#![forbid(unsafe_code)]

mod vec_mmu;

pub use vec_mmu::VecMmu;

use fs_mmu::Fault;
use fs_riscv::{decode, decode_compressed, AluOp, AmoOp, BranchOp, Inst, LoadOp, MulOp, StoreOp};
use std::simd::prelude::*;

/// Lanes per SoA batch. One AVX-512 ZMM register holds 16 packed `u32`s (architecture.md §2) —
/// this is the eventual hardware vector width the AVX-512 executor will target directly.
pub const LANES: usize = 16;

/// Why a lane stopped advancing and was masked off (`active[lane] = false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneExit {
    /// `ecall` — carries `a0`, mirroring the scalar core's `Exit::Ecall` harness convention.
    Ecall { a0: u32 },
    /// `ebreak`.
    Ebreak,
    /// A memory fault (unmapped/permission/unaligned) from the lane's access to the shared
    /// [`VecMmu`].
    Fault(Fault),
    /// Decoded to `Inst::Illegal`, or to a privileged/CSR form this executor does not implement
    /// (CSR access, MRET/SRET/WFI/SFENCE.VMA). Full-system privilege is `fs-arch`'s job (M2+);
    /// masking the lane off here is honest — silently no-opping would corrupt results.
    Illegal { pc: u32, raw: u32 },
}

/// SoA lane-batched RV32IM(A) state.
///
/// `regs[reg][lane]` lays out each architectural register as a `LANES`-wide vector — the
/// mechanical shape of a future AVX-512 transpose (architecture.md §2). `pc`, `active`,
/// `reservation`, `insns_retired`, and `exit` ride alongside per lane.
pub struct VecCpu {
    /// `regs[i][lane]` is register `x{i}` for `lane`. `regs[0][*]` is never read (x0 is
    /// hardwired zero via `rd_reg`/`wr_reg`) but is kept in the array so the layout stays a
    /// uniform `[[u32; LANES]; 32]` — mechanical for a ZMM transpose.
    pub regs: [[u32; LANES]; 32],
    pub pc: [u32; LANES],
    /// Per-lane active mask. A lane is masked off once it faults, halts, or hits an unsupported
    /// decode; `step` skips masked-off lanes. This is the seam a k-mask-predicated AVX-512 `step`
    /// will reuse verbatim (see DESIGN.md).
    pub active: [bool; LANES],
    pub insns_retired: [u64; LANES],
    /// Set exactly when a lane transitions from active to inactive; `None` while still running.
    pub exit: [Option<LaneExit>; LANES],
    /// Per-lane LR/SC reservation (A-extension), independent per hart/lane.
    reservation: [Option<u32>; LANES],
    /// Number of `step()` calls that took the converged-lane SIMD ALU fast path (`try_simd_alu`),
    /// as opposed to falling through to the scalar-over-lanes loop. One `step()` call retiring
    /// all `LANES` lanes' ALU instruction counts once here — this is a diagnostic/benchmark
    /// counter, not part of the executor's correctness contract (see the
    /// `converged_straight_line_alu_program_uses_the_simd_fast_path` test and `examples/bench.rs`).
    pub simd_alu_steps: u64,
    /// Number of `step()` calls that took the converged-lane, same-address SIMD memory fast path
    /// (`try_simd_load`/`try_simd_store`) — a diagnostic/benchmark counter alongside
    /// `simd_alu_steps`, not part of the correctness contract.
    pub simd_mem_steps: u64,
}

impl VecCpu {
    /// All lanes start at the same `pc` with a zeroed register file and active mask — the
    /// "byte-identical lanes" starting condition architecture.md §7 relies on for differential
    /// coverage. Seed per-lane inputs afterwards with [`VecCpu::set_reg`].
    pub fn new(entry: u32) -> Self {
        Self {
            regs: [[0; LANES]; 32],
            pc: [entry; LANES],
            active: [true; LANES],
            insns_retired: [0; LANES],
            exit: [None; LANES],
            reservation: [None; LANES],
            simd_alu_steps: 0,
            simd_mem_steps: 0,
        }
    }

    #[inline]
    pub fn rd_reg(&self, lane: usize, i: u8) -> u32 {
        if i == 0 { 0 } else { self.regs[i as usize][lane] }
    }

    #[inline]
    fn wr_reg(&mut self, lane: usize, i: u8, v: u32) {
        if i != 0 {
            self.regs[i as usize][lane] = v;
        }
    }

    /// Seed one lane's register before the first `step` — e.g. to give lanes different per-lane
    /// inputs while every lane shares the same starting `pc` and program (writes to x0 are
    /// dropped, matching hardware).
    pub fn set_reg(&mut self, lane: usize, i: u8, v: u32) {
        self.wr_reg(lane, i, v);
    }

    /// Whether any lane is still active (i.e. `step` still has work to do).
    pub fn any_active(&self) -> bool {
        self.active.iter().any(|&a| a)
    }

    /// True iff every *active* lane currently shares the same `pc`. This is the AVX-512 fast-path
    /// precondition (architecture.md §2): converged lanes can fetch/decode as a single vector
    /// instruction; on divergence the hybrid executor must scalarize the disagreeing lanes.
    /// Informational only in this scalar-over-lanes cut — `step` makes progress regardless.
    pub fn lanes_converged(&self) -> bool {
        let mut pcs = self
            .active
            .iter()
            .zip(self.pc.iter())
            .filter(|&(&active, _)| active)
            .map(|(_, &pc)| pc);
        match pcs.next() {
            Some(first) => pcs.all(|pc| pc == first),
            None => true,
        }
    }

    /// Advance every active lane by exactly one instruction, all against the ONE shared
    /// [`VecMmu`] (DESIGN.md §3's interleaved layout — no more per-lane `Mmu`).
    ///
    /// Tries the converged-lane SIMD fast path first ([`VecCpu::try_simd_fast_step`]); if that
    /// declines (returns `false`, having mutated nothing), falls back to the scalar-over-lanes
    /// loop (decision #45, correctness-first): each lane independently fetches/decodes/executes
    /// via `fs_riscv::decode`/`decode_compressed` and the M-extension semantics copied below, but
    /// the SoA layout + active mask keep a future AVX-512 lockstep step a mechanical drop-in.
    pub fn step(&mut self, mmu: &mut VecMmu) {
        if self.try_simd_fast_step(mmu) {
            return;
        }
        for lane in 0..LANES {
            if self.active[lane] {
                self.step_lane(lane, mmu);
            }
        }
    }

    /// Converged-lane SIMD fast path dispatcher (architecture.md §2, DESIGN.md "Lockstep
    /// fetch/decode" + "Memory: same-address fast path").
    ///
    /// Preconditions, all checked before anything is mutated:
    /// - every active lane shares the same `pc` ([`VecCpu::lanes_converged`]);
    /// - the whole group fetches its instruction **once** from the shared [`VecMmu`]
    ///   ([`fetch_converged`]) — one shared read instead of `LANES` per-lane ones, and every
    ///   active lane's line must agree byte-for-byte (checked, not assumed: lanes whose memory
    ///   has diverged decline here and fall back to `step_lane`'s per-lane fetch).
    ///
    /// Given a single decode, dispatches to the payload that matches the instruction class:
    /// - `Inst::OpImm`/`Inst::Op` -> [`VecCpu::try_simd_alu`] (packed ALU, `Simd<u32, LANES>`).
    /// - `Inst::Load`/`Inst::Store` whose effective address also agrees across every active lane
    ///   -> [`VecCpu::try_simd_load`]/[`VecCpu::try_simd_store`] (`VecMmu`'s same-address fast
    ///   path, one shared access instead of `LANES`).
    /// - anything else (branches/jumps, MUL/DIV/REM, divergent addresses, system) -> `false`,
    ///   `step`'s scalar-over-lanes loop handles it exactly as before.
    ///
    /// Re-fetching in the fallback is free of side effects (fetch/load are pure reads until a
    /// same-address store commits), so speculatively decoding here first is always safe to
    /// discard.
    fn try_simd_fast_step(&mut self, mmu: &mut VecMmu) -> bool {
        let active = self.active;
        if !self.lanes_converged() {
            return false;
        }
        let Some(first) = active.iter().position(|&a| a) else {
            return false; // no active lanes at all
        };
        let pc = self.pc[first];

        let Some((iword, ilen)) = fetch_converged(mmu, pc, active) else {
            return false;
        };
        let inst = if ilen == 2 { decode_compressed(iword as u16) } else { decode(iword) };

        match inst {
            Inst::OpImm { .. } | Inst::Op { .. } => self.try_simd_alu(inst, ilen, active),
            Inst::Load { .. } => self.try_simd_load(inst, ilen, active, mmu),
            Inst::Store { .. } => self.try_simd_store(inst, ilen, active, mmu),
            _ => false, // branch / jump / mul-div / system: scalar path handles it
        }
    }

    /// Advance every active lane's `pc` by `ilen`, masked so inactive lanes are untouched — the
    /// pc-update tail shared by every converged fast-path payload below.
    fn advance_pc_converged(&mut self, active_mask: Mask<i32, LANES>, ilen: u32) {
        let pc_vec: Simd<u32, LANES> = Simd::from_array(self.pc);
        self.pc = active_mask.select(pc_vec + Simd::splat(ilen), pc_vec).to_array();
    }

    /// Bump `insns_retired` for every lane that was active this step — shared by every converged
    /// fast-path payload below.
    fn retire_active(&mut self, active: [bool; LANES]) {
        for (lane, &was_active) in active.iter().enumerate() {
            if was_active {
                self.insns_retired[lane] += 1;
            }
        }
    }

    /// Packed ALU payload: `inst` must be `Inst::OpImm`/`Inst::Op` (checked by the caller,
    /// `try_simd_fast_step`'s match arm). Executes across all `LANES` lanes with one masked
    /// `Simd<u32, LANES>` operation ([`simd_alu`]) instead of `LANES` scalar `alu` calls — the
    /// original throughput win `examples/bench.rs` measures.
    fn try_simd_alu(&mut self, inst: Inst, ilen: u32, active: [bool; LANES]) -> bool {
        let (op, rd, rs1, b): (AluOp, u8, u8, Simd<u32, LANES>) = match inst {
            Inst::OpImm { op, rd, rs1, imm } => (op, rd, rs1, Simd::splat(imm as u32)),
            Inst::Op { op, rd, rs1, rs2 } => {
                (op, rd, rs1, Simd::from_array(self.regs[rs2 as usize]))
            }
            _ => return false,
        };

        let active_mask: Mask<i32, LANES> = Mask::from_array(active);
        let a: Simd<u32, LANES> = Simd::from_array(self.regs[rs1 as usize]);
        let result = simd_alu(op, a, b);

        // x0 is hardwired zero (never written, mirroring `wr_reg`); every other destination lane
        // keeps its previous value where the active mask is clear.
        if rd != 0 {
            let prev: Simd<u32, LANES> = Simd::from_array(self.regs[rd as usize]);
            self.regs[rd as usize] = active_mask.select(result, prev).to_array();
        }

        // No branch/jump/load/store in this instruction class, so every active lane's `pc`
        // advances by the same `ilen` — still a masked vector op, not a per-lane scalar add.
        self.advance_pc_converged(active_mask, ilen);
        self.retire_active(active);
        self.simd_alu_steps += 1;

        true
    }

    /// Same-address vectorized load payload (DESIGN.md "Memory: same-address fast path"):
    /// `inst` must be `Inst::Load` (checked by the caller). Computes every active lane's
    /// effective address as one `Simd<u32, LANES>` add; if every active lane's address agrees,
    /// issues one `VecMmu::load_same` for the whole group instead of `LANES` separate loads.
    /// Declines (returns `false`, mutating nothing) on divergent addresses or if `VecMmu` itself
    /// declines (misalignment/OOB/permission) — `step`'s scalar-over-lanes loop then services the
    /// instruction lane-by-lane via `VecMmu::load_lane` (the divergent-address path's scalar
    /// body).
    fn try_simd_load(&mut self, inst: Inst, ilen: u32, active: [bool; LANES], mmu: &VecMmu) -> bool {
        let Inst::Load { op, rd, rs1, imm } = inst else {
            return false;
        };
        let Some(first) = active.iter().position(|&a| a) else {
            return false;
        };
        let rs1v: Simd<u32, LANES> = Simd::from_array(self.regs[rs1 as usize]);
        let addrs = rs1v + Simd::splat(imm as u32);
        let addr0 = addrs.to_array()[first];
        let active_mask: Mask<i32, LANES> = Mask::from_array(active);
        if (active_mask & !addrs.simd_eq(Simd::splat(addr0))).any() {
            return false; // divergent effective addresses: scalar-over-lanes handles it lane-by-lane
        }

        let (size, signed) = match op {
            LoadOp::Lb => (1u8, true),
            LoadOp::Lbu => (1u8, false),
            LoadOp::Lh => (2u8, true),
            LoadOp::Lhu => (2u8, false),
            LoadOp::Lw => (4u8, false),
        };
        let Some(raw) = mmu.load_same(addr0, size, active) else {
            return false;
        };
        let value = if signed {
            match size {
                1 => sign_extend_simd(raw, 8),
                2 => sign_extend_simd(raw, 16),
                _ => raw,
            }
        } else {
            raw
        };

        if rd != 0 {
            let prev: Simd<u32, LANES> = Simd::from_array(self.regs[rd as usize]);
            self.regs[rd as usize] = active_mask.select(value, prev).to_array();
        }
        self.advance_pc_converged(active_mask, ilen);
        self.retire_active(active);
        self.simd_mem_steps += 1;
        true
    }

    /// Same-address vectorized store payload, the twin of [`VecCpu::try_simd_load`]: `inst` must
    /// be `Inst::Store`. Declines under the same conditions (divergent effective address, or
    /// `VecMmu::store_same` itself declining).
    fn try_simd_store(
        &mut self,
        inst: Inst,
        ilen: u32,
        active: [bool; LANES],
        mmu: &mut VecMmu,
    ) -> bool {
        let Inst::Store { op, rs1, rs2, imm } = inst else {
            return false;
        };
        let Some(first) = active.iter().position(|&a| a) else {
            return false;
        };
        let rs1v: Simd<u32, LANES> = Simd::from_array(self.regs[rs1 as usize]);
        let addrs = rs1v + Simd::splat(imm as u32);
        let addr0 = addrs.to_array()[first];
        let active_mask: Mask<i32, LANES> = Mask::from_array(active);
        if (active_mask & !addrs.simd_eq(Simd::splat(addr0))).any() {
            return false; // divergent effective addresses: scalar-over-lanes handles it lane-by-lane
        }

        let size = match op {
            StoreOp::Sb => 1u8,
            StoreOp::Sh => 2u8,
            StoreOp::Sw => 4u8,
        };
        let vals: Simd<u32, LANES> = Simd::from_array(self.regs[rs2 as usize]);
        if !mmu.store_same(addr0, size, active, vals) {
            return false;
        }
        self.advance_pc_converged(active_mask, ilen);
        self.retire_active(active);
        self.simd_mem_steps += 1;
        true
    }

    fn halt(&mut self, lane: usize, exit: LaneExit) {
        self.active[lane] = false;
        self.exit[lane] = Some(exit);
    }

    fn step_lane(&mut self, lane: usize, mmu: &mut VecMmu) {
        let pc = self.pc[lane];

        // Variable-length fetch, identical to the scalar core: a half-word whose low 2 bits
        // != 0b11 is a 16-bit compressed instruction (IALIGN=16); otherwise it is 32-bit. Reads
        // this lane's own slot of the shared `VecMmu` (`ifetch16_lane` — the divergent-address
        // path's scalar body, DESIGN.md §3), since a lane running `step_lane` may have diverged
        // in `pc` from the rest of the group.
        let lo = match mmu.ifetch16_lane(lane, pc) {
            Ok(v) => v,
            Err(f) => return self.halt(lane, LaneExit::Fault(f)),
        };
        let (inst, ilen, iword) = if lo & 0x3 != 0x3 {
            (decode_compressed(lo), 2u32, lo as u32)
        } else {
            let hi = match mmu.ifetch16_lane(lane, pc.wrapping_add(2)) {
                Ok(v) => v,
                Err(f) => return self.halt(lane, LaneExit::Fault(f)),
            };
            let w = (lo as u32) | ((hi as u32) << 16);
            (decode(w), 4u32, w)
        };
        let mut next = pc.wrapping_add(ilen);

        macro_rules! try_mem {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(f) => return self.halt(lane, LaneExit::Fault(f)),
                }
            };
        }

        match inst {
            Inst::Lui { rd, imm } => self.wr_reg(lane, rd, imm),
            Inst::Auipc { rd, imm } => self.wr_reg(lane, rd, pc.wrapping_add(imm)),
            Inst::Jal { rd, imm } => {
                self.wr_reg(lane, rd, next);
                next = pc.wrapping_add(imm as u32);
            }
            Inst::Jalr { rd, rs1, imm } => {
                let target = self.rd_reg(lane, rs1).wrapping_add(imm as u32) & !1;
                self.wr_reg(lane, rd, next);
                next = target;
            }
            Inst::Branch { op, rs1, rs2, imm } => {
                let a = self.rd_reg(lane, rs1);
                let b = self.rd_reg(lane, rs2);
                let taken = match op {
                    BranchOp::Eq => a == b,
                    BranchOp::Ne => a != b,
                    BranchOp::Lt => (a as i32) < (b as i32),
                    BranchOp::Ge => (a as i32) >= (b as i32),
                    BranchOp::Ltu => a < b,
                    BranchOp::Geu => a >= b,
                };
                if taken {
                    next = pc.wrapping_add(imm as u32);
                }
            }
            Inst::Load { op, rd, rs1, imm } => {
                let addr = self.rd_reg(lane, rs1).wrapping_add(imm as u32);
                let (size, signed) = match op {
                    LoadOp::Lb => (1, true),
                    LoadOp::Lbu => (1, false),
                    LoadOp::Lh => (2, true),
                    LoadOp::Lhu => (2, false),
                    LoadOp::Lw => (4, false),
                };
                let raw = try_mem!(mmu.load_lane(lane, addr, size));
                let v = if signed {
                    match size {
                        1 => raw as u8 as i8 as i32 as u32,
                        2 => raw as u16 as i16 as i32 as u32,
                        _ => raw,
                    }
                } else {
                    raw
                };
                self.wr_reg(lane, rd, v);
            }
            Inst::Store { op, rs1, rs2, imm } => {
                let addr = self.rd_reg(lane, rs1).wrapping_add(imm as u32);
                let val = self.rd_reg(lane, rs2);
                let size = match op {
                    StoreOp::Sb => 1,
                    StoreOp::Sh => 2,
                    StoreOp::Sw => 4,
                };
                try_mem!(mmu.store_lane(lane, addr, size, val));
            }
            Inst::OpImm { op, rd, rs1, imm } => {
                let v = alu(op, self.rd_reg(lane, rs1), imm as u32);
                self.wr_reg(lane, rd, v);
            }
            Inst::Op { op, rd, rs1, rs2 } => {
                let v = alu(op, self.rd_reg(lane, rs1), self.rd_reg(lane, rs2));
                self.wr_reg(lane, rd, v);
            }
            Inst::Mul { op, rd, rs1, rs2 } => {
                let v = muldiv(op, self.rd_reg(lane, rs1), self.rd_reg(lane, rs2));
                self.wr_reg(lane, rd, v);
            }
            Inst::LrW { rd, rs1, .. } => {
                let addr = self.rd_reg(lane, rs1);
                let v = try_mem!(mmu.load_lane(lane, addr, 4));
                self.reservation[lane] = Some(addr);
                self.wr_reg(lane, rd, v);
            }
            Inst::ScW { rd, rs1, rs2, .. } => {
                let addr = self.rd_reg(lane, rs1);
                let success = self.reservation[lane] == Some(addr);
                if success {
                    let v = self.rd_reg(lane, rs2);
                    try_mem!(mmu.store_lane(lane, addr, 4, v));
                }
                // A reservation is single-use, and any trap/context-switch would clear it too.
                self.reservation[lane] = None;
                self.wr_reg(lane, rd, if success { 0 } else { 1 });
            }
            Inst::AmoW { op, rd, rs1, rs2, .. } => {
                let addr = self.rd_reg(lane, rs1);
                let old = try_mem!(mmu.load_lane(lane, addr, 4));
                let src = self.rd_reg(lane, rs2);
                let result = match op {
                    AmoOp::Swap => src,
                    AmoOp::Add => old.wrapping_add(src),
                    AmoOp::Xor => old ^ src,
                    AmoOp::And => old & src,
                    AmoOp::Or => old | src,
                    AmoOp::Min => (old as i32).min(src as i32) as u32,
                    AmoOp::Max => (old as i32).max(src as i32) as u32,
                    AmoOp::Minu => old.min(src),
                    AmoOp::Maxu => old.max(src),
                };
                try_mem!(mmu.store_lane(lane, addr, 4, result));
                self.reservation[lane] = None;
                self.wr_reg(lane, rd, old);
            }
            Inst::Fence => {}
            Inst::Ecall => {
                let a0 = self.rd_reg(lane, 10);
                return self.halt(lane, LaneExit::Ecall { a0 });
            }
            Inst::Ebreak => return self.halt(lane, LaneExit::Ebreak),
            // CSR/privileged forms: no CSR file or privilege modes in this rv32im(a)-only
            // executor yet (that surface is fs-arch's, M2+). Mask off rather than mis-execute.
            Inst::Csr { .. } | Inst::Mret | Inst::Sret | Inst::Wfi | Inst::SfenceVma => {
                return self.halt(lane, LaneExit::Illegal { pc, raw: iword });
            }
            Inst::Illegal(raw) => return self.halt(lane, LaneExit::Illegal { pc, raw }),
        }

        self.pc[lane] = next;
        self.insns_retired[lane] += 1;
    }
}

/// Vectorized instruction fetch for a converged group (DESIGN.md "Lockstep fetch/decode" +
/// "Vectorized interleaving"): every `active` lane already shares `pc` (checked by the caller,
/// `VecCpu::lanes_converged`), so this issues ONE [`VecMmu::ifetch16_same`] (or two, for a 32-bit
/// instruction) against the shared interleaved store instead of `LANES` independent per-lane
/// fetches — the fetch deduplication that removes the original SIMD fast path's ~2x throughput
/// ceiling (DESIGN.md).
///
/// Returns `None`, having read nothing lanes disagree on, if `VecMmu` declines (misalignment/
/// out-of-bounds/any active lane missing `PERM_EXEC`) or if the active lanes' fetched halfword(s)
/// are not byte-identical (checked, not assumed: a lane whose memory has diverged — e.g.
/// self-modifying code taking different edits per lane — must not be silently treated as
/// converged). Either way the caller falls back to `step_lane`'s per-lane fetch
/// (`VecMmu::ifetch16_lane`), which produces the correct per-lane result/fault even when lanes'
/// own memory has diverged.
fn fetch_converged(mmu: &VecMmu, pc: u32, active: [bool; LANES]) -> Option<(u32, u32)> {
    let active_mask: Mask<i32, LANES> = Mask::from_array(active);
    let first = active.iter().position(|&a| a)?;

    let lo_line = mmu.ifetch16_same(pc, active)?;
    let lo0 = lo_line.to_array()[first];
    if (active_mask & !lo_line.simd_eq(Simd::splat(lo0))).any() {
        return None; // active lanes disagree on the fetched halfword
    }
    if lo0 & 0x3 != 0x3 {
        return Some((lo0, 2));
    }

    let hi_line = mmu.ifetch16_same(pc.wrapping_add(2), active)?;
    let hi0 = hi_line.to_array()[first];
    if (active_mask & !hi_line.simd_eq(Simd::splat(hi0))).any() {
        return None;
    }
    Some((lo0 | (hi0 << 16), 4))
}

/// Sign-extend the low `bits` bits of every lane of `v` to a full 32-bit signed value packed back
/// into `Simd<u32, LANES>` — the packed twin of the scalar `as u8 as i8 as i32 as u32` / `as u16
/// as i16 as i32 as u32` casts `step_lane`'s `Inst::Load` arm uses for `LB`/`LH`, via the
/// arithmetic-shift-left-then-right idiom (`(v << (32-bits)) >> (32-bits)` in signed lanes).
#[inline]
fn sign_extend_simd(v: Simd<u32, LANES>, bits: u32) -> Simd<u32, LANES> {
    let shift: Simd<i32, LANES> = Simd::splat((32 - bits) as i32);
    let vi: Simd<i32, LANES> = v.cast();
    ((vi << shift) >> shift).cast()
}

/// Copied verbatim from `fs_riscv`'s private `alu` (single source of truth is the scalar core;
/// this executor mirrors it exactly rather than depending on a private fn — see DESIGN.md).
#[inline]
fn alu(op: AluOp, a: u32, b: u32) -> u32 {
    match op {
        AluOp::Add => a.wrapping_add(b),
        AluOp::Sub => a.wrapping_sub(b),
        AluOp::Sll => a.wrapping_shl(b & 31),
        AluOp::Slt => ((a as i32) < (b as i32)) as u32,
        AluOp::Sltu => (a < b) as u32,
        AluOp::Xor => a ^ b,
        AluOp::Srl => a.wrapping_shr(b & 31),
        AluOp::Sra => ((a as i32).wrapping_shr(b & 31)) as u32,
        AluOp::Or => a | b,
        AluOp::And => a & b,
    }
}

/// The `Simd<u32, LANES>`-packed twin of [`alu`] above: identical op semantics (wrapping
/// arithmetic, `& 31` shift-amount masking, signed vs. unsigned compares), all `LANES` lanes at
/// once instead of one lane at a time. This is the SIMD fast path's payload
/// (`VecCpu::try_simd_alu`); the `simd_alu_fast_path_matches_scalar_alu_exhaustively` test below
/// is what actually enforces that this and `alu` never disagree.
///
/// `std::simd`'s wrapping-integer arithmetic matches the plain scalar `u32` semantics required
/// here (add/sub wrap silently, no overflow trap), so this needs no `wrapping_*` equivalents of
/// its own — only the shift amount needs the explicit `& 31` mask that hardware shifts impose.
#[inline]
fn simd_alu(op: AluOp, a: Simd<u32, LANES>, b: Simd<u32, LANES>) -> Simd<u32, LANES> {
    let shamt = b & Simd::splat(31u32);
    match op {
        AluOp::Add => a + b,
        AluOp::Sub => a - b,
        AluOp::Sll => a << shamt,
        AluOp::Slt => {
            let ai: Simd<i32, LANES> = a.cast();
            let bi: Simd<i32, LANES> = b.cast();
            ai.simd_lt(bi).select(Simd::splat(1), Simd::splat(0))
        }
        AluOp::Sltu => a.simd_lt(b).select(Simd::splat(1), Simd::splat(0)),
        AluOp::Xor => a ^ b,
        AluOp::Srl => a >> shamt,
        AluOp::Sra => {
            let ai: Simd<i32, LANES> = a.cast();
            let shamt_i: Simd<i32, LANES> = shamt.cast();
            (ai >> shamt_i).cast()
        }
        AluOp::Or => a | b,
        AluOp::And => a & b,
    }
}

/// Copied verbatim from `fs_riscv`'s private `muldiv`, including the exact M-extension edge
/// cases (architecture.md §5): DIV/0 -> `0xffff_ffff`, REM/0 -> dividend, signed `INT_MIN/-1`
/// overflow -> DIV=`0x8000_0000`/REM=0, MULH*/via 64-bit widening. These do not have a clean
/// packed SIMD form and are the instructions the eventual AVX-512 executor must scalarize
/// (masked scalarize-16 fallback — see DESIGN.md); keeping this a byte-for-byte copy of the
/// golden model is what makes that fallback exact.
#[inline]
fn muldiv(op: MulOp, a: u32, b: u32) -> u32 {
    match op {
        MulOp::Mul => a.wrapping_mul(b),
        MulOp::Mulh => (((a as i32 as i64) * (b as i32 as i64)) >> 32) as u32,
        MulOp::Mulhsu => (((a as i32 as i64) * (b as i64)) >> 32) as u32,
        MulOp::Mulhu => (((a as u64) * (b as u64)) >> 32) as u32,
        MulOp::Div => {
            if b == 0 {
                0xffff_ffff
            } else if a == 0x8000_0000 && b == 0xffff_ffff {
                0x8000_0000
            } else {
                ((a as i32).wrapping_div(b as i32)) as u32
            }
        }
        MulOp::Divu => a.checked_div(b).unwrap_or(0xffff_ffff),
        MulOp::Rem => {
            if b == 0 {
                a
            } else if a == 0x8000_0000 && b == 0xffff_ffff {
                0
            } else {
                ((a as i32).wrapping_rem(b as i32)) as u32
            }
        }
        MulOp::Remu => a.checked_rem(b).unwrap_or(a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::{Mmu, PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_riscv::{asm, A0, A7, T0, T1, X0};

    const BASE: u32 = 0x8000_0000;

    /// Lay `prog` out in a fresh shared [`VecMmu`] — `map`/`protect` broadcast identically to
    /// every lane (DESIGN.md "byte-identical starting snapshot"), which is exactly what these
    /// tests want: all `LANES` lanes start byte-identical, diverging only via per-lane register
    /// seeds/execution, never via different underlying memory content.
    fn make_mmu(prog: &[u32]) -> VecMmu {
        let mut mmu = VecMmu::new(BASE, 0x1_0000);
        mmu.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mmu.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        mmu
    }

    /// The scalar golden model's own `fs_mmu::Mmu` — unrelated to `VecMmu`, used only to run
    /// `fs_riscv::Cpu` (the oracle these tests cross-check `VecCpu` against) independently.
    fn make_scalar_mmu(prog: &[u32]) -> Mmu {
        let mut mmu = Mmu::new(BASE, 0x1_0000);
        mmu.protect(BASE, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        let mut bytes = Vec::new();
        for w in prog {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mmu.map(BASE, &bytes, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();
        mmu
    }

    /// Run the scalar golden model (`fs_riscv::Cpu`) to its `ecall` exit, optionally seeding one
    /// register first, and return the final CPU.
    fn run_scalar(prog: &[u32], seed: Option<(u8, u32)>) -> fs_riscv::Cpu {
        let mut mmu = make_scalar_mmu(prog);
        let mut cpu = fs_riscv::Cpu::new(BASE);
        if let Some((reg, val)) = seed {
            cpu.regs[reg as usize] = val;
        }
        for _ in 0..10_000 {
            match cpu.step(&mut mmu).unwrap() {
                fs_riscv::Exit::Continue => {}
                fs_riscv::Exit::Ecall => return cpu,
                other => panic!("unexpected scalar exit: {other:?}"),
            }
        }
        panic!("scalar program did not terminate");
    }

    // a0 = sum(1..=10) = 55, then ecall exit — same hand-encoded loop as fs_riscv's own test,
    // reused here as the identical-lanes validation program.
    fn sum_1_to_10_program() -> [u32; 9] {
        use asm::*;
        [
            addi(A0, X0, 0),  // 0: sum = 0
            addi(T0, X0, 1),  // 1: i = 1
            addi(T1, X0, 11), // 2: limit = 11
            bge(T0, T1, 16),  // 3: if i >= 11 -> done (idx 7)
            add(A0, A0, T0),  // 4: sum += i
            addi(T0, T0, 1),  // 5: i++
            jal(X0, -12),     // 6: -> idx 3
            addi(A7, X0, 93), // 7: done: a7 = exit
            ecall(),          // 8
        ]
    }

    /// Requirement (a)+(b): all `LANES` lanes run the SAME program from IDENTICAL starting state
    /// (decision-#3/#45 lockstep contract). Every lane's final registers must match every other
    /// lane's, and must match a single scalar `fs_riscv::Cpu` run of the same program.
    #[test]
    fn identical_lanes_match_each_other_and_the_scalar_golden_model() {
        let prog = sum_1_to_10_program();
        let mut mmu = make_mmu(&prog);
        let mut vcpu = VecCpu::new(BASE);

        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut mmu);
            guard += 1;
            assert!(guard < 10_000, "vectorized program did not converge");
        }

        for lane in 0..LANES {
            assert_eq!(vcpu.exit[lane], Some(LaneExit::Ecall { a0: 55 }), "lane {lane}");
        }

        // (a) every lane agrees with every other lane, register-for-register.
        for reg in 0..32 {
            for lane in 1..LANES {
                assert_eq!(
                    vcpu.regs[reg][0], vcpu.regs[reg][lane],
                    "register x{reg} disagreed between lane 0 and lane {lane}"
                );
            }
        }

        // (b) and the whole bank agrees with the scalar golden model.
        let scalar = run_scalar(&prog, None);
        for reg in 0..32 {
            assert_eq!(vcpu.regs[reg][0], scalar.regs[reg], "register x{reg} vs scalar");
        }
    }

    /// Requirement 4's second half: DIFFERENT per-lane inputs must make lanes genuinely diverge
    /// (different `pc` trajectories, different retirement counts, different final results) while
    /// the active mask correctly tracks each lane finishing independently, and every lane's
    /// individual result still matches an independent scalar run seeded with that lane's input.
    #[test]
    fn lanes_diverge_on_different_inputs_and_the_active_mask_tracks_it() {
        use asm::*;
        // t2 (x7) is the per-lane seed N; a0 = sum(1..=N).
        const T2: u8 = 7;
        let prog: [u32; 9] = [
            addi(A0, X0, 0),  // 0: sum = 0
            addi(T0, X0, 1),  // 1: i = 1
            addi(T1, T2, 1),  // 2: limit = N + 1
            bge(T0, T1, 16),  // 3: if i >= limit -> done (idx 7)
            add(A0, A0, T0),  // 4: sum += i
            addi(T0, T0, 1),  // 5: i++
            jal(X0, -12),     // 6: -> idx 3
            addi(A7, X0, 93), // 7: done
            ecall(),          // 8
        ];

        let mut mmu = make_mmu(&prog);
        let mut vcpu = VecCpu::new(BASE);
        let ns: [u32; LANES] = std::array::from_fn(|lane| lane as u32); // N = 0, 1, .., 15
        for (lane, &n) in ns.iter().enumerate() {
            vcpu.set_reg(lane, T2, n);
        }

        let mut saw_pc_divergence = false;
        let mut saw_partial_completion = false;
        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut mmu);
            if !vcpu.lanes_converged() {
                saw_pc_divergence = true;
            }
            let active_count = vcpu.active.iter().filter(|&&a| a).count();
            if active_count > 0 && active_count < LANES {
                saw_partial_completion = true;
            }
            guard += 1;
            assert!(guard < 10_000, "vectorized program did not converge");
        }

        assert!(
            saw_pc_divergence,
            "lanes running the same program with different inputs must take different branches \
             (differing pc) at some point"
        );
        assert!(
            saw_partial_completion,
            "the lane seeded N=0 finishes in far fewer steps than N=15; the active mask should \
             show some lanes done while others are still running"
        );

        for (lane, &n) in ns.iter().enumerate() {
            let expected = n * (n + 1) / 2;
            assert_eq!(vcpu.exit[lane], Some(LaneExit::Ecall { a0: expected }), "lane {lane}");
            assert_eq!(vcpu.regs[A0 as usize][lane], expected, "lane {lane} final a0");

            // Cross-check this one lane against an independently-seeded scalar run.
            let scalar = run_scalar(&prog, Some((T2, n)));
            assert_eq!(scalar.regs[A0 as usize], expected, "lane {lane} scalar oracle");
            assert_eq!(
                vcpu.regs[A0 as usize][lane], scalar.regs[A0 as usize],
                "lane {lane} vs its scalar oracle"
            );
        }

        // Lanes must NOT all agree with each other (that would mean the different inputs were
        // silently ignored).
        let distinct: std::collections::HashSet<u32> =
            (0..LANES).map(|lane| vcpu.regs[A0 as usize][lane]).collect();
        assert_eq!(distinct.len(), LANES, "every lane should have a distinct sum");
    }

    /// A tiny xorshift PRNG (no external `rand` dependency, decision #26-style hermeticism) used
    /// to fuzz [`simd_alu`] against the scalar [`alu`] it must never disagree with.
    struct XorShift64(u64);
    impl XorShift64 {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as u32
        }
    }

    /// [`simd_alu`] is the SIMD fast path's payload; this is what actually proves it never
    /// disagrees with the scalar [`alu`] it mirrors, across every `AluOp` and a wide spread of
    /// `a`/`b` values (including the shift-amount-masking and signed-compare edge cases).
    #[test]
    fn simd_alu_fast_path_matches_scalar_alu_exhaustively() {
        let ops = [
            AluOp::Add,
            AluOp::Sub,
            AluOp::Sll,
            AluOp::Slt,
            AluOp::Sltu,
            AluOp::Xor,
            AluOp::Srl,
            AluOp::Sra,
            AluOp::Or,
            AluOp::And,
        ];
        let mut rng = XorShift64(0x243f_6a88_85a3_08d3);
        for op in ops {
            for _ in 0..2_000 {
                let a_arr: [u32; LANES] = std::array::from_fn(|_| rng.next_u32());
                let b_arr: [u32; LANES] = std::array::from_fn(|_| rng.next_u32());
                let result = simd_alu(op, Simd::from_array(a_arr), Simd::from_array(b_arr)).to_array();
                for lane in 0..LANES {
                    assert_eq!(
                        result[lane],
                        alu(op, a_arr[lane], b_arr[lane]),
                        "op {op:?} lane {lane} a={:#x} b={:#x}",
                        a_arr[lane],
                        b_arr[lane]
                    );
                }
            }
        }
    }

    // --- Minimal R-type/I-type encoders for the ALU ops `fs_riscv::asm` doesn't expose helpers
    // for (its `i_type`/`r_type` are private). Mirrors `decode`'s funct3/funct7 tables exactly
    // (fs-riscv/src/lib.rs) — used only to hand-assemble the SIMD-fast-path test program below.
    fn i_type(op: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
        ((imm as u32 & 0xfff) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | op
    }
    fn r_type(op: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
        (funct7 << 25)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | (funct3 << 12)
            | ((rd as u32) << 7)
            | op
    }
    fn andi(rd: u8, rs1: u8, imm: i32) -> u32 {
        i_type(0x13, 7, rd, rs1, imm)
    }
    fn xori(rd: u8, rs1: u8, imm: i32) -> u32 {
        i_type(0x13, 4, rd, rs1, imm)
    }
    fn slli(rd: u8, rs1: u8, shamt: u8) -> u32 {
        r_type(0x13, 1, 0x00, rd, rs1, shamt)
    }
    fn srli(rd: u8, rs1: u8, shamt: u8) -> u32 {
        r_type(0x13, 5, 0x00, rd, rs1, shamt)
    }
    fn and_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 7, 0x00, rd, rs1, rs2)
    }
    fn or_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 6, 0x00, rd, rs1, rs2)
    }
    fn xor_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 4, 0x00, rd, rs1, rs2)
    }
    fn sll_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 1, 0x00, rd, rs1, rs2)
    }
    fn srl_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 5, 0x00, rd, rs1, rs2)
    }
    fn sra_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 5, 0x20, rd, rs1, rs2)
    }
    fn slt_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 2, 0x00, rd, rs1, rs2)
    }
    fn sltu_(rd: u8, rs1: u8, rs2: u8) -> u32 {
        r_type(0x33, 3, 0x00, rd, rs1, rs2)
    }

    /// A converged, branch-free, all-ALU program: every lane stays at the same `pc` for the
    /// program's entire run (no branches to diverge on), so `step` should take the SIMD fast path
    /// for every one of its `OP-IMM`/`OP` instructions. `T2` (x7) is the per-lane seed.
    fn straight_line_alu_program() -> Vec<u32> {
        use asm::*;
        const T2: u8 = 7;
        vec![
            addi(T0, T2, 5),    // t0 = seed + 5
            addi(T1, T2, -3),   // t1 = seed - 3
            and_(T0, T0, T1),   // t0 &= t1
            or_(T0, T0, T2),    // t0 |= seed
            xor_(T0, T0, T1),   // t0 ^= t1
            slli(T0, T0, 2),    // t0 <<= 2
            srli(T1, T1, 1),    // t1 >>= 1 (logical)
            add(T0, T0, T1),    // t0 += t1
            sub(T0, T0, T2),    // t0 -= seed
            sll_(T0, T0, T2),   // t0 <<= (seed & 31)
            srl_(T0, T0, T2),   // t0 >>= (seed & 31) (logical)
            sra_(T0, T0, T2),   // t0 >>= (seed & 31) (arithmetic)
            slt_(T1, T0, T2),   // t1 = (t0 < seed) signed
            sltu_(T1, T0, T2),  // t1 = (t0 < seed) unsigned (overwrites the signed result above)
            andi(T0, T0, 0xff), // t0 &= 0xff
            xori(T0, T0, 0x2a), // t0 ^= 0x2a
            add(A0, T0, T1),    // a0 = t0 + t1 (final result)
            addi(A7, X0, 93),   // a7 = exit
            ecall(),
        ]
    }

    /// Requirement 3: a converged straight-line ALU program — no branches, so lanes never
    /// diverge in `pc` — must take `VecCpu::try_simd_alu`'s SIMD fast path for every ALU
    /// instruction, and still land on exactly the same per-lane results as an independent scalar
    /// `fs_riscv::Cpu` run seeded with that lane's input.
    #[test]
    fn converged_straight_line_alu_program_uses_the_simd_fast_path() {
        const T2: u8 = 7;
        let prog = straight_line_alu_program();
        let mut mmu = make_mmu(&prog);
        let mut vcpu = VecCpu::new(BASE);
        let seeds: [u32; LANES] = std::array::from_fn(|lane| (lane as u32) * 7 + 1);
        for (lane, &seed) in seeds.iter().enumerate() {
            vcpu.set_reg(lane, T2, seed);
        }

        let alu_insn_count = (prog.len() - 1) as u64; // every instruction except the trailing ecall

        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut mmu);
            guard += 1;
            assert!(guard < 1_000, "straight-line ALU program did not converge");
        }

        // The whole run (bar the final ecall) must have gone through the SIMD fast path — lanes
        // never diverge in pc here, so there is no reason to ever fall back to scalar-over-lanes.
        assert_eq!(
            vcpu.simd_alu_steps, alu_insn_count,
            "every ALU instruction in a converged, branch-free program should hit the SIMD fast \
             path exactly once"
        );

        for (lane, &seed) in seeds.iter().enumerate() {
            let scalar = run_scalar(&prog, Some((T2, seed)));
            assert_eq!(
                vcpu.exit[lane],
                Some(LaneExit::Ecall { a0: scalar.regs[A0 as usize] }),
                "lane {lane}"
            );
            for reg in 0..32 {
                assert_eq!(
                    vcpu.regs[reg][lane], scalar.regs[reg],
                    "lane {lane} register x{reg} vs its scalar oracle"
                );
            }
        }

        // Sanity check that per-lane inputs really drove the computation rather than being
        // ignored (the `0xff` mask and shifts make some seeds collide, so this only checks that
        // results are not all identical, not that every one of the 16 is distinct).
        let distinct: std::collections::HashSet<u32> =
            (0..LANES).map(|lane| vcpu.regs[A0 as usize][lane]).collect();
        assert!(distinct.len() > 1, "lanes should not all compute the same result");
    }

    /// The new `VecMmu`-backed memory fast path (DESIGN.md "Memory: same-address fast path"):
    /// every lane shares the same effective address (`t5`) but stores/loads a distinct per-lane
    /// value (`t2`) — this must take `VecCpu::try_simd_store`/`try_simd_load`'s same-address
    /// vectorized access exactly twice (once for the `sw`, once for the `lw`), never falling back
    /// to `step_lane`'s per-lane path, and every lane must read back exactly what it (not some
    /// other lane) stored — proving the interleaved layout keeps each lane's copy of the shared
    /// word genuinely independent.
    #[test]
    fn converged_same_address_load_store_uses_the_simd_mem_fast_path() {
        use asm::*;
        const T2: u8 = 7; // per-lane value to store
        const T5: u8 = 30; // shared data pointer (same for every lane)
        let prog = vec![
            lui(T5, BASE),          // 0: t5 = BASE
            addi(T5, T5, 0x100),    // 1: t5 += 0x100  (same address for every lane)
            sw(T5, T2, 0),          // 2: store this lane's seed at [t5]
            lw(A0, T5, 0),          // 3: a0 = load back from [t5]
            addi(A7, X0, 93),       // 4: exit
            ecall(),                // 5
        ];
        let mut mmu = make_mmu(&prog);
        let mut vcpu = VecCpu::new(BASE);
        let seeds: [u32; LANES] = std::array::from_fn(|lane| (lane as u32) * 37 + 11);
        for (lane, &seed) in seeds.iter().enumerate() {
            vcpu.set_reg(lane, T2, seed);
        }

        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut mmu);
            guard += 1;
            assert!(guard < 1_000, "converged same-address program did not converge");
        }

        assert_eq!(
            vcpu.simd_mem_steps, 2,
            "both the store and the load-back should take the same-address SIMD fast path"
        );
        for (lane, &seed) in seeds.iter().enumerate() {
            assert_eq!(vcpu.regs[A0 as usize][lane], seed, "lane {lane} did not read back its own value");
            assert_eq!(vcpu.exit[lane], Some(LaneExit::Ecall { a0: seed }), "lane {lane}");
        }
    }

    /// The divergent-address path (DESIGN.md "Divergent-address path"): every lane targets a
    /// *different* guest address (set directly per lane, rather than computed identically), so
    /// `try_simd_store`/`try_simd_load` must decline every time and `step_lane` services each
    /// lane through `VecMmu::store_lane`/`load_lane` instead — the scalar body the eventual
    /// `vpgatherdd`/`vpscatterdd` batching wraps. Each lane must still read back exactly its own
    /// stored value, with zero cross-lane interference.
    #[test]
    fn divergent_addresses_fall_back_to_the_per_lane_path() {
        use asm::*;
        const T2: u8 = 7; // per-lane value to store
        const T5: u8 = 30; // per-lane absolute data address, set directly so it diverges by construction
        let prog = vec![sw(T5, T2, 0), lw(A0, T5, 0), addi(A7, X0, 93), ecall()];
        let mut mmu = make_mmu(&prog);
        let mut vcpu = VecCpu::new(BASE);
        let seeds: [u32; LANES] = std::array::from_fn(|lane| (lane as u32) * 13 + 3);
        for (lane, &seed) in seeds.iter().enumerate() {
            vcpu.set_reg(lane, T2, seed);
            vcpu.set_reg(lane, T5, BASE + 0x300 + (lane as u32) * 4);
        }

        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut mmu);
            guard += 1;
            assert!(guard < 1_000, "divergent-address program did not converge");
        }

        assert_eq!(
            vcpu.simd_mem_steps, 0,
            "per-lane-divergent addresses must never take the same-address SIMD fast path"
        );
        for (lane, &seed) in seeds.iter().enumerate() {
            assert_eq!(vcpu.regs[A0 as usize][lane], seed, "lane {lane} did not read back its own value");
            assert_eq!(vcpu.exit[lane], Some(LaneExit::Ecall { a0: seed }), "lane {lane}");
        }
    }

    /// A same-address store the shared `VecMmu` must decline (missing `PERM_WRITE`) has to fall
    /// back to `step_lane`'s per-lane store — which, since every lane shares the same permission
    /// state here, means every lane independently faults rather than the fast path silently
    /// "succeeding" for some lanes and not others.
    #[test]
    fn converged_store_permission_fault_falls_back_and_halts_every_lane() {
        use asm::*;
        const T5: u8 = 30;
        let data_addr = BASE + 0x200;
        let prog = vec![
            lui(T5, BASE),       // 0: t5 = BASE
            addi(T5, T5, 0x200), // 1: t5 += 0x200 (same address for every lane)
            addi(A0, X0, 0x11),  // 2: a0 = dummy value to store
            sw(T5, A0, 0),       // 3: store (should fault: read-only)
            addi(A7, X0, 93),    // 4: exit (unreached)
            ecall(),             // 5
        ];
        let mut mmu = make_mmu(&prog);
        // Downgrade just the target word to read-only: every lane's store must now fault.
        mmu.protect(data_addr, 4, PERM_READ).unwrap();
        let mut vcpu = VecCpu::new(BASE);

        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut mmu);
            guard += 1;
            assert!(guard < 1_000, "permission-fault program did not converge");
        }

        assert_eq!(
            vcpu.simd_mem_steps, 0,
            "a same-address store that VecMmu declines must never count as a fast-path success"
        );
        for lane in 0..LANES {
            match vcpu.exit[lane] {
                Some(LaneExit::Fault(f)) => {
                    assert_eq!(f.kind, fs_mmu::FaultKind::Permission, "lane {lane}");
                }
                other => panic!("lane {lane}: expected a permission fault, got {other:?}"),
            }
        }
    }
}
