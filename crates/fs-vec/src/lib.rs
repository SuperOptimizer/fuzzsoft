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
//! See `DESIGN.md` in this crate for the full AVX-512 target (interleaved MMU, `vmovdqa32`
//! same-address fast path vs `vpgatherdd`/`vpscatterdd`, masked scalarize-16 fallback for
//! DIV/REM/MULH, and where `unsafe` will eventually live).

#![forbid(unsafe_code)]

use fs_mmu::{Bus, Fault, Mmu};
use fs_riscv::{decode, decode_compressed, AluOp, AmoOp, BranchOp, Inst, LoadOp, MulOp, StoreOp};

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
    /// A memory fault (unmapped/permission/unaligned) from the lane's own `Mmu`.
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

    /// Advance every active lane by exactly one instruction, each against its own `Mmu`
    /// (`buses[lane]` — see DESIGN.md for the future interleaved single-shared layout).
    ///
    /// Scalar-over-lanes loop (decision #45, correctness-first): each lane independently
    /// fetches/decodes/executes via `fs_riscv::decode`/`decode_compressed` and the M-extension
    /// semantics copied below, but the SoA layout + active mask keep a future AVX-512 lockstep
    /// step a mechanical drop-in.
    pub fn step(&mut self, buses: &mut [Mmu]) {
        assert_eq!(buses.len(), LANES, "fs-vec: exactly one Mmu per lane");
        for (lane, bus) in buses.iter_mut().enumerate() {
            if self.active[lane] {
                self.step_lane(lane, bus);
            }
        }
    }

    fn halt(&mut self, lane: usize, exit: LaneExit) {
        self.active[lane] = false;
        self.exit[lane] = Some(exit);
    }

    fn step_lane(&mut self, lane: usize, bus: &mut Mmu) {
        let pc = self.pc[lane];

        // Variable-length fetch, identical to the scalar core: a half-word whose low 2 bits
        // != 0b11 is a 16-bit compressed instruction (IALIGN=16); otherwise it is 32-bit.
        let lo = match bus.ifetch16(pc) {
            Ok(v) => v,
            Err(f) => return self.halt(lane, LaneExit::Fault(f)),
        };
        let (inst, ilen, iword) = if lo & 0x3 != 0x3 {
            (decode_compressed(lo), 2u32, lo as u32)
        } else {
            let hi = match bus.ifetch16(pc.wrapping_add(2)) {
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
                let raw = try_mem!(bus.load(addr, size));
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
                try_mem!(bus.store(addr, size, val));
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
                let v = try_mem!(bus.load(addr, 4));
                self.reservation[lane] = Some(addr);
                self.wr_reg(lane, rd, v);
            }
            Inst::ScW { rd, rs1, rs2, .. } => {
                let addr = self.rd_reg(lane, rs1);
                let success = self.reservation[lane] == Some(addr);
                if success {
                    let v = self.rd_reg(lane, rs2);
                    try_mem!(bus.store(addr, 4, v));
                }
                // A reservation is single-use, and any trap/context-switch would clear it too.
                self.reservation[lane] = None;
                self.wr_reg(lane, rd, if success { 0 } else { 1 });
            }
            Inst::AmoW { op, rd, rs1, rs2, .. } => {
                let addr = self.rd_reg(lane, rs1);
                let old = try_mem!(bus.load(addr, 4));
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
                try_mem!(bus.store(addr, 4, result));
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
    use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_riscv::{asm, A0, A7, T0, T1, X0};

    const BASE: u32 = 0x8000_0000;

    fn make_mmu(prog: &[u32]) -> Mmu {
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
        let mut mmu = make_mmu(prog);
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
        let mut buses: Vec<Mmu> = (0..LANES).map(|_| make_mmu(&prog)).collect();
        let mut vcpu = VecCpu::new(BASE);

        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut buses);
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

        let mut buses: Vec<Mmu> = (0..LANES).map(|_| make_mmu(&prog)).collect();
        let mut vcpu = VecCpu::new(BASE);
        let ns: [u32; LANES] = std::array::from_fn(|lane| lane as u32); // N = 0, 1, .., 15
        for (lane, &n) in ns.iter().enumerate() {
            vcpu.set_reg(lane, T2, n);
        }

        let mut saw_pc_divergence = false;
        let mut saw_partial_completion = false;
        let mut guard = 0;
        while vcpu.any_active() {
            vcpu.step(&mut buses);
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
}
