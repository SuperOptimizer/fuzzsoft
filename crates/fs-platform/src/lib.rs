//! The full-system machine: physical RAM + MMIO devices, addressed by physical address, plus the
//! driver loop that couples the CLINT timer to the CPU's deterministic virtual time.
//!
//! Memory map follows QEMU `virt` (decision #15) so stock kernels/DTBs and the QEMU/Spike
//! references line up. M2 currently models RAM + the CLINT; PLIC/UART/virtio come later.

#![forbid(unsafe_code)]

use fs_mmu::{Access, Bus, CowRam, Fault, FaultKind, Golden, Mmu};
use fs_riscv::{Cpu, SysExit};
use std::sync::Arc;

/// Core-Local Interruptor: software interrupt (msip), timer compare (mtimecmp), and the
/// monotonic timer (mtime). Base and register offsets match the SiFive/ACLINT CLINT.
pub const CLINT_BASE: u32 = 0x0200_0000;
pub const CLINT_SIZE: u32 = 0x0001_0000;
const CLINT_MSIP: u32 = 0x0000;
const CLINT_MTIMECMP: u32 = 0x4000;
const CLINT_MTIME: u32 = 0xbff8;

#[derive(Debug, Default, Clone)]
pub struct Clint {
    pub msip: u32,
    pub mtimecmp: u64,
    /// Snapshot of virtual time, refreshed by the driver each step so MMIO reads see it.
    pub mtime: u64,
}

impl Clint {
    fn load(&self, off: u32, size: u8) -> u32 {
        let _ = size; // all CLINT registers are read in 32-bit halves
        match off {
            CLINT_MSIP => self.msip & 1,
            o if o == CLINT_MTIMECMP => self.mtimecmp as u32,
            o if o == CLINT_MTIMECMP + 4 => (self.mtimecmp >> 32) as u32,
            o if o == CLINT_MTIME => self.mtime as u32,
            o if o == CLINT_MTIME + 4 => (self.mtime >> 32) as u32,
            _ => 0,
        }
    }
    fn store(&mut self, off: u32, val: u32) {
        match off {
            CLINT_MSIP => self.msip = val & 1,
            o if o == CLINT_MTIMECMP => {
                self.mtimecmp = (self.mtimecmp & 0xffff_ffff_0000_0000) | val as u64
            }
            o if o == CLINT_MTIMECMP + 4 => {
                self.mtimecmp = (self.mtimecmp & 0xffff_ffff) | ((val as u64) << 32)
            }
            _ => {} // mtime is read-only (driven by the CPU)
        }
    }
}

/// A polled ns16550 UART: bytes written to THR are captured; LSR always reports "ready".
/// Enough for OpenSBI/kernel console output (we don't model input or interrupts yet).
pub const UART_BASE: u32 = 0x1000_0000;
pub const UART_SIZE: u32 = 0x100;
const UART_THR: u32 = 0; // write: transmit; read: RBR
const UART_LSR: u32 = 5; // line status

#[derive(Debug, Default, Clone)]
pub struct Uart {
    pub out: Vec<u8>,
}

impl Uart {
    fn load(&self, off: u32) -> u32 {
        match off {
            UART_LSR => 0x60, // THR empty (0x20) | transmitter empty (0x40)
            _ => 0,
        }
    }
    fn store(&mut self, off: u32, val: u32) {
        if off == UART_THR {
            self.out.push(val as u8);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Shared physical-address routing (PR2 of the software COW design, `docs/cow-shared-ram.md`):
// both `Machine` (owned `Mmu`) and `CowMachine` (shared-golden `CowRam`) dispatch a physical
// address to RAM / CLINT / UART identically. Factored here once so the two `Bus` impls cannot
// silently drift apart.
// ---------------------------------------------------------------------------------------------

pub(crate) fn in_ram(addr: u32, ram_base: u32, ram_end: u32) -> bool {
    (ram_base..ram_end).contains(&addr)
}
pub(crate) fn in_clint(addr: u32) -> bool {
    (CLINT_BASE..CLINT_BASE + CLINT_SIZE).contains(&addr)
}
pub(crate) fn in_uart(addr: u32) -> bool {
    (UART_BASE..UART_BASE + UART_SIZE).contains(&addr)
}

pub(crate) fn mmio_fault(addr: u32, size: u8, access: Access) -> Fault {
    Fault { addr, len: size as u32, access, kind: FaultKind::Unmapped }
}

/// The physical machine: RAM (soft-MMU) + MMIO devices.
#[derive(Clone)]
pub struct Machine {
    pub ram: Mmu,
    ram_base: u32,
    ram_end: u32,
    pub clint: Clint,
    pub uart: Uart,
}

impl Machine {
    pub fn new(ram_base: u32, ram_size: u32) -> Self {
        Self {
            ram: Mmu::new(ram_base, ram_size as usize),
            ram_base,
            ram_end: ram_base.wrapping_add(ram_size),
            clint: Clint::default(),
            uart: Uart::default(),
        }
    }

    fn in_ram(&self, addr: u32) -> bool {
        in_ram(addr, self.ram_base, self.ram_end)
    }
    fn in_clint(&self, addr: u32) -> bool {
        in_clint(addr)
    }
    fn in_uart(&self, addr: u32) -> bool {
        in_uart(addr)
    }

    fn fault(addr: u32, size: u8, access: Access) -> Fault {
        mmio_fault(addr, size, access)
    }
}

impl Bus for Machine {
    fn load(&mut self, addr: u32, size: u8) -> Result<u32, Fault> {
        if self.in_clint(addr) {
            Ok(self.clint.load(addr - CLINT_BASE, size))
        } else if self.in_uart(addr) {
            Ok(self.uart.load(addr - UART_BASE))
        } else if self.in_ram(addr) {
            self.ram.load(addr, size)
        } else {
            Err(Self::fault(addr, size, Access::Read))
        }
    }
    fn store(&mut self, addr: u32, size: u8, val: u32) -> Result<(), Fault> {
        if self.in_clint(addr) {
            self.clint.store(addr - CLINT_BASE, val);
            Ok(())
        } else if self.in_uart(addr) {
            self.uart.store(addr - UART_BASE, val);
            Ok(())
        } else if self.in_ram(addr) {
            self.ram.store(addr, size, val)
        } else {
            Err(Self::fault(addr, size, Access::Write))
        }
    }
    fn ifetch16(&mut self, addr: u32) -> Result<u16, Fault> {
        if self.in_ram(addr) {
            self.ram.ifetch16(addr)
        } else {
            Err(Self::fault(addr, 2, Access::Exec))
        }
    }
    fn store_may_assert_interrupt(&self, addr: u32, _size: u8) -> bool {
        self.in_clint(addr)
    }
}

// ---------------------------------------------------------------------------------------------
// `CowMachine`: the same physical machine, but RAM is a per-lane `CowRam` copy-on-write view over
// a shared, immutable `Arc<Golden>` image instead of an owned `Mmu`. CLINT/UART stay per-lane
// (small, not shared/COW'd). Routing/dispatch is byte-identical to `Machine` (both call the
// shared `in_ram`/`in_clint`/`in_uart`/`mmio_fault` free functions above), so the audited scalar
// full-system core (`fs_riscv::Cpu::step_system(&mut dyn Bus)`) drives a `CowMachine` with zero
// `fs-riscv` changes. See `docs/cow-shared-ram.md` (PR2) and `cow_machine.rs`'s differential test.
pub struct CowMachine {
    pub ram: CowRam,
    ram_base: u32,
    ram_end: u32,
    pub clint: Clint,
    pub uart: Uart,
}

impl CowMachine {
    /// A fresh per-lane view over `golden`: RAM starts entirely golden (no overlay pages
    /// allocated yet); CLINT/UART start at their defaults, same as `Machine::new`.
    pub fn from_golden(golden: Arc<Golden>, ram_base: u32, ram_size: u32) -> Self {
        debug_assert_eq!(golden.base(), ram_base, "golden base must match ram_base");
        debug_assert_eq!(golden.size(), ram_size as usize, "golden size must match ram_size");
        Self {
            ram: CowRam::new(golden),
            ram_base,
            ram_end: ram_base.wrapping_add(ram_size),
            clint: Clint::default(),
            uart: Uart::default(),
        }
    }

    /// Convenience for callers that don't already have a `Golden`: snapshot `m.ram` as the golden
    /// image and build a `CowMachine` over it. PR3 (VecSystem) will instead capture one `Golden`
    /// and call `from_golden` up to 16×32 times over the *same* `Arc`, so it does not use this.
    pub fn from_machine(m: &Machine) -> Self {
        let golden = Arc::new(Golden::from_mmu(&m.ram));
        let ram_size = golden.size() as u32;
        Self::from_golden(golden, m.ram_base, ram_size)
    }

    /// Revert this lane's RAM to golden (O(dirty), zero byte copy-back). CLINT/UART reset
    /// orchestration (mirroring `Snapshot::reset`) is left to PR3, which owns the per-lane
    /// case-reset loop.
    pub fn reset_case(&mut self) {
        self.ram.reset();
    }
}

impl Bus for CowMachine {
    fn load(&mut self, addr: u32, size: u8) -> Result<u32, Fault> {
        if in_clint(addr) {
            Ok(self.clint.load(addr - CLINT_BASE, size))
        } else if in_uart(addr) {
            Ok(self.uart.load(addr - UART_BASE))
        } else if in_ram(addr, self.ram_base, self.ram_end) {
            self.ram.load(addr, size)
        } else {
            Err(mmio_fault(addr, size, Access::Read))
        }
    }
    fn store(&mut self, addr: u32, size: u8, val: u32) -> Result<(), Fault> {
        if in_clint(addr) {
            self.clint.store(addr - CLINT_BASE, val);
            Ok(())
        } else if in_uart(addr) {
            self.uart.store(addr - UART_BASE, val);
            Ok(())
        } else if in_ram(addr, self.ram_base, self.ram_end) {
            self.ram.store(addr, size, val)
        } else {
            Err(mmio_fault(addr, size, Access::Write))
        }
    }
    fn ifetch16(&mut self, addr: u32) -> Result<u16, Fault> {
        if in_ram(addr, self.ram_base, self.ram_end) {
            self.ram.ifetch16(addr)
        } else {
            Err(mmio_fault(addr, 2, Access::Exec))
        }
    }
    fn store_may_assert_interrupt(&self, addr: u32, _size: u8) -> bool {
        in_clint(addr)
    }
}

const MIP_MSIP: u32 = 1 << 3;

/// A golden whole-machine snapshot for fast reset fuzzing: full RAM contents+permissions plus the
/// hart and device state. Reset restores only the dirtied blocks (O(bytes touched)).
#[derive(Clone)]
pub struct Snapshot {
    gmem: Vec<u8>,
    gperms: Vec<u8>,
    cpu: Cpu,
    clint: Clint,
    uart_len: usize,
}

impl Snapshot {
    /// Capture the current state as golden and switch RAM into dirty-tracking mode.
    pub fn capture(cpu: &Cpu, machine: &mut Machine) -> Self {
        let (m, p) = machine.ram.planes();
        let gmem = m.to_vec();
        let gperms = p.to_vec();
        machine.ram.enable_dirty_tracking();
        Snapshot {
            gmem,
            gperms,
            cpu: cpu.clone(),
            clint: machine.clint.clone(),
            uart_len: machine.uart.out.len(),
        }
    }

    /// Restore to the golden state in O(bytes dirtied since the last reset/capture).
    pub fn reset(&self, cpu: &mut Cpu, machine: &mut Machine) {
        machine.ram.reset_dirty(&self.gmem, &self.gperms);
        *cpu = self.cpu.clone();
        machine.clint = self.clint.clone();
        machine.uart.out.truncate(self.uart_len);
    }
}

/// Why a run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Halt(u32),
    Hypercall(u32),
    Budget,
}

/// Shared body of `sync_timer`/`sync_timer_cow`: push a `Clint`'s compare/pending-IPI state into
/// the hart. Factored so `Machine` and `CowMachine` can't drift on timer semantics either.
#[inline]
fn apply_clint(cpu: &mut Cpu, clint: &Clint) {
    cpu.csr.mtimecmp = clint.mtimecmp;
    if clint.msip & 1 != 0 {
        cpu.csr.mip |= MIP_MSIP;
    } else {
        cpu.csr.mip &= !MIP_MSIP;
    }
}

/// Sync the CLINT timer/IPI state into the hart for one step.
#[inline]
pub fn sync_timer(cpu: &mut Cpu, machine: &Machine) {
    apply_clint(cpu, &machine.clint);
}

/// Same as `sync_timer`, for a `CowMachine` (PR3's VecSystem per-lane fallback driver needs
/// this — the CLINT/timer logic is identical to the scalar `Machine` path, only the RAM backing
/// differs).
#[inline]
pub fn sync_timer_cow(cpu: &mut Cpu, machine: &CowMachine) {
    apply_clint(cpu, &machine.clint);
}

/// Run until an HTIF halt, a fuzzing hypercall, or `cpu.insns_retired >= deadline`.
pub fn run_until(cpu: &mut Cpu, machine: &mut Machine, deadline: u64) -> Stop {
    while cpu.insns_retired < deadline {
        machine.clint.mtime = cpu.virtual_time();
        sync_timer(cpu, machine);
        match cpu.step_system(machine) {
            SysExit::Continue => {}
            SysExit::Halt(c) => return Stop::Halt(c),
            SysExit::Hypercall(c) => return Stop::Hypercall(c),
        }
    }
    Stop::Budget
}

/// Drive `cpu` on `machine` until an HTIF halt or the instruction budget. Each step syncs the
/// CLINT (mtime = virtual time; the M-timer compare and software-interrupt pending bits) into the
/// hart before stepping, keeping timer interrupts a strict function of retired instructions.
pub fn run(cpu: &mut Cpu, machine: &mut Machine, max_insns: u64) -> Option<u32> {
    while cpu.insns_retired < max_insns {
        machine.clint.mtime = cpu.virtual_time();
        apply_clint(cpu, &machine.clint);
        if let SysExit::Halt(code) = cpu.step_system(machine) {
            return Some(code);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};
    use fs_riscv::{asm, A0, T0, T1, X0};

    #[test]
    fn clint_mmio_roundtrip() {
        let mut m = Machine::new(0x8000_0000, 0x1000);
        // Program mtimecmp low/high via MMIO, read it back.
        m.store(CLINT_BASE + CLINT_MTIMECMP, 4, 0xdead_beef).unwrap();
        m.store(CLINT_BASE + CLINT_MTIMECMP + 4, 4, 0x0000_0007).unwrap();
        assert_eq!(m.load(CLINT_BASE + CLINT_MTIMECMP, 4).unwrap(), 0xdead_beef);
        assert_eq!(m.load(CLINT_BASE + CLINT_MTIMECMP + 4, 4).unwrap(), 7);
        // mtime reflects the driver-provided snapshot.
        m.clint.mtime = 0x1_0000_002a;
        assert_eq!(m.load(CLINT_BASE + CLINT_MTIME, 4).unwrap(), 0x2a);
        assert_eq!(m.load(CLINT_BASE + CLINT_MTIME + 4, 4).unwrap(), 1);
    }

    #[test]
    fn clint_machine_timer_interrupt() {
        let base = 0x8000_0000u32;
        let tohost = base + 0x2000;
        let handler = base + 0x100;
        let mut m = Machine::new(base, 0x1_0000);
        m.ram.protect(base, 0x1_0000, PERM_READ | PERM_WRITE).unwrap();
        // Loop forever until the M-timer fires.
        m.ram
            .map(base, &0x0000_006fu32.to_le_bytes(), PERM_READ | PERM_WRITE | PERM_EXEC)
            .unwrap();
        // Handler: a0 = 99, HTIF exit.
        let mut code = Vec::new();
        for w in [asm::addi(A0, X0, 99), asm::lui(T0, tohost), asm::addi(T1, X0, 1), asm::sw(T0, T1, 0)] {
            code.extend_from_slice(&w.to_le_bytes());
        }
        m.ram.map(handler, &code, PERM_READ | PERM_WRITE | PERM_EXEC).unwrap();

        let mut cpu = Cpu::new(base);
        cpu.htif_tohost = Some(tohost);
        cpu.csr.mtvec = handler;
        cpu.csr.mie |= 1 << 7; // MTIE
        cpu.csr.mstatus |= fs_riscv::sys::MSTATUS_MIE;
        m.clint.mtimecmp = 5; // fire after 5 retired instructions

        assert_eq!(run(&mut cpu, &mut m, 10_000), Some(0));
        assert_eq!(cpu.regs[A0 as usize], 99); // handler ran
        assert_eq!(cpu.csr.mcause, 0x8000_0007); // interrupt | machine-timer(7)
    }
}
