//! Cooperative hypercall path: a guest agent reports its own `malloc`/`free` calls via a
//! reserved `ecall`, and the emulator stamps sanitizer permissions in response. This is the
//! cheap, precise path for targets we *can* recompile (or that already ship such an agent);
//! `hooks.rs` is the fallback for when we cannot.
//!
//! fuzzsoft's fuzzing hypercall (`fs-riscv::Cpu.hypercall_eid` / `SysExit::Hypercall(a0)`,
//! decision #6) already reserves one `a7` value for `ecall` and hands the caller `a0`. This
//! module defines the sanitizer's own sub-protocol *within that channel*: `a0` carries the
//! command, `a1`/`a2` carry its arguments (read directly out of `Cpu::regs`, since
//! `SysExit::Hypercall` only surfaces `a0` — the registers themselves are public).
//!
//! ```ignore
//! // In the runner, after `SysExit::Hypercall(a0) = cpu.step_system(&mut mmu)`:
//! let a1 = cpu.regs[11];
//! let a2 = cpu.regs[12];
//! match fs_san::hypercall::dispatch(&mut sanitizer, &mut mmu, a0, a1, a2) {
//!     Some(Ok(())) => {}                       // handled
//!     Some(Err(e)) => report_bug(e),           // sanitizer-detected bug (double-free, etc.)
//!     None => { /* not a sanitizer command; try other hypercall handlers */ }
//! }
//! ```

use fs_mmu::Mmu;

use crate::alloc::{SanError, Sanitizer};

/// `cmd=malloc`: `a1` = returned pointer, `a2` = requested size.
pub const CMD_MALLOC: u32 = 0x5A4E_0001;
/// `cmd=free`: `a1` = pointer being freed.
pub const CMD_FREE: u32 = 0x5A4E_0002;

/// Dispatch one sanitizer hypercall. `cmd` is the value the runner got back as
/// `SysExit::Hypercall(cmd)` (guest `a0`); `a1`/`a2` are read by the caller directly out of the
/// guest register file (see module docs).
///
/// Returns `None` if `cmd` is not a sanitizer command (the caller should try other handlers, or
/// treat it as unrecognized), `Some(result)` if it was.
pub fn dispatch(
    san: &mut Sanitizer,
    mmu: &mut Mmu,
    cmd: u32,
    a1: u32,
    a2: u32,
) -> Option<Result<(), SanError>> {
    match cmd {
        CMD_MALLOC => Some(san.alloc(mmu, a1, a2)),
        CMD_FREE => Some(san.free(mmu, a1)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs_mmu::FaultKind;

    #[test]
    fn malloc_then_free_hypercalls_round_trip() {
        let mut mmu = Mmu::new(0x8000_0000, 0x10000);
        let mut san = Sanitizer::new(16);
        let addr = 0x8000_1000u32;

        assert_eq!(
            dispatch(&mut san, &mut mmu, CMD_MALLOC, addr, 32),
            Some(Ok(()))
        );
        assert!(san.is_live(addr));

        mmu.write(addr, &[7; 32]).unwrap();

        assert_eq!(
            dispatch(&mut san, &mut mmu, CMD_FREE, addr, 0),
            Some(Ok(()))
        );
        assert!(!san.is_live(addr));
        assert_eq!(mmu.read_u8(addr).unwrap_err().kind, FaultKind::Permission);
    }

    #[test]
    fn unknown_command_is_none() {
        let mut mmu = Mmu::new(0x8000_0000, 0x1000);
        let mut san = Sanitizer::new(16);
        assert_eq!(dispatch(&mut san, &mut mmu, 0xdead_beef, 0, 0), None);
    }
}
