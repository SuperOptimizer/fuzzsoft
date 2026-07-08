//! Curated dictionary of "interesting" 32-bit constants that real kernel code actually branches
//! on — ioctl request codes, netlink message types/flags, socket family/protocol/option numbers,
//! errno values, fcntl/prctl commands, epoll/eventfd/timerfd flag bits, and page/size boundaries.
//!
//! Motivation (this module exists to fix a specific, measured gap): the cmplog agent
//! (`crate::cmplog::mutate_cmplog`) found that many real-kernel branches are gated on magic
//! constants (`if (cmd == TCGETS)`, `if (nlh->nlmsg_type == RTM_NEWLINK)`, ...), but cmplog can
//! only *substitute* a value it discovers was compared against — it can't invent one from
//! nothing. If the generator/mutator never happens to place a value anywhere near a
//! constant-gated branch, cmplog has nothing to log and nothing to substitute. This dictionary is
//! the input-side fix: bias fresh generation (`genr::gen_arg_value`, wired into the `Int`/`Flags`
//! cases) and a dedicated mutation operator (`mutate::mutate_dict_const`) to draw from this list a
//! fraction of the time, so real kernel constants show up in the corpus organically — both
//! reaching those branches directly and giving cmplog concrete values to log comparisons against
//! in the first place.
//!
//! Deliberately excludes made-up "sentinel" magic numbers (e.g. `0xdeadbeef`) per this module's
//! brief — every entry below is a real constant with a cited kernel source, the same citation
//! discipline `syscalls.rs` already uses for its `ioctl`/`sockaddr` literals.
//!
//! Each group is a separate `pub const` (so it stays independently reviewable/citable and
//! reusable, e.g. a future desc could reference `IOCTL_REQCODES` directly); [`DICTIONARY_GROUPS`]
//! lists them together for [`pick_dict_const`]'s uniform "any dictionary constant" access.

use crate::rng::Rng;

/// ioctl request codes: this crate's own `ioctl$*` descriptions' cmd literals (see `syscalls.rs`'s
/// `TCGETS`/`TIOCGWINSZ`/etc. consts, and the new wave-10 consts alongside them) plus a couple of
/// widely-hit ones not yet backed by a dedicated description — uapi/asm-generic/ioctls.h,
/// uapi/linux/sockios.h.
pub const IOCTL_REQCODES: &[u32] = &[
    0x5401, // TCGETS
    0x5402, // TCSETS
    0x5413, // TIOCGWINSZ
    0x5414, // TIOCSWINSZ
    0x541B, // FIONREAD
    0x5421, // FIONBIO
    0x540F, // TIOCGPGRP
    0x5410, // TIOCSPGRP
    0x5450, // FIONCLEX
    0x5451, // FIOCLEX
    0x5452, // FIOASYNC
    0x8912, // SIOCGIFCONF
    0x8913, // SIOCGIFFLAGS
    0x8914, // SIOCSIFFLAGS
    0x8927, // SIOCGIFHWADDR
];

/// netlink message types (`NLMSG_*`/`RTM_*`) and `nlmsg_flags` bits (`NLM_F_*`) —
/// uapi/linux/netlink.h, uapi/linux/rtnetlink.h.
pub const NETLINK_CONSTS: &[u32] = &[
    1, // NLMSG_NOOP
    2, // NLMSG_ERROR
    3, // NLMSG_DONE
    4, // NLMSG_OVERRUN
    16, // NLMSG_MIN_TYPE / RTM_NEWLINK
    18, // RTM_GETLINK
    20, // RTM_NEWADDR
    22, // RTM_GETADDR
    24, // RTM_NEWROUTE
    26, // RTM_GETROUTE
    0x1,   // NLM_F_REQUEST
    0x2,   // NLM_F_MULTI
    0x4,   // NLM_F_ACK
    0x8,   // NLM_F_ECHO
    0x100, // NLM_F_ROOT / NLM_F_REPLACE
    0x200, // NLM_F_MATCH / NLM_F_EXCL
    0x400, // NLM_F_ATOMIC / NLM_F_CREATE
    0x300, // NLM_F_DUMP (ROOT|MATCH)
    0x800, // NLM_F_APPEND
];

/// socket family/type/protocol/option numbers — uapi/asm-generic/socket.h, uapi/linux/in.h,
/// uapi/linux/socket.h.
pub const SOCKOPT_CONSTS: &[u32] = &[
    1,  // AF_UNIX
    2,  // AF_INET
    16, // AF_NETLINK
    1,  // SOCK_STREAM
    2,  // SOCK_DGRAM
    3,  // SOCK_RAW
    6,  // IPPROTO_TCP
    17, // IPPROTO_UDP
    1,  // SOL_SOCKET
    270, // SOL_NETLINK
    2, // SO_REUSEADDR
    4, // SO_ERROR
    6, // SO_BROADCAST
    7, // SO_SNDBUF
    8, // SO_RCVBUF
    9, // SO_KEEPALIVE
];

/// errno values — uapi/asm-generic/errno-base.h + uapi/asm-generic/errno.h.
pub const ERRNO_CONSTS: &[u32] = &[
    1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32,
    34, 38, 39, 40,
];

/// fcntl (uapi/asm-generic/fcntl.h) `F_*` command numbers and prctl (uapi/linux/prctl.h) `PR_*`
/// option numbers.
pub const FCNTL_PRCTL_CONSTS: &[u32] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1030, // F_DUPFD..F_GETSIG, F_DUPFD_CLOEXEC
    15, 16, 38, 4, 1, 2, 3, 22, 23, 24, 21, 29, 30, 47, // PR_SET_NAME..PR_CAP_AMBIENT
];

/// epoll (uapi/linux/eventpoll.h) / eventfd (uapi/linux/eventfd.h) / timerfd
/// (uapi/linux/timerfd.h) flag bits.
pub const EPOLL_EVENTFD_TIMERFD_CONSTS: &[u32] = &[
    0x1,        // EPOLLIN
    0x4,        // EPOLLOUT
    0x8,        // EPOLLERR
    0x10,       // EPOLLHUP
    0x2000,     // EPOLLRDHUP
    0x40000000, // EPOLLONESHOT
    0x80000000, // EPOLLET
    1,          // EFD_SEMAPHORE / TFD_TIMER_ABSTIME
];

/// Page/size boundaries and off-by-ones a data-dependent length/range check commonly gates on.
pub const BOUNDARY_CONSTS: &[u32] = &[
    0, 1, 2, 4095, 4096, 4097, 0xFFFF, 0x1_0000, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF,
];

/// All groups together, for uniform "pick any dictionary constant" access ([`pick_dict_const`])
/// and for tests that check overall coverage. Order matches the doc comment citation order above.
pub static DICTIONARY_GROUPS: &[&[u32]] = &[
    IOCTL_REQCODES,
    NETLINK_CONSTS,
    SOCKOPT_CONSTS,
    ERRNO_CONSTS,
    FCNTL_PRCTL_CONSTS,
    EPOLL_EVENTFD_TIMERFD_CONSTS,
    BOUNDARY_CONSTS,
];

/// Total distinct-slot count across every group (includes intra-group duplicates like `1`/`2`
/// appearing in more than one group — that's intentional: those values are genuinely common
/// across subsystems, so weighting them slightly higher is correct, not a bug).
pub fn dictionary_len() -> usize {
    DICTIONARY_GROUPS.iter().map(|g| g.len()).sum()
}

/// Pick one constant from the whole dictionary. The group is chosen uniformly first, then a
/// constant uniformly within that group — this keeps a large group (e.g. `ERRNO_CONSTS`) from
/// dominating the small ones (e.g. `IOCTL_REQCODES`) purely by virtue of having more entries.
pub fn pick_dict_const(rng: &mut Rng) -> u32 {
    let group = rng.pick(DICTIONARY_GROUPS);
    *rng.pick(group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_group_is_nonempty() {
        for (i, g) in DICTIONARY_GROUPS.iter().enumerate() {
            assert!(!g.is_empty(), "dictionary group {i} is empty");
        }
    }

    #[test]
    fn dictionary_has_a_healthy_number_of_constants() {
        assert!(
            dictionary_len() >= 50,
            "expected >=50 total dictionary entries, got {}",
            dictionary_len()
        );
    }

    #[test]
    fn pick_dict_const_is_deterministic_for_same_seed() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..50 {
            assert_eq!(pick_dict_const(&mut a), pick_dict_const(&mut b));
        }
    }

    #[test]
    fn pick_dict_const_only_ever_returns_a_cataloged_value() {
        let all: Vec<u32> = DICTIONARY_GROUPS.iter().flat_map(|g| g.iter().copied()).collect();
        let mut rng = Rng::new(9);
        for _ in 0..500 {
            let v = pick_dict_const(&mut rng);
            assert!(all.contains(&v), "{v:#x} not in any dictionary group");
        }
    }

    #[test]
    fn pick_dict_const_reaches_multiple_groups_across_many_draws() {
        // Across enough draws, values from at least a handful of distinct groups should show up
        // — proves group selection isn't somehow collapsing onto just one group.
        let mut rng = Rng::new(4);
        let mut hit_groups: Vec<bool> = vec![false; DICTIONARY_GROUPS.len()];
        for _ in 0..2000 {
            let v = pick_dict_const(&mut rng);
            for (i, g) in DICTIONARY_GROUPS.iter().enumerate() {
                if g.contains(&v) {
                    hit_groups[i] = true;
                }
            }
        }
        let hit_count = hit_groups.iter().filter(|b| **b).count();
        assert!(
            hit_count >= DICTIONARY_GROUPS.len() - 1,
            "only hit {hit_count}/{} groups in 2000 draws",
            DICTIONARY_GROUPS.len()
        );
    }
}
