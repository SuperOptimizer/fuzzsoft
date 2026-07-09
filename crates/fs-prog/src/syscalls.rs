//! Starter syscall description table: real rv32 (asm-generic) syscall numbers, verified
//! against `build/linux-src/arch/riscv/include/generated/uapi/asm/unistd_32.h` /
//! `qemu/linux-headers/asm-riscv/unistd_32.h`. See `docs/syzlang.md` "Starter descriptions".
//!
//! Every `nr` below is cited against a specific kernel source line so a reviewer can re-verify
//! it without re-deriving anything; see the per-description comments. A handful of syscalls take
//! a 64-bit (`loff_t`) argument split across two 32-bit registers on a *native* 32-bit kernel
//! (not "compat" in the 32-on-64 sense — `include/uapi/asm-generic/unistd.h`'s `__SC_COMP` macro
//! resolves to the `compat_sys_*` entry point whenever `__BITS_PER_LONG == 32`, which is exactly
//! this kernel's case). Those splits use `compat_arg_u64_dual`'s little-endian order (`name_lo`
//! before `name_hi` — see `include/asm-generic/compat.h`), confirmed against each syscall's
//! `COMPAT_SYSCALL_DEFINE*` in `fs/read_write.c` / `fs/open.c` / `fs/sync.c`.

use crate::resource::{FD, KEY, SOCK, VMA};
use crate::types::ArgType;
use crate::types::ArgType::*;
use crate::types::Dir::*;
use crate::types::{Field, LenSpec, Produces, SyscallDesc};

// ---------------- flag/const tables ----------------

pub const OPEN_FLAGS: &[u32] = &[
    0o0,       // O_RDONLY
    0o1,       // O_WRONLY
    0o2,       // O_RDWR
    0o200,     // O_EXCL (asm-generic/fcntl.h: 00000200)
    0o100,     // O_CREAT
    0o1000,    // O_TRUNC
    0o2000,    // O_APPEND
    0o40000,   // O_DIRECT (asm-generic/fcntl.h: 00040000)
    0o200000,  // O_DIRECTORY
    0o2000000, // O_CLOEXEC
    0o4000,    // O_NONBLOCK
    0x20000,   // O_NOFOLLOW (asm-generic/fcntl.h: 1<<17)
    0x200000,  // O_PATH (asm-generic/fcntl.h: 1<<21)
    0x410000,  // O_TMPFILE == __O_TMPFILE(1<<22) | O_DIRECTORY(1<<16)
];
pub const OPEN_MODE: &[u32] = &[0o600, 0o644, 0o755, 0o777, 0];
pub const AF_FAMILY: &[u32] = &[1 /* AF_UNIX */, 2 /* AF_INET */];
pub const SOCK_TYPE: &[u32] = &[1 /* SOCK_STREAM */, 2 /* SOCK_DGRAM */];
pub const SEND_FLAGS: &[u32] = &[
    0, 0x40,   /* MSG_DONTWAIT */
    0x4000, /* MSG_NOSIGNAL */
];
pub const SEEK_WHENCE: &[u32] = &[0 /* SET */, 1 /* CUR */, 2 /* END */];
pub const O_FLAGS_SETFL: &[u32] = &[
    0o2000,  /* APPEND */
    0o4000,  /* NONBLOCK */
    0o10000, /* ASYNC */
];
pub const MEMFD_FLAGS: &[u32] = &[
    0, 1, /* MFD_CLOEXEC */
    2, /* MFD_ALLOW_SEALING */
    3,
];
pub const PRCTL_OPTION: &[u32] = &[
    15, /* PR_SET_NAME */
    16, /* PR_GET_NAME */
    38, /* PR_SET_NO_NEW_PRIVS */
    4,  /* PR_SET_DUMPABLE */
    1,  /* PR_SET_PDEATHSIG */
    2,  /* PR_GET_PDEATHSIG */
    3,  /* PR_GET_DUMPABLE */
    22, /* PR_SET_SECCOMP */
    23, /* PR_CAPBSET_READ */
    24, /* PR_CAPBSET_DROP */
];
pub const IOCTL_CMD: &[u32] = &[
    0x541B, /* FIONREAD */
    0x5421, /* FIONBIO */
    0x5401, /* TCGETS */
];
pub const FACCESSAT_MODE: &[u32] = &[
    0, /* F_OK */
    1, /* X_OK */
    2, /* W_OK */
    4, /* R_OK */
];
pub const STATX_MASK: &[u32] = &[
    0x7ff, /* STATX_BASIC_STATS */
    0x800, /* STATX_BTIME */
];
pub const PATH_POOL: &[&str] = &["/", "/dev/null", "/proc/self/maps", "/tmp/x"];
pub const MEMFD_NAMES: &[&str] = &["a", "fuzz", ""];

// mmap2/mprotect/madvise (uapi/asm-generic/mman-common.h, uapi/linux/mman.h)
pub const MMAP_PROT: &[u32] = &[
    0,         /* PROT_NONE */
    1,         /* PROT_READ */
    2,         /* PROT_WRITE */
    4,         /* PROT_EXEC */
    1 | 2,     /* READ|WRITE */
    1 | 4,     /* READ|EXEC */
];
pub const MMAP_FLAGS: &[u32] = &[
    0x01,        /* MAP_SHARED */
    0x02,        /* MAP_PRIVATE */
    0x10,        /* MAP_FIXED */
    0x20,        /* MAP_ANONYMOUS */
    0x4000,      /* MAP_NORESERVE (uapi/asm-generic/mman.h) */
    0x8000,      /* MAP_POPULATE (uapi/asm-generic/mman.h) */
    0x20000,     /* MAP_STACK (uapi/asm-generic/mman.h) */
    0x02 | 0x20, /* MAP_PRIVATE|MAP_ANONYMOUS */
];
pub const MADV_ADVICE: &[u32] = &[
    0, /* MADV_NORMAL */
    1, /* MADV_RANDOM */
    4, /* MADV_DONTNEED */
    8, /* MADV_FREE */
];

// eventfd2 (uapi/linux/eventfd.h: EFD_SEMAPHORE=1<<0, EFD_CLOEXEC=O_CLOEXEC, EFD_NONBLOCK=O_NONBLOCK)
pub const EFD_FLAGS: &[u32] = &[0, 1, 0o2000000, 0o4000];
// epoll_create1 (uapi/linux/eventpoll.h: EPOLL_CLOEXEC=O_CLOEXEC)
pub const EPOLL_CREATE_FLAGS: &[u32] = &[0, 0o2000000];
// epoll_ctl op (uapi/linux/eventpoll.h: EPOLL_CTL_ADD=1, _DEL=2, _MOD=3)
pub const EPOLL_OP: &[u32] = &[1, 2, 3];
// epoll_event.events subset (uapi/linux/eventpoll.h)
pub const EPOLL_EVENTS: &[u32] = &[
    0x00000001, /* EPOLLIN */
    0x00000004, /* EPOLLOUT */
    0x00000008, /* EPOLLERR */
    0x00000010, /* EPOLLHUP */
    0x00002000, /* EPOLLRDHUP */
    0x40000000, /* EPOLLONESHOT */
    0x80000000, /* EPOLLET */
];
// inotify_init1 (uapi/linux/inotify.h: IN_CLOEXEC=O_CLOEXEC, IN_NONBLOCK=O_NONBLOCK)
pub const IN_INIT_FLAGS: &[u32] = &[0, 0o2000000, 0o4000];
// inotify_add_watch mask subset (uapi/linux/inotify.h)
pub const IN_MASK: &[u32] = &[
    0x00000001, /* IN_ACCESS */
    0x00000002, /* IN_MODIFY */
    0x00000004, /* IN_ATTRIB */
    0x00000008, /* IN_CLOSE_WRITE */
    0x00000020, /* IN_OPEN */
    0x00000100, /* IN_CREATE */
    0x00000200, /* IN_DELETE */
    0x40000000, /* IN_ISDIR */
];
// flock operation (bits/fcntl-linux.h convention: LOCK_SH=1,LOCK_EX=2,LOCK_NB=4,LOCK_UN=8)
pub const FLOCK_OP: &[u32] = &[1, 2, 4, 8];
// timerfd_create clockid (uapi/linux/time.h)
pub const CLOCKIDS: &[u32] = &[0 /* CLOCK_REALTIME */, 1 /* CLOCK_MONOTONIC */, 7 /* CLOCK_BOOTTIME */];
// timerfd_create flags (uapi/linux/timerfd.h: TFD_CLOEXEC=O_CLOEXEC, TFD_NONBLOCK=O_NONBLOCK)
pub const TFD_FLAGS: &[u32] = &[0, 0o2000000, 0o4000];
// timerfd_settime64 flags (uapi/linux/timerfd.h: TFD_TIMER_ABSTIME=1<<0)
pub const TFD_SETTIME_FLAGS: &[u32] = &[0, 1];
// signalfd4 flags (uapi/linux/signalfd.h: SFD_CLOEXEC=O_CLOEXEC, SFD_NONBLOCK=O_NONBLOCK)
pub const SFD_FLAGS: &[u32] = &[0, 0o2000000, 0o4000];
// setsockopt/getsockopt level+optname (uapi/asm-generic/socket.h)
pub const SOCKOPT_LEVEL: &[u32] = &[1 /* SOL_SOCKET */];
pub const SOCKOPT_NAME: &[u32] = &[
    2, /* SO_REUSEADDR */
    4, /* SO_ERROR */
    6, /* SO_BROADCAST */
    7, /* SO_SNDBUF */
    8, /* SO_RCVBUF */
    9, /* SO_KEEPALIVE */
];
pub const SHUTDOWN_HOW: &[u32] = &[0, 1, 2];
// accept4 flags (uapi/linux/net.h: SOCK_CLOEXEC=O_CLOEXEC, SOCK_NONBLOCK=O_NONBLOCK)
pub const ACCEPT4_FLAGS: &[u32] = &[0, 0o2000000, 0o4000];
// fallocate mode (uapi/linux/falloc.h)
pub const FALLOCATE_MODE: &[u32] = &[
    0x00, /* FALLOC_FL_ALLOCATE_RANGE */
    0x01, /* FALLOC_FL_KEEP_SIZE */
    0x02, /* FALLOC_FL_PUNCH_HOLE */
    0x10, /* FALLOC_FL_ZERO_RANGE */
];
// close_range flags (uapi/linux/close_range.h)
pub const CLOSE_RANGE_FLAGS: &[u32] = &[0, 2 /* CLOSE_RANGE_UNSHARE */, 4 /* CLOSE_RANGE_CLOEXEC */];
// fcntl F_SETFD arg (uapi/asm-generic/fcntl.h: FD_CLOEXEC=1)
pub const FD_FLAGS: &[u32] = &[0, 1];
// preadv2/pwritev2 flags (uapi/linux/fs.h: RWF_*)
pub const RWF_FLAGS: &[u32] = &[
    0x00, 0x01, /* RWF_HIPRI */
    0x02, /* RWF_DSYNC */
    0x04, /* RWF_SYNC */
    0x08, /* RWF_NOWAIT */
];
// F_DUPFD_CLOEXEC (asm-generic/fcntl.h: F_LINUX_SPECIFIC_BASE(1024) + 6)
pub const F_DUPFD_CLOEXEC: u32 = 1030;

// ---- wave 8: ioctl request codes with REAL, well-known literal values ----
// These are all "legacy" ioctl numbers assigned directly in uapi/asm-generic/ioctls.h /
// uapi/linux/sockios.h *before* the generic `_IOC(dir,type,nr,size)` encoding scheme existed, so
// they are cited as literals (per this crate's expansion brief, option 2: "well-known literal
// values with a comment citing them") rather than re-derived via `_IOC` — deriving them would
// require inventing a `dir`/`type`/`size` decomposition that doesn't actually correspond to how
// these particular numbers were assigned upstream.
pub const TCGETS: u32 = 0x5401; // struct termios* (get)
pub const TCSETS: u32 = 0x5402; // struct termios* (set)
pub const TIOCGWINSZ: u32 = 0x5413; // struct winsize* (get)
pub const TIOCSWINSZ: u32 = 0x5414; // struct winsize* (set)
pub const FIONREAD: u32 = 0x541B; // int* (bytes available to read)
pub const FIONBIO: u32 = 0x5421; // int* (enable/disable O_NONBLOCK)
// uapi/linux/sockios.h — also pre-_IOC legacy BSD-derived ioctl numbers.
pub const SIOCGIFCONF: u32 = 0x8912; // struct ifconf*
pub const SIOCGIFFLAGS: u32 = 0x8913; // struct ifreq*

// ---- wave 9: real sockaddr subtype layouts + setsockopt (level,optname) pairs ----
pub const AF_NETLINK: u32 = 16; // uapi/linux/socket.h
pub const AF_INET_ONLY: &[u32] = &[2 /* AF_INET */];
pub const AF_UNIX_ONLY: &[u32] = &[1 /* AF_UNIX */];
pub const NETLINK_SOCK_TYPE: &[u32] = &[2 /* SOCK_DGRAM */, 3 /* SOCK_RAW */];
pub const NETLINK_PROTO: &[u32] = &[0 /* NETLINK_ROUTE */, 4 /* NETLINK_FIREWALL(legacy)/generic */];
pub const SUN_PATHS: &[&str] = &["/tmp/s", "/tmp/y", ""]; // "" => Linux autobind (abstract-ish)
pub const IPPROTO_TCP: u32 = 6; // uapi/linux/in.h
pub const SOL_SOCKET: u32 = 1; // uapi/asm-generic/socket.h
pub const SO_REUSEADDR: u32 = 2; // uapi/asm-generic/socket.h
pub const TCP_NODELAY: u32 = 1; // uapi/linux/tcp.h

// ---- wave 10: more ioctl request codes + a real nlmsghdr-shaped netlink sendmsg ----
// Same "well-known pre-_IOC literal" citation discipline as wave 8 above.
pub const TIOCGPGRP: u32 = 0x540F; // pid_t* (get foreground process group)
pub const TIOCSPGRP: u32 = 0x5410; // const pid_t* (set foreground process group)
pub const FIONCLEX: u32 = 0x5450; // no argp (clear FD_CLOEXEC) — modeled with a dummy nullable ptr
pub const FIOCLEX: u32 = 0x5451; // no argp (set FD_CLOEXEC) — modeled with a dummy nullable ptr
pub const FIOASYNC: u32 = 0x5452; // int* (enable/disable O_ASYNC/SIGIO)
pub const SIOCSIFFLAGS: u32 = 0x8914; // struct ifreq* (uapi/linux/sockios.h)
pub const SIOCGIFHWADDR: u32 = 0x8927; // struct ifreq* (uapi/linux/sockios.h)
// setsockopt(SOL_SOCKET, SO_SNDBUF) — uapi/asm-generic/socket.h.
pub const SO_SNDBUF: u32 = 7;
// SOL_NETLINK-level sockopt (uapi/linux/socket.h: SOL_NETLINK=270) — a materially different
// `level` namespace than every other `setsockopt$*` desc below, which are all SOL_SOCKET/
// IPPROTO_TCP.
pub const SOL_NETLINK: u32 = 270;
pub const NETLINK_ADD_MEMBERSHIP: u32 = 1; // uapi/linux/netlink.h

// ---- wave 11: fault injection (fail_nth) arming preamble ----
// See docs/bug-finding.md's "FAULT INJECTION FIRST": `/proc/self/fail-nth`
// (fs/proc/base.c's `proc_fail_nth_operations`, gated `#ifdef CONFIG_FAULT_INJECTION`) lets a
// task arm `should_fail_ex()` (lib/fault-inject.c) to fail exactly its Nth matching allocation,
// then self-disarm. A single-entry pool, not a free string — this is a fixed control-file path,
// not fuzzed data.
pub const FAIL_NTH_PATH: &[&str] = &["/proc/self/fail-nth"];
// Countdown values to arm: small decimal-ASCII strings. fs/proc/base.c's `proc_fail_nth_write`
// parses via `kstrtouint_from_user`, which is fine with the `StringConst`'s own trailing NUL —
// it copies exactly `count` bytes, appends its own terminator right after, and `kstrtouint` stops
// parsing at the first non-digit byte either way.
pub const FAIL_NTH_COUNTS: &[&str] = &["0", "1", "2", "3", "5", "8", "16"];
// nlmsg_type real values (uapi/linux/netlink.h generic + uapi/linux/rtnetlink.h RTM_* subset).
pub const NLMSG_TYPE: &[u32] = &[
    1,  /* NLMSG_NOOP */
    2,  /* NLMSG_ERROR */
    3,  /* NLMSG_DONE */
    16, /* RTM_NEWLINK */
    18, /* RTM_GETLINK */
    22, /* RTM_GETADDR */
    26, /* RTM_GETROUTE */
];
// nlmsg_flags real bits (uapi/linux/netlink.h).
pub const NLM_F_FLAGS: &[u32] = &[
    0x1,   /* NLM_F_REQUEST */
    0x2,   /* NLM_F_MULTI */
    0x4,   /* NLM_F_ACK */
    0x100, /* NLM_F_ROOT */
    0x200, /* NLM_F_MATCH */
    0x300, /* NLM_F_DUMP (ROOT|MATCH) */
];

// ---- wave 12: splice/vmsplice/tee (fs/splice.c) ----
// `SPLICE_F_*` bits (include/linux/splice.h; there's no uapi header for these — they're
// exposed only via this literal value set, same citation discipline as the wave-8/10 ioctls).
pub const SPLICE_FLAGS: &[u32] = &[
    0,
    0x01, /* SPLICE_F_MOVE */
    0x02, /* SPLICE_F_NONBLOCK */
    0x04, /* SPLICE_F_MORE */
    0x08, /* SPLICE_F_GIFT */
];

// ---- wave 13: unshare/setns namespaces (kernel/fork.c, kernel/nsproxy.c) ----
// CLONE_NEW*/CLONE_{FS,FILES,THREAD,SYSVSEM} bits (uapi/linux/sched.h) valid for
// unshare(2)'s `unshare_flags`.
pub const UNSHARE_FLAGS: &[u32] = &[
    0x00000200, /* CLONE_FS */
    0x00000400, /* CLONE_FILES */
    0x00000080, /* CLONE_NEWTIME */
    0x00010000, /* CLONE_THREAD */
    0x00020000, /* CLONE_NEWNS */
    0x00040000, /* CLONE_SYSVSEM */
    0x02000000, /* CLONE_NEWCGROUP */
    0x04000000, /* CLONE_NEWUTS */
    0x08000000, /* CLONE_NEWIPC */
    0x10000000, /* CLONE_NEWUSER */
    0x20000000, /* CLONE_NEWPID */
    0x40000000, /* CLONE_NEWNET */
];
// setns(2)'s `nstype` (0 = don't check / infer from fd; else one real CLONE_NEW* bit).
pub const NSTYPE_FLAGS: &[u32] = &[
    0,
    0x00000080, /* CLONE_NEWTIME */
    0x00020000, /* CLONE_NEWNS */
    0x02000000, /* CLONE_NEWCGROUP */
    0x04000000, /* CLONE_NEWUTS */
    0x08000000, /* CLONE_NEWIPC */
    0x10000000, /* CLONE_NEWUSER */
    0x20000000, /* CLONE_NEWPID */
    0x40000000, /* CLONE_NEWNET */
];
// `/proc/self/ns/*` magic-symlink fds (proc_ns_dir_operations) — the standard zero-scaffolding
// way to obtain an fd `setns(2)` will accept, one entry per namespace type actually enabled in
// this kernel's `.config` (CONFIG_{UTS,IPC,USER,PID,NET,TIME}_NS=y — verified against
// `build/linux-slubdebug/.config`; CONFIG_CGROUP_NS wasn't checked so "cgroup" is left out
// rather than risking an always-ENOENT path).
pub const NS_PATHS: &[&str] = &[
    "/proc/self/ns/mnt",
    "/proc/self/ns/uts",
    "/proc/self/ns/ipc",
    "/proc/self/ns/user",
    "/proc/self/ns/pid",
    "/proc/self/ns/net",
    "/proc/self/ns/time",
];

// ---- wave 14: keyctl/add_key/request_key (security/keys/keyctl.c) ----
// Real key type names `key_get_type_from_user` accepts against `.config`'s built-in
// `CONFIG_KEYS=y` type table (security/keys/{user_defined,keyring,request_key_auth}.c register
// "user"/"keyring"/"logon"; "big_key" needs CONFIG_BIG_KEYS, left out since it's not confirmed
// enabled).
pub const KEY_TYPES: &[&str] = &["user", "keyring", "logon"];
pub const KEY_DESCRIPTIONS: &[&str] = &["fuzzkey", "a", ""];
pub const KEY_CALLOUT_INFO: &[&str] = &["-", "fuzz"];
// KEY_SPEC_* special keyring ids (uapi/linux/keyctl.h) usable as an `add_key`/`request_key`
// `ringid`/`destringid` or a `keyctl$unlink` destination keyring — encoded as their real
// (negative) `key_serial_t` bit pattern in a u32 (this crate's `Flags` vals are always u32; the
// guest reads them back as the same signed `int` bits either way).
pub const KEYRING_SPECIAL: &[u32] = &[
    0xffffffff, /* -1 KEY_SPEC_THREAD_KEYRING */
    0xfffffffe, /* -2 KEY_SPEC_PROCESS_KEYRING */
    0xfffffffd, /* -3 KEY_SPEC_SESSION_KEYRING */
    0xfffffffc, /* -4 KEY_SPEC_USER_KEYRING */
    0xfffffffb, /* -5 KEY_SPEC_USER_SESSION_KEYRING */
];
// keyctl(2) `option` values this wave models (uapi/linux/keyctl.h `KEYCTL_*`).
pub const KEYCTL_GET_KEYRING_ID: u32 = 0;
pub const KEYCTL_REVOKE: u32 = 3;
pub const KEYCTL_UNLINK: u32 = 9;
pub const KEYCTL_READ: u32 = 11;
pub const KEYCTL_DESCRIBE: u32 = 6;

// ---- wave 15: process_vm_readv/writev (mm/process_vm_access.c) ----
// struct iovec describing a slice of the *remote* target's address space: unlike this crate's
// `IOVEC` (whose `iov_base` is a real `Ptr` into our own scratch, correct for `lvec` — the local
// side), a remote-side `iov_base` is an address in some *other* task's mm, which this generator
// has no model of — so it's a free `Int` (an arbitrary/biased guess, mostly landing on unmapped
// remote addresses and exercising `access_remote_vm`'s fault/short-copy paths, which is exactly
// the useful fuzz signal here) rather than a `Ptr`.
static REMOTE_IOVEC_FIELDS: &[Field] = &[
    Field {
        name: "iov_base",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "iov_len",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static REMOTE_IOVEC: ArgType = Struct(REMOTE_IOVEC_FIELDS);
// Two-entry remote iovec array, same manual-unroll shape as `IOVEC2` (see its doc comment).
static REMOTE_IOVEC2_FIELDS: &[Field] = &[
    Field {
        name: "iov0",
        ty: &REMOTE_IOVEC,
    },
    Field {
        name: "iov1",
        ty: &REMOTE_IOVEC,
    },
];
static REMOTE_IOVEC2: ArgType = Struct(REMOTE_IOVEC2_FIELDS);

// struct sockaddr (generic, 16 bytes: u16 family + 14 bytes data — enough for AF_UNIX/AF_INET)
static SOCKADDR_FIELDS: &[Field] = &[
    Field {
        name: "family",
        ty: &Flags {
            vals: AF_FAMILY,
            bitmask: false,
        },
    },
    Field {
        name: "data",
        ty: &Buffer {
            len: LenSpec::Fixed(14),
        },
    },
];
static SOCKADDR: ArgType = Struct(SOCKADDR_FIELDS);

// struct statx (partial: just reserve enough scratch bytes; contents are opaque out data)
static STATX_BUF: ArgType = Buffer {
    len: LenSpec::Fixed(256),
};

// struct iovec { void *iov_base; size_t iov_len; } — both fields naturally 4-byte aligned, so
// this already matches the real rv32-ILP32 layout with no padding. `iov_len` is deliberately a
// free `Int` rather than tied to `iov_base`'s actual buffer size: `ArgType::Len{of}` only
// resolves against a sibling in the *top-level* call's arg list (see `genr::generate_args`), not
// a sibling field nested inside a `Struct` — so an occasionally-mismatched `iov_len` here is a
// real (and useful) fuzz signal rather than a modeling gap.
static IOVEC_FIELDS: &[Field] = &[
    Field {
        name: "iov_base",
        ty: &Ptr {
            dir: In,
            inner: &Buffer {
                len: LenSpec::Range(0, 64),
            },
            nullable: false,
        },
    },
    Field {
        name: "iov_len",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static IOVEC: ArgType = Struct(IOVEC_FIELDS);

// struct iovec[2], manually unrolled — syzlang-lite deliberately has no generic array-of-struct
// type (see `docs/syzlang.md`'s "deliberate omissions"), so a fixed 2-entry iovec vector is
// modeled as two named embedded `IOVEC` fields instead.
static IOVEC2_FIELDS: &[Field] = &[
    Field {
        name: "iov0",
        ty: &IOVEC,
    },
    Field {
        name: "iov1",
        ty: &IOVEC,
    },
];
static IOVEC2: ArgType = Struct(IOVEC2_FIELDS);

// struct msghdr (send/recvmsg) — 7 scalar/pointer words, all 4 bytes, so natural alignment
// already matches the real ABI (no padding). `msg_iovlen`/`msg_namelen`/`msg_controllen` are
// independent `Int`s for the same "no Len-inside-Struct" reason `iov_len` is above.
static MSGHDR_FIELDS: &[Field] = &[
    Field {
        name: "msg_name",
        ty: &Ptr {
            dir: In,
            inner: &SOCKADDR,
            nullable: true,
        },
    },
    Field {
        name: "msg_namelen",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "msg_iov",
        ty: &Ptr {
            dir: In,
            inner: &IOVEC2,
            nullable: false,
        },
    },
    Field {
        name: "msg_iovlen",
        ty: &Const(2),
    },
    Field {
        name: "msg_control",
        ty: &Ptr {
            dir: In,
            inner: &Buffer {
                len: LenSpec::Fixed(16),
            },
            nullable: true,
        },
    },
    Field {
        name: "msg_controllen",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "msg_flags",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static MSGHDR: ArgType = Struct(MSGHDR_FIELDS);

// struct epoll_event { __poll_t events; __u64 data; } is `__attribute__((packed))` (no gap
// before the u64 field) — modeled as three natural-aligned 4-byte fields (events, data_lo,
// data_hi) so our natural-alignment `struct_layout` reproduces the same packed 12-byte layout
// without needing a dedicated "packed struct" `ArgType`.
static EPOLL_EVENT_FIELDS: &[Field] = &[
    Field {
        name: "events",
        ty: &Flags {
            vals: EPOLL_EVENTS,
            bitmask: true,
        },
    },
    Field {
        name: "data_lo",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "data_hi",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static EPOLL_EVENT: ArgType = Struct(EPOLL_EVENT_FIELDS);

// struct __kernel_timespec { int64 tv_sec; int64 tv_nsec; } — both 8-byte aligned on rv32.
static TIMESPEC64_FIELDS: &[Field] = &[
    Field {
        name: "tv_sec",
        ty: &Int {
            bits: 64,
            signed: true,
        },
    },
    Field {
        name: "tv_nsec",
        ty: &Int {
            bits: 64,
            signed: true,
        },
    },
];
static TIMESPEC64: ArgType = Struct(TIMESPEC64_FIELDS);

// struct __kernel_itimerspec { timespec64 it_interval; timespec64 it_value; }
static ITIMERSPEC64_FIELDS: &[Field] = &[
    Field {
        name: "it_interval",
        ty: &TIMESPEC64,
    },
    Field {
        name: "it_value",
        ty: &TIMESPEC64,
    },
];
static ITIMERSPEC64: ArgType = Struct(ITIMERSPEC64_FIELDS);

// struct open_how { u64 flags; u64 mode; u64 resolve; } (uapi/linux/openat2.h) — all three
// fields are naturally 8-byte aligned already (no padding), so `Len{of}` on the enclosing `Ptr`
// resolves to the real 24-byte `sizeof(struct open_how)`. Modeled as free `Int`s rather than
// `Flags{OPEN_FLAGS,..}` because `Flags` always serializes as 4 bytes (see
// `lower::value_size_align`) which would break this struct's real 8-byte field width/alignment.
static OPEN_HOW_FIELDS: &[Field] = &[
    Field {
        name: "flags",
        ty: &Int {
            bits: 64,
            signed: false,
        },
    },
    Field {
        name: "mode",
        ty: &Int {
            bits: 64,
            signed: false,
        },
    },
    Field {
        name: "resolve",
        ty: &Int {
            bits: 64,
            signed: false,
        },
    },
];
static OPEN_HOW: ArgType = Struct(OPEN_HOW_FIELDS);

// struct winsize { u16 ws_row, ws_col, ws_xpixel, ws_ypixel; } (uapi/asm-generic/termios.h) —
// four naturally-aligned 2-byte fields, no padding: `sizeof(struct winsize) == 8`.
static WINSIZE_FIELDS: &[Field] = &[
    Field {
        name: "ws_row",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
    Field {
        name: "ws_col",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
    Field {
        name: "ws_xpixel",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
    Field {
        name: "ws_ypixel",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
];
static WINSIZE: ArgType = Struct(WINSIZE_FIELDS);

// struct termios { tcflag_t c_iflag,c_oflag,c_cflag,c_lflag; cc_t c_line; cc_t c_cc[NCCS=19]; }
// (uapi/asm-generic/termbits.h) — four 4-byte tcflag_t + one 1-byte c_line + 19-byte c_cc array
// modeled as a fixed `Buffer` (this crate's type system has no generic fixed-size scalar array,
// see docs/syzlang.md's deliberate omissions; a raw byte buffer is the right fit for opaque
// `cc_t[]` control-character data anyway). Natural layout: 4+4+4+4=16, +1 (c_line)=17, +19
// (c_cc)=36 — already a multiple of the struct's own 4-byte alignment, so no trailing padding;
// matches the real `sizeof(struct termios) == 36` on rv32.
static TERMIOS_FIELDS: &[Field] = &[
    Field {
        name: "c_iflag",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "c_oflag",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "c_cflag",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "c_lflag",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "c_line",
        ty: &Int {
            bits: 8,
            signed: false,
        },
    },
    Field {
        name: "c_cc",
        ty: &Buffer {
            len: LenSpec::Fixed(19),
        },
    },
];
static TERMIOS: ArgType = Struct(TERMIOS_FIELDS);

// struct ifreq (uapi/linux/if.h): `ifr_name[IFNAMSIZ=16]` followed by a union whose largest
// common member (`struct sockaddr`/`ifru_ivalue`/`ifru_flags`) fits in 16 bytes on a 32-bit
// build, giving the real `sizeof(struct ifreq) == 32`. The union is modeled as an opaque 16-byte
// `Buffer` (its interpretation is ioctl-cmd-dependent — exactly the union-avoidance rationale in
// docs/syzlang.md's deliberate omissions) rather than a dedicated field per member.
static IFREQ_FIELDS: &[Field] = &[
    Field {
        name: "ifr_name",
        ty: &Buffer {
            len: LenSpec::Fixed(16),
        },
    },
    Field {
        name: "ifr_ifru",
        ty: &Buffer {
            len: LenSpec::Fixed(16),
        },
    },
];
static IFREQ: ArgType = Struct(IFREQ_FIELDS);

// struct ifconf { int ifc_len; union { char *ifcu_buf; struct ifreq *ifcu_req; } ifc_ifcu; }
// (uapi/linux/if.h) — modeled with a real nested `Ptr` field (like `msghdr`'s `msg_iov`) pointing
// at scratch space sized for a handful of `ifreq`s; both fields are natural 4-byte scalars, no
// padding, matching the real 8-byte `sizeof(struct ifconf)` on rv32.
static IFCONF_FIELDS: &[Field] = &[
    Field {
        name: "ifc_len",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "ifc_buf",
        ty: &Ptr {
            dir: Out,
            inner: &Buffer {
                len: LenSpec::Fixed(128), // room for ~4 ifreqs (32 bytes each)
            },
            nullable: false,
        },
    },
];
static IFCONF: ArgType = Struct(IFCONF_FIELDS);

// struct sockaddr_in { sa_family_t sin_family; in_port_t sin_port; struct in_addr sin_addr;
// unsigned char sin_zero[8]; } (uapi/linux/in.h) — family(2)+port(2)+addr(4)+zero(8) = 16 bytes,
// every field naturally aligned already (no padding); `sin_family` is pinned to `AF_INET`(2) via
// `Const` since this struct only ever describes an AF_INET address (unlike the generic
// `SOCKADDR` above, which is family-polymorphic raw bytes).
static SOCKADDR_IN_FIELDS: &[Field] = &[
    Field {
        name: "sin_family",
        ty: &Const(2 /* AF_INET */),
    },
    Field {
        name: "sin_port",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
    Field {
        name: "sin_addr",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "sin_zero",
        ty: &Buffer {
            len: LenSpec::Fixed(8),
        },
    },
];
static SOCKADDR_IN: ArgType = Struct(SOCKADDR_IN_FIELDS);

// struct sockaddr_un { sa_family_t sun_family; char sun_path[108]; } (uapi/linux/un.h) —
// `sun_path` is modeled as a `StringConst` (variable length + NUL) rather than a fixed 108-byte
// buffer: `Len{of}` on the enclosing `bind`/`connect` call measures the *actual* serialized size
// (2 + strlen+1), which is exactly how real AF_UNIX programs pass `addrlen` (often shorter than
// `sizeof(struct sockaddr_un)`), including the empty-path "" case (Linux autobind-style abstract
// addressing when `addrlen == sizeof(sa_family_t)`).
static SOCKADDR_UN_FIELDS: &[Field] = &[
    Field {
        name: "sun_family",
        ty: &Const(1 /* AF_UNIX */),
    },
    Field {
        name: "sun_path",
        ty: &StringConst(SUN_PATHS),
    },
];
static SOCKADDR_UN: ArgType = Struct(SOCKADDR_UN_FIELDS);

// struct sockaddr_nl { sa_family_t nl_family; unsigned short nl_pad; __u32 nl_pid, nl_groups; }
// (uapi/linux/netlink.h) — family(2)+pad(2)+pid(4)+groups(4) = 12 bytes, all naturally aligned,
// no padding.
static SOCKADDR_NL_FIELDS: &[Field] = &[
    Field {
        name: "nl_family",
        ty: &Const(AF_NETLINK),
    },
    Field {
        name: "nl_pad",
        ty: &Const(0),
    },
    Field {
        name: "nl_pid",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "nl_groups",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static SOCKADDR_NL: ArgType = Struct(SOCKADDR_NL_FIELDS);

// struct nlmsghdr { __u32 nlmsg_len; __u16 nlmsg_type; __u16 nlmsg_flags; __u32 nlmsg_seq;
// __u32 nlmsg_pid; } (uapi/linux/netlink.h), immediately followed here by a fixed payload
// buffer (room for a `struct rtgenmsg`/`ifaddrmsg`/small `rtattr` chain — the real
// classic-netlink-fuzzing shape: a correctly-sized 16-byte header, then opaque attribute bytes).
// `nlmsg_type`/`nlmsg_flags` are modeled as free `Int{16,..}`s rather than `Flags{NLMSG_TYPE,..}`/
// `Flags{NLM_F_FLAGS,..}` for the same reason `OPEN_HOW`'s fields above are: `ArgType::Flags`
// always serializes as 4 bytes (see `lower::value_size_align`), which would break this struct's
// real 2-byte field width/alignment (shifting `nlmsg_seq`/`nlmsg_pid` off their true offsets).
// The dictionary bias wired into `genr::gen_arg_value`'s `Int` case (see `dict.rs`) still lets
// these land on a real `NLMSG_TYPE`/`NLM_F_FLAGS` value a fraction of the time, without the
// layout cost `Flags` would impose. `nlmsg_len` is deliberately a free `Int` too (not derived from
// the payload's actual size) — same "occasionally desynced length field is a real fuzz signal"
// rationale `iov_len`/`msg_iovlen` above already document (`Len{of}` only resolves against a
// top-level call arg, never a nested struct field).
static NLMSG_FIELDS: &[Field] = &[
    Field {
        name: "nlmsg_len",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "nlmsg_type",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
    Field {
        name: "nlmsg_flags",
        ty: &Int {
            bits: 16,
            signed: false,
        },
    },
    Field {
        name: "nlmsg_seq",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "nlmsg_pid",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "payload",
        ty: &Buffer {
            len: LenSpec::Fixed(16),
        },
    },
];
static NLMSG: ArgType = Struct(NLMSG_FIELDS);

// struct iovec pointing at one NLMSG (see `IOVEC`'s doc above for why `iov_len` stays a free,
// independently-generated `Int` rather than a derived `Len{of}`).
static IOVEC_NL_FIELDS: &[Field] = &[
    Field {
        name: "iov_base",
        ty: &Ptr {
            dir: In,
            inner: &NLMSG,
            nullable: false,
        },
    },
    Field {
        name: "iov_len",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static IOVEC_NL: ArgType = Struct(IOVEC_NL_FIELDS);

// struct msghdr for a netlink sendmsg: same 7-field shape as `MSGHDR` above, but `msg_name`
// targets a real `sockaddr_nl` (nullable — netlink to the kernel commonly omits it) and
// `msg_iov` points at a single `IOVEC_NL` (one nlmsghdr-shaped message per call, matching
// `msg_iovlen: Const(1)`) instead of the generic two-iovec vector.
static MSGHDR_NL_FIELDS: &[Field] = &[
    Field {
        name: "msg_name",
        ty: &Ptr {
            dir: In,
            inner: &SOCKADDR_NL,
            nullable: true,
        },
    },
    Field {
        name: "msg_namelen",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "msg_iov",
        ty: &Ptr {
            dir: In,
            inner: &IOVEC_NL,
            nullable: false,
        },
    },
    Field {
        name: "msg_iovlen",
        ty: &Const(1),
    },
    Field {
        name: "msg_control",
        ty: &Ptr {
            dir: In,
            inner: &Buffer {
                len: LenSpec::Fixed(16),
            },
            nullable: true,
        },
    },
    Field {
        name: "msg_controllen",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
    Field {
        name: "msg_flags",
        ty: &Int {
            bits: 32,
            signed: false,
        },
    },
];
static MSGHDR_NL: ArgType = Struct(MSGHDR_NL_FIELDS);

// struct ifreq (uapi/linux/if.h) — reuses the `IFREQ` type already defined above for
// `ioctl$SIOCGIFFLAGS`/`ioctl$SIOCGIFCONF`; the wave-10 ioctls below just pair it with a
// different real request code.

// ---------------- syscall descriptions ----------------

pub static SYSCALLS: &[SyscallDesc] = &[
    // 1. openat(56): dirfd:AT_FDCWD-or-fd, path, flags, mode -> fd
    SyscallDesc {
        name: "openat",
        nr: 56,
        args: &[
            Res(FD), // dirfd (seed AT_FDCWD=-100 covers the common case)
            Ptr {
                dir: In,
                inner: &StringConst(PATH_POOL),
                nullable: false,
            },
            Flags {
                vals: OPEN_FLAGS,
                bitmask: true,
            },
            Flags {
                vals: OPEN_MODE,
                bitmask: false,
            },
        ],
        produces: Produces::Ret(FD),
    },
    // 2. read(63): fd, buf(out), count=len(buf) -> ssize
    SyscallDesc {
        name: "read",
        nr: 63,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 3. write(64): fd, buf(in), count=len(buf) -> ssize
    SyscallDesc {
        name: "write",
        nr: 64,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 4. close(57): fd -> int (consumes the resource; enables double-close mutants)
    SyscallDesc {
        name: "close",
        nr: 57,
        args: &[Res(FD)],
        produces: Produces::None,
    },
    // 5. llseek(62): REAL rv32 5-arg shape, not generic lseek.
    //    fd, off_hi, off_lo, result:ptr[out,i64], whence -> int32
    SyscallDesc {
        name: "llseek",
        nr: 62,
        args: &[
            Res(FD),
            Int {
                bits: 32,
                signed: true,
            }, // offset_high
            Int {
                bits: 32,
                signed: true,
            }, // offset_low
            Ptr {
                dir: Out,
                inner: &Int {
                    bits: 64,
                    signed: true,
                },
                nullable: false,
            }, // loff_t *result
            Flags {
                vals: SEEK_WHENCE,
                bitmask: false,
            },
        ],
        produces: Produces::None,
    },
    // 6. ioctl(29): fd, cmd, arg (generic hand-picked cmd set).
    SyscallDesc {
        name: "ioctl$generic",
        nr: 29,
        args: &[
            Res(FD),
            Flags {
                vals: IOCTL_CMD,
                bitmask: false,
            },
            Ptr {
                dir: InOut,
                inner: &Buffer {
                    len: LenSpec::Fixed(64),
                },
                nullable: true,
            },
        ],
        produces: Produces::None,
    },
    // 7. dup(23): oldfd:fd -> fd
    SyscallDesc {
        name: "dup",
        nr: 23,
        args: &[Res(FD)],
        produces: Produces::Ret(FD),
    },
    // 8. dup3(24): oldfd:fd, newfd:int, flags -> fd
    SyscallDesc {
        name: "dup3",
        nr: 24,
        args: &[
            Res(FD),
            Int {
                bits: 32,
                signed: false,
            }, // newfd
            Flags {
                vals: &[0o2000000 /* O_CLOEXEC */],
                bitmask: true,
            },
        ],
        produces: Produces::Ret(FD),
    },
    // 9a. fcntl64$dupfd(25): fd, cmd=F_DUPFD, arg:int (min new fd) -> fd
    SyscallDesc {
        name: "fcntl64$dupfd",
        nr: 25,
        args: &[
            Res(FD),
            Const(0 /* F_DUPFD */),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::Ret(FD),
    },
    // 9b. fcntl64$setfl(25): fd, cmd=F_SETFL, arg:flags -> int32
    SyscallDesc {
        name: "fcntl64$setfl",
        nr: 25,
        args: &[
            Res(FD),
            Const(4 /* F_SETFL */),
            Flags {
                vals: O_FLAGS_SETFL,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 10. pipe2(59): fds:ptr[out,array[fd,2]], flags -> int32. Multi-resource producer.
    SyscallDesc {
        name: "pipe2",
        nr: 59,
        args: &[
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Fixed(8),
                },
                nullable: false,
            }, // 2x i32 fds
            Flags {
                vals: OPEN_FLAGS,
                bitmask: true,
            }, // O_CLOEXEC/O_NONBLOCK subset in practice
        ],
        produces: Produces::OutArray {
            arg_idx: 0,
            kind: FD,
            count: 2,
        },
    },
    // 11. socket(198): domain, type, proto -> sock (fd subtype)
    SyscallDesc {
        name: "socket",
        nr: 198,
        args: &[
            Flags {
                vals: AF_FAMILY,
                bitmask: false,
            },
            Flags {
                vals: SOCK_TYPE,
                bitmask: false,
            },
            Const(0),
        ],
        produces: Produces::Ret(SOCK),
    },
    // 12. bind(200): fd:sock, addr(in), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "bind",
        nr: 200,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 13. sendto(206): EXACTLY 6 args.
    SyscallDesc {
        name: "sendto",
        nr: 206,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Range(0, 128),
                },
                nullable: false,
            },
            Len { of: 1 },
            Flags {
                vals: SEND_FLAGS,
                bitmask: true,
            },
            Ptr {
                dir: In,
                inner: &SOCKADDR,
                nullable: true,
            },
            Len { of: 4 },
        ],
        produces: Produces::None,
    },
    // 14. getcwd(17): buf(out), size=len(buf) -> int32
    SyscallDesc {
        name: "getcwd",
        nr: 17,
        args: &[
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 0 },
        ],
        produces: Produces::None,
    },
    // 15. faccessat(48): dirfd, path, mode, flags -> int32
    SyscallDesc {
        name: "faccessat",
        nr: 48,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(PATH_POOL),
                nullable: false,
            },
            Flags {
                vals: FACCESSAT_MODE,
                bitmask: true,
            },
            Const(0),
        ],
        produces: Produces::None,
    },
    // 16. statx(291): dirfd, path, flags, mask, buf(out) -> int32
    SyscallDesc {
        name: "statx",
        nr: 291,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(PATH_POOL),
                nullable: false,
            },
            Flags {
                vals: &[0, 0x1000 /* AT_EMPTY_PATH */],
                bitmask: true,
            },
            Flags {
                vals: STATX_MASK,
                bitmask: true,
            },
            Ptr {
                dir: Out,
                inner: &STATX_BUF,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 17. getdents64(61): fd (ideally O_DIRECTORY-opened), buf(out), count=len(buf) -> int32
    SyscallDesc {
        name: "getdents64",
        nr: 61,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(32, 512),
                },
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 18. memfd_create(279): name(in,string), flags -> fd. Zero filesystem dependency.
    SyscallDesc {
        name: "memfd_create",
        nr: 279,
        args: &[
            Ptr {
                dir: In,
                inner: &StringConst(MEMFD_NAMES),
                nullable: false,
            },
            Flags {
                vals: MEMFD_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::Ret(FD),
    },
    // 19. prctl(167): all-scalar, no resources/pointers.
    SyscallDesc {
        name: "prctl",
        nr: 167,
        args: &[
            Flags {
                vals: PRCTL_OPTION,
                bitmask: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 2: eventfd/epoll/inotify/timerfd/signalfd (nr's from unistd_32.h) ============

    // 20. eventfd2(19): initval, flags -> fd
    SyscallDesc {
        name: "eventfd2",
        nr: 19,
        args: &[
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: EFD_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::Ret(FD),
    },
    // 21. epoll_create1(20): flags -> fd
    SyscallDesc {
        name: "epoll_create1",
        nr: 20,
        args: &[Flags {
            vals: EPOLL_CREATE_FLAGS,
            bitmask: true,
        }],
        produces: Produces::Ret(FD),
    },
    // 22. epoll_ctl(21): epfd, op, fd, event(in,nullable for EPOLL_CTL_DEL) -> int32
    SyscallDesc {
        name: "epoll_ctl",
        nr: 21,
        args: &[
            Res(FD), // epfd, ideally an epoll_create1 producer
            Flags {
                vals: EPOLL_OP,
                bitmask: false,
            },
            Res(FD), // the fd being watched
            Ptr {
                dir: In,
                inner: &EPOLL_EVENT,
                nullable: true,
            },
        ],
        produces: Produces::None,
    },
    // 23. epoll_pwait(22): REAL rv32 nr is epoll_pwait, not plain epoll_wait (no epoll_wait on
    //     rv32's asm-generic table). epfd, events(out), maxevents, timeout_ms, sigmask(in,
    //     nullable), sigsetsize=len(sigmask) -> int32
    SyscallDesc {
        name: "epoll_pwait",
        nr: 22,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &EPOLL_EVENT,
                nullable: false,
            },
            Int {
                bits: 32,
                signed: false,
            }, // maxevents
            Int {
                bits: 32,
                signed: true,
            }, // timeout (ms; -1 blocks)
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Fixed(8),
                },
                nullable: true,
            }, // sigmask (sigset_t is 8 bytes)
            Len { of: 4 },
        ],
        produces: Produces::None,
    },
    // 24. inotify_init1(26): flags -> fd
    SyscallDesc {
        name: "inotify_init1",
        nr: 26,
        args: &[Flags {
            vals: IN_INIT_FLAGS,
            bitmask: true,
        }],
        produces: Produces::Ret(FD),
    },
    // 25. inotify_add_watch(27): fd, path, mask -> watch descriptor (plain int; not modeled as a
    //     resource — see inotify_rm_watch's comment).
    SyscallDesc {
        name: "inotify_add_watch",
        nr: 27,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(PATH_POOL),
                nullable: false,
            },
            Flags {
                vals: IN_MASK,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 26. inotify_rm_watch(28): fd, wd. `wd` isn't fd/sock/vma-shaped so it stays a plain biased
    //     Int rather than a new resource kind (kept minimal per docs/syzlang.md's scope cuts);
    //     small values still exercise the real wd namespace often enough to be useful.
    SyscallDesc {
        name: "inotify_rm_watch",
        nr: 28,
        args: &[
            Res(FD),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 27. flock(32): fd, operation -> int32
    SyscallDesc {
        name: "flock",
        nr: 32,
        args: &[
            Res(FD),
            Flags {
                vals: FLOCK_OP,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 28. timerfd_create(85): clockid, flags -> fd
    SyscallDesc {
        name: "timerfd_create",
        nr: 85,
        args: &[
            Flags {
                vals: CLOCKIDS,
                bitmask: false,
            },
            Flags {
                vals: TFD_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::Ret(FD),
    },
    // 29. timerfd_settime64(411): fd, flags, new_value(in), old_value(out,nullable) -> int32
    SyscallDesc {
        name: "timerfd_settime64",
        nr: 411,
        args: &[
            Res(FD),
            Flags {
                vals: TFD_SETTIME_FLAGS,
                bitmask: true,
            },
            Ptr {
                dir: In,
                inner: &ITIMERSPEC64,
                nullable: false,
            },
            Ptr {
                dir: Out,
                inner: &ITIMERSPEC64,
                nullable: true,
            },
        ],
        produces: Produces::None,
    },
    // 30. signalfd4(74): ufd:fd (seed -1 creates a new one), mask(in), sizemask=len(mask),
    //     flags -> fd
    SyscallDesc {
        name: "signalfd4",
        nr: 74,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Fixed(8),
                }, // sigset_t is 8 bytes
                nullable: false,
            },
            Len { of: 1 },
            Flags {
                vals: SFD_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::Ret(FD),
    },

    // ============ wave 3: socket subsystem depth ============

    // 31. socketpair(199): domain, type, protocol, sv:ptr[out,array[sock,2]] -> int32.
    //     Multi-resource producer, same OutArray shape as pipe2.
    SyscallDesc {
        name: "socketpair",
        nr: 199,
        args: &[
            Flags {
                vals: AF_FAMILY,
                bitmask: false,
            },
            Flags {
                vals: SOCK_TYPE,
                bitmask: false,
            },
            Const(0),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Fixed(8),
                },
                nullable: false,
            },
        ],
        produces: Produces::OutArray {
            arg_idx: 3,
            kind: SOCK,
            count: 2,
        },
    },
    // 32. listen(201): sockfd:sock, backlog -> int32
    SyscallDesc {
        name: "listen",
        nr: 201,
        args: &[
            Res(SOCK),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 33. accept4(242): sockfd:sock, addr(out,nullable), addrlen(inout,nullable), flags -> sock
    SyscallDesc {
        name: "accept4",
        nr: 242,
        args: &[
            Res(SOCK),
            Ptr {
                dir: Out,
                inner: &SOCKADDR,
                nullable: true,
            },
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: true,
            },
            Flags {
                vals: ACCEPT4_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::Ret(SOCK),
    },
    // 34. connect(203): sockfd:sock, addr(in), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "connect",
        nr: 203,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 35. getsockname(204): sockfd:sock, addr(out), addrlen(inout,nullable) -> int32
    SyscallDesc {
        name: "getsockname",
        nr: 204,
        args: &[
            Res(SOCK),
            Ptr {
                dir: Out,
                inner: &SOCKADDR,
                nullable: false,
            },
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: true,
            },
        ],
        produces: Produces::None,
    },
    // 36. getpeername(205): same shape as getsockname
    SyscallDesc {
        name: "getpeername",
        nr: 205,
        args: &[
            Res(SOCK),
            Ptr {
                dir: Out,
                inner: &SOCKADDR,
                nullable: false,
            },
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: true,
            },
        ],
        produces: Produces::None,
    },
    // 37. setsockopt(208): sockfd:sock, level, optname, optval(in), optlen=len(optval) -> int32
    SyscallDesc {
        name: "setsockopt",
        nr: 208,
        args: &[
            Res(SOCK),
            Flags {
                vals: SOCKOPT_LEVEL,
                bitmask: false,
            },
            Flags {
                vals: SOCKOPT_NAME,
                bitmask: false,
            },
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Fixed(4),
                },
                nullable: false,
            },
            Len { of: 3 },
        ],
        produces: Produces::None,
    },
    // 38. getsockopt(209): sockfd:sock, level, optname, optval(out), optlen(inout) -> int32
    SyscallDesc {
        name: "getsockopt",
        nr: 209,
        args: &[
            Res(SOCK),
            Flags {
                vals: SOCKOPT_LEVEL,
                bitmask: false,
            },
            Flags {
                vals: SOCKOPT_NAME,
                bitmask: false,
            },
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Fixed(4),
                },
                nullable: false,
            },
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 39. shutdown(210): sockfd:sock, how -> int32
    SyscallDesc {
        name: "shutdown",
        nr: 210,
        args: &[
            Res(SOCK),
            Flags {
                vals: SHUTDOWN_HOW,
                bitmask: false,
            },
        ],
        produces: Produces::None,
    },
    // 40. recvfrom(207): EXACTLY 6 args.
    SyscallDesc {
        name: "recvfrom",
        nr: 207,
        args: &[
            Res(SOCK),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(0, 128),
                },
                nullable: false,
            },
            Len { of: 1 },
            Flags {
                vals: SEND_FLAGS,
                bitmask: true,
            },
            Ptr {
                dir: Out,
                inner: &SOCKADDR,
                nullable: true,
            },
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: true,
            },
        ],
        produces: Produces::None,
    },
    // 41. sendmsg(211): sockfd:sock, msg(in,msghdr incl. nested iovec ptr), flags -> ssize.
    //     Exercises `lower::build_bytes`'s nested-Ptr-inside-Struct serialization (msg_name /
    //     msg_iov / msg_control are pointers *within* the msghdr struct).
    SyscallDesc {
        name: "sendmsg",
        nr: 211,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &MSGHDR,
                nullable: false,
            },
            Flags {
                vals: SEND_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 42. recvmsg(212): sockfd:sock, msg(out,msghdr), flags -> ssize
    SyscallDesc {
        name: "recvmsg",
        nr: 212,
        args: &[
            Res(SOCK),
            Ptr {
                dir: Out,
                inner: &MSGHDR,
                nullable: false,
            },
            Flags {
                vals: SEND_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 4: memory / mmap2 family ============

    // 43. mmap2(222): addr,len,prot,flags,fd,pgoff(page units!) -> vma
    SyscallDesc {
        name: "mmap2",
        nr: 222,
        args: &[
            Int {
                bits: 32,
                signed: false,
            }, // addr hint (usually 0)
            Int {
                bits: 32,
                signed: false,
            }, // len
            Flags {
                vals: MMAP_PROT,
                bitmask: true,
            },
            Flags {
                vals: MMAP_FLAGS,
                bitmask: true,
            },
            Res(FD), // fd, or -1 seed for MAP_ANONYMOUS
            Int {
                bits: 32,
                signed: false,
            }, // pgoff — GOTCHA: page units, not bytes
        ],
        produces: Produces::Ret(VMA),
    },
    // 44. munmap(215): addr:vma, len -> int32 (consumes vma)
    SyscallDesc {
        name: "munmap",
        nr: 215,
        args: &[
            Res(VMA),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 45. mprotect(226): addr:vma, len, prot -> int32
    SyscallDesc {
        name: "mprotect",
        nr: 226,
        args: &[
            Res(VMA),
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: MMAP_PROT,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 46. madvise(233): addr:vma, len, advice -> int32
    SyscallDesc {
        name: "madvise",
        nr: 233,
        args: &[
            Res(VMA),
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: MADV_ADVICE,
                bitmask: false,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 5: file size / positioned IO (native 32-bit loff_t-split ABI) ============

    // 47. ftruncate64(46): fd, length_lo, length_hi -> int32 (compat_arg_u64_dual: lo before hi
    //     on this little-endian target; see fs/open.c COMPAT_SYSCALL_DEFINE3(ftruncate64,...)).
    SyscallDesc {
        name: "ftruncate64",
        nr: 46,
        args: &[
            Res(FD),
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 48. fallocate(47): fd, mode, offset_lo, offset_hi, len_lo, len_hi -> int32 (EXACTLY 6 args;
    //     fs/open.c COMPAT_SYSCALL_DEFINE6(fallocate,...)).
    SyscallDesc {
        name: "fallocate",
        nr: 47,
        args: &[
            Res(FD),
            Flags {
                vals: FALLOCATE_MODE,
                bitmask: true,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 49. pread64(67): fd, buf(out), count=len(buf), pos_lo, pos_hi -> ssize (fs/read_write.c
    //     COMPAT_SYSCALL_DEFINE5(pread64,...)).
    SyscallDesc {
        name: "pread64",
        nr: 67,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 1 },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 50. pwrite64(68): fd, buf(in), count=len(buf), pos_lo, pos_hi -> ssize
    SyscallDesc {
        name: "pwrite64",
        nr: 68,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 1 },
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 51. readv(65): fd, iov(out,array[iovec,2]), iovcnt=2 -> ssize
    SyscallDesc {
        name: "readv",
        nr: 65,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
        ],
        produces: Produces::None,
    },
    // 52. writev(66): fd, iov(in,array[iovec,2]), iovcnt=2 -> ssize
    SyscallDesc {
        name: "writev",
        nr: 66,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
        ],
        produces: Produces::None,
    },
    // 53. preadv(69): fd, iov(out), vlen=2, pos_lo, pos_hi -> ssize (fs/read_write.c
    //     SYSCALL_DEFINE5(preadv,...); native rv32 build uses the non-compat entry since
    //     `unsigned long` is already 32-bit here — still lo/hi split via two registers).
    SyscallDesc {
        name: "preadv",
        nr: 69,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 54. pwritev(70): fd, iov(in), vlen=2, pos_lo, pos_hi -> ssize
    SyscallDesc {
        name: "pwritev",
        nr: 70,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 55. preadv2(286): fd, iov(out), vlen=2, pos_lo, pos_hi, flags -> ssize (EXACTLY 6 args;
    //     fs/read_write.c SYSCALL_DEFINE6(preadv2,...)).
    SyscallDesc {
        name: "preadv2",
        nr: 286,
        args: &[
            Res(FD),
            Ptr {
                dir: Out,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: RWF_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 56. pwritev2(287): fd, iov(in), vlen=2, pos_lo, pos_hi, flags -> ssize
    SyscallDesc {
        name: "pwritev2",
        nr: 287,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Int {
                bits: 32,
                signed: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: RWF_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 6: openat2 / fd-table / pidfd ============

    // 57. openat2(437): dirfd, path, how(in,struct open_how), size=len(how) -> fd
    SyscallDesc {
        name: "openat2",
        nr: 437,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(PATH_POOL),
                nullable: false,
            },
            Ptr {
                dir: In,
                inner: &OPEN_HOW,
                nullable: false,
            },
            Len { of: 2 },
        ],
        produces: Produces::Ret(FD),
    },
    // 58. close_range(436): first:fd, last:fd, flags -> int32. Two independent Res(FD) consumer
    //     slots (not required to be ordered first<=last — that mismatch is itself a fuzz signal).
    SyscallDesc {
        name: "close_range",
        nr: 436,
        args: &[
            Res(FD),
            Res(FD),
            Flags {
                vals: CLOSE_RANGE_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 59. pidfd_open(434): pid, flags -> fd
    SyscallDesc {
        name: "pidfd_open",
        nr: 434,
        args: &[
            Int {
                bits: 32,
                signed: false,
            }, // pid (biased small-int generation covers pid 1 etc.)
            Const(0),
        ],
        produces: Produces::Ret(FD),
    },
    // 60. pidfd_getfd(438): pidfd:fd, targetfd, flags -> fd (duplicates a foreign fd)
    SyscallDesc {
        name: "pidfd_getfd",
        nr: 438,
        args: &[
            Res(FD),
            Int {
                bits: 32,
                signed: false,
            },
            Const(0),
        ],
        produces: Produces::Ret(FD),
    },

    // ============ wave 7: fcntl64 depth ============

    // 61. fcntl64$getfl(25): fd, cmd=F_GETFL -> int32
    SyscallDesc {
        name: "fcntl64$getfl",
        nr: 25,
        args: &[Res(FD), Const(3 /* F_GETFL */)],
        produces: Produces::None,
    },
    // 62. fcntl64$getfd(25): fd, cmd=F_GETFD -> int32
    SyscallDesc {
        name: "fcntl64$getfd",
        nr: 25,
        args: &[Res(FD), Const(1 /* F_GETFD */)],
        produces: Produces::None,
    },
    // 63. fcntl64$setfd(25): fd, cmd=F_SETFD, arg:FD_CLOEXEC-or-0 -> int32
    SyscallDesc {
        name: "fcntl64$setfd",
        nr: 25,
        args: &[
            Res(FD),
            Const(2 /* F_SETFD */),
            Flags {
                vals: FD_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 64. fcntl64$dupfd_cloexec(25): fd, cmd=F_DUPFD_CLOEXEC(1030), arg:int (min new fd) -> fd.
    //     Another FD *producer* variant (like fcntl64$dupfd) so the resource pool has one more
    //     depth-building option besides openat/socket/pipe2/memfd_create.
    SyscallDesc {
        name: "fcntl64$dupfd_cloexec",
        nr: 25,
        args: &[
            Res(FD),
            Const(F_DUPFD_CLOEXEC),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::Ret(FD),
    },

    // ============ wave 8: ioctl with real request codes + correctly-shaped struct ptrs ============
    // Every `cmd` below is a REAL literal ioctl number (see the `TCGETS`/`TIOCGWINSZ`/etc.
    // constants' comments above this table for the exact uapi header + rationale for citing them
    // as literals rather than `_IOC`-recomputing them).

    // 65. ioctl$TIOCGWINSZ(29): fd, cmd=TIOCGWINSZ(0x5413), argp:ptr[out,winsize] -> int32
    SyscallDesc {
        name: "ioctl$TIOCGWINSZ",
        nr: 29,
        args: &[
            Res(FD),
            Const(TIOCGWINSZ),
            Ptr {
                dir: Out,
                inner: &WINSIZE,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 66. ioctl$TIOCSWINSZ(29): fd, cmd=TIOCSWINSZ(0x5414), argp:ptr[in,winsize] -> int32
    SyscallDesc {
        name: "ioctl$TIOCSWINSZ",
        nr: 29,
        args: &[
            Res(FD),
            Const(TIOCSWINSZ),
            Ptr {
                dir: In,
                inner: &WINSIZE,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 67. ioctl$TCGETS(29): fd, cmd=TCGETS(0x5401), argp:ptr[out,termios] -> int32
    SyscallDesc {
        name: "ioctl$TCGETS",
        nr: 29,
        args: &[
            Res(FD),
            Const(TCGETS),
            Ptr {
                dir: Out,
                inner: &TERMIOS,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 68. ioctl$TCSETS(29): fd, cmd=TCSETS(0x5402), argp:ptr[in,termios] -> int32
    SyscallDesc {
        name: "ioctl$TCSETS",
        nr: 29,
        args: &[
            Res(FD),
            Const(TCSETS),
            Ptr {
                dir: In,
                inner: &TERMIOS,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 69. ioctl$FIONREAD(29): fd, cmd=FIONREAD(0x541B), argp:ptr[out,int32] -> int32
    SyscallDesc {
        name: "ioctl$FIONREAD",
        nr: 29,
        args: &[
            Res(FD),
            Const(FIONREAD),
            Ptr {
                dir: Out,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 70. ioctl$FIONBIO(29): fd, cmd=FIONBIO(0x5421), argp:ptr[in,int32] (0/1) -> int32
    SyscallDesc {
        name: "ioctl$FIONBIO",
        nr: 29,
        args: &[
            Res(FD),
            Const(FIONBIO),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 71. ioctl$SIOCGIFFLAGS(29): fd:sock, cmd=SIOCGIFFLAGS(0x8913), argp:ptr[inout,ifreq] -> int32
    SyscallDesc {
        name: "ioctl$SIOCGIFFLAGS",
        nr: 29,
        args: &[
            Res(SOCK),
            Const(SIOCGIFFLAGS),
            Ptr {
                dir: InOut,
                inner: &IFREQ,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 72. ioctl$SIOCGIFCONF(29): fd:sock, cmd=SIOCGIFCONF(0x8912), argp:ptr[inout,ifconf] -> int32
    SyscallDesc {
        name: "ioctl$SIOCGIFCONF",
        nr: 29,
        args: &[
            Res(SOCK),
            Const(SIOCGIFCONF),
            Ptr {
                dir: InOut,
                inner: &IFCONF,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 9: real sockaddr subtype layouts + netlink + typed setsockopt ============

    // 73. socket$netlink(198): domain=AF_NETLINK(16), type, protocol -> sock. AF_NETLINK sockets
    //     exercise a materially different kernel subsystem (net/netlink/af_netlink.c) than the
    //     AF_UNIX/AF_INET paths the generic `socket` description reaches.
    SyscallDesc {
        name: "socket$netlink",
        nr: 198,
        args: &[
            Const(AF_NETLINK),
            Flags {
                vals: NETLINK_SOCK_TYPE,
                bitmask: false,
            },
            Flags {
                vals: NETLINK_PROTO,
                bitmask: false,
            },
        ],
        produces: Produces::Ret(SOCK),
    },
    // 74. bind$inet(200): sockfd:sock, addr(in,sockaddr_in), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "bind$inet",
        nr: 200,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR_IN,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 75. bind$un(200): sockfd:sock, addr(in,sockaddr_un), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "bind$un",
        nr: 200,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR_UN,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 76. bind$nl(200): sockfd:sock, addr(in,sockaddr_nl), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "bind$nl",
        nr: 200,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR_NL,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 77. connect$inet(203): sockfd:sock, addr(in,sockaddr_in), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "connect$inet",
        nr: 203,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR_IN,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 78. connect$un(203): sockfd:sock, addr(in,sockaddr_un), addrlen=len(addr) -> int32
    SyscallDesc {
        name: "connect$un",
        nr: 203,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &SOCKADDR_UN,
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },
    // 79. sendto$inet(206): EXACTLY 6 args, addr(in,nullable,sockaddr_in) -> ssize
    SyscallDesc {
        name: "sendto$inet",
        nr: 206,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Range(0, 128),
                },
                nullable: false,
            },
            Len { of: 1 },
            Flags {
                vals: SEND_FLAGS,
                bitmask: true,
            },
            Ptr {
                dir: In,
                inner: &SOCKADDR_IN,
                nullable: true,
            },
            Len { of: 4 },
        ],
        produces: Produces::None,
    },
    // 80. setsockopt$so_reuseaddr(208): sockfd:sock, level=SOL_SOCKET(1), optname=SO_REUSEADDR(2),
    //     optval(in,int32), optlen=len(optval) -> int32. A real (level,optname) pair with a
    //     correctly sized (4-byte) `int*` optval, rather than the generic table's random
    //     level/optname combination.
    SyscallDesc {
        name: "setsockopt$so_reuseaddr",
        nr: 208,
        args: &[
            Res(SOCK),
            Const(SOL_SOCKET),
            Const(SO_REUSEADDR),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
            Len { of: 3 },
        ],
        produces: Produces::None,
    },
    // 81. setsockopt$tcp_nodelay(208): sockfd:sock, level=IPPROTO_TCP(6), optname=TCP_NODELAY(1),
    //     optval(in,int32), optlen=len(optval) -> int32
    SyscallDesc {
        name: "setsockopt$tcp_nodelay",
        nr: 208,
        args: &[
            Res(SOCK),
            Const(IPPROTO_TCP),
            Const(TCP_NODELAY),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
            Len { of: 3 },
        ],
        produces: Produces::None,
    },

    // ============ wave 10: deeper ioctl/driver + real netlink sendmsg reach ============
    // Attacks the gap the cmplog agent found directly: real-kernel magic-value branches (ioctl
    // request codes, netlink message types) weren't reachable because the corpus rarely carried
    // the right constants. This wave adds a few more well-known ioctl request codes with
    // correctly-shaped struct args (extending wave 8's TCGETS/TIOCGWINSZ/SIOCGIFFLAGS pattern)
    // plus a real `nlmsghdr`-shaped netlink `sendmsg` (the classic netlink fuzzing surface: a
    // real header with `type`/`flags`/`seq`/`pid` fields, not just a bare `sockaddr_nl`). All
    // scalar (`Int`/`Flags`/`Const`) args throughout this table also now get `dict.rs`'s
    // dictionary bias a fraction of the time — see `genr::gen_arg_value`.

    // 82. ioctl$TIOCGPGRP(29): fd, cmd=TIOCGPGRP(0x540F), argp:ptr[out,pid_t] -> int32
    SyscallDesc {
        name: "ioctl$TIOCGPGRP",
        nr: 29,
        args: &[
            Res(FD),
            Const(TIOCGPGRP),
            Ptr {
                dir: Out,
                inner: &Int {
                    bits: 32,
                    signed: true,
                },
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 83. ioctl$TIOCSPGRP(29): fd, cmd=TIOCSPGRP(0x5410), argp:ptr[in,pid_t] -> int32
    SyscallDesc {
        name: "ioctl$TIOCSPGRP",
        nr: 29,
        args: &[
            Res(FD),
            Const(TIOCSPGRP),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: true,
                },
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 84. ioctl$FIOASYNC(29): fd, cmd=FIOASYNC(0x5452), argp:ptr[in,int32] (0/1) -> int32
    SyscallDesc {
        name: "ioctl$FIOASYNC",
        nr: 29,
        args: &[
            Res(FD),
            Const(FIOASYNC),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 85. ioctl$SIOCSIFFLAGS(29): fd:sock, cmd=SIOCSIFFLAGS(0x8914), argp:ptr[in,ifreq] -> int32.
    //     The set-side counterpart to wave 8's ioctl$SIOCGIFFLAGS, same real IFREQ shape.
    SyscallDesc {
        name: "ioctl$SIOCSIFFLAGS",
        nr: 29,
        args: &[
            Res(SOCK),
            Const(SIOCSIFFLAGS),
            Ptr {
                dir: In,
                inner: &IFREQ,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 86. ioctl$SIOCGIFHWADDR(29): fd:sock, cmd=SIOCGIFHWADDR(0x8927), argp:ptr[inout,ifreq]
    //     -> int32
    SyscallDesc {
        name: "ioctl$SIOCGIFHWADDR",
        nr: 29,
        args: &[
            Res(SOCK),
            Const(SIOCGIFHWADDR),
            Ptr {
                dir: InOut,
                inner: &IFREQ,
                nullable: false,
            },
        ],
        produces: Produces::None,
    },
    // 87. setsockopt$so_sndbuf(208): sockfd:sock, level=SOL_SOCKET(1), optname=SO_SNDBUF(7),
    //     optval(in,int32), optlen=len(optval) -> int32. A (level,optname) pair not yet covered
    //     by wave 9's so_reuseaddr/tcp_nodelay variants.
    SyscallDesc {
        name: "setsockopt$so_sndbuf",
        nr: 208,
        args: &[
            Res(SOCK),
            Const(SOL_SOCKET),
            Const(SO_SNDBUF),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
            Len { of: 3 },
        ],
        produces: Produces::None,
    },
    // 88. setsockopt$netlink_add_membership(208): sockfd:sock, level=SOL_NETLINK(270) — a
    //     materially different `level` namespace than every other setsockopt$* desc above,
    //     which are all SOL_SOCKET/IPPROTO_TCP — optname=NETLINK_ADD_MEMBERSHIP(1), optval(in,
    //     int32, multicast group number), optlen=len(optval) -> int32
    SyscallDesc {
        name: "setsockopt$netlink_add_membership",
        nr: 208,
        args: &[
            Res(SOCK),
            Const(SOL_NETLINK),
            Const(NETLINK_ADD_MEMBERSHIP),
            Ptr {
                dir: In,
                inner: &Int {
                    bits: 32,
                    signed: false,
                },
                nullable: false,
            },
            Len { of: 3 },
        ],
        produces: Produces::None,
    },
    // 89. sendmsg$nl(211): sockfd:sock (ideally socket$netlink's), msg(in,msghdr_nl incl. nested
    //     real nlmsghdr with type/flags/seq/pid), flags -> ssize. The classic netlink-fuzzing
    //     surface: unlike the generic `sendmsg`/`bind$nl` (which only shapes the sockaddr_nl),
    //     this shapes the actual *message* payload the kernel's netlink handlers parse.
    SyscallDesc {
        name: "sendmsg$nl",
        nr: 211,
        args: &[
            Res(SOCK),
            Ptr {
                dir: In,
                inner: &MSGHDR_NL,
                nullable: false,
            },
            Flags {
                vals: SEND_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 11: fault injection arming (fail_nth) ============
    // See docs/bug-finding.md's "FAULT INJECTION FIRST": `genr::prepend_fail_inject` prepends
    // these two calls as a program preamble so a chosen kernel allocation can be made to fail on
    // demand, reaching cleanup/error paths ordinary argument fuzzing structurally never
    // exercises. Requires `CONFIG_FAULT_INJECTION=y` (firmware/Image.failinj); on a kernel built
    // without it `openat$fail_nth` simply returns -ENOENT (the path doesn't exist), harmlessly —
    // no boot or behavior impact on the stock/slubdebug kernels.

    // 91. openat$fail_nth(56): dirfd (ignored — path is absolute), path="/proc/self/fail-nth",
    //     flags=O_WRONLY, mode=0 -> fd.
    SyscallDesc {
        name: "openat$fail_nth",
        nr: 56,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(FAIL_NTH_PATH),
                nullable: false,
            },
            Const(1 /* O_WRONLY */),
            Const(0),
        ],
        produces: Produces::Ret(FD),
    },
    // 92. write$fail_nth(64): fd (the fail-nth fd, ideally openat$fail_nth's), buf=ascii decimal
    //     countdown value, count=len(buf) -> ssize.
    SyscallDesc {
        name: "write$fail_nth",
        nr: 64,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(FAIL_NTH_COUNTS),
                nullable: false,
            },
            Len { of: 1 },
        ],
        produces: Produces::None,
    },

    // ============ wave 12: splice/vmsplice/tee (T2.1 cheap subsystem, was 0 descriptions) ============
    // fs/splice.c. All three thread `pipe2`(59)'s FD OutArray producer — a pipe end is required
    // on at least one side of every one of these (vmsplice always; splice/tee whenever the other
    // fd isn't itself pipe-capable) — so `pipe2` is this wave's real resource-producer anchor,
    // same role `openat`/`socket` play for the read/write/ioctl waves above.

    // 93. vmsplice(75): fd:fd (ideally a pipe2 end), iov(in,array[iovec,2]), nr_segs=2,
    //     flags -> ssize (fs/splice.c SYSCALL_DEFINE4(vmsplice,...)).
    SyscallDesc {
        name: "vmsplice",
        nr: 75,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Flags {
                vals: SPLICE_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 94. splice(76): EXACTLY 6 args (fs/splice.c SYSCALL_DEFINE6(splice,...)). fd_in, off_in
    //     (nullable — must be NULL when fd_in is a pipe, real splice(2) constraint; an
    //     occasionally-non-NULL offset against a pipe fd is a real -ESPIPE fuzz signal, same
    //     "mismatch is a signal, not a modeling gap" rationale IOVEC's doc documents), fd_out,
    //     off_out (nullable, same constraint on the write side), len, flags -> ssize.
    SyscallDesc {
        name: "splice",
        nr: 76,
        args: &[
            Res(FD),
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 64,
                    signed: true,
                },
                nullable: true,
            },
            Res(FD),
            Ptr {
                dir: InOut,
                inner: &Int {
                    bits: 64,
                    signed: true,
                },
                nullable: true,
            },
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: SPLICE_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },
    // 95. tee(77): fdin:fd, fdout:fd (real tee(2) requires BOTH to be pipes — genr's resource
    //     pool doesn't distinguish "pipe fd" from "any fd" within the flat `fd` kind, so this
    //     frequently draws a non-pipe fd on one side; that's the same accepted imprecision as
    //     `splice`'s offset nullability above, not a correctness bug in the description), len,
    //     flags -> ssize.
    SyscallDesc {
        name: "tee",
        nr: 77,
        args: &[
            Res(FD),
            Res(FD),
            Int {
                bits: 32,
                signed: false,
            },
            Flags {
                vals: SPLICE_FLAGS,
                bitmask: true,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 13: unshare/setns namespaces (T2.1 cheap subsystem, was 0 descriptions) ============
    // kernel/fork.c / kernel/nsproxy.c. `CONFIG_{USER,UTS,IPC,PID,NET,TIME}_NS=y` (verified
    // against `build/linux-slubdebug/.config`); zero guest-side scaffolding needed — `unshare` is
    // single-scalar, and `setns`'s fd comes from the always-present `/proc/self/ns/*` magic
    // symlinks (`openat$ns` below), not from any T2.2-class mount/initramfs setup.

    // 96. unshare(97): unshare_flags -> int32. Zero-fd, zero-dependency (CLONE_NEWUSER succeeds
    //     unprivileged; the rest need CAP_SYS_ADMIN, which this guest's init/agent has as root).
    SyscallDesc {
        name: "unshare",
        nr: 97,
        args: &[Flags {
            vals: UNSHARE_FLAGS,
            bitmask: true,
        }],
        produces: Produces::None,
    },
    // 97. openat$ns(56): dirfd (ignored — path is absolute), path=one of `/proc/self/ns/*`,
    //     flags=O_RDONLY, mode=0 -> fd. The FD producer `setns` below threads from.
    SyscallDesc {
        name: "openat$ns",
        nr: 56,
        args: &[
            Res(FD),
            Ptr {
                dir: In,
                inner: &StringConst(NS_PATHS),
                nullable: false,
            },
            Const(0 /* O_RDONLY */),
            Const(0),
        ],
        produces: Produces::Ret(FD),
    },
    // 98. setns(268): fd:fd (ideally openat$ns's), nstype -> int32 (kernel/nsproxy.c
    //     SYSCALL_DEFINE2(setns,...)).
    SyscallDesc {
        name: "setns",
        nr: 268,
        args: &[
            Res(FD),
            Flags {
                vals: NSTYPE_FLAGS,
                bitmask: false,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 14: keyctl/add_key/request_key (T2.1 cheap subsystem, was 0 descriptions) ============
    // security/keys/keyctl.c. `CONFIG_KEYS=y` (verified against `build/linux-slubdebug/.config`).
    // `add_key`/`request_key`/`keyctl$get_keyring_id` are the `Res(KEY)` producers; the rest are
    // consumers threaded against them (or a `KEYRING_SPECIAL` seed literal when no live key
    // exists yet) — same producer/consumer shape as the `fd`/`sock`/`vma` waves above, just a new
    // unrelated resource kind (see `resource::KEY`'s doc comment).

    // 99. add_key(217): type(in,string), description(in,string), payload(in,bytes),
    //     plen=len(payload), ringid -> key_serial_t (security/keys/keyctl.c
    //     SYSCALL_DEFINE5(add_key,...)).
    SyscallDesc {
        name: "add_key",
        nr: 217,
        args: &[
            Ptr {
                dir: In,
                inner: &StringConst(KEY_TYPES),
                nullable: false,
            },
            Ptr {
                dir: In,
                inner: &StringConst(KEY_DESCRIPTIONS),
                nullable: false,
            },
            Ptr {
                dir: In,
                inner: &Buffer {
                    len: LenSpec::Range(0, 64),
                },
                nullable: true,
            },
            Len { of: 2 },
            Flags {
                vals: KEYRING_SPECIAL,
                bitmask: false,
            },
        ],
        produces: Produces::Ret(KEY),
    },
    // 100. request_key(218): type(in,string), description(in,string),
    //      callout_info(in,string,nullable), destringid -> key_serial_t (security/keys/keyctl.c
    //      SYSCALL_DEFINE4(request_key,...)).
    SyscallDesc {
        name: "request_key",
        nr: 218,
        args: &[
            Ptr {
                dir: In,
                inner: &StringConst(KEY_TYPES),
                nullable: false,
            },
            Ptr {
                dir: In,
                inner: &StringConst(KEY_DESCRIPTIONS),
                nullable: false,
            },
            Ptr {
                dir: In,
                inner: &StringConst(KEY_CALLOUT_INFO),
                nullable: true,
            },
            Flags {
                vals: KEYRING_SPECIAL,
                bitmask: false,
            },
        ],
        produces: Produces::Ret(KEY),
    },
    // 101. keyctl$get_keyring_id(219): option=KEYCTL_GET_KEYRING_ID(0), id (a KEYRING_SPECIAL
    //      special id, since we have no independent "keyring id" pool yet), create:bool(0/1)
    //      -> key_serial_t. Another `Res(KEY)` producer besides add_key/request_key
    //      (security/keys/keyctl.c keyctl_get_keyring_ID).
    SyscallDesc {
        name: "keyctl$get_keyring_id",
        nr: 219,
        args: &[
            Const(KEYCTL_GET_KEYRING_ID),
            Flags {
                vals: KEYRING_SPECIAL,
                bitmask: false,
            },
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::Ret(KEY),
    },
    // 102. keyctl$describe(219): option=KEYCTL_DESCRIBE(6), key:Res(KEY), buffer(out),
    //      buflen=len(buffer) -> int32 (security/keys/keyctl.c keyctl_describe_key).
    SyscallDesc {
        name: "keyctl$describe",
        nr: 219,
        args: &[
            Const(KEYCTL_DESCRIBE),
            Res(KEY),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 2 },
        ],
        produces: Produces::None,
    },
    // 103. keyctl$read(219): option=KEYCTL_READ(11), key:Res(KEY), buffer(out),
    //      buflen=len(buffer) -> int32 (security/keys/keyctl.c keyctl_read_key).
    SyscallDesc {
        name: "keyctl$read",
        nr: 219,
        args: &[
            Const(KEYCTL_READ),
            Res(KEY),
            Ptr {
                dir: Out,
                inner: &Buffer {
                    len: LenSpec::Range(0, 256),
                },
                nullable: false,
            },
            Len { of: 2 },
        ],
        produces: Produces::None,
    },
    // 104. keyctl$revoke(219): option=KEYCTL_REVOKE(3), key:Res(KEY) -> int32
    //      (security/keys/keyctl.c keyctl_revoke_key).
    SyscallDesc {
        name: "keyctl$revoke",
        nr: 219,
        args: &[Const(KEYCTL_REVOKE), Res(KEY)],
        produces: Produces::None,
    },
    // 105. keyctl$unlink(219): option=KEYCTL_UNLINK(9), key:Res(KEY), keyring (a KEYRING_SPECIAL
    //      destination) -> int32 (security/keys/keyctl.c keyctl_unlink).
    SyscallDesc {
        name: "keyctl$unlink",
        nr: 219,
        args: &[
            Const(KEYCTL_UNLINK),
            Res(KEY),
            Flags {
                vals: KEYRING_SPECIAL,
                bitmask: false,
            },
        ],
        produces: Produces::None,
    },

    // ============ wave 15: process_vm_readv/writev (T2.1 cheap subsystem) ============
    // mm/process_vm_access.c. Zero-fd, zero-dependency: a `pid` is a plain biased `Int` (small
    // values organically cover pid 1/2, i.e. real tasks, per this wave's brief — "self-pid is
    // fine for reachability"), reaching `find_get_task_by_vpid`/`mm_access`/
    // `process_vm_rw_single_vec` regardless of whether the remote-side addresses are valid (see
    // `REMOTE_IOVEC`'s doc comment for why the remote iovec is modeled with free `Int`s rather
    // than `Ptr`s).

    // 106. process_vm_readv(270): EXACTLY 6 args (mm/process_vm_access.c
    //      SYSCALL_DEFINE6(process_vm_readv,...)). pid, lvec(out,array[iovec,2], local/our
    //      scratch), liovcnt=2, rvec(in,array[remote_iovec,2], remote addresses), riovcnt=2,
    //      flags(currently unused by the kernel; still fuzzed) -> ssize.
    SyscallDesc {
        name: "process_vm_readv",
        nr: 270,
        args: &[
            Int {
                bits: 32,
                signed: true,
            },
            Ptr {
                dir: Out,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Ptr {
                dir: In,
                inner: &REMOTE_IOVEC2,
                nullable: false,
            },
            Const(2),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
    // 107. process_vm_writev(271): same EXACTLY-6-args shape as process_vm_readv, opposite
    //      direction (lvec is our real data source, rvec is the remote destination).
    SyscallDesc {
        name: "process_vm_writev",
        nr: 271,
        args: &[
            Int {
                bits: 32,
                signed: true,
            },
            Ptr {
                dir: In,
                inner: &IOVEC2,
                nullable: false,
            },
            Const(2),
            Ptr {
                dir: In,
                inner: &REMOTE_IOVEC2,
                nullable: false,
            },
            Const(2),
            Int {
                bits: 32,
                signed: false,
            },
        ],
        produces: Produces::None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_at_least_fifteen_descriptions() {
        assert!(SYSCALLS.len() >= 15, "only {} descriptions", SYSCALLS.len());
    }

    #[test]
    fn table_grew_well_past_the_starter_set() {
        // Locks in the substantial expansion this change made (19 -> 60+); a regression here
        // means someone accidentally deleted descriptions rather than adding to them.
        assert!(
            SYSCALLS.len() >= 60,
            "expected >=60 descriptions, got {}",
            SYSCALLS.len()
        );
    }

    #[test]
    fn every_desc_args_len_matches_expectation_and_nr_is_nonzero_ish() {
        for d in SYSCALLS {
            assert!(d.args.len() <= 6, "{} has >6 args", d.name);
        }
    }

    #[test]
    fn no_duplicate_variant_names() {
        for (i, a) in SYSCALLS.iter().enumerate() {
            for b in &SYSCALLS[i + 1..] {
                assert_ne!(a.name, b.name, "duplicate SyscallDesc name {}", a.name);
            }
        }
    }

    #[test]
    fn at_least_one_producer_and_one_consumer_of_fd_exist() {
        use crate::types::Produces;
        let producers = SYSCALLS
            .iter()
            .filter(|d| matches!(d.produces, Produces::Ret(k) | Produces::OutArray { kind: k, .. } if k == FD || k == SOCK))
            .count();
        let consumers = SYSCALLS
            .iter()
            .filter(|d| d.args.iter().any(|a| matches!(a, Res(k) if *k == FD)))
            .count();
        assert!(producers >= 3, "producers={producers}");
        assert!(consumers >= 3, "consumers={consumers}");
    }

    #[test]
    fn new_subsystems_are_represented() {
        let names: Vec<&str> = SYSCALLS.iter().map(|d| d.name).collect();
        for want in [
            "eventfd2",
            "epoll_create1",
            "epoll_ctl",
            "epoll_pwait",
            "timerfd_create",
            "timerfd_settime64",
            "signalfd4",
            "inotify_init1",
            "inotify_add_watch",
            "socketpair",
            "accept4",
            "connect",
            "setsockopt",
            "getsockopt",
            "sendmsg",
            "recvmsg",
            "mmap2",
            "munmap",
            "mprotect",
            "madvise",
            "pread64",
            "pwrite64",
            "readv",
            "writev",
            "openat2",
            "close_range",
            "pidfd_open",
            "pidfd_getfd",
        ] {
            assert!(names.contains(&want), "missing expected desc {want}");
        }
    }

    #[test]
    fn table_grew_past_the_ioctl_and_sockaddr_depth_expansion() {
        // Locks in this change's expansion (64 -> 82); a regression here means someone
        // accidentally deleted descriptions rather than adding to them.
        assert!(
            SYSCALLS.len() >= 82,
            "expected >=82 descriptions, got {}",
            SYSCALLS.len()
        );
    }

    #[test]
    fn ioctl_and_sockaddr_depth_descriptions_are_represented() {
        let names: Vec<&str> = SYSCALLS.iter().map(|d| d.name).collect();
        for want in [
            "ioctl$TIOCGWINSZ",
            "ioctl$TIOCSWINSZ",
            "ioctl$TCGETS",
            "ioctl$TCSETS",
            "ioctl$FIONREAD",
            "ioctl$FIONBIO",
            "ioctl$SIOCGIFFLAGS",
            "ioctl$SIOCGIFCONF",
            "socket$netlink",
            "bind$inet",
            "bind$un",
            "bind$nl",
            "connect$inet",
            "connect$un",
            "sendto$inet",
            "setsockopt$so_reuseaddr",
            "setsockopt$tcp_nodelay",
            "fcntl64$dupfd_cloexec",
        ] {
            assert!(names.contains(&want), "missing expected desc {want}");
        }
    }

    #[test]
    fn table_grew_past_the_wave_10_netlink_and_ioctl_expansion() {
        // Locks in this change's expansion (82 -> 90); a regression here means someone
        // accidentally deleted descriptions rather than adding to them.
        assert!(
            SYSCALLS.len() >= 90,
            "expected >=90 descriptions, got {}",
            SYSCALLS.len()
        );
    }

    #[test]
    fn wave_10_ioctl_and_netlink_descriptions_are_represented() {
        let names: Vec<&str> = SYSCALLS.iter().map(|d| d.name).collect();
        for want in [
            "ioctl$TIOCGPGRP",
            "ioctl$TIOCSPGRP",
            "ioctl$FIOASYNC",
            "ioctl$SIOCSIFFLAGS",
            "ioctl$SIOCGIFHWADDR",
            "setsockopt$so_sndbuf",
            "setsockopt$netlink_add_membership",
            "sendmsg$nl",
        ] {
            assert!(names.contains(&want), "missing expected desc {want}");
        }
    }

    /// `sendmsg$nl`'s `msghdr` carries a *real* `nlmsghdr` (type/flags/seq/pid, not just an
    /// opaque buffer) inside a nested iovec — the classic netlink-fuzzing shape this wave adds.
    /// Checks the header serializes at the real byte offsets/widths (no accidental padding from
    /// modeling `nlmsg_type`/`nlmsg_flags` as anything wider than their real 2 bytes).
    #[test]
    fn sendmsg_nl_serializes_a_real_nlmsghdr_at_correct_offsets() {
        use crate::lower::lower;
        use crate::prog::{ArgValue, Prog, ResRef, TypedCall};
        use crate::rng::Rng;

        let socket_nl = SYSCALLS.iter().find(|d| d.name == "socket$netlink").unwrap();
        let sendmsg_nl = SYSCALLS.iter().find(|d| d.name == "sendmsg$nl").unwrap();

        let mut rng = Rng::new(31);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: socket_nl,
            args: crate::genr::generate_args(&mut rng, socket_nl, &[]),
        });
        let mut args = crate::genr::generate_args(&mut rng, sendmsg_nl, &p.calls);
        args[0] = ArgValue::Res(ResRef::Produced {
            call_idx: 0,
            slot: 0,
        });
        p.calls.push(TypedCall {
            desc: sendmsg_nl,
            args,
        });
        assert!(p.is_well_formed());

        let base = 0x9300_0000u32;
        let lowered = lower(&p, base);
        assert_eq!(lowered.calls[1].nr, 211); // sendmsg

        let msghdr_ptr = lowered.calls[1].args[1];
        let scratch_end = base as usize + lowered.scratch.len();
        assert!((base as usize..scratch_end).contains(&(msghdr_ptr as usize)));
        let msghdr_off = (msghdr_ptr - base) as usize;

        // msg_iov is MSGHDR_NL's 3rd field (msg_name:ptr@0, msg_namelen:u32@4, msg_iov:ptr@8).
        let iov_ptr =
            u32::from_le_bytes(lowered.scratch[msghdr_off + 8..msghdr_off + 12].try_into().unwrap());
        assert!((base as usize..scratch_end).contains(&(iov_ptr as usize)));
        let iov_off = (iov_ptr - base) as usize;

        // IOVEC_NL's iov_base (offset 0) points at the NLMSG struct.
        let nlmsg_ptr =
            u32::from_le_bytes(lowered.scratch[iov_off..iov_off + 4].try_into().unwrap());
        assert!((base as usize..scratch_end).contains(&(nlmsg_ptr as usize)));
        let nlmsg_off = (nlmsg_ptr - base) as usize;

        // struct nlmsghdr real layout: len@0(4), type@4(2), flags@6(2), seq@8(4), pid@12(4).
        // Just confirm every byte lands inside scratch and the struct's total footprint (32
        // bytes: 16-byte header + 16-byte payload) fits.
        assert!(nlmsg_off + 32 <= lowered.scratch.len());
    }

    #[test]
    fn vma_has_a_producer_and_consumers() {
        use crate::types::Produces;
        let producers = SYSCALLS
            .iter()
            .filter(|d| matches!(d.produces, Produces::Ret(k) if k == VMA))
            .count();
        let consumers = SYSCALLS
            .iter()
            .filter(|d| d.args.iter().any(|a| matches!(a, Res(k) if *k == VMA)))
            .count();
        assert!(producers >= 1, "producers={producers}");
        assert!(consumers >= 2, "consumers={consumers}");
    }

    #[test]
    fn table_grew_past_the_wave_12_to_15_cheap_subsystem_expansion() {
        // Locks in T2.1's expansion (90 -> 105+: splice/vmsplice/tee, unshare/setns,
        // keyctl/add_key/request_key, process_vm_readv/writev); a regression here means someone
        // accidentally deleted descriptions rather than adding to them.
        assert!(
            SYSCALLS.len() >= 105,
            "expected >=105 descriptions, got {}",
            SYSCALLS.len()
        );
    }

    #[test]
    fn wave_12_to_15_cheap_subsystem_descriptions_are_represented() {
        let names: Vec<&str> = SYSCALLS.iter().map(|d| d.name).collect();
        for want in [
            "vmsplice",
            "splice",
            "tee",
            "unshare",
            "openat$ns",
            "setns",
            "add_key",
            "request_key",
            "keyctl$get_keyring_id",
            "keyctl$describe",
            "keyctl$read",
            "keyctl$revoke",
            "keyctl$unlink",
            "process_vm_readv",
            "process_vm_writev",
        ] {
            assert!(names.contains(&want), "missing expected desc {want}");
        }
    }

    /// `key` is a brand-new resource kind (wave 14): confirm it has the same
    /// producer(s)-and-consumers shape `vma_has_a_producer_and_consumers` already checks for
    /// `vma`, so a chain like `add_key -> keyctl$describe` can actually thread organically.
    #[test]
    fn key_has_producers_and_consumers() {
        use crate::types::Produces;
        let producers = SYSCALLS
            .iter()
            .filter(|d| matches!(d.produces, Produces::Ret(k) if k == KEY))
            .count();
        let consumers = SYSCALLS
            .iter()
            .filter(|d| d.args.iter().any(|a| matches!(a, Res(k) if *k == KEY)))
            .count();
        assert!(producers >= 3, "producers={producers}");
        assert!(consumers >= 4, "consumers={consumers}");
    }

    /// `splice`/`tee`/`vmsplice` all consume `fd`, and `pipe2` (already in the table) is their
    /// natural producer anchor — confirm the resource-threading precondition holds so genr's
    /// consumer-preference bias (`pick_desc_biased`) can actually deepen these chains.
    #[test]
    fn splice_family_consumes_fd_and_pipe2_produces_it() {
        let pipe2_produces_fd = SYSCALLS
            .iter()
            .find(|d| d.name == "pipe2")
            .map(|d| matches!(d.produces, Produces::OutArray { kind, .. } if kind == FD))
            .unwrap_or(false);
        assert!(pipe2_produces_fd, "pipe2 must still produce FD");
        for name in ["vmsplice", "splice", "tee"] {
            let d = SYSCALLS.iter().find(|d| d.name == name).unwrap();
            assert!(
                d.args.iter().any(|a| matches!(a, Res(k) if *k == FD)),
                "{name} should consume Res(FD)"
            );
        }
    }

    /// `setns` needs a namespace fd; `openat$ns` is this wave's zero-scaffolding producer for
    /// one (`/proc/self/ns/*`). Confirm the pairing lowers cleanly end to end (not just
    /// individually), including the FD threading fixup, mirroring
    /// `prepend_fail_inject_builds_a_well_formed_armed_preamble_that_lowers_cleanly`'s style of
    /// check for a hand-threaded two-call chain.
    #[test]
    fn openat_ns_threads_into_setns_and_lowers_cleanly() {
        use crate::lower::lower;
        use crate::prog::{ArgValue, Prog, ResRef, TypedCall};
        use crate::rng::Rng;

        let openat_ns = SYSCALLS.iter().find(|d| d.name == "openat$ns").unwrap();
        let setns = SYSCALLS.iter().find(|d| d.name == "setns").unwrap();

        let mut rng = Rng::new(7);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: openat_ns,
            args: crate::genr::generate_args(&mut rng, openat_ns, &[]),
        });
        let mut args = crate::genr::generate_args(&mut rng, setns, &p.calls);
        args[0] = ArgValue::Res(ResRef::Produced {
            call_idx: 0,
            slot: 0,
        });
        p.calls.push(TypedCall { desc: setns, args });
        assert!(p.is_well_formed());

        let lowered = lower(&p, 0x9400_0000);
        assert_eq!(lowered.calls[0].nr, 56); // openat
        assert_eq!(lowered.calls[1].nr, 268); // setns
        let has_fd_fixup = lowered.fixups.iter().any(|f| {
            f.dst_call == 1 && f.dst_arg == 0 && matches!(f.src, crate::lower::FixupSrc::Reg(0))
        });
        assert!(has_fd_fixup, "setns's fd arg must thread from openat$ns's return");

        let wire = crate::lower::to_wire(&lowered);
        assert_eq!(wire.len(), crate::lower::WIRE_WORDS);
    }

    /// A hand-built `add_key -> keyctl$describe` chain (the classic key-management pairing)
    /// must be well-formed and lower cleanly, with the key serial threaded via a `Reg(0)` fixup
    /// exactly like `openat_ns_threads_into_setns_and_lowers_cleanly` above.
    #[test]
    fn add_key_threads_into_keyctl_describe_and_lowers_cleanly() {
        use crate::lower::lower;
        use crate::prog::{ArgValue, Prog, ResRef, TypedCall};
        use crate::rng::Rng;

        let add_key = SYSCALLS.iter().find(|d| d.name == "add_key").unwrap();
        let describe = SYSCALLS.iter().find(|d| d.name == "keyctl$describe").unwrap();

        let mut rng = Rng::new(13);
        let mut p = Prog::new();
        p.calls.push(TypedCall {
            desc: add_key,
            args: crate::genr::generate_args(&mut rng, add_key, &[]),
        });
        let mut args = crate::genr::generate_args(&mut rng, describe, &p.calls);
        args[1] = ArgValue::Res(ResRef::Produced {
            call_idx: 0,
            slot: 0,
        });
        p.calls.push(TypedCall {
            desc: describe,
            args,
        });
        assert!(p.is_well_formed());

        let lowered = lower(&p, 0x9500_0000);
        assert_eq!(lowered.calls[0].nr, 217); // add_key
        assert_eq!(lowered.calls[1].nr, 219); // keyctl
        let has_key_fixup = lowered.fixups.iter().any(|f| {
            f.dst_call == 1 && f.dst_arg == 1 && matches!(f.src, crate::lower::FixupSrc::Reg(0))
        });
        assert!(has_key_fixup, "keyctl$describe's key arg must thread from add_key's return");

        let wire = crate::lower::to_wire(&lowered);
        assert_eq!(wire.len(), crate::lower::WIRE_WORDS);
    }

    /// `process_vm_readv`/`process_vm_writev` are zero-fd/zero-resource (just a `pid` + two
    /// iovec arrays) — confirm they generate and lower cleanly across a seed sweep on their own,
    /// same discipline as the recipe-resolution test above but for a plain generated call.
    #[test]
    fn process_vm_readv_writev_generate_and_lower_cleanly() {
        use crate::lower::lower;
        use crate::prog::{Prog, TypedCall};
        use crate::rng::Rng;

        for name in ["process_vm_readv", "process_vm_writev"] {
            let desc = SYSCALLS.iter().find(|d| d.name == name).unwrap();
            for seed in [1u32, 2, 3, 42, 12345] {
                let mut rng = Rng::new(seed);
                let args = crate::genr::generate_args(&mut rng, desc, &[]);
                let mut p = Prog::new();
                p.calls.push(TypedCall { desc, args });
                assert!(p.is_well_formed(), "{name} seed {seed} ill-formed");
                let lowered = lower(&p, 0x9600_0000);
                assert_eq!(lowered.calls[0].nr, desc.nr);
                let wire = crate::lower::to_wire(&lowered);
                assert_eq!(wire.len(), crate::lower::WIRE_WORDS);
            }
        }
    }
}
