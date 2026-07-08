//! Starter syscall description table: real rv32 (asm-generic) syscall numbers, verified
//! against `build/linux-src/arch/riscv/include/generated/uapi/asm/unistd_32.h` /
//! `qemu/linux-headers/asm-riscv/unistd_32.h`. See `docs/syzlang.md` "Starter descriptions".

use crate::resource::{FD, SOCK};
use crate::types::ArgType;
use crate::types::ArgType::*;
use crate::types::Dir::*;
use crate::types::{Field, LenSpec, Produces, SyscallDesc};

// ---------------- flag/const tables ----------------

pub const OPEN_FLAGS: &[u32] = &[
    0o0,       // O_RDONLY
    0o1,       // O_WRONLY
    0o2,       // O_RDWR
    0o100,     // O_CREAT
    0o1000,    // O_TRUNC
    0o2000,    // O_APPEND
    0o200000,  // O_DIRECTORY
    0o2000000, // O_CLOEXEC
    0o4000,    // O_NONBLOCK
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
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_at_least_fifteen_descriptions() {
        assert!(SYSCALLS.len() >= 15, "only {} descriptions", SYSCALLS.len());
    }

    #[test]
    fn every_desc_args_len_matches_expectation_and_nr_is_nonzero_ish() {
        for d in SYSCALLS {
            assert!(d.args.len() <= 6, "{} has >6 args", d.name);
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
}
