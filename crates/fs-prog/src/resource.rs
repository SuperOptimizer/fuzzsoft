//! Resource kinds (syzkaller's `resource fd[int32]: ...`), flat single-parent subtyping,
//! and the seed-literal pool usable even with no live producer in the program.
//!
//! See `docs/syzlang.md` §1.

/// A named resource kind. Subtyping is a single flat parent link: `sock` is-a `fd`, so a
/// consumer that declares `Res(FD)` accepts values produced as FD *or* SOCK.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ResourceKind(pub &'static str);

pub const FD: ResourceKind = ResourceKind("fd");
pub const SOCK: ResourceKind = ResourceKind("sock"); // subtype of fd
pub const VMA: ResourceKind = ResourceKind("vma");
/// A `key_serial_t` (security/keys): produced by `add_key`/`request_key`/
/// `keyctl$get_keyring_id`, consumed by `keyctl$*`'s key/keyring args. Unrelated to `fd` (a key
/// serial number, not a file descriptor) — same "new unrelated resource kind" shape as `VMA`.
pub const KEY: ResourceKind = ResourceKind("key");

#[derive(Debug)]
pub struct ResourceDef {
    pub kind: ResourceKind,
    pub subtype_of: Option<ResourceKind>,
    /// Seed literals usable even with no live producer (syzkaller's `-1, AT_FDCWD`).
    pub seeds: &'static [i64],
}

pub static RESOURCES: &[ResourceDef] = &[
    ResourceDef {
        kind: FD,
        subtype_of: None,
        seeds: &[-1, -100 /* AT_FDCWD */, 0, 1, 2],
    },
    ResourceDef {
        kind: SOCK,
        subtype_of: Some(FD),
        seeds: &[-1],
    },
    ResourceDef {
        kind: VMA,
        subtype_of: None,
        seeds: &[0, -1],
    }, // 0 = let kernel pick, -1 = bad
    ResourceDef {
        kind: KEY,
        subtype_of: None,
        // uapi/linux/keyctl.h KEY_SPEC_* special keyring ids: usable as a keyring/key arg even
        // with no live add_key/request_key producer in the program.
        seeds: &[-1 /* THREAD_KEYRING */, -3 /* SESSION_KEYRING */, -4 /* USER_KEYRING */],
    },
];

/// Whether a value produced as `have` may satisfy a consumer that wants `want`.
pub fn kind_compat(want: ResourceKind, have: ResourceKind) -> bool {
    if want == have {
        return true;
    }
    RESOURCES
        .iter()
        .find(|r| r.kind == have)
        .and_then(|r| r.subtype_of)
        .is_some_and(|p| kind_compat(want, p))
}

/// Seed literals declared for `kind` (empty slice if `kind` is unknown to `RESOURCES`).
pub fn seeds_for(kind: ResourceKind) -> &'static [i64] {
    RESOURCES
        .iter()
        .find(|r| r.kind == kind)
        .map(|r| r.seeds)
        .unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sock_is_compat_with_fd_but_not_reverse() {
        assert!(kind_compat(FD, SOCK)); // a Res(FD) consumer accepts a SOCK producer
        assert!(!kind_compat(SOCK, FD)); // a Res(SOCK) consumer does NOT accept a bare FD
        assert!(kind_compat(FD, FD));
        assert!(kind_compat(SOCK, SOCK));
    }

    #[test]
    fn vma_is_unrelated_to_fd() {
        assert!(!kind_compat(FD, VMA));
        assert!(!kind_compat(VMA, FD));
    }

    #[test]
    fn key_is_unrelated_to_fd_and_vma() {
        assert!(!kind_compat(FD, KEY));
        assert!(!kind_compat(KEY, FD));
        assert!(!kind_compat(VMA, KEY));
        assert!(!kind_compat(KEY, VMA));
        assert!(kind_compat(KEY, KEY));
    }

    #[test]
    fn seeds_present_for_all_kinds() {
        assert!(!seeds_for(FD).is_empty());
        assert!(!seeds_for(SOCK).is_empty());
        assert!(!seeds_for(VMA).is_empty());
        assert!(!seeds_for(KEY).is_empty());
        assert!(seeds_for(ResourceKind("nonexistent")).is_empty());
    }
}
