//! End-to-end tests: generation -> mutation -> lowering, exercising the parts that matter most
//! for the fuzzer (resource threading through the fixup table, buffer/length round-tripping,
//! and the exact wire encoding the guest agent will read).

use fs_prog::{
    ArgValue, CALL_WORDS, FIXUP_WORDS, FixupSrc, MAX_CALLS, MAX_FIXUPS, Prog, ResRef, Rng,
    SYSCALLS, TypedCall, WIRE_WORDS, generate, kind_compat, lower, mutate, to_wire,
};

fn desc(name: &str) -> &'static fs_prog::SyscallDesc {
    SYSCALLS.iter().find(|d| d.name == name).unwrap()
}

/// Build `openat -> read -> close`, threading the fd openat produces through both consumers,
/// entirely by hand (no reliance on the generator's randomness) so the lowering assertions
/// below are unambiguous.
fn build_open_read_close() -> Prog {
    let openat = desc("openat");
    let read = desc("read");
    let close = desc("close");

    let mut rng = Rng::new(7);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: openat,
        args: fs_prog::genr::generate_args(&mut rng, openat, &[]),
    });

    let mut read_args = fs_prog::genr::generate_args(&mut rng, read, &p.calls);
    read_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    }); // fd <- openat's return
    // Force a known buffer length so we can assert the Len arg lowers correctly.
    read_args[1] = ArgValue::Ptr(Box::new(ArgValue::Bytes(vec![0u8; 37])));
    read_args[2] = ArgValue::Imm(37);
    p.calls.push(TypedCall {
        desc: read,
        args: read_args,
    });

    let mut close_args = fs_prog::genr::generate_args(&mut rng, close, &p.calls);
    close_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    }); // same fd
    p.calls.push(TypedCall {
        desc: close,
        args: close_args,
    });

    assert!(p.is_well_formed());
    p
}

#[test]
fn resource_threading_produces_open_read_close_with_fixups() {
    let p = build_open_read_close();
    let lowered = lower(&p, 0x1000_0000);

    assert_eq!(lowered.calls.len(), 3);
    assert_eq!(lowered.calls[0].nr, 56); // openat
    assert_eq!(lowered.calls[1].nr, 63); // read
    assert_eq!(lowered.calls[2].nr, 57); // close

    // Both read's fd arg (slot 0) and close's fd arg (slot 0) must be fixed up from openat's
    // return value (call 0), and neither placeholder should have leaked a nonzero literal.
    assert_eq!(lowered.calls[1].args[0], 0);
    assert_eq!(lowered.calls[2].args[0], 0);

    let read_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 1 && f.dst_arg == 0)
        .expect("read's fd arg must have a fixup");
    assert_eq!(read_fixup.source_call_index, 0);
    assert_eq!(read_fixup.source_slot, 0);
    assert_eq!(read_fixup.src, FixupSrc::Reg(0));

    let close_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 2 && f.dst_arg == 0)
        .expect("close's fd arg must have a fixup");
    assert_eq!(close_fixup.source_call_index, 0);
    assert_eq!(close_fixup.src, FixupSrc::Reg(0));

    // read's count arg (Len{of:1}) must have lowered to the buffer's actual length (37).
    assert_eq!(lowered.calls[1].args[2], 37);

    // read's buf pointer must land inside the scratch region we handed lower() as base.
    let buf_ptr = lowered.calls[1].args[1];
    assert!(buf_ptr >= 0x1000_0000 && buf_ptr < 0x1000_0000 + lowered.scratch.len() as u32);
}

#[test]
fn pipe2_out_array_threads_two_fds_via_mem_fixups() {
    let pipe2 = desc("pipe2");
    let read = desc("read");
    let write = desc("write");

    let mut rng = Rng::new(3);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: pipe2,
        args: fs_prog::genr::generate_args(&mut rng, pipe2, &[]),
    });

    let mut write_args = fs_prog::genr::generate_args(&mut rng, write, &p.calls);
    write_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 1,
    }); // write end
    p.calls.push(TypedCall {
        desc: write,
        args: write_args,
    });

    let mut read_args = fs_prog::genr::generate_args(&mut rng, read, &p.calls);
    read_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    }); // read end
    p.calls.push(TypedCall {
        desc: read,
        args: read_args,
    });

    assert!(p.is_well_formed());
    let lowered = lower(&p, 0x2000_0000);

    let write_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 1 && f.dst_arg == 0)
        .unwrap();
    let read_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 2 && f.dst_arg == 0)
        .unwrap();

    // Both come from call 0's OutArray, at different byte offsets (slot 1 vs slot 0), i.e. Mem
    // fixups 4 bytes apart, not Reg fixups (pipe2 has no a0 resource — it's Produces::OutArray).
    match (write_fixup.src, read_fixup.src) {
        (FixupSrc::Mem(w_off), FixupSrc::Mem(r_off)) => assert_eq!(w_off, r_off + 4),
        other => panic!("expected two Mem fixups 4 bytes apart, got {other:?}"),
    }
    assert_eq!(write_fixup.source_slot, 1);
    assert_eq!(read_fixup.source_slot, 0);
}

#[test]
fn wire_encoding_matches_doc_layout() {
    let p = build_open_read_close();
    let lowered = lower(&p, 0x1000_0000);
    let wire = to_wire(&lowered);

    // §4: prog[0] = n
    assert_eq!(wire[0], 3);
    // §4: prog[1..1+MAX_CALLS*7) = MAX_CALLS call-slots, 7 words each: nr, a0..a5
    assert_eq!(wire[1], 56); // openat's nr
    assert_eq!(wire[1 + CALL_WORDS], 63); // read's nr
    assert_eq!(wire[1 + 2 * CALL_WORDS], 57); // close's nr
    // §4: prog[1+MAX_CALLS*7] = nfix
    let fixup_base = 1 + MAX_CALLS * CALL_WORDS;
    assert_eq!(wire[fixup_base], lowered.fixups.len() as u32);
    // §4: total word count is exactly 186 for MAX_CALLS=8, MAX_FIXUPS=32
    assert_eq!(wire.len(), 186);

    // Spot-check one fixup slot's encoding: dst_call, dst_arg, src_kind(0=Reg,1=Mem), src_val.
    let f = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 1 && f.dst_arg == 0)
        .unwrap();
    let slot = lowered
        .fixups
        .iter()
        .position(|x| std::ptr::eq(x, f))
        .unwrap();
    let base = fixup_base + 1 + slot * FIXUP_WORDS;
    assert_eq!(wire[base], 1); // dst_call
    assert_eq!(wire[base + 1], 0); // dst_arg
    assert_eq!(wire[base + 2], 0); // src_kind = Reg
    assert_eq!(wire[base + 3], 0); // src_val = call_idx 0
}

#[test]
fn generation_is_well_formed_across_many_seeds() {
    for seed in 1..500u32 {
        let mut rng = Rng::new(seed);
        let p = generate(&mut rng);
        assert!(p.is_well_formed(), "seed {seed}");
        assert!(p.calls.len() <= MAX_CALLS);
        let _ = lower(&p, 0x8000_0000); // must not panic on any generated program
    }
}

#[test]
fn mutation_chain_stays_well_formed_and_lowerable() {
    let mut rng = Rng::new(42);
    let mut p = generate(&mut rng);
    for _ in 0..2000 {
        p = mutate(&mut rng, &p);
        assert!(p.is_well_formed());
        assert!(!p.calls.is_empty());
        assert!(p.calls.len() <= MAX_CALLS);
        let _ = lower(&p, 0x8000_0000); // must not panic across a long mutation chain
    }
}

#[test]
fn repeated_mutation_eventually_threads_a_resource() {
    // Starting from a single resource-consuming call with only a Seed reference, mutation
    // (via the "wire two calls together" move) should, across enough tries, end up with a
    // real Produced{..} reference at least once.
    let close = desc("close");
    let mut rng = Rng::new(99);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: close,
        args: vec![ArgValue::Res(ResRef::Seed(-1))],
    });

    let mut saw_produced = false;
    for _ in 0..500 {
        p = mutate(&mut rng, &p);
        assert!(p.is_well_formed());
        if p.calls.iter().any(|c| {
            c.args
                .iter()
                .any(|a| matches!(a, ArgValue::Res(ResRef::Produced { .. })))
        }) {
            saw_produced = true;
            break;
        }
    }
    assert!(
        saw_produced,
        "500 mutations never threaded a resource reference"
    );
}

#[test]
fn kind_compat_allows_sock_to_satisfy_fd_consumers_end_to_end() {
    assert!(kind_compat(fs_prog::FD, fs_prog::SOCK));
    let socket = desc("socket");
    let close = desc("close");
    let mut rng = Rng::new(5);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: socket,
        args: fs_prog::genr::generate_args(&mut rng, socket, &[]),
    });
    let mut close_args = fs_prog::genr::generate_args(&mut rng, close, &p.calls);
    close_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    });
    p.calls.push(TypedCall {
        desc: close,
        args: close_args,
    });
    assert!(p.is_well_formed());
}

/// `socketpair` is the second `Produces::OutArray` producer in the table (after `pipe2`), and
/// it produces `SOCK` (not plain `FD`) — checks the OutArray Mem-fixup path also works for a
/// resource kind that has a subtyping parent.
#[test]
fn socketpair_out_array_threads_two_socks_via_mem_fixups() {
    let socketpair = desc("socketpair");
    let shutdown = desc("shutdown");
    let setsockopt = desc("setsockopt");

    let mut rng = Rng::new(17);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: socketpair,
        args: fs_prog::genr::generate_args(&mut rng, socketpair, &[]),
    });

    let mut shutdown_args = fs_prog::genr::generate_args(&mut rng, shutdown, &p.calls);
    shutdown_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    });
    p.calls.push(TypedCall {
        desc: shutdown,
        args: shutdown_args,
    });

    let mut setsockopt_args = fs_prog::genr::generate_args(&mut rng, setsockopt, &p.calls);
    setsockopt_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 1,
    });
    p.calls.push(TypedCall {
        desc: setsockopt,
        args: setsockopt_args,
    });

    assert!(p.is_well_formed());
    let lowered = lower(&p, 0x3000_0000);

    let shutdown_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 1 && f.dst_arg == 0)
        .unwrap();
    let setsockopt_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 2 && f.dst_arg == 0)
        .unwrap();
    match (shutdown_fixup.src, setsockopt_fixup.src) {
        (FixupSrc::Mem(a), FixupSrc::Mem(b)) => assert_eq!(b, a + 4),
        other => panic!("expected two Mem fixups 4 bytes apart, got {other:?}"),
    }
    assert_eq!(shutdown_fixup.source_slot, 0);
    assert_eq!(setsockopt_fixup.source_slot, 1);
}

/// `sendmsg`'s `msghdr` contains a *nested* `Ptr` field (`msg_iov`) inside a top-level `Struct`
/// arg — this exercises the `lower::build_bytes` extension that lets a struct field itself be a
/// pointer into scratch (needed for real `msghdr`/`iovec` shapes; without it `msg_iov` would
/// silently lower to a dangling/zero pointer instead of a real scratch address).
#[test]
fn sendmsg_nested_iovec_pointer_lowers_inside_scratch() {
    let socket = desc("socket");
    let sendmsg = desc("sendmsg");

    let mut rng = Rng::new(41);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: socket,
        args: fs_prog::genr::generate_args(&mut rng, socket, &[]),
    });
    let mut args = fs_prog::genr::generate_args(&mut rng, sendmsg, &p.calls);
    args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    });
    p.calls.push(TypedCall {
        desc: sendmsg,
        args,
    });
    assert!(p.is_well_formed());

    let base = 0x4000_0000u32;
    let lowered = lower(&p, base);
    let scratch_end = base as usize + lowered.scratch.len();

    let msghdr_ptr = lowered.calls[1].args[1];
    assert!((base as usize..scratch_end).contains(&(msghdr_ptr as usize)));

    // msg_iov is MSGHDR's 3rd field (msg_name:ptr@0, msg_namelen:u32@4, msg_iov:ptr@8), all
    // 4-byte scalars/pointers with no padding.
    let msghdr_off = (msghdr_ptr - base) as usize;
    let iov_ptr_bytes: [u8; 4] = lowered.scratch[msghdr_off + 8..msghdr_off + 12]
        .try_into()
        .unwrap();
    let iov_ptr = u32::from_le_bytes(iov_ptr_bytes);

    assert_ne!(iov_ptr, 0, "msg_iov is nullable:false, must not lower to NULL");
    assert!(
        (base as usize..scratch_end).contains(&(iov_ptr as usize)),
        "msg_iov ({iov_ptr:#x}) must point inside scratch ({base:#x}..{scratch_end:#x})"
    );
    // build_bytes serializes a nested Ptr's pointee *before* bump-allocating the enclosing
    // struct's own bytes, so the nested iovec must land at a strictly lower scratch offset.
    assert!(iov_ptr < msghdr_ptr);
}

/// `mmap2 -> madvise -> munmap`: `VMA` is a resource kind with no subtyping parent (unlike
/// `SOCK`), so this is the from-scratch check that `Res(VMA)` threading produces `Reg` fixups
/// exactly like `Res(FD)` does.
#[test]
fn mmap2_produces_vma_threaded_into_madvise_and_munmap() {
    let mmap2 = desc("mmap2");
    let madvise = desc("madvise");
    let munmap = desc("munmap");

    let mut rng = Rng::new(2);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: mmap2,
        args: fs_prog::genr::generate_args(&mut rng, mmap2, &[]),
    });
    let mut madvise_args = fs_prog::genr::generate_args(&mut rng, madvise, &p.calls);
    madvise_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    });
    p.calls.push(TypedCall {
        desc: madvise,
        args: madvise_args,
    });
    let mut munmap_args = fs_prog::genr::generate_args(&mut rng, munmap, &p.calls);
    munmap_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    });
    p.calls.push(TypedCall {
        desc: munmap,
        args: munmap_args,
    });
    assert!(p.is_well_formed());

    let lowered = lower(&p, 0x5000_0000);
    assert_eq!(lowered.calls[0].nr, 222); // mmap2
    assert_eq!(lowered.calls[1].nr, 233); // madvise
    assert_eq!(lowered.calls[2].nr, 215); // munmap
    for (call_idx, arg_idx) in [(1usize, 0usize), (2, 0)] {
        let f = lowered
            .fixups
            .iter()
            .find(|f| f.dst_call as usize == call_idx && f.dst_arg as usize == arg_idx)
            .unwrap();
        assert_eq!(f.src, FixupSrc::Reg(0));
    }
}

/// `epoll_create1 -> memfd_create -> epoll_ctl(epfd, ADD, memfd)`: a single call consuming two
/// *different* earlier producers in two different arg slots simultaneously.
#[test]
fn epoll_ctl_threads_two_independent_fd_producers_in_one_call() {
    let epoll_create1 = desc("epoll_create1");
    let memfd_create = desc("memfd_create");
    let epoll_ctl = desc("epoll_ctl");

    let mut rng = Rng::new(6);
    let mut p = Prog::new();
    p.calls.push(TypedCall {
        desc: epoll_create1,
        args: fs_prog::genr::generate_args(&mut rng, epoll_create1, &[]),
    });
    p.calls.push(TypedCall {
        desc: memfd_create,
        args: fs_prog::genr::generate_args(&mut rng, memfd_create, &p.calls),
    });
    let mut ctl_args = fs_prog::genr::generate_args(&mut rng, epoll_ctl, &p.calls);
    ctl_args[0] = ArgValue::Res(ResRef::Produced {
        call_idx: 0,
        slot: 0,
    }); // epfd
    ctl_args[2] = ArgValue::Res(ResRef::Produced {
        call_idx: 1,
        slot: 0,
    }); // watched fd
    p.calls.push(TypedCall {
        desc: epoll_ctl,
        args: ctl_args,
    });
    assert!(p.is_well_formed());

    let lowered = lower(&p, 0x6000_0000);
    let epfd_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 2 && f.dst_arg == 0)
        .unwrap();
    let watched_fixup = lowered
        .fixups
        .iter()
        .find(|f| f.dst_call == 2 && f.dst_arg == 2)
        .unwrap();
    assert_eq!(epfd_fixup.src, FixupSrc::Reg(0));
    assert_eq!(watched_fixup.src, FixupSrc::Reg(1));
}

/// Broad sweep: every description in the (now 60+ entry) table gets generated at least once
/// across many seeds, and `lower()`/`to_wire()` never panics and always yields exactly
/// `WIRE_WORDS` words with `nfix <= MAX_FIXUPS`, no matter which new descriptions a program
/// happens to contain.
#[test]
fn every_description_gets_generated_and_lowers_cleanly_across_many_seeds() {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    for seed in 1..4000u32 {
        let mut rng = Rng::new(seed);
        let p = generate(&mut rng);
        for c in &p.calls {
            seen.insert(c.desc.name);
        }
        let lowered = lower(&p, 0x8000_0000);
        let wire = to_wire(&lowered);
        assert_eq!(wire.len(), WIRE_WORDS);
        let fixup_base = 1 + MAX_CALLS * CALL_WORDS;
        let nfix = wire[fixup_base] as usize;
        assert!(nfix <= MAX_FIXUPS);
    }
    let missing: Vec<&str> = SYSCALLS
        .iter()
        .map(|d| d.name)
        .filter(|n| !seen.contains(n))
        .collect();
    assert!(missing.is_empty(), "never generated: {missing:?}");
}

/// Same sweep, but through long mutation chains starting from a fresh program each time —
/// exercises the four new mutation operators (splice/toggle/resize/interesting-int) alongside
/// the original four, across descriptions the mutator inserts via `wire_producer`/`insert_call`.
#[test]
fn mutation_chains_across_many_seeds_stay_lowerable_within_wire_limits() {
    for seed in 1..300u32 {
        let mut rng = Rng::new(seed);
        let mut p = generate(&mut rng);
        for _ in 0..300 {
            p = mutate(&mut rng, &p);
            assert!(p.is_well_formed());
            let lowered = lower(&p, 0x7000_0000);
            let wire = to_wire(&lowered);
            assert_eq!(wire.len(), WIRE_WORDS);
            let fixup_base = 1 + MAX_CALLS * CALL_WORDS;
            assert!(wire[fixup_base] as usize <= MAX_FIXUPS);
        }
    }
}
