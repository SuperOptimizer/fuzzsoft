//! End-to-end tests: generation -> mutation -> lowering, exercising the parts that matter most
//! for the fuzzer (resource threading through the fixup table, buffer/length round-tripping,
//! and the exact wire encoding the guest agent will read).

use fs_prog::{
    ArgValue, CALL_WORDS, FIXUP_WORDS, FixupSrc, MAX_CALLS, Prog, ResRef, Rng, SYSCALLS, TypedCall,
    generate, kind_compat, lower, mutate, to_wire,
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
