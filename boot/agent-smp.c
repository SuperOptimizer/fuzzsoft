/* fuzzsoft guest SMP fuzzing agent (T5.1c, docs/smp-design.md): a SEPARATE copy of boot/agent.c
 * (never touch that file — the default single-hart `fuzzsoft fuzz`/`smp-boot` kernel embeds
 * unmodified boot/agent.c and MUST stay byte-for-byte unregressed) that additionally spawns a
 * second, independent syscall-issuing flow so TWO harts can race real syscalls against the SAME
 * shared kernel — the whole point of SMP fuzzing (a UAF-via-race/TOCTOU needs two concurrent
 * callers to ever trigger).
 *
 * Same snapshot-fuzz protocol as boot/agent.c (see that file's header comment for the wire
 * format), just duplicated across two independent (prog, scratch, results) buffer sets — "hart A"
 * (this process, the original PID) and "hart B" (a `clone()`ed second task). Mechanism:
 *
 *   - `raw_clone(CLONE_VM | CLONE_FS | CLONE_FILES, hart_b_stack_top)`: a genuine second Linux
 *     task (its own pid, its own kernel-side `current`/task_struct — the reason this can't be
 *     faked by the emulator just repointing hart 0's PC into hart 1's already-running process:
 *     real syscalls need a real `current` for the kernel's syscall entry path to resolve the
 *     right mm/fd table/credentials) that SHARES this process's address space (CLONE_VM) and open
 *     file descriptor table (CLONE_FILES) — deliberately NOT a full CLONE_THREAD (which additionally
 *     needs CLONE_SIGHAND and shared thread-group/tid bookkeeping neither flow needs here, since
 *     nothing here signals or waits on the other).
 *   - Each flow independently calls `hypercall(HC_SNAPSHOT, own_prog, own_scratch)` as the very
 *     first thing it does after being pinned to its hart via `sched_setaffinity` (hart A -> cpu 0,
 *     hart B -> cpu 1) — matching `SnapshotSmp`'s `cpus[0]`/`cpus[1]` indexing on the fs-cli side
 *     by construction, not by discovery.
 *   - `fs_platform::run_smp`'s existing `stop_on_hypercall = false` mode (already exercised by the
 *     T5.1a mechanical tests) runs every hart independently until EACH has hit its own
 *     hypercall/halt/deadline — so ONE `run_smp(..., false)` call, with no new scheduler code,
 *     naturally waits for BOTH flows to reach their own first HC_SNAPSHOT before fs-cli captures
 *     `SnapshotSmp`. No barrier/rendezvous needed: the CLINT/mtime state is shared and the
 *     scheduler itself provides "wait for both".
 *
 * Build: clang --target=riscv32 -march=rv32ima -mabi=ilp32 -static -nostdlib -fuse-ld=lld -O2
 *        -o build/agent-smp boot/agent-smp.c
 * Packed into a DEDICATED kernel image (firmware/Image.smp, via a private initramfs spec) — never
 * firmware/Image, which stays exactly the unmodified-boot/agent.c kernel the single-hart path and
 * `smp-boot` already use.
 */

#define HC_EID 0x0A550000
#define HC_SNAPSHOT 0
#define HC_DONE 1
#define SYS_mount 40
#define SYS_write 64
#define SYS_openat 56
#define SYS_close 57
#define SYS_sched_setaffinity 122
#define SYS_clone 220
#define AT_FDCWD (-100)
#define O_WRONLY 1

#define CLONE_VM 0x00000100
#define CLONE_FS 0x00000200
#define CLONE_FILES 0x00000400

#define MAX_CALLS 8
#define MAX_FIXUPS 32
#define CALL_WORDS 7  /* nr, a0..a5 */
#define FIXUP_WORDS 4 /* dst_call, dst_arg, src_kind, src_val */
#define FIXUP_BASE (1 + MAX_CALLS * CALL_WORDS)
#define WIRE_WORDS (FIXUP_BASE + 1 + MAX_FIXUPS * FIXUP_WORDS) /* = 186 */
#define SCRATCH_SIZE (32 * 1024)

static long hypercall(long cmd, long a, long b) {
    register long a7 asm("a7") = HC_EID;
    register long a0 asm("a0") = cmd;
    register long a1 asm("a1") = a;
    register long a2 asm("a2") = b;
    asm volatile("ecall" : "+r"(a0) : "r"(a7), "r"(a1), "r"(a2) : "memory");
    return a0;
}

static long do_syscall(unsigned nr, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                       unsigned a4, unsigned a5) {
    register long r7 asm("a7") = nr;
    register long r0 asm("a0") = a0;
    register long r1 asm("a1") = a1;
    register long r2 asm("a2") = a2;
    register long r3 asm("a3") = a3;
    register long r4 asm("a4") = a4;
    register long r5 asm("a5") = a5;
    asm volatile("ecall"
                 : "+r"(r0)
                 : "r"(r7), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5)
                 : "memory");
    return r0;
}

static void print(const char *s) {
    unsigned n = 0;
    while (s[n]) n++;
    do_syscall(SYS_write, 1, (unsigned)(long)s, n, 0, 0, 0);
}

/* Same fault-injection preamble as boot/agent.c — see that file's doc comment. Run once, by hart
 * A only, before hart B is spawned (so it costs nothing extra and races nothing). */
static void try_mount(const char *dev, const char *dir, const char *type) {
    do_syscall(SYS_mount, (unsigned)(long)dev, (unsigned)(long)dir, (unsigned)(long)type, 0, 0, 0);
}

static void write_knob_zero(const char *path) {
    long fd = do_syscall(SYS_openat, (unsigned)AT_FDCWD, (unsigned)(long)path, O_WRONLY, 0, 0, 0);
    if ((int)fd < 0) return;
    do_syscall(SYS_write, (unsigned)fd, (unsigned)(long)"0", 1, 0, 0, 0);
    do_syscall(SYS_close, (unsigned)fd, 0, 0, 0, 0, 0);
}

static void arm_fault_injection_knobs(void) {
    try_mount("none", "/proc", "proc");
    try_mount("none", "/sys", "sysfs");
    try_mount("none", "/sys/kernel/debug", "debugfs");

    write_knob_zero("/sys/kernel/debug/failslab/ignore-gfp-wait");
    write_knob_zero("/sys/kernel/debug/fail_page_alloc/ignore-gfp-wait");
    write_knob_zero("/sys/kernel/debug/fail_page_alloc/ignore-gfp-highmem");
    write_knob_zero("/sys/kernel/debug/fail_page_alloc/min-order");
}

static void pin_to_cpu(unsigned cpu) {
    unsigned long mask = 1ul << cpu;
    do_syscall(SYS_sched_setaffinity, 0, (unsigned)sizeof(mask), (unsigned)(long)&mask, 0, 0, 0);
}

/* Raw clone(2) on riscv (arch/riscv selects CONFIG_CLONE_BACKWARDS, kernel/fork.c): syscall args
 * are (clone_flags, newsp, parent_tidptr, tls, child_tidptr) — NOT the libc clone(fn, stack, ...)
 * wrapper shape. Passing a nonzero `child_stack_top` gives the CHILD a fresh stack on return (both
 * flows return from the SAME `ecall`, distinguished only by the return value — 0 in the child,
 * the child's pid in the parent, a negative errno on failure — exactly like fork()). We don't
 * track parent/child tids or set up TLS: neither flow waits on or signals the other. */
static long raw_clone(unsigned long flags, void *child_stack_top) {
    register long a0 asm("a0") = (long)flags;
    register long a1 asm("a1") = (long)child_stack_top;
    register long a2 asm("a2") = 0;
    register long a3 asm("a3") = 0;
    register long a4 asm("a4") = 0;
    register long a7 asm("a7") = SYS_clone;
    asm volatile("ecall"
                 : "+r"(a0)
                 : "r"(a7), "r"(a1), "r"(a2), "r"(a3), "r"(a4)
                 : "memory");
    return a0;
}

/* Per-hart buffer set: (prog, scratch, results) — hart A's set is the original agent.c layout
 * (kept at the SAME static addresses, in the SAME declaration order, so its VA is stable and
 * documented like the original); hart B's is a second, independent, otherwise identical set. Both
 * are handed to the emulator (which writes program/scratch at each per-case reset) — see fs-cli's
 * `smp-fuzz` per-hart injection, which learns each hart's own (prog_va, scratch_va) straight from
 * that hart's own `a1`/`a2` register at its own HC_SNAPSHOT hypercall (no fixed-VA convention
 * needed on the host side — CLONE_VM makes both flows' VAs live in the SAME address space, but
 * fs-cli never has to assume or hardcode them; it just reads each cpu's own registers). */
static volatile unsigned prog[WIRE_WORDS];
static char scratch[SCRATCH_SIZE] __attribute__((aligned(64)));
static unsigned results[MAX_CALLS];

static volatile unsigned prog_b[WIRE_WORDS];
static char scratch_b[SCRATCH_SIZE] __attribute__((aligned(64)));
static unsigned results_b[MAX_CALLS];
static char hart_b_stack[16384] __attribute__((aligned(16)));

static void fuzz_loop(volatile unsigned *p, char *sc, unsigned *res) {
    for (;;) {
        hypercall(HC_SNAPSHOT, (long)p, (long)sc);

        unsigned n = p[0];
        if (n > MAX_CALLS) n = MAX_CALLS;
        unsigned nfix = p[FIXUP_BASE];
        if (nfix > MAX_FIXUPS) nfix = MAX_FIXUPS;

        for (unsigned i = 0; i < n; i++) {
            volatile unsigned *c = &p[1 + i * CALL_WORDS];
            unsigned a[6] = {c[1], c[2], c[3], c[4], c[5], c[6]};

            for (unsigned f = 0; f < nfix; f++) {
                volatile unsigned *fr = &p[FIXUP_BASE + 1 + f * FIXUP_WORDS];
                if (fr[0] != i) continue;
                unsigned v = fr[2] == 0 ? res[fr[3]] : *(volatile unsigned *)(sc + fr[3]);
                if (fr[1] < 6) a[fr[1]] = v;
            }

            res[i] = (unsigned)do_syscall(c[0], a[0], a[1], a[2], a[3], a[4], a[5]);
        }
        hypercall(HC_DONE, 0, 0);
    }
}

void _start(void) {
    print("\n=== fuzzsoft SMP agent: two-hart syscall racing, typed resource-threaded fuzzing ===\n");

    /* Fault in both harts' buffers up front (single flow, before the clone) so the emulator can
     * translate all four regions at snapshot time. */
    for (unsigned i = 0; i < WIRE_WORDS; i++) {
        prog[i] = 0;
        prog_b[i] = 0;
    }
    for (unsigned i = 0; i < SCRATCH_SIZE; i += 4096) {
        scratch[i] = 0;
        scratch_b[i] = 0;
    }
    scratch[SCRATCH_SIZE - 1] = 0;
    scratch_b[SCRATCH_SIZE - 1] = 0;
    hart_b_stack[0] = 0; /* fault in the child's stack page(s) too */

    arm_fault_injection_knobs();

    /* Spawn hart B's flow. `child == 0` in the new task; `child > 0` (its pid) in this one;
     * `child < 0` means clone() failed (e.g. a kernel/config without CLONE_VM support) — in that
     * case this degrades to running ONLY hart A's loop, which fs-cli's `smp-fuzz` detects (hart 1
     * never reaches its own HC_SNAPSHOT, budget-out) and reports rather than silently misbehaving. */
    long child = raw_clone(CLONE_VM | CLONE_FS | CLONE_FILES, hart_b_stack + sizeof(hart_b_stack));
    if (child == 0) {
        pin_to_cpu(1);
        fuzz_loop(prog_b, scratch_b, results_b);
    }
    pin_to_cpu(0);
    fuzz_loop(prog, scratch, results);
}
