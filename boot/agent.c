/* fuzzsoft guest fuzzing agent (runs as PID 1 / init).
 *
 * Snapshot-fuzz protocol via reserved "hypercall" ecalls (a7 == HC_EID), which the emulator
 * intercepts. Programs are typed syscall *sequences* with resource threading: the emulator writes
 * a program (in the fs-prog wire format) plus a scratch data image at each reset, and the agent
 * interprets it, then signals DONE.
 *   - hypercall(SNAPSHOT, prog, scratch): golden snapshot captured just after this; the emulator
 *     learns the program buffer and scratch buffer addresses.
 *   - the interpret loop applies each call's resource fixups, runs the syscall, and records the
 *     a0 return value into results[] so a later call can consume it (open->read->close chains).
 *   - hypercall(DONE): emulator records coverage/crashes and resets to the snapshot.
 *
 * Wire layout (must match crates/fs-prog/src/lower.rs exactly; see crates/fs-prog/DESIGN.md):
 *   prog[0]                                  = n     (number of calls, <= MAX_CALLS)
 *   prog[1 .. 1+MAX_CALLS*CALL_WORDS)        = MAX_CALLS call slots: nr, a0..a5
 *   prog[FIXUP_BASE]                         = nfix  (number of fixups, <= MAX_FIXUPS)
 *   prog[FIXUP_BASE+1 ..]                    = MAX_FIXUPS fixup slots: dst_call, dst_arg,
 *                                              src_kind, src_val
 * A fixup means: before running call dst_call, overwrite its dst_arg-th register with either
 *   src_kind==0 (Reg): results[src_val]                     (src_val is a call index)
 *   src_kind==1 (Mem): *(u32*)(scratch + src_val)          (src_val is a scratch byte offset)
 *
 * Build: clang --target=riscv32 -march=rv32ima -mabi=ilp32 -static -nostdlib -fuse-ld=lld -O2
 */

#define HC_EID 0x0A550000
#define HC_SNAPSHOT 0
#define HC_DONE 1
#define SYS_write 64

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

/* Program buffer (fs-prog wire form) and a scratch data buffer for pointer args. Both are handed
 * to the emulator, which writes the program/scratch here and passes `scratch` as a valid pointer
 * base. results[] holds each call's a0 so later calls can thread produced resources (fds). */
static volatile unsigned prog[WIRE_WORDS];
static char scratch[SCRATCH_SIZE] __attribute__((aligned(64)));
static unsigned results[MAX_CALLS];

void _start(void) {
    print("\n=== fuzzsoft agent: userspace ready, typed resource-threaded fuzzing ===\n");

    /* Fault in the buffers (Linux demand-pages .bss) so the emulator can translate them at
     * snapshot time and write programs/scratch into them. */
    for (unsigned i = 0; i < WIRE_WORDS; i++) prog[i] = 0;
    for (unsigned i = 0; i < SCRATCH_SIZE; i += 4096) scratch[i] = 0;
    scratch[SCRATCH_SIZE - 1] = 0;

    for (;;) {
        hypercall(HC_SNAPSHOT, (long)prog, (long)scratch);

        unsigned n = prog[0];
        if (n > MAX_CALLS) n = MAX_CALLS;
        unsigned nfix = prog[FIXUP_BASE];
        if (nfix > MAX_FIXUPS) nfix = MAX_FIXUPS;

        for (unsigned i = 0; i < n; i++) {
            volatile unsigned *c = &prog[1 + i * CALL_WORDS];
            unsigned a[6] = {c[1], c[2], c[3], c[4], c[5], c[6]};

            /* Apply all fixups targeting this call before invoking it. */
            for (unsigned f = 0; f < nfix; f++) {
                volatile unsigned *fr = &prog[FIXUP_BASE + 1 + f * FIXUP_WORDS];
                if (fr[0] != i) continue;                      /* dst_call != this call */
                unsigned v = fr[2] == 0
                                 ? results[fr[3]]              /* Reg(src_val = call idx) */
                                 : *(volatile unsigned *)(scratch + fr[3]); /* Mem(byte off) */
                if (fr[1] < 6) a[fr[1]] = v;                   /* dst_arg */
            }

            results[i] = (unsigned)do_syscall(c[0], a[0], a[1], a[2], a[3], a[4], a[5]);
        }
        hypercall(HC_DONE, 0, 0);
    }
}
