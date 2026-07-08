/* fuzzsoft guest fuzzing agent (runs as PID 1 / init).
 *
 * Snapshot-fuzz protocol via reserved "hypercall" ecalls (a7 == HC_EID), which the emulator
 * intercepts. Programs are syscall *sequences*: the emulator writes a program into `prog` (laid
 * out as [count][nr,a0..a5]*) at each reset, and the agent interprets it, then signals DONE.
 *   - hypercall(SNAPSHOT, prog, scratch): golden snapshot captured just after this; the emulator
 *     learns the program and scratch buffer addresses.
 *   - the interpret loop runs each syscall in the freshly-written program.
 *   - hypercall(DONE): emulator records coverage/crashes and resets to the snapshot.
 *
 * Build: clang --target=riscv32 -march=rv32ima -mabi=ilp32 -static -nostdlib -fuse-ld=lld -O2
 */

#define HC_EID 0x0A550000
#define HC_SNAPSHOT 0
#define HC_DONE 1
#define SYS_write 64
#define MAX_CALLS 8

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

/* Program buffer ([count][nr,a0..a5]*) and a data buffer for pointer args. Both are handed to
 * the emulator, which writes the program here and passes `scratch` as a valid pointer argument. */
static volatile unsigned prog[1 + MAX_CALLS * 7];
static char scratch[4096] __attribute__((aligned(64)));

void _start(void) {
    print("\n=== fuzzsoft agent: userspace ready, snapshot syscall-sequence fuzzing ===\n");

    /* Fault in the buffers (Linux demand-pages .bss) so the emulator can translate them at
     * snapshot time and write programs into them. */
    for (unsigned i = 0; i < sizeof(prog) / 4; i++) prog[i] = 0;
    for (unsigned i = 0; i < sizeof(scratch); i += 4096) scratch[i] = 0;
    scratch[sizeof(scratch) - 1] = 0;

    for (;;) {
        hypercall(HC_SNAPSHOT, (long)prog, (long)scratch);
        unsigned n = prog[0];
        if (n > MAX_CALLS) n = MAX_CALLS;
        for (unsigned i = 0; i < n; i++) {
            volatile unsigned *c = &prog[1 + i * 7];
            do_syscall(c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        }
        hypercall(HC_DONE, 0, 0);
    }
}
