/* Tiny nostdlib rv32 init for fuzzsoft (decision #31): raw ecall syscalls, no libc.
 * Prints a banner proving we reached userspace, then loops issuing getpid (a cheap
 * syscall) so the fuzzer has a live process to drive later.
 *
 * Build (see boot/build-initramfs.sh):
 *   clang --target=riscv32 -march=rv32ima -mabi=ilp32 -static -nostdlib \
 *         -fuse-ld=lld -o init boot/init.c
 */

static long syscall3(long n, long a0, long a1, long a2) {
    register long rn asm("a7") = n;
    register long ra0 asm("a0") = a0;
    register long ra1 asm("a1") = a1;
    register long ra2 asm("a2") = a2;
    asm volatile("ecall" : "+r"(ra0) : "r"(rn), "r"(ra1), "r"(ra2) : "memory");
    return ra0;
}

#define SYS_read 63
#define SYS_write 64
#define SYS_exit 93
#define SYS_getpid 172

static unsigned slen(const char *s) {
    unsigned n = 0;
    while (s[n]) n++;
    return n;
}

void _start(void) {
    const char *msg = "\n=== fuzzsoft init: reached rv32 Linux userspace! ===\n";
    syscall3(SYS_write, 1, (long)msg, slen(msg));

    /* Stay alive so PID 1 doesn't die (which would panic the kernel). */
    for (;;) {
        syscall3(SYS_getpid, 0, 0, 0);
    }
}
