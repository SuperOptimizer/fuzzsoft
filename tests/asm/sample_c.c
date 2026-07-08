/* Compressed-instruction validator for fuzzsoft M0/M1 (decision #26).
 * Computes sum(1..=100) = 5050 with a non-foldable loop, then exits via ecall a7=93.
 * Built with -march=rv32imac so clang emits c.* (compressed) instructions. */

void _start(void) {
    volatile int n = 100; /* volatile so the loop is not constant-folded away */
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i;
    }
    register int a0 asm("a0") = sum; /* 5050 */
    register int a7 asm("a7") = 93;  /* __NR_exit */
    asm volatile("ecall" ::"r"(a0), "r"(a7));
    for (;;) {
    }
}
