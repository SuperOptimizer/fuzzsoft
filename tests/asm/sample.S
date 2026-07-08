# Minimal bare-metal RV32 sample for fuzzsoft M0 (clang-built test input, decision #26).
# Computes 6 * 7 = 42 (exercises the M extension) and exits via ecall a7=93.
#
# Build (see tests/asm/build.sh):
#   clang --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -nostdlib \
#         -fuse-ld=lld -Wl,-Ttext=0x80000000 -o sample.elf sample.S

.section .text
.globl _start
_start:
    li   a1, 6
    li   a2, 7
    mul  a0, a1, a2      # a0 = 42
    li   a7, 93          # __NR_exit
    ecall
