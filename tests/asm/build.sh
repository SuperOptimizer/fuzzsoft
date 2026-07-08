#!/usr/bin/env bash
# Build the clang bare-metal RV32 sample(s) into ELFs fuzzsoft can run.
set -euo pipefail
cd "$(dirname "$0")"
clang --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -nostdlib \
      -fuse-ld=lld -Wl,-Ttext=0x80000000 -o sample.elf sample.S
echo "built: $(pwd)/sample.elf"; file sample.elf

# Compressed-instruction sample (exercises the C extension decoder).
clang --target=riscv32-unknown-elf -march=rv32imac -mabi=ilp32 -Os -nostdlib \
      -fuse-ld=lld -Wl,-Ttext=0x80000000 -o sample_c.elf sample_c.c
echo "built: $(pwd)/sample_c.elf"; file sample_c.elf
