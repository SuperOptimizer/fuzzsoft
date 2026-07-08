#!/usr/bin/env bash
# Build the tiny rv32 init and pack it into an initramfs cpio (newc format).
set -euo pipefail
cd "$(dirname "$0")/.."

clang --target=riscv32 -march=rv32ima -mabi=ilp32 -static -nostdlib -fuse-ld=lld -O2 \
      -o build/init boot/init.c
clang --target=riscv32 -march=rv32ima -mabi=ilp32 -static -nostdlib -fuse-ld=lld -O2 \
      -o build/agent boot/agent.c
llvm-strip build/init build/agent 2>/dev/null || true

rm -rf build/initramfs
mkdir -p build/initramfs/dev
cp build/init build/initramfs/init
chmod +x build/initramfs/init

( cd build/initramfs && find . -print0 | cpio --null --create --format=newc --quiet ) > firmware/initramfs.cpio
echo "init: $(stat -c%s build/init) bytes; initramfs.cpio: $(stat -c%s firmware/initramfs.cpio) bytes"
file build/init
