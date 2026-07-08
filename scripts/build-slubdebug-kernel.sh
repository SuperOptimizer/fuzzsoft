#!/usr/bin/env bash
# Build a second RV32 Linux kernel Image with kernel-side SLUB heap debugging
# (CONFIG_SLUB_DEBUG_ON) turned on, so the *kernel itself* red-zones/poisons
# every slab object and BUG()s/oopses on real heap corruption. This is the
# kernel-cooperative alternative to emulator-side sanitizer poisoning, which
# false-positived on stock SLUB (see docs/kernel-san.md, "Experiment result").
#
# Produces:
#   firmware/Image.slubdebug
#   firmware/System.map.slubdebug
#
# Does NOT touch firmware/Image, firmware/System.map, build/linux-src, or any
# crates//boot/ files. The stock kernel used by the main fuzzer is untouched.
#
# Why a *second* kernel source worktree (build/linux-slubdebug-src) instead of
# building straight out of build/linux-src with O=...: build/linux-src already
# has an in-tree stock build (its own .config, include/config/,
# arch/riscv/include/generated/ are populated). Kbuild's own out-of-tree-build
# cleanliness check (Makefile's `outputmakefile` target) refuses to build with
# O= against a source tree that isn't pristine ("The source tree is not
# clean, please run 'make ARCH=riscv mrproper'"). We must not mrproper
# build/linux-src (that would destroy the stock build), so instead we add a
# second, clean git worktree of the same kernel repo and do the real O= build
# from there. Both worktrees point at the same commit, so this is not a
# meaningfully different kernel source, just a clean checkout to build from.
set -euo pipefail

ROOT="/home/forrest/fuzzsoft"
KERNEL_REPO="${ROOT}/linux"                          # main kernel git repo
STOCK_SRC="${ROOT}/build/linux-src"                   # existing stock worktree (untouched)
SLUBDEBUG_SRC="${ROOT}/build/linux-slubdebug-src"     # clean worktree, created if missing
OUT_DIR="${ROOT}/build/linux-slubdebug"               # O= out-of-tree build dir
BUILD_LOG="${ROOT}/build/kernel-slubdebug-build.log"
NPROC="$(nproc)"

echo "== build-slubdebug-kernel: using $NPROC parallel jobs =="

# 1. Make sure we have a clean second worktree of the kernel source, checked
#    out at the same commit as the stock build/linux-src worktree.
if [ ! -d "$SLUBDEBUG_SRC" ]; then
    if [ ! -d "$STOCK_SRC" ]; then
        echo "error: $STOCK_SRC does not exist; cannot determine kernel commit" >&2
        exit 1
    fi
    commit="$(git -C "$STOCK_SRC" rev-parse HEAD)"
    echo "== adding clean worktree $SLUBDEBUG_SRC @ $commit =="
    git -C "$KERNEL_REPO" worktree add --detach "$SLUBDEBUG_SRC" "$commit"
else
    echo "== reusing existing worktree $SLUBDEBUG_SRC =="
fi

mkdir -p "$OUT_DIR"

# 2. Seed the O= .config from the stock .config (same base kernel config,
#    including CONFIG_INITRAMFS_SOURCE pointing at boot/initramfs.spec so the
#    fuzzer's snapshot/hypercall agent (build/agent, packed as /init) still
#    runs), then run olddefconfig once to normalize it for this tree.
if [ ! -f "$OUT_DIR/.config" ]; then
    echo "== seeding $OUT_DIR/.config from $STOCK_SRC/.config =="
    cp "$STOCK_SRC/.config" "$OUT_DIR/.config"
fi

make -C "$SLUBDEBUG_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig

# 3. Enable kernel-side SLUB debugging: CONFIG_SLUB_DEBUG_ON turns on
#    red-zoning + poisoning + sanity checks by default for every slab cache
#    (equivalent to booting stock SLUB_DEBUG=y with slub_debug= on the
#    cmdline, but on unconditionally with no cmdline dependency).
"$SLUBDEBUG_SRC/scripts/config" --file "$OUT_DIR/.config" -e SLUB_DEBUG -e SLUB_DEBUG_ON

# Re-normalize after editing, then verify the edits actually stuck (olddefconfig
# can silently drop an option if something else deselected its dependency).
make -C "$SLUBDEBUG_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig

echo "== verifying config =="
grep -E '^CONFIG_SLUB_DEBUG_ON=y' "$OUT_DIR/.config" || {
    echo "error: CONFIG_SLUB_DEBUG_ON did not stick in $OUT_DIR/.config" >&2
    exit 1
}
grep -E '^CONFIG_INITRAMFS_SOURCE=' "$OUT_DIR/.config" | grep -q 'boot/initramfs.spec' || {
    echo "error: CONFIG_INITRAMFS_SOURCE lost the fuzzer's initramfs.spec path" >&2
    exit 1
}
grep -E '^CONFIG_SLUB_DEBUG_ON=y|^CONFIG_SLUB_DEBUG=y|^CONFIG_SLUB=y|^CONFIG_INITRAMFS_SOURCE=' "$OUT_DIR/.config"

# 4. Full build.
echo "== building Image (this is a full from-scratch build, several minutes) =="
make -C "$SLUBDEBUG_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 -j"$NPROC" Image 2>&1 | tee "$BUILD_LOG"

IMAGE="$OUT_DIR/arch/riscv/boot/Image"
SYSMAP="$OUT_DIR/System.map"
[ -f "$IMAGE" ] || { echo "error: $IMAGE missing after build" >&2; exit 1; }
[ -f "$SYSMAP" ] || { echo "error: $SYSMAP missing after build" >&2; exit 1; }

# 5. Publish under the new firmware/*.slubdebug names. Never touch
#    firmware/Image or firmware/System.map (the stock fuzzer kernel).
cp "$IMAGE" "$ROOT/firmware/Image.slubdebug"
cp "$SYSMAP" "$ROOT/firmware/System.map.slubdebug"

echo "== done =="
ls -la "$ROOT/firmware/Image.slubdebug" "$ROOT/firmware/System.map.slubdebug"
echo "Smoke-test with:"
echo "  ${ROOT}/target/release/fuzzsoft fuzz --cases 50 --seed 1 --kernel ${ROOT}/firmware/Image.slubdebug"
