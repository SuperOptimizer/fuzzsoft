#!/usr/bin/env bash
# Build a THIRD RV32 Linux kernel Image with a deliberately-planted, clearly
# fenced heap out-of-bounds write in memfd_create(), on top of the same
# CONFIG_SLUB_DEBUG_ON kernel-side heap debugging used by
# scripts/build-slubdebug-kernel.sh. This exists purely to VALIDATE the
# fuzzer's crash oracle: prove that a real, controlled kernel heap corruption
# is actually detected end-to-end (SLUB_DEBUG report on the console ->
# kernel_crash_sig() scan -> "[KERNEL CRASH]"), since the fuzzer has so far
# only ever run against clean kernels (0 crashes, which proves nothing about
# detection).
#
# Produces:
#   firmware/Image.buggy
#   firmware/System.map.buggy
#
# Does NOT touch firmware/Image, firmware/Image.slubdebug, System.map*,
# build/linux-src, build/linux-slubdebug-src, crates/, boot/, or Cargo files.
# The bug lives ONLY in build/linux-buggy-src, applied from the reviewable,
# reproducible patch scripts/planted-bug.patch.
#
# THIS IMAGE IS FOR ORACLE VALIDATION ONLY. It is never a real fuzzing
# target — it contains a deliberate, unconditional kernel heap bug.
#
# Why a *third* clean worktree (build/linux-buggy-src) instead of building
# straight out of build/linux-src or reusing build/linux-slubdebug-src: same
# reasoning as build-slubdebug-kernel.sh — Kbuild's O= build refuses to run
# against a source tree that already holds an in-tree build, and we must not
# mrproper (or otherwise mutate) either existing worktree. A dedicated third
# worktree also keeps the planted bug source-level isolated from both the
# stock and the slub_debug-only trees.
set -euo pipefail

ROOT="/home/forrest/fuzzsoft"
KERNEL_REPO="${ROOT}/linux"                  # main kernel git repo
STOCK_SRC="${ROOT}/build/linux-src"          # existing stock worktree (untouched)
BUGGY_SRC="${ROOT}/build/linux-buggy-src"    # clean worktree, created if missing
OUT_DIR="${ROOT}/build/linux-buggy"          # O= out-of-tree build dir
PATCH="${ROOT}/scripts/planted-bug.patch"
BUILD_LOG="${ROOT}/build/kernel-buggy-build.log"
NPROC="$(nproc)"

echo "== build-buggy-kernel: using $NPROC parallel jobs =="

# 1. Make sure we have a clean third worktree of the kernel source, checked
#    out at the same commit as the stock build/linux-src worktree.
if [ ! -d "$BUGGY_SRC" ]; then
    if [ ! -d "$STOCK_SRC" ]; then
        echo "error: $STOCK_SRC does not exist; cannot determine kernel commit" >&2
        exit 1
    fi
    commit="$(git -C "$STOCK_SRC" rev-parse HEAD)"
    echo "== adding clean worktree $BUGGY_SRC @ $commit =="
    # NOTE: pass an absolute path here. `git -C "$KERNEL_REPO" worktree add
    # build/linux-buggy-src ...` would resolve the relative path against
    # $KERNEL_REPO (i.e. linux/build/linux-buggy-src), not $ROOT/build/ —
    # this bit us once during development.
    git -C "$KERNEL_REPO" worktree add --detach "$BUGGY_SRC" "$commit"
else
    echo "== reusing existing worktree $BUGGY_SRC =="
fi

# 2. Apply the planted-bug patch if not already applied (idempotent: skip if
#    the fenced marker is already present in the tree).
if grep -q "FUZZSOFT PLANTED BUG" "$BUGGY_SRC/mm/memfd.c" 2>/dev/null; then
    echo "== planted bug already present in $BUGGY_SRC/mm/memfd.c =="
else
    echo "== applying $PATCH =="
    git -C "$BUGGY_SRC" apply "$PATCH"
fi
grep -q "FUZZSOFT PLANTED BUG" "$BUGGY_SRC/mm/memfd.c" || {
    echo "error: planted bug marker missing from $BUGGY_SRC/mm/memfd.c after apply" >&2
    exit 1
}

mkdir -p "$OUT_DIR"

# 3. Seed the O= .config from the stock .config (same base kernel config,
#    including CONFIG_INITRAMFS_SOURCE pointing at boot/initramfs.spec so the
#    fuzzer's snapshot/hypercall agent (build/agent, packed as /init) still
#    runs), then run olddefconfig once to normalize it for this tree.
if [ ! -f "$OUT_DIR/.config" ]; then
    echo "== seeding $OUT_DIR/.config from $STOCK_SRC/.config =="
    cp "$STOCK_SRC/.config" "$OUT_DIR/.config"
fi

make -C "$BUGGY_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig

# 4. Enable kernel-side SLUB debugging: CONFIG_SLUB_DEBUG_ON turns on
#    red-zoning + poisoning + sanity checks by default for every slab cache
#    (equivalent to booting stock SLUB_DEBUG=y with slub_debug= on the
#    cmdline, but on unconditionally with no cmdline dependency). This is
#    what makes the planted overwrite actually get caught and reported.
"$BUGGY_SRC/scripts/config" --file "$OUT_DIR/.config" -e SLUB_DEBUG -e SLUB_DEBUG_ON

# Re-normalize after editing, then verify the edits actually stuck (olddefconfig
# can silently drop an option if something else deselected its dependency).
make -C "$BUGGY_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig

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

# 5. Full build.
echo "== building Image (this is a full from-scratch build, several minutes) =="
make -C "$BUGGY_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 -j"$NPROC" Image 2>&1 | tee "$BUILD_LOG"

IMAGE="$OUT_DIR/arch/riscv/boot/Image"
SYSMAP="$OUT_DIR/System.map"
[ -f "$IMAGE" ] || { echo "error: $IMAGE missing after build" >&2; exit 1; }
[ -f "$SYSMAP" ] || { echo "error: $SYSMAP missing after build" >&2; exit 1; }

# 6. Publish under the new firmware/*.buggy names. Never touch
#    firmware/Image, firmware/Image.slubdebug, or System.map* (the other two
#    fuzzer kernels).
cp "$IMAGE" "$ROOT/firmware/Image.buggy"
cp "$SYSMAP" "$ROOT/firmware/System.map.buggy"

echo "== done =="
ls -la "$ROOT/firmware/Image.buggy" "$ROOT/firmware/System.map.buggy"
echo "Smoke-test with (should report [KERNEL CRASH] with a SLUB Redzone report within a few hundred cases):"
echo "  ${ROOT}/target/release/fuzzsoft fuzz --cases 400 --seed 1 --kernel ${ROOT}/firmware/Image.buggy"
