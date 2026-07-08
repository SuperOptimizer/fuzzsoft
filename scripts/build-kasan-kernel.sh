#!/usr/bin/env bash
# Attempt to build a 4th RV32 Linux kernel Image with the strongest available
# heap out-of-bounds/UAF oracle beyond CONFIG_SLUB_DEBUG_ON (which only checks
# redzone/poison at FREE time, per firmware/Image.slubdebug + docs/kernel-san.md).
# KASAN and KFENCE are the gold-standard oracles for this (they catch OOB
# *reads* and *immediate* use-after-free, not just corruption-at-free), so this
# script tries them first, in that order, and PROVES (via .config grep, not
# assumption) whether each one actually stuck.
#
# EMPIRICAL RESULT on this kernel tree (verified by running exactly the config
# steps below against build/linux-kasan-src): NEITHER CONFIG_KASAN NOR
# CONFIG_KFENCE IS AVAILABLE on rv32. This is a hard architectural gate, not a
# missed config option:
#
#   arch/riscv/Kconfig:
#       select HAVE_ARCH_KASAN        if MMU && 64BIT
#       select HAVE_ARCH_KASAN_VMALLOC if MMU && 64BIT
#       select HAVE_ARCH_KFENCE       if MMU && 64BIT
#
#   lib/Kconfig.kasan:   menuconfig KASAN   depends on HAVE_ARCH_KASAN (generic) or
#                        HAVE_ARCH_KASAN_SW_TAGS (arm64-only) or HAVE_ARCH_KASAN_HW_TAGS
#   lib/Kconfig.kfence:  menuconfig KFENCE  depends on HAVE_ARCH_KFENCE
#
# This build runs CONFIG_ARCH_RV32I=y (CONFIG_64BIT is not set), so
# HAVE_ARCH_KASAN and HAVE_ARCH_KFENCE are never selected, and the KASAN/KFENCE
# Kconfig prompts are never satisfiable. Proof (reproduced by this script):
# enabling CONFIG_KASAN/CONFIG_KFENCE with scripts/config and then running
# `make olddefconfig` makes the symbols vanish from .config ENTIRELY (not even
# emitted as `# CONFIG_KASAN is not set`) -- olddefconfig silently drops a
# selection whose `depends on` can never be true. There is no rv32 KASAN/KFENCE
# arch support to fall back onto within this kernel source tree; adding it
# would mean implementing arch_kfence_init_pool()/shadow-memory offset
# calculations for Sv32 page tables from scratch, which is arch bring-up work,
# not a kernel config change, and is out of scope here.
#
# So this script builds the strongest oracle that IS actually available on
# rv32 with no arch bring-up required: CONFIG_DEBUG_PAGEALLOC (unmaps pages
# from the kernel linear map immediately on free_pages() -- any subsequent
# read OR write faults immediately, a real hard fault, not a free-time check)
# plus CONFIG_PAGE_POISONING (fills freed pages with a poison pattern and
# verifies it on next alloc, catching corruption even where DEBUG_PAGEALLOC's
# unmap doesn't apply), on top of the same CONFIG_SLUB_DEBUG_ON already used by
# Image.slubdebug.
#
# HONEST LIMITATION: DEBUG_PAGEALLOC/PAGE_POISONING operate at *page*
# granularity (arch/riscv ARCH_SUPPORTS_DEBUG_PAGEALLOC, gated only on MMU --
# no 64BIT restriction). They catch immediate UAF/OOB on whole pages returned
# to the buddy allocator (vmalloc, order>0 allocations, a fully-emptied slab
# page reclaimed back to the page allocator) -- they do NOT give
# KASAN/KFENCE's byte-level, every-kmalloc-object redzone coverage, since SLUB
# packs multiple small objects per page and the page stays mapped as long as
# any object on it is still live. This is a real, narrower oracle than
# KASAN/KFENCE would have been -- it is what's actually buildable on this
# target, not a full substitute.
#
# Produces:
#   firmware/Image.dpalloc
#   firmware/System.map.dpalloc
#
# Does NOT touch firmware/Image, firmware/Image.slubdebug, firmware/Image.buggy,
# build/linux-src, or any other existing worktree/build dir. build/linux-kasan-src
# and build/linux-kasan are new, dedicated to this script.
set -euo pipefail

ROOT="/home/forrest/fuzzsoft"
KERNEL_REPO="${ROOT}/linux"                   # main kernel git repo
STOCK_SRC="${ROOT}/build/linux-src"           # existing stock worktree (untouched)
KASAN_SRC="${ROOT}/build/linux-kasan-src"     # clean worktree, created if missing
OUT_DIR="${ROOT}/build/linux-kasan"           # O= out-of-tree build dir
BUILD_LOG="${ROOT}/build/kernel-kasan-build.log"
NPROC="$(nproc)"

echo "== build-kasan-kernel: using $NPROC parallel jobs =="

# 1. Make sure we have a clean worktree of the kernel source, checked out at
#    the same commit as the stock build/linux-src worktree (same reasoning as
#    build-slubdebug-kernel.sh / build-buggy-kernel.sh: Kbuild's O= build
#    refuses to run against a source tree that already holds an in-tree build).
if [ ! -d "$KASAN_SRC" ]; then
    if [ ! -d "$STOCK_SRC" ]; then
        echo "error: $STOCK_SRC does not exist; cannot determine kernel commit" >&2
        exit 1
    fi
    commit="$(git -C "$STOCK_SRC" rev-parse HEAD)"
    echo "== adding clean worktree $KASAN_SRC @ $commit =="
    git -C "$KERNEL_REPO" worktree add --detach "$KASAN_SRC" "$commit"
else
    echo "== reusing existing worktree $KASAN_SRC =="
fi

mkdir -p "$OUT_DIR"

# 2. Seed the O= .config from the stock .config, then olddefconfig once.
if [ ! -f "$OUT_DIR/.config" ]; then
    echo "== seeding $OUT_DIR/.config from $STOCK_SRC/.config =="
    cp "$STOCK_SRC/.config" "$OUT_DIR/.config"
fi

make -C "$KASAN_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig

# 3. Try KASAN first (the gold standard, if available).
"$KASAN_SRC/scripts/config" --file "$OUT_DIR/.config" -e KASAN -e KASAN_GENERIC -e KASAN_INLINE
make -C "$KASAN_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig
if grep -qE '^CONFIG_KASAN=y' "$OUT_DIR/.config"; then
    echo "== CONFIG_KASAN stuck! rv32 KASAN IS available on this tree -- building it. =="
    ORACLE="kasan"
else
    echo "== CONFIG_KASAN did NOT stick (expected on rv32: arch/riscv/Kconfig gates"
    echo "   HAVE_ARCH_KASAN behind MMU && 64BIT). Falling back to CONFIG_KFENCE. =="

    # 4. Try KFENCE next.
    "$KASAN_SRC/scripts/config" --file "$OUT_DIR/.config" -e KFENCE
    "$KASAN_SRC/scripts/config" --file "$OUT_DIR/.config" --set-val KFENCE_SAMPLE_INTERVAL 100
    make -C "$KASAN_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig
    if grep -qE '^CONFIG_KFENCE=y' "$OUT_DIR/.config"; then
        echo "== CONFIG_KFENCE stuck! rv32 KFENCE IS available on this tree -- building it. =="
        ORACLE="kfence"
    else
        echo "== CONFIG_KFENCE did NOT stick either (expected: same MMU && 64BIT gate on"
        echo "   HAVE_ARCH_KFENCE in arch/riscv/Kconfig). Neither KASAN nor KFENCE is"
        echo "   available on rv32 in this kernel tree -- falling back to"
        echo "   CONFIG_DEBUG_PAGEALLOC + CONFIG_PAGE_POISONING, the strongest oracle"
        echo "   that IS available with no arch bring-up (see header comment above). =="
        "$KASAN_SRC/scripts/config" --file "$OUT_DIR/.config" \
            -e DEBUG_KERNEL -e DEBUG_PAGEALLOC -e PAGE_POISONING -e SLUB_DEBUG -e SLUB_DEBUG_ON
        make -C "$KASAN_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 olddefconfig
        ORACLE="dpalloc"
    fi
fi

echo "== verifying config (oracle=$ORACLE) =="
grep -E '^CONFIG_INITRAMFS_SOURCE=' "$OUT_DIR/.config" | grep -q 'boot/initramfs.spec' || {
    echo "error: CONFIG_INITRAMFS_SOURCE lost the fuzzer's initramfs.spec path" >&2
    exit 1
}
case "$ORACLE" in
    kasan)
        grep -qE '^CONFIG_KASAN=y' "$OUT_DIR/.config" || { echo "error: CONFIG_KASAN did not stick" >&2; exit 1; }
        ;;
    kfence)
        grep -qE '^CONFIG_KFENCE=y' "$OUT_DIR/.config" || { echo "error: CONFIG_KFENCE did not stick" >&2; exit 1; }
        ;;
    dpalloc)
        grep -qE '^CONFIG_DEBUG_PAGEALLOC=y' "$OUT_DIR/.config" || { echo "error: CONFIG_DEBUG_PAGEALLOC did not stick" >&2; exit 1; }
        ;;
esac
grep -E '^CONFIG_KASAN=|^CONFIG_KFENCE=|^CONFIG_DEBUG_PAGEALLOC=|^CONFIG_PAGE_POISONING=|^CONFIG_SLUB_DEBUG(_ON)?=|^CONFIG_INITRAMFS_SOURCE=' "$OUT_DIR/.config"

# 5. Full build.
echo "== building Image (this is a full from-scratch build, several minutes) =="
make -C "$KASAN_SRC" O="$OUT_DIR" ARCH=riscv LLVM=1 -j"$NPROC" Image 2>&1 | tee "$BUILD_LOG"

IMAGE="$OUT_DIR/arch/riscv/boot/Image"
SYSMAP="$OUT_DIR/System.map"
[ -f "$IMAGE" ] || { echo "error: $IMAGE missing after build" >&2; exit 1; }
[ -f "$SYSMAP" ] || { echo "error: $SYSMAP missing after build" >&2; exit 1; }

# 6. Publish under a name matching whichever oracle actually got built. Never
#    touch firmware/Image, firmware/Image.slubdebug, firmware/Image.buggy.
cp "$IMAGE" "$ROOT/firmware/Image.${ORACLE}"
cp "$SYSMAP" "$ROOT/firmware/System.map.${ORACLE}"

echo "== done: built oracle = $ORACLE =="
ls -la "$ROOT/firmware/Image.${ORACLE}" "$ROOT/firmware/System.map.${ORACLE}"
echo "Smoke-test with:"
echo "  ${ROOT}/target/release/fuzzsoft fuzz --cases 100 --seed 1 --kernel ${ROOT}/firmware/Image.${ORACLE}"
