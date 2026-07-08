#!/usr/bin/env bash
# Build the fault-injection RV32 Linux kernel Images: the #1 lever in docs/bug-finding.md's
# bug-finding plan. Random argument fuzzing essentially never makes a real kmalloc/alloc_pages
# legitimately fail in a small guest, so every "the allocation failed, clean up" branch is dead
# code no campaign has executed — exactly where UAF/double-free/leak bugs concentrate. This turns
# on the kernel's own fault-injection framework (fail_nth: a task can arm should_fail_ex() to
# fail exactly its Nth matching allocation, then self-disarm — see lib/fault-inject.c), combined
# with CONFIG_SLUB_DEBUG_ON as the oracle that catches the resulting corruption via the *existing*
# kernel_crash_sig() console scan.
#
# THE LOAD-BEARING GOTCHA (see docs/bug-finding.md): mm/failslab.c and mm/fail_page_alloc.c both
# default `ignore_gfp_reclaim = true`, which silently exempts ordinary GFP_KERNEL allocations (the
# common case) from ever failing. CONFIG_FAULT_INJECTION_DEBUG_FS exposes debugfs knobs
# (failslab/ignore-gfp-wait, fail_page_alloc/ignore-gfp-wait, fail_page_alloc/ignore-gfp-highmem,
# fail_page_alloc/min-order) that must be flipped to 0 — boot/agent.c's
# arm_fault_injection_knobs() does this once, pre-snapshot, so it's baked into the golden image.
#
# Produces TWO kernel images:
#   firmware/Image.failinj + firmware/System.map.failinj
#       — clean fault-injection kernel: FAULT_INJECTION + FAILSLAB + FAIL_PAGE_ALLOC +
#         FAULT_INJECTION_DEBUG_FS + SLUB_DEBUG_ON. FAULT_INJECTION_USERCOPY is deliberately left
#         OFF (docs/bug-finding.md: it "drowns signal in -EFAULT").
#   firmware/Image.failinj.buggy + firmware/System.map.failinj.buggy
#       — the SAME config, plus scripts/failinj-bug.patch: a hand-planted double-free in
#         memfd_create() that is dead code UNLESS fault injection is armed against its second
#         kmalloc. Exists purely to validate the arm -> inject -> crash -> kernel_crash_sig chain
#         end-to-end (a "0 crashes" result on the clean Image.failinj alone would be ambiguous
#         between "mechanism works, target just has no reachable bugs" and "mechanism is inert").
#
# Does NOT touch firmware/Image, firmware/Image.slubdebug, firmware/Image.buggy,
# firmware/Image.dpalloc, firmware/Image.kasan, or any of their System.map* siblings — every
# other Image variant in firmware/ is untouched. build/linux-src (the stock worktree/build) is
# untouched.
#
# Why separate clean worktrees per image (same pattern as build-slubdebug-kernel.sh /
# build-buggy-kernel.sh): Kbuild's O= out-of-tree build refuses to run against a source tree that
# already holds an in-tree build ("The source tree is not clean, please run 'make mrproper'"), and
# we must not mrproper/mutate any existing worktree (stock, slubdebug, buggy, kasan, dpalloc, ...).
# Each variant therefore gets its own clean `git worktree add --detach` checkout at the same
# commit as the stock build/linux-src worktree.
set -euo pipefail

ROOT="/home/forrest/fuzzsoft"
KERNEL_REPO="${ROOT}/linux"                              # main kernel git repo
STOCK_SRC="${ROOT}/build/linux-src"                       # existing stock worktree (untouched)

FAILINJ_SRC="${ROOT}/build/linux-failinj-src"             # clean worktree, created if missing
FAILINJ_OUT="${ROOT}/build/linux-failinj"                 # O= out-of-tree build dir

BUGGY_SRC="${ROOT}/build/linux-failinj-buggy-src"         # clean worktree, created if missing
BUGGY_OUT="${ROOT}/build/linux-failinj-buggy"             # O= out-of-tree build dir

PATCH="${ROOT}/scripts/failinj-bug.patch"
NPROC="$(nproc)"

echo "== build-failinj-kernel: using $NPROC parallel jobs =="

if [ ! -d "$STOCK_SRC" ]; then
    echo "error: $STOCK_SRC does not exist; cannot determine kernel commit" >&2
    exit 1
fi
COMMIT="$(git -C "$STOCK_SRC" rev-parse HEAD)"

# The fault-injection config knobs shared by BOTH images below. FAULT_INJECTION_USERCOPY is
# deliberately never enabled here.
enable_failinj_config() {
    local src_tree="$1"
    local out_dir="$2"
    "$src_tree/scripts/config" --file "$out_dir/.config" \
        -e FAULT_INJECTION \
        -e FAILSLAB \
        -e FAIL_PAGE_ALLOC \
        -e FAULT_INJECTION_DEBUG_FS \
        -e SLUB_DEBUG \
        -e SLUB_DEBUG_ON
}

verify_config() {
    local out_dir="$1"
    local label="$2"
    echo "== verifying config ($label) =="
    for opt in CONFIG_FAULT_INJECTION=y CONFIG_FAILSLAB=y CONFIG_FAIL_PAGE_ALLOC=y \
               CONFIG_FAULT_INJECTION_DEBUG_FS=y CONFIG_SLUB_DEBUG_ON=y; do
        grep -qE "^${opt}$" "$out_dir/.config" || {
            echo "error: $opt did not stick in $out_dir/.config" >&2
            exit 1
        }
    done
    grep -E '^CONFIG_FAULT_INJECTION_USERCOPY=y' "$out_dir/.config" && {
        echo "error: CONFIG_FAULT_INJECTION_USERCOPY unexpectedly enabled in $out_dir/.config" >&2
        exit 1
    }
    grep -E '^CONFIG_INITRAMFS_SOURCE=' "$out_dir/.config" | grep -q 'boot/initramfs.spec' || {
        echo "error: CONFIG_INITRAMFS_SOURCE lost the fuzzer's initramfs.spec path in $out_dir/.config" >&2
        exit 1
    }
    grep -E '^CONFIG_FAULT_INJECTION=y|^CONFIG_FAILSLAB=y|^CONFIG_FAIL_PAGE_ALLOC=y|^CONFIG_FAULT_INJECTION_DEBUG_FS=y|^CONFIG_SLUB_DEBUG_ON=y|^CONFIG_INITRAMFS_SOURCE=' "$out_dir/.config"
    echo "== ($label) config verified OK =="
}

build_image() {
    local src_tree="$1"
    local out_dir="$2"
    local build_log="$3"
    echo "== building Image in $out_dir (full from-scratch build, several minutes) =="
    make -C "$src_tree" O="$out_dir" ARCH=riscv LLVM=1 -j"$NPROC" Image 2>&1 | tee "$build_log"
    [ -f "$out_dir/arch/riscv/boot/Image" ] || { echo "error: Image missing in $out_dir" >&2; exit 1; }
    [ -f "$out_dir/System.map" ] || { echo "error: System.map missing in $out_dir" >&2; exit 1; }
}

# ---------------------------------------------------------------------------
# 1. Clean fault-injection kernel: firmware/Image.failinj
# ---------------------------------------------------------------------------
if [ ! -d "$FAILINJ_SRC" ]; then
    echo "== adding clean worktree $FAILINJ_SRC @ $COMMIT =="
    git -C "$KERNEL_REPO" worktree add --detach "$FAILINJ_SRC" "$COMMIT"
else
    echo "== reusing existing worktree $FAILINJ_SRC =="
fi

mkdir -p "$FAILINJ_OUT"
if [ ! -f "$FAILINJ_OUT/.config" ]; then
    echo "== seeding $FAILINJ_OUT/.config from $STOCK_SRC/.config =="
    cp "$STOCK_SRC/.config" "$FAILINJ_OUT/.config"
fi

make -C "$FAILINJ_SRC" O="$FAILINJ_OUT" ARCH=riscv LLVM=1 olddefconfig
enable_failinj_config "$FAILINJ_SRC" "$FAILINJ_OUT"
make -C "$FAILINJ_SRC" O="$FAILINJ_OUT" ARCH=riscv LLVM=1 olddefconfig
verify_config "$FAILINJ_OUT" "Image.failinj"

build_image "$FAILINJ_SRC" "$FAILINJ_OUT" "${ROOT}/build/kernel-failinj-build.log"

cp "$FAILINJ_OUT/arch/riscv/boot/Image" "$ROOT/firmware/Image.failinj"
cp "$FAILINJ_OUT/System.map" "$ROOT/firmware/System.map.failinj"

echo "== done: Image.failinj =="
ls -la "$ROOT/firmware/Image.failinj" "$ROOT/firmware/System.map.failinj"

# ---------------------------------------------------------------------------
# 2. Buggy fault-injection kernel (oracle-validation only): firmware/Image.failinj.buggy
# ---------------------------------------------------------------------------
if [ ! -d "$BUGGY_SRC" ]; then
    echo "== adding clean worktree $BUGGY_SRC @ $COMMIT =="
    # NOTE: pass an absolute path. `git -C "$KERNEL_REPO" worktree add build/linux-failinj-buggy-src
    # ...` would resolve the relative path against $KERNEL_REPO (i.e. linux/build/..., wrong),
    # not $ROOT/build/... — this bit build-buggy-kernel.sh's author once, documented there.
    git -C "$KERNEL_REPO" worktree add --detach "$BUGGY_SRC" "$COMMIT"
else
    echo "== reusing existing worktree $BUGGY_SRC =="
fi

if grep -q "FUZZSOFT PLANTED BUG" "$BUGGY_SRC/mm/memfd.c" 2>/dev/null; then
    echo "== planted fault-injection bug already present in $BUGGY_SRC/mm/memfd.c =="
else
    echo "== applying $PATCH =="
    git -C "$BUGGY_SRC" apply "$PATCH"
fi
grep -q "FUZZSOFT PLANTED BUG" "$BUGGY_SRC/mm/memfd.c" || {
    echo "error: planted bug marker missing from $BUGGY_SRC/mm/memfd.c after apply" >&2
    exit 1
}

mkdir -p "$BUGGY_OUT"
if [ ! -f "$BUGGY_OUT/.config" ]; then
    echo "== seeding $BUGGY_OUT/.config from $STOCK_SRC/.config =="
    cp "$STOCK_SRC/.config" "$BUGGY_OUT/.config"
fi

make -C "$BUGGY_SRC" O="$BUGGY_OUT" ARCH=riscv LLVM=1 olddefconfig
enable_failinj_config "$BUGGY_SRC" "$BUGGY_OUT"
make -C "$BUGGY_SRC" O="$BUGGY_OUT" ARCH=riscv LLVM=1 olddefconfig
verify_config "$BUGGY_OUT" "Image.failinj.buggy"

build_image "$BUGGY_SRC" "$BUGGY_OUT" "${ROOT}/build/kernel-failinj-buggy-build.log"

cp "$BUGGY_OUT/arch/riscv/boot/Image" "$ROOT/firmware/Image.failinj.buggy"
cp "$BUGGY_OUT/System.map" "$ROOT/firmware/System.map.failinj.buggy"

echo "== done: Image.failinj.buggy =="
ls -la "$ROOT/firmware/Image.failinj.buggy" "$ROOT/firmware/System.map.failinj.buggy"

echo
echo "Smoke-test with:"
echo "  ${ROOT}/target/release/fuzzsoft fuzz --cases 20 --seed 1 --kernel ${ROOT}/firmware/Image.failinj"
echo "  ${ROOT}/target/release/fuzzsoft fuzz --cases 400 --seed 1 --kernel ${ROOT}/firmware/Image.failinj.buggy"
echo "(the latter only actually crashes once a --fail-inject generator flag is wired into fs-cli"
echo " to prepend fs-prog's genr::prepend_fail_inject onto a fraction of generated programs)"
