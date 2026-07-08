#!/usr/bin/env bash
#
# fuzzsoft dependency installer.
#
# Installs the full toolchain we need for milestones M0-M3:
#   - clang/LLVM + lld           : build RV32 test binaries, userland, and (M2) the kernel with LLVM=1
#   - device-tree-compiler (dtc) : build DTBs (M2) and required to build Spike
#   - qemu-system-misc           : qemu-system-riscv32/64 as the full-system boot oracle (M2)
#   - Spike (riscv-isa-sim)       : canonical ISA golden model, built from source with --enable-commitlog
#                                   (per-instruction register diffs for M1 differential testing)
#   - kernel/buildroot host deps : bc bison flex libelf/ssl-dev cpio zstd rsync ... (M2/M3)
#   - rustup + nightly           : for std::simd (portable_simd) in the vectorized executor (M4)
#
# Usage:   sudo bash scripts/install-deps.sh
#
# apt + Spike install need root; rustup is installed for the INVOKING user (not root),
# detected via $SUDO_USER. Safe to re-run (idempotent).

set -euo pipefail

# --- privilege / user detection ------------------------------------------------
if [[ "${EUID}" -ne 0 ]]; then
    echo "!! This script needs root for apt and the Spike install."
    echo "   Re-run:  sudo bash scripts/install-deps.sh"
    exit 1
fi

REAL_USER="${SUDO_USER:-root}"
if [[ "${REAL_USER}" == "root" ]]; then
    echo "!! Could not detect a non-root user (SUDO_USER is empty)."
    echo "   rustup should not be installed as root. Run this via 'sudo bash ...' from your normal user."
    exit 1
fi
REAL_HOME="$(getent passwd "${REAL_USER}" | cut -d: -f6)"
echo "==> Privileged steps run as root; rustup will be installed for user '${REAL_USER}' (${REAL_HOME})"

run_as_user() { sudo -u "${REAL_USER}" -H bash -lc "$1"; }

SPIKE_SRC="/opt/riscv-isa-sim"
PREFIX="/usr/local"
JOBS="$(nproc)"

# --- 1. apt packages -----------------------------------------------------------
echo
echo "==> [1/4] Installing apt packages ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates pkg-config python3 python3-pip \
    cmake ninja-build \
    clang lld llvm \
    device-tree-compiler \
    autoconf automake \
    qemu-system-misc qemu-user \
    bc bison flex libelf-dev libssl-dev cpio gzip zstd xz-utils \
    unzip rsync file libncurses-dev \
    gdb-multiarch

# Best-effort perf (kernel-version specific; don't fail the run if unavailable)
apt-get install -y linux-tools-generic 2>/dev/null || echo "   (skipped linux-tools-generic — install manually if you want 'perf')"

# --- 2. Spike (riscv-isa-sim) from source -------------------------------------
echo
echo "==> [2/4] Building Spike (riscv-isa-sim) with commit-log ..."
if command -v spike >/dev/null 2>&1; then
    echo "   spike already installed at $(command -v spike) — skipping build. Delete ${SPIKE_SRC} and re-run to rebuild."
else
    if [[ ! -d "${SPIKE_SRC}/.git" ]]; then
        rm -rf "${SPIKE_SRC}"
        git clone --depth 1 https://github.com/riscv-software-src/riscv-isa-sim "${SPIKE_SRC}"
    fi
    mkdir -p "${SPIKE_SRC}/build"
    pushd "${SPIKE_SRC}/build" >/dev/null
    ../configure --prefix="${PREFIX}" --enable-commitlog
    make -j"${JOBS}"
    make install
    popd >/dev/null
    ldconfig
    echo "   Spike installed to ${PREFIX}/bin/spike"
fi

# --- 3. rustup + nightly (as the invoking user) --------------------------------
echo
echo "==> [3/4] Installing rustup + nightly for user '${REAL_USER}' ..."
if run_as_user 'command -v rustup >/dev/null 2>&1'; then
    echo "   rustup already present — ensuring nightly toolchain + components ..."
    run_as_user 'rustup toolchain install nightly --profile default'
    run_as_user 'rustup default nightly'
else
    echo "   Installing rustup (default toolchain: nightly) ..."
    run_as_user 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly --profile default'
fi
# rust-src helps rust-analyzer and -Zbuild-std experiments; clippy/rustfmt come with the default profile.
run_as_user 'source "$HOME/.cargo/env" 2>/dev/null; rustup component add rust-src clippy rustfmt' || true

echo
echo "   NOTE: rustup put its shims in ${REAL_HOME}/.cargo/bin. Make sure that is FIRST on PATH"
echo "         (the installer adds it to your shell profile; open a new shell or 'source ~/.cargo/env')."
echo "         There is also a system rustc 1.93 (stable, tarball) — the rustup nightly shim should win."

# --- 4. Verify -----------------------------------------------------------------
echo
echo "==> [4/4] Verifying toolchain ..."
ok()   { printf '   [ ok ] %-22s %s\n' "$1" "$2"; }
miss() { printf '   [MISS] %-22s %s\n' "$1" "$2"; }

check() { if command -v "$1" >/dev/null 2>&1; then ok "$1" "$($2 2>&1 | head -1)"; else miss "$1" "not found"; fi; }

check clang               "clang --version"
check ld.lld              "ld.lld --version"
check llvm-objcopy        "llvm-objcopy --version"
check dtc                 "dtc --version"
check qemu-system-riscv32 "qemu-system-riscv32 --version"
check qemu-riscv32        "qemu-riscv32 --version"
check spike               "spike --help"

echo "   --- rust (as ${REAL_USER}) ---"
run_as_user 'source "$HOME/.cargo/env" 2>/dev/null; printf "   [ ok ] %-22s %s\n" rustup  "$(rustup --version 2>&1 | head -1)"; printf "   [ ok ] %-22s %s\n" rustc "$(rustc --version 2>&1 | head -1)"; printf "   [ ok ] %-22s %s\n" cargo "$(cargo --version 2>&1 | head -1)"' || miss rustup "verify failed"

echo
echo "==> Done. If any line says [MISS], tell me and I'll adjust the script."
echo "    Next: quick sanity check that clang can emit RV32 ->"
echo "      echo 'int _start(){return 0;}' | clang --target=riscv32 -march=rv32im -mabi=ilp32 -nostdlib -x c - -o /tmp/rv32test.elf && file /tmp/rv32test.elf"
