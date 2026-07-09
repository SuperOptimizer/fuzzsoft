#!/usr/bin/env bash
# Build the 2-hart SMP device-tree blob (SMP epic, docs/smp-design.md §2 Phase 1 / roadmap T5.1b)
# from boot/fuzzsoft-smp.dts. Separate from the single-hart firmware/fuzzsoft.dtb (built by hand
# from boot/fuzzsoft.dts, decision #40) so the default fuzzer's DTB is never touched.
#
# Produces:  firmware/fuzzsoft-smp.dtb
#
# Usage:  scripts/build-smp-dtb.sh
#   fuzzsoft smp-boot --firmware firmware/fw_jump.bin --dtb firmware/fuzzsoft-smp.dtb \
#                     --kernel firmware/Image --replay
set -euo pipefail

ROOT="/home/forrest/fuzzsoft"
SRC="${ROOT}/boot/fuzzsoft-smp.dts"
OUT="${ROOT}/firmware/fuzzsoft-smp.dtb"

command -v dtc >/dev/null 2>&1 || { echo "error: dtc (device-tree-compiler) not found on PATH" >&2; exit 1; }
[ -f "$SRC" ] || { echo "error: $SRC missing" >&2; exit 1; }

mkdir -p "${ROOT}/firmware"
dtc -I dts -O dtb -o "$OUT" "$SRC"
echo "== built $OUT ($(stat -c%s "$OUT") bytes) from $SRC =="
