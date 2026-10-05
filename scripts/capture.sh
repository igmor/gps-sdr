#!/usr/bin/env bash
# Record raw GPS L1 with a HackRF One (receive only).
#
# Requirements:
#   - An *active* GPS antenna (patch antenna with an LNA, roughly $10-20) with a
#     clear view of the sky. A passive whip antenna won't pick up GPS.
#   - Bias-tee power (-p 1) sends 3.3 V up the coax to run the antenna's LNA.
#     Only use -p 1 with an active antenna: it can damage other devices on that port.
#
# Usage: scripts/capture.sh [out_file] [seconds]
#   ~40 s is enough to decode a full navigation frame (30 s) with margin.
set -euo pipefail

OUT=${1:-data/gps.bin}
SECONDS_TO_RECORD=${2:-40}
FS=4000000           # 4 MS/s: the C/A main lobe is +/-1.023 MHz
FREQ=1575420000      # GPS L1
# Use radioconda's tools, which match the board's firmware (2024.02.1).
HACKRF_TRANSFER=${HACKRF_TRANSFER:-$HOME/radioconda/bin/hackrf_transfer}

"$HACKRF_TRANSFER" -r "$OUT" \
  -f "$FREQ" \
  -s "$FS" \
  -n $((FS * SECONDS_TO_RECORD)) \
  -a 1 \
  -l 32 \
  -g 40 \
  -p 1

echo "Saved $OUT ($(du -h "$OUT" | cut -f1)). Next: python acquire.py $OUT --fs $FS"
