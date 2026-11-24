#!/usr/bin/env bash
# One-shot local QC: build sam2qc.exe, stage sam2_ncnn.dll + the ncnn models,
# generate the deterministic test image (or take $1), run the QC pass, and
# cross-check the logits against the Python reference.
#
#   ./example/run_qc.sh [image.png]
#
# Env: SAM2_MODEL_DIR - directory holding sam2_*_fixed.ncnn.param/.bin and
#      sam2_prompt.bin (default: repo root); SAM2_RUNS (default 5),
#      SAM2_EXPECT (default Cpu).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BIN="$REPO/bin"
WORK="$HERE/out"
LOG="$WORK/qc_output.log"
MODELS="${SAM2_MODEL_DIR:-$REPO}"
USER_IMAGE="${1:-}"

have_models() {
  [ -f "$1/sam2_encoder_fixed.ncnn.param" ] && [ -f "$1/sam2_encoder_fixed.ncnn.bin" ] &&
  [ -f "$1/sam2_decoder_fixed.ncnn.param" ] && [ -f "$1/sam2_decoder_fixed.ncnn.bin" ] &&
  [ -f "$1/sam2_prompt.bin" ]
}

DLLSRC=""
for cand in "$REPO/native/build" "$REPO/native/build-ncnn" "$BIN"; do
  if [ -f "$cand/sam2_ncnn.dll" ]; then DLLSRC="$cand"; break; fi
done
if [ -z "$DLLSRC" ]; then
  echo "missing sam2_ncnn.dll; looked in $REPO/native/build(-ncnn) and $BIN" >&2
  echo "build it with ./native/build.sh" >&2
  exit 1
fi
if ! have_models "$MODELS"; then
  echo "missing ncnn models in $MODELS (need sam2_*_fixed.ncnn.param/.bin + sam2_prompt.bin)" >&2
  echo "grab them from the v0.2.0 release assets" >&2
  exit 1
fi

command -v go >/dev/null || { echo "go toolchain not found on PATH" >&2; exit 1; }

mkdir -p "$BIN" "$WORK"

echo "== build (CGO_ENABLED=0) -> $BIN/sam2qc.exe"
(
  cd "$HERE"
  CGO_ENABLED=0 go build -trimpath -o "$BIN/sam2qc.exe" .
)

echo "== stage DLL + models -> $BIN"
cp -f "$DLLSRC/sam2_ncnn.dll" "$BIN/"
for f in sam2_encoder_fixed.ncnn.param sam2_encoder_fixed.ncnn.bin \
         sam2_decoder_fixed.ncnn.param sam2_decoder_fixed.ncnn.bin sam2_prompt.bin; do
  cp -f "$MODELS/$f" "$BIN/"
done

if [ -n "$USER_IMAGE" ]; then
  [ -f "$USER_IMAGE" ] || { echo "input image not found: $USER_IMAGE" >&2; exit 1; }
  IMAGE="$(cd "$(dirname "$USER_IMAGE")" && pwd)/$(basename "$USER_IMAGE")"
  echo "== using caller-supplied image: $IMAGE"
else
  IMAGE="$WORK/test_image.png"
  echo "== generate deterministic test image: $IMAGE"
  "$BIN/sam2qc.exe" --gen-test-image "$IMAGE"
fi

echo "== QC pass (log: $LOG)"
(
  cd "$WORK"
  SAM2_DLL="$(cygpath -w "$BIN/sam2_ncnn.dll")" \
  SAM2_ENCODER="$(cygpath -w "$BIN/sam2_encoder_fixed.ncnn.param")" \
  SAM2_DECODER="$(cygpath -w "$BIN/sam2_decoder_fixed.ncnn.param")" \
  SAM2_RUNS="${SAM2_RUNS:-5}" \
  SAM2_EXPECT="${SAM2_EXPECT:-Cpu}" \
    "$BIN/sam2qc.exe" "$IMAGE" | tee "$LOG"
)

if [ "${CROSSCHECK:-1}" != "0" ]; then
  echo "== cross-check vs Python reference"
  "$HERE/crosscheck.sh" "$IMAGE"
fi
