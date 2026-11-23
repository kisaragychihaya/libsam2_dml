#!/usr/bin/env bash
# Builds the pure-Go QC binary (CGO disabled), stages the three DLLs next to the
# exe, then runs the QC pass. Everything the run writes lands in example/out/.
#
#   ./example/run_qc.sh                    # deterministic synthetic test image
#   ./example/run_qc.sh photo.jpg          # your own image instead
#
# Environment:
#   SAM2_DLL / SAM2_ENCODER / SAM2_DECODER   override the paths derived below
#   SAM2_DEVICE                             DirectML adapter index (default 0)
#   SAM2_RUNS                               timed iterations per stage (default 5)
#   SAM2_MODEL_DIR                          where the .onnx models live (default: repo root)
#
# The three DLLs are taken from native/build/ when present (i.e. after
# ./native/build.sh), otherwise from bin/ (where the release zip unpacks them).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BIN="$REPO/bin"
WORK="$HERE/out"
LOG="$WORK/qc_output.log"
MODELS="${SAM2_MODEL_DIR:-$REPO}"
USER_IMAGE="${1:-}"

have_dlls() { [ -f "$1/sam2_dml.dll" ] && [ -f "$1/onnxruntime.dll" ] && [ -f "$1/DirectML.dll" ]; }

DLLSRC=""
for cand in "$REPO/native/build" "$BIN"; do
  if have_dlls "$cand"; then DLLSRC="$cand"; break; fi
done
if [ -z "$DLLSRC" ]; then
  cat >&2 <<EOF
missing sam2_dml.dll / onnxruntime.dll / DirectML.dll
  looked in: $REPO/native/build and $BIN
  build them with ./native/build.sh, or unpack the release zip into $BIN
EOF
  exit 1
fi

command -v go >/dev/null || { echo "go toolchain not found on PATH" >&2; exit 1; }

mkdir -p "$BIN" "$WORK"

echo "== build (CGO_ENABLED=0) -> $BIN/sam2qc.exe"
(
  cd "$HERE"
  CGO_ENABLED=0 go build -trimpath -o "$BIN/sam2qc.exe" .
)

echo "== stage DLLs from $DLLSRC -> $BIN"
for dll in sam2_dml.dll onnxruntime.dll DirectML.dll; do
  cp -f "$DLLSRC/$dll" "$BIN/"
done

if [ -n "$USER_IMAGE" ]; then
  [ -f "$USER_IMAGE" ] || { echo "input image not found: $USER_IMAGE" >&2; exit 1; }
  IMAGE="$(cd "$(dirname "$USER_IMAGE")" && pwd)/$(basename "$USER_IMAGE")"
  echo "== using caller-supplied image: $IMAGE"
else
  IMAGE="$WORK/test_image.png"
  echo "== generate synthetic test image -> $IMAGE"
  "$BIN/sam2qc.exe" --gen-test-image "$(cygpath -w "$IMAGE")"
fi

# The exe resolves sam2_dml.dll next to itself; point it at the staged copy so
# the run does not depend on the build tree.
export SAM2_DLL="${SAM2_DLL:-$(cygpath -w "$BIN/sam2_dml.dll")}"
export SAM2_ENCODER="${SAM2_ENCODER:-$(cygpath -w "$MODELS/sam2_hiera_small.encoder.onnx")}"
export SAM2_DECODER="${SAM2_DECODER:-$(cygpath -w "$MODELS/sam2_hiera_small.decoder.onnx")}"

echo "== run"
set +e
(
  cd "$WORK"
  "$BIN/sam2qc.exe" "$(cygpath -w "$IMAGE")"
) 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
set -e

echo
if [ "$rc" -ne 0 ]; then
  echo "QC FAILED (exit $rc); see $LOG" >&2
  exit "$rc"
fi
echo "QC PASSED (log: $LOG)"
