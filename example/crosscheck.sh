#!/usr/bin/env bash
# Compares Go pure-FFI logits with the PyTorch single-mask reference.
# Run ./example/run_qc.sh first to produce go_logits.bin and go_rgb.bin.
#
# The deterministic synthetic image uses the checked-in golden reference. For a
# different image, set SAM2_REF_LOGITS or install PyTorch/SAM2 and provide
# SAM2_CHECKPOINT so example/ref_dump.py can generate a matching reference.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK="$HERE/out"
IMAGE="${1:-$WORK/test_image.png}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/.venv/Scripts/python.exe" ]; then
    PYTHON="$REPO/.venv/Scripts/python.exe"
  elif [ -x "$REPO/../.venv/Scripts/python.exe" ]; then
    PYTHON="$REPO/../.venv/Scripts/python.exe"
  else
    PYTHON="$(command -v python || command -v python3 || true)"
  fi
fi
[ -n "$PYTHON" ] || { echo "python not found; set PYTHON=/path/to/python" >&2; exit 1; }
[ -f "$WORK/go_logits.bin" ] || { echo "run ./example/run_qc.sh first" >&2; exit 1; }
[ -f "$IMAGE" ] || { echo "input image not found: $IMAGE" >&2; exit 1; }

if [ -n "${SAM2_REF_LOGITS:-}" ]; then
  REF="$SAM2_REF_LOGITS"
elif [ "$IMAGE" = "$WORK/test_image.png" ]; then
  REF="$HERE/test_data/sam2_qc_reference.bin"
else
  REF="$WORK/ref_logits.bin"
  CHECKPOINT="${SAM2_CHECKPOINT:-$REPO/sam2_hiera_small.pt}"
  "$PYTHON" "$HERE/ref_dump.py" --rgb "$WORK/go_rgb.bin" \
    --checkpoint "$CHECKPOINT" --out "$REF"
fi

"$PYTHON" "$HERE/compare.py" "$WORK/go_logits.bin" "$REF"
