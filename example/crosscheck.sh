#!/usr/bin/env bash
# Cross-checks the Go pure-FFI logits against the Python/DirectML reference.
# Run ./example/run_qc.sh first: it produces the canonical go_logits.bin /
# go_rgb.bin for the test image. This script adds a PNG control run and every
# reference dump, then prints the layered comparison.
#
#   ./example/crosscheck.sh                # synthetic test image (default)
#   ./example/crosscheck.sh photo.jpg      # same image you gave to run_qc.sh
#
# Requires onnxruntime-directml + pillow + numpy:
#   python -m pip install onnxruntime-directml pillow numpy
#
# Layered comparisons (compare.py, atol=1e-3). The two GATE pairs must pass, the
# JPEG pairs are diagnostic because Go's image/jpeg and libjpeg only agree to
# ~14 LSB per channel:
#   go_logits.bin     vs ref_logits_gorxgb.bin  same pixels the DLL saw -> inference only
#   go_logits_png.bin vs ref_logits_png.bin      same file, decoded by both sides
#   go_logits.bin     vs ref_logits_src.bin      input decoded independently per side
#   go_logits.bin     vs ref_logits_pil.bin      PIL's own resize kernel too
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK="$HERE/out"
LOG="$WORK/compare_output.log"
MODELS="${SAM2_MODEL_DIR:-$REPO}"
IMAGE="${1:-$WORK/test_image.png}"
PNG="$WORK/test_image_png.png"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/.venv/Scripts/python.exe" ]; then
    PYTHON="$REPO/.venv/Scripts/python.exe"
  else
    PYTHON="$(command -v python || command -v python3 || true)"
  fi
fi
[ -n "$PYTHON" ] || { echo "python not found; set PYTHON=/path/to/python" >&2; exit 1; }

[ -f "$WORK/go_logits.bin" ] || { echo "run ./example/run_qc.sh first ($WORK/go_logits.bin missing)" >&2; exit 1; }
[ -f "$IMAGE" ] || { echo "input image not found: $IMAGE" >&2; exit 1; }

export SAM2_DLL="${SAM2_DLL:-$(cygpath -w "$REPO/bin/sam2_dml.dll")}"
export SAM2_ENCODER="${SAM2_ENCODER:-$(cygpath -w "$MODELS/sam2_hiera_small.encoder.onnx")}"
export SAM2_DECODER="${SAM2_DECODER:-$(cygpath -w "$MODELS/sam2_hiera_small.decoder.onnx")}"

mkdir -p "$WORK"
cd "$WORK"
: > "$LOG"

# A lossless PNG copy of the test image: the Go PNG decoder and PIL then see
# bit-identical pixels, so that comparison isolates the DLL from decoder
# differences (Go's image/jpeg and libjpeg disagree by up to 14 LSB per channel).
{
  "$PYTHON" - "$IMAGE" "$PNG" <<'PYEOF'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
with Image.open(src) as im:
    im.convert("RGB").save(dst, format="PNG", optimize=True)
    print(f"png copy        : {dst} ({im.width}x{im.height}) from {src}")
PYEOF
} 2>&1 | tee -a "$LOG"

echo "== control run on the PNG copy" | tee -a "$LOG"
PNG_RUN_DIR="$WORK/.png_run"
rm -rf "$PNG_RUN_DIR" && mkdir -p "$PNG_RUN_DIR"
(cd "$PNG_RUN_DIR" && "$REPO/bin/sam2qc.exe" "$(cygpath -w "$PNG")" > run.log 2>&1) || {
  cat "$PNG_RUN_DIR/run.log" >&2; echo "PNG control run failed" >&2; exit 1;
}
mv -f "$PNG_RUN_DIR/go_logits.bin" go_logits_png.bin
mv -f "$PNG_RUN_DIR/go_rgb.bin" go_rgb_png.bin
cp -f "$PNG_RUN_DIR/run.log" png_run.log
rm -rf "$PNG_RUN_DIR"
grep -E "provider|encode|decode|prompt" png_run.log | tee -a "$LOG"

echo "== python reference dumps" | tee -a "$LOG"
ref() {
  "$PYTHON" "$HERE/ref_dump.py" --encoder "$SAM2_ENCODER" --decoder "$SAM2_DECODER" "$@" 2>&1 | tee -a "$LOG"
}
ref --image "$IMAGE"   --resize float --out ref_logits_src.bin
ref --image "$IMAGE"   --resize pil   --out ref_logits_pil.bin
ref --rgb   go_rgb.bin --resize float --out ref_logits_gorxgb.bin
ref --image "$PNG"     --resize float --out ref_logits_png.bin

echo | tee -a "$LOG"
set +e
"$PYTHON" "$HERE/compare.py" --dir "$WORK" 2>&1 | tee -a "$LOG"
rc="${PIPESTATUS[0]}"
set -e
echo "cross-check log : $LOG"
exit "$rc"
