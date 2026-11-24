"""Compares the Go QC logits dump against the PyTorch SAM2 reference.

Both files use the container written by example/main.go (writeLogits) and
example/ref_dump.py (write_logits):

    8 bytes  magic "S2LOGITS"
    int32    num_masks, int32 mask_h, int32 mask_w
    float32  num_masks * mask_h * mask_w logits, row-major, little endian

Usage:
    python example/compare.py                       # every pair found in example/out
    python example/compare.py --dir path/to/out     # other work directory
    python example/compare.py go_logits.bin ref_logits.bin
    python example/compare.py --strict ...          # every pair becomes a gate

Pairs are classed as a gate (must be allclose) or diagnostic (reported, does not
affect the exit code). The pairs that decode the input independently on each side
are diagnostic on purpose: Go's image/jpeg and libjpeg agree only to within
~14 LSB per channel, and that pixel difference alone moves the logits by ~1, far
more than any DLL defect. Feed the same pixels (go_rgb.bin, or a lossless PNG
copy of the image) so the comparison isolates inference numerics.

Exit code 0 when every gate pair is within atol=2e-2, 1 otherwise.
"""

import os
import struct
import sys

import numpy as np

MAGIC = b"S2LOGITS"
ATOL = 2e-2
HEADER = 8 + 12


def read_logits(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != MAGIC:
        raise SystemExit(f"{path}: bad magic {blob[:8]!r}, expected {MAGIC!r}")
    num, h, w = struct.unpack("<iii", blob[8:HEADER])
    data = np.frombuffer(blob, dtype="<f4", offset=HEADER)
    if data.size != num * h * w:
        raise SystemExit(f"{path}: payload {data.size} floats != {num}x{h}x{w}")
    return data.reshape(num, h, w)


def compare(label, go_path, ref_path, gate):
    a = read_logits(go_path)
    b = read_logits(ref_path)
    kind = "GATE" if gate else "diagnostic"
    print(f"\n=== [{kind}] {label}")
    print(f"  files    : {os.path.basename(go_path)}  vs  {os.path.basename(ref_path)}")
    print(f"  shape    : go {a.shape}  ref {b.shape}")
    if a.shape != b.shape:
        print("  FAIL: shape mismatch")
        return False, gate

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rms = float(np.sqrt((diff ** 2).mean()))
    print(f"  max |d|  : {max_abs:.3e}")
    print(f"  mean |d| : {mean_abs:.3e}")
    print(f"  rms  |d| : {rms:.3e}")
    print(f"  max |ref|: {float(np.abs(b).max()):.3e}   (diff scale context)")
    print(f"  elems    : {a.size}  over atol: {int((diff > ATOL).sum())}")
    for i in range(a.shape[0]):
        d = np.abs(a[i].astype(np.float64) - b[i].astype(np.float64))
        print(f"  mask[{i}]  : max |d| {d.max():.3e}  mean |d| {d.mean():.3e}")

    ga = a.reshape(a.shape[0], -1) > 0
    gb = b.reshape(b.shape[0], -1) > 0
    union = int((ga | gb).sum())
    iou = float((ga & gb).sum() / union) if union else 1.0
    print(f"  binary IoU (logit>0, all masks): {iou:.6f}")
    print(f"  positive ratio: go {float((a > 0).mean()):.6f}  ref {float((b > 0).mean()):.6f}")

    ok = max_abs <= ATOL
    print(f"  max |d| <= atol={ATOL:g}: {'PASS' if ok else 'FAIL'}")
    return ok, gate


def default_pairs(work):
    p = lambda name: os.path.join(work, name)
    candidates = [
        ("same pixels the DLL saw (inference only)",
         p("go_logits.bin"), p("ref_logits_gorxgb.bin"), True),
        ("same PNG file decoded by both sides (lossless copy)",
         p("go_logits_png.bin"), p("ref_logits_png.bin"), True),
        ("input decoded independently on each side (Go decoder vs PIL)",
         p("go_logits.bin"), p("ref_logits_src.bin"), False),
        ("input decoded independently + PIL resize kernel",
         p("go_logits.bin"), p("ref_logits_pil.bin"), False),
    ]
    return [c for c in candidates if os.path.isfile(c[1]) and os.path.isfile(c[2])]


def main(argv):
    args = argv[1:]
    strict = "--strict" in args
    args = [a for a in args if a != "--strict"]

    here = os.path.dirname(os.path.abspath(__file__))
    work = os.path.join(here, "out")
    if "--dir" in args:
        i = args.index("--dir")
        if i + 1 >= len(args):
            raise SystemExit("--dir needs a path")
        work = args[i + 1]
        del args[i:i + 2]

    if not args:
        pairs = default_pairs(work)
        if not pairs:
            raise SystemExit(f"no logits dumps found under {work}; run ./example/run_qc.sh "
                             f"and ./example/crosscheck.sh first")
    elif len(args) % 2 != 0:
        raise SystemExit("pass pairs of files: go_logits.bin ref_logits.bin [more pairs]")
    else:
        pairs = [("explicit pair", args[i], args[i + 1], True) for i in range(0, len(args), 2)]

    if strict:
        pairs = [(label, a, b, True) for label, a, b, _ in pairs]

    gate_total = gate_ok = 0
    diag_fail = 0
    for label, go_path, ref_path, gate in pairs:
        for path in (go_path, ref_path):
            if not os.path.isfile(path):
                raise SystemExit(f"missing file: {path}")
        ok, is_gate = compare(label, go_path, ref_path, gate)
        if is_gate:
            gate_total += 1
            gate_ok += int(ok)
        elif not ok:
            diag_fail += 1

    print()
    if gate_total:
        print(f"GATE PAIRS      : {gate_ok}/{gate_total} within atol={ATOL:g}")
    if diag_fail:
        print(f"DIAGNOSTIC PAIRS: {diag_fail} exceed atol={ATOL:g} "
              f"(expected only when the two sides decode the input independently)")
    verdict = gate_total > 0 and gate_ok == gate_total
    print(f"OVERALL: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
