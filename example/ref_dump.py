"""Python/DirectML reference for the Go pure-FFI QC run.

Runs the same encoder+decoder pair with onnxruntime-directml, following the
anylabeling rules (resize to 1024x1024 -> /255 -> ImageNet mean/std -> NCHW
float32, single positive point at the image center, zero mask_input,
has_mask_input=[0]) and writes the logits with the same container layout as the
Go side's go_logits.bin:

    8 bytes  magic "S2LOGITS"
    int32    num_masks
    int32    mask_h
    int32    mask_w
    float32  num_masks * mask_h * mask_w logits, row-major, little endian

Usage (from the repo root):
    python example/ref_dump.py --image example/out/test_image.png --out example/out/ref_logits_src.bin
    python example/ref_dump.py --rgb example/out/go_rgb.bin \\
        --resize float --out example/out/ref_logits_gorxgb.bin

--rgb uses the exact RGB24 buffer dumped by the Go binary (go_rgb.bin), which
removes JPEG-decoder differences from the comparison.
--resize float reproduces the DLL's half-pixel bilinear resize in float32;
--resize pil uses PIL's Image.BILINEAR (what test_ort_dml.py does).
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

ENC_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOGITS_MAGIC = b"S2LOGITS"
RGB_MAGIC = b"S2RGB123"


def log(*args):
    print(*args, flush=True)


def ms(seconds):
    return f"{seconds * 1000:.2f}"


def load_image_rgb(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return rgb


def load_go_rgb(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != RGB_MAGIC:
        raise SystemExit(f"{path}: bad magic, expected {RGB_MAGIC!r}")
    w, h = struct.unpack("<ii", blob[8:16])
    pix = np.frombuffer(blob, dtype=np.uint8, offset=16)
    if pix.size != w * h * 3:
        raise SystemExit(f"{path}: payload {pix.size} != {w}x{h}x3")
    return pix.reshape(h, w, 3).copy()


def resize_float(rgb, size=ENC_SIZE):
    """Half-pixel bilinear in float, the same sampling grid as sam2_dml.cpp."""
    h, w = rgb.shape[:2]
    src = rgb.astype(np.float32)

    dx = np.arange(size, dtype=np.float32)
    sx = (dx + 0.5) * (np.float32(w) / np.float32(size)) - 0.5
    sx = np.clip(sx, 0.0, float(w - 1))
    x0 = np.floor(sx).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (sx - x0).astype(np.float32)[None, :, None]

    dy = np.arange(size, dtype=np.float32)
    sy = (dy + 0.5) * (np.float32(h) / np.float32(size)) - 0.5
    sy = np.clip(sy, 0.0, float(h - 1))
    y0 = np.floor(sy).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    fy = (sy - y0).astype(np.float32)[:, None, None]

    top = src[y0][:, x0] * (1.0 - fx) + src[y0][:, x1] * fx
    bot = src[y1][:, x0] * (1.0 - fx) + src[y1][:, x1] * fx
    return top * (1.0 - fy) + bot * fy


def resize_pil(rgb, size=ENC_SIZE):
    im = Image.fromarray(rgb, mode="RGB")
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)


def preprocess(rgb, mode):
    resized = resize_float(rgb) if mode == "float" else resize_pil(rgb)
    norm = (resized / 255.0 - MEAN) / STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def make_session(path, device_id):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Keep stdout/stderr readable: the default WARNING level prints graph-merge
    # warnings that the C runtime emits as UTF-16.
    so.log_severity_level = 3
    providers = [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def timed(runs, fn):
    fn()  # warmup
    times = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, min(times), sum(times) / len(times), max(times)


def write_logits(path, masks):
    if masks.ndim != 3:
        raise SystemExit(f"masks must be [num][h][w], got {masks.shape}")
    num, h, w = masks.shape
    payload = np.ascontiguousarray(masks, dtype=np.float32).tobytes()
    with open(path, "wb") as f:
        f.write(LOGITS_MAGIC)
        f.write(struct.pack("<iii", num, h, w))
        f.write(payload)
    log(f"wrote {path}: {num} x {h} x {w} float32 ({len(payload)} bytes payload, 20-byte header)")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap.add_argument("--image", default=os.path.join(here, "out", "test_image.png"),
                    help="source JPEG/PNG (ignored when --rgb is given)")
    ap.add_argument("--rgb", default=None,
                    help="go_rgb.bin dumped by the Go QC binary (bypasses JPEG decoding)")
    ap.add_argument("--resize", choices=("float", "pil"), default="float",
                    help="float = same bilinear as the DLL, pil = PIL Image.BILINEAR")
    ap.add_argument("--out", default=os.path.join(here, "ref_logits.bin"))
    ap.add_argument("--encoder", default=os.path.join(repo, "sam2_hiera_small.encoder.onnx"))
    ap.add_argument("--decoder", default=os.path.join(repo, "sam2_hiera_small.decoder.onnx"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dump-input", default=None, help="optional path for the 1x3x1024x1024 input tensor")
    args = ap.parse_args()

    log(f"onnxruntime     : {ort.__version__}  providers={ort.get_available_providers()}")
    for p in (args.encoder, args.decoder):
        if not os.path.isfile(p):
            raise SystemExit(f"missing model: {p}")

    if args.rgb:
        rgb = load_go_rgb(args.rgb)
        src_desc = f"{args.rgb} (Go RGB24 dump)"
    else:
        rgb = load_image_rgb(args.image)
        src_desc = f"{args.image} (PIL decode)"
    h, w = rgb.shape[:2]
    log(f"image           : {src_desc} {w}x{h}")
    log(f"resize mode     : {args.resize}")

    tensor = preprocess(rgb, args.resize)
    if args.dump_input:
        np.ascontiguousarray(tensor, dtype=np.float32).tofile(args.dump_input)
        log(f"input tensor    : {args.dump_input} shape={tensor.shape}")

    enc = make_session(args.encoder, args.device)
    dec = make_session(args.decoder, args.device)
    log(f"encoder provider: {enc.get_providers()}")
    log(f"decoder provider: {dec.get_providers()}")

    enc_in = enc.get_inputs()[0].name
    enc_out_names = [o.name for o in enc.get_outputs()]
    (enc_out, *enc_t) = timed(args.runs, lambda: enc.run(enc_out_names, {enc_in: tensor}))
    log(f"encoder         : min {ms(enc_t[0])} / avg {ms(enc_t[1])} / max {ms(enc_t[2])} ms ({args.runs} runs)")
    outs = dict(zip(enc_out_names, enc_out))
    hr0 = outs["high_res_feats_0"]
    hr1 = outs["high_res_feats_1"]
    emb = outs["image_embed"]
    log(f"shapes          : hr0 {hr0.shape} hr1 {hr1.shape} embed {emb.shape}")

    point_coords = np.array([[[w / 2.0 / w * ENC_SIZE, h / 2.0 / h * ENC_SIZE]]], dtype=np.float32)
    point_labels = np.array([[1]], dtype=np.float32)
    mask_input = np.zeros((1, 1, ENC_SIZE // 4, ENC_SIZE // 4), dtype=np.float32)
    has_mask_input = np.array([0], dtype=np.float32)
    feeds = {
        "image_embed": emb,
        "high_res_feats_0": hr0,
        "high_res_feats_1": hr1,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
    }
    # The decoder consumes fixed input names; keep the model's own order.
    feeds = {i.name: feeds[i.name] for i in dec.get_inputs()}
    log(f"prompt          : center point ({point_coords[0, 0, 0]:.1f}, {point_coords[0, 0, 1]:.1f}) label=1")

    dec_out_names = [o.name for o in dec.get_outputs()]
    (dec_out, *dec_t) = timed(args.runs, lambda: dec.run(dec_out_names, feeds))
    log(f"decoder         : min {ms(dec_t[0])} / avg {ms(dec_t[1])} / max {ms(dec_t[2])} ms ({args.runs} runs)")

    masks = np.asarray(dec_out[0], dtype=np.float32)  # [1, num_masks, h, w]
    iou = np.asarray(dec_out[1], dtype=np.float32).reshape(-1)
    if masks.ndim != 4 or masks.shape[0] != 1:
        raise SystemExit(f"unexpected masks shape {masks.shape}")
    masks = masks[0]
    for i, v in enumerate(iou):
        log(f"  mask[{i}] iou    : {v:.6f}")
    best = int(np.argmax(iou))
    flat = masks[best].reshape(-1)
    ratio = float((flat > 0).mean())
    log(f"best mask       : index {best}, iou {iou[best]:.6f}")
    log(f"positive ratio  : {(flat > 0).sum()} / {flat.size} = {ratio:.4f} (logit > 0)")
    log(f"best mask stats : logit min {flat.min():.4f} / max {flat.max():.4f} / mean {flat.mean():.4f}")

    write_logits(args.out, masks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
