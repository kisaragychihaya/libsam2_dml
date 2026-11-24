"""Generate a PyTorch SAM2 single-mask reference for the Go pure-FFI QC run.

The ncnn decoder was converted with multimask_output=False, so the reference uses
the same branch. --rgb accepts the exact RGB24 buffer dumped by the Go client and
therefore removes image-decoder differences from the comparison.

Usage from the repository root:
    python example/ref_dump.py --rgb example/out/go_rgb.bin \
        --checkpoint sam2_hiera_small.pt --out example/out/ref_logits.bin
"""

import argparse
import os
import struct

import numpy as np
from PIL import Image

ENC_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RGB_MAGIC = b"S2RGB123"
LOGITS_MAGIC = b"S2LOGITS"


def load_image_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


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


def write_logits(path, masks):
    masks = np.ascontiguousarray(masks, dtype=np.float32)
    if masks.ndim != 3:
        raise SystemExit(f"masks must be [num][h][w], got {masks.shape}")
    num, h, w = masks.shape
    with open(path, "wb") as f:
        f.write(LOGITS_MAGIC)
        f.write(struct.pack("<iii", num, h, w))
        f.write(masks.tobytes())
    print(f"wrote {path}: {num} x {h} x {w} float32", flush=True)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap.add_argument("--image", default=os.path.join(here, "out", "test_image.png"))
    ap.add_argument("--rgb", default=None, help="go_rgb.bin dumped by the Go QC client")
    ap.add_argument("--out", default=os.path.join(here, "ref_logits.bin"))
    ap.add_argument("--checkpoint", default=os.path.join(repo, "sam2_hiera_small.pt"))
    ap.add_argument("--config", default="configs/sam2/sam2_hiera_s.yaml")
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    rgb = load_go_rgb(args.rgb) if args.rgb else load_image_rgb(args.image)
    h, w = rgb.shape[:2]
    resized = resize_float(rgb)
    tensor = np.ascontiguousarray(
        ((resized / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None],
        dtype=np.float32,
    )

    import torch
    from sam2.build_sam import build_sam2

    model = build_sam2(args.config, args.checkpoint, device="cpu").eval()
    model.sam_mask_decoder.dynamic_multimask_via_stability = False
    encoder = model.image_encoder
    prompt_encoder = model.sam_prompt_encoder
    decoder = model.sam_mask_decoder
    points = torch.tensor([[[w / 2.0 / w * ENC_SIZE, h / 2.0 / h * ENC_SIZE]]])
    labels = torch.tensor([[1]], dtype=torch.int64)

    with torch.no_grad():
        features = encoder(torch.from_numpy(tensor))
        vision = features["vision_features"]
        hr0, hr1 = features["backbone_fpn"][0], features["backbone_fpn"][1]
        sparse = prompt_encoder._embed_points(points, labels, pad=True)
        dense = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            1, -1, vision.shape[-2], vision.shape[-1]
        )
        masks, iou, _, _ = decoder(
            image_embeddings=vision,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[decoder.conv_s0(hr0), decoder.conv_s1(hr1)],
        )

    masks_np = masks.numpy()[0]
    iou_np = iou.numpy().reshape(-1)
    print(f"PyTorch reference: {w}x{h}, center point, mask shape {masks_np.shape}", flush=True)
    print(f"mask[0] iou: {float(iou_np[0]):.7f}", flush=True)
    write_logits(args.out, masks_np)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
