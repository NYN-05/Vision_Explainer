"""Deterministic image perturbation engine for robustness testing.

Each op is registered with a kind (photometric = pixels only, geometry/visibility
unchanged; geometric = boxes must be remapped or explicitly invalidated), three
severity levels, and an apply function. All randomness comes from a seeded
``random.Random`` + numpy Generator so fixed (image, op, severity, seed) inputs
give byte-identical outputs. Failures are recorded, never silently skipped.

New ops: add an entry to OP_REGISTRY — no pipeline changes needed.
"""
from __future__ import annotations
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

MAX_PERTURB_MP = 50.0  # resource guard: refuse absurd inputs
SEVERITIES = (1, 2, 3)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(img: Image.Image) -> None:
    if img.width * img.height / 1e6 > MAX_PERTURB_MP:
        raise ValueError(f"image too large ({img.width}x{img.height})")


def _identity(boxes: list, *a) -> tuple[list, list]:
    return boxes, []


def _blur(img, rng, sev, params):
    return img.filter(ImageFilter.GaussianBlur(radius={1: 1.0, 2: 2.5, 3: 5.0}[sev])), _identity


def _brightness(img, rng, sev, params):
    return ImageEnhance.Brightness(img).enhance({1: 1.25, 2: 1.6, 3: 0.55}[sev]), _identity


def _contrast(img, rng, sev, params):
    return ImageEnhance.Contrast(img).enhance({1: 1.3, 2: 1.7, 3: 0.5}[sev]), _identity


def _noise(img, rng, sev, params):
    std = {1: 8.0, 2: 18.0, 3: 35.0}[sev]
    gen = np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    arr = np.asarray(img).astype(np.float32) + gen.normal(0, std, (*img.size[::-1], 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), _identity


def _jpeg(img, rng, sev, params):
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality={1: 75, 2: 45, 3: 15}[sev])
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out, _identity


def _resize(img, rng, sev, params):
    # Downscale then restore original size: tests resolution robustness while
    # boxes stay aligned (identity remap). Final size always == input size.
    s = {1: 0.75, 2: 0.5, 3: 0.25}[sev]
    small = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
    return small.resize(img.size, Image.BICUBIC), _identity


def _occlusion(img, rng, sev, params):
    frac = {1: 0.10, 2: 0.20, 3: 0.35}[sev]
    bw, bh = int(img.width * frac), int(img.height * frac)
    x0, y0 = rng.randint(0, max(0, img.width - bw)), rng.randint(0, max(0, img.height - bh))
    occ = (x0, y0, x0 + bw, y0 + bh)
    out = img.copy()
    solid = Image.new("RGB", (bw, bh), tuple(params.get("color", (0, 0, 0))) if params else (0, 0, 0))
    out.paste(solid, (x0, y0))

    def remap(boxes, *_):
        kept, invalid = [], []
        ox0, oy0, ox1, oy1 = occ
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            inter = max(0, min(x2, ox1) - max(x1, ox0)) * max(0, min(y2, oy1) - max(y1, oy0))
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if inter / area > 0.5:
                invalid.append({"id": b.get("id"), "reason": f">50% occluded by {occ}"})
            else:
                kept.append(b)
        return kept, invalid

    return out, remap


def _crop(img, rng, sev, params):
    # Random (seeded) crop resized back to original size; boxes remapped by
    # translate+scale, clipped, dropped (<20% remaining) as invalid with reason.
    frac = {1: 0.9, 2: 0.75, 3: 0.6}[sev]
    cw, ch = int(img.width * frac), int(img.height * frac)
    x0 = rng.randint(0, img.width - cw) if img.width > cw else 0
    y0 = rng.randint(0, img.height - ch) if img.height > ch else 0
    out = img.crop((x0, y0, x0 + cw, y0 + ch)).resize(img.size, Image.BICUBIC)
    sx, sy = img.width / cw, img.height / ch

    def remap(boxes, *_):
        kept, invalid = [], []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            nx1, ny1 = (x1 - x0) * sx, (y1 - y0) * sy
            nx2, ny2 = (x2 - x0) * sx, (y2 - y0) * sy
            cx1, cy1 = max(0.0, nx1), max(0.0, ny1)
            cx2, cy2 = min(float(img.width), nx2), min(float(img.height), ny2)
            orig = max(1.0, (x2 - x1) * (y2 - y1))
            rem = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            if rem / orig < 0.2:
                invalid.append({"id": b.get("id"), "reason": f"<20% remains after crop {(x0, y0, cw, ch)}"})
            else:
                nb = dict(b)
                nb["bbox"] = [cx1, cy1, cx2, cy2]
                kept.append(nb)
        return kept, invalid

    return out, remap


OP_REGISTRY: dict[str, dict] = {
    "gaussian_blur": {"kind": "photometric", "apply": _blur},
    "brightness": {"kind": "photometric", "apply": _brightness},
    "contrast": {"kind": "photometric", "apply": _contrast},
    "gaussian_noise": {"kind": "photometric", "apply": _noise},
    "jpeg_compression": {"kind": "photometric", "apply": _jpeg},
    "resize": {"kind": "geometric", "apply": _resize},
    "occlusion": {"kind": "geometric", "apply": _occlusion},
    "crop": {"kind": "geometric", "apply": _crop},
}


def list_ops() -> list[dict]:
    return [{"name": n, "kind": m["kind"], "severities": list(SEVERITIES)}
            for n, m in OP_REGISTRY.items()]


def apply_perturbation(image_path: str | Path, op: str, severity: int, seed: int,
                       out_path: str | Path, params: dict | None = None) -> dict:
    """Apply one op, save JPEG, return the record (or error record, never raise
    for transform failures; validation errors raise ValueError)."""
    if op not in OP_REGISTRY:
        raise ValueError(f"unknown perturbation '{op}'. Valid: {sorted(OP_REGISTRY)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 1/2/3, got {severity}")
    rec: dict = {"op": op, "kind": OP_REGISTRY[op]["kind"], "severity": severity,
                 "seed": seed, "params": params or {}}
    try:
        with Image.open(image_path).convert("RGB") as im:
            _check(im)
            rng = random.Random(seed)
            out, remap = OP_REGISTRY[op]["apply"](im, rng, severity, params or {})
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.save(out_path, "JPEG", quality=92)
        rec.update({"output": str(out_path), "sha256": _sha256(out_path),
                    "remap": remap.__name__ if remap is not _identity else "identity"})
        rec["_remap_fn"] = remap  # caller-side only; stripped before persistence
    except Exception as e:
        rec.update({"error": f"{type(e).__name__}: {e}"})
    return rec


def remap_boxes(record: dict, boxes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply a perturbation record's remap to COCO-style box dicts.
    Returns (kept, invalid). Identity for photometric ops."""
    fn = record.get("_remap_fn")
    if fn is None:
        return boxes, []
    return fn(list(boxes), None)
