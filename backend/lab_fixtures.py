"""Synthetic demo fixtures (no dataset needed for v1).

make_still_life() draws a red square + blue circle on white and returns
(image_path, coco_gt). The shapes are out-of-distribution for COCO YOLO, which
is intentional: the demo shows graceful low scores, measurable degradation and
stability-vs-accuracy separation — not inflated accuracy. Real evaluation needs
a labeled COCO-format dataset (see docs/implementation_audit.md).
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw

CLAIMS_PATH = Path(__file__).resolve().parents[1] / "lab" / "claims" / "still_life.json"


def make_still_life(out_dir: str | Path, size: tuple[int, int] = (640, 480)) -> tuple[str, dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    sq = (80, 140, 260, 320)       # red square, left
    ci = (380, 140, 560, 320)      # blue circle, right
    d.rectangle(sq, fill=(220, 30, 30))
    d.ellipse(ci, fill=(30, 80, 220))
    ip = out_dir / "still_life.jpg"
    img.save(ip, "JPEG", quality=92)
    gt = {
        "images": [{"id": 1, "file_name": "still_life.jpg",
                    "width": size[0], "height": size[1]}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1000,
             "bbox": [sq[0], sq[1], sq[2] - sq[0], sq[3] - sq[1]]},
            {"id": 2, "image_id": 1, "category_id": 1001,
             "bbox": [ci[0], ci[1], ci[2] - ci[0], ci[3] - ci[1]]},
        ],
        "categories": [{"id": 1000, "name": "square"}, {"id": 1001, "name": "circle"}],
    }
    return str(ip), gt
