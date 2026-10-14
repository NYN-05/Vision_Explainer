"""Download YOLO26 X-series checkpoints (1 size x 7 tasks = 7 files).

Run from anywhere (weights always land in this file's folder):
    python model/load.py
"""
from pathlib import Path
import shutil

from ultralytics import YOLO

# HOW TO CHANGE WHAT GETS DOWNLOADED:
#   - Sizes:  add/remove letters from SIZES. Valid YOLO sizes are
#             "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (x-large).
#             Example: SIZES = ["n", "s"] downloads only nano + small.
#             NOTE: this app runs X-only (see config.yaml), so other sizes are
#             just local inventory unless you change the app config too.
#   - Tasks:  add/remove suffixes from TASKS. "" = detection, "-seg" = instance
#             segmentation, "-pose" = pose, "-obb" = oriented boxes,
#             "-cls" = classification, "-sem" = semantic segmentation,
#             "-depth" = depth. Example: TASKS = [""] downloads detection only.
#   - YOLO generation: file names follow yolo<VERSION><size><task>.pt.
#             This script uses VERSION = "26". For another generation
#             (e.g. YOLOv8/v11), change PREFIX below, e.g. PREFIX = "yolov8".
SIZES = ["x"]
TASKS = ["", "-seg", "-pose", "-obb", "-cls", "-sem", "-depth"]
PREFIX = "yolo26"

TARGET = Path(__file__).resolve().parent  # .../obj_detection/model


def download(name: str) -> str:
    dest = TARGET / name
    if dest.exists() and dest.stat().st_size > 0:
        return f"Skipped (exists): {name}"
    YOLO(name)  # downloads to cwd if not already cached
    src = Path.cwd() / name
    if src.exists() and src.resolve() != dest.resolve():
        shutil.move(str(src), str(dest))
    return f"Downloaded: {name}"


def main() -> None:
    names = [f"{PREFIX}{s}{t}.pt" for s in SIZES for t in TASKS]
    print(f"Downloading {len(names)} checkpoints sequentially (sizes={SIZES}, tasks={len(TASKS)}) ...")
    ok, failed = 0, []
    for i, n in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {n} ...")
        try:
            print(download(n))
            ok += 1
        except Exception as err:
            print(f"Failed {n}: {err}")
            failed.append(n)
    print(f"Done: {ok}/{len(names)} downloaded" + (f", failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
