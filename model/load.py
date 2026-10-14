"""Download every YOLO26 checkpoint: all sizes x all tasks (5 x 7 = 35 files).

Run from the directory where you want the weights (e.g. model/):
    cd model && python load.py
"""
from pathlib import Path
import shutil

from ultralytics import YOLO

SIZES = ["n", "s", "m", "l", "x"]
TASKS = ["", "-seg", "-pose", "-obb", "-cls", "-sem", "-depth"]

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
    names = [f"yolo26{s}{t}.pt" for s in SIZES for t in TASKS]
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
