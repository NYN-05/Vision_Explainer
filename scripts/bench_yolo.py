"""Benchmark registered YOLO26X heads on a user-supplied labeled dataset.

Writes outputs/bench/<task>.json with precision, recall, mAP, latency, peak VRAM.
Never invents numbers: every figure comes from an actual YOLO.val() run.
Requires benchmarks.data_root in config.yaml (ultralytics dataset yaml or data dir).

Usage:
    python scripts/bench_yolo.py [--task detect] [--data bench/coco8.yaml]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml as pyyaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import MODEL_REGISTRY, VALID_TASKS  # noqa: E402


def bench_task(weight: str, data: str, device: str) -> dict:
    model = YOLO(weight)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    metrics = model.val(data=data, device=device, verbose=False)
    wall_s = round(time.perf_counter() - t0, 1)
    out = {"weight": weight, "wall_s": wall_s,
           "precision": round(float(getattr(metrics.box, "mp", 0) or 0), 4),
           "recall": round(float(getattr(metrics.box, "mr", 0) or 0), 4),
           "mAP50": round(float(getattr(metrics.box, "map50", 0) or 0), 4),
           "mAP50_95": round(float(getattr(metrics.box, "map", 0) or 0), 4)}
    try:
        out["latency_ms_per_img"] = round(float(metrics.speed["inference"]), 2)
    except Exception:
        pass
    if torch.cuda.is_available() and device == "cuda":
        try:
            out["peak_vram_MB"] = round(torch.cuda.max_memory_allocated(0) / 1e6)
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="", help="single task or empty for all present heads")
    ap.add_argument("--data", default="", help="dataset yaml/dir (overrides config)")
    args = ap.parse_args()
    cfg = pyyaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data = args.data or cfg.get("benchmarks", {}).get("data_root", "")
    if not data:
        print("ERROR: no dataset. Set benchmarks.data_root or pass --data <yaml>.", file=sys.stderr)
        sys.exit(2)
    device = cfg["yolo"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tasks = [args.task] if args.task else list(VALID_TASKS)
    outdir = ROOT / "outputs" / "bench"
    outdir.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        w = ROOT / cfg["yolo"]["weights"][t]
        if not w.exists():
            print(f"SKIP {t}: missing {w}")
            continue
        print(f"bench {t} on {data} ...", flush=True)
        res = bench_task(str(w), data, device)
        res.update({"task": t, "device": device,
                    "domain": MODEL_REGISTRY[t]["domain"],
                    "validation": MODEL_REGISTRY[t]["validation"]})
        (outdir / f"{t}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"  -> {res}")


if __name__ == "__main__":
    main()
