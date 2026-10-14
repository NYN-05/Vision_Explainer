"""Prediction comparison: baseline vs perturbed detections (detection task, v1).

Matching is IoU-based and order-independent — detections are never compared by
list position. Thresholds are explicit arguments. Stability (same output under
perturbation) is reported separately from correctness (match to ground truth);
a stable-but-wrong model scores high here and low in lab_metrics, by design.
"""
from __future__ import annotations

from backend.lab_metrics import iou_xywh


def _box(d: dict) -> list:
    return [float(d["x1"]), float(d["y1"]), float(d["x2"] - d["x1"]), float(d["y2"] - d["y1"])]


def _has_box(d: dict) -> bool:
    return all(k in d for k in ("x1", "y1", "x2", "y2"))


def compare(baseline: list[dict], perturbed: list[dict], iou_match: float = 0.5,
            drift_iou: float = 0.7, conf_delta: float = 0.25) -> dict:
    """Compare two detection lists. Pure function; never raises on bad rows."""
    base = [d for d in (baseline or []) if isinstance(d, dict) and _has_box(d)]
    pert = [d for d in (perturbed or []) if isinstance(d, dict) and _has_box(d)]
    base = sorted(base, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    used = [False] * len(pert)
    pairs, disappeared, misclassified, drifted, conf_drops = [], [], [], [], []
    for i, b in enumerate(base):
        best, best_j = iou_match, -1
        for j, p in enumerate(pert):
            if used[j]:
                continue
            v = iou_xywh(_box(b), _box(p))
            if v >= best:
                best, best_j = v, j
        if best_j < 0:
            disappeared.append(i)
            continue
        used[best_j] = True
        p = pert[best_j]
        entry = {"base": i, "pert": best_j, "iou": round(best, 4),
                 "base_label": b.get("label"), "pert_label": p.get("label"),
                 "conf_change": round(float(p.get("conf", 0.0)) - float(b.get("conf", 0.0)), 4)}
        pairs.append(entry)
        if p.get("label") != b.get("label"):
            misclassified.append(entry)
        else:
            if best < drift_iou:
                drifted.append(entry)
            if entry["conf_change"] <= -conf_delta:
                conf_drops.append(entry)
    new_fps = [j for j, u in enumerate(used) if not u]
    tracked_ok = len(pairs) - len(misclassified) - len(drifted)
    n_base = len(base)
    return {
        "n_baseline": n_base, "n_perturbed": len(pert),
        "matched": len(pairs), "disappeared": disappeared,
        "misclassified": misclassified, "drifted": drifted,
        "confidence_drops": conf_drops, "new_false_positives": new_fps,
        # Stability only: fraction of baseline detections tracked with same label
        # and no drift. NOT accuracy — see lab_metrics.evaluate for correctness.
        "stability": round(tracked_ok / n_base, 4) if n_base else None,
    }
