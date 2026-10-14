"""Ground-truth detection evaluation (COCO-style, dependency-free).

Ground-truth format (COCO-compatible subset):
  {"images": [{"id", "file_name", "width", "height"}],
   "annotations": [{"id", "image_id", "category_id", "bbox": [x,y,w,h]}],
   "categories": [{"id", "name"}]}
Predictions: [{"image_id", "category_id", "bbox": [x,y,w,h], "score"}]

Matching: per (image, category), predictions sorted by score descending, greedy
match to the highest-IoU unmatched GT at IoU >= threshold. Duplicates are false
positives. AP uses COCO 101-point recall interpolation over IoU 0.5:0.05:0.95.
Classes with zero ground truth get ap=None and are excluded from means (their
counts are still reported). No GT and no predictions on an image counts as
correct-but-vacuous (recorded, not scored).
"""
from __future__ import annotations

IOU_THRS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def iou_xywh(a: list | tuple, b: list | tuple) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match(image_gt: list[dict], image_preds: list[dict], thr: float) -> tuple[int, int, list[float], list[int]]:
    """Greedy match. Returns (tp, fp, scores_sorted, tp_flags_aligned)."""
    preds = sorted(image_preds, key=lambda p: float(p.get("score", 0.0)), reverse=True)
    used = [False] * len(image_gt)
    tp_flags, scores = [], []
    for p in preds:
        best, best_i = thr, -1
        for i, g in enumerate(image_gt):
            if used[i]:
                continue
            v = iou_xywh(p["bbox"], g["bbox"])
            if v >= best:
                best, best_i = v, i
        scores.append(float(p.get("score", 0.0)))
        if best_i >= 0:
            used[best_i] = True
            tp_flags.append(1)
        else:
            tp_flags.append(0)
    tp = sum(tp_flags)
    return tp, len(preds) - tp, scores, tp_flags


def _ap_101(tp_flags: list[int], n_gt: int) -> float | None:
    if n_gt == 0:
        return None  # no ground truth: class excluded from means
    if not tp_flags:
        return 0.0  # ground truth exists but nothing predicted: worst score
    tp_cum, prec, rec = 0, [], []
    for i, f in enumerate(tp_flags):
        tp_cum += f
        prec.append(tp_cum / (i + 1))
        rec.append(tp_cum / n_gt)
    ap = 0.0
    for t in (i / 100 for i in range(101)):
        p = max([pr for pr, r in zip(prec, rec) if r >= t], default=0.0)
        ap += p
    return round(ap / 101, 4)


def evaluate(gt: dict, preds: list[dict], iou_thrs: tuple = IOU_THRS,
             cat_ids: list | None = None) -> dict:
    """Full COCO-style eval. Never raises on empty/degenerate inputs."""
    gt = gt or {}
    preds = preds or []
    images = {im["id"]: im for im in gt.get("images", [])}
    anns = gt.get("annotations", [])
    cats = {c["id"]: c.get("name", str(c["id"])) for c in gt.get("categories", [])}
    if cat_ids is None:
        cat_ids = sorted({a["category_id"] for a in anns} | {p.get("category_id") for p in preds
                                                             if "category_id" in p})
    by_img_cat: dict = {}
    for a in anns:
        by_img_cat.setdefault((a["image_id"], a["category_id"]), []).append(a)
    by_img_cat_p: dict = {}
    for p in preds:
        if "category_id" not in p or "bbox" not in p:
            continue
        by_img_cat_p.setdefault((p.get("image_id"), p["category_id"]), []).append(p)

    per_class, ap50s, ap_all = {}, [], []
    total_tp = total_fp = total_fn = 0
    per_image: dict = {}
    for cid in cat_ids:
        aps, ap50 = [], None
        c_tp = c_fp = c_fn = 0
        for thr in iou_thrs:
            t_tp = t_fp = 0
            for img_id in set([k[0] for k in list(by_img_cat) + list(by_img_cat_p)
                                if k[1] == cid] + [im for im in images]):
                g = by_img_cat.get((img_id, cid), [])
                pr = by_img_cat_p.get((img_id, cid), [])
                tp, fp, _, _ = _match(g, pr, thr)
                t_tp += tp
                t_fp += fp
            n_gt = sum(len(by_img_cat.get((img_id, cid), [])) for img_id in images)
            flags: list[int] = []
            for img_id in sorted(images):
                _, _, _, fl = _match(by_img_cat.get((img_id, cid), []),
                                     by_img_cat_p.get((img_id, cid), []), thr)
                flags.extend(fl)
            ap = _ap_101(flags, n_gt)
            if ap is not None:
                aps.append(ap)
                if abs(thr - 0.5) < 1e-9:
                    ap50 = ap
            if abs(thr - 0.5) < 1e-9:
                c_tp, c_fp = t_tp, t_fp
                c_fn = n_gt - t_tp
        mean_ap = round(sum(aps) / len(aps), 4) if aps else None
        if mean_ap is not None:
            ap_all.append(mean_ap)
        if ap50 is not None:
            ap50s.append(ap50)
        total_tp += c_tp
        total_fp += c_fp
        total_fn += c_fn
        per_class[str(cid)] = {"name": cats.get(cid, str(cid)), "ap": mean_ap, "ap50": ap50,
                               "precision": round(c_tp / (c_tp + c_fp), 4) if c_tp + c_fp else None,
                               "recall": round(c_tp / (c_tp + c_fn), 4) if c_tp + c_fn else None,
                               "tp": c_tp, "fp": c_fp, "fn": c_fn,
                               "n_gt": sum(len(by_img_cat.get((i, cid), [])) for i in images)}

    for img_id in images:
        itp = ifp = ifn = 0
        for cid in cat_ids:
            tp, fp, _, _ = _match(by_img_cat.get((img_id, cid), []),
                                  by_img_cat_p.get((img_id, cid), []), 0.5)
            itp += tp
            ifp += fp
            ifn += len(by_img_cat.get((img_id, cid), [])) - tp
        per_image[str(img_id)] = {"tp": itp, "fp": ifp, "fn": ifn}

    n_gt_total = sum(len(v) for k, v in by_img_cat.items())
    return {
        "mAP": round(sum(ap_all) / len(ap_all), 4) if ap_all else None,
        "mAP50": round(sum(ap50s) / len(ap50s), 4) if ap50s else None,
        "precision": round(total_tp / (total_tp + total_fp), 4) if total_tp + total_fp else None,
        "recall": round(total_tp / (total_tp + total_fn), 4) if total_tp + total_fn else None,
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "n_gt": n_gt_total,
        "n_pred": len(preds), "per_class": per_class, "per_image": per_image,
        "iou_thresholds": list(iou_thrs),
    }


def degradation(baseline: dict, perturbed: dict) -> dict:
    """Metric deltas (perturbed − baseline). None-safe; lower is worse for mAP/P/R."""
    out = {}
    for k in ("mAP", "mAP50", "precision", "recall"):
        b, p = baseline.get(k), perturbed.get(k)
        out[k + "_delta"] = round(p - b, 4) if isinstance(b, (int, float)) and isinstance(p, (int, float)) else None
    return out
