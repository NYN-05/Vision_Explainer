"""Experiment orchestrator: baseline → perturb → infer → evaluate → compare → report.

Stages (each recorded in the stage manifest, failures recorded not skipped):
  validate, baseline_config, baseline_infer, perturb, perturbed_infer,
  evaluate, compare, explain, aggregate, report.

Detection-only runs never touch Gemma. Explanation eval is opt-in and marked
in the report. GPU work goes through YoloService (internally serialized).
Long runs execute in a daemon thread; status is polled from disk.
"""
from __future__ import annotations
import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend import lab_compare, lab_metrics
from backend.lab_explain import evaluate_explanation, load_claims
from backend.lab_fixtures import make_still_life
from backend.perturb import OP_REGISTRY, apply_perturbation
from backend.utils import atomic_write_text, build_evidence

STAGES = ("validate", "baseline_config", "baseline_infer", "perturb",
          "perturbed_infer", "evaluate", "compare", "explain", "aggregate", "report")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_config(cfg: dict) -> list[str]:
    errs = []
    if not isinstance(cfg, dict):
        return ["config must be an object"]
    if cfg.get("task", "detect") != "detect":
        errs.append("v1 supports task=detect only")
    for k in ("conf", "iou"):
        v = cfg.get(k)
        if v is not None and not (0.05 <= float(v) <= 0.95):
            errs.append(f"{k}={v} out of range [0.05, 0.95]")
    ds = cfg.get("dataset") or {}
    if not ds.get("demo") and not (ds.get("gt") and ds.get("images")):
        errs.append("dataset needs {demo:true} or {gt:<coco.json>, images:<dir>}")
    for p in cfg.get("perturbations", []):
        if p.get("op") not in OP_REGISTRY:
            errs.append(f"unknown perturbation '{p.get('op')}'")
        for s in p.get("severities", []):
            if s not in (1, 2, 3):
                errs.append(f"severity must be 1/2/3, got {s}")
        if not isinstance(p.get("seed", 0), int):
            errs.append("seed must be int")
    if not cfg.get("perturbations"):
        errs.append("at least one perturbation required")
    return errs


def _to_coco_preds(image_id, detections: list[dict]) -> list[dict]:
    out = []
    for d in detections or []:
        if not isinstance(d, dict) or "cls" not in d or "conf" not in d:
            continue
        if all(k in d for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (float(d[k]) for k in ("x1", "y1", "x2", "y2"))
            out.append({"image_id": image_id, "category_id": int(d["cls"]),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(d["conf"])})
    return out


class LabRunner:
    def __init__(self, yolo, gemma, lab_dir: str | Path):
        self.yolo = yolo
        self.gemma = gemma
        self.lab_dir = Path(lab_dir)
        self._threads: dict[str, threading.Thread] = {}

    # ---- public API ----
    def start(self, cfg: dict) -> dict:
        errs = validate_config(cfg)
        exp_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        exp_dir = self.lab_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(exp_dir / "config.json", json.dumps(cfg, indent=2))
        state = {"exp_id": exp_id, "status": "validation_failed" if errs else "queued",
                 "errors": errs, "stage": "validate", "stages": {}, "updated_at": _now()}
        atomic_write_text(exp_dir / "status.json", json.dumps(state, indent=2))
        if not errs:
            t = threading.Thread(target=self._execute, args=(exp_id, cfg), daemon=True)
            self._threads[exp_id] = t
            t.start()
        return {"exp_id": exp_id, "status": state["status"], "errors": errs}

    def status(self, exp_id: str) -> dict | None:
        p = self.lab_dir / exp_id / "status.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"exp_id": exp_id, "status": "unknown"}

    def report(self, exp_id: str) -> dict | None:
        p = self.lab_dir / exp_id / "report.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list(self) -> list[dict]:
        out = []
        if self.lab_dir.exists():
            for d in sorted(self.lab_dir.iterdir(), reverse=True):
                if (d / "status.json").exists():
                    s = self.status(d.name) or {}
                    out.append({"exp_id": d.name, "status": s.get("status"),
                                "stage": s.get("stage"), "updated_at": s.get("updated_at")})
        return out

    # ---- internals ----
    def _mark(self, exp_dir: Path, state: dict, stage: str, **kw) -> None:
        state["stage"] = stage
        state["stages"][stage] = {"at": _now(), **kw}
        state["updated_at"] = _now()
        atomic_write_text(exp_dir / "status.json", json.dumps(state, indent=2))

    def _execute(self, exp_id: str, cfg: dict) -> None:
        exp_dir = self.lab_dir / exp_id
        state = {"exp_id": exp_id, "status": "running", "errors": [],
                 "stage": "validate", "stages": {"validate": {"at": _now(), "ok": True}},
                 "updated_at": _now()}
        task = cfg.get("task", "detect")
        conf, iou = cfg.get("conf"), cfg.get("iou")
        cmp_cfg = cfg.get("compare", {})
        try:
            # baseline_config: resolve dataset (demo fixtures generated in-exp_dir)
            self._mark(exp_dir, state, "baseline_config")
            ds = cfg["dataset"]
            if ds.get("demo"):
                img_path, gt = make_still_life(exp_dir / "dataset")
                img_dir = exp_dir / "dataset"
            else:
                gt = json.loads(Path(ds["gt"]).read_text(encoding="utf-8"))
                img_dir = Path(ds["images"])
            ann_dir = exp_dir / "annotated"
            images = gt.get("images", [])

            # baseline_infer
            self._mark(exp_dir, state, "baseline_infer")
            base_dets: dict = {}
            for im in images:
                p = img_dir / im["file_name"]
                try:
                    r = self.yolo.predict(p, task, ann_dir, conf=conf, iou=iou,
                                          ann_name=f"base_{im['id']}.jpg")
                    base_dets[str(im["id"])] = r["detections"]
                except Exception as e:
                    state["errors"].append(f"baseline {im.get('file_name')}: {type(e).__name__}: {e}")
                    base_dets[str(im["id"])] = []
            base_preds = [pr for im in images for pr in _to_coco_preds(im["id"], base_dets[str(im["id"])])]
            base_metrics = lab_metrics.evaluate(gt, base_preds)

            # perturb + perturbed_infer
            self._mark(exp_dir, state, "perturb")
            conditions = []
            for spec in cfg["perturbations"]:
                for sev in spec.get("severities", [1]):
                    seed = int(spec.get("seed", 0))
                    precs, gt_variant_anns, invalid_total = [], [], 0
                    variant_gt = {"images": images, "annotations": [],
                                  "categories": gt.get("categories", [])}
                    ok = True
                    for im in images:
                        src = img_dir / im["file_name"]
                        outp = exp_dir / "perturbed" / f"{spec['op']}_s{sev}" / im["file_name"]
                        rec = apply_perturbation(src, spec["op"], sev, seed, outp,
                                                 spec.get("params"))
                        remap_fn = rec.pop("_remap_fn", None)
                        if "error" in rec:
                            ok = False
                        g_boxes = [{"id": a.get("id"), "bbox": list(a["bbox"])}
                                   for a in gt.get("annotations", []) if a.get("image_id") == im["id"]]
                        if "error" not in rec and remap_fn is not None:
                            kept, invalid = remap_fn(list(g_boxes), None)
                            invalid_total += len(invalid)
                            rec["invalid"] = invalid
                        else:
                            kept = []
                        for a in gt.get("annotations", []):
                            if a.get("image_id") == im["id"] and any(k["id"] == a.get("id") for k in kept):
                                na = dict(a)
                                nb = next(k["bbox"] for k in kept if k["id"] == a.get("id"))
                                na["bbox"] = nb
                                variant_gt["annotations"].append(na)
                        precs.append(rec)
                    self._mark(exp_dir, state, "perturbed_infer",
                               info=f"{spec['op']} s{sev}")
                    cond_preds: list[dict] = []
                    cond_dets: dict = {}
                    if ok:
                        for im, rec in zip(images, precs):
                            if "error" in rec:
                                cond_dets[str(im["id"])] = []
                                continue
                            try:
                                r = self.yolo.predict(rec["output"], task, ann_dir,
                                                      conf=conf, iou=iou,
                                                      ann_name=f"{spec['op']}_s{sev}_{im['id']}.jpg")
                                cond_dets[str(im["id"])] = r["detections"]
                            except Exception as e:
                                state["errors"].append(
                                    f"{spec['op']} s{sev} {im.get('file_name')}: {type(e).__name__}: {e}")
                                cond_dets[str(im["id"])] = []
                    cond_preds = [pr for im in images for pr in _to_coco_preds(im["id"], cond_dets[str(im["id"])])]
                    conditions.append({"op": spec["op"], "severity": sev, "seed": seed,
                                       "perturbations": precs, "detections": cond_dets,
                                       "variant_gt": variant_gt, "invalid_total": invalid_total,
                                       "preds": cond_preds})

            # evaluate
            self._mark(exp_dir, state, "evaluate")
            for c in conditions:
                c["metrics"] = lab_metrics.evaluate(
                    {"images": images, "annotations": c["variant_gt"]["annotations"],
                     "categories": gt.get("categories", [])}, c["preds"])
                c["degradation"] = lab_metrics.degradation(base_metrics, c["metrics"])

            # compare
            self._mark(exp_dir, state, "compare")
            for c in conditions:
                per_img, stabs = {}, []
                for im in images:
                    r = lab_compare.compare(base_dets[str(im["id"])],
                                            c["detections"][str(im["id"])], **cmp_cfg)
                    per_img[str(im["id"])] = r
                    if r["stability"] is not None:
                        stabs.append(r["stability"])
                c["comparison"] = per_img
                c["mean_stability"] = round(sum(stabs) / len(stabs), 4) if stabs else None
                del c["preds"]
                del c["detections"]

            # explain (opt-in; detection-only when disabled or Gemma down)
            self._mark(exp_dir, state, "explain")
            explain_rep: dict = {"enabled": bool(cfg.get("explain", False)), "results": [],
                                 "skipped": ""}
            if explain_rep["enabled"]:
                if self.gemma is None or not self.gemma.health().get("ok"):
                    explain_rep["skipped"] = "gemma unavailable"
                else:
                    from backend.agents import ReasonerAgent
                    claims_path = cfg.get("claims")
                    claims_by_image: dict = {}
                    if claims_path and Path(claims_path).exists():
                        claims_by_image["*"] = load_claims(claims_path)
                    n = 0
                    for im in images[: int(cfg.get("explain_max_images", 2))]:
                        cl = claims_by_image.get("*", [])
                        dets = base_dets[str(im["id"])]
                        ev = build_evidence(dets, task)
                        out, _ = ReasonerAgent(self.gemma).run(
                            img_dir / im["file_name"],
                            cfg.get("query", "Describe this image in full detail using the YOLO detections."),
                            {"detections": dets, "model": task, "task": task},
                            crops=None, evidence=ev)
                        explain_rep["results"].append(
                            {"image": im.get("file_name"),
                             **evaluate_explanation(out.get("answer", ""),
                                                    [e["id"] for e in ev], cl)})
                        n += 1
                    if not n:
                        explain_rep["skipped"] = "no images evaluated"

            # aggregate
            self._mark(exp_dir, state, "aggregate")
            failures = []
            for c in conditions:
                for img_id, r in c["comparison"].items():
                    score = len(r["disappeared"]) + len(r["misclassified"]) + len(r["drifted"])
                    if score:
                        failures.append({"op": c["op"], "severity": c["severity"],
                                         "image_id": img_id, "score": score,
                                         "disappeared": len(r["disappeared"]),
                                         "misclassified": len(r["misclassified"]),
                                         "drifted": len(r["drifted"]),
                                         "new_fps": len(r["new_false_positives"])})
            failures.sort(key=lambda f: -f["score"])
            topk = failures[: int(cfg.get("save_failures_top_k", 5))]
            for f in topk:
                src = next((p["output"] for c in conditions
                            if c["op"] == f["op"] and c["severity"] == f["severity"]
                            for p in c["perturbations"]
                            if Path(p.get("output", "")).name ==
                            next(i["file_name"] for i in images if str(i["id"]) == str(f["image_id"]))), None)
                if src and Path(src).exists():
                    dest = exp_dir / "failures" / f"{f['op']}_s{f['severity']}_{f['image_id']}.jpg"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dest)
                    f["artifact"] = f"/outputs/lab/{exp_id}/failures/{dest.name}"

            # report
            self._mark(exp_dir, state, "report")
            try:
                import torch as _t
                prov = {"torch": _t.__version__}
            except Exception:
                prov = {}
            try:
                import ultralytics as _u
                prov["ultralytics"] = _u.__version__
            except Exception:
                pass
            report = {
                "exp_id": exp_id, "created_at": _now(), "product": "vision-reliability-lab",
                "config": cfg, "stages": state["stages"], "errors": state["errors"],
                "baseline_metrics": base_metrics,
                "conditions": [{k: c[k] for k in ("op", "severity", "seed", "metrics",
                                                  "degradation", "mean_stability",
                                                  "invalid_total", "comparison")} for c in conditions],
                "failures": topk, "explanation": explain_rep,
                "consistency_note": "stability measures output repeatability, not correctness; "
                                    "see baseline_metrics/conditions metrics for accuracy",
                "provenance": prov,
            }
            atomic_write_text(exp_dir / "report.json", json.dumps(report, indent=2))
            state["status"] = "done"
            self._mark(exp_dir, state, "report", ok=True)
        except Exception as e:
            state["status"] = "failed"
            state["errors"].append(f"orchestrator: {type(e).__name__}: {e}")
            self._mark(exp_dir, state, "report", ok=False)
