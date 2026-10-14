"""AI agents: Detector -> Reasoner -> Reporter, coordinated by Orchestrator."""
from __future__ import annotations
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.services import MODEL_REGISTRY, hw_info, select_model
from backend.utils import (SYSTEM_PROMPT, SYSTEM_PROMPT_ALL, atomic_write_text,
                           build_evidence, build_fused_summary, build_provenance,
                           build_user_text, build_yolo_summary, crop_detections,
                           save_report, append_history, validate_answer)


# ---- detector (Agent 1: thin wrapper over YoloService) ----
class DetectorAgent:
    def __init__(self, yolo_service):
        self.yolo = yolo_service

    def run(self, image_path: str | Path, task: str, conf: float | None = None,
            iou: float | None = None, annotated_dir: str | Path = "outputs/annotated",
            ann_name: str | None = None) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        result = self.yolo.predict(image_path, task, annotated_dir, conf=conf, iou=iou,
                                   ann_name=ann_name)
        ms = (time.perf_counter() - t0) * 1000.0
        step = {"agent": "detector",
                "action": f"yolo26x-{task}",
                "detail": f"{result['num_objects']} obj(s), conf={result['conf']}, {result['inference_ms']}ms yolo",
                "ms": round(ms, 1)}
        return result, step

    def run_all(self, image_path: str | Path, conf: float | None = None,
                iou: float | None = None,
                annotated_dir: str | Path = "outputs/annotated") -> tuple[dict[str, dict], list[dict]]:
        """Run all 7 X heads sequentially. Returns (fused, steps)."""
        t0 = time.perf_counter()
        fused = self.yolo.predict_all(image_path, annotated_dir, conf=conf, iou=iou)
        ms = (time.perf_counter() - t0) * 1000.0
        steps = []
        for task, res in fused.items():
            steps.append({"agent": "detector", "action": f"yolo26x-{task}",
                          "detail": f"{res['num_objects']} obj(s), {res['inference_ms']}ms yolo",
                          "ms": 0})
        steps.append({"agent": "detector", "action": "yolo26x-all-done",
                      "detail": f"7 heads fused, total {round(ms, 1)}ms", "ms": round(ms, 1)})
        return fused, steps


# ---- reasoner (Agent 2: Gemma-4 vision, bounded verification) ----
class ReasonerAgent:
    def __init__(self, gemma_client, low_conf_threshold: float = 0.35):
        self.gemma = gemma_client
        self.low_conf_threshold = low_conf_threshold

    def run(self, image_path: str | Path, query: str, detection: dict, refine_note: str = "",
            crops: list[dict] | None = None,
            evidence: list[dict] | None = None) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        summary = build_yolo_summary(detection["detections"], detection["model"], detection["task"])
        user_text = build_user_text(query, summary, refine_note, crops, evidence)
        out = self.gemma.chat_vision(image_path, user_text, SYSTEM_PROMPT,
                                     extra_images=[c["path"] for c in (crops or [])])
        ms = (time.perf_counter() - t0) * 1000.0
        step = {"agent": "reasoner", "action": "gemma-4-vision",
                "detail": f"{len(crops or [])} crops, thought={len(out['thought'])}ch answer={len(out['answer'])}ch",
                "ms": round(ms, 1)}
        return out, step

    def run_fused(self, image_path: str | Path, query: str, fused: dict[str, dict],
                  crops: list[dict] | None = None,
                  evidence: list[dict] | None = None) -> tuple[dict, dict]:
        """Single Gemma pass over all 7 heads (no per-head refine — too costly)."""
        t0 = time.perf_counter()
        summary = build_fused_summary(fused)
        user_text = build_user_text(query, summary, crops=crops, evidence=evidence)
        out = self.gemma.chat_vision(image_path, user_text, SYSTEM_PROMPT_ALL,
                                     extra_images=[c["path"] for c in (crops or [])])
        ms = (time.perf_counter() - t0) * 1000.0
        step = {"agent": "reasoner", "action": "gemma-4-vision-all",
                "detail": f"7 heads fused, {len(crops or [])} crops, "
                          f"thought={len(out['thought'])}ch answer={len(out['answer'])}ch",
                "ms": round(ms, 1)}
        return out, step

    # Queries that imply objects are expected (empty result is suspicious, not proof).
    EXPECTS_OBJECTS = ("detect", "count", "find", "where", "how many", "person", "people",
                       "car", "vehicle", "cat", "dog", "animal", "object", "thing")

    def needs_refine(self, detection: dict, query: str = "") -> tuple[bool, str]:
        """Bounded verification signal. Empty/low-conf alone is NOT failure proof.

        Returns True only when justified: low average confidence, or an empty result
        that contradicts an object-expecting query. Callers must ALSO corroborate
        (e.g. Gemma uncertainty) and respect the retry budget.
        """
        dets = detection.get("detections", [])
        if dets:
            confs = [d.get("conf", 0.0) for d in dets
                     if isinstance(d.get("conf"), (int, float))]
            if confs and sum(confs) / len(confs) < self.low_conf_threshold:
                avg = sum(confs) / len(confs)
                return True, f"Average confidence low ({avg:.2f}); retry justified."
            return False, "accepted"
        q = (query or "").lower()
        if any(k in q for k in self.EXPECTS_OBJECTS):
            return True, "Empty result contradicts object-expecting query; one retry justified."
        return False, "empty result accepted as valid — no retry"


# ---- reporter (Agent 3: report dict + disk persist) ----
class ReporterAgent:
    def __init__(self, reports_dir: str | Path = "outputs/reports",
                 history_path: str | Path = "outputs/history.jsonl"):
        self.reports_dir = Path(reports_dir)
        self.history_path = Path(history_path)

    def run(self, image_name: str, query: str, task: str, requested_model: str,
            detection: dict, reason: dict, steps: list[dict],
            annotated_url: str, refined: bool = False,
            all_detections: dict | None = None,
            annotated_gallery: list | None = None,
            per_task: dict | None = None,
            selection_rationale: str = "",
            validation_status=None,
            fallback_actions: list | None = None,
            hardware: dict | None = None,
            run_id: str | None = None,
            run_dir: str | Path | None = None,
            run_files: list | None = None,
            evidence: list | None = None,
            grounding: dict | None = None,
            crop_provenance: list | None = None,
            retry_reason: str = "",
            provenance: dict | None = None) -> dict:
        rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        dets = detection.get("detections", [])
        if task == "all" and all_detections is not None:
            dets = all_detections
        report = {
            "id": rid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": image_name,
            "query": query,
            "task": task,
            "requested_model": requested_model,
            "model": detection.get("model"),
            "device": detection.get("device"),
            "conf": detection.get("conf"),
            "num_objects": detection.get("num_objects", 0),
            "detections": dets,
            "yolo_ms": detection.get("inference_ms"),
            "thought": reason.get("thought", ""),
            "final_answer": reason.get("answer", ""),
            "usage": reason.get("usage", {}),
            "steps": steps,
            "refined": refined,
            "annotated_url": annotated_url,
            "selection_rationale": selection_rationale,
            "validation_status": validation_status if validation_status is not None
            else detection.get("validation", "unvalidated"),
            "fallback_actions": fallback_actions or [],
            "hardware": hardware or {"device": detection.get("device")},
            "run_id": rid,
        }
        if run_files is not None:
            report["run_files"] = run_files
        if all_detections is not None:
            # flat compat list (detect head) for old clients
            flat = all_detections.get("detect", []) if isinstance(all_detections, dict) else []
            report["detections_flat"] = flat
            report["all_detections"] = all_detections
        if annotated_gallery is not None:
            report["annotated_gallery"] = annotated_gallery
        if per_task is not None:
            report["per_task"] = per_task
        if evidence is not None:
            report["evidence"] = evidence
        if grounding is not None:
            report["grounding"] = grounding
        if crop_provenance is not None:
            report["crop_provenance"] = crop_provenance
        if retry_reason:
            report["retry_reason"] = retry_reason
        if provenance is not None:
            report["provenance"] = provenance
        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            path = atomic_write_text(run_dir / "report.json", json.dumps(report, indent=2))
        else:
            path = save_report(report, self.reports_dir)
        report["report_path"] = str(path)
        append_history(report, self.history_path)
        return report


# ---- orchestrator (Select -> Detect -> Reason -> [verify] -> Report) ----
VERIFY_BUDGET = 1  # hard bound on extra YOLO passes per request


def _image_meta(image_path: Path) -> dict:
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return {"width": im.width, "height": im.height}
    except Exception:
        return {}


class Orchestrator:
    def __init__(self, yolo_service, gemma_client, reports_dir="outputs/reports",
                 history_path="outputs/history.jsonl", annotated_dir="outputs/annotated",
                 runs_dir: str | Path = "outputs/runs",
                 max_refine_passes: int = 1, retry_conf: float = 0.15,
                 crop_top_k: int = 5, crop_min_conf: float = 0.35):
        self.detector = DetectorAgent(yolo_service)
        self.reasoner = ReasonerAgent(gemma_client)
        self.reporter = ReporterAgent(reports_dir, history_path)
        self.annotated_dir = Path(annotated_dir)
        self.runs_dir = Path(runs_dir)
        self.max_refine_passes = max_refine_passes
        self.retry_conf = retry_conf
        self.crop_top_k = crop_top_k
        self.crop_min_conf = crop_min_conf

    def _close_ups(self, image_path: Path, detections: list[dict],
                   run_dir: Path, run_id: str, steps: list[dict]) -> tuple[list[dict], list[dict]]:
        """Crop top-k confident boxes for Gemma close-ups. Returns (crops, run_file entries)."""
        crops = crop_detections(image_path, detections, run_dir / "crops",
                                top_k=self.crop_top_k, min_conf=self.crop_min_conf)
        if crops:
            steps.append({"agent": "orchestrator", "action": "crop",
                          "detail": f"{len(crops)} close-ups for Gemma", "ms": 0})
        files = [{"name": f"crops/{Path(c['path']).name}",
                  "url": self._run_file_url(run_id, f"crops/{Path(c['path']).name}")}
                 for c in crops]
        return crops, files

    @staticmethod
    def _run_file_url(run_id: str, filename: str) -> str:
        return f"/outputs/runs/{run_id}/{filename}"

    def _new_run(self, image_path: Path) -> tuple[str, Path, str]:
        """Create outputs/runs/<run_id>/ and stage the input image. Returns (id, dir, input name)."""
        rid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        run_dir = self.runs_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        input_name = f"input{image_path.suffix.lower() or '.jpg'}"
        shutil.copy(image_path, run_dir / input_name)
        return rid, run_dir, input_name

    def run(self, image_path: str | Path, query: str = "", model_choice: str = "auto",
            conf: float | None = None, iou: float | None = None) -> dict:
        image_path = Path(image_path)
        steps: list[dict] = []
        fallback_actions: list[str] = []
        hw = hw_info(self.detector.yolo.device)
        task, rationale = select_model(query, model_choice, _image_meta(image_path), hw)
        steps.append({"agent": "orchestrator", "action": "select",
                      "detail": f"task={task} (requested={model_choice}): {rationale}", "ms": 0})

        if task == "all":
            return self._run_all(image_path, query, model_choice, conf, iou, steps,
                                 rationale, fallback_actions, hw)

        run_id, run_dir, input_name = self._new_run(image_path)
        ann_name = f"annotated_{task}.jpg"
        try:
            detection, s1 = self.detector.run(image_path, task, conf=conf, iou=iou,
                                              annotated_dir=run_dir, ann_name=ann_name)
        except RuntimeError as e:
            fallback_actions.append(f"inference failed: {e}")
            raise
        steps.append(s1)

        crops, crop_files = self._close_ups(image_path, detection.get("detections", []),
                                            run_dir, run_id, steps)
        evidence = build_evidence(detection.get("detections", []), task)
        reason, s2 = self.reasoner.run(image_path, query, detection, crops=crops,
                                       evidence=evidence)
        steps.append(s2)

        def _avg_conf(dets: list[dict]) -> float | None:
            confs = [d.get("conf", 0.0) for d in (dets or [])
                     if isinstance(d.get("conf"), (int, float))]
            return sum(confs) / len(confs) if confs else None

        refined = False
        retry_reason = ""
        # Bounded verification: at most VERIFY_BUDGET extra passes. Triggered by
        # MEASURABLE detector signals; Gemma uncertainty wording corroborates but
        # very-low confidence alone (< retry_conf) is sufficient — an empty result
        # alone is still a valid outcome, never a trigger by itself.
        used = 0
        ans = (reason.get("answer", "") or "").lower()
        need, note = self.reasoner.needs_refine(detection, query)
        avg_conf = _avg_conf(detection.get("detections", []))
        gemma_uncertain = ("no object" in ans or "uncertain" in ans or "nothing" in ans)
        measurement_driven = need and avg_conf is not None and avg_conf < self.retry_conf
        if need and used < min(self.max_refine_passes, VERIFY_BUDGET) and (
                gemma_uncertain or measurement_driven):
            retry_reason = (f"measurement-driven: avg_conf={avg_conf:.2f} < retry_conf={self.retry_conf}"
                            if measurement_driven
                            else f"gemma-corroborated: {note}")
            steps.append({"agent": "orchestrator", "action": "verify-retry",
                          "detail": f"retry conf={self.retry_conf}: {retry_reason}", "ms": 0})
            fallback_actions.append(f"verify-retry yolo26x-{task} conf={self.retry_conf}: {retry_reason}")
            detection, s3 = self.detector.run(image_path, task, conf=self.retry_conf,
                                              iou=iou, annotated_dir=run_dir,
                                              ann_name=f"annotated_{task}_retry.jpg")
            steps.append(s3)
            crops, crop_files = self._close_ups(image_path, detection.get("detections", []),
                                                run_dir, run_id, steps)
            evidence = build_evidence(detection.get("detections", []), task)
            reason, s4 = self.reasoner.run(image_path, query, detection,
                                           refine_note=f"Verify pass conf={self.retry_conf}. {note}",
                                           crops=crops, evidence=evidence)
            steps.append(s4)
            refined = True
            used += 1

        grounding = validate_answer(reason.get("answer", ""), [e["id"] for e in evidence])
        crop_provenance = [{k: c[k] for k in ("n", "det_no", "label", "conf", "box", "pad", "image_size")
                            if k in c} for c in crops]
        provenance = build_provenance(image_path, [task],
                                      self.detector.yolo.weights,
                                      detection.get("device", ""))

        annotated_name = Path(detection["annotated_path"]).name
        annotated_url = self._run_file_url(run_id, annotated_name)

        run_files = [{"name": input_name, "url": self._run_file_url(run_id, input_name)},
                     {"name": annotated_name, "url": annotated_url}]
        run_files.extend(crop_files)
        if refined:
            retry_name = f"annotated_{task}_retry.jpg"
            if (run_dir / retry_name).exists():
                run_files.append({"name": retry_name,
                                  "url": self._run_file_url(run_id, retry_name)})
        run_files.append({"name": "report.json",
                          "url": self._run_file_url(run_id, "report.json")})

        report = self.reporter.run(image_name=image_path.name, query=query, task=task,
                                   requested_model=model_choice, detection=detection,
                                   reason=reason, steps=steps,
                                   annotated_url=annotated_url, refined=refined,
                                   selection_rationale=rationale,
                                   validation_status=MODEL_REGISTRY[task]["validation"],
                                   fallback_actions=fallback_actions, hardware=hw,
                                   run_id=run_id, run_dir=run_dir, run_files=run_files,
                                   evidence=evidence, grounding=grounding,
                                   crop_provenance=crop_provenance,
                                   retry_reason=retry_reason, provenance=provenance)
        return report

    def _run_all(self, image_path: Path, query: str, model_choice: str,
                 conf: float | None, iou: float | None, steps: list[dict],
                 rationale: str = "", fallback_actions: list | None = None,
                 hw: dict | None = None) -> dict:
        """Run all 7 X heads sequentially, single fused Gemma pass, no refine."""
        run_id, run_dir, input_name = self._new_run(image_path)
        fused, det_steps = self.detector.run_all(image_path, conf=conf, iou=iou,
                                                  annotated_dir=run_dir)
        steps.extend(det_steps)

        crops, crop_files = self._close_ups(
            image_path, fused.get("detect", {}).get("detections", []), run_dir, run_id, steps)
        evidence = build_evidence(fused.get("detect", {}).get("detections", []), "detect")
        reason, rs = self.reasoner.run_fused(image_path, query, fused, crops=crops,
                                             evidence=evidence)
        steps.append(rs)
        grounding = validate_answer(reason.get("answer", ""), [e["id"] for e in evidence])
        crop_provenance = [{k: c[k] for k in ("n", "det_no", "label", "conf", "box", "pad", "image_size")
                            if k in c} for c in crops]
        provenance = build_provenance(image_path, list(fused.keys()),
                                      self.detector.yolo.weights,
                                      fused.get("detect", {}).get("device", ""))

        per_task: dict[str, dict] = {}
        gallery: list[str] = []
        total = 0
        for t, res in fused.items():
            url = self._run_file_url(run_id, Path(res["annotated_path"]).name)
            gallery.append({"task": t, "url": url})
            per_task[t] = {"model": res.get("model"), "num_objects": res.get("num_objects", 0),
                           "inference_ms": res.get("inference_ms"), "annotated_url": url}
            total += res.get("num_objects", 0)

        primary_url = gallery[0]["url"] if gallery else ""
        pseudo = {"task": "all", "model": "all-7x", "device": fused["detect"].get("device"),
                  "conf": conf, "num_objects": total, "detections": {},
                  "inference_ms": sum(r.get("inference_ms", 0) for r in fused.values())}
        run_files = ([{"name": input_name, "url": self._run_file_url(run_id, input_name)}]
                     + [{"name": Path(g["url"]).name, "url": g["url"]} for g in gallery]
                     + crop_files
                     + [{"name": "report.json", "url": self._run_file_url(run_id, "report.json")}])
        report = self.reporter.run(
            image_name=image_path.name, query=query, task="all",
            requested_model=model_choice, detection=pseudo, reason=reason, steps=steps,
            annotated_url=primary_url, refined=False,
            all_detections={t: r.get("detections", []) for t, r in fused.items()},
            annotated_gallery=gallery, per_task=per_task,
            selection_rationale=rationale or "explicit all-mode: all 7 X heads fused",
            validation_status={t: MODEL_REGISTRY[t]["validation"] for t in fused},
            fallback_actions=fallback_actions or [], hardware=hw or {},
            run_id=run_id, run_dir=run_dir, run_files=run_files,
            evidence=evidence, grounding=grounding, crop_provenance=crop_provenance,
            provenance=provenance)
        return report
