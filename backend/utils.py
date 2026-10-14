"""Shared utils: Gemma prompt builders + disk persistence."""
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ---- prompts (image goes BEFORE text per Gemma-4 model card) ----
SYSTEM_PROMPT = (
    "You are a vision reasoning agent in a Detect-Reason-Report pipeline. "
    "You receive: (1) the original image, (2) YOLO26X detections as JSON, "
    "optionally (3) close-up crops of the most confident objects, named crop_1..N "
    "with a mapping to detection numbers. "
    "YOLO is authoritative for object locations. Do not hallucinate objects YOLO did not find. "
    "Describe the image in rich forensic detail, organized as: "
    "SCENE (setting, background vs foreground, lighting, colors, mood); "
    "OBJECTS (each detection: appearance, color, size, position relative to the image and to other objects, "
    "using the crops for fine detail); RELATIONS (spatial arrangement, interactions, occlusions); "
    "TEXT (any visible text, signs, labels, transcribed exactly); "
    "CLOSE (counts reconciled with YOLO, uncertainties, suggested next action). "
    "If detections are empty or low-confidence, say so and suggest a concrete refine action "
    "(lower confidence, crop-zoom, or try seg/pose/obb). Be thorough but factual."
)

def build_yolo_summary(detections: list[dict], model: str, task: str) -> str:
    if not detections:
        return f"YOLO task={task} model={model}: no objects above threshold."
    lines = [f"YOLO task={task} model={model}: {len(detections)} object(s)."]
    for i, d in enumerate(detections[:50], 1):
        if not isinstance(d, dict):
            continue
        conf = d.get("conf")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        label = d.get("label", "?")
        if task == "cls" or "x1" not in d:
            lines.append(f"{i}. {label} conf={conf_s}")
        else:
            try:
                box = f"box=[{float(d['x1']):.0f},{float(d['y1']):.0f},{float(d['x2']):.0f},{float(d['y2']):.0f}]"
            except (KeyError, TypeError, ValueError):
                box = "box=[?]"
            lines.append(f"{i}. {label} conf={conf_s} {box}")
    if len(detections) > 50:
        lines.append(f"... +{len(detections)-50} more")
    return "\n".join(lines)


SYSTEM_PROMPT_ALL = (
    "You are a vision reasoning agent in a Detect-Reason-Report pipeline (ALL-MODE). "
    "You receive: (1) the original image, (2) fused YOLO26X outputs from all 7 X heads "
    "(detect/seg/pose/obb/cls/depth/sem). "
    "Each head is authoritative only for its own task. Cross-check them: e.g. detect boxes vs seg masks, "
    "pose keypoints vs detect persons. Do not hallucinate objects no head found. "
    "Explain per-head counts, agreements/disagreements, spatial relations, and what to do next. "
    "Describe the scene richly: setting, colors, lighting, per-object appearance and positions. "
    "GROUNDING RULES (mandatory): every claim about a localized object must cite its evidence id "
    "as [#dN]. Per-head confidences are NOT comparable across heads — never rank heads by raw "
    "confidence. Mark disagreements between heads explicitly instead of silently picking one. "
    "If evidence is insufficient, say UNCERTAIN or abstain rather than guessing. "
    "Keep the final answer factual and detailed."
)


def build_fused_summary(fused: dict[str, dict]) -> str:
    parts = []
    for task in ("detect", "seg", "pose", "obb", "cls", "depth", "sem"):
        det = fused.get(task)
        if det is None:
            continue
        parts.append(build_yolo_summary(det.get("detections", []), det.get("model", task), task))
    return "\n\n".join(parts)


def build_evidence(detections: list[dict], task: str) -> list[dict]:
    """Normalize detections into citable evidence records.

    Each record gets a stable id (`d1..dN`, 1-based in detection order) that
    Gemma must cite as `[#dN]` for every object claim. Returns
    [{id, n, task, label, conf, box}] where box is [x1,y1,x2,y2] or None for
    box-less outputs (e.g. cls). Pure function — safe to unit-test.
    """
    out = []
    for i, d in enumerate(detections or [], 1):
        if not isinstance(d, dict):
            continue
        box = None
        try:
            if all(k in d for k in ("x1", "y1", "x2", "y2")):
                box = [round(float(d["x1"])), round(float(d["y1"])),
                       round(float(d["x2"])), round(float(d["y2"]))]
        except (TypeError, ValueError):
            box = None
        conf = d.get("conf")
        out.append({"id": f"d{i}", "n": i, "task": task,
                    "label": str(d.get("label", "?")),
                    "conf": float(conf) if isinstance(conf, (int, float)) else None,
                    "box": box})
    return out


def build_evidence_text(evidence: list[dict]) -> str:
    """Render evidence records as the citation source Gemma must reference."""
    if not evidence:
        return "EVIDENCE: no detections. There is nothing to cite; say so explicitly."
    lines = [f"EVIDENCE ({len(evidence)} detections — cite these ids):"]
    for e in evidence:
        box = f" box={e['box']}" if e["box"] else " (no box)"
        conf = f"{e['conf']:.2f}" if isinstance(e["conf"], float) else "?"
        lines.append(f"[#{e['id']}] {e['label']} conf={conf}{box}")
    return "\n".join(lines)


def build_user_text(query: str, yolo_summary: str, refine_note: str = "",
                    crops: list[dict] | None = None,
                    evidence: list[dict] | None = None) -> str:
    q = query.strip() or "Describe this image in full detail using the YOLO detections."
    text = f"USER QUERY:\n{q}\n\nYOLO26X OUTPUT:\n{yolo_summary}\n"
    if evidence is not None:
        text += f"\n{build_evidence_text(evidence)}\n"
    if crops:
        mapping = "\n".join(
            f"crop_{c['n']} = detection #{c['det_no']} ({c['label']}, conf={c['conf']:.2f})"
            for c in crops)
        text += f"\nCLOSE-UP CROPS (in order after the full image):\n{mapping}\n"
    if refine_note:
        text += f"\nNOTE:\n{refine_note}\n"
    text += (
        "\nRespond in two parts:\n"
        "THOUGHT: step-by-step reasoning (what YOLO found per object, what each crop reveals, "
        "layout, uncertainties).\n"
        "ANSWER: detailed description following SCENE / OBJECTS / RELATIONS / TEXT / CLOSE. "
        "Every object gets at least one sentence on appearance and position.\n"
        "GROUNDING RULES (mandatory): every claim about an object must cite its evidence id "
        "as [#dN] (e.g. [#d2]). Never cite an id that is not listed above. "
        "Do not infer relationships (e.g. person DRIVING a vehicle) without visible evidence — "
        "describe positions, mark the relationship UNCERTAIN. "
        "If evidence is insufficient, say UNCERTAIN or abstain explicitly rather than guessing.\n"
        "If nothing detected, say so explicitly."
    )
    return text


ABSTAIN_PHRASES = ("uncertain", "cannot tell", "can't tell", "unclear", "abstain",
                   "insufficient evidence", "not visible", "cannot confirm")

import re as _re
_CITE_RE = _re.compile(r"\[#(d\d+)\]")


def validate_answer(answer: str, evidence_ids: list[str]) -> dict:
    """Check Gemma's answer against the evidence ids it was given.

    Returns {cited, unknown, coverage, abstained}. Heuristic, never raises:
    it records whether claims are traceable, it does not prove truth.
    """
    try:
        cited = sorted(set(_CITE_RE.findall(answer or "")), key=lambda s: int(s[1:]))
    except Exception:
        cited = []
    valid = set(evidence_ids or [])
    unknown = [c for c in cited if c not in valid]
    coverage = round(len([c for c in cited if c in valid]) / len(valid), 3) if valid else 1.0
    low = (answer or "").lower()
    abstained = any(p in low for p in ABSTAIN_PHRASES)
    return {"cited": cited, "unknown": unknown, "coverage": coverage,
            "abstained": abstained,
            "evidence_count": len(valid)}


def crop_detections(image_path: str | Path, detections: list[dict], out_dir: str | Path,
                    top_k: int = 5, min_conf: float = 0.35, pad: float = 0.12) -> list[dict]:
    """Crop top-k confident boxes (with padding) for Gemma close-ups.

    Returns [{n, path, det_no, label, conf, box, pad}]. box is the padded
    [x1,y1,x2,y2] pixel coords in the ORIGINAL image (crop provenance).
    Skips box-less entries (e.g. cls). Never raises: returns [] when nothing qualifies.
    """
    try:
        from PIL import Image
        boxes = [d for d in (detections or [])
                 if isinstance(d, dict) and all(k in d for k in ("x1", "y1", "x2", "y2"))
                 and isinstance(d.get("conf"), (int, float)) and d["conf"] >= min_conf]
        boxes.sort(key=lambda d: d["conf"], reverse=True)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        crops = []
        with Image.open(image_path).convert("RGB") as im:
            w, h = im.size
            for n, d in enumerate(boxes[:max(0, top_k)], 1):
                pw, ph = (d["x2"] - d["x1"]) * pad, (d["y2"] - d["y1"]) * pad
                box = (max(0, int(d["x1"] - pw)), max(0, int(d["y1"] - ph)),
                       min(w, int(d["x2"] + pw)), min(h, int(d["y2"] + ph)))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                name = f"crop_{n}.jpg"
                im.crop(box).save(out_dir / name, "JPEG", quality=88)
                crops.append({"n": n, "path": str(out_dir / name),
                              "det_no": detections.index(d) + 1,
                              "label": str(d.get("label", "?")), "conf": float(d["conf"]),
                              "box": list(box), "pad": pad,
                              "image_size": [w, h]})
        return crops
    except Exception:
        return []


# ---- store (disk persistence: reports + history.jsonl) ----
def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text atomically (tmp file + os.replace) so concurrent readers
    never see a half-written report. Creates parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _locked_append_line(path: Path, line: str) -> None:
    """Append one line with an OS file lock (msvcrt on Windows, fcntl on
    POSIX) so concurrent requests cannot interleave history.jsonl rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, len(line.encode("utf-8")))
                try:
                    f.write(line)
                finally:
                    f.flush()
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, len(line.encode("utf-8")))
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError, ValueError):
            # Locking unavailable (odd FS) — plain append, never fail the request.
            f.write(line)


def save_report(report: dict, reports_dir: str | Path) -> Path:
    ensure_dirs(reports_dir)
    rid = report.get("id", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    return atomic_write_text(Path(reports_dir) / f"{rid}.json", json.dumps(report, indent=2))


def append_history(report: dict, history_path: str | Path) -> None:
    p = Path(history_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: report.get(k) for k in (
        "id", "run_id", "created_at", "query", "task", "model", "num_objects",
        "refined", "annotated_url", "report_path", "final_answer")}
    _locked_append_line(p, json.dumps(slim) + "\n")


def read_history(history_path: str | Path, limit: int = 20) -> list[dict]:
    p = Path(history_path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-limit:][::-1]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
