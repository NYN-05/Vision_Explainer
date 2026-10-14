"""YOLO26X service. Strict X-only, no fallback. Detect-first routing."""
from __future__ import annotations
import time
import uuid
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VALID_TASKS = ("detect", "seg", "pose", "obb", "cls", "depth", "sem")

# Verified registry. `domain`/`capabilities` are claims about the checkpoint family
# (COCO-pretrained YOLO26X); per-head `validation` is filled by validate_registry(),
# which loads the actual file and checks task + class map. Never trust filenames alone.
# NOTE: verified 2026-10-14 — obb head carries 15 aerial-oriented classes
# (plane, ship, ...); it is the aerial pick. No separate satellite checkpoint ships.
MODEL_REGISTRY: dict[str, dict] = {
    "detect": {"domain": "general (COCO)", "output": "boxes+labels",
               "capabilities": ["boxes", "counts", "small-object-via-low-conf"],
               "vram_reserve_MB": 2000, "validation": "unvalidated"},
    "seg": {"domain": "general (COCO)", "output": "masks+boxes+labels",
            "capabilities": ["pixel-masks"], "vram_reserve_MB": 2200, "validation": "unvalidated"},
    "pose": {"domain": "general (COCO-person)", "output": "boxes+keypoints",
             "capabilities": ["keypoints-17"], "vram_reserve_MB": 2200, "validation": "unvalidated"},
    "obb": {"domain": "aerial-oriented (15 classes: plane, ship, ...)",
            "output": "rotated-boxes+labels",
            "capabilities": ["oriented-boxes", "aerial"],
            "vram_reserve_MB": 2000, "validation": "unvalidated"},
    "cls": {"domain": "general (COCO)", "output": "top5-labels",
            "capabilities": ["single-image-class"], "vram_reserve_MB": 1500, "validation": "unvalidated"},
    "depth": {"domain": "general", "output": "depth-map",
              "capabilities": ["depth"], "vram_reserve_MB": 2200, "validation": "unvalidated"},
    "sem": {"domain": "street-scene (19 classes: road, sidewalk, ...)",
            "output": "semantic-map",
            "capabilities": ["scene-parsing"], "vram_reserve_MB": 2200, "validation": "unvalidated"},
}

# Domain cues checked before generic task keywords (deterministic, no Gemma call).
DOMAIN_CUES: dict[str, tuple[str, ...]] = {
    "aerial": ("aerial", "satellite", "drone", "overhead", "top-down", "remote sensing"),
    "dense": ("dense", "crowd", "small object", "tiny", "far away"),
}

# keyword -> task (checked in order)
ROUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pose": ("pose", "keypoint", "skeleton", "joint", "yoga", "posture", "person"),
    "seg": ("segment", "mask", "outline", "silhouette", "instance seg"),
    "obb": ("oriented", "rotat", "aerial", "satellite", "obb", "diagonal box"),
    "cls": ("classif", "what is this", "which category", "single object"),
    "depth": ("depth", "distance map", "how far", "3d depth"),
    "sem": ("semantic", "scene pars", "stuff segment", "road scene"),
}


def route_task(query: str, requested: str = "auto") -> str:
    """Detect-first routing. Explicit choice wins; 'all' runs every X head; else keyword match; else detect."""
    r = (requested or "auto").lower().strip()
    if r == "all":
        return "all"
    if r in VALID_TASKS:
        return r
    q = (query or "").lower()
    for task, kws in ROUTE_KEYWORDS.items():
        if any(k in q for k in kws):
            # 'person' alone is too generic -> keep detect unless pose-specific word present
            if task == "pose" and not any(k in q for k in ("pose", "keypoint", "skeleton", "joint", "yoga", "posture")):
                continue
            return task
    return "detect"


def hw_info(device: str) -> dict:
    """CUDA availability + free/total VRAM. Never raises."""
    info: dict = {"device": device, "cuda": torch.cuda.is_available()}
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            info.update({"gpu": torch.cuda.get_device_name(0),
                         "free_MB": round(free / 1e6), "total_MB": round(total / 1e6)})
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def select_model(query: str, requested: str = "auto", image_meta: dict | None = None,
                 hw: dict | None = None) -> tuple[str, str]:
    """Deterministic task-aware selection. Returns (task, rationale). No Gemma call.

    Priority: explicit choice > task keywords (+ domain cues) > default detect.
    `image_meta` (e.g. {"width": w, "height": h}) only annotates the rationale
    (dense/small-object strategy note); it never overrides an explicit choice.
    """
    r = (requested or "auto").lower().strip()
    if r == "all":
        return "all", "explicit request: run all 7 X heads and fuse"
    if r in VALID_TASKS:
        return r, f"explicit request: {r} ({MODEL_REGISTRY[r]['domain']})"
    q = (query or "").lower()
    aerial = any(k in q for k in DOMAIN_CUES["aerial"])
    dense = any(k in q for k in DOMAIN_CUES["dense"]) or (
        image_meta is not None and max(image_meta.get("width", 0),
                                        image_meta.get("height", 0)) >= 2000)
    task = route_task(query, requested)
    why = f"query keywords -> {task}" if task != "detect" else "no task keywords -> default detect"
    extra = []
    if aerial:
        extra.append("aerial-domain cues present; obb selected "
                     "(verified 15-class aerial-oriented head: plane, ship, ...)" if task == "obb"
                     else "aerial-domain cues present but no aerial-specific head requested; "
                     "obb (verified aerial-oriented) available on demand")
    if dense and task == "detect":
        extra.append("dense/small-object cues: detect with low-conf strategy suggested")
    if extra:
        why += "; " + "; ".join(extra)
    why += f" [{MODEL_REGISTRY[task]['domain']}]"
    _ = hw  # reserved: hardware gates execution via check_fit, not selection
    return task, why


class YoloService:
    def __init__(self, weights: dict[str, str], device: str = "cuda",
                 conf_default: float = 0.25, iou_default: float = 0.7,
                 vram_safety_MB: int = 800):
        self.weights = dict(weights)
        self.conf_default = conf_default
        self.iou_default = iou_default
        self.vram_safety_MB = vram_safety_MB
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self._loaded_task: str | None = None
        self._model: YOLO | None = None

    def weight_for(self, task: str) -> Path:
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown YOLO task '{task}'. Valid: {VALID_TASKS}")
        p = Path(self.weights[task])
        if not p.exists():
            # STRICT: no fallback to s/m/l, no auto-download of another size.
            raise FileNotFoundError(
                f"Missing X weight for task '{task}': {p.resolve()}. "
                f"Expected yolo26x variant. No fallback allowed per config."
            )
        return p

    def validate_registry(self) -> dict[str, dict]:
        """Load each present checkpoint and verify actual task + class map.

        Updates MODEL_REGISTRY[task]["validation"] in place. Statuses:
        verified | task-mismatch:<actual> | missing | load-error:<err>.
        Weight loading only (no inference) — cheap, CPU-side.
        """
        out: dict[str, dict] = {}
        for task in VALID_TASKS:
            p = Path(self.weights[task])
            if not p.exists():
                MODEL_REGISTRY[task]["validation"] = "missing"
                out[task] = {"validation": "missing", "weight": str(p)}
                continue
            try:
                m = YOLO(str(p))
                actual = getattr(m, "task", "?")
                names = getattr(m, "names", {}) or {}
                # Ultralytics task names differ from weight suffixes; normalize.
                aliases = {"seg": "segment", "cls": "classify", "sem": "semantic"}
                ok = actual == task or aliases.get(task) == actual
                entry = {"validation": "verified" if ok else f"task-mismatch:{actual}",
                         "weight": str(p), "actual_task": actual,
                         "num_classes": len(names),
                         "classes_sample": list(names.values())[:5]}
                MODEL_REGISTRY[task].update(validation=entry["validation"],
                                            actual_task=actual, num_classes=len(names))
                out[task] = entry
            except Exception as e:
                MODEL_REGISTRY[task]["validation"] = f"load-error:{type(e).__name__}"
                out[task] = {"validation": MODEL_REGISTRY[task]["validation"], "weight": str(p)}
        return out

    def check_fit(self, task: str, safety_MB: int = 800) -> tuple[bool, str]:
        """VRAM feasibility pre-check. Estimates only; never silently swaps model size."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return True, "cpu or cuda-unavailable: no VRAM gate"
        try:
            free, total = torch.cuda.mem_get_info(0)
            need = (MODEL_REGISTRY[task]["vram_reserve_MB"] + safety_MB) * 1e6
            ok = free >= need
            return ok, (f"free={free/1e6:.0f}MB need~{need/1e6:.0f}MB (reserve+safety) "
                        f"for yolo26x-{task}")
        except Exception as e:
            return True, f"VRAM query failed, proceeding: {type(e).__name__}: {e}"

    def _unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_task = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_model(self, task: str) -> YOLO:
        if self._loaded_task == task and self._model is not None:
            return self._model
        # unload previous to save VRAM (X models are large, RTX 4050 ~6GB)
        self._unload()
        w = self.weight_for(task)
        self._model = YOLO(str(w))
        self._loaded_task = task
        return self._model

    def list_models(self) -> list[dict]:
        out = []
        for task in VALID_TASKS:
            p = Path(self.weights[task])
            reg = MODEL_REGISTRY[task]
            out.append({"task": task, "weight": str(p), "exists": p.exists(), "size": "x",
                        "domain": reg["domain"], "output": reg["output"],
                        "validation": reg["validation"]})
        return out

    def predict(self, image_path: str | Path, task: str, annotated_dir: str | Path,
                conf: float | None = None, iou: float | None = None,
                vram_safety_MB: int | None = None,
                ann_name: str | None = None) -> dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        conf = self.conf_default if conf is None else conf
        iou = self.iou_default if iou is None else iou
        if vram_safety_MB is None:
            vram_safety_MB = self.vram_safety_MB
        # Unload a different resident head BEFORE measuring: the gate must see
        # steady-state free VRAM (one head at a time), not the overlap moment.
        if task != self._loaded_task:
            self._unload()
        ok, fit_note = self.check_fit(task, vram_safety_MB)
        if not ok:
            raise RuntimeError(
                f"Insufficient VRAM for yolo26x-{task}: {fit_note}. "
                f"X-only policy: refusing to silently fall back to s/m. "
                f"Free VRAM (close other GPU apps / stop llama-server layers) and retry.")
        try:
            model = self._get_model(task)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load yolo26x-{task}: {type(e).__name__}: {e}")

        t0 = time.perf_counter()
        try:
            results = model.predict(source=str(image_path), conf=conf, iou=iou,
                                    device=self.device, verbose=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                try:
                    del self._model
                except Exception:
                    pass
                self._model = None
                self._loaded_task = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise RuntimeError(f"YOLO26X-{task} inference failed: {e}")
        ms = (time.perf_counter() - t0) * 1000.0
        if not results:
            # e.g. backend could not decode the image ("Image Read Error"):
            # fail loudly with meaning instead of IndexError downstream.
            raise RuntimeError(
                f"YOLO26X-{task} returned no result for {image_path.name}; "
                f"the image bytes may be corrupt or in an unreadable format.")
        r = results[0]

        detections = self._parse(r, task)
        annotated_path = self._save_plot(r, annotated_dir, ann_name)
        # Honesty marker: depth/sem output parsing runs but has never been
        # verified against model specs or fixed test images (see CHANGES.md).
        # Downstream consumers must not treat these as validated results.
        output_note = ("unvalidated head: parsing not empirically verified"
                       if task in ("depth", "sem") else "")
        return {
            "task": task,
            "model": str(self.weight_for(task)),
            "device": self.device,
            "conf": conf,
            "iou": iou,
            "inference_ms": round(ms, 1),
            "num_objects": len(detections),
            "detections": detections,
            "annotated_path": str(annotated_path),
            "names": getattr(r, "names", {}),
            "vram_fit_note": fit_note,
            "validation": MODEL_REGISTRY[task]["validation"],
            "output_note": output_note,
        }

    def predict_all(self, image_path: str | Path, annotated_dir: str | Path,
                      conf: float | None = None, iou: float | None = None) -> dict[str, dict]:
        """Run every X head sequentially (VRAM-safe: one model at a time). No fallback."""
        fused: dict[str, dict] = {}
        for task in VALID_TASKS:
            fused[task] = self.predict(image_path, task, annotated_dir, conf=conf, iou=iou,
                                       ann_name=f"annotated_{task}.jpg")
        return fused

    @staticmethod
    def _parse(r, task: str) -> list[dict]:
        names = getattr(r, "names", {}) or {}
        # classification: probs only
        probs = getattr(r, "probs", None)
        if task == "cls" or (probs is not None and getattr(r, "boxes", None) is None):
            out = []
            try:
                top = probs.top5 if hasattr(probs, "top5") else []
                confs = probs.top5conf if hasattr(probs, "top5conf") else []
                for cls_id, c in zip(list(top), [float(x) for x in confs]):
                    out.append({"cls": int(cls_id), "label": str(names.get(int(cls_id), cls_id)),
                                "conf": float(c)})
            except Exception:
                pass
            return out
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        try:
            xyxy = boxes.xyxy.cpu().tolist()
            cls_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else [0] * len(xyxy)
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
        except Exception:
            return []
        for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs):
            out.append({"cls": int(cid), "label": str(names.get(int(cid), cid)),
                        "conf": float(cf),
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
        # keypoints for pose (optional, first 5 pts count)
        kpts = getattr(r, "keypoints", None)
        if kpts is not None:
            try:
                ky = kpts.xy.cpu().tolist()
                for d, pts in zip(out, ky):
                    d["keypoints"] = [[float(a), float(b)] for a, b in pts[:17]]
            except Exception:
                pass
        return out

    @staticmethod
    def _save_plot(r, annotated_dir: str | Path, name: str | None = None) -> Path:
        Path(annotated_dir).mkdir(parents=True, exist_ok=True)
        try:
            plotted = r.plot()  # BGR numpy
        except Exception as e:
            raise RuntimeError(f"Failed to render YOLO annotation: {e}")
        name = name or f"ann_{uuid.uuid4().hex[:12]}.jpg"
        out = Path(annotated_dir) / name
        cv2.imwrite(str(out), plotted)
        return out


# ---- gemma_client: HTTP client for llama.cpp server (OpenAI-compatible) ----
import base64
import io
import re
import httpx
from PIL import Image

THOUGHT_RE = re.compile(r"<\|channel\|>thought(.*?)<channel\|>", re.DOTALL)


def split_thought(text: str) -> tuple[str, str]:
    """Return (thought, answer). Handles Gemma-4 thinking tags; falls back to THOUGHT:/ANSWER: split."""
    m = THOUGHT_RE.search(text or "")
    if m:
        thought = m.group(1).strip()
        answer = (text[:m.start()] + text[m.end():]).strip()
        # also support explicit ANSWER: marker after tags
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:", 1)[1].strip()
        return thought, answer or text.strip()
    t = text or ""
    if "THOUGHT:" in t and "ANSWER:" in t:
        try:
            thought = t.split("THOUGHT:", 1)[1].split("ANSWER:", 1)[0].strip()
            answer = t.split("ANSWER:", 1)[1].strip()
            return thought, answer
        except Exception:
            pass
    return "", t.strip()


def image_to_data_url(image_path: str | Path, max_edge: int = 1024) -> str:
    p = Path(image_path)
    img = Image.open(p).convert("RGB")
    img.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GemmaClient:
    def __init__(self, base_url: str, chat_endpoint: str = "/v1/chat/completions",
                 models_endpoint: str = "/v1/models", timeout_s: int = 120,
                 temperature: float = 1.0, top_p: float = 0.95, top_k: int = 64,
                 max_tokens: int = 1024, image_max_edge: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.chat_url = self.base_url + chat_endpoint
        self.models_url = self.base_url + models_endpoint
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.image_max_edge = image_max_edge

    def health(self) -> dict:
        try:
            r = httpx.get(self.models_url, timeout=10.0)
            r.raise_for_status()
            data = r.json()
            models = data.get("data", data.get("models", []))
            arch = {}
            try:
                arch = models[0].get("architecture", {}) if models else {}
            except Exception:
                pass
            return {"ok": True, "base_url": self.base_url, "models": models, "architecture": arch}
        except Exception as e:
            return {"ok": False, "base_url": self.base_url, "error": f"{type(e).__name__}: {e}"}

    def chat_vision(self, image_path: str | Path, user_text: str, system_text: str,
                    extra_images: list[str | Path] | None = None) -> dict:
        parts: list[dict] = [
            {"type": "image_url", "image_url": {
                "url": image_to_data_url(image_path, self.image_max_edge)}},
        ]
        for extra in extra_images or []:
            parts.append({"type": "image_url", "image_url": {
                "url": image_to_data_url(extra, self.image_max_edge)}})
        parts.append({"type": "text", "text": user_text})
        payload = {
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": parts},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            r = httpx.post(self.chat_url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"llama.cpp chat failed at {self.chat_url}: {type(e).__name__}: {e}")
        try:
            raw = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected chat response shape: {e}; keys={list(data.keys())}")
        thought, answer = split_thought(raw)
        if not answer:
            # truncated thinking / no final part yet — keep raw so UI never shows empty
            answer = raw.strip()
        usage = data.get("usage", {})
        return {"raw": raw, "thought": thought, "answer": answer, "usage": usage}
