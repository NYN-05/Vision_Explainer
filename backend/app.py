"""FastAPI backend: simple HTML UI + Detect-Reason-Report API."""
from __future__ import annotations
import logging
import shutil
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agents import Orchestrator
from backend.services import GemmaClient, YoloService
from backend.utils import ensure_dirs, read_history

log = logging.getLogger("vision_explainer")

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

UPLOADS = ROOT / CFG["paths"]["uploads"]
ANNOTATED = ROOT / CFG["paths"]["annotated"]
REPORTS = ROOT / CFG["paths"]["reports"]
RUNS = ROOT / CFG["paths"]["runs"]
HISTORY = ROOT / CFG["paths"]["history"]
ensure_dirs(UPLOADS, ANNOTATED, REPORTS, RUNS)

SERVER_CFG = CFG.get("server", {})
MAX_UPLOAD_MB = float(SERVER_CFG.get("max_upload_MB", 25))
MAX_IMAGE_MP = float(SERVER_CFG.get("max_image_MP", 50))  # megapixels
MAX_QUERY_CHARS = int(SERVER_CFG.get("max_query_chars", 2000))

MODEL_CHOICES = ("auto", "all", "detect", "seg", "pose", "obb", "cls", "depth", "sem")
CONF_MIN, CONF_MAX = 0.05, 0.95
IOU_MIN, IOU_MAX = 0.10, 0.95

yolo = YoloService(weights={k: str(ROOT / v) for k, v in CFG["yolo"]["weights"].items()},
                   device=CFG["yolo"].get("device", "cuda"),
                   conf_default=CFG["yolo"].get("conf_default", 0.25),
                   iou_default=CFG["yolo"].get("iou_default", 0.7),
                   vram_safety_MB=CFG.get("selection", {}).get("vram_safety_MB", 800))
try:
    yolo.validate_registry()  # verify actual task/class-map per checkpoint; never crashes boot
except Exception:
    pass
gemma = GemmaClient(base_url=CFG["gemma"]["base_url"],
                    chat_endpoint=CFG["gemma"]["chat_endpoint"],
                    models_endpoint=CFG["gemma"]["models_endpoint"],
                    timeout_s=CFG["gemma"].get("timeout_s", 120),
                    temperature=CFG["gemma"].get("temperature", 1.0),
                    top_p=CFG["gemma"].get("top_p", 0.95),
                    top_k=CFG["gemma"].get("top_k", 64),
                    max_tokens=CFG["gemma"].get("max_tokens", 1024),
                    image_max_edge=CFG["gemma"].get("image_max_edge", 1024))
orch = Orchestrator(yolo, gemma, reports_dir=REPORTS, history_path=HISTORY,
                    annotated_dir=ANNOTATED, runs_dir=RUNS,
                    max_refine_passes=CFG["agent"].get("max_refine_passes", 1),
                    retry_conf=CFG["agent"].get("retry_conf", 0.15),
                    crop_top_k=CFG["agent"].get("crop_top_k", 5),
                    crop_min_conf=CFG["agent"].get("crop_min_conf", 0.35))

app = FastAPI(title="Vision Explainer")
app.mount("/outputs", StaticFiles(directory=ROOT / "outputs"), name="outputs")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    html = ROOT / "static" / "index.html"
    if not html.exists():
        return JSONResponse({"ok": False, "error": "static/index.html missing"})
    return FileResponse(str(html))


@app.get("/api/health")
def health():
    import torch
    g = gemma.health()
    vram = {}
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            vram = {"gpu": torch.cuda.get_device_name(0), "free_MB": round(free / 1e6),
                    "total_MB": round(total / 1e6)}
    except Exception as e:
        vram = {"error": str(e)}
    return {"ok": True, "gemma": g, "yolo_device": yolo.device,
            "yolo_loaded_task": yolo._loaded_task, "vram": vram,
            "x_only_no_fallback": True}


@app.get("/api/models")
def models():
    return {"models": yolo.list_models(), "routing": "detect-first + all",
            "choices": ["auto", "all", "detect", "seg", "pose", "obb", "cls", "depth", "sem"],
            "size": "x", "fallback": False}


@app.get("/api/history")
def history(limit: int = 20):
    limit = max(1, min(int(limit), 100))
    return {"history": read_history(HISTORY, limit=limit)}


@app.post("/api/analyze")
async def analyze(request: Request, image: UploadFile = File(...), query: str = Form(""),
                  model_choice: str = Form("auto"),
                  conf: float | None = Form(None), iou: float | None = Form(None)):
    if (model_choice or "auto").lower().strip() not in MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model_choice '{model_choice}'. Valid: {', '.join(MODEL_CHOICES)}.")
    if conf is not None and not (CONF_MIN <= conf <= CONF_MAX):
        raise HTTPException(400, f"conf={conf} out of range [{CONF_MIN}, {CONF_MAX}].")
    if iou is not None and not (IOU_MIN <= iou <= IOU_MAX):
        raise HTTPException(400, f"iou={iou} out of range [{IOU_MIN}, {IOU_MAX}].")
    if len(query or "") > MAX_QUERY_CHARS:
        raise HTTPException(400, f"query too long ({len(query)} chars, max {MAX_QUERY_CHARS}).")
    if not (image.content_type or "").startswith("image/"):
        # allow by extension too
        if Path(image.filename or "").suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            raise HTTPException(400, "Upload an image file (jpg/png/webp/bmp).")
    try:
        declared = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        declared = 0
    if declared and declared > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"Upload too large (>{MAX_UPLOAD_MB:.0f} MB declared).")
    ext = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        ext = ".jpg"
    name = f"up_{uuid.uuid4().hex[:10]}{ext}"
    dest = UPLOADS / name
    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(image.file, f)
    finally:
        await image.close()
    if dest.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        try:
            dest.unlink()
        except Exception:
            pass
        raise HTTPException(400, f"Upload too large (>{MAX_UPLOAD_MB:.0f} MB).")
    # Verify actual decodability (extension/content-type lie: e.g. AVIF named .jpg,
    # which OpenCV/YOLO cannot read) and normalize everything to real JPEG so the
    # whole pipeline (YOLO, Gemma, thumbnails) sees one format.
    try:
        from PIL import Image
        with Image.open(dest) as im:
            im.load()
            mp = im.width * im.height / 1e6
            if mp > MAX_IMAGE_MP:
                raise HTTPException(400, f"Image too large ({mp:.1f} MP, max {MAX_IMAGE_MP:.0f} MP).")
            if (im.format or "").upper() not in ("JPEG", "JPG", "PNG", "BMP", "WEBP"):
                rgb = im.convert("RGB")
                dest.unlink()
                dest = dest.with_suffix(".jpg")
                rgb.save(dest, "JPEG", quality=92)
    except HTTPException:
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise
    except Exception:
        try:
            dest.unlink()
        except Exception:
            pass
        raise HTTPException(400, "Unreadable image file. Upload a valid JPG/PNG/WEBP/BMP (AVIF/HEIC are converted when readable).")
    try:
        report = orch.run(dest, query=query, model_choice=model_choice, conf=conf, iou=iou)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception:
        eid = uuid.uuid4().hex[:8]
        log.exception("analyze failed eid=%s file=%s", eid, dest.name)
        raise HTTPException(500, f"Internal error (eid={eid}). Details logged server-side.")
    return report
