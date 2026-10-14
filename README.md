# Vision Explainer

Local image understanding: YOLO26X detection + Gemma-4 vision reasoning.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-✓-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26X-blue)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Gemma--4-orange)

A local, GPU-accelerated **Detect → Reason → Report** pipeline for still images.
YOLO26X finds *where* objects are; Gemma-4 (via a local llama.cpp server) explains
*what it means*. Three small custom agents orchestrate the flow, and every run is
persisted to disk with its full reasoning trace.

## Key features

- **3-agent crew** — Detector (YOLO26X) → Reasoner (Gemma-4 vision) → Reporter (disk).
- **7 verified X-series heads** — detect, seg, pose, obb, cls, depth, sem; strict
  X-only policy, never silently falls back to smaller models.
- **Intelligent model selection** — deterministic, explainable routing
  (explicit choice > query/domain keywords > default `detect`); every checkpoint is
  validated at startup and every report records the selection rationale.
- **Run-all fusion mode** — run all 7 heads sequentially and fuse them in one
  Gemma pass, with a per-head annotated gallery.
- **Robust uploads** — files are verified and normalized to JPEG (AVIF/HEIC handled);
  corrupt uploads get a clear `400`, never a cryptic `500`.
- **Close-up crops** — top confident boxes are cropped and sent with the full image,
  so Gemma describes fine detail instead of glossing over small objects.
- **Full audit trail** — annotated images, JSON reports, agent step log, Gemma
  thought + answer, persistent `history.jsonl`.

## Architecture

<p align="center">
  <a href="https://gitdiagram.com/nyn-05/vision_explainer?utm_source=readme&utm_medium=picture">
    <img src="https://gitdiagram.com/nyn-05/vision_explainer/diagram.png" alt="Architecture diagram of Vision Explainer" width="85%" />
  </a>
</p>

| Module                    | Responsibility                                                                |
| ------------------------- | ----------------------------------------------------------------------------- |
| `backend/app.py`        | FastAPI routes, upload handling, static + `/outputs` serving                |
| `backend/agents.py`     | Detector, Reasoner, Reporter, Orchestrator                                    |
| `backend/services.py`   | `YoloService` (registry, selection, VRAM gate, inference) + `GemmaClient` |
| `backend/utils.py`      | Gemma prompt builders, report/history persistence                             |
| `static/index.html`     | Upload UI, agent-step viewer, gallery, history                                |
| `scripts/bench_yolo.py` | Per-head benchmark runner (precision/recall/mAP/latency/VRAM)                 |

## Prerequisites

- Python 3.11, NVIDIA GPU with CUDA (tested on RTX 4050 6 GB).
- [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` with a Gemma-4
  GGUF **plus its `--mmproj` file** (vision input requires it).
- YOLO26X weights in `model/` (`yolo26x.pt`, `yolo26x-seg.pt`, `yolo26x-pose.pt`,
  `yolo26x-obb.pt`, `yolo26x-cls.pt`, `yolo26x-depth.pt`, `yolo26x-sem.pt`).
  Fetch them with the bundled downloader (all sizes × tasks, skips existing):
  `python model/load.py`.

## Quickstart

**0. Clone**

```powershell
git clone https://github.com/NYN-05/Vision_Explainer.git
cd Vision_Explainer
```

**1. Install dependencies**

```powershell
pip install -r requirements.txt
```

**2. Start the llama.cpp server (with vision)**

> Replace the example paths below with wherever your GGUF files live.
> The `--mmproj` flag is mandatory — without it the model loads but image input fails.

```powershell
llama-server.exe -m "C:\Users\JHASHANK\models\gemma-4-E2B_q4_0-it.gguf" `
  --mmproj "C:\Users\JHASHANK\models\gemma-4-E2B-it-mmproj.gguf" `
  --host 127.0.0.1 --port 8080 --ctx-size 8192 -ngl 99
curl.exe http://127.0.0.1:8080/v1/models   # expect a model list, not a connection error
```

Common pitfall: `gguf_init_from_file: failed to open GGUF file` means the `-m` /
`--mmproj` path is wrong (e.g. an unreplaced `<models-dir>` placeholder) — the
server exits before serving anything.

**3. Start the backend**

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

Upload an image, type a question (e.g. *"Describe people and vehicles and their positions"*),
pick `auto` (or a specific head / `all`), and inspect the annotated image, agent steps,
Gemma reasoning, and final answer.

## Configuration (`config.yaml`)

| Section        | Key options                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo`       | `weights` (7 X checkpoints), `device`, `conf_default` (0.25), `iou_default` (0.7)                                                         |
| `gemma`      | `base_url`, endpoints, `temperature` 1.0, `top_p` 0.95, `top_k` 64, `max_tokens` 4096, `image_max_edge` 1536                          |
| `selection`  | `vram_safety_MB` (VRAM pre-check margin), aerial/dense routing cues documented in code                                                          |
| `agent`      | `verify_budget` (max extra YOLO passes: 1), `retry_conf` (0.15), `crop_top_k` (close-ups per request, 0 disables), `crop_min_conf` (0.35) |
| `benchmarks` | `data_root` (labeled dataset path; empty = disabled)                                                                                            |
| `paths`      | `uploads`, `annotated` (legacy), `reports` (legacy), `runs`, `history` under `outputs/`                                               |

## API reference

| Endpoint                              | Description                                                                                                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /`                             | Web UI                                                                                                                                                                        |
| `GET /api/health`                   | YOLO device, Gemma reachability, VRAM, loaded head                                                                                                                            |
| `GET /api/models`                   | 7 heads: weight path, existence, domain, output format, `validation` status                                                                                                 |
| `GET /api/history?limit=20`         | Recent analyses (newest first)                                                                                                                                                |
| `POST /api/analyze`                 | Multipart: `image` (file), `query` (text), `model_choice` (`auto`\|`all`\|`detect`\|`seg`\|`pose`\|`obb`\|`cls`\|`depth`\|`sem`), `conf`?, `iou`? |
| `GET /outputs/runs/<run_id>/<file>` | Per-run files: input, annotated images, `report.json`                                                                                                                       |

**Run folders:** every request creates `outputs/runs/<run_id>/` holding its input image,
annotated image(s), and `report.json`; the report's `run_files` lists each file's URL.
Legacy `outputs/annotated/` + `outputs/reports/` files from older runs stay readable.

**Report schema** (`outputs/runs/<run_id>/report.json`): `id`, `run_id`, `created_at`, `image`, `query`,
`task`, `requested_model`, `model`, `device`, `conf`, `num_objects`, `detections`
(+ `all_detections`, `annotated_gallery`, `per_task` in `all`-mode), `thought`,
`final_answer`, `selection_rationale`, `validation_status`, `fallback_actions`,
`hardware`, `steps`, `refined`, plus evidence keys: `evidence` (citable `[#dN]` records),
`grounding` (cited/unknown ids, coverage, abstention), `crop_provenance` (crop boxes in
original-image coords), `provenance` (image hash, library versions, weight sizes/mtimes),
`retry_reason`, and `cross_checks` (`all`-mode head-agreement checks). New keys are
additive — old clients keep working.

**Status codes:** `400` unreadable/oversize upload, unknown `model_choice`, or `conf`/`iou`
out of range · `500` internal error (generic message, details logged server-side) ·
`502` YOLO/Gemma inference failure (e.g. VRAM exhaustion — X-only policy refuses silent downgrade).

## Model registry

Verified at startup by `validate_registry()` (actual checkpoint task + class count —
compatibility is never inferred from filenames):

| Head   | Weight               | Domain             | Classes                 | Output               |
| ------ | -------------------- | ------------------ | ----------------------- | -------------------- |
| detect | `yolo26x.pt`       | general (COCO)     | 80                      | boxes + labels       |
| seg    | `yolo26x-seg.pt`   | general (COCO)     | 80                      | masks + boxes        |
| pose   | `yolo26x-pose.pt`  | COCO-person        | 1                       | boxes + 17 keypoints |
| obb    | `yolo26x-obb.pt`   | aerial-oriented    | 15 (plane, ship, …)    | rotated boxes        |
| cls    | `yolo26x-cls.pt`   | general (ImageNet) | 1000                    | top-5 labels         |
| depth  | `yolo26x-depth.pt` | general            | 1                       | depth map            |
| sem    | `yolo26x-sem.pt`   | street-scene       | 19 (road, sidewalk, …) | semantic map         |

## Benchmarks

No claim of "best model" without evidence. Supply a labeled dataset, then:

```powershell
python scripts/bench_yolo.py --data bench/coco8.yaml          # all heads
python scripts/bench_yolo.py --task detect --data bench/dota.yaml
# → outputs/bench/<task>.json: precision, recall, mAP50, mAP50-95, latency, peak VRAM
python scripts/eval_grounding.py                              # citation hygiene over stored runs
# → outputs/bench/grounding.json: coverage, unknown-citation rate, abstention rate
```

## Troubleshooting

| Symptom                          | Cause / fix                                                                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `400 Unreadable image file`    | Corrupt upload, or a format PIL can't decode. Note: AVIF files misnamed `.jpg` are auto-converted and work.                                           |
| `502 … Insufficient VRAM`     | ~6 GB GPUs fit one X-head + Gemma. Free VRAM (close GPU apps / reduce llama-server offload) and retry. The app never silently downgrades to S/M models. |
| `502 … llama.cpp chat failed` | llama.cpp server down or started without `--mmproj`. Check `:8080/v1/models` and restart with `--mmproj`.                                         |
| Empty detections, valid result   | Empty is a legitimate outcome. A verify-retry runs at most once, only when low confidence coincides with Gemma uncertainty.                             |
| `depth`/`sem` look odd       | Those heads' output parsing is code-complete but empirically unvalidated — flagged in `CHANGES.md`.                                                  |

## Project structure

```text
obj_detection/
├── backend/{app,agents,services,utils}.py
├── static/index.html
├── scripts/bench_yolo.py
├── model/load.py                 # weight downloader (all sizes × tasks)
├── model/yolo26x*.pt            # 7 X weights (via load.py, git-ignored)
├── outputs/{uploads,runs,annotated,reports,bench,history.jsonl}
├── config.yaml  requirements.txt  CHANGES.md
```

See `CHANGES.md` for the dated change history.

## Roadmap

- Webcam/video-stream mode (frame sampling + tracking).
- Custom fine-tuned heads registered with provenance + per-head benchmarks.
- Evaluation harness with stored bench baselines gating model-selection claims.

## License

No license chosen yet — add a `LICENSE` file (e.g. MIT or Apache-2.0) before distributing.
