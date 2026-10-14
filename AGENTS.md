# AGENTS.md — Vision Reliability Lab

Local Detect → Reason → Report + robustness lab: YOLO26X (CUDA) + Gemma-4 via llama.cpp. No cloud, no pytest/lint/CI in repo.

## Run it (order matters)
- Install: `pip install -r requirements.txt` (Python 3.11, NVIDIA CUDA expected; falls back to CPU)
- Weights: `python model/load.py` — 7× `model/yolo26x*.pt`, skipped if present. Never commit (`*.pt` git-ignored).
- Gemma first: `llama-server.exe -m "<gemma-gguf>" --mmproj "<mmproj-gguf>" --host 127.0.0.1 --port 8080 --ctx-size 8192 -ngl 99` — `--mmproj` is mandatory for vision. Check `curl.exe http://127.0.0.1:8080/v1/models`.
- Backend: `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`, open `http://127.0.0.1:8000/`
- Bench (needs dataset): `python scripts/bench_yolo.py --data <yaml>` or set `benchmarks.data_root` in `config.yaml`

## Architecture (4 backend files, don't re-split)
- `backend/app.py` — FastAPI routes, upload verify/normalize, mounts `/outputs` + `/static`. Entry `backend.app:app`.
- `backend/agents.py` — Detector/Reasoner/Reporter + Orchestrator (select → detect → crops → reason → ≤1 verify-retry → report).
- `backend/services.py` — `YoloService` (registry, VRAM gate, inference) + `GemmaClient`. Source of truth for routing/selection.
- `backend/utils.py` — Gemma prompts, `crop_detections`, report/history persistence.
- `static/index.html` — entire UI, zero deps. `scripts/bench_yolo.py`, `model/load.py` standalone.
- Lab (new, additive — `/api/analyze` contract untouched): `backend/perturb.py` (8 seeded
  perturbation ops), `backend/lab_metrics.py` (COCO-style eval, no new deps),
  `backend/lab_compare.py` (IoU matching, stability≠accuracy), `backend/lab_explain.py`
  + `lab/claims/` (structural + hand-reviewed factual), `backend/lab_fixtures.py`
  (synthetic demo), `backend/lab_run.py` (background orchestrator → `outputs/lab/<id>/`),
  `static/lab.html` (`/lab`), `scripts/test_lab.py` (stdlib unittest), `scripts/eval_grounding.py`.
- `config.yaml` is runtime truth (weights, Gemma params, `agent.crop_top_k/crop_min_conf`, `selection.vram_safety_MB`, `paths.runs`, `paths.lab`).

## Hard constraints (from user, do not relax)
- **X-only, no fallback:** never load/suggest/download `n/s/m/l` weights. Missing weight → `FileNotFoundError`; low VRAM → `RuntimeError`/502 with message, never silent downgrade.
- **One X-head at a time:** always `_unload()` before `check_fit()`/load. RTX 4050 6GB fits one head + Gemma.
- **Registry trust:** never infer task from filename. `validate_registry()` loads each checkpoint and sets `verified | task-mismatch:<actual> | missing | load-error`.
- **Empty detections are valid:** retry only when `ReasonerAgent.needs_refine()` signals AND Gemma answer contains `no object`/`uncertain`/`nothing`, max 1 extra pass (`VERIFY_BUDGET`).

## Gotchas agents hit
- Uploads lie: AVIF named `.jpg` breaks OpenCV/YOLO. `app.py` PIL-verifies (`Image.open().load()`), converts readable non-JPEG to JPEG, 400s corrupt files. `predict()` raises `RuntimeError` on empty result list, not `IndexError`.
- Gemma failures look like: `gguf_init_from_file failed` = wrong `-m/--mmproj` path; `502 llama.cpp chat failed` = server down or started without `--mmproj`.
- `depth`/`sem` parsing is code-complete but empirically unvalidated — say so, don't claim it works.
- `outputs/`, `*.pt`, `test_*.jpg/avif` are git-ignored. Per-run output is `outputs/runs/<run_id>/{input.*, annotated_<task>.jpg, crops/, report.json}`; report keys are additive (old clients must keep working).
- Frontend: keep the 16 legacy element IDs (`img,q,model,conf,go,status,ann,summary,steps,thought,dets,runfiles,hist,health,gallery,answer`). Markdown renderer is escape-first — never inject raw HTML.

## Verify without test suite
- Lab unit checks: `python scripts/test_lab.py` (20 tests, no GPU/Gemma/dataset needed).
- Serving: `python -c "from fastapi.testclient import TestClient; from backend.app import app; ..."` for `/api/health`, `/api/models`, `auto→detect`, `all`-mode, selection (`satellite→obb`).
- Frontend: `node --check` on extracted `<script>`; check contrast/focus/keyboard if touching CSS.
- After backend changes, confirm `GET /` 200 and one live `/api/analyze` run creates 3 files (single) or 9 files (`all`) with valid history row.
