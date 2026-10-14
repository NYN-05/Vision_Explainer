# Changes

Changelog for `obj_detection` (YOLO26X + Gemma-4 agent). Newest first.
(Times are local, from file modification times.)

## 2026-10-14 — Architecture audit fixes, phases 0–3 (0,1,2,4,3)

- Phase 0: verified 14 code-backed defects (D1–D14: unbounded conf/iou, 500s leaking
  internals, unlocked history, non-atomic reports, no upload caps, Gemma-worded retry
  gating, no citation requirement/output validation, comparable fused confidences,
  unvalidated depth/sem, no crop/provenance records, shared-service mutation,
  unclamped history limit).
- Phase 1 (hardening): `model_choice`/`conf`/`iou`/`query` validated (400s);
  upload caps (`max_upload_MB` 25, `max_image_MP` 50, content-length pre-check);
  500s sanitized with server-side logged eid; atomic report writes; locked
  `history.jsonl` appends (msvcrt/fcntl); history limit clamped 1–100.
- Phase 2 (evidence boundary): `build_evidence()` assigns citable `[#dN]` ids;
  prompts mandate per-claim citations + UNCERTAIN/abstain; `validate_answer()`
  records `grounding` (cited/unknown/coverage/abstained); crops carry box/pad
  provenance; retry also triggers measurement-driven (avg_conf < retry_conf).
- Phase 4 (provenance + eval): per-run `provenance` (image sha256/size, torch/
  ultralytics versions, weight sizes/mtimes); new `scripts/eval_grounding.py`
  → `outputs/bench/grounding.json` (coverage, unknown-citation/abstention rates).
- Phase 3 (fusion honesty): `cross_check_fused()` (detect-persons-vs-pose,
  detect-vs-seg; cls/depth/sem marked na with reasons); fused prompts/summaries
  state confidences are not cross-head comparable; `predict()` tags depth/sem
  with `output_note: unvalidated head`.
- All new report keys additive (`evidence`, `grounding`, `crop_provenance`,
  `provenance`, `retry_reason`, `cross_checks`, `output_note`).
- Verified: py_compile, pure-function unit checks, TestClient (health/models/400s),
  eval_grounding on 4 legacy reports. Live YOLO+Gemma run not re-verified here.

## 2026-10-14 22:33 — Audit fixes: keyboard, contrast, gallery, polish

- `static/index.html` only: keyboard-focusable file input (overlay + `:focus-within`
  ring), `--faint` → `#5f6875` (5.64:1 on white, 5.07:1 on panels), gallery rebuilt
  with `addEventListener` (no inline handlers), dead `.sp` rule now used, 44px ghost
  buttons, `alert()` → `#status.err` (aria-live announced), escaped history tooltips.
- Verified: `node --check`, md vectors, live run 200 (9 objs, history linked).

## 2026-10-14 22:17 — SaaS workspace redesign (single file)

- `static/index.html` rebuilt Linear/Vercel-style: top bar with brand + status pills,
  compact control card (dropzone, query, model, conf, Analyze), answer-first results
  (summary stats incl. model + YOLO ms, Markdown answer + copy, image compare, gallery),
  collapsible steps/thought/JSON (+copy), run files, history as a table
  (time, run, question, task, objs, refined-status, image link).
- One additive backend line: history entries now store `refined` for the Status column.
- Empty state is a 3-step guide; loading/error/broken-image states explicit.
- Verified: `node --check`, md vectors, 16 legacy IDs, `GET /` 200, live run
  (9 objs, SCENE/…/CLOSE Markdown, 5 crops, history row complete). Real-browser
  render and touch gestures not verified.

## 2026-10-14 22:14 — Richer descriptions: detail prompts + close-up crops

- Prompts demand forensic structure (SCENE / OBJECTS / RELATIONS / TEXT / CLOSE);
  `max_tokens` 2048 → 4096, `image_max_edge` 1024 → 1536.
- Top-5 confident boxes are cropped (`runs/<id>/crops/crop_N.jpg`) and sent with the
  full image; prompt maps each crop to its detection. Bounded, configurable
  (`crop_top_k`, `crop_min_conf`); skipped for box-less outputs.
- Verified: same test image went from ~500 to ~5900 answer chars with 5 crops used.

## 2026-10-14 22:08 — Dashboard redesign (single-file, no framework)

- `static/index.html` rebuilt as a minimal AI dashboard: header + status pills
  (YOLO device/head, Gemma OK/DOWN, VRAM, X-only policy), control panel with
  drag-and-drop upload, results with summary stats, image compare, agent timeline,
  rendered-Markdown answer, collapsible thought/JSON (+copy button), run files,
  improved history. Still one file, zero dependencies.
- Markdown rendered via escape-first inline formatter (verified: `<script>`/`<img onerror>`
  neutralized); also fixed audit gaps en passant: `h1`, `aria-live` status,
  `:focus-visible`, `pre` overflow, 44px-class touch targets.
- Verified: full JS syntax (`node --check`), md unit vectors, all 16 legacy IDs intact,
  `GET /` 200, `/api/health` OK. Real-browser render and touch gestures not verified.

## 2026-10-14 22:02 — Per-run output folders

- Every request now creates `outputs/runs/<run_id>/` holding its input image,
  annotated image(s) (`annotated_<task>.jpg`, `annotated_<task>_retry.jpg` on verify-retry),
  and `report.json`; reports link them via `run_files`. Global `history.jsonl` kept
  (entries gained `run_id`). Legacy `outputs/annotated|reports` stay readable.
- `YoloService.predict()` takes `ann_name` (deterministic names; `predict_all` uses
  `annotated_<task>.jpg`); `config.yaml` gained `paths.runs`; UI gained a Run-folder section.
- Fixed along the way: VRAM gate measured with the previous head still resident and
  wrongly blocked mid-sequence in `all`-mode — now unloads first, then measures.
- Verified: single → 3 files/run; `all` → 9 files/run, gallery URLs + history all
  pointing inside the run folder.

## 2026-10-14 21:50 — Renamed project to Vision Explainer

- UI title/heading (`static/index.html`), API title (`backend/app.py`),
  README heading → "Vision Explainer". No logic touched; verified via TestClient.

## 2026-10-14 20:59 — AVIF upload 500 fix

- Cause: AVIF bytes misnamed `.jpg` (your `test_2.jpg`) — OpenCV/YOLO can't decode AVIF,
  `results[0]` threw `IndexError` → generic 500.
- `backend/app.py`: verify decodability with PIL at upload; unreadable → 400 with a clear
  message; readable non-JPEG/PNG (AVIF/HEIC/WEBP) → converted to real JPEG so the whole
  pipeline sees one format.
- `backend/services.py` `predict()`: empty YOLO result list now raises `RuntimeError`
  with meaning (→ 502) instead of `IndexError` (→ 500).
- `requirements.txt`: added `pi-heif` (AVIF/HEIC decode; ultralytics auto-installed it).
- Verified: AVIF-as-JPG → 200 (was 500), 0-byte file → 400, normal JPG → 200 unchanged.

## 2026-10-14 20:54 — Codebase consolidation (no behavior change)

- Merged `backend/agents/{detector,reasoner,reporter,orchestrator}.py` → `backend/agents.py`;
  `backend/services/{yolo_service,gemma_client}.py` → `backend/services.py`;
  `backend/utils/{prompts,store}.py` → `backend/utils.py`. Removed empty
  `backend/agents/`, `backend/services/`, `backend/utils/` dirs and all 4 empty
  `__init__.py` (namespace packages; imports verified). Updated imports in
  `backend/app.py`, `scripts/bench_yolo.py`.
- Removed 1 unused import (`import json` in prompts). No logic touched.
- Regression: health/models/validation, `auto→detect` (9 objs), `all` (7 gallery),
  selection (`satellite→obb`, `segment→seg`), bench `--help` — all pass.

## 2026-10-14 20:44 — Intelligent model selection

- `backend/services/yolo_service.py`: `MODEL_REGISTRY` (domain/output/capabilities/VRAM
  reserve/validation), `validate_registry()` (loads each checkpoint, checks actual task +
  class map), `select_model()` (explicit > keywords+domain > default; no Gemma call),
  `hw_info()`, `check_fit()` VRAM gate (X-only, errors instead of S/M fallback),
  `list_models()` exposes `domain/output/validation`.
- `backend/agents/orchestrator.py`: `select` step with rationale, `VERIFY_BUDGET=1`,
  retry only on detector signal + Gemma corroboration, `fallback_actions`, HW snapshot.
- `backend/agents/reporter.py`: additive `selection_rationale`, `validation_status`,
  `fallback_actions`, `hardware` (old clients unaffected).
- `backend/agents/reasoner.py`: `needs_refine(detection, query)` — empty/low-conf alone
  is never failure proof.
- `backend/app.py`: passes `vram_safety_MB`, startup `validate_registry()` (guarded).
- `config.yaml`: `selection:` + `benchmarks:` blocks.
- `scripts/bench_yolo.py` (new): `YOLO.val()` per head → `outputs/bench/<task>.json`.
- Verified: 7/7 heads `verified` (detect 80 COCO, seg 80, pose 1, obb 15 aerial,
  cls 1000 ImageNet, depth 1, sem 19 street-scene); selection/API/`all`-mode tests pass.

## 2026-10-14 20:37 — Run-all option

- `model_choice=all` runs all 7 X heads sequentially + single fused Gemma pass, no refine.
- Reports add `all_detections`, `detections_flat`, `annotated_gallery` (7 urls), `per_task`.
- UI: `all 7 YOLO26X` dropdown entry + gallery grid + dict-safe detections render.
- Fixed `build_yolo_summary` crash on `None`-confidence entries (safe formatting).

## 2026-10-14 20:18 — Initial build

- FastAPI backend (`backend/app.py`): `/`, `/api/health`, `/api/models`, `/api/history`,
  `POST /api/analyze`; serves `static/index.html`, mounts `/outputs`.
- 3 custom agents: detector → reasoner (Gemma-4 vision via llama.cpp `:8080`) → reporter.
- Detect-first routing, strict X-only (`model/yolo26x*.pt`), disk persist
  (`outputs/reports/*.json`, `outputs/history.jsonl`).
- Fixes during verification: Gemma `max_tokens` 1024 → 2048 (thought+answer truncation);
  empty-answer fallback keeps raw text.
