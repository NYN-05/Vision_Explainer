# Implementation Audit — Vision Explainer → Vision Reliability Lab

Date: 2026-10-15. Source of truth: repository code as read (backend/*.py, static/index.html,
config.yaml, scripts/*). Decisions per user direction: no v1 dataset (synthetic fixtures only),
no new metric deps (pycocotools absent — hand-rolled COCO-style eval validated by fixtures),
full lab UI, product renamed with existing functionality preserved.

## 1. Verified architecture (code, not docs)

- `backend/app.py` — FastAPI; `POST /api/analyze` (validated conf/iou/model_choice/query,
  upload caps, sanitized 500s); `GET /api/{health,models,history}`; serves `static/`, `/outputs`.
- `backend/agents.py` — Detector/Reasoner/Reporter + Orchestrator (select → detect → crops →
  evidence → reason → ≤1 verify-retry → report). Per-run `outputs/runs/<id>/` folders.
- `backend/services.py` — `YoloService`: X-only registry, `validate_registry()`, VRAM gate,
  one-head-at-a-time; `GemmaClient` (llama.cpp). **Known race:** `_model`/`_loaded_task`
  mutated without a lock (audit D13) — must fix for lab concurrency.
- `backend/utils.py` — prompts, `build_evidence`/`validate_answer` grounding primitives,
  `crop_detections` (box provenance), `build_provenance`, atomic writes, locked history,
  `cross_check_fused`.
- `static/index.html` — single-file UI; 16 legacy element IDs must stay intact.
- `scripts/bench_yolo.py` (aggregate YOLO val), `scripts/eval_grounding.py` (citation hygiene).
- Report keys additive by convention; `config.yaml` is runtime truth.

## 2. Reuse inventory

Reuse unchanged: model registry/validation/selection, VRAM gate, upload verify/normalize,
evidence + grounding helpers, provenance builder, atomic persistence, Gemma client,
report/history conventions. Lab lives in new modules + new routes; `/api/analyze` untouched.

## 3. Planned modifications

| # | Piece | Location | Notes |
|---|---|---|---|
| 1 | Perturbation engine | `backend/perturb.py` (new) | PIL-only registry; photometric vs geometric; seeded determinism; box remap or explicit invalid marking; sha256 + params recorded |
| 2 | GT eval engine | `backend/lab_metrics.py` (new) | COCO-format GT in, greedy IoU matching, P/R, mAP@0.5, mAP@0.5:0.95 (101-pt interp), per-class, FP/FN, conf histograms; baseline vs perturbed; consistency separate from accuracy |
| 3 | Comparison engine | `backend/lab_compare.py` (new) | IoU-based (never positional) matching baseline↔perturbed; disappearances, flips, drift; failure triplets |
| 4 | Explanation eval | `backend/lab_explain.py` (new) + `lab/claims/*.json` fixtures | Structural checks via `validate_answer`; factual Supported/Refuted/Insufficient-evidence verdicts vs hand-reviewed claims; judge output labeled automated, off by default |
| 5 | GPU serialization | `services.py` lock | `threading.Lock` around predict/predict_all; shared by serving + lab |
| 6 | Orchestrator | `backend/lab_run.py` (new) | 10-stage sequence, stage manifest, background thread + status file; detection-only mode (no Gemma); explain opt-in |
| 7 | API | `app.py` additions | `POST /api/experiments`, `GET /api/experiments/{id}`, `GET /api/experiments` (list); validation mirrors analyze |
| 8 | UI | `static/lab.html` (new) + nav link | Configure, run, poll status, view metrics/failures, open report JSON; reuse index.html styles; legacy IDs untouched |
| 9 | Rename | strings only | "Vision Reliability Lab" in README/app title/UI; keep routes, schema, behavior |
| 10 | Tests | `scripts/test_lab.py` (stdlib unittest) | Determinism hashes, synthetic P/R/mAP hand-checks, positional-mismatch trap, empty/duplicate edge cases, stability-vs-accuracy split |

## 4. Risks

- Hand-rolled mAP diverging from COCO semantics → mitigated by tiny fixtures with hand-computed
  expectations plus cross-check of mAP@0.5==1.0 on a perfect-prediction fixture.
- Geometric box remap errors → each op declares remap math; invalid cases excluded with reasons,
  counted separately, never silently scored.
- GPU lock serializes serving inference too → acceptable (single GPU was already effectively serial).
- Background-thread experiments vs server reload → status/manifest on disk; thread is daemon.
- Scope creep into 7 heads → v1 detection-only; other heads reuse the same stage interfaces later.

## 5. Acceptance criteria (v1)

1. `python scripts/test_lab.py` green: determinism, synthetic metrics, edge cases, no-invented-numbers.
2. Detection-only experiment runs end-to-end on a synthetic COCO fixture (generated images +
   hand labels) with no Gemma server: report.json contains baseline/perturbed metrics,
   degradation, consistency, failure artifacts, config, provenance.
3. Explanation eval runs on claim fixtures; structural and factual verdicts stored separately.
4. Existing `/api/analyze` contract unchanged (regression: health/models/400s + live run).
5. No metric in any report lacks a stored source artifact; empty/invalid cases listed with reasons.
