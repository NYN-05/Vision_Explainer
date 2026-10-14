"""Grounding hygiene eval over stored run reports (no manual labels needed).

Reads outputs/runs/*/report.json, aggregates the `grounding` block written by
the orchestrator (cited ids, unknown ids, coverage, abstention) and writes
outputs/bench/grounding.json. Measures citation discipline, not truth:
a high-coverage run may still misdescribe an object, and a low-coverage run
with abstention can be the correct behavior on hard images.

Usage:
    python scripts/eval_grounding.py [--runs outputs/runs]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "outputs" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "bench" / "grounding.json"))
    args = ap.parse_args()

    rundir = Path(args.runs)
    reports = sorted(rundir.glob("*/report.json")) if rundir.exists() else []
    if not reports:
        print(f"no reports in {rundir}", file=sys.stderr)
        sys.exit(2)

    per_run = []
    for rp in reports:
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        g = rep.get("grounding") or {}
        per_run.append({"run_id": rep.get("run_id"), "task": rep.get("task"),
                        "num_objects": rep.get("num_objects", 0),
                        "coverage": g.get("coverage"),
                        "unknown": g.get("unknown", []),
                        "abstained": g.get("abstained", False),
                        "refined": rep.get("refined", False)})

    coverages = [r["coverage"] for r in per_run if isinstance(r["coverage"], (int, float))]
    out = {
        "num_reports": len(per_run),
        "avg_coverage": round(sum(coverages) / len(coverages), 3) if coverages else None,
        "pct_with_unknown_citations": round(
            sum(1 for r in per_run if r["unknown"]) / len(per_run), 3) if per_run else None,
        "pct_abstained": round(
            sum(1 for r in per_run if r["abstained"]) / len(per_run), 3) if per_run else None,
        "pct_refined": round(
            sum(1 for r in per_run if r["refined"]) / len(per_run), 3) if per_run else None,
        "runs": per_run,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
