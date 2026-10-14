"""Explanation reliability evaluation (v1: detection + grounding focus).

Two independent layers (never conflated):
  structural — deterministic citation checks via validate_answer
               (valid ids? coverage? abstention markers?). Says nothing about truth.
  factual    — hand-reviewed claim fixtures (lab/claims/*.json) with expected
               verdicts. Answer stance per claim is found by a documented keyword
               heuristic (negation window); results are rates over claims, and the
               heuristic's limits are reported, not hidden. Judge-model output, if
               supplied, is stored verbatim as `automated_judgment` — never as truth.

Claim types: presence | attribute | action | spatial | scene.
A cited detection id proves the reference is valid, not that the sentence is true.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from backend.utils import ABSTAIN_PHRASES, validate_answer

NEGATIONS = ("no", "not", "n't", "never", "without", "absent", "none", "neither", "nor")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def stance(answer: str, keywords: list[str]) -> str:
    """asserted | denied | absent. Keyword present + negation in the 4 words
    before the match → denied. Multi-word keywords match as phrases."""
    for s in _SENT_SPLIT.split(answer or ""):
        low = s.lower()
        for k in keywords:
            kl = k.lower()
            idx = low.find(kl)
            if idx < 0:
                continue
            words_before = re.findall(r"[a-z']+", low[:idx])
            window = words_before[-4:]
            if any(n in " ".join(window) for n in NEGATIONS):
                return "denied"
            return "asserted"
    return "absent"


def _uncertain_near(answer: str, keywords: list[str]) -> bool:
    for s in _SENT_SPLIT.split(answer or ""):
        low = s.lower()
        if any(k.lower() in low for k in keywords) and any(p in low for p in ABSTAIN_PHRASES):
            return True
    return False


def judge_claim(expected: str, st: str, uncertain: bool) -> str:
    """Map (expected verdict, detected stance) to an outcome label."""
    if expected == "supported":
        return {"asserted": "supported_correct", "absent": "omission",
                "denied": "contradicted"}[st]
    if expected == "refuted":
        return "hallucinated" if st == "asserted" else "correct_rejection"
    # expected == "insufficient"
    if st in ("absent", "denied") or (st == "asserted" and uncertain):
        return "correct_abstention"
    return "unqualified"


def evaluate_explanation(answer: str, evidence_ids: list[str],
                         claims: list[dict]) -> dict:
    """Score one answer against hand-reviewed claims. Pure; never raises."""
    structural = validate_answer(answer, evidence_ids or [])
    results = []
    for cl in claims or []:
        st = stance(answer, cl.get("keywords", []))
        unc = _uncertain_near(answer, cl.get("keywords", []))
        outcome = judge_claim(cl.get("expected", "insufficient"), st, unc)
        results.append({"id": cl.get("id"), "type": cl.get("type", "presence"),
                        "expected": cl.get("expected"), "stance": st,
                        "uncertain_marked": unc, "outcome": outcome})
    n = len(results) or 1
    bad = sum(1 for r in results if r["outcome"] in ("hallucinated", "contradicted", "unqualified"))
    return {
        "structural": structural,
        "evidence_reference_valid": not structural.get("unknown"),
        "claims": results,
        "unsupported_claim_rate": round(bad / n, 4),
        "supported_correct_rate": round(sum(1 for r in results if r["outcome"] == "supported_correct") / n, 4),
        "omission_rate": round(sum(1 for r in results if r["outcome"] == "omission") / n, 4),
        "correct_abstention_rate": round(sum(1 for r in results if r["outcome"] == "correct_abstention") / n, 4),
        "n_claims": len(results),
        "method_note": "stance by keyword heuristic; factual labels are hand-reviewed; "
                       "citation validity is structural only, not proof of truth",
    }


def load_claims(path: str | Path) -> list[dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d.get("claims", [])
