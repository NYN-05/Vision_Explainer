"""Vision Reliability Lab tests (stdlib only). Synthetic fixtures, known expectations.

Run:  python scripts/test_lab.py
Covers: perturbation determinism/validation/remap, COCO-style metrics incl.
duplicates/order-independence/empty edge cases, stability-vs-accuracy split,
explanation stance/judgment matrix. No dataset, no GPU, no Gemma required.
"""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import lab_compare, lab_metrics
from backend.lab_explain import evaluate_explanation, judge_claim, stance
from backend.lab_fixtures import make_still_life
from backend.perturb import OP_REGISTRY, apply_perturbation, list_ops

GT2 = {
    "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20]},
    ],
    "categories": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}],
}


def P(img, cat, x, y, w, h, s):
    return {"image_id": img, "category_id": cat, "bbox": [x, y, w, h], "score": s}


class TestPerturb(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.img, _ = make_still_life(self.tmp)

    def test_registry_kinds(self):
        ops = {o["name"]: o["kind"] for o in list_ops()}
        self.assertEqual(ops["gaussian_blur"], "photometric")
        self.assertEqual(ops["occlusion"], "geometric")
        self.assertEqual(ops["crop"], "geometric")
        self.assertEqual(len(ops), 8)

    def test_determinism(self):
        outs = []
        for _ in range(2):
            r = apply_perturbation(self.img, "gaussian_noise", 2, 7, self.tmp / "n.jpg")
            self.assertNotIn("error", r)
            outs.append(Path(r["output"]).read_bytes())
        self.assertEqual(outs[0], outs[1])

    def test_seed_matters_for_stochastic(self):
        a = apply_perturbation(self.img, "gaussian_noise", 2, 7, self.tmp / "a.jpg")["sha256"]
        b = apply_perturbation(self.img, "gaussian_noise", 2, 8, self.tmp / "b.jpg")["sha256"]
        self.assertNotEqual(a, b)

    def test_validation(self):
        with self.assertRaises(ValueError):
            apply_perturbation(self.img, "nope", 1, 0, self.tmp / "x.jpg")
        with self.assertRaises(ValueError):
            apply_perturbation(self.img, "gaussian_blur", 9, 0, self.tmp / "x.jpg")

    def test_resize_keeps_size(self):
        from PIL import Image
        r = apply_perturbation(self.img, "resize", 3, 0, self.tmp / "r.jpg")
        with Image.open(r["output"]) as im:
            self.assertEqual(im.size, (640, 480))

    def test_crop_remap_math(self):
        from backend.perturb import remap_boxes
        from PIL import Image
        tiny = self.tmp / "t.jpg"
        Image.new("RGB", (100, 100), "white").save(tiny)
        rec = apply_perturbation(tiny, "crop", 1, 3, self.tmp / "c.jpg")  # frac .9 → 90x90
        boxes = [{"id": 1, "bbox": [0, 0, 100, 100]}]  # whole image
        kept, invalid = remap_boxes(rec, boxes)
        self.assertEqual(len(kept), 1)
        x1, y1, x2, y2 = kept[0]["bbox"]
        self.assertAlmostEqual(x2 - x1, 100.0, places=4)  # scaled back to full size
        self.assertAlmostEqual(y2 - y1, 100.0, places=4)
        # tiny box fully cropped out → invalid with reason
        kept2, invalid2 = remap_boxes(rec, [{"id": 2, "bbox": [0, 0, 1, 1]}])
        self.assertTrue(kept2 == [] or invalid2 != [])
        if invalid2:
            self.assertIn("reason", invalid2[0])

    def test_occlusion_invalid(self):
        from backend.perturb import remap_boxes
        rec = {"_remap_fn": None}
        from PIL import Image
        tiny = self.tmp / "o.jpg"
        Image.new("RGB", (100, 100), "white").save(tiny)
        rec = apply_perturbation(tiny, "occlusion", 3, 1, self.tmp / "o2.jpg")
        import random
        from backend.perturb import OP_REGISTRY as R
        with Image.open(tiny).convert("RGB") as im:
            _, fn = R["occlusion"]["apply"](im, random.Random(1), 3, {})
        kept, invalid = fn([{"id": 1, "bbox": [0, 0, 100, 100]}], None)
        self.assertEqual(invalid, [])  # 35% < 50% of full-image box
        kept, invalid = fn([{"id": 2, "bbox": [0, 0, 10, 10]}], None)
        # depends on rng position; just check schema when invalid
        for inv in invalid:
            self.assertIn("reason", inv)


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        m = lab_metrics.evaluate(GT2, [P(1, 0, 10, 10, 20, 20, .9), P(1, 1, 60, 60, 20, 20, .8)])
        self.assertEqual(m["mAP"], 1.0)
        self.assertEqual(m["mAP50"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)

    def test_empty_preds(self):
        m = lab_metrics.evaluate(GT2, [])
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fn"], 2)
        self.assertIsNone(m["precision"])
        self.assertEqual(m["mAP"], 0.0)  # GT exists, nothing predicted: worst score
        self.assertEqual(m["mAP50"], 0.0)

    def test_duplicates_second_is_fp(self):
        m = lab_metrics.evaluate(GT2, [P(1, 0, 10, 10, 20, 20, .9), P(1, 0, 10, 10, 20, 20, .8),
                                       P(1, 1, 60, 60, 20, 20, .7)])
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["fp"], 1)

    def test_order_independence(self):
        a = lab_metrics.evaluate(GT2, [P(1, 1, 60, 60, 20, 20, .9), P(1, 0, 10, 10, 20, 20, .8)])
        b = lab_metrics.evaluate(GT2, [P(1, 0, 10, 10, 20, 20, .8), P(1, 1, 60, 60, 20, 20, .9)])
        self.assertEqual(a["mAP"], b["mAP"])
        self.assertEqual(a["tp"], 2)

    def test_partial_recall(self):
        m = lab_metrics.evaluate(GT2, [P(1, 0, 10, 10, 20, 20, .9)])
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["fn"], 1)

    def test_degradation(self):
        d = lab_metrics.degradation({"mAP": 0.8, "mAP50": 0.9, "precision": None, "recall": 0.5},
                                    {"mAP": 0.6, "mAP50": 0.7, "precision": 1.0, "recall": 0.5})
        self.assertEqual(d["mAP_delta"], -0.2)
        self.assertIsNone(d["precision_delta"])


class TestCompare(unittest.TestCase):
    B = lambda self, l, c, x=0: {"label": l, "conf": c, "x1": x, "y1": 0, "x2": x + 10, "y2": 10}

    def test_stable(self):
        r = lab_compare.compare([self.B("cat", .9)], [self.B("cat", .88)])
        self.assertEqual(r["stability"], 1.0)
        self.assertEqual(r["disappeared"], [])

    def test_disappeared_and_fp(self):
        r = lab_compare.compare([self.B("cat", .9), self.B("dog", .8, 50)], [self.B("cat", .9)])
        self.assertEqual(r["disappeared"], [1])
        r2 = lab_compare.compare([self.B("cat", .9)], [self.B("cat", .9), self.B("dog", .7, 50)])
        self.assertEqual(r2["new_false_positives"], [1])

    def test_misclassified(self):
        r = lab_compare.compare([self.B("cat", .9)], [self.B("dog", .9)])
        self.assertEqual(len(r["misclassified"]), 1)
        self.assertEqual(r["stability"], 0.0)

    def test_stability_is_not_accuracy(self):
        # identical wrong outputs: stability 1.0, but GT eval recall 0.0
        wrong = [self.B("dog", .9)]
        r = lab_compare.compare(wrong, [self.B("dog", .9)])
        self.assertEqual(r["stability"], 1.0)
        m = lab_metrics.evaluate(GT2, [P(1, 5, 0, 0, 10, 10, .9)])
        self.assertEqual(m["recall"], 0.0)


class TestExplain(unittest.TestCase):
    def test_stance(self):
        self.assertEqual(stance("There is a red square here.", ["red square"]), "asserted")
        self.assertEqual(stance("There is no red square here.", ["red square"]), "denied")
        self.assertEqual(stance("Nothing to report.", ["red square"]), "absent")

    def test_judge_matrix(self):
        self.assertEqual(judge_claim("supported", "asserted", False), "supported_correct")
        self.assertEqual(judge_claim("supported", "absent", False), "omission")
        self.assertEqual(judge_claim("supported", "denied", False), "contradicted")
        self.assertEqual(judge_claim("refuted", "asserted", False), "hallucinated")
        self.assertEqual(judge_claim("refuted", "absent", False), "correct_rejection")
        self.assertEqual(judge_claim("insufficient", "asserted", True), "correct_abstention")
        self.assertEqual(judge_claim("insufficient", "asserted", False), "unqualified")

    def test_full(self):
        claims = [{"id": "c1", "type": "presence", "expected": "supported", "keywords": ["red square"]},
                  {"id": "c2", "type": "presence", "expected": "refuted", "keywords": ["cat"]}]
        ans = "The red square [#d1] is on the left. No cat is visible."
        r = evaluate_explanation(ans, ["d1"], claims)
        self.assertTrue(r["evidence_reference_valid"])
        self.assertEqual(r["unsupported_claim_rate"], 0.0)
        self.assertEqual(r["n_claims"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
