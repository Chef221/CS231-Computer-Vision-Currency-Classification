import json
import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_recall_fscore_support,
)

from src.evaluation.value_metrics import CLASS_ORDER, evaluate_classification


class EvaluateClassificationTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = evaluate_classification(CLASS_ORDER, CLASS_ORDER)
        self.assertEqual(result["confusion_matrix"], np.eye(6, dtype=int).tolist())
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["mae_vnd"], 0.0)
        self.assertEqual(result["total_gap_vnd"], 0)
        self.assertEqual(result["error_pairs"], [])

    def test_documented_seven_image_example(self):
        true = [10000, 10000, 20000, 50000, 100000, 200000, 500000]
        pred = [10000, 20000, 20000, 50000, 100000, 200000, 200000]
        result = evaluate_classification(true, pred)
        self.assertEqual(result["class_order_vnd"], list(CLASS_ORDER))
        self.assertEqual(result["n_samples"], 7)
        self.assertAlmostEqual(result["accuracy"], 5 / 7)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertAlmostEqual(result["mae_vnd"], 310000 / 7)
        self.assertEqual(result["total_gap_vnd"], 310000)
        self.assertEqual(result["confusion_matrix"][0][1], 1)
        self.assertEqual(result["confusion_matrix"][5][4], 1)
        self.assertEqual(result["per_class"][0]["tp"], 1)
        self.assertEqual(result["per_class"][0]["fn"], 1)
        self.assertAlmostEqual(result["per_class"][1]["precision"], 0.5)
        self.assertEqual(result["error_pairs"], [
            {"true_vnd": 10000, "predicted_vnd": 20000, "count": 1,
             "gap_vnd": 10000, "total_gap_vnd": 10000},
            {"true_vnd": 500000, "predicted_vnd": 200000, "count": 1,
             "gap_vnd": 300000, "total_gap_vnd": 300000},
        ])

    def test_same_accuracy_but_larger_gap(self):
        small = list(CLASS_ORDER)
        small[1] = 10000
        large = list(CLASS_ORDER)
        large[5] = 10000
        a = evaluate_classification(CLASS_ORDER, small)
        b = evaluate_classification(CLASS_ORDER, large)
        self.assertEqual(a["accuracy"], b["accuracy"])
        self.assertAlmostEqual(a["macro_f1"], b["macro_f1"])
        self.assertAlmostEqual(b["mae_vnd"], 49 * a["mae_vnd"])

    def test_all_wrong_still_distinguishes_gap(self):
        near = evaluate_classification([10000] * 6, [20000] * 6)
        far = evaluate_classification([10000] * 6, [500000] * 6)
        self.assertEqual(near["accuracy"], 0.0)
        self.assertEqual(far["macro_f1"], 0.0)
        self.assertEqual(near["mae_vnd"], 10000)
        self.assertEqual(far["mae_vnd"], 490000)
        self.assertEqual(far["error_pairs"][0]["count"], 6)

    def test_all_six_classes_remain_when_absent(self):
        result = evaluate_classification([10000], [10000])
        self.assertEqual(len(result["per_class"]), 6)
        self.assertEqual(result["per_class"][5]["support"], 0)
        self.assertEqual(result["per_class"][5]["precision"], 0.0)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertEqual(result["mae_vnd"], 0.0)

    def test_scores_match_sklearn_and_pair_totals_match(self):
        rng = np.random.default_rng(2312026)
        true = rng.choice(CLASS_ORDER, size=500)
        pred = rng.choice(CLASS_ORDER, size=500)
        result = evaluate_classification(true, pred)
        p, r, f, support = precision_recall_fscore_support(
            true, pred, labels=CLASS_ORDER, average=None, zero_division=0,
        )
        self.assertAlmostEqual(result["accuracy"], accuracy_score(true, pred))
        self.assertAlmostEqual(result["macro_f1"], f1_score(
            true, pred, labels=CLASS_ORDER, average="macro", zero_division=0,
        ))
        self.assertAlmostEqual(result["mae_vnd"], mean_absolute_error(true, pred))
        self.assertEqual(result["confusion_matrix"], confusion_matrix(
            true, pred, labels=CLASS_ORDER,
        ).tolist())
        np.testing.assert_allclose([row["precision"] for row in result["per_class"]], p)
        np.testing.assert_allclose([row["recall"] for row in result["per_class"]], r)
        np.testing.assert_allclose([row["f1"] for row in result["per_class"]], f)
        self.assertEqual([row["support"] for row in result["per_class"]], support.tolist())
        self.assertEqual(sum(row["total_gap_vnd"] for row in result["error_pairs"]),
                         result["total_gap_vnd"])
        self.assertEqual(sum(row["count"] for row in result["error_pairs"]),
                         int(np.sum(true != pred)))

    def test_large_total_uses_more_than_int32(self):
        result = evaluate_classification([10000] * 5000, [500000] * 5000)
        self.assertEqual(result["total_gap_vnd"], 2450000000)
        self.assertEqual(result["mae_vnd"], 490000)

    def test_numeric_strings(self):
        result = evaluate_classification(["10000", "500000"], ["10000", "200000"])
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["mae_vnd"], 150000)

    def test_json_schema_has_only_the_current_metrics(self):
        result = evaluate_classification(CLASS_ORDER, CLASS_ORDER)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(set(result), {
            "class_order_vnd", "n_samples", "confusion_matrix", "accuracy",
            "macro_f1", "mae_vnd", "total_gap_vnd", "per_class", "error_pairs",
        })

    def test_repetition_and_sample_order_do_not_change_scores(self):
        true = [10000, 10000, 20000, 50000, 100000, 200000, 500000]
        pred = [10000, 20000, 20000, 50000, 100000, 200000, 200000]
        baseline = evaluate_classification(true, pred)
        repeated = evaluate_classification(true * 3, pred * 3)
        reordered = evaluate_classification(true[::-1], pred[::-1])
        for key in ("accuracy", "macro_f1", "mae_vnd"):
            self.assertAlmostEqual(baseline[key], repeated[key])
            self.assertAlmostEqual(baseline[key], reordered[key])
        self.assertEqual(baseline["confusion_matrix"], reordered["confusion_matrix"])
        self.assertEqual(repeated["total_gap_vnd"], 3 * baseline["total_gap_vnd"])


if __name__ == "__main__":
    unittest.main()
