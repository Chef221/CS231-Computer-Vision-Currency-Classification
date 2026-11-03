import json
import unittest

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.evaluation.value_metrics import CLASS_ORDER, evaluate_classification


class EvaluateClassificationTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = evaluate_classification(CLASS_ORDER, CLASS_ORDER)
        self.assertEqual(result["confusion_matrix"], np.eye(6, dtype=int).tolist())
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["weighted_f1"], 1.0)
        self.assertEqual(result["macro_value_f1"], 1.0)

    def test_result_is_json_serializable(self):
        result = evaluate_classification(CLASS_ORDER, CLASS_ORDER)
        self.assertIsInstance(json.loads(json.dumps(result)), dict)

    def test_manual_value_gap_calculation_and_label_order(self):
        y_true = [10_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
        y_pred = [10_000, 20_000, 20_000, 50_000, 100_000, 200_000, 200_000]
        result = evaluate_classification(y_true, y_pred)
        self.assertEqual(result["class_order_vnd"], list(CLASS_ORDER))
        self.assertEqual(result["confusion_matrix"][0][1], 1)
        self.assertEqual(result["confusion_matrix"][5][4], 1)
        self.assertEqual(result["per_class"][1]["incoming_gap_vnd"], 10_000)
        self.assertEqual(result["per_class"][5]["outgoing_gap_vnd"], 300_000)
        self.assertAlmostEqual(result["per_class"][0]["value_precision"], 1.0)
        self.assertAlmostEqual(result["per_class"][0]["value_recall"], 0.5)
        self.assertAlmostEqual(result["per_class"][0]["value_f1"], 2 / 3)
        self.assertAlmostEqual(result["per_class"][1]["value_precision"], 2 / 3)
        self.assertAlmostEqual(result["per_class"][1]["value_recall"], 1.0)
        self.assertAlmostEqual(result["per_class"][1]["value_f1"], 4 / 5)
        self.assertEqual(result["per_class"][4]["incoming_gap_vnd"], 300_000)
        self.assertAlmostEqual(result["per_class"][4]["value_f1"], 4 / 7)
        self.assertAlmostEqual(result["macro_value_f1"], 212 / 315)

    def test_absent_classes_remain_in_macro_average(self):
        result = evaluate_classification([10_000], [10_000])
        self.assertEqual(len(result["per_class"]), 6)
        self.assertEqual(result["per_class"][5]["support"], 0)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertAlmostEqual(result["macro_value_f1"], 1 / 6)

    def test_numeric_strings_are_normalized_to_denominations(self):
        result = evaluate_classification(["10000", "500000"], ["10000", "500000"])
        self.assertEqual(result["class_order_vnd"], list(CLASS_ORDER))
        self.assertEqual(result["n_samples"], 2)

    def test_documented_three_image_example(self):
        result = evaluate_classification(
            [10_000, 10_000, 20_000], [10_000, 20_000, 20_000]
        )
        self.assertAlmostEqual(result["per_class"][0]["value_f1"], 2 / 3)
        self.assertAlmostEqual(result["per_class"][1]["value_f1"], 0.8)
        self.assertAlmostEqual(result["macro_value_f1"], 11 / 45)

    def test_traditional_f1_matches_sklearn_fixed_class_order(self):
        y_true = [10_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
        y_pred = [10_000, 20_000, 20_000, 50_000, 100_000, 200_000, 200_000]
        result = evaluate_classification(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_ORDER, zero_division=0
        )
        self.assertAlmostEqual(result["macro_f1"], float(f1.mean()))
        self.assertAlmostEqual(result["weighted_f1"], float(np.average(f1, weights=support)))
        self.assertEqual(
            [row["support"] for row in result["per_class"]], support.tolist()
        )

    def test_all_wrong_predictions_have_zero_scores(self):
        result = evaluate_classification(CLASS_ORDER, list(CLASS_ORDER[1:]) + [CLASS_ORDER[0]])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["macro_f1"], 0.0)
        self.assertEqual(result["weighted_f1"], 0.0)
        self.assertEqual(result["macro_value_f1"], 0.0)

    def test_repetition_and_input_permutation_do_not_change_scores(self):
        y_true = [10_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
        y_pred = [10_000, 20_000, 20_000, 50_000, 100_000, 200_000, 200_000]
        baseline = evaluate_classification(y_true, y_pred)
        repeated = evaluate_classification(y_true * 3, y_pred * 3)
        permutation = [6, 2, 4, 0, 5, 1, 3]
        reordered = evaluate_classification(
            [y_true[i] for i in permutation], [y_pred[i] for i in permutation]
        )
        for key in ("accuracy", "macro_f1", "weighted_f1", "macro_value_f1"):
            self.assertAlmostEqual(baseline[key], repeated[key])
            self.assertAlmostEqual(baseline[key], reordered[key])
        self.assertEqual(baseline["confusion_matrix"], reordered["confusion_matrix"])

if __name__ == "__main__":
    unittest.main()
