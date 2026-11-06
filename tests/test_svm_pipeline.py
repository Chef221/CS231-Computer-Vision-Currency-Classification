"""Small CPU fixtures for feature semantics, cache alignment and SVC dev selection.

Temporary data only; these are not the saved currency experiment results.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.SVM.pipeline import (CLASSES, candidates, hsv12, hog_raw, l2, extract_matrix,
                              save_features, load_features, same_manifest,
                              run_dev_grid, save_dev_result)
from src.evaluation.value_metrics import evaluate_classification


class SVMPipelineTests(unittest.TestCase):
    def test_joint_hsv_histogram_and_hog_dimensions(self):
        image = np.full((224, 224, 3), [40, 120, 200], dtype=np.uint8)
        descriptor = hsv12(image)
        self.assertEqual(descriptor.shape, (1728,))
        self.assertEqual(descriptor.dtype, np.float32)
        self.assertEqual(np.count_nonzero(descriptor), 1)
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=6)
        image[:, 80:140] = 240
        before = image.copy()
        raw = hog_raw(image)
        self.assertEqual(raw.shape, (6084,))
        self.assertTrue(np.isfinite(raw).all())
        self.assertGreater(float(np.linalg.norm(raw)), 0)
        self.assertAlmostEqual(float(np.linalg.norm(l2(raw[None])[0])), 1.0, places=6)
        np.testing.assert_array_equal(image, before)

    def test_features_cache_and_manifest_alignment(self):
        with tempfile.TemporaryDirectory(prefix="cs231_svm_test_") as directory:
            root = Path(directory)
            manifest = pd.DataFrame({"relative_path": [f"train/{v}/{v}.png" for v in CLASSES],
                                     "label_vnd": CLASSES})
            for index, relative in enumerate(manifest.relative_path):
                path = root / relative
                path.parent.mkdir(parents=True)
                image = np.full((40, 80, 3), [20 + index * 30, 80, 150], dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path), image))
            features = extract_matrix(manifest, root, "HSV12")
            self.assertEqual(features.shape, (6, 1728))
            cache = root / "cache"
            save_features(cache, manifest, manifest, features, features)
            loaded, labels, ordered = load_features(cache, "train")
            np.testing.assert_array_equal(loaded, features)
            np.testing.assert_array_equal(labels, CLASSES)
            same_manifest(manifest, ordered)
            with self.assertRaises(ValueError):
                same_manifest(manifest, ordered.iloc[::-1].reset_index(drop=True))
            np.save(cache / "y_train.npy", labels[::-1])
            with self.assertRaises(ValueError):
                load_features(cache, "train")

    def test_seven_candidates_choose_by_f1_then_vnd_mae(self):
        train = np.repeat(np.eye(6, dtype=np.float32), 2, axis=0)
        labels = np.repeat(np.asarray(CLASSES), 2)
        dev = np.eye(6, dtype=np.float32)
        truth = np.asarray(CLASSES)
        grid, predictions, params = run_dev_grid(train, labels, dev, truth)
        self.assertEqual(len(grid), 7)
        self.assertEqual(set(grid.order), set(range(7)))
        self.assertEqual(params, candidates()[int(grid.iloc[0].order)])
        scores = evaluate_classification(truth, predictions)
        self.assertEqual(scores["macro_f1"], grid.iloc[0].macro_f1)
        self.assertEqual(scores["mae_vnd"], float(np.abs(truth - predictions).sum() / len(truth)))
        self.assertTrue(set(predictions).issubset(CLASSES))
        with tempfile.TemporaryDirectory(prefix="cs231_svm_grid_") as directory:
            folder = Path(directory)
            manifest = pd.DataFrame({"relative_path": [f"dev/{v}/{v}.png" for v in CLASSES],
                                     "label_vnd": CLASSES})
            saved = save_dev_result(folder, "fixture", grid, predictions, manifest)
            self.assertEqual(json.loads((folder / "metrics_dev.json").read_text()), saved)
            self.assertEqual(pd.read_csv(folder / "predictions_dev.csv").y_true_vnd.tolist(), CLASSES)

    def test_exact_ties_keep_first_declared_candidate(self):
        train = np.ones((12, 3), dtype=np.float32)
        truth = np.repeat(np.asarray(CLASSES), 2)
        dev = np.ones((6, 3), dtype=np.float32)
        grid, predictions, params = run_dev_grid(train, truth, dev, np.asarray(CLASSES))
        self.assertEqual(grid.macro_f1.nunique(), 1)
        self.assertEqual(grid.mae_vnd.nunique(), 1)
        self.assertEqual(params, candidates()[0])
        self.assertEqual(int(grid.iloc[0].order), 0)

    def test_notebook_error_tables_accept_perfect_predictions(self):
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        root = Path(__file__).resolve().parents[1]
        scores = evaluate_classification(np.asarray(CLASSES), np.asarray(CLASSES))
        with tempfile.TemporaryDirectory(prefix="cs231_svm_notebook_") as directory:
            scope = {"scores": scores, "CLASSES": CLASSES, "np": np, "pd": pd,
                     "plt": plt, "OUTPUT": Path(directory), "display": lambda *args: None}
            for filename, marker in (("07_EVALUATE_TEST.ipynb", "ConfusionMatrixDisplay"),
                                     ("08_ERROR_ANALYSIS.ipynb", "directed = pd.DataFrame")):
                notebook = json.loads((root / "src/SVM/224" / filename).read_text())
                source = next("".join(c["source"]) for c in notebook["cells"]
                              if c["cell_type"] == "code" and marker in "".join(c["source"]))
                with patch("matplotlib.pyplot.show"):
                    exec(compile(source, filename, "exec"), scope)
            self.assertTrue(scope["error_pairs"].empty)
            self.assertTrue(scope["directed"].empty)
            self.assertEqual(list(scope["directed"].columns),
                             ["true_vnd", "predicted_vnd", "count", "gap_vnd", "total_gap_vnd"])
            self.assertEqual(len(pd.read_csv(Path(directory) / "error_pairs_cost.csv")), 0)
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
