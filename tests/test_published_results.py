"""Đối chiếu số liệu công bố và khả năng dùng checkpoint thật của nhóm."""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.evaluation.value_metrics import evaluate_classification

ROOT = Path(__file__).resolve().parents[1]


class PublishedResultsTests(unittest.TestCase):
    def test_metrics_match_every_saved_test_prediction(self):
        expected_correct = {"KNN": 1187, "SVM": 1362, "CNN": 1512}
        for kind, count in expected_correct.items():
            root = ROOT / f"src/{kind}/results_224"
            frame = pd.read_csv(root / "test_final/predictions_test.csv")
            scores = evaluate_classification(frame.y_true_vnd, frame.y_pred_vnd)
            self.assertEqual(
                scores, json.loads((root / "test_final/metrics_test.json").read_text())
            )
            self.assertEqual(len(frame), 1553)
            self.assertEqual(int((frame.y_true_vnd == frame.y_pred_vnd).sum()), count)
            self.assertEqual(frame.relative_path.nunique(), 1553)

    def test_same_splits_and_paired_comparison(self):
        knn = ROOT / "src/KNN/results_224"
        for kind in ["SVM", "CNN"]:
            for split in ["train", "dev"]:
                left = pd.read_csv(knn / f"inputs/{split}_manifest.csv")
                right = pd.read_csv(ROOT / f"src/{kind}/results_224/inputs/{split}_manifest.csv")
                pd.testing.assert_frame_equal(
                    left[["relative_path", "label_vnd"]], right[["relative_path", "label_vnd"]]
                )
        svm = pd.read_csv(ROOT / "src/SVM/results_224/test_final/predictions_test.csv")
        cnn = pd.read_csv(ROOT / "src/CNN/results_224/test_final/predictions_test.csv")
        paired = svm.merge(
            cnn, on="relative_path", suffixes=("_svm", "_cnn"), validate="one_to_one"
        )
        self.assertTrue((paired.y_true_vnd_svm == paired.y_true_vnd_cnn).all())
        a = paired.y_true_vnd_svm == paired.y_pred_vnd_svm
        b = paired.y_true_vnd_cnn == paired.y_pred_vnd_cnn
        self.assertEqual(
            [int((a & b).sum()), int((~a & b).sum()), int((a & ~b).sum()), int((~a & ~b).sum())],
            [1343, 169, 19, 22],
        )

    def test_actual_checkpoint_sample_if_available(self):
        checkpoint = ROOT / "src/CNN/results_224/candidates/C04_DEEP_AUG/best.keras"
        if not checkpoint.is_file():
            self.skipTest("Model lớn được Git bỏ qua; kiểm tra khi có Output của nhóm.")
        from src.CNN.model import inference_model
        from src.CNN.data import image_array, CLASSES
        import tensorflow as tf

        frame = pd.read_csv(ROOT / "src/CNN/results_224/test_final/predictions_test.csv")
        sample = pd.concat(
            [
                frame.groupby("y_true_vnd").head(1),
                frame.loc[frame.y_true_vnd != frame.y_pred_vnd].head(6),
            ]
        ).drop_duplicates("relative_path")
        if not all((ROOT / "DATA" / p).is_file() for p in sample.relative_path):
            self.skipTest("Cần ảnh DATA để đối chiếu checkpoint thật.")
        model = inference_model(tf.keras.models.load_model(checkpoint, compile=False))
        images = np.stack([image_array(ROOT / "DATA" / p) for p in sample.relative_path])
        probabilities = model(images, training=False).numpy()
        self.assertEqual(model.count_params(), 1208806)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_array_equal(
            np.array(CLASSES)[probabilities.argmax(axis=1)], sample.y_pred_vnd
        )
