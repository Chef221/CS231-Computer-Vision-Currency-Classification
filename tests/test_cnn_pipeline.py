"""CPU integration fixtures, not official CNN experiment results.

Run: python -m unittest discover -s tests -p test_cnn_pipeline.py -v
Requires src/CNN/requirements.txt. All generated data/checkpoints stay in /tmp.
"""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

from src.CNN import data, pipeline, analysis
from src.CNN.model import build_model, architecture_table
from src.CNN.runtime import restore_previous_output, mark_stage

ROOT = Path(__file__).resolve().parents[1]


class CNNIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="cs231_cnn_test_")
        cls.root = Path(cls.temporary.name)
        cls.images = cls.root / "DATA"
        cls.results = cls.root / "results_224"
        cls.protocol = copy.deepcopy(data.protocol(ROOT))
        cls.protocol["split_sizes"] = {"train": 12, "dev": 6, "test": 6}
        cls.protocol["training"].update(max_epochs=2, batch_size=4, num_workers=0, amp_cuda=False)
        for index, config in enumerate(cls.protocol["candidates"]):
            config["channels"] = [4, 8, 16] if index == 3 else [4, 8]
        for split, count in (("train", 2), ("dev", 1), ("test", 1)):
            for index, value in enumerate(data.CLASSES):
                folder = cls.images / split / str(value)
                folder.mkdir(parents=True)
                for number in range(count):
                    image = np.zeros((36, 48, 3), dtype=np.uint8)
                    image[:] = [(index * 37 + 40) % 255, (index * 63 + 50) % 255, (index * 97 + 60) % 255]
                    cv2.circle(image, (12 + number * 5, 18), 7 + index, (230, 230, 230), -1)
                    cv2.imwrite(str(folder / f"{value}_{number:06d}.png"), image)
        cls.protocol_patch = patch("src.CNN.data.protocol", return_value=cls.protocol)
        cls.protocol_patch.start()
        pipeline.prepare_inputs(cls.images, cls.results, ROOT)
        original_save = pipeline._atomic_torch_save
        stopped = False

        def interrupt_after_saved_epoch(path, content):
            nonlocal stopped
            original_save(path, content)
            if Path(path).name == "last.pt" and not stopped:
                stopped = True
                raise InterruptedError("simulated interruption after epoch checkpoint")

        try:
            with patch("src.CNN.pipeline._atomic_torch_save", side_effect=interrupt_after_saved_epoch):
                pipeline.train_candidate("C01_RGB", cls.images, cls.results, ROOT)
        except InterruptedError:
            pass
        else:
            raise AssertionError("Resume fixture did not interrupt")
        for config in cls.protocol["candidates"]:
            pipeline.train_candidate(config["candidate_id"], cls.images, cls.results, ROOT)
        cls.table, cls.lock = pipeline.select_and_lock(cls.results, ROOT)
        cls.metrics, cls.predictions = pipeline.evaluate_test(cls.images, cls.results, ROOT)
        spec = importlib.util.spec_from_file_location("cnn_report", ROOT / "docs/CNN_REPORT/build_report_224.py")
        cls.report_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.report_module)

    @classmethod
    def tearDownClass(cls):
        cls.protocol_patch.stop()
        cls.temporary.cleanup()

    def test_vnd_mapping_and_eval_augmentation(self):
        manifest = pd.read_csv(self.results / "inputs/dev_manifest.csv")
        dataset = data.BanknoteDataset(manifest, self.images, self.protocol["candidates"][2], training=False)
        with patch("src.CNN.data.augment_rgb", side_effect=AssertionError("eval augmentation")):
            tensor, index = dataset[2]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(data.CLASSES[index], 50000)
        self.assertGreaterEqual(float(tensor.min()), 0)
        self.assertLessEqual(float(tensor.max()), 1)
        with self.assertRaises(ValueError):
            data.BanknoteDataset(manifest, self.images, self.protocol["candidates"][2], training=True)
        actual = self.predictions.y_true_vnd.to_numpy()
        predicted = self.predictions.y_pred_vnd.to_numpy()
        self.assertEqual(self.metrics["mae_vnd"], float(np.abs(actual - predicted).sum() / len(actual)))
        self.assertEqual(set(actual), set(data.CLASSES))

    def test_resume_completion_and_selection_rules(self):
        history = pd.read_csv(self.results / "candidates/C01_RGB/history.csv", float_precision="round_trip")
        self.assertEqual(history.epoch.tolist(), [1, 2])
        self.assertEqual(len(self.table), 4)
        best = self.table.sort_values(["macro_f1", "mae_vnd", "candidate_order", "best_epoch"],
                                     ascending=[False, True, True, True]).iloc[0]
        self.assertEqual(self.lock["candidate"]["candidate_id"], best.candidate_id)
        checkpoint_hash = data.sha256(self.results / "candidates/C01_RGB/best.pt")
        with patch("src.CNN.pipeline.evaluate_loader", side_effect=AssertionError("completed run retrained")):
            pipeline.train_candidate("C01_RGB", self.images, self.results, ROOT)
        self.assertEqual(checkpoint_hash, data.sha256(self.results / "candidates/C01_RGB/best.pt"))
        for config in self.protocol["candidates"]:
            folder = self.results / "candidates" / config["candidate_id"]
            hist = pd.read_csv(folder / "history.csv", float_precision="round_trip")
            selected = hist.sort_values(["dev_macro_f1", "dev_mae_vnd", "epoch"],
                                       ascending=[False, True, True]).iloc[0]
            self.assertEqual(json.loads((folder / "summary.json").read_text())["best_epoch"], int(selected.epoch))

    def test_locked_checkpoint_tamper_is_rejected(self):
        path = self.results / self.lock["checkpoint"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"tampered")
            with self.assertRaises(ValueError):
                pipeline.verify_lock(self.results, ROOT)
        finally:
            path.write_bytes(original)
        pipeline.verify_lock(self.results, ROOT)

    def test_test_result_reuses_frozen_weights(self):
        with patch("src.CNN.pipeline.load_locked_model", side_effect=AssertionError("second test inference")):
            metrics, predictions = pipeline.evaluate_test(self.images, self.results, ROOT)
        self.assertEqual(metrics, self.metrics)
        self.assertEqual(len(predictions), 6)
        used = self.results / "test_final/locked_cnn_config_used.json"
        self.assertEqual(used.read_bytes(), (self.results / "selection/locked_cnn_config.json").read_bytes())
        self.assertFalse(json.loads((self.results / "test_final/run_info.json").read_text())["refit"])

    def test_learned_features_gradcam_and_error_edge_case(self):
        coords, _ = analysis.inspect_features(self.images, self.results, ROOT)
        self.assertEqual(len(coords), 6)
        model, lock, device = pipeline.load_locked_model(self.results, ROOT)
        image = data.image_tensor(np.full((224, 224, 3), 100, dtype=np.uint8), lock["candidate"]["color"])[None].to(device)
        before = len(model.blocks[-1][5]._forward_hooks)
        cam, zero, probabilities = analysis.gradcam(model, image, 0)
        self.assertEqual(cam.shape, (224, 224))
        self.assertTrue(np.isfinite(cam).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        self.assertEqual(before, len(model.blocks[-1][5]._forward_hooks))
        cases, _ = analysis.analyze_errors(self.images, self.results, ROOT)
        self.assertEqual(cases.relative_path.nunique(), len(cases))
        correct_only = self.predictions.copy()
        correct_only["y_pred_vnd"] = correct_only.y_true_vnd
        correct_only["gap_vnd"] = 0
        self.assertEqual(len(analysis.select_cases(correct_only)), len(correct_only))

    def _fixture_references(self):
        folder = self.root / "reference_fixtures"
        provenance = {"baselines": {}, "sha256": {}}
        for model in ("KNN", "SVM"):
            destination = folder / model
            destination.mkdir(parents=True, exist_ok=True)
            pred = self.predictions.copy()
            pred["y_pred_vnd"] = pred.y_true_vnd
            pred["correct"] = True
            pred["gap_vnd"] = 0
            pred.to_csv(destination / "predictions_test.csv", index=False)
            data.save_json(destination / "metrics_test.json", pipeline.score_predictions(pred))
            for split in ("train", "dev", "test"):
                source = self.results / ("test_final" if split == "test" else "inputs") / f"{split}_manifest.csv"
                (destination / source.name).write_bytes(source.read_bytes())
            provenance["baselines"][model] = {"folder": model, "source_commit": "synthetic fixture, not experiment"}
            for path in destination.iterdir():
                provenance["sha256"][path.relative_to(folder).as_posix()] = data.sha256(path)
        data.save_json(folder / "provenance.json", provenance)
        return folder

    def test_paired_comparison_and_report_from_evidence(self):
        references = self._fixture_references()
        comparison, paired = analysis.compare_baselines(self.results, references)
        self.assertEqual(set(comparison.model), {"CNN", "KNN", "SVM"})
        for row in paired.itertuples():
            self.assertEqual(row.both_correct + row.only_cnn_correct + row.only_baseline_correct + row.both_wrong, 6)
        report = self.report_module.build_report(self.results, self.root / "report", ROOT, references)
        text = report.read_text()
        self.assertNotIn("{{", text)
        self.assertIn("báo cáo kết quả đã đo", text)
        self.assertIn(self.report_module.number(self.metrics["mae_vnd"], 2), text)
        draft = self.report_module.build_report(self.root / "not_run", self.root / "draft", ROOT, references)
        self.assertIn("chưa có kết quả CNN", draft.read_text())

    def test_runtime_requires_exact_previous_stage(self):
        input_root = self.root / "kaggle_input"
        source = input_root / "notebooks/user/cnn01/CS231_CNN/results_224"
        (source / "inputs").mkdir(parents=True, exist_ok=True)
        data.save_json(source / "inputs/experiment_config.json", {"test_fixture": True})
        mark_stage(source, "01")
        target = self.root / "restored"
        restore_previous_output(target, "01", "inputs/experiment_config.json", input_root)
        self.assertTrue((target / "inputs/experiment_config.json").is_file())
        with self.assertRaises(ValueError):
            restore_previous_output(target, "02", "inputs/experiment_config.json", input_root)


if __name__ == "__main__":
    unittest.main()
