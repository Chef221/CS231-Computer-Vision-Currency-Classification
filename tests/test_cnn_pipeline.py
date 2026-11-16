"""Real TensorFlow CPU integration on tiny fixtures, never official experiment data."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.CNN import data, pipeline, analysis
from src.CNN.model import (
    build_model,
    TorchBatchNormalization,
    conv_initializer,
    compile_model,
    cosine_learning_rate,
    inference_model,
)
from src.experiments.figures import sha256

ROOT = Path(__file__).resolve().parents[1]


class TensorFlowPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="cs231_tf_fixture_")
        cls.root = Path(cls.temp.name)
        cls.images = cls.root / "DATA"
        cls.results = cls.root / "results_tf_224"
        cls.configs = copy.deepcopy(data.CANDIDATES[:2])
        for config in cls.configs:
            config["channels"] = [4, 8]
        for split, n in [("train", 2), ("dev", 1), ("test", 1)]:
            for i, v in enumerate(data.CLASSES):
                folder = cls.images / split / str(v)
                folder.mkdir(parents=True)
                for k in range(n):
                    image = np.full(
                        (30, 50, 3),
                        [(30 + i * 40) % 256, (70 + i * 20) % 256, (120 + i * 10) % 256],
                        dtype="uint8",
                    )
                    cv2.circle(image, (15 + k, 15), 5 + i, (200, 200, 200), -1)
                    cv2.imwrite(str(folder / f"{v}_{k:06d}.png"), image)
        data.prepare(cls.images, cls.results, candidates=cls.configs, epochs=2, expected=False)
        for config in cls.configs:
            pipeline.train_candidate(
                build_model(config), config["candidate_id"], cls.images, cls.results
            )
        cls.comparison, cls.lock = pipeline.select_and_lock(cls.results)
        cls.scores, cls.predictions = pipeline.evaluate_test(cls.images, cls.results)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_streaming_mapping_rgb_gray_and_inference(self):
        manifest = pd.read_csv(self.results / "inputs/dev_manifest.csv")
        for config in self.configs:
            images, labels = next(
                iter(data.make_dataset(manifest, self.images, config, batch_size=6))
            )
            self.assertEqual(images.shape, (6, 224, 224, 1 if config["color"] == "gray" else 3))
            np.testing.assert_array_equal(labels, np.arange(6))
            expected = data.image_array(
                self.images / manifest.iloc[0].relative_path, config["color"]
            )
            np.testing.assert_array_equal(images[0], expected)
        model = build_model(data.CANDIDATES[2])
        x = np.ones((1, 224, 224, 3), dtype="float32") * 100
        np.testing.assert_allclose(model(x, training=False), model(x, training=False))

    def test_best_epoch_and_portable_checkpoint(self):
        history = pd.read_csv(self.results / "candidates/C01_RGB/history.csv")
        self.assertEqual(history.epoch.tolist(), [1, 2])
        np.testing.assert_allclose(history.learning_rate, [0.001, 0.000505], rtol=1e-6)
        best = self.comparison.iloc[0]
        self.assertEqual(self.lock["candidate"], best.candidate_id)
        model, lock = pipeline.load_locked_model(self.results)
        self.assertEqual(model.output_shape, (None, 6))
        self.assertIn(int(lock["best_epoch"]), [1, 2])
        self.assertEqual(pipeline.metric_key({"macro_f1": 0.8, "mae_vnd": 1}), (-0.8, 1.0))

    def test_vnd_metrics_and_test_rerun(self):
        pred = self.predictions
        self.assertEqual(
            self.scores["mae_vnd"], float(np.abs(pred.y_true_vnd - pred.y_pred_vnd).mean())
        )
        scores, frame = pipeline.evaluate_test(self.images, self.results)
        self.assertEqual(scores, self.scores)
        self.assertEqual(len(frame), 6)

    def test_gradcam_feature_maps_and_error_figures(self):
        model, lock = pipeline.load_locked_model(self.results)
        image = data.image_array(
            self.images / self.predictions.iloc[0].relative_path, lock["config"]["color"]
        )
        heat, check = analysis.gradcam(model, image, 0)
        self.assertEqual(heat.shape, (224, 224))
        self.assertTrue(check["finite"])
        self.assertGreaterEqual(heat.min(), 0)
        self.assertLessEqual(heat.max(), 1.00001)
        coordinates, channels = analysis.inspect_features(self.images, self.results)
        self.assertEqual(len(coordinates), 6)
        self.assertGreater(len(channels), 0)
        cases, pairs = analysis.analyze_errors(self.images, self.results)
        self.assertEqual(cases.case_id.nunique(), len(cases))
        self.assertTrue((self.results / "feature_analysis/CNN_FEATURE_MAPS.png").is_file())
        self.assertTrue((self.results / "error_analysis/CNN_GRADCAM_01.png").is_file())


class ZProtocolParityTests(unittest.TestCase):
    def setUp(self):
        data.seed_everything(42)

    def test_batchnorm_biased_forward_unbiased_running_statistics(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
        layer = TorchBatchNormalization()
        y = layer(x, training=True).numpy()
        mean = x.mean(axis=(0, 1, 2))
        var = x.var(axis=(0, 1, 2))
        np.testing.assert_allclose(y, (x - mean) / np.sqrt(var + 1e-5), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(layer.moving_mean, 0.1 * mean)
        np.testing.assert_allclose(layer.moving_variance, 0.9 + 0.1 * var * 8 / 7)
        y_eval = layer(x, training=False)
        np.testing.assert_allclose(
            y_eval, (x - 0.1 * mean) / np.sqrt(0.9 + 0.1 * var * 8 / 7 + 1e-5), rtol=1e-6
        )
        layer(x + 3, training=True)
        np.testing.assert_allclose(
            layer.moving_mean, 0.9 * 0.1 * mean + 0.1 * (mean + 3), rtol=1e-6
        )

    def test_original_architecture_and_initialization(self):
        for config, expected in [(data.CANDIDATES[0], 304742), (data.CANDIDATES[3], 1206886)]:
            model = build_model(config)
            self.assertEqual(sum(np.prod(w.shape) for w in model.trainable_weights), expected)
            self.assertEqual(model.output_shape, (None, 6))
            self.assertEqual(
                len([l for l in model.layers if isinstance(l, TorchBatchNormalization)]),
                2 * len(config["channels"]),
            )
        self.assertEqual(conv_initializer().get_config()["mode"], "fan_out")
        self.assertEqual(conv_initializer().get_config()["distribution"], "untruncated_normal")

    def test_adamw_and_cosine_match_original_epoch_schedule(self):
        settings = {
            "learning_rate": 0.001,
            "eta_min": 0.00001,
            "epochs": 25,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "gradient_clip_norm": 5.0,
        }
        model = compile_model(build_model({**data.CANDIDATES[0], "channels": [2]}), settings)
        optimizer = model.optimizer.get_config()
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.AdamW)
        for key in ("weight_decay", "epsilon", "global_clipnorm"):
            self.assertEqual(
                optimizer[key], settings[{"global_clipnorm": "gradient_clip_norm"}.get(key, key)]
            )
        self.assertEqual((optimizer["beta_1"], optimizer["beta_2"]), (0.9, 0.999))
        self.assertEqual(cosine_learning_rate(0, settings), 0.001)
        self.assertAlmostEqual(cosine_learning_rate(24, settings), 0.0000139032228493, places=14)
        self.assertEqual(cosine_learning_rate(25, settings), 0.00001)

    def test_mixed_precision_checkpoint_and_float32_evaluation(self):
        old = tf.keras.mixed_precision.global_policy().name
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            config = {**data.CANDIDATES[0], "channels": [2]}
            model = build_model(config)
            settings = {
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "gradient_clip_norm": 5.0,
            }
            compile_model(model, settings)
            x = np.full((2, 224, 224, 3), 100, dtype=np.float32)
            y = np.array([0, 1])
            self.assertEqual(model.get_layer("rescale").compute_dtype, "float32")
            self.assertEqual(model.get_layer("conv1_1").compute_dtype, "float16")
            self.assertTrue(np.isfinite(model.train_on_batch(x, y)).all())
            evaluated = inference_model(model)
            self.assertTrue(all(l.compute_dtype == "float32" for l in evaluated.layers))
            for before, after in zip(model.get_weights(), evaluated.get_weights()):
                np.testing.assert_array_equal(before, after)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "mixed.keras"
                model.save(path)
                loaded = tf.keras.models.load_model(path)
                np.testing.assert_allclose(inference_model(loaded)(x), evaluated(x), rtol=1e-6)
                self.assertEqual(int(loaded.optimizer.iterations.numpy()), 1)
                loaded.optimizer.learning_rate.assign(0.0005)
                self.assertTrue(np.isfinite(loaded.train_on_batch(x, y)).all())
                self.assertEqual(int(loaded.optimizer.iterations.numpy()), 2)
            heat, check = analysis.gradcam(evaluated, x[0], 0)
            self.assertTrue(check["finite"])
            self.assertEqual(heat.shape, (224, 224))
        finally:
            tf.keras.mixed_precision.set_global_policy(old)

    def test_full_dataset_shuffle_and_train_only_reproducible_augmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for split in ["train", "dev"]:
                for index, v in enumerate(data.CLASSES):
                    path = root / split / str(v) / "example.png"
                    path.parent.mkdir(parents=True)
                    cv2.imwrite(str(path), np.full((224, 224, 3), 30 * index, dtype=np.uint8))
                    rows.append(
                        {"relative_path": path.relative_to(root).as_posix(), "label_vnd": v}
                    )
            frame = pd.DataFrame(rows)
            train = frame.iloc[:6]
            dev = frame.iloc[6:]
            config = data.CANDIDATES[2]

            def epoch(n):
                return next(iter(data.make_dataset(train, root, config, 6, True, epoch=n)))

            first, labels = epoch(1)
            same, other_labels = epoch(1)
            second, _ = epoch(2)
            np.testing.assert_array_equal(first, same)
            np.testing.assert_array_equal(labels, other_labels)
            np.testing.assert_array_equal(labels, np.random.default_rng(43).permutation(6))
            self.assertFalse(np.array_equal(first, second))
            actual, _ = next(iter(data.make_dataset(dev, root, config, 6)))
            np.testing.assert_array_equal(
                actual[0], data.image_array(root / dev.iloc[0].relative_path)
            )
            with self.assertRaises(ValueError):
                data.make_dataset(dev, root, config, 6, True)

    def test_augmentation_matches_original_pytorch_function_fixture(self):
        # Golden pixels generated by augment_rgb in the original Git HEAD code,
        # with an isolated random.Random(2026), using the same OpenCV backend.
        import random
        import hashlib

        image = (np.arange(224 * 224 * 3).reshape(224, 224, 3) % 256).astype(np.uint8)
        actual = data.augment_rgb(image, random.Random(2026))
        self.assertEqual(
            hashlib.sha256(actual.tobytes()).hexdigest(),
            "997690122b5fcabbaccd93063472c4dc12dfa87d70ac74e8db2dc1884a62d6d8",
        )


if __name__ == "__main__":
    unittest.main()
