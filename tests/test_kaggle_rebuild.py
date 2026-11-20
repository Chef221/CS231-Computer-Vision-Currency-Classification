"""Chạy thật chuỗi SVM nhỏ; kiểm tra notebook, đường dẫn và bộ code xuất ra."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
import nbformat
from src.experiments.runtime import restore_previous_output, setup
from src.CNN.model import build_model
from src.CNN.data import CANDIDATES
from tools.build_bundle import build_bundle

ROOT = Path(__file__).resolve().parents[1]


class KaggleRebuildTests(unittest.TestCase):
    def test_notebooks_valid_and_inline_cnn_matches_factory(self):
        for kind in ["CNN", "SVM"]:
            for path in (ROOT / f"src/{kind}/224").glob("*.ipynb"):
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        compile(cell.source, str(path), "exec")
                        self.assertFalse(cell.outputs)
        import tensorflow as tf

        for config in CANDIDATES:
            path = next(
                p
                for p in (ROOT / "src/CNN/224").glob("*TRAIN*.ipynb")
                if f"model = models.Sequential(name='{config['candidate_id']}')" in p.read_text()
            )
            nb = nbformat.read(path, as_version=4)
            source = next(
                c.source
                for c in nb.cells
                if c.cell_type == "code" and "model = models.Sequential" in c.source
            )
            scope = {"models": tf.keras.models, "layers": tf.keras.layers}
            exec(source, scope)
            self.assertEqual(scope["model"].to_json(), build_model(config).to_json())

    def test_restore_kaggle_output_and_protect_published_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "input"
            previous = inputs / "notebooks/user/step/CS231_CNN/results_224"
            (previous / "inputs").mkdir(parents=True)
            (previous / "inputs/experiment_config.json").write_text("{}")
            target = root / "working/results_224"
            restore_previous_output(target, "CNN", "inputs/experiment_config.json", inputs)
            self.assertTrue((target / "inputs/experiment_config.json").is_file())
            with self.assertRaises(FileNotFoundError):
                restore_previous_output(root / "other", "CNN", "missing.keras", inputs)
            data = root / "DATA"
            for split in ["train", "dev"]:
                (data / split / "10000").mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "CS231_DATA_ROOT": str(data),
                    "CS231_CNN_RESULTS_ROOT": str(ROOT / "src/CNN/results_224"),
                },
            ):
                with self.assertRaises(ValueError):
                    setup("CNN", ROOT)

    def test_svm_notebook_chain_and_exports(self):
        from src.SVM import pipeline

        with tempfile.TemporaryDirectory(prefix="cs231_svm_") as directory:
            root = Path(directory)
            images = root / "DATA"
            results = root / "results"
            for split, count in [("train", 2), ("dev", 1), ("test", 1)]:
                for i, value in enumerate(pipeline.CLASSES):
                    path = images / split / str(value)
                    path.mkdir(parents=True)
                    for k in range(count):
                        im = np.full(
                            (32, 45, 3),
                            [(i * 35 + 40) % 255, (i * 55 + 30) % 255, (i * 20 + 60) % 255],
                            dtype=np.uint8,
                        )
                        cv2.line(im, (2 + k, 4), (20, 25), (230, 230, 230), 2 + i)
                        cv2.imwrite(str(path / f"{value}_{k}.png"), im)
            env = {
                "CS231_PROJECT_ROOT": str(ROOT),
                "CS231_DATA_ROOT": str(images),
                "CS231_SVM_RESULTS_ROOT": str(results),
            }
            with (
                patch.dict(os.environ, env),
                patch.dict(pipeline.SPLIT_SIZES, {"train": 12, "dev": 6, "test": 6}),
                patch("matplotlib.pyplot.show"),
            ):
                for path in sorted((ROOT / "src/SVM/224").glob("*.ipynb")):
                    if path.name.startswith("00_"):
                        continue
                    scope = {"__name__": "__main__"}
                    for cell in nbformat.read(path, as_version=4).cells:
                        if cell.cell_type == "code":
                            exec(compile(cell.source, str(path), "exec"), scope)
                    # Each notebook has a fresh scope, just like a separate kernel.
                    scope.clear()
            self.assertEqual(
                len(pd.read_csv(results / "selection/all_35_configurations_dev.csv")), 35
            )
            self.assertTrue((results / "test_final/model.joblib").is_file())
            self.assertTrue((results / "error_analysis/SVM_TRAIN_REFERENCES_01.png").is_file())
            self.assertEqual(len(pd.read_csv(results / "error_analysis/all_errors.csv")), 0)
            for kind in ["github", "kaggle"]:
                destination, _ = build_bundle(kind, root / f"{kind}.zip")
                with ZipFile(destination) as archive:
                    names = archive.namelist()
                    self.assertIn("src/experiments/runtime.py", names)
                    self.assertFalse(
                        any(n.endswith((".keras", ".pt", ".npy", ".joblib")) for n in names)
                    )
                    self.assertFalse(any(".local_backup" in n for n in names))
                    if kind == "github":
                        self.assertIn("src/CNN/results_224/test_final/metrics_test.json", names)
                    else:
                        self.assertFalse(any("/results_224/" in n for n in names))


if __name__ == "__main__":
    unittest.main()
