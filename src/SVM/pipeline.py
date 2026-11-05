"""Shared, deterministic feature extraction and dev evaluation for SVM notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from src.evaluation.value_metrics import CLASS_ORDER, evaluate_classification
from src.preprocessing import read_rgb224

CLASSES = list(CLASS_ORDER)
SPLIT_SIZES = {"train": 12494, "dev": 1564, "test": 1553}
FEATURE_ORDER = ["HSV12", "HOG", "HOG_PCA", "HSV12_HOG", "HSV12_HOG_PCA"]
FEATURE_FOLDERS = {
    "HSV12": "hsv12", "HOG": "hog", "HOG_PCA": "hog_pca",
    "HSV12_HOG": "hsv12_hog", "HSV12_HOG_PCA": "hsv12_hog_pca",
}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_for_split(data_root: Path, split: str) -> pd.DataFrame:
    rows = []
    for label in CLASSES:
        folder = data_root / split / str(label)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.glob("*.png")):
            rows.append({"relative_path": path.relative_to(data_root).as_posix(), "label_vnd": label})
    manifest = pd.DataFrame(rows, columns=["relative_path", "label_vnd"])
    if len(manifest) != SPLIT_SIZES[split]:
        raise ValueError(f"{split}: expected {SPLIT_SIZES[split]} images, found {len(manifest)}")
    return manifest


def hsv12(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 12, 12], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).reshape(-1).astype(np.float32)


def hog_raw(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vector = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), block_norm="L2-Hys",
                 transform_sqrt=False, feature_vector=True, channel_axis=None)
    return np.asarray(vector, dtype=np.float32)


def extract_matrix(manifest: pd.DataFrame, data_root: Path, feature: str) -> np.ndarray:
    """Extract either HSV12 or raw HOG in manifest order, one image at a time."""
    if feature not in {"HSV12", "HOG_RAW"}:
        raise ValueError(feature)
    dimensions = 1728 if feature == "HSV12" else 6084
    matrix = np.empty((len(manifest), dimensions), dtype=np.float32)
    extractor = hsv12 if feature == "HSV12" else hog_raw
    for i, relative_path in enumerate(manifest["relative_path"]):
        matrix[i] = extractor(read_rgb224(data_root / relative_path))
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    return matrix


def l2(matrix: np.ndarray) -> np.ndarray:
    return normalize(matrix, norm="l2").astype(np.float32)


def save_features(folder: Path, train: pd.DataFrame, dev: pd.DataFrame,
                  x_train: np.ndarray, x_dev: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if len(x_train) != len(train) or len(x_dev) != len(dev):
        raise ValueError("Feature rows do not match manifests")
    for split, manifest, matrix in [("train", train, x_train), ("dev", dev, x_dev)]:
        np.save(folder / f"X_{split}.npy", matrix)
        np.save(folder / f"y_{split}.npy", manifest.label_vnd.to_numpy(dtype=np.int64))
        manifest.to_csv(folder / f"{split}_manifest.csv", index=False)


def load_features(folder: Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(folder / f"{split}_manifest.csv")
    matrix = np.load(folder / f"X_{split}.npy", mmap_mode="r")
    labels = np.load(folder / f"y_{split}.npy")
    if len(matrix) != len(manifest) or not np.array_equal(labels, manifest.label_vnd):
        raise ValueError(f"Feature/label/manifest mismatch: {folder}, {split}")
    return matrix, labels, manifest


def same_manifest(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if not left[["relative_path", "label_vnd"]].equals(right[["relative_path", "label_vnd"]]):
        raise ValueError("Feature branches use different image or label order")


def candidates() -> list[dict]:
    """Seven predeclared SVC configurations, ordered for deterministic ties."""
    configs = [{"kernel": "linear", "C": c, "gamma": "scale"} for c in (0.1, 1.0, 10.0)]
    configs += [{"kernel": "rbf", "C": c, "gamma": gamma}
                for c in (1.0, 10.0) for gamma in ("scale", 10.0)]
    return configs


def run_dev_grid(x_train: np.ndarray, y_train: np.ndarray,
                 x_dev: np.ndarray, y_dev: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, dict]:
    records = []
    best_key = None
    best_pred = None
    best_params = None
    for order, params in enumerate(candidates()):
        model = SVC(**params, cache_size=512, decision_function_shape="ovr", break_ties=True)
        model.fit(x_train, y_train)
        pred = model.predict(x_dev)
        scores = evaluate_classification(y_dev, pred)
        record = {"order": order, **params, "accuracy": scores["accuracy"],
                  "macro_f1": scores["macro_f1"], "mae_vnd": scores["mae_vnd"],
                  "n_support_vectors": int(model.n_support_.sum())}
        records.append(record)
        key = (-scores["macro_f1"], scores["mae_vnd"], order)
        if best_key is None or key < best_key:
            best_key, best_pred, best_params = key, pred.copy(), params.copy()
    grid = pd.DataFrame(records).sort_values(
        ["macro_f1", "mae_vnd", "order"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return grid, best_pred, best_params


def save_dev_result(folder: Path, feature_id: str, grid: pd.DataFrame,
                    pred: np.ndarray, manifest: pd.DataFrame) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    scores = evaluate_classification(manifest.label_vnd.to_numpy(), pred)
    grid.to_csv(folder / "svm_grid_dev.csv", index=False)
    best = grid.iloc[0].to_dict()
    pd.DataFrame([{"feature_id": feature_id, **best}]).to_csv(folder / "svm_best_dev.csv", index=False)
    predictions = manifest[["relative_path"]].copy()
    predictions["y_true_vnd"] = manifest.label_vnd.to_numpy()
    predictions["y_pred_vnd"] = pred
    predictions.to_csv(folder / "predictions_dev.csv", index=False)
    pd.DataFrame(scores["per_class"]).to_csv(folder / "per_class_dev.csv", index=False)
    pd.DataFrame(scores["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        folder / "confusion_matrix_dev.csv")
    save_json(folder / "metrics_dev.json", scores)
    return scores
