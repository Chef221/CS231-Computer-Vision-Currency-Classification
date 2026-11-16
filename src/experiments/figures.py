"""Vẽ ảnh đầu vào, bảng lỗi và ảnh đối chiếu từ dữ liệu thực nghiệm."""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.exposure import rescale_intensity
from sklearn.metrics import ConfusionMatrixDisplay
from src.preprocessing import read_rgb224
from src.evaluation.value_metrics import CLASS_ORDER, evaluate_classification
from src.experiments.runtime import save_json
import hashlib


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


ERROR_COLUMNS = ["true_vnd", "predicted_vnd", "count", "gap_vnd", "total_gap_vnd"]


def finish(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def input_figures(data_root, manifest, folder, prefix):
    """Two fixed train examples, preferring the same two classes as the KNN report."""
    samples = []
    for label in (10000, 500000):
        pool = manifest.loc[manifest.label_vnd == label]
        if len(pool):
            samples.append(pool.iloc[0])
    if not samples:
        raise ValueError("Không có ảnh minh họa train")
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 3 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        raw = cv2.cvtColor(
            cv2.imread(str(Path(data_root) / sample.relative_path)), cv2.COLOR_BGR2RGB
        )
        for ax, im, title in zip(
            axes[row],
            [raw, read_rgb224(Path(data_root) / sample.relative_path)],
            [f"Original {raw.shape[1]}×{raw.shape[0]}", "RGB 224×224"],
        ):
            ax.imshow(im)
            ax.set_title(f"{sample.label_vnd:,} VND | {title}")
            ax.axis("off")
    finish(fig, folder, f"{prefix}_1_RESIZE")
    fig, axes = plt.subplots(len(samples), 4, figsize=(13, 3 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        image = read_rgb224(Path(data_root) / sample.relative_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        axes[row, 0].imshow(image)
        for channel, (cmap, vmax) in enumerate([("hsv", 179), ("gray", 255), ("gray", 255)], 1):
            im = axes[row, channel].imshow(hsv[:, :, channel - 1], cmap=cmap, vmin=0, vmax=vmax)
            fig.colorbar(im, ax=axes[row, channel], fraction=0.046)
        for ax, title in zip(
            axes[row],
            [f"RGB {sample.label_vnd:,} VND", "Hue [0,179]", "Saturation [0,255]", "Value [0,255]"],
        ):
            ax.set_title(title)
            ax.axis("off")
    finish(fig, folder, f"{prefix}_HSV_CHANNELS")
    fig, axes = plt.subplots(len(samples), 3, figsize=(11, 3 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        rgb = read_rgb224(Path(data_root) / sample.relative_path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, rendered = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=True,
            transform_sqrt=False,
        )
        for ax, image, title in zip(
            axes[row],
            [rgb, gray, rescale_intensity(rendered)],
            [f"RGB {sample.label_vnd:,} VND", "Gray", "HOG visualization (6084-D descriptor)"],
        ):
            ax.imshow(image, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
    finish(fig, folder, f"{prefix}_2_HOG")
    save_json(
        Path(folder) / "input_samples.json",
        {
            "split": "train",
            "selection": "first filename in train/10000 and train/500000",
            "samples": [
                {
                    "relative_path": s.relative_path,
                    "label_vnd": int(s.label_vnd),
                    "sha256": sha256(Path(data_root) / s.relative_path),
                }
                for s in samples
            ],
            "hsv_hog_role": "SVM input descriptors; for CNN these figures illustrate inputs only, not learned features",
        },
    )


def save_scores(folder, split, manifest, pred, extra=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    truth = manifest.label_vnd.to_numpy(dtype=np.int64)
    scores = evaluate_classification(truth, pred)
    predictions = manifest.rename(columns={"label_vnd": "y_true_vnd"}).copy()
    predictions["y_pred_vnd"] = pred
    predictions["correct"] = truth == pred
    predictions["gap_vnd"] = np.abs(truth - pred)
    if extra:
        for key, value in extra.items():
            predictions[key] = value
    predictions.to_csv(folder / f"predictions_{split}.csv", index=False)
    manifest.to_csv(folder / f"{split}_manifest.csv", index=False)
    save_json(folder / f"metrics_{split}.json", scores)
    pd.DataFrame(scores["per_class"]).to_csv(folder / f"per_class_{split}.csv", index=False)
    pd.DataFrame(scores["error_pairs"], columns=ERROR_COLUMNS).to_csv(
        folder / f"error_pairs_{split}.csv", index=False
    )
    pd.DataFrame(scores["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        folder / f"confusion_matrix_{split}.csv"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(np.array(scores["confusion_matrix"]), display_labels=CLASS_ORDER).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=45
    )
    ax.set_title(f"Confusion matrix — {split} (rows: true; columns: predicted)")
    finish(fig, folder, f"confusion_matrix_{split}")
    return scores, predictions


def error_tables(predictions, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scores = evaluate_classification(predictions.y_true_vnd, predictions.y_pred_vnd)
    directed = pd.DataFrame(scores["error_pairs"], columns=ERROR_COLUMNS).sort_values(
        ["total_gap_vnd", "true_vnd", "predicted_vnd"], ascending=[False, True, True]
    )
    directed.to_csv(folder / "error_pairs_cost.csv", index=False)
    per_class = pd.DataFrame(scores["per_class"])
    per_class["error_rate"] = per_class.fn / per_class.support.replace(0, np.nan)
    per_class.to_csv(folder / "per_class_analysis.csv", index=False)
    cm = np.array(scores["confusion_matrix"])
    pairs = pd.DataFrame(
        [
            {
                "class_a": a,
                "class_b": b,
                "count": int(cm[i, j] + cm[j, i]),
                "total_gap_vnd": int(cm[i, j] + cm[j, i]) * abs(a - b),
            }
            for i, a in enumerate(CLASS_ORDER)
            for j, b in enumerate(CLASS_ORDER)
            if i < j
        ]
    )
    pairs.sort_values(["count", "class_a", "class_b"], ascending=[False, True, True]).to_csv(
        folder / "error_pairs_bidirectional.csv", index=False
    )
    predictions.loc[~predictions.correct.astype(bool)].to_csv(
        folder / "all_errors.csv", index=False
    )
    return per_class, directed


def select_cases(predictions, limit=12):
    """Deterministic: common directions, costly mistakes, small/large gaps, correct controls."""
    frame = predictions.copy().reset_index(drop=True)
    frame["correct"] = frame.y_true_vnd == frame.y_pred_vnd
    frame["gap_vnd"] = np.abs(frame.y_true_vnd - frame.y_pred_vnd)
    if "score_gap" not in frame:
        frame["score_gap"] = 0.0
    selected, reasons = [], []

    def choose(pool, sort, ascending, reason):
        pool = pool.loc[~pool.index.isin(selected)].sort_values(
            sort + ["relative_path"], ascending=ascending + [True]
        )
        if len(pool) and len(selected) < limit:
            selected.append(int(pool.index[0]))
            reasons.append(reason)

    errors = frame.loc[~frame.correct]
    counts = (
        errors.groupby(["y_true_vnd", "y_pred_vnd"])
        .size()
        .sort_values(ascending=False, kind="stable")
    )
    for (a, b), _ in counts.head(5).items():
        choose(
            errors.loc[(errors.y_true_vnd == a) & (errors.y_pred_vnd == b)],
            ["score_gap"],
            [False],
            "frequent_direction",
        )
    choose(errors, ["gap_vnd"], [False], "largest_VND_gap")
    choose(errors, ["score_gap"], [False], "error_large_score_gap")
    choose(errors, ["score_gap"], [True], "error_small_score_gap")
    choose(frame.loc[frame.correct], ["score_gap"], [True], "correct_small_score_gap")
    choose(frame.loc[frame.correct], ["score_gap"], [False], "correct_large_score_gap")
    cases = frame.loc[selected].copy().reset_index(drop=True)
    cases.insert(0, "case_id", [f"C{i + 1:02d}" for i in range(len(cases))])
    cases["selection_reason"] = reasons
    return cases


def error_gallery(cases, data_root, folder, prefix, scores=None):
    cases.to_csv(Path(folder) / "selected_cases.csv", index=False)
    for start in range(0, len(cases), 4):
        page = cases.iloc[start : start + 4]
        fig, axes = plt.subplots(len(page), 2, figsize=(12, 3.2 * len(page)), squeeze=False)
        for r, row in enumerate(page.itertuples()):
            axes[r, 0].imshow(read_rgb224(Path(data_root) / row.relative_path))
            axes[r, 0].axis("off")
            axes[r, 0].set_title(
                f"{row.case_id} true {row.y_true_vnd:,} → predicted {row.y_pred_vnd:,}"
            )
            if scores is not None:
                axes[r, 1].bar(
                    [f"{v // 1000}k" for v in CLASS_ORDER],
                    scores[int(getattr(row, "test_index", row.Index))],
                )
                axes[r, 1].set_title(f"{row.selection_reason}; gap={row.score_gap:.4f}")
            else:
                axes[r, 1].axis("off")
                axes[r, 1].text(
                    0.05, 0.5, f"{row.selection_reason}\n{row.relative_path}", wrap=True
                )
        finish(fig, folder, f"{prefix}_ERRORS_{start // 4 + 1:02d}")


def hsv_histograms(data_root, manifest, folder, prefix):
    """Match PDF slide 23: RGB + 12-bin marginal pixel histograms for H/S/V."""
    samples = []
    preferred = {10000: "train/10000/10000_000896.png", 500000: "train/500000/500000_001722.png"}
    for value in (10000, 500000):
        pool = manifest.loc[manifest.label_vnd == value]
        if len(pool):
            match = pool.loc[pool.relative_path == preferred[value]]
            samples.append(match.iloc[0] if len(match) else pool.iloc[0])
    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 3.2 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        rgb = read_rgb224(Path(data_root) / sample.relative_path)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        axes[row, 0].imshow(rgb)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{sample.label_vnd:,} VND")
        for j, (name, end, color) in enumerate(
            [
                ("Hue (H)", 180, "#b17b00"),
                ("Saturation (S)", 256, "#5e4c98"),
                ("Value (V)", 256, "#177e89"),
            ],
            1,
        ):
            counts, edges = np.histogram(hsv[:, :, j - 1], bins=12, range=(0, end))
            axes[row, j].bar(
                edges[:-1],
                counts / counts.sum() * 100,
                width=np.diff(edges),
                align="edge",
                color=color,
                edgecolor="white",
            )
            axes[row, j].set_title(name)
            axes[row, j].set_xlabel("Channel value")
            axes[row, j].set_ylabel("Pixels (%)")
            axes[row, j].set_xlim(0, end)
            axes[row, j].grid(axis="y", alpha=0.2)
    finish(fig, folder, f"{prefix}_HSV_HISTOGRAMS")
    save_json(
        Path(folder) / "hsv_histogram_samples.json",
        {
            "samples": [s.to_dict() for s in samples],
            "bins_per_channel": 12,
            "role": "Marginal histograms for illustration; classifier uses JOINT 1728-D histogram.",
        },
    )


def svm_reference_gallery(cases, train_manifest, x_train, x_test, data_root, folder):
    """Nearest train examples are visual context, not an explanation of SVM boundary."""
    records = []
    for start in range(0, len(cases), 3):
        page = cases.iloc[start : start + 3]
        fig, axes = plt.subplots(len(page), 3, figsize=(13, 3.5 * len(page)), squeeze=False)
        for r, row in enumerate(page.itertuples()):
            query = x_test[int(row.test_index)]
            chosen = []
            for label in [row.y_pred_vnd, row.y_true_vnd]:
                indices = np.flatnonzero(train_manifest.label_vnd.to_numpy() == label)
                # Bounded chunks prevent a full 12k × feature_dim subtraction allocation.
                best = (float("inf"), None)
                for begin in range(0, len(indices), 256):
                    batch = indices[begin : begin + 256]
                    distances = np.linalg.norm(np.asarray(x_train[batch]) - query, axis=1)
                    k = int(distances.argmin())
                    best = min(best, (float(distances[k]), int(batch[k])))
                chosen.append(best)
            paths = [row.relative_path] + [
                train_manifest.iloc[index].relative_path for _, index in chosen
            ]
            titles = [
                f"{row.case_id} TEST true {row.y_true_vnd:,}\npredicted {row.y_pred_vnd:,}",
                f"TRAIN predicted class {row.y_pred_vnd:,}\nEuclidean d={chosen[0][0]:.4f}",
                f"TRAIN true class {row.y_true_vnd:,}\nEuclidean d={chosen[1][0]:.4f}",
            ]
            for ax, path, title in zip(axes[r], paths, titles):
                ax.imshow(read_rgb224(Path(data_root) / path))
                ax.axis("off")
                ax.set_title(title, fontsize=10)
            records.append(
                {
                    "case_id": row.case_id,
                    "test_path": paths[0],
                    "predicted_class_reference": paths[1],
                    "true_class_reference": paths[2],
                    "predicted_class_distance": chosen[0][0],
                    "true_class_distance": chosen[1][0],
                }
            )
        fig.suptitle(
            "Train examples in feature space; these neighbors do not determine SVM predictions",
            fontsize=11,
        )
        finish(fig, folder, f"SVM_TRAIN_REFERENCES_{start // 3 + 1:02d}")
    pd.DataFrame(records).to_csv(Path(folder) / "train_references.csv", index=False)
