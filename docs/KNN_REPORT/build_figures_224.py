"""Tạo hình báo cáo từ ảnh và kết quả RGB224 đã có; không chạy mô hình."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.preprocessing import read_rgb224

REPORT = Path(__file__).resolve().parent
RESULTS = ROOT / "src/KNN/results_224"
FIGURES = REPORT / "KNN224_FIGURES"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
sources = set()
outputs = []


def read_table(relative):
    path = RESULTS / relative
    sources.add(path)
    return pd.read_csv(path)


def rgb(relative):
    path = ROOT / "DATA" / relative
    sources.add(path)
    return read_rgb224(path)


def save(fig, name, vector=False):
    path = FIGURES / (name + ".png")
    fig.savefig(path, dpi=300)
    outputs.append(path)
    if vector:
        pdf = FIGURES / (name + ".pdf")
        fig.savefig(pdf)
        outputs.append(pdf)
    plt.close(fig)


def money(value):
    return f"{int(value):,}".replace(",", ".")


manifest = read_table("hsv12/features/train_manifest.csv")
example_paths = [manifest.loc[manifest.label_vnd == label, "relative_path"].iloc[0]
                 for label in [10000, 500000]]
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for row, path in enumerate(example_paths):
    image_path = ROOT / "DATA" / path
    original = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    for col, image in enumerate([original, rgb(path)]):
        axes[row, col].imshow(image)
        axes[row, col].set_title(
            f"{'Trước resize' if col == 0 else 'Sau resize'}: {image.shape[1]} × {image.shape[0]}")
        axes[row, col].axis("off")
fig.tight_layout()
save(fig, "KNN_1_RESIZE")

hsv_examples = [
    ("train/10000/10000_000896.png", 10000),
    ("train/500000/500000_001722.png", 500000),
]
fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
for row, (path, label) in enumerate(hsv_examples):
    hsv_rgb = rgb(path)
    hsv = cv2.cvtColor(hsv_rgb, cv2.COLOR_RGB2HSV)
    axes[row, 0].imshow(hsv_rgb)
    axes[row, 0].set_title(f"Ảnh RGB224 — {money(label)}đ")
    axes[row, 0].axis("off")
    for channel, title, cmap, maximum in [
        (0, "H — Sắc độ", "hsv", 179),
        (1, "S — Độ bão hòa", "gray", 255),
        (2, "V — Độ sáng", "gray", 255),
    ]:
        ax = axes[row, channel + 1]
        plane = ax.imshow(hsv[:, :, channel], cmap=cmap, vmin=0, vmax=maximum)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(plane, ax=ax, orientation="horizontal", fraction=0.05,
                     pad=0.04, ticks=[0, maximum])
fig.tight_layout()
save(fig, "KNN_HSV_CHANNELS")

sample = rgb(example_paths[0])
gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
descriptor, visual = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), block_norm="L2-Hys",
                         transform_sqrt=False, visualize=True, feature_vector=True,
                         channel_axis=None)
assert descriptor.size == 6084
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, image, title in zip(axes, [sample, gray, visual],
                            ["RGB224", "Ảnh xám", "Trực quan hướng gradient HOG"]):
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.tight_layout()
save(fig, "KNN_2_HOG")

cm_path = RESULTS / "test_final/confusion_matrix_test.csv"
sources.add(cm_path)
cm = pd.read_csv(cm_path, index_col=0)
fig, ax = plt.subplots(figsize=(6.8, 5.8))
ax.imshow(cm.to_numpy(), cmap="Blues")
ticks = [money(v) for v in cm.index]
ax.set_xticks(range(6), ticks, rotation=35, ha="right")
ax.set_yticks(range(6), ticks)
ax.set_xlabel("Mệnh giá dự đoán (đ)")
ax.set_ylabel("Mệnh giá thật (đ)")
for i in range(6):
    for j in range(6):
        value = cm.iloc[i, j]
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.to_numpy().max() / 2 else "black")
fig.tight_layout()
save(fig, "KNN_3_CONFUSION_TEST", vector=True)

cases = read_table("error_analysis/selected_cases.csv")
train = read_table("hsv12/features/train_manifest.csv")
for case_ids, filename in [(["C04", "C09"], "KNN_4_ERRORS"),
                            (["C10", "C11"], "KNN_5_BOUNDARY")]:
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4))
    for r, case_id in enumerate(case_ids):
        case = cases.loc[cases.case_id == case_id].iloc[0]
        contrast_path = case.other_train_path if case.correct else case.same_train_path
        contrast_index = int(case.other_train_index if case.correct else case.same_train_index)
        contrast_distance = case.other_distance if case.correct else case.same_distance
        contrast_label = train.iloc[contrast_index].label_vnd
        chosen_label = train.iloc[int(case.nearest_train_index)].label_vnd
        names = [f"{case_id} — Test: {money(case.y_true_vnd)}đ\nDự đoán: {money(case.y_pred_vnd)}đ",
                 f"Train được chọn: {money(chosen_label)}đ\nd = {case.nearest_distance:.6f}",
                 f"Train đối chiếu: {money(contrast_label)}đ\nd = {contrast_distance:.6f}"]
        for c, path in enumerate([case.relative_path, case.nearest_train_path, contrast_path]):
            axes[r, c].imshow(rgb(path))
            axes[r, c].set_title(names[c], fontsize=10)
            axes[r, c].axis("off")
    fig.tight_layout()
    save(fig, filename)

def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


provenance = {
    "script": Path(__file__).relative_to(ROOT).as_posix(),
    "preprocessing_examples": example_paths,
    "hsv_examples": [path for path, label in hsv_examples],
    "case_groups": [["C04", "C09"], ["C10", "C11"]],
    "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
    "outputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
    "new_model_training": False,
}
(FIGURES / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
print("Created six report figures from existing images/results.")
