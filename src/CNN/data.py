"""Fixed VND mapping, shared RGB224 decoding and training-only augmentation."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.evaluation.value_metrics import CLASS_ORDER
from src.preprocessing import read_rgb224

CLASSES = list(CLASS_ORDER)
CLASS_TO_IDX = {value: index for index, value in enumerate(CLASSES)}
SPLIT_SIZES = {"train": 12494, "dev": 1564, "test": 1553}
CANDIDATES = [
    {"candidate_id": "C01_RGB", "color": "rgb", "augmentation": False,
     "channels": [32, 64, 128]},
    {"candidate_id": "C02_GRAY", "color": "gray", "augmentation": False,
     "channels": [32, 64, 128]},
    {"candidate_id": "C03_RGB_AUG", "color": "rgb", "augmentation": True,
     "channels": [32, 64, 128]},
    {"candidate_id": "C04_DEEP_AUG", "color": "rgb", "augmentation": True,
     "channels": [32, 64, 128, 256]},
]
CODE_FILES = ["src/preprocessing.py", "src/evaluation/value_metrics.py",
              *[f"src/CNN/{name}.py" for name in
                ("model", "data", "pipeline", "analysis", "runtime")]]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def manifest_for_split(data_root, split, expected_size=None):
    rows = []
    for label in CLASSES:
        folder = Path(data_root) / split / str(label)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.glob("*.png")):
            rows.append({"relative_path": path.relative_to(data_root).as_posix(),
                         "label_vnd": label})
    frame = pd.DataFrame(rows, columns=["relative_path", "label_vnd"])
    expected_size = SPLIT_SIZES[split] if expected_size is None else expected_size
    if len(frame) != expected_size or set(frame.label_vnd) != set(CLASSES):
        raise ValueError(f"{split}: expected {expected_size} images of six classes, found {len(frame)}")
    return frame


def same_manifest(left, right):
    columns = ["relative_path", "label_vnd"]
    if not left[columns].reset_index(drop=True).equals(right[columns].reset_index(drop=True)):
        raise ValueError("Đường dẫn, nhãn hoặc thứ tự ảnh không khớp manifest.")


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def augment_rgb(image):
    angle = random.uniform(-12.0, 12.0)
    scale = random.uniform(0.95, 1.05)
    matrix = cv2.getRotationMatrix2D((111.5, 111.5), angle, scale)
    matrix[:, 2] += [random.uniform(-0.05, 0.05) * 224,
                     random.uniform(-0.05, 0.05) * 224]
    image = cv2.warpAffine(image, matrix, (224, 224), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    contrast = random.uniform(0.9, 1.1)
    brightness = random.uniform(-0.1, 0.1) * 255
    return np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                   0, 255).astype(np.uint8)


def image_tensor(rgb, color="rgb"):
    if color == "gray":
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None]
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255)


class BanknoteDataset(Dataset):
    def __init__(self, manifest, data_root, config, training=False):
        self.frame = manifest.reset_index(drop=True)
        self.data_root, self.config, self.training = Path(data_root), config, training
        if training and not self.frame.relative_path.str.startswith("train/").all():
            raise ValueError("Training/augmentation chỉ được dùng trên train.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rgb = read_rgb224(self.data_root / row.relative_path)
        if self.training and self.config["augmentation"]:
            rgb = augment_rgb(rgb)
        return image_tensor(rgb, self.config["color"]), CLASS_TO_IDX[int(row.label_vnd)]


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def make_loader(manifest, data_root, config, training, settings, device, epoch=0):
    generator = torch.Generator().manual_seed(settings["seed"] + epoch)
    return DataLoader(
        BanknoteDataset(manifest, data_root, config, training),
        batch_size=settings["batch_size"], shuffle=training,
        num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker, generator=generator,
        persistent_workers=False, drop_last=False,
    )


def protocol(project_root):
    return {
        "model_family": "CNN", "protocol_id": "cnn_rgb224_v1",
        "classes_vnd": CLASSES, "split_sizes": SPLIT_SIZES,
        "preprocessing": {"reader": "src.preprocessing.read_rgb224", "size": [224, 224],
                          "resize": "direct OpenCV INTER_LINEAR", "scale": "float32 / 255",
                          "normalization": "no ImageNet mean/std; no fitted pixel scaler"},
        "architecture": {"block": "2 x (Conv3x3, BN, ReLU), MaxPool2, Dropout2d0.1",
                         "head": "GAP, Dense128, ReLU, Dropout0.5, Dense6 logits",
                         "initialization": "Kaiming conv; Xavier linear; BN weight1 bias0",
                         "pretrained": False},
        "augmentation": {"train_only": True, "degrees": [-12, 12],
                         "translate_fraction": [-0.05, 0.05], "scale": [0.95, 1.05],
                         "brightness_add_fraction": [-0.1, 0.1], "contrast": [0.9, 1.1],
                         "border_fill": [127, 127, 127], "flip": False,
                         "hue_saturation_jitter": False},
        "candidates": CANDIDATES,
        "training": {"seed": 42, "batch_size": 64, "num_workers": 2,
                     "max_epochs": 25, "patience": 6, "learning_rate": 0.001,
                     "weight_decay": 0.0001, "optimizer": "AdamW",
                     "betas": [0.9, 0.999], "scheduler": "CosineAnnealingLR",
                     "eta_min": 0.00001, "loss": "CrossEntropyLoss",
                     "label_smoothing": 0.0, "class_weights": None,
                     "amp_cuda": True, "gradient_clip_norm": 5.0},
        "selection": "max unrounded dev macro_f1; min mae_vnd; candidate order; earliest epoch",
        "checkpoint_selection": "max unrounded dev macro_f1; min mae_vnd; earliest epoch",
        "refit_train_plus_dev": False,
        "determinism": "fixed seeds; deterministic warn_only; no cross-device equality guarantee",
        "test_history": "Previously inspected in earlier project experiments; not a new holdout.",
        "source_sha256": {name: sha256(Path(project_root) / name) for name in CODE_FILES},
    }


def assert_protocol(results_root, project_root):
    p = json.loads(Path(results_root, "inputs/experiment_config.json").read_text())
    current = protocol(project_root)
    if p != current:
        raise ValueError("Protocol hoặc code đã đổi. Dùng thư mục thực nghiệm mới và chạy lại từ 01.")
    for split in ("train", "dev"):
        path = Path(results_root, "inputs", f"{split}_manifest.csv")
        recorded = json.loads(Path(results_root, "inputs/manifest_hashes.json").read_text())
        if sha256(path) != recorded[split]:
            raise ValueError(f"Manifest {split} đã đổi sau bước 01.")
    return p
