"""Ảnh RGB224 như KNN, augmentation trên train và đọc ảnh theo batch."""

from pathlib import Path
import json
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.preprocessing import read_rgb224
from src.evaluation.value_metrics import CLASS_ORDER
from src.experiments.runtime import save_json

# Giữ thiết lập của lượt TensorFlow cuối đã dùng trong slide.
TRAINING = {
    "seed": 42,
    "epochs": 25,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "gradient_clip_norm": 5.0,
    "patience": 6,
    "eta_min": 0.00001,
}

CLASSES = list(CLASS_ORDER)
CLASS_TO_IDX = {value: index for index, value in enumerate(CLASSES)}
SPLIT_SIZES = {"train": 12494, "dev": 1564, "test": 1553}
CANDIDATES = [
    {"candidate_id": "C01_RGB", "color": "rgb", "augmentation": False, "channels": [32, 64, 128]},
    {"candidate_id": "C02_GRAY", "color": "gray", "augmentation": False, "channels": [32, 64, 128]},
    {
        "candidate_id": "C03_RGB_AUG",
        "color": "rgb",
        "augmentation": True,
        "channels": [32, 64, 128],
    },
    {
        "candidate_id": "C04_DEEP_AUG",
        "color": "rgb",
        "augmentation": True,
        "channels": [32, 64, 128, 256],
    },
]


def seed_everything(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def configure_gpu(mixed_precision=True):
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    tf.config.experimental.enable_tensor_float_32_execution(False)
    use_amp = mixed_precision and bool(tf.config.list_physical_devices("GPU"))
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if use_amp else "float32")
    print(
        "TensorFlow:",
        tf.__version__,
        "| GPUs:",
        tf.config.list_physical_devices("GPU"),
        "| precision:",
        tf.keras.mixed_precision.global_policy().name,
    )


def manifest_for_split(data_root, split, expected=True):
    rows = []
    for label in CLASSES:
        folder = Path(data_root) / split / str(label)
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise ValueError(f"Lớp rỗng: {folder}")
        rows.extend(
            {"relative_path": p.relative_to(data_root).as_posix(), "label_vnd": label}
            for p in paths
        )
    frame = pd.DataFrame(rows)
    if expected and len(frame) != SPLIT_SIZES[split]:
        raise ValueError(f"{split}: cần {SPLIT_SIZES[split]}, có {len(frame)} ảnh")
    return frame


def augment_rgb(image, rng):
    """Original uint8 OpenCV affine + photometric transform, same operation order."""
    angle = rng.uniform(-12.0, 12.0)
    scale = rng.uniform(0.95, 1.05)
    matrix = cv2.getRotationMatrix2D((111.5, 111.5), angle, scale)
    matrix[:, 2] += [rng.uniform(-0.05, 0.05) * 224, rng.uniform(-0.05, 0.05) * 224]
    image = cv2.warpAffine(
        image,
        matrix,
        (224, 224),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(127, 127, 127),
    )
    contrast = rng.uniform(0.9, 1.1)
    brightness = rng.uniform(-0.1, 0.1) * 255
    return np.clip(
        (image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness, 0, 255
    ).astype(np.uint8)


def image_array(path, color="rgb", augmentation_seed=None):
    image = read_rgb224(path)
    if augmentation_seed is not None:
        image = augment_rgb(image, random.Random(int(augmentation_seed)))
    if color == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None]
    return image.astype(np.float32)


def make_dataset(manifest, data_root, config, batch_size=64, training=False, epoch=1, seed=42):
    if training and not manifest.relative_path.str.startswith("train/").all():
        raise ValueError("Training/augmentation chỉ được dùng trên train.")
    paths = np.array([str(Path(data_root) / p) for p in manifest.relative_path])
    labels = np.array([CLASS_TO_IDX[int(v)] for v in manifest.label_vnd], dtype=np.int32)
    indices = np.arange(len(paths))
    if training:
        # Permute ALL path strings before decoding; no full-image RAM cache.
        indices = np.random.default_rng(seed + epoch).permutation(indices)
    sample_seeds = ((seed + epoch) * 1000003 + indices).astype(np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((paths[indices], labels[indices], sample_seeds))

    def decode(path, label, sample_seed):
        def read(value, augmentation_seed):
            augment_seed = (
                int(augmentation_seed.numpy()) if training and config["augmentation"] else None
            )
            return image_array(value.numpy().decode("utf-8"), config["color"], augment_seed)

        image = tf.py_function(read, [path, sample_seed], Tout=tf.float32)
        image.set_shape((224, 224, 1 if config["color"] == "gray" else 3))
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=2, deterministic=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.batch(batch_size, drop_remainder=False).with_options(options).prefetch(1)


def prepare(data_root, results_root, candidates=None, epochs=25, expected=True):
    """Lưu danh sách train/dev; chưa đọc ảnh test ở bước chuẩn bị."""
    folder = Path(results_root) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    manifests = {
        split: manifest_for_split(data_root, split, expected) for split in ("train", "dev")
    }
    config = {
        "framework": "tensorflow",
        "classes_vnd": CLASSES,
        "seed": 42,
        "candidates": candidates if candidates is not None else CANDIDATES,
        "training": {**TRAINING, "epochs": epochs},
        "expected_full_dataset": expected,
    }
    save_json(folder / "experiment_config.json", config)
    for split, frame in manifests.items():
        frame.to_csv(folder / f"{split}_manifest.csv", index=False)
    counts = [
        {"split": split, "label_vnd": value, "count": int((frame.label_vnd == value).sum())}
        for split, frame in manifests.items()
        for value in CLASSES
    ]
    pd.DataFrame(counts).to_csv(folder / "image_counts.csv", index=False)
    return config, manifests


def load_config(results_root):
    return json.loads((Path(results_root) / "inputs/experiment_config.json").read_text())
