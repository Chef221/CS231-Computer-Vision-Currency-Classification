"""Vẽ augmentation, feature maps, PCA fit train và Grad-CAM của CNN."""

from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from src.CNN.data import CLASSES, CLASS_TO_IDX, image_array, make_dataset, load_config
from src.CNN.pipeline import load_locked_model
from src.experiments.figures import finish, error_tables, select_cases
from src.experiments.runtime import save_json
from src.experiments.figures import sha256


def probe_model(model, names):
    # Explicit functional replay also works with Sequential models reloaded in Keras 3.
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    outputs = {}
    for layer in model.layers:
        x = layer(x, training=False)
        if layer.name in names:
            outputs[layer.name] = x
    return tf.keras.Model(inputs, [outputs[n] for n in names])


def gradcam(model, image, class_index):
    last = next(
        layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)
    )
    feature_index = next(i for i, l in enumerate(model.layers) if l.name == last)
    # Gradients of pre-softmax logits avoid saturation and do not mutate the model.
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(image[None], dtype=tf.float32)
        tape.watch(x)
        maps = None
        for index, layer in enumerate(model.layers[:-1]):
            x = layer(x, training=False)
            if index == feature_index:
                maps = x
        dense = model.layers[-1]
        logits = tf.matmul(tf.cast(x, tf.float32), tf.cast(dense.kernel, tf.float32)) + tf.cast(
            dense.bias, tf.float32
        )
        target = logits[:, int(class_index)]
    gradients = tape.gradient(target, maps)
    if gradients is None:
        raise ValueError("Grad-CAM gradient bị ngắt")
    weights = tf.reduce_mean(tf.cast(gradients, tf.float32), axis=(1, 2), keepdims=True)
    heat = tf.nn.relu(tf.reduce_sum(weights * tf.cast(maps, tf.float32), axis=-1))[0].numpy()
    maximum = float(heat.max())
    if maximum > 0:
        heat /= maximum
    heat = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_LINEAR)
    return heat, {
        "target_index": int(class_index),
        "target_vnd": CLASSES[int(class_index)],
        "conv_layer": last,
        "zero_heatmap": maximum == 0,
        "finite": bool(np.isfinite(heat).all()),
    }


def augmentation_examples(data_root, results_root):
    import random
    from src.CNN.data import augment_rgb

    train = pd.read_csv(Path(results_root) / "inputs/train_manifest.csv")
    row = train.iloc[0]
    rgb = cv2.cvtColor(cv2.imread(str(Path(data_root) / row.relative_path)), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    rng = random.Random(42)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB224")
    for ax in axes[1:]:
        ax.imshow(augment_rgb(rgb, rng))
        ax.set_title("Train: original OpenCV augmentation")
    for ax in axes:
        ax.axis("off")
    finish(fig, Path(results_root) / "inputs", "CNN_AUGMENTATION")
    save_json(
        Path(results_root) / "inputs/augmentation_example.json",
        {
            "relative_path": row.relative_path,
            "seed": 42,
            "implementation": "same augment_rgb function used by training; uint8 before Rescaling",
        },
    )


def inspect_features(data_root, results_root):
    root = Path(results_root)
    model, lock = load_locked_model(root)
    protocol = load_config(root)
    train = pd.read_csv(root / "inputs/train_manifest.csv")
    dev = pd.read_csv(root / "inputs/dev_manifest.csv")
    embed = probe_model(model, ["embedding"])

    # One requested output is returned as a tensor by some Keras versions.
    def vectors(frame):
        chunks = []
        for images, _ in make_dataset(
            frame, data_root, lock["config"], protocol["training"]["batch_size"]
        ):
            output = embed(images, training=False)
            chunks.append(np.asarray(output[0] if isinstance(output, (list, tuple)) else output))
        return np.concatenate(chunks)

    x_train = vectors(train)
    x_dev = vectors(dev)
    pca = PCA(n_components=2, svd_solver="full").fit(x_train)
    xy = pca.transform(x_dev)
    folder = root / "feature_analysis"
    folder.mkdir(exist_ok=True)
    coords = dev.copy()
    coords["pc1"] = xy[:, 0]
    coords["pc2"] = xy[:, 1]
    coords.to_csv(folder / "embedding_pca_dev.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for v in CLASSES:
        mask = dev.label_vnd == v
        ax.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=0.5, label=f"{v:,}")
    ax.legend(title="VND")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("CNN embedding: PCA fit train, dev projected")
    finish(fig, folder, "CNN_EMBEDDING_PCA")
    names = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    probes = probe_model(model, [names[0], names[-1]])
    image = image_array(Path(data_root) / train.iloc[0].relative_path, lock["config"]["color"])
    maps = probes(image[None], training=False)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    channels = []
    for row, (name, activation) in enumerate(zip([names[0], names[-1]], maps)):
        activation = activation[0].numpy()
        # Stable high-energy channels; independent scaling per feature map.
        chosen = np.argsort(np.mean(np.abs(activation), axis=(0, 1)))[-6:][::-1]
        for ax, c in zip(axes[row], chosen):
            ax.imshow(activation[:, :, c], cmap="viridis")
            ax.axis("off")
            ax.set_title(f"{name} ch{c}")
            channels.append(
                {"layer": name, "channel": int(c), "relative_path": train.iloc[0].relative_path}
            )
        for ax in axes[row, len(chosen) :]:
            ax.axis("off")
    finish(fig, folder, "CNN_FEATURE_MAPS")
    pd.DataFrame(channels).to_csv(folder / "selected_channels.csv", index=False)
    save_json(
        folder / "analysis_record.json",
        {
            "checkpoint_sha256": sha256(root / lock["checkpoint"]),
            "pca_fit_split": "train",
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "embedding_dim": int(x_train.shape[1]),
            "sample": train.iloc[0].relative_path,
        },
    )
    return coords, pd.DataFrame(channels)


def analyze_errors(data_root, results_root):
    root = Path(results_root)
    predictions = pd.read_csv(root / "test_final/predictions_test.csv")
    model, lock = load_locked_model(root)
    folder = root / "error_analysis"
    folder.mkdir(exist_ok=True)
    per_class, directed = error_tables(predictions, folder)
    cases = select_cases(predictions)
    cases.to_csv(folder / "selected_cases.csv", index=False)
    checks = []
    for start in range(0, len(cases), 3):
        page = cases.iloc[start : start + 3]
        fig, axes = plt.subplots(len(page), 4, figsize=(16, 3.6 * len(page)), squeeze=False)
        for r, row in enumerate(page.itertuples()):
            rgb = cv2.cvtColor(
                cv2.imread(str(Path(data_root) / row.relative_path)), cv2.COLOR_BGR2RGB
            )
            rgb = cv2.resize(rgb, (224, 224))
            image = image_array(Path(data_root) / row.relative_path, lock["config"]["color"])
            axes[r, 0].imshow(rgb)
            axes[r, 0].set_title(
                f"{row.case_id} true {row.y_true_vnd:,}\npred {row.y_pred_vnd:,} | {row.selection_reason}",
                fontsize=10,
            )
            for col, value in [(1, row.y_pred_vnd), (2, row.y_true_vnd)]:
                heat, check = gradcam(model, image, CLASS_TO_IDX[int(value)])
                check.update(
                    {
                        "case_id": row.case_id,
                        "relative_path": row.relative_path,
                        "role": "predicted" if col == 1 else "true",
                    }
                )
                checks.append(check)
                axes[r, col].imshow(rgb)
                axes[r, col].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[r, col].set_title(
                    f"Grad-CAM target {value:,}" + (" (zero)" if check["zero_heatmap"] else "")
                )
            for ax in axes[r, :3]:
                ax.axis("off")
            axes[r, 3].bar(
                [f"{v // 1000}k" for v in CLASSES], [getattr(row, f"p_{v}") for v in CLASSES]
            )
            axes[r, 3].set_ylim(0, 1)
            axes[r, 3].set_title("Softmax (uncalibrated)")
        finish(fig, folder, f"CNN_GRADCAM_{start // 3 + 1:02d}")
    pd.DataFrame(checks).to_csv(folder / "gradcam_checks.csv", index=False)
    notes = cases[["case_id", "relative_path", "selection_reason"]].copy()
    for col in ["observation", "hypothesis", "not_verified"]:
        notes[col] = ""
    if not (folder / "visual_review_notes.csv").exists():
        notes.to_csv(folder / "visual_review_notes.csv", index=False)
    save_json(
        folder / "analysis_record.json",
        {
            "n_test": len(predictions),
            "n_errors": int((~predictions.correct).sum()),
            "predictions_sha256": sha256(root / "test_final/predictions_test.csv"),
            "checkpoint_sha256": sha256(root / lock["checkpoint"]),
            "test_used_for_retuning": False,
        },
    )
    return cases, directed
