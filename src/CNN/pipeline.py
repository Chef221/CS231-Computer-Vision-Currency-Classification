"""Huấn luyện trên train, chọn checkpoint bằng dev, sau đó đánh giá test."""

from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from src.CNN.data import CLASSES, load_config, make_dataset, manifest_for_split, seed_everything
from src.CNN.model import compile_model, cosine_learning_rate, inference_model
from src.evaluation.value_metrics import evaluate_classification
from src.experiments.runtime import save_json
from src.experiments.figures import finish, save_scores


def predict(model, dataset):
    model = inference_model(model)
    return np.concatenate([model(images, training=False).numpy() for images, _ in dataset])


def metric_key(scores):
    # So sánh theo F1 cao trước; nếu bằng nhau thì MAE thấp trước.
    return (-float(scores["macro_f1"]), float(scores["mae_vnd"]))


def evaluate_dev(model, dataset, manifest):
    model = inference_model(model)
    chunks = []
    loss_sum = 0.0
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(reduction="sum")
    for images, labels in dataset:
        probabilities = model(images, training=False)
        loss_sum += float(cross_entropy(labels, probabilities).numpy())
        chunks.append(probabilities.numpy())
    probabilities = np.concatenate(chunks)
    if not np.isfinite(probabilities).all() or not np.isfinite(loss_sum):
        raise FloatingPointError("Dev có giá trị NaN hoặc vô hạn.")
    predictions = np.array(CLASSES)[probabilities.argmax(axis=1)]
    return evaluate_classification(manifest.label_vnd, predictions), loss_sum / len(manifest)


class DevCheckpoint(tf.keras.callbacks.Callback):
    """Mỗi epoch: chấm dev một lần, lưu mạng tốt nhất và đếm patience."""

    def __init__(self, dataset, manifest, folder, settings):
        super().__init__()
        self.dataset = dataset
        self.manifest = manifest
        self.folder = Path(folder)
        self.settings = settings
        self.rows = []
        self.best = None
        self.wait = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()
        self.model.optimizer.learning_rate.assign(cosine_learning_rate(epoch, self.settings))

    def on_train_batch_end(self, batch, logs=None):
        if not np.isfinite((logs or {}).get("loss", np.nan)):
            raise FloatingPointError("Loss train có giá trị NaN hoặc vô hạn.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        start = time.perf_counter()
        train_seconds = start - self.started
        scores, dev_loss = evaluate_dev(self.model, self.dataset, self.manifest)
        logs.update(
            {
                "epoch": epoch + 1,
                "val_loss": dev_loss,
                "val_accuracy": scores["accuracy"],
                "dev_macro_f1": scores["macro_f1"],
                "dev_mae_vnd": scores["mae_vnd"],
                "learning_rate": float(self.model.optimizer.learning_rate.numpy()),
                "train_seconds": train_seconds,
                "dev_seconds": time.perf_counter() - start,
            }
        )
        self.rows.append(dict(logs))
        if self.best is None or metric_key(scores) < self.best:
            self.best = metric_key(scores)
            self.wait = 0
            self.model.save(self.folder / "best.keras")
        else:
            self.wait += 1
        pd.DataFrame(self.rows).to_csv(self.folder / "history.csv", index=False)
        print(f"Dev: Macro-F1={scores['macro_f1']:.6f} | MAE={scores['mae_vnd']:,.2f} đồng/ảnh")
        if self.wait >= self.settings["patience"]:
            self.model.stop_training = True


def learning_curves(history, folder):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history.epoch, history.loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="dev")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].plot(history.epoch, history.accuracy, label="train")
    axes[1].plot(history.epoch, history.val_accuracy, label="dev")
    axes[1].set_ylabel("Accuracy")
    axes[2].plot(history.epoch, history.dev_macro_f1, label="dev Macro-F1")
    axes[2].set_ylabel("Macro-F1")
    axes[3].plot(history.epoch, history.learning_rate, label="cosine LR")
    axes[3].set_ylabel("Learning rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.25)
    finish(fig, folder, "learning_curves")


def train_candidate(model, candidate_id, data_root, results_root):
    root = Path(results_root)
    experiment = load_config(root)
    config = next(c for c in experiment["candidates"] if c["candidate_id"] == candidate_id)
    settings = experiment["training"]
    folder = root / "candidates" / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / "inputs/train_manifest.csv")
    dev = pd.read_csv(root / "inputs/dev_manifest.csv")
    dev_ds = make_dataset(dev, data_root, config, settings["batch_size"])
    seed_everything(settings["seed"])
    compile_model(model, settings)
    save_json(folder / "run_config.json", {"config": config, "training": settings})
    callback = DevCheckpoint(dev_ds, dev, folder, settings)
    # Giữ lịch seed, shuffle và augmentation của baseline theo từng epoch.
    for epoch in range(settings["epochs"]):
        seed_everything(settings["seed"] + epoch + 1)
        train_ds = make_dataset(
            train,
            data_root,
            config,
            settings["batch_size"],
            training=True,
            epoch=epoch + 1,
            seed=settings["seed"],
        )
        model.fit(
            train_ds,
            initial_epoch=epoch,
            epochs=epoch + 1,
            callbacks=[callback],
            shuffle=False,
            verbose=2,
        )
        if callback.wait >= settings["patience"]:
            break
    history = pd.DataFrame(callback.rows)
    best_epoch = int(
        history.sort_values(["dev_macro_f1", "dev_mae_vnd", "epoch"], ascending=[False, True, True])
        .iloc[0]
        .epoch
    )
    best_model = tf.keras.models.load_model(folder / "best.keras", compile=False)
    probabilities = predict(best_model, dev_ds)
    scores, _ = save_scores(folder, "dev", dev, np.array(CLASSES)[probabilities.argmax(axis=1)])
    summary = {
        **config,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameters": int(best_model.count_params()),
        **{name: scores[name] for name in ("accuracy", "macro_f1", "mae_vnd")},
        "tensorflow": tf.__version__,
        "precision": tf.keras.mixed_precision.global_policy().name,
    }
    save_json(folder / "summary.json", summary)
    lines = []
    best_model.summary(print_fn=lambda line, **kwargs: lines.append(line))
    (folder / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    learning_curves(history, folder)
    return history, summary


def select_and_lock(results_root):
    root = Path(results_root)
    rows = []
    for order, config in enumerate(load_config(root)["candidates"]):
        summary = json.loads(
            (root / "candidates" / config["candidate_id"] / "summary.json").read_text()
        )
        rows.append({**summary, "candidate_order": order})
    comparison = pd.DataFrame(rows).sort_values(
        ["macro_f1", "mae_vnd", "candidate_order"], ascending=[False, True, True]
    )
    best = comparison.iloc[0]
    config = next(
        c for c in load_config(root)["candidates"] if c["candidate_id"] == best.candidate_id
    )
    lock = {
        "candidate": best.candidate_id,
        "config": config,
        "best_epoch": int(best.best_epoch),
        "checkpoint": f"candidates/{best.candidate_id}/best.keras",
        "classes_vnd": CLASSES,
        "dev_metrics": {key: float(best[key]) for key in ("accuracy", "macro_f1", "mae_vnd")},
    }
    save_json(root / "selection/locked_cnn_config.json", lock)
    comparison.to_csv(root / "selection/best_by_candidate_dev.csv", index=False)
    return comparison, lock


def load_locked_model(results_root):
    root = Path(results_root)
    lock = json.loads((root / "selection/locked_cnn_config.json").read_text())
    model = tf.keras.models.load_model(root / lock["checkpoint"], compile=False)
    return inference_model(model), lock


def evaluate_test(data_root, results_root):
    root = Path(results_root)
    model, lock = load_locked_model(root)
    experiment = load_config(root)
    manifest = manifest_for_split(data_root, "test", experiment["expected_full_dataset"])
    probabilities = predict(
        model,
        make_dataset(manifest, data_root, lock["config"], experiment["training"]["batch_size"]),
    )
    ranked = np.sort(probabilities, axis=1)
    extra = {"confidence": ranked[:, -1], "score_gap": ranked[:, -1] - ranked[:, -2]}
    extra.update({f"p_{value}": probabilities[:, i] for i, value in enumerate(CLASSES)})
    return save_scores(
        root / "test_final",
        "test",
        manifest,
        np.array(CLASSES)[probabilities.argmax(axis=1)],
        extra,
    )
