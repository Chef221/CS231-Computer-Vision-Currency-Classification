"""Train on train, select on dev, lock weights and evaluate test exactly as saved."""

from __future__ import annotations

import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn

from src.CNN import data
from src.CNN.data import CLASSES, sha256, save_json, make_loader, same_manifest
from src.CNN.model import build_model, architecture_table
from src.evaluation.value_metrics import evaluate_classification
from src.preprocessing import read_rgb224


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
            "sklearn": sklearn.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda}


def prepare_inputs(data_root, results_root, project_root):
    results_root = Path(results_root)
    folder = results_root / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    readme = results_root / "README.md"
    source_readme = Path(project_root) / "src/CNN/results_224/README.md"
    if not readme.is_file() and source_readme.is_file():
        shutil.copy2(source_readme, readme)
    p = data.protocol(project_root)
    protocol_file = folder / "experiment_config.json"
    if protocol_file.is_file() and json.loads(protocol_file.read_text()) != p:
        raise ValueError("Thư mục kết quả đã có protocol khác. Dùng root mới.")
    counts, checks, hashes = [], [], {}
    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    for split in ("train", "dev"):
        manifest = data.manifest_for_split(data_root, split, p["split_sizes"][split])
        path = folder / f"{split}_manifest.csv"
        if path.is_file():
            same_manifest(pd.read_csv(path), manifest)
        manifest.to_csv(path, index=False)
        hashes[split] = sha256(path)
        for index, value in enumerate(CLASSES):
            subset = manifest[manifest.label_vnd == value]
            counts.append({"split": split, "denomination_vnd": value, "count": len(subset)})
            sample = read_rgb224(Path(data_root) / subset.relative_path.iloc[0])
            checks.append({"split": split, "label_vnd": value,
                           "relative_path": subset.relative_path.iloc[0],
                           "shape": str(sample.shape), "dtype": str(sample.dtype)})
            if split == "train":
                raw = cv2.cvtColor(cv2.imread(str(Path(data_root) / subset.relative_path.iloc[0])),
                                   cv2.COLOR_BGR2RGB)
                axes[0, index].imshow(raw)
                axes[0, index].set_title(f"{value:,} VND; {raw.shape[1]}×{raw.shape[0]}")
                axes[1, index].imshow(sample)
                axes[1, index].set_title("RGB 224×224")
                axes[0, index].axis("off")
                axes[1, index].axis("off")
    fig.tight_layout()
    fig.savefig(folder / "rgb224_examples.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(counts).to_csv(folder / "image_counts.csv", index=False)
    pd.DataFrame(checks).to_csv(folder / "input_checks.csv", index=False)
    save_json(folder / "manifest_hashes.json", hashes)
    save_json(protocol_file, p)
    rows = []
    for config in p["candidates"]:
        table = architecture_table(config)
        for row in table:
            rows.append({"candidate_id": config["candidate_id"], **row})
    pd.DataFrame(rows).to_csv(folder / "architecture.csv", index=False)
    return p, pd.DataFrame(counts), pd.DataFrame(rows)


def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _atomic_torch_save(path, content):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def _cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def score_predictions(frame):
    return evaluate_classification(frame.y_true_vnd.to_numpy(), frame.y_pred_vnd.to_numpy())


def evaluate_loader(model, loader, manifest, device):
    model.eval()
    predictions, probabilities = [], []
    loss_sum, count = 0.0, 0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite logits")
            loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            count += len(targets)
            probabilities.append(logits.softmax(1).cpu().numpy())
            predictions.extend(logits.argmax(1).cpu().tolist())
    probs = np.concatenate(probabilities)
    frame = manifest[["relative_path"]].copy().reset_index(drop=True)
    frame["y_true_vnd"] = manifest.label_vnd.to_numpy()
    frame["y_pred_vnd"] = np.asarray(CLASSES)[predictions]
    frame["correct"] = frame.y_true_vnd == frame.y_pred_vnd
    frame["gap_vnd"] = (frame.y_true_vnd - frame.y_pred_vnd).abs()
    ordered = np.sort(probs, axis=1)
    frame["top_softmax"] = ordered[:, -1]
    frame["softmax_gap"] = ordered[:, -1] - ordered[:, -2]
    for i, value in enumerate(CLASSES):
        frame[f"softmax_{value}"] = probs[:, i]
    return score_predictions(frame), frame, loss_sum / count


def save_evaluation(folder, split, scores, predictions):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / f"metrics_{split}.json", scores)
    predictions.to_csv(folder / f"predictions_{split}.csv", index=False)
    pd.DataFrame(scores["per_class"]).to_csv(folder / f"per_class_{split}.csv", index=False)
    pd.DataFrame(scores["error_pairs"]).to_csv(folder / f"error_pairs_{split}.csv", index=False)
    pd.DataFrame(scores["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        folder / f"confusion_matrix_{split}.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.asarray(scores["confusion_matrix"]),
                           display_labels=[f"{v // 1000}k" for v in CLASSES]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(folder / f"confusion_matrix_{split}.png", dpi=180)
    plt.close(fig)


def plot_history(history, destination):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(history.epoch, history.train_loss, label="train (online)")
    axes[0, 0].plot(history.epoch, history.dev_loss, label="dev")
    axes[0, 0].set_ylabel("Cross entropy")
    axes[0, 1].plot(history.epoch, history.train_accuracy, label="train (online)")
    axes[0, 1].plot(history.epoch, history.dev_accuracy, label="dev")
    axes[0, 1].set_ylabel("Accuracy")
    axes[1, 0].plot(history.epoch, history.dev_macro_f1, label="dev Macro-F1")
    axes[1, 0].set_ylabel("Macro-F1")
    axes[1, 1].plot(history.epoch, history.dev_mae_vnd, label="dev MAE")
    axes[1, 1].set_ylabel("VND / image")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _verify_run(folder, metadata):
    record = json.loads((folder / "run_complete.json").read_text())
    if record["metadata"] != metadata:
        raise ValueError(f"Run metadata changed: {folder}")
    for name, digest in record["sha256"].items():
        if not (folder / name).is_file() or sha256(folder / name) != digest:
            raise ValueError(f"Run artifact changed/missing: {folder / name}")
    return record


def train_candidate(candidate_id, data_root, results_root, project_root):
    results_root = Path(results_root)
    p = data.assert_protocol(results_root, project_root)
    config = next(c for c in p["candidates"] if c["candidate_id"] == candidate_id)
    settings = p["training"]
    folder = results_root / "candidates" / candidate_id
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"candidate": config, "training": settings,
                "protocol_sha256": sha256(results_root / "inputs/experiment_config.json")}
    if (folder / "run_complete.json").is_file():
        _verify_run(folder, metadata)
        print(candidate_id, "đã hoàn tất; đọc lại kết quả đã lưu.")
        return pd.read_csv(folder / "history.csv"), json.loads((folder / "summary.json").read_text())
    config_path = folder / "run_config.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != metadata:
        raise ValueError("Không ghi đè run với cấu hình khác. Dùng root mới.")
    save_json(config_path, metadata)
    train = pd.read_csv(results_root / "inputs/train_manifest.csv")
    dev = pd.read_csv(results_root / "inputs/dev_manifest.csv")
    device = _device()
    data.seed_everything(settings["seed"])
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["max_epochs"], eta_min=settings["eta_min"])
    use_amp = settings["amp_cuda"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dev_loader = make_loader(dev, data_root, config, False, settings, device)
    history, start_epoch, best_key, best_epoch, wait = [], 1, None, None, 0
    last = folder / "last.pt"
    if last.is_file():
        state = _load_checkpoint(last, device)
        if state["metadata"] != metadata:
            raise ValueError("Resume checkpoint belongs to another protocol/config.")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, start_epoch = state["history"], state["epoch"] + 1
        best_key, best_epoch, wait = tuple(state["best_key"]), state["best_epoch"], state["wait"]
        print("Resume", candidate_id, "từ epoch", start_epoch)
    for epoch in range(start_epoch, settings["max_epochs"] + 1):
        if wait >= settings["patience"]:
            break
        # Epoch-specific seeds also seed new DataLoader workers, so resume keeps the epoch schedule.
        data.seed_everything(settings["seed"] + epoch)
        train_loader = make_loader(train, data_root, config, True, settings, device, epoch)
        model.train()
        started = time.perf_counter()
        total_loss, total_correct, total = 0.0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets,
                                                    label_smoothing=settings["label_smoothing"])
            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(targets)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores, _, dev_loss = evaluate_loader(model, dev_loader, dev, device)
        row = {"epoch": epoch, "learning_rate": learning_rate,
               "train_loss": total_loss / total, "train_accuracy": total_correct / total,
               "dev_loss": dev_loss, "dev_accuracy": scores["accuracy"],
               "dev_macro_f1": scores["macro_f1"], "dev_mae_vnd": scores["mae_vnd"],
               "train_seconds": train_seconds, "dev_seconds": time.perf_counter() - started}
        history.append(row)
        key = (-scores["macro_f1"], scores["mae_vnd"], epoch)
        if best_key is None or key < best_key:
            best_key, best_epoch, wait = key, epoch, 0
            _atomic_torch_save(folder / "best.pt", {
                "state_dict": _cpu_weights(model), "metadata": metadata,
                "epoch": epoch, "dev_metrics": {k: scores[k] for k in ("macro_f1", "accuracy", "mae_vnd")}})
        else:
            wait += 1
        scheduler.step()
        _atomic_torch_save(last, {
            "state_dict": _cpu_weights(model), "metadata": metadata, "epoch": epoch,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "history": history,
            "best_key": list(best_key), "best_epoch": best_epoch, "wait": wait,
        })
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"{candidate_id} epoch {epoch:02d}: dev F1={scores['macro_f1']:.6f}, "
              f"MAE={scores['mae_vnd']:.2f}, best={best_epoch}, wait={wait}", flush=True)
    checkpoint = _load_checkpoint(folder / "best.pt", device)
    model.load_state_dict(checkpoint["state_dict"])
    scores, predictions, _ = evaluate_loader(model, dev_loader, dev, device)
    save_evaluation(folder, "dev", scores, predictions)
    history = pd.DataFrame(history)
    history.to_csv(folder / "history.csv", index=False)
    plot_history(history, folder / "learning_curves.png")
    pd.DataFrame(architecture_table(config)).to_csv(folder / "architecture.csv", index=False)
    summary = {"candidate_id": candidate_id, "best_epoch": best_epoch,
               "epochs_run": int(history.epoch.max()), "embedding_dim": config["channels"][-1],
               "parameters": sum(param.numel() for param in model.parameters()),
               **{k: scores[k] for k in ("accuracy", "macro_f1", "mae_vnd")},
               "train_seconds": float(history.train_seconds.sum()),
               "environment": environment(device)}
    save_json(folder / "summary.json", summary)
    artifacts = ["best.pt", "history.csv", "metrics_dev.json", "predictions_dev.csv",
                 "summary.json", "run_config.json"]
    save_json(folder / "run_complete.json", {"metadata": metadata,
              "sha256": {name: sha256(folder / name) for name in artifacts}})
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history, summary


def verify_lock(results_root, project_root):
    results_root = Path(results_root)
    data.assert_protocol(results_root, project_root)
    lock_path = results_root / "selection/locked_cnn_config.json"
    lock = json.loads(lock_path.read_text())
    for key, expected in lock["sha256"].items():
        base, relative = key.split("/", 1)
        path = (results_root if base == "results" else Path(project_root)) / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Locked file changed/missing: {path}")
    return lock


def select_and_lock(results_root, project_root):
    results_root = Path(results_root)
    p = data.assert_protocol(results_root, project_root)
    folder = results_root / "selection"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "locked_cnn_config.json").is_file():
        lock = verify_lock(results_root, project_root)
        return pd.read_csv(folder / "best_by_candidate_dev.csv"), lock
    rows, histories, hashed = [], [], ["inputs/experiment_config.json", "inputs/manifest_hashes.json",
                                       "inputs/train_manifest.csv", "inputs/dev_manifest.csv"]
    dev = pd.read_csv(results_root / "inputs/dev_manifest.csv")
    for order, config in enumerate(p["candidates"]):
        candidate_id = config["candidate_id"]
        run = results_root / "candidates" / candidate_id
        metadata = {"candidate": config, "training": p["training"],
                    "protocol_sha256": sha256(results_root / "inputs/experiment_config.json")}
        _verify_run(run, metadata)
        predictions = pd.read_csv(run / "predictions_dev.csv")
        same_manifest(dev, predictions.rename(columns={"y_true_vnd": "label_vnd"}))
        scores = score_predictions(predictions)
        summary = json.loads((run / "summary.json").read_text())
        for key in ("accuracy", "macro_f1", "mae_vnd"):
            if scores[key] != summary[key]:
                raise ValueError(f"Metric/prediction mismatch: {candidate_id}, {key}")
        hist = pd.read_csv(run / "history.csv", float_precision="round_trip")
        chosen = hist.sort_values(["dev_macro_f1", "dev_mae_vnd", "epoch"],
                                 ascending=[False, True, True]).iloc[0]
        if int(chosen.epoch) != summary["best_epoch"]:
            raise ValueError(f"Checkpoint epoch does not follow dev rule: {candidate_id}")
        histories.append(hist.assign(candidate_id=candidate_id, candidate_order=order))
        rows.append({"candidate_id": candidate_id, "candidate_order": order,
                     **{key: summary[key] for key in ("best_epoch", "epochs_run", "parameters",
                                                     "embedding_dim", "accuracy", "macro_f1", "mae_vnd")}})
        hashed.extend(f"candidates/{candidate_id}/{name}" for name in
                      ("run_complete.json", "best.pt", "summary.json", "history.csv", "run_config.json",
                       "metrics_dev.json", "predictions_dev.csv"))
    table = pd.DataFrame(rows).sort_values(["macro_f1", "mae_vnd", "candidate_order", "best_epoch"],
                                           ascending=[False, True, True, True]).reset_index(drop=True)
    table.to_csv(folder / "best_by_candidate_dev.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(folder / "all_epochs_dev.csv", index=False)
    hashed += ["selection/best_by_candidate_dev.csv", "selection/all_epochs_dev.csv"]
    best = table.iloc[0]
    config = next(c for c in p["candidates"] if c["candidate_id"] == best.candidate_id)
    lock = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": config, "best_epoch": int(best.best_epoch),
            "checkpoint": f"candidates/{best.candidate_id}/best.pt",
            "classes_vnd": CLASSES, "training_split": "train", "refit_train_plus_dev": False,
            "dev_metrics": {key: float(best[key]) for key in ("accuracy", "macro_f1", "mae_vnd")},
            "protocol": p, "environment_at_lock": environment(_device()),
            "sha256": {**{f"results/{name}": sha256(results_root / name) for name in hashed},
                       **{f"project/{name}": value for name, value in p["source_sha256"].items()}}}
    save_json(folder / "locked_cnn_config.json", lock)
    return table, lock


def load_locked_model(results_root, project_root, device=None):
    device = _device() if device is None else device
    lock = verify_lock(results_root, project_root)
    model = build_model(lock["candidate"]).to(device)
    checkpoint = _load_checkpoint(Path(results_root) / lock["checkpoint"], device)
    if checkpoint["epoch"] != lock["best_epoch"] or checkpoint["metadata"]["candidate"] != lock["candidate"]:
        raise ValueError("Checkpoint metadata does not match lock.")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, lock, device


def evaluate_test(data_root, results_root, project_root):
    results_root = Path(results_root)
    lock = verify_lock(results_root, project_root)
    lock_path = results_root / "selection/locked_cnn_config.json"
    folder = results_root / "test_final"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "run_info.json").is_file():
        record = json.loads((folder / "run_info.json").read_text())
        if record["lock_sha256"] != sha256(lock_path):
            raise ValueError("Test result belongs to another lock.")
        for name, value in record["sha256"].items():
            if sha256(folder / name) != value:
                raise ValueError(f"Test artifact changed: {name}")
        print("Đọc kết quả test đã lưu; không dự đoán lại.")
        return json.loads((folder / "metrics_test.json").read_text()), pd.read_csv(folder / "predictions_test.csv")
    model, lock, device = load_locked_model(results_root, project_root)
    settings = lock["protocol"]["training"]
    manifest = data.manifest_for_split(data_root, "test", lock["protocol"]["split_sizes"]["test"])
    loader = make_loader(manifest, data_root, lock["candidate"], False, settings, device)
    started = time.perf_counter()
    scores, predictions, loss = evaluate_loader(model, loader, manifest, device)
    seconds = time.perf_counter() - started
    save_evaluation(folder, "test", scores, predictions)
    manifest.to_csv(folder / "test_manifest.csv", index=False)
    shutil.copy2(lock_path, folder / "locked_cnn_config_used.json")
    pd.DataFrame([{"split": "dev", **lock["dev_metrics"]},
                  {"split": "test", **{k: scores[k] for k in
                                       ("accuracy", "macro_f1", "mae_vnd")}}]).to_csv(
        folder / "summary_test.csv", index=False)
    files = ["metrics_test.json", "predictions_test.csv", "test_manifest.csv", "locked_cnn_config_used.json"]
    (results_root / "README.md").write_text(
        "# Kết quả CNN RGB224\n\n"
        f"Đã đánh giá checkpoint {lock['candidate']['candidate_id']}, epoch {lock['best_epoch']}, "
        f"trên {len(manifest)} ảnh test. Accuracy: {scores['accuracy']:.6%}; "
        f"Macro-F1: {scores['macro_f1']:.6f}; MAE: {scores['mae_vnd']:.2f} VND/ảnh.\n\n"
        "Đọc `test_final/metrics_test.json` và `predictions_test.csv` để đối chiếu số gốc. "
        "`selection/locked_cnn_config.json` ghi cấu hình, trọng số và hash; test không refit.\n\n"
        "Notebook 08 bổ sung `error_analysis/`, `comparison/` và `report/`. Giữ checkpoint "
        "trong `candidates/` để xem feature map, Grad-CAM và tái sử dụng mô hình.\n",
        encoding="utf-8")
    save_json(folder / "run_info.json", {
        "lock_sha256": sha256(lock_path), "checkpoint_sha256": sha256(results_root / lock["checkpoint"]),
        "loaded_best_epoch": lock["best_epoch"], "refit": False, "test_used_for_tuning": False,
        "test_samples": len(manifest), "cross_entropy_test": loss,
        "inference_with_io_seconds": seconds, "environment": environment(device),
        "sha256": {name: sha256(folder / name) for name in files},
    })
    return scores, predictions
