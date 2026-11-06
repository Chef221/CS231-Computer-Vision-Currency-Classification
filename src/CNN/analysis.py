"""Inspect learned representations and saved mistakes without tuning on test."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src.CNN.data import (CLASSES, CLASS_TO_IDX, make_loader, image_tensor,
                          same_manifest, save_json, sha256, seed_everything, augment_rgb)
from src.CNN.pipeline import load_locked_model, score_predictions, evaluate_test
from src.preprocessing import read_rgb224


def augmentation_examples(data_root, results_root):
    folder = Path(results_root) / "inputs"
    train = pd.read_csv(folder / "train_manifest.csv")
    seed_everything(42)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, label in enumerate((10000, 500000)):
        image = read_rgb224(Path(data_root) / train[train.label_vnd == label].relative_path.iloc[0])
        for col in range(4):
            axes[row, col].imshow(image if col == 0 else augment_rgb(image))
            axes[row, col].set_title(f"{label:,}: " + ("RGB224" if col == 0 else f"augmentation {col}"))
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(folder / "augmentation_examples.png", dpi=180)
    plt.close(fig)


def inspect_features(data_root, results_root, project_root):
    results_root = Path(results_root)
    folder = results_root / "feature_analysis"
    folder.mkdir(parents=True, exist_ok=True)
    model, lock, device = load_locked_model(results_root, project_root)
    config, settings = lock["candidate"], lock["protocol"]["training"]
    frames, matrices = {}, {}
    for split in ("train", "dev"):
        frames[split] = pd.read_csv(results_root / "inputs" / f"{split}_manifest.csv")
        loader = make_loader(frames[split], data_root, config, False, settings, device)
        parts = []
        with torch.inference_mode():
            for images, _ in loader:
                parts.append(model.embed(images.to(device)).cpu().numpy())
        matrices[split] = np.concatenate(parts)
        np.save(folder / f"embeddings_{split}.npy", matrices[split])
        frames[split].to_csv(folder / f"{split}_manifest.csv", index=False)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(matrices["train"])
    projected = pca.transform(matrices["dev"])
    coords = frames["dev"].assign(pc1=projected[:, 0], pc2=projected[:, 1])
    coords.to_csv(folder / "pca_dev_coordinates.csv", index=False)
    np.savez(folder / "pca_train_projection.npz", components=pca.components_, mean=pca.mean_)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASSES:
        sample = coords[coords.label_vnd == label]
        ax.scatter(sample.pc1, sample.pc2, s=8, alpha=0.5, label=f"{label // 1000}k")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} train variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} train variance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "embedding_pca_dev.png", dpi=180)
    plt.close(fig)
    relative = frames["train"].relative_path.iloc[0]
    rgb = read_rgb224(Path(data_root) / relative)
    tensor = image_tensor(rgb, config["color"]).unsqueeze(0).to(device)
    stages, x = [], tensor
    with torch.inference_mode():
        for block in model.blocks:
            x = block(x)
            stages.append(x[0].cpu().numpy())
    fig, axes = plt.subplots(2, 7, figsize=(17, 5))
    selected_channels = []
    for row, stage_index in enumerate((0, len(stages) - 1)):
        maps = stages[stage_index]
        indices = np.argsort(-maps.mean(axis=(1, 2)), kind="stable")[:6]
        selected_channels.append({"block": stage_index + 1, "channels": indices.tolist()})
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Train input; block {stage_index + 1}")
        axes[row, 0].axis("off")
        for col, channel in enumerate(indices, 1):
            axes[row, col].imshow(maps[channel], cmap="magma")
            axes[row, col].set_title(f"channel {channel}; {maps.shape[1]}×{maps.shape[2]}")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(folder / "feature_maps.png", dpi=180)
    plt.close(fig)
    save_json(folder / "analysis_record.json", {
        "candidate_id": config["candidate_id"], "fit_split_pca": "train", "plot_split": "dev",
        "embedding_dim": int(matrices["train"].shape[1]),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_map_example": relative, "selected_channels": selected_channels,
        "channel_selection": "six highest spatial mean activations per displayed block",
        "lock_sha256": sha256(results_root / "selection/locked_cnn_config.json"),
        "interpretation": "PCA2 is descriptive; channel semantics and causal importance are not established.",
    })
    return coords, pd.DataFrame(selected_channels)


def gradcam(model, tensor, class_index):
    """Grad-CAM on the last block's second ReLU, before its spatial max pool."""
    model.eval()
    activations = []
    handle = model.blocks[-1][5].register_forward_hook(lambda module, args, out: activations.append(out))
    try:
        with torch.enable_grad():
            logits = model(tensor)
            maps = activations[0]
            gradients, = torch.autograd.grad(logits[0, class_index], maps)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * maps).sum(1).relu()[0].detach().cpu().numpy()
        maximum = float(cam.max())
        if maximum > 0:
            cam = cam / maximum
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam, maximum == 0, logits.detach().softmax(1)[0].cpu().numpy()
    finally:
        handle.remove()


def _overlay(rgb, cam):
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip(0.60 * rgb.astype(np.float32) / 255 + 0.40 * heatmap, 0, 1)


def select_cases(predictions, limit=12):
    selected, seen = [], set()
    errors = predictions[predictions.y_true_vnd != predictions.y_pred_vnd].copy()

    def take(frame, reason, count=1):
        added = 0
        for _, row in frame.iterrows():
            if row.relative_path not in seen and len(selected) < limit:
                selected.append({**row.to_dict(), "selection_reason": reason})
                seen.add(row.relative_path)
                added += 1
                if added == count:
                    break

    directed = errors.groupby(["y_true_vnd", "y_pred_vnd"], as_index=False).agg(
        count=("relative_path", "size"), total_gap_vnd=("gap_vnd", "sum"))
    for _, pair in directed.sort_values(["count", "y_true_vnd", "y_pred_vnd"],
                                        ascending=[False, True, True]).head(4).iterrows():
        subset = errors[(errors.y_true_vnd == pair.y_true_vnd) & (errors.y_pred_vnd == pair.y_pred_vnd)]
        take(subset.sort_values("relative_path"), "frequent_directed_pair")
    for _, pair in directed.sort_values(["total_gap_vnd", "y_true_vnd", "y_pred_vnd"],
                                        ascending=[False, True, True]).head(3).iterrows():
        subset = errors[(errors.y_true_vnd == pair.y_true_vnd) & (errors.y_pred_vnd == pair.y_pred_vnd)]
        take(subset.sort_values(["top_softmax", "relative_path"], ascending=[False, True]), "large_vnd_gap")
    take(errors.sort_values(["top_softmax", "relative_path"], ascending=[False, True]), "high_softmax_error")
    take(errors.sort_values(["softmax_gap", "relative_path"]), "small_softmax_gap_error")
    correct = predictions[predictions.y_true_vnd == predictions.y_pred_vnd]
    take(correct.sort_values(["softmax_gap", "relative_path"]), "small_softmax_gap_correct")
    take(correct.sort_values(["top_softmax", "relative_path"], ascending=[False, True]), "high_softmax_correct")
    take(errors.sort_values("relative_path"), "additional_error", count=limit)
    take(correct.sort_values("relative_path"), "additional_correct", count=limit)
    cases = pd.DataFrame(selected)
    cases.insert(0, "case_id", [f"C{index:02d}" for index in range(1, len(cases) + 1)])
    return cases


def analyze_errors(data_root, results_root, project_root):
    results_root = Path(results_root)
    scores, predictions = evaluate_test(data_root, results_root, project_root)
    model, lock, device = load_locked_model(results_root, project_root)
    folder = results_root / "error_analysis"
    folder.mkdir(parents=True, exist_ok=True)
    errors = predictions[predictions.y_true_vnd != predictions.y_pred_vnd].copy()
    errors.to_csv(folder / "all_errors.csv", index=False)
    directed = pd.DataFrame(scores["error_pairs"], columns=["true_vnd", "predicted_vnd", "count", "gap_vnd", "total_gap_vnd"])
    directed.sort_values(["total_gap_vnd", "true_vnd", "predicted_vnd"],
                          ascending=[False, True, True]).to_csv(folder / "error_pairs_cost.csv", index=False)
    combined = {}
    for row in scores["error_pairs"]:
        pair = tuple(sorted((row["true_vnd"], row["predicted_vnd"])))
        record = combined.setdefault(pair, {"lower_vnd": pair[0], "higher_vnd": pair[1],
                                            "count": 0, "total_gap_vnd": 0})
        record["count"] += row["count"]
        record["total_gap_vnd"] += row["total_gap_vnd"]
    pd.DataFrame(list(combined.values()), columns=["lower_vnd", "higher_vnd", "count", "total_gap_vnd"]).sort_values(
        ["count", "lower_vnd", "higher_vnd"], ascending=[False, True, True]).to_csv(
            folder / "error_pairs_bidirectional.csv", index=False)
    cases = select_cases(predictions)
    cases.to_csv(folder / "selected_cases.csv", index=False)
    records = []
    for start in range(0, len(cases), 4):
        batch = cases.iloc[start:start + 4]
        fig, axes = plt.subplots(len(batch), 3, figsize=(12, 3.2 * len(batch)), squeeze=False)
        for row_index, row in enumerate(batch.itertuples()):
            rgb = read_rgb224(Path(data_root) / row.relative_path)
            tensor = image_tensor(rgb, lock["candidate"]["color"]).unsqueeze(0).to(device)
            pred_cam, pred_zero, probability = gradcam(model, tensor, CLASS_TO_IDX[row.y_pred_vnd])
            true_cam, true_zero, _ = gradcam(model, tensor, CLASS_TO_IDX[row.y_true_vnd])
            if CLASSES[int(probability.argmax())] != row.y_pred_vnd:
                raise ValueError("Grad-CAM inference does not match saved test prediction.")
            if lock["candidate"]["color"] == "gray":
                rgb = np.repeat(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
            axes[row_index, 0].imshow(rgb)
            axes[row_index, 0].set_title(f"{row.case_id}: true {row.y_true_vnd:,}; pred {row.y_pred_vnd:,}")
            axes[row_index, 1].imshow(_overlay(rgb, pred_cam))
            axes[row_index, 1].set_title(f"Pred CAM; softmax={row.top_softmax:.3f}")
            axes[row_index, 2].imshow(_overlay(rgb, true_cam))
            axes[row_index, 2].set_title(f"True-class CAM; score gap={row.softmax_gap:.3f}")
            for ax in axes[row_index]:
                ax.axis("off")
            records.append({"case_id": row.case_id, "pred_cam_all_zero": pred_zero,
                            "true_cam_all_zero": true_zero})
        fig.tight_layout()
        fig.savefig(folder / f"cases_page_{start // 4 + 1:02d}.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(records).to_csv(folder / "gradcam_checks.csv", index=False)
    notes = cases[["case_id", "relative_path", "y_true_vnd", "y_pred_vnd", "selection_reason"]].copy()
    for column in ("observation", "hypothesis", "not_verified"):
        notes[column] = ""
    if not (folder / "visual_review_notes.csv").is_file():
        notes.to_csv(folder / "visual_review_notes.csv", index=False)
    save_json(folder / "analysis_record.json", {
        "candidate_id": lock["candidate"]["candidate_id"], "cases": len(cases),
        "errors": len(errors), "total_gap_vnd": scores["total_gap_vnd"],
        "lock_sha256": sha256(results_root / "selection/locked_cnn_config.json"),
        "predictions_sha256": sha256(results_root / "test_final/predictions_test.csv"),
        "cam_layer": "last block second ReLU, before MaxPool",
        "cam_scaling": "ReLU then independent max normalization per image and target; zero maps preserved",
        "softmax": "uncalibrated; not a guarantee of correctness",
    })
    return cases, directed


def compare_baselines(results_root, reference_root):
    results_root, reference_root = Path(results_root), Path(reference_root)
    provenance = json.loads((reference_root / "provenance.json").read_text())
    for relative, digest in provenance["sha256"].items():
        if sha256(reference_root / relative) != digest:
            raise ValueError(f"Baseline reference changed: {relative}")
    cnn = pd.read_csv(results_root / "test_final/predictions_test.csv")
    cnn_scores = score_predictions(cnn)
    comparisons = [{"model": "CNN", **{k: cnn_scores[k] for k in
                                        ("n_samples", "accuracy", "macro_f1", "mae_vnd", "total_gap_vnd")}}]
    paired, per_class = [], []
    for name, relative in provenance["baselines"].items():
        base = reference_root / relative["folder"]
        for split in ("train", "dev", "test"):
            current = results_root / ("test_final" if split == "test" else "inputs") / f"{split}_manifest.csv"
            same_manifest(pd.read_csv(current), pd.read_csv(base / f"{split}_manifest.csv"))
        baseline = pd.read_csv(base / "predictions_test.csv")
        same_manifest(cnn.rename(columns={"y_true_vnd": "label_vnd"}),
                      baseline.rename(columns={"y_true_vnd": "label_vnd"}))
        baseline_scores = score_predictions(baseline)
        stored = json.loads((base / "metrics_test.json").read_text())
        matching_scalars = all(np.isclose(baseline_scores[key], stored[key], rtol=1e-12, atol=1e-12)
                               for key in ("accuracy", "macro_f1", "mae_vnd"))
        if (not matching_scalars or baseline_scores["confusion_matrix"] != stored["confusion_matrix"]
                or baseline_scores["total_gap_vnd"] != stored["total_gap_vnd"]):
            raise ValueError(f"Baseline metrics/predictions mismatch: {name}")
        comparisons.append({"model": name, **{k: stored[k] for k in
                            ("n_samples", "accuracy", "macro_f1", "mae_vnd", "total_gap_vnd")}})
        a = cnn.y_true_vnd.to_numpy() == cnn.y_pred_vnd.to_numpy()
        b = baseline.y_true_vnd.to_numpy() == baseline.y_pred_vnd.to_numpy()
        paired.append({"baseline": name, "both_correct": int((a & b).sum()),
                       "only_cnn_correct": int((a & ~b).sum()),
                       "only_baseline_correct": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum())})
        for cnn_class, base_class in zip(cnn_scores["per_class"], stored["per_class"]):
            per_class.append({"baseline": name, "denomination_vnd": cnn_class["denomination_vnd"],
                              "cnn_f1": cnn_class["f1"], "baseline_f1": base_class["f1"],
                              "difference": cnn_class["f1"] - base_class["f1"]})
    folder = results_root / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(comparisons)
    table.to_csv(folder / "metrics_comparison_test.csv", index=False)
    pd.DataFrame(paired).to_csv(folder / "paired_summary_test.csv", index=False)
    pd.DataFrame(per_class).to_csv(folder / "per_class_comparison_test.csv", index=False)
    save_json(folder / "comparison_record.json", {
        "paired_manifest_check": "train/dev/test paths, VND labels and row order all match",
        "cnn_predictions_sha256": sha256(results_root / "test_final/predictions_test.csv"),
        "reference_provenance_sha256": sha256(reference_root / "provenance.json"),
        "baseline_sources": provenance["baselines"],
        "limitation": "CNN may use augmentation and learned features; hardware/libraries differ; one seed and split.",
    })
    return table, pd.DataFrame(paired)
