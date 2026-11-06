"""Verify the saved CNN experiment and baseline evidence without training or inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("cnn_report", ROOT / "docs/CNN_REPORT/build_report_224.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def close(actual, expected, name):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name}: {actual} != {expected}")


def verify_per_class(metrics):
    matrix = metrics["confusion_matrix"]
    for index, value in enumerate(report.CLASSES):
        row = metrics["per_class"][index]
        support, predicted, tp = sum(matrix[index]), sum(r[index] for r in matrix), matrix[index][index]
        expected = {"denomination_vnd": value, "support": support, "predicted_count": predicted,
                    "tp": tp, "fp": predicted - tp, "fn": support - tp,
                    "precision": tp / predicted if predicted else 0,
                    "recall": tp / support if support else 0,
                    "f1": 2 * tp / (support + predicted) if support + predicted else 0}
        for name, number in expected.items():
            close(row[name], number, f"per-class {value} {name}")


def verify_prediction_columns(rows):
    for row in rows:
        actual, predicted = int(row["y_true_vnd"]), int(row["y_pred_vnd"])
        if "gap_vnd" in row:
            close(row["gap_vnd"], abs(actual - predicted), "image gap")
        if "correct" in row and (row["correct"].lower() == "true") != (actual == predicted):
            raise ValueError("Image correct flag disagrees with VND labels")
        if "top_softmax" in row:
            probabilities = [float(row[f"softmax_{value}"]) for value in report.CLASSES]
            if not all(0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
                raise ValueError("Invalid softmax row")
            if report.CLASSES[max(range(6), key=lambda i: probabilities[i])] != predicted:
                raise ValueError("Softmax argmax disagrees with prediction")
            ordered = sorted(probabilities)
            if (not math.isclose(float(row["top_softmax"]), ordered[-1], abs_tol=1e-6)
                    or not math.isclose(float(row["softmax_gap"]), ordered[-1] - ordered[-2], abs_tol=1e-6)):
                raise ValueError("Softmax summary disagrees with class scores")


def verify(results_root, project_root=ROOT):
    results_root, project_root = Path(results_root).resolve(), Path(project_root).resolve()
    lock, metrics, predictions, count, missing = report._verify_measured(results_root, project_root)
    stage = report.read_json(results_root / "stage_completed.json")
    if stage != {"model": "CNN", "stage": "08"}:
        raise ValueError(f"Expected completed notebook 08, found {stage}")
    verify_per_class(metrics)
    verify_prediction_columns(predictions)
    protocol = report.read_json(results_root / "inputs/experiment_config.json")
    candidate_rows, histories, timing, run_hashes = [], {}, [], 0
    source_protocol_hash = report.digest(results_root / "inputs/experiment_config.json")
    for order, config in enumerate(protocol["candidates"]):
        candidate = config["candidate_id"]
        folder = results_root / "candidates" / candidate
        complete = report.read_json(folder / "run_complete.json")
        expected_metadata = {"candidate": config, "training": protocol["training"],
                             "protocol_sha256": source_protocol_hash}
        if complete["metadata"] != expected_metadata:
            raise ValueError(f"Run protocol mismatch: {candidate}")
        for name, expected in complete["sha256"].items():
            path = folder / name
            if not path.exists() and path.suffix == ".pt":
                continue
            if report.digest(path) != expected:
                raise ValueError(f"Run hash mismatch: {path}")
            run_hashes += 1
        history = report.read_csv(folder / "history.csv")
        if [int(r["epoch"]) for r in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Missing/duplicate history epochs: {candidate}")
        if len(history) > protocol["training"]["max_epochs"]:
            raise ValueError("More epochs than declared")
        selected = min(history, key=lambda r: (-float(r["dev_macro_f1"]), float(r["dev_mae_vnd"]), int(r["epoch"])))
        summary = report.read_json(folder / "summary.json")
        if summary["best_epoch"] != int(selected["epoch"]) or summary["epochs_run"] != len(history):
            raise ValueError(f"Incorrect checkpoint epoch: {candidate}")
        scores = report.read_json(folder / "metrics_dev.json")
        pred = report.read_csv(folder / "predictions_dev.csv")
        report.verify_scores(pred, scores)
        verify_per_class(scores)
        verify_prediction_columns(pred)
        if [(r["relative_path"], int(r["y_true_vnd"])) for r in pred] != report.manifest_rows(results_root / "inputs/dev_manifest.csv"):
            raise ValueError(f"Dev order mismatch: {candidate}")
        for name in ("accuracy", "macro_f1", "mae_vnd"):
            close(summary[name], scores[name], f"summary {candidate} {name}")
            close(selected["dev_" + name], scores[name], f"history {candidate} {name}")
        close(summary["parameters"], sum(int(r["parameters"]) for r in report.read_csv(folder / "architecture.csv")), "parameter count")
        close(summary["embedding_dim"], config["channels"][-1], "embedding dimension")
        train_seconds = sum(float(r["train_seconds"]) for r in history)
        dev_seconds = sum(float(r["dev_seconds"]) for r in history)
        close(summary["train_seconds"], train_seconds, "training time")
        timing.append({"candidate_id": candidate, "epochs": len(history), "best_epoch": summary["best_epoch"],
                       "train_seconds": train_seconds, "dev_seconds": dev_seconds,
                       "train_plus_dev_seconds": train_seconds + dev_seconds})
        candidate_rows.append({"candidate_id": candidate, "order": order, **summary})
        histories[candidate] = {int(row["epoch"]): row for row in history}
    winner = min(candidate_rows, key=lambda r: (-r["macro_f1"], r["mae_vnd"], r["order"], r["best_epoch"]))
    if winner["candidate_id"] != lock["candidate"]["candidate_id"] or winner["best_epoch"] != lock["best_epoch"]:
        raise ValueError("Global selection violates declared rule")
    grid = report.read_csv(results_root / "selection/best_by_candidate_dev.csv")
    if {r["candidate_id"] for r in grid} != set(histories):
        raise ValueError("Incomplete dev candidate table")
    by_id = {row["candidate_id"]: row for row in candidate_rows}
    for row in grid:
        candidate = row["candidate_id"]
        for key in ("best_epoch", "epochs_run", "parameters", "embedding_dim", "accuracy", "macro_f1", "mae_vnd"):
            close(row[key], by_id[candidate][key], f"dev table {candidate} {key}")
    all_epochs = report.read_csv(results_root / "selection/all_epochs_dev.csv")
    if len(all_epochs) != sum(len(value) for value in histories.values()):
        raise ValueError("Combined history length mismatch")
    for row in all_epochs:
        original = histories[row["candidate_id"]][int(row["epoch"])]
        for key in original:
            close(row[key], original[key], f"combined history {key}")
    test_record = report.read_json(results_root / "test_final/run_info.json")
    if (test_record["loaded_best_epoch"] != lock["best_epoch"] or test_record["refit"]
            or test_record["test_used_for_tuning"]):
        raise ValueError("Test record violates locked protocol")
    checkpoint = results_root / lock["checkpoint"]
    if checkpoint.is_file() and report.digest(checkpoint) != test_record["checkpoint_sha256"]:
        raise ValueError("Test checkpoint hash mismatch")
    refs = project_root / "src/CNN/reference_results"
    provenance = report.read_json(refs / "provenance.json")
    for relative, expected in provenance["sha256"].items():
        if report.digest(refs / relative) != expected:
            raise ValueError(f"Baseline hash mismatch: {relative}")
    comparison = {row["model"]: row for row in report.read_csv(results_root / "comparison/metrics_comparison_test.csv")}
    paired_saved = {row["baseline"]: row for row in report.read_csv(results_root / "comparison/paired_summary_test.csv")}
    paired, all_metrics, wrong_sets = {}, {"CNN": metrics}, []
    cnn_correct = [r["y_true_vnd"] == r["y_pred_vnd"] for r in predictions]
    wrong_sets.append({r["relative_path"] for r, correct in zip(predictions, cnn_correct) if not correct})
    per_class = report.read_csv(results_root / "comparison/per_class_comparison_test.csv")
    for model, record in provenance["baselines"].items():
        base = refs / record["folder"]
        scores = report.read_json(base / "metrics_test.json")
        pred = report.read_csv(base / "predictions_test.csv")
        report.verify_scores(pred, scores)
        verify_per_class(scores)
        all_metrics[model] = scores
        for split in ("train", "dev", "test"):
            current = results_root / ("test_final" if split == "test" else "inputs") / f"{split}_manifest.csv"
            if report.manifest_rows(current) != report.manifest_rows(base / f"{split}_manifest.csv"):
                raise ValueError(f"Manifest mismatch: {model} {split}")
        b = [r["y_true_vnd"] == r["y_pred_vnd"] for r in pred]
        wrong_sets.append({r["relative_path"] for r, correct in zip(pred, b) if not correct})
        paired[model] = {"both_correct": sum(x and y for x, y in zip(cnn_correct, b)),
                         "only_cnn_correct": sum(x and not y for x, y in zip(cnn_correct, b)),
                         "only_baseline_correct": sum(not x and y for x, y in zip(cnn_correct, b)),
                         "both_wrong": sum(not x and not y for x, y in zip(cnn_correct, b))}
        for key, value in paired[model].items():
            close(paired_saved[model][key], value, f"paired {model} {key}")
        class_rows = [r for r in per_class if r["baseline"] == model]
        if len(class_rows) != 6:
            raise ValueError("Missing per-class comparison")
        for row, cnn_class, baseline_class in zip(class_rows, metrics["per_class"], scores["per_class"]):
            close(row["denomination_vnd"], cnn_class["denomination_vnd"], "class comparison label")
            close(row["cnn_f1"], cnn_class["f1"], "class comparison CNN F1")
            close(row["baseline_f1"], baseline_class["f1"], "class comparison baseline F1")
            close(row["difference"], cnn_class["f1"] - baseline_class["f1"], "class F1 difference")
    for model, scores in all_metrics.items():
        for key in ("n_samples", "accuracy", "macro_f1", "mae_vnd", "total_gap_vnd"):
            close(comparison[model][key], scores[key], f"comparison {model} {key}")
    feature = report.read_json(results_root / "feature_analysis/analysis_record.json")
    errors = report.read_json(results_root / "error_analysis/analysis_record.json")
    lock_hash = report.digest(results_root / "selection/locked_cnn_config.json")
    if feature["lock_sha256"] != lock_hash or errors["lock_sha256"] != lock_hash:
        raise ValueError("Analysis belongs to another lock")
    if errors["predictions_sha256"] != report.digest(results_root / "test_final/predictions_test.csv"):
        raise ValueError("Analysis prediction hash mismatch")
    wrong = [row for row in predictions if row["y_true_vnd"] != row["y_pred_vnd"]]
    if report.read_csv(results_root / "error_analysis/all_errors.csv") != wrong:
        raise ValueError("Error image table does not match test predictions")
    bidirectional = report.read_csv(results_root / "error_analysis/error_pairs_bidirectional.csv")
    if sum(int(row["count"]) for row in bidirectional) != len(wrong):
        raise ValueError("Error-pair counts mismatch")
    if sum(int(row["total_gap_vnd"]) for row in bidirectional) != metrics["total_gap_vnd"]:
        raise ValueError("Error-pair total VND mismatch")
    cases = report.read_csv(results_root / "error_analysis/selected_cases.csv")
    original = {row["relative_path"]: row for row in predictions}
    if len({row["relative_path"] for row in cases}) != len(cases):
        raise ValueError("Duplicate selected case")
    for row in cases:
        for key in original[row["relative_path"]]:
            if row[key] != original[row["relative_path"]][key]:
                raise ValueError(f"Case differs from test prediction: {row['case_id']} {key}")
    return {"status": "verified", "lock_sha256": lock_hash,
            "locked_files_verified": count, "missing_large_checkpoints": missing,
            "run_complete_files_verified": run_hashes, "baseline_files_verified": len(provenance["sha256"]),
            "source_code_matches_lock": True, "train_dev_test_manifests_match_both_baselines": True,
            "selected_candidate": winner["candidate_id"], "selected_epoch": winner["best_epoch"],
            "correct_test_images": sum(cnn_correct), "test_errors": len(wrong),
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "mae_vnd": metrics["mae_vnd"],
            "total_gap_vnd": metrics["total_gap_vnd"], "candidate_timing": timing, "paired": paired,
            "all_three_models_wrong": len(set.intersection(*wrong_sets)),
            "scope": "Saved evidence and metadata verified; no retraining or GPU inference rerun; image bytes not compared across original training sessions."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "src/CNN/results_224")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "validation.json")
    args = parser.parse_args()
    record = verify(args.results)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
