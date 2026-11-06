"""Independent checks of saved SVM results, selection, report appendix and figures.

This script uses the standard library and does not fit SVC/PCA or extract features.
Large caches may be absent after cloning; present caches must match their hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLASSES = [10000, 20000, 50000, 100000, 200000, 500000]
FEATURES = ["HSV12", "HOG", "HOG_PCA", "HSV12_HOG", "HSV12_HOG_PCA"]
FOLDERS = ["hsv12", "hog", "hog_pca", "hsv12_hog", "hsv12_hog_pca"]
PARAMS = [("linear", 0.1, "scale"), ("linear", 1.0, "scale"), ("linear", 10.0, "scale"),
          ("rbf", 1.0, "scale"), ("rbf", 1.0, "10.0"), ("rbf", 10.0, "scale"), ("rbf", 10.0, "10.0")]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b, name):
    if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Mismatch {name}: {a} != {b}")


def manifests(path):
    return [(r["relative_path"], int(r["label_vnd"])) for r in rows(path)]


def verify_metrics(predictions, scores):
    matrix = [[0] * 6 for _ in CLASSES]
    gap = 0
    for row in predictions:
        actual, predicted = int(row["y_true_vnd"]), int(row["y_pred_vnd"])
        matrix[CLASSES.index(actual)][CLASSES.index(predicted)] += 1
        gap += abs(actual - predicted)
    if matrix != scores["confusion_matrix"] or gap != scores["total_gap_vnd"]:
        raise ValueError("Prediction/confusion/value-gap mismatch")
    if len(predictions) != scores["n_samples"] or scores["class_order_vnd"] != CLASSES:
        raise ValueError("Wrong sample count or class order")
    f1 = []
    for i, value in enumerate(CLASSES):
        support, predicted, tp = sum(matrix[i]), sum(r[i] for r in matrix), matrix[i][i]
        expected = {"denomination_vnd": value, "support": support, "predicted_count": predicted,
                    "tp": tp, "fp": predicted - tp, "fn": support - tp,
                    "precision": tp / predicted if predicted else 0,
                    "recall": tp / support if support else 0,
                    "f1": 2 * tp / (support + predicted) if support + predicted else 0}
        for key, number in expected.items():
            close(scores["per_class"][i][key], number, f"per class {value} {key}")
        f1.append(expected["f1"])
    close(scores["accuracy"], sum(matrix[i][i] for i in range(6)) / len(predictions), "accuracy")
    close(scores["macro_f1"], sum(f1) / 6, "macro_f1")
    close(scores["mae_vnd"], gap / len(predictions), "mae_vnd")


def key(row):
    return -float(row["macro_f1"]), float(row["mae_vnd"]), int(float(row["order"]))


def formatted(value, places):
    return f"{float(value):,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def verify(results_root, project_root=ROOT):
    result, project = Path(results_root).resolve(), Path(project_root).resolve()
    lock_path = result / "selection/locked_svm_config.json"
    lock = read_json(lock_path)
    verified, missing = 0, []
    for name, expected in lock["sha256"].items():
        kind, relative = name.split("/", 1)
        path = (result if kind == "results" else project) / relative
        if not path.exists() and path.suffix in {".npy", ".pkl"}:
            missing.append(name)
            continue
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Locked artifact changed/missing: {name}")
        verified += 1
    combined, summaries = [], []
    for order, (feature, folder_name) in enumerate(zip(FEATURES, FOLDERS)):
        folder = result / folder_name
        grid = rows(folder / "svm_grid_dev.csv")
        if len(grid) != 7 or {int(float(r["order"])) for r in grid} != set(range(7)):
            raise ValueError(f"Incomplete candidate grid: {feature}")
        for row in grid:
            index = int(float(row["order"]))
            kernel, c, gamma = PARAMS[index]
            if (row["kernel"], float(row["C"]), row["gamma"]) != (kernel, c, gamma):
                raise ValueError(f"Parameters differ from declared grid: {feature}, {index}")
            combined.append({**row, "feature_id": feature, "feature_order": order})
        best = min(grid, key=key)
        for split in ("train", "dev"):
            if manifests(folder / f"{split}_manifest.csv") != manifests(result / "inputs" / f"{split}_manifest.csv"):
                raise ValueError(f"Feature manifest mismatch: {feature}, {split}")
        predictions = rows(folder / "predictions_dev.csv")
        scores = read_json(folder / "metrics_dev.json")
        verify_metrics(predictions, scores)
        if [(r["relative_path"], int(r["y_true_vnd"])) for r in predictions] != manifests(result / "inputs/dev_manifest.csv"):
            raise ValueError(f"Dev prediction order mismatch: {feature}")
        saved = rows(folder / "svm_best_dev.csv")[0]
        for name in ("accuracy", "macro_f1", "mae_vnd"):
            close(scores[name], best[name], f"best grid {feature} {name}")
            close(saved[name], best[name], f"best CSV {feature} {name}")
        summaries.append({"feature_id": feature, "feature_order": order,
                          "correct_dev": sum(r["y_true_vnd"] == r["y_pred_vnd"] for r in predictions),
                          **{name: float(best[name]) for name in ("accuracy", "macro_f1", "mae_vnd")}})
    all_rows = rows(result / "selection/all_35_configurations_dev.csv")
    if len(all_rows) != 35:
        raise ValueError("Expected 35 combined rows")
    by_key = {(r["feature_id"], int(float(r["order"]))): r for r in combined}
    for row in all_rows:
        original = by_key[(row["feature_id"], int(float(row["order"])))]
        for name in ("accuracy", "macro_f1", "mae_vnd", "n_support_vectors"):
            close(row[name], original[name], "combined grid " + name)
    winner = min(combined, key=lambda r: (-float(r["macro_f1"]), float(r["mae_vnd"]), r["feature_order"], int(float(r["order"]))))
    if winner["feature_id"] != lock["feature_id"]:
        raise ValueError("Winner does not follow declared rule")
    if (winner["kernel"], float(winner["C"]), winner["gamma"]) != (lock["svm"]["kernel"], lock["svm"]["C"], str(lock["svm"]["gamma"])):
        raise ValueError("Locked SVC parameters differ from dev winner")
    for name in ("accuracy", "macro_f1", "mae_vnd"):
        close(lock["dev_metrics"][name], winner[name], "locked dev " + name)
    test = result / "test_final"
    scores = read_json(test / "metrics_test.json")
    predictions = rows(test / "predictions_test.csv")
    verify_metrics(predictions, scores)
    if [(r["relative_path"], int(r["y_true_vnd"])) for r in predictions] != manifests(test / "test_manifest.csv"):
        raise ValueError("Test prediction order mismatch")
    if lock_path.read_bytes() != (test / "locked_svm_config_used.json").read_bytes():
        raise ValueError("Test used another lock")
    run = read_json(test / "run_info.json")
    if run["lock_sha256"] != digest(lock_path) or run["test_used_for_tuning"] or lock["refit_train_plus_dev"]:
        raise ValueError("Test record violates locked protocol")
    knn = project / "src/KNN/results_224"
    for split in ("train", "dev", "test"):
        part = "test_final" if split == "test" else "inputs"
        if manifests(result / part / f"{split}_manifest.csv") != manifests(knn / part / f"{split}_manifest.csv"):
            raise ValueError(f"KNN/SVM manifest mismatch: {split}")
    baseline = rows(knn / "test_final/predictions_test.csv")
    baseline_scores = read_json(knn / "test_final/metrics_test.json")
    verify_metrics(baseline, baseline_scores)
    a = [r["y_true_vnd"] == r["y_pred_vnd"] for r in predictions]
    b = [r["y_true_vnd"] == r["y_pred_vnd"] for r in baseline]
    pairs = {"both_correct": sum(x and y for x, y in zip(a, b)),
             "only_svm_correct": sum(x and not y for x, y in zip(a, b)),
             "only_knn_correct": sum(not x and y for x, y in zip(a, b)),
             "both_wrong": sum(not x and not y for x, y in zip(a, b))}
    analysis = result / "error_analysis"
    decisions = rows(analysis / "all_test_decisions.csv")
    if len(decisions) != len(predictions):
        raise ValueError("Missing decision scores")
    for row, original in zip(decisions, predictions):
        for name in ("relative_path", "y_true_vnd", "y_pred_vnd"):
            if row[name] != original[name]:
                raise ValueError("Decision score order mismatch")
        ordered = sorted(float(row[f"score_{value}"]) for value in CLASSES)
        close(row["top_decision_score"], ordered[-1], "top decision score")
        close(row["score_gap"], ordered[-1] - ordered[-2], "decision gap")
        close(row["true_class_score"], row["score_" + row["y_true_vnd"]], "true-class score")
    errors = [row for row in decisions if row["y_true_vnd"] != row["y_pred_vnd"]]
    if rows(analysis / "all_errors.csv") != errors:
        raise ValueError("Error list differs from predictions")
    cost_pairs = rows(analysis / "error_pairs_bidirectional.csv")
    if sum(int(r["total_errors"]) for r in cost_pairs) != len(errors) or sum(int(r["total_gap_vnd"]) for r in cost_pairs) != scores["total_gap_vnd"]:
        raise ValueError("Error-pair totals do not match test")
    record = read_json(analysis / "analysis_record.json")
    if record["sha256"]["lock"] != digest(lock_path) or record["sha256"]["test_predictions"] != digest(test / "predictions_test.csv"):
        raise ValueError("Analysis belongs to another test/lock")
    if (test / "X_test.npy").is_file() and digest(test / "X_test.npy") != record["sha256"]["test_feature"]:
        raise ValueError("Analysis test feature changed")
    doc = project / "docs/SVM_REPORT"
    if (doc / "PHU_LUC_35_CAU_HINH_SVM.csv").read_bytes() != (result / "selection/all_35_configurations_dev.csv").read_bytes():
        raise ValueError("Report appendix differs from original grid")
    appendix = (doc / "BAO_CAO_THUC_NGHIEM_SVM.md").read_text().split("## Phụ lục A.", 1)[1]
    appendix_rows = [line for line in appendix.splitlines() if line.startswith("| ")][1:]
    if len(appendix_rows) != 35:
        raise ValueError("Markdown appendix needs 35 configurations")
    labels = {"HSV12": "HSV12", "HOG": "HOG", "HOG-PCA": "HOG_PCA", "HSV12+HOG": "HSV12_HOG", "HSV12+HOG-PCA": "HSV12_HOG_PCA"}
    for line in appendix_rows:
        fields = [s.strip() for s in line.strip("|").split("|")]
        feature, order, kernel, c, gamma, accuracy, f1, mae, support_vectors = fields
        original = by_key[(labels[feature], int(order) - 1)]
        expected_gamma = "—" if original["kernel"] == "linear" else "scale" if original["gamma"] == "scale" else formatted(original["gamma"], 1)
        expected = (original["kernel"], formatted(original["C"], 1), expected_gamma,
                    formatted(original["accuracy"], 6), formatted(original["macro_f1"], 6),
                    formatted(original["mae_vnd"], 2), original["n_support_vectors"])
        if (kernel.lower(), c, gamma, accuracy, f1, mae, support_vectors) != expected:
            raise ValueError(f"Markdown appendix number mismatch: {feature}, {order}")
    figures = read_json(doc / "SVM224_FIGURES/provenance.json")["figures"]
    for name, meta in figures.items():
        for path in (project / meta["source"], doc / "SVM224_FIGURES" / name):
            if digest(path) != meta["sha256"] or path.stat().st_size != meta["bytes"]:
                raise ValueError(f"Figure/provenance mismatch: {path}")
    return {"status": "verified", "source_commit": "6d87bc4792554b7e38ac065f9b7546b9b954bee6",
            "locked_files_verified": verified, "missing_large_caches": missing,
            "source_code_matches_lock": True, "dev_grid_rows": len(combined),
            "report_appendix_rows_verified": len(appendix_rows), "figure_sources_verified": len(figures),
            "selected_feature": lock["feature_id"], "selected_svc": lock["svm"], "dev_features": summaries,
            "test_correct": sum(a), "test_errors": len(errors), "accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"], "mae_vnd": scores["mae_vnd"], "total_gap_vnd": scores["total_gap_vnd"],
            "knn_train_dev_test_manifests_match": True, "paired_knn": pairs,
            "scope": "Saved evidence, code hashes and selection checked; no SVC/PCA refit or test inference rerun; source-image bytes across sessions not compared."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "src/SVM/results_224")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "validation.json")
    args = parser.parse_args()
    result = verify(args.results)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
