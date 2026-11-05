"""Đối chiếu báo cáo với evidence; không chạy lại thí nghiệm."""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.evaluation.value_metrics import evaluate_classification

REPORT = ROOT / "docs/KNN_REPORT"
RESULTS = ROOT / "src/KNN/results_224"
text = (REPORT / "BAO_CAO_THUC_NGHIEM_KNN.md").read_text(encoding="utf-8")
clean = text.replace("**", "")
evidence = []


def number(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def check(fragment, source, selector):
    assert fragment in clean, fragment
    evidence.append({"report_text": fragment, "source": source, "selector": selector})


best_path = "selection/best_by_feature_dev.csv"
best = pd.read_csv(RESULTS / best_path).set_index("feature_id")
reference = pd.read_csv(RESULTS / "selection/reference_dev.csv").set_index("feature_id")
lock = json.loads((RESULTS / "selection/locked_knn_config.json").read_text())
dims = {"HSV12": 1728, "HOG": 6084, "HOG_PCA": 1418,
        "HSV12_HOG": 7812, "HSV12_HOG_PCA": 3146}
names = {"HSV12": "HSV12", "HOG": "HOG", "HOG_PCA": "HOG-PCA",
         "HSV12_HOG": "HSV12+HOG", "HSV12_HOG_PCA": "HSV12+HOG-PCA"}
for key, row in best.iterrows():
    fragment = (f"| {names[key]} | {number(dims[key])} | {int(row.k)} / {row.weights} | "
                f"{number(row.accuracy * 100, 2)} | {number(row.macro_f1, 6)} | {number(row.mae_vnd, 2)} |")
    check(fragment, best_path, f"feature_id={key}; dimensions from pipeline/cache")
for key in reference.index:
    digits = 6
    check(number(reference.loc[key, "macro_f1"], digits), "selection/reference_dev.csv", f"feature_id={key}; macro_f1")

pred = pd.read_csv(RESULTS / "test_final/predictions_test.csv")
scores = evaluate_classification(pred.y_true_vnd, pred.y_pred_vnd)
assert scores == json.loads((RESULTS / "test_final/metrics_test.json").read_text())
check(f"| Test | {number(len(pred))} | {number(np.trace(scores['confusion_matrix']))} | "
      f"{number(scores['accuracy']*100, 2)} | {number(scores['macro_f1'], 6)} | {number(scores['mae_vnd'], 2)} |",
      "test_final/predictions_test.csv", "all rows; shared evaluate_classification")
dev = best.loc["HSV12"]
check(f"| Dev | 1.564 | 1.245 | {number(dev.accuracy*100, 2)} | {number(dev.macro_f1, 6)} | {number(dev.mae_vnd, 2)} |",
      best_path, "HSV12; count from dev confusion matrix")
dev_cm = pd.read_csv(RESULTS / "hsv12/hsv12_confusion_matrix_dev.csv", index_col=0).to_numpy()
assert dev_cm.sum() == 1564 and np.trace(dev_cm) == 1245

for row in scores["per_class"]:
    fragment = (f"| {number(row['denomination_vnd'])}đ | {row['tp']} / {row['support']} | "
                f"{number(row['precision'], 4)} | {number(row['recall'], 4)} | {number(row['f1'], 4)} |")
    check(fragment, "test_final/predictions_test.csv", f"shared metrics; class={row['denomination_vnd']}")

pairs = pd.read_csv(RESULTS / "error_analysis/error_pairs_bidirectional.csv")
pairs["gap"] = abs(pairs.class_a - pairs.class_b)
pairs["total_gap"] = pairs.total_errors * pairs.gap
for row in pairs.sort_values("total_gap", ascending=False).head(5).itertuples():
    fragment = (f"| {number(row.class_a)}đ ↔ {number(row.class_b)}đ | {row.total_errors} | "
                f"{number(row.gap)} | {number(row.total_gap)} |")
    check(fragment, "error_analysis/error_pairs_bidirectional.csv", f"pair={row.class_a},{row.class_b}; count*gap")
check(number(scores["total_gap_vnd"]), "test_final/predictions_test.csv", "sum absolute VND gaps")
check(number(100*(dev.accuracy-scores["accuracy"]), 2), best_path, "HSV12 accuracy minus test, percentage points")
check(number(dev.macro_f1-scores["macro_f1"], 6), best_path, "HSV12 macro_f1 minus test")
check(number(scores["mae_vnd"]-dev.mae_vnd, 2), best_path, "test MAE minus HSV12 dev")

pca = json.loads((RESULTS / "hog_pca/run_config.json").read_text())
assert pca["pca_dim"] == 1418 and pca["original_dim"] == 6084
check(number(pca["explained_variance_ratio_sum"]*100, 4), "hog_pca/run_config.json", "explained_variance_ratio_sum*100")
check(number((1-1418/6084)*100, 2), "hog_pca/run_config.json", "dimension reduction percent")

cases = pd.read_csv(RESULTS / "error_analysis/selected_cases.csv").set_index("case_id")
for key, digits in [("C04", 6), ("C09", 6), ("C10", 8), ("C11", 8)]:
    row = cases.loc[key]
    contrast = row.other_distance if row.correct else row.same_distance
    for column, value in [("nearest_distance", row.nearest_distance), ("contrast_distance", contrast)]:
        check(number(value, digits), "error_analysis/selected_cases.csv", f"case_id={key}; {column}")

images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
assert len(images) == 6
assert not re.search(r"^\*Hình KNN-\d+\.", text, re.M)
assert all((REPORT / image).is_file() for image in images)
assert text.count("\\[") == text.count("\\]") == 5
assert "test hiện tại đã có lịch sử được xem" in text
assert "không phải độ đo chính" in text
assert "HSV8" not in text and "Color Moments" not in text and "Bray–Curtis" not in text
assert len(re.findall(r"\*\*Bảng KNN-\d\.", text)) == 5

def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


provenance = json.loads((REPORT / "KNN224_FIGURES/provenance.json").read_text())
for category in ["sources", "outputs"]:
    for path, expected in provenance[category].items():
        assert sha(ROOT / path) == expected, path
for name, expected in lock["sha256"].items():
    assert sha(ROOT / name) == expected, name

record = {"status": "PASS", "format": "course-report-section-markdown",
          "numeric_evidence": evidence, "embedded_figures": images,
          "source_hashes": {name: sha(RESULTS / name) for name in [best_path,
              "selection/reference_dev.csv", "test_final/predictions_test.csv",
              "hog_pca/run_config.json", "error_analysis/selected_cases.csv"]},
          "report_sha256": sha(REPORT / "BAO_CAO_THUC_NGHIEM_KNN.md"),
          "limits": ["Markdown and six figures checked; no final Word/PDF page-layout check",
                     "No grid rerun or new model training", "Single reused test split; no causal visual ablations"]}
(REPORT / "review/verification_224.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"PASS: {len(evidence)} numeric checks; six image links; source and figure hashes.")
