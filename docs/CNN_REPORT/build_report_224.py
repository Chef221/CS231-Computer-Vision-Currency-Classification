"""Render a draft or measured CNN report from saved evidence; no training dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = [10000, 20000, 50000, 100000, 200000, 500000]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value, digits=0):
    return f"{float(value):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def signed(value, digits=0):
    return ("+" if float(value) > 0 else "") + number(value, digits)


def table(header, rows):
    rows = list(rows)
    alignment = []
    for index in range(len(header)):
        cells = [str(row[index]).replace("**", "").strip() for row in rows]
        numeric = bool(cells) and all(cell == "—" or re.fullmatch(r"[+\-]?\d[\d.,]*(?:%|đ)?", cell) for cell in cells)
        alignment.append("---:" if numeric else "---")
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(alignment) + "|",
                      *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]])


def scores_from_csv(rows):
    if not rows:
        raise ValueError("Empty predictions")
    matrix = [[0] * 6 for _ in CLASSES]
    total_gap = 0
    for row in rows:
        actual, predicted = int(row["y_true_vnd"]), int(row["y_pred_vnd"])
        matrix[CLASSES.index(actual)][CLASSES.index(predicted)] += 1
        total_gap += abs(actual - predicted)
    f1 = []
    for i in range(6):
        support, predicted_count = sum(matrix[i]), sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / (support + predicted_count) if support + predicted_count else 0)
    return {"n_samples": len(rows), "accuracy": sum(matrix[i][i] for i in range(6)) / len(rows),
            "macro_f1": sum(f1) / 6, "mae_vnd": total_gap / len(rows),
            "total_gap_vnd": total_gap, "confusion_matrix": matrix}


def verify_scores(rows, metrics):
    computed = scores_from_csv(rows)
    for key in ("accuracy", "macro_f1", "mae_vnd"):
        if not math.isclose(computed[key], metrics[key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Prediction/metric mismatch: {key}")
    for key in ("n_samples", "total_gap_vnd", "confusion_matrix"):
        if computed[key] != metrics[key]:
            raise ValueError(f"Prediction/metric mismatch: {key}")


def manifest_rows(path):
    return [(row["relative_path"], int(row["label_vnd"])) for row in read_csv(path)]


def candidate_commentary(grid, results_root):
    rows = {r["candidate_id"]: r for r in grid}
    paragraphs = []
    titles = {"C01_RGB": "RGB: mốc CNN cơ sở", "C02_GRAY": "Ảnh xám: bỏ thông tin màu",
              "C03_RGB_AUG": "RGB có augmentation", "C04_DEEP_AUG": "Thêm khối tích chập 256 kênh"}
    for index, row in enumerate(sorted(grid, key=lambda r: int(r["candidate_order"])), 1):
        candidate = row["candidate_id"]
        predictions = read_csv(results_root / "candidates" / candidate / "predictions_dev.csv")
        correct = sum(p["y_true_vnd"] == p["y_pred_vnd"] for p in predictions)
        text = (f"### 4.{index}. {titles.get(candidate, candidate)}\n\n"
                f"{candidate} đúng **{number(correct)}/{number(len(predictions))} ảnh dev**, "
                f"đạt Accuracy **{number(float(row['accuracy']) * 100, 2)}%**, "
                f"Macro-F1 **{number(row['macro_f1'], 6)}**, MAE **{number(row['mae_vnd'], 2)}đ/ảnh** "
                f"ở epoch **{row['best_epoch']}**. ")
        if candidate == "C01_RGB":
            text += "Đây là mốc mạng ba khối nhận RGB và không augmentation. Kết quả thể hiện khả năng học trực tiếp từ pixel, chưa tách được tác dụng của màu tờ tiền khỏi nền, ánh sáng và bố cục."
        elif candidate == "C02_GRAY" and "C01_RGB" in rows:
            rgb = rows["C01_RGB"]
            history = read_csv(results_root / "candidates" / candidate / "history.csv")
            epoch = next(r for r in history if int(r["epoch"]) == int(row["best_epoch"]))
            text += (f"Macro-F1 chênh **{signed(float(row['macro_f1']) - float(rgb['macro_f1']), 6)}** "
                     f"so với C01. Tại checkpoint này, train Accuracy online là **{number(float(epoch['train_accuracy']) * 100, 2)}%**, "
                     f"dev loss là **{number(epoch['dev_loss'], 4)}**. ")
            if float(row["macro_f1"]) < float(rgb["macro_f1"]):
                text += "Nhánh xám chưa học tốt trong cấu hình/ngân sách này. Việc bỏ màu của toàn ảnh có liên quan đến khác biệt biểu diễn, nhưng cũng đổi Conv đầu và quá trình tối ưu; chưa chứng minh mọi CNN xám đều kém hoặc màu nền không góp phần. "
            if int(row["best_epoch"]) == int(row["epochs_run"]):
                text += "Checkpoint tốt nhất ở đúng cuối lượt chạy; chưa có bằng chứng cấu hình đã hội tụ hoàn toàn."
        elif candidate == "C03_RGB_AUG" and "C01_RGB" in rows:
            base = rows["C01_RGB"]
            difference = float(row["macro_f1"]) - float(base["macro_f1"])
            mae_difference = float(row["mae_vnd"]) - float(base["mae_vnd"])
            text += (f"So với C01, Macro-F1 chênh **{signed(difference, 6)}**, còn MAE chênh "
                     f"**{signed(mae_difference, 2)}đ/ảnh**. ")
            if float(row["accuracy"]) == float(base["accuracy"]):
                text += "Hai cấu hình có cùng số ảnh đúng nhưng phân bố dự đoán theo lớp và mức chênh mệnh giá khác nhau. "
            if difference > 0 and mae_difference > 0:
                text += "Đây là đánh đổi: Macro-F1 tăng nhưng MAE xấu hơn; không kết luận augmentation cải thiện mọi tiêu chí. "
            text += "Quy tắc ưu tiên Macro-F1 trước MAE được giữ nguyên; gói augmentation chưa được tách từng phép để kiểm định nguyên nhân."
        elif candidate == "C04_DEEP_AUG" and "C03_RGB_AUG" in rows:
            base = rows["C03_RGB_AUG"]
            baseline_predictions = read_csv(results_root / "candidates/C03_RGB_AUG/predictions_dev.csv")
            baseline_correct = sum(p["y_true_vnd"] == p["y_pred_vnd"] for p in baseline_predictions)
            text += (f"So với C03 có cùng augmentation, số ảnh đúng chênh **{signed(correct - baseline_correct)}**, "
                     f"Macro-F1 chênh **{signed(float(row['macro_f1']) - float(base['macro_f1']), 6)}**, "
                     f"MAE chênh **{signed(float(row['mae_vnd']) - float(base['mae_vnd']), 2)}đ/ảnh**. "
                     "Thêm khối đồng thời tăng số tham số, số chiều GAP và đổi độ phân giải feature map cuối, nên kết quả đánh giá cả cấu hình mở rộng; chưa cô lập riêng hiệu quả độ sâu.")
        paragraphs.append(text)
    return "\n\n".join(paragraphs)


def case_commentary(results_root):
    path = results_root / "error_analysis/selected_cases.csv"
    if not path.is_file():
        return ""
    cases = read_csv(path)
    parts = []
    for reason in ("high_softmax_error", "small_softmax_gap_error", "small_softmax_gap_correct"):
        row = next((r for r in cases if r["selection_reason"] == reason), None)
        if row is None:
            continue
        prefix = (f"**{row['case_id']}**: ảnh thật {number(row['y_true_vnd'])}đ, "
                  f"dự đoán {number(row['y_pred_vnd'])}đ; softmax lớp thắng **{number(row['top_softmax'], 6)}**")
        if reason == "high_softmax_error":
            parts.append(prefix + ". Điểm cao vẫn đi kèm nhãn sai, nên không xem softmax là xác suất đúng được bảo đảm.")
        elif reason == "small_softmax_gap_error":
            ranked = sorted((float(row[f'softmax_{value}']), value) for value in CLASSES)
            second = ranked[-2][1]
            parts.append(prefix + f", chênh hai điểm đầu **{number(row['softmax_gap'], 6)}**. "
                         f"Lớp đứng thứ hai là {number(second)}đ; điểm lớp thật là "
                         f"**{number(row['softmax_' + row['y_true_vnd']], 6)}**. Hai điểm đầu sát nhau không có nghĩa lớp thật gần được chọn.")
        else:
            parts.append(prefix + f", chênh hai điểm đầu **{number(row['softmax_gap'], 6)}**. "
                         "Đây là ảnh đúng có điểm thấp/sát nhau dùng làm đối chứng; chưa triển khai hoặc chọn ngưỡng từ chối trên test.")
    return "\n\n".join(parts)


def _verify_measured(results_root, project_root):
    lock_path = results_root / "selection/locked_cnn_config.json"
    lock = read_json(lock_path)
    run = read_json(results_root / "test_final/run_info.json")
    if digest(lock_path) != run["lock_sha256"]:
        raise ValueError("Test/selection lock mismatch")
    if lock_path.read_bytes() != (results_root / "test_final/locked_cnn_config_used.json").read_bytes():
        raise ValueError("Test used another lock")
    missing_weights, verified = [], 0
    for relative, expected in lock["sha256"].items():
        kind, name = relative.split("/", 1)
        path = (results_root if kind == "results" else project_root) / name
        if not path.exists() and path.suffix == ".pt":
            missing_weights.append(relative)
            continue
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Locked artifact changed/missing: {path}")
        verified += 1
    for name, expected in run["sha256"].items():
        if digest(results_root / "test_final" / name) != expected:
            raise ValueError(f"Test artifact changed: {name}")
    predictions = read_csv(results_root / "test_final/predictions_test.csv")
    metrics = read_json(results_root / "test_final/metrics_test.json")
    verify_scores(predictions, metrics)
    expected = manifest_rows(results_root / "test_final/test_manifest.csv")
    if expected != [(r["relative_path"], int(r["y_true_vnd"])) for r in predictions]:
        raise ValueError("Test prediction order does not match manifest")
    return lock, metrics, predictions, verified, missing_weights


def build_report(results_root, output_root, project_root=ROOT, reference_root=None):
    results_root, output_root, project_root = (
        Path(path).resolve() for path in (results_root, output_root, project_root))
    output_root.mkdir(parents=True, exist_ok=True)
    figures = output_root / "CNN224_FIGURES"
    figures.mkdir(parents=True, exist_ok=True)
    reference_root = (Path(reference_root).resolve() if reference_root is not None
                      else project_root / "src/CNN/reference_results")
    provenance = read_json(reference_root / "provenance.json")
    baselines = {}
    for name, source in provenance["baselines"].items():
        folder = reference_root / source["folder"]
        predictions = read_csv(folder / "predictions_test.csv")
        metrics = read_json(folder / "metrics_test.json")
        verify_scores(predictions, metrics)
        baselines[name] = {"folder": folder, "metrics": metrics, "predictions": predictions}
    for relative, expected in provenance["sha256"].items():
        if digest(reference_root / relative) != expected:
            raise ValueError(f"Reference artifact changed: {relative}")
    measured = (results_root / "test_final/metrics_test.json").is_file()
    lock = metrics = predictions = None
    verified, missing_weights = 0, []
    if measured:
        lock, metrics, predictions, verified, missing_weights = _verify_measured(results_root, project_root)
        for baseline in baselines.values():
            for split in ("train", "dev", "test"):
                folder = results_root / ("test_final" if split == "test" else "inputs")
                if manifest_rows(folder / f"{split}_manifest.csv") != manifest_rows(baseline["folder"] / f"{split}_manifest.csv"):
                    raise ValueError(f"CNN/baseline manifest mismatch: {split}")
    counts = {}
    rows = []
    for split in ("train", "dev", "test"):
        source = (results_root / ("test_final" if split == "test" else "inputs") / f"{split}_manifest.csv"
                  if measured else baselines["KNN"]["folder"] / f"{split}_manifest.csv")
        counts[split] = Counter(value for _, value in manifest_rows(source))
    for value in CLASSES:
        nums = [counts[split][value] for split in ("train", "dev", "test")]
        rows.append([number(value) + "đ", *map(number, nums), number(sum(nums))])
    totals = {split: sum(counts[split].values()) for split in counts}
    rows.append(["**Tổng**", *[number(totals[s]) for s in ("train", "dev", "test")], number(sum(totals.values()))])
    pending = "Chưa có kết quả chạy CNN để điền bảng; số liệu sẽ được lấy từ CSV/JSON sau khi hoàn tất notebook."
    values = {"STATUS": ("**Trạng thái: báo cáo kết quả đã đo.** Các bảng CNN dưới đây được tạo từ output đã kiểm tra."
                         if measured else "**Trạng thái: thiết kế thực nghiệm, chưa có kết quả CNN.** Các mục phương pháp đã hoàn thiện; bảng CNN được để chờ lần chạy Kaggle. Số KNN/SVM là kết quả tham chiếu đã có."),
              "TRAIN_N": number(totals["train"]), "DEV_N": number(totals["dev"]),
              "TEST_N": number(totals["test"]),
              "DATA_TABLE": table(["Mệnh giá", "Train", "Dev", "Test", "Tổng"], rows)}
    values.update({token: "" for token in ("OVERVIEW", "PCA_TEXT", "PER_CLASS_TEXT", "ERROR_SUMMARY",
                                           "CASE_TEXT", "PAIRED_TEXT", "BOUNDARY_FIGURE", "HISTORY_LINK")})
    values["RUNTIME_TABLE"] = values["RUNTIME_TEXT"] = values["PER_CLASS_COMPARISON_TABLE"] = pending
    figure_sources = {}

    def figure(token, source, filename, alt):
        if source.is_file():
            destination = figures / filename
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            values[token] = f"![{alt}](CNN224_FIGURES/{filename})"
            try:
                relative_source = source.relative_to(project_root).as_posix()
            except ValueError:
                relative_source = "results/" + source.relative_to(results_root).as_posix()
            figure_sources[filename] = {"source": relative_source, "sha256": digest(source)}
        else:
            values[token] = f"*Hình {alt.lower()} được tạo sau khi chạy notebook tương ứng.*"

    figure("ARCHITECTURE_UNUSED", HERE / "CNN224_FIGURES/CNN_ARCHITECTURE.svg",
           "CNN_ARCHITECTURE.svg", "Kiến trúc CNN")
    figure("INPUT_FIGURE", results_root / "inputs/rgb224_examples.png", "CNN_1_RESIZE.png", "Ảnh train trước và sau RGB224")
    figure("AUGMENTATION_FIGURE", results_root / "inputs/augmentation_examples.png", "CNN_2_AUGMENTATION.png", "Minh họa augmentation chỉ dùng cho train")
    figure("EMBEDDING_FIGURE", results_root / "feature_analysis/embedding_pca_dev.png", "CNN_4_EMBEDDING_PCA.png", "Đặc trưng dev chiếu lên PCA fit bằng train")
    figure("FEATURE_MAP_FIGURE", results_root / "feature_analysis/feature_maps.png", "CNN_5_FEATURE_MAPS.png", "Feature map của một ảnh train")
    figure("CONFUSION_FIGURE", results_root / "test_final/confusion_matrix_test.png", "CNN_6_CONFUSION_TEST.png", "Ma trận nhầm lẫn CNN trên test")
    pages = sorted((results_root / "error_analysis").glob("cases_page_*.png"))
    figure("ERROR_FIGURE", pages[-2] if len(pages) > 1 else pages[0] if pages else results_root / "error_analysis/cases_page_01.png",
           "CNN_7_GRADCAM.png", "Các trường hợp nhầm có chênh lệch lớn và softmax cao")
    if len(pages) > 1:
        figure("BOUNDARY_FIGURE", pages[-1], "CNN_8_GRADCAM_BOUNDARY.png",
               "Đối chứng điểm sát nhau, ảnh đúng và ảnh sai")
    dev_rows = []
    if measured:
        grid = read_csv(results_root / "selection/best_by_candidate_dev.csv")
        for row in sorted(grid, key=lambda r: int(r["candidate_order"])):
            dev_rows.append([row["candidate_id"], row["best_epoch"], row["epochs_run"], number(row["parameters"]),
                             number(float(row["accuracy"]) * 100, 2) + "%", number(row["macro_f1"], 6), number(row["mae_vnd"], 2)])
        values["DEV_TABLE"] = table(["Cấu hình", "Epoch chọn", "Epoch chạy", "Tham số", "Accuracy", "Macro-F1", "MAE (đ/ảnh)"], dev_rows)
        winner = lock["candidate"]["candidate_id"]
        values["SELECTION_TEXT"] = f"Phương án đã khóa là **{winner}**, checkpoint epoch **{lock['best_epoch']}**; biểu diễn GAP có **{lock['candidate']['channels'][-1]} chiều**. Trọng số được giữ nguyên khi chuyển sang test."
        values["CANDIDATE_OBSERVATIONS"] = candidate_commentary(grid, results_root)
        histories = {r["candidate_id"]: read_csv(results_root / "candidates" / r["candidate_id"] / "history.csv") for r in grid}
        last_epoch = histories[winner][-1]
        if int(last_epoch["epoch"]) != lock["best_epoch"]:
            values["SELECTION_TEXT"] += (f" Epoch cuối {last_epoch['epoch']} có Macro-F1 "
                f"**{number(last_epoch['dev_macro_f1'], 6)}**, MAE **{number(last_epoch['dev_mae_vnd'], 2)}đ/ảnh**; "
                "không thay checkpoint đã chọn bằng trọng số của epoch cuối.")
        runtime_rows, total_seconds = [], 0.0
        for row in sorted(grid, key=lambda r: int(r["candidate_order"])):
            hist = histories[row["candidate_id"]]
            train_seconds = sum(float(r["train_seconds"]) for r in hist)
            dev_seconds = sum(float(r["dev_seconds"]) for r in hist)
            total_seconds += train_seconds + dev_seconds
            runtime_rows.append([row["candidate_id"], len(hist), number(train_seconds / 60, 2),
                                 number(dev_seconds / 60, 2), number((train_seconds + dev_seconds) / 60, 2)])
        values["RUNTIME_TABLE"] = table(["Cấu hình", "Epoch chạy", "Train (phút)", "Dev (phút)", "Tổng (phút)"], runtime_rows)
        env = read_json(results_root / "candidates" / winner / "summary.json")["environment"]
        values["RUNTIME_TEXT"] = (f"Tổng thời gian train và chấm dev của bốn cấu hình là **{number(total_seconds / 60, 2)} phút**. "
            f"Môi trường được ghi: Python {env['python']}, PyTorch {env['torch']}, NumPy {env['numpy']}, "
            f"OpenCV {env['opencv']}, scikit-learn {env['sklearn']}, thiết bị {env['device']} ({env.get('gpu') or 'CPU'}). "
            "Tổng tính từ giây gốc trước khi làm tròn từng hàng. Thời gian cộng từ log epoch, chưa gồm sao chép output, vẽ hình, phân tích đặc trưng và các bước khác; không dùng làm benchmark tốc độ với KNN/SVM.")
        maximum = lock["protocol"]["training"]["max_epochs"]
        if all(len(hist) == maximum for hist in histories.values()):
            values["RUNTIME_TEXT"] += f" Cả bốn cấu hình chạy đủ {maximum} epoch; không lượt nào dừng sớm."
        figure("LEARNING_FIGURE", results_root / "candidates" / winner / "learning_curves.png", "CNN_3_LEARNING_CURVES.png", "Learning curve của cấu hình được chọn bằng dev")
        feature_record = results_root / "feature_analysis/analysis_record.json"
        if feature_record.is_file():
            feature = read_json(feature_record)
            if feature["lock_sha256"] != digest(results_root / "selection/locked_cnn_config.json"):
                raise ValueError("Feature analysis belongs to another lock")
            ratios = feature["pca_explained_variance_ratio"]
            values["PCA_TEXT"] = (f"Biểu diễn của cấu hình đã chọn có **{feature['embedding_dim']} chiều**. "
                f"Hai trục PCA giữ lần lượt **{number(ratios[0] * 100, 2)}%** và **{number(ratios[1] * 100, 2)}%**, "
                f"tổng **{number(sum(ratios) * 100, 2)}% phương sai train**. Phần phương sai còn lại không xuất hiện trên hình, "
                "nên việc hai lớp chồng lấn ở PCA2 chưa chứng minh chúng không thể phân biệt trong không gian GAP đầy đủ.")
        metric_rows = []
        dev_predictions = read_csv(results_root / "candidates" / winner / "predictions_dev.csv")
        for split, entry in (("Dev", lock["dev_metrics"]), ("Test", metrics)):
            rows_for_split = dev_predictions if split == "Dev" else predictions
            n_correct = sum(r["y_true_vnd"] == r["y_pred_vnd"] for r in rows_for_split)
            metric_rows.append([split, number(len(rows_for_split)), number(n_correct), number(float(entry["accuracy"]) * 100, 2) + "%", number(entry["macro_f1"], 6), number(entry["mae_vnd"], 2)])
        values["TEST_TABLE"] = table(["Tập", "Số ảnh", "Số đúng", "Accuracy", "Macro-F1", "MAE (đ/ảnh)"], metric_rows)
        correct = sum(row["y_true_vnd"] == row["y_pred_vnd"] for row in predictions)
        values["OVERVIEW"] = (f"CNN được chọn là **{winner}, epoch {lock['best_epoch']}**, "
            f"đúng **{number(correct)}/{number(len(predictions))} ảnh test**: Accuracy **{number(metrics['accuracy'] * 100, 2)}%**, "
            f"Macro-F1 **{number(metrics['macro_f1'], 6)}**, MAE **{number(metrics['mae_vnd'], 2)}đ/ảnh**. "
            "Các kết quả và so sánh dưới đây thuộc cùng split của dự án.")
        values["TEST_TEXT"] = f"CNN dự đoán đúng **{number(correct)}/{number(len(predictions))} ảnh test**, còn **{number(len(predictions) - correct)} lỗi**. Tổng chênh lệch danh nghĩa là **{number(metrics['total_gap_vnd'])}đ**; chia cho toàn bộ {number(len(predictions))} ảnh thu được MAE đã báo cáo."
        values["TEST_TEXT"] += (f" So với dev, Accuracy chênh **{signed((metrics['accuracy'] - lock['dev_metrics']['accuracy']) * 100, 2)} điểm phần trăm**, "
            f"Macro-F1 **{signed(metrics['macro_f1'] - lock['dev_metrics']['macro_f1'], 6)}**, "
            f"MAE **{signed(metrics['mae_vnd'] - lock['dev_metrics']['mae_vnd'], 2)}đ/ảnh**. "
            "Khác biệt giữa hai split không chứng minh mô hình tốt hơn trên mọi dữ liệu mới.")
        test_run = read_json(results_root / "test_final/run_info.json")
        values["TEST_TEXT"] += (f" Suy luận kèm đọc ảnh mất **{number(test_run['inference_with_io_seconds'], 2)} giây** "
                               "ở lượt này; không có bước fit lại trong notebook test.")
        values["PER_CLASS_TABLE"] = table(["Mệnh giá", "Support", "Precision", "Recall", "F1"],
            [[number(r["denomination_vnd"]) + "đ", number(r["support"]), number(r["precision"], 4), number(r["recall"], 4), number(r["f1"], 4)] for r in metrics["per_class"]])
        low_f1 = min(metrics["per_class"], key=lambda r: r["f1"])
        high_f1 = max(metrics["per_class"], key=lambda r: r["f1"])
        low_recall = min(metrics["per_class"], key=lambda r: r["recall"])
        values["PER_CLASS_TEXT"] = (f"Lớp {number(high_f1['denomination_vnd'])}đ có F1 cao nhất "
            f"(**{number(high_f1['f1'], 4)}**), còn {number(low_f1['denomination_vnd'])}đ thấp nhất "
            f"(**{number(low_f1['f1'], 4)}**). Recall thấp nhất thuộc lớp {number(low_recall['denomination_vnd'])}đ "
            f"(**{number(low_recall['recall'] * 100, 2)}%**). Số ảnh mỗi lớp không đều; bảng này bổ sung cho Accuracy và Macro-F1 tổng thể.")
        errors = sorted(metrics["error_pairs"], key=lambda r: (-r["total_gap_vnd"], r["true_vnd"], r["predicted_vnd"]))[:5]
        values["ERROR_TABLE"] = table(["Thật → dự đoán", "Số lỗi", "Chênh/lỗi", "Tổng chênh"],
            [[f"{number(r['true_vnd'])} → {number(r['predicted_vnd'])}", number(r["count"]), number(r["gap_vnd"]), number(r["total_gap_vnd"])] for r in errors]) if errors else "Không có lỗi trên test của lượt chạy này."
        pairs = {}
        for row in metrics["error_pairs"]:
            key = tuple(sorted((row["true_vnd"], row["predicted_vnd"])))
            count_pair, cost_pair = pairs.get(key, (0, 0))
            pairs[key] = (count_pair + row["count"], cost_pair + row["total_gap_vnd"])
        if pairs:
            main_pair = max(pairs, key=lambda p: pairs[p][1])
            count_pair, cost_pair = pairs[main_pair]
            values["ERROR_SUMMARY"] = (f"Gộp hai chiều, cặp **{number(main_pair[0])}đ ↔ {number(main_pair[1])}đ** "
                f"có **{count_pair} lỗi**, tổng chênh **{number(cost_pair)}đ**, chiếm "
                f"**{number(cost_pair / metrics['total_gap_vnd'] * 100, 2)}%** tổng chênh của CNN. "
                "Dù tỷ lệ đúng tổng cao, các cặp mệnh giá cách xa nhau vẫn có thể chi phối MAE; cần đọc cả số lỗi và độ lớn chênh lệch.")
        values["CASE_TEXT"] = case_commentary(results_root)
        shutil.copy2(results_root / "selection/best_by_candidate_dev.csv", output_root / "BANG_CAU_HINH_DEV.csv")
        epoch_count = sum(len(history) for history in histories.values())
        shutil.copy2(results_root / "selection/all_epochs_dev.csv", output_root / "PHU_LUC_LICH_SU_EPOCH_CNN.csv")
        values["HISTORY_LINK"] = (f"[Phụ lục lịch sử epoch](PHU_LUC_LICH_SU_EPOCH_CNN.csv) giữ đủ **{epoch_count} dòng** "
                                  "với loss, Accuracy, Macro-F1, MAE, learning rate và thời gian gốc; không làm tròn trước khi chọn checkpoint.")
    else:
        for token in ("DEV_TABLE", "TEST_TABLE", "PER_CLASS_TABLE", "ERROR_TABLE", "CANDIDATE_OBSERVATIONS"):
            values[token] = pending
        values["SELECTION_TEXT"] = "Chưa chọn cấu hình hoặc checkpoint. Không đặt trước giả thuyết RGB, augmentation hoặc mạng sâu sẽ thắng."
        values["TEST_TEXT"] = "Chưa có số đo test CNN. Chỉ chạy notebook 07 sau khi đã khóa đủ bốn kết quả dev."
        values["LEARNING_FIGURE"] = "*Learning curve sẽ được tạo từ lịch sử huấn luyện thật.*"
    comparison, paired, differences, paired_text = [], [], [], []
    wrong_sets = [{r["relative_path"] for r in predictions if r["y_true_vnd"] != r["y_pred_vnd"]}] if measured else []
    for name, entry in baselines.items():
        m = entry["metrics"]
        pipeline_name = ("HSV12; k=1, uniform, Euclidean" if name == "KNN"
                         else "HSV12+HOG; RBF C=10, gamma=scale" if name == "SVM" else name)
        baseline_correct = sum(r["y_true_vnd"] == r["y_pred_vnd"] for r in entry["predictions"])
        comparison.append([name, pipeline_name, number(baseline_correct), number(m["accuracy"] * 100, 2) + "%", number(m["macro_f1"], 6), number(m["mae_vnd"], 2)])
        if measured:
            a = [row["y_true_vnd"] == row["y_pred_vnd"] for row in predictions]
            b = [row["y_true_vnd"] == row["y_pred_vnd"] for row in entry["predictions"]]
            if [(r["relative_path"], r["y_true_vnd"]) for r in predictions] != [(r["relative_path"], r["y_true_vnd"]) for r in entry["predictions"]]:
                raise ValueError("Paired prediction order mismatch")
            paired.append([name, sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                           sum(not x and y for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))])
            differences.append(f"CNN − {name}: số đúng **{signed(sum(a) - sum(b))} ảnh**, "
                f"Accuracy **{signed((metrics['accuracy'] - m['accuracy']) * 100, 2)} điểm phần trăm**, "
                f"Macro-F1 **{signed(metrics['macro_f1'] - m['macro_f1'], 6)}**, "
                f"MAE **{signed(metrics['mae_vnd'] - m['mae_vnd'], 2)}đ/ảnh**. "
                f"Tổng chênh lệch VND thay đổi **{signed(metrics['total_gap_vnd'] - m['total_gap_vnd'])}đ**.")
            if metrics["mae_vnd"] < m["mae_vnd"]:
                differences[-1] += f" MAE giảm **{number((1 - metrics['mae_vnd'] / m['mae_vnd']) * 100, 2)}%** so với {name}."
            paired_text.append(f"So với {name}, CNN đúng **{paired[-1][2]}** ảnh đối chứng sai, "
                              f"nhưng cũng sai **{paired[-1][3]}** ảnh đối chứng đúng; "
                              f"chênh ròng **{signed(paired[-1][2] - paired[-1][3])} ảnh**.")
            wrong_sets.append({r["relative_path"] for r in entry["predictions"] if r["y_true_vnd"] != r["y_pred_vnd"]})
    cnn_description = lock["candidate"]["candidate_id"] + f"; epoch {lock['best_epoch']}" if measured else "Bốn cấu hình, chưa chọn"
    comparison.append(["CNN", cnn_description, number(correct) if measured else "—",
                       number(metrics["accuracy"] * 100, 2) + "%" if measured else "Chưa chạy",
                       number(metrics["macro_f1"], 6) if measured else "—", number(metrics["mae_vnd"], 2) if measured else "—"])
    values["COMPARISON_TABLE"] = table(["Phương pháp", "Biểu diễn/cấu hình", "Số đúng", "Accuracy", "Macro-F1", "MAE (đ/ảnh)"], comparison)
    values["COMPARISON_TEXT"] = "\n\n".join(differences) if measured else "Chưa có cơ sở kết luận CNN cao hơn hoặc thấp hơn KNN/SVM. So sánh ghép cặp chỉ được xác nhận khi có manifest và dự đoán CNN thật."
    values["PAIRED_TABLE"] = table(["So với", "Cả hai đúng", "Chỉ CNN đúng", "Chỉ đối chứng đúng", "Cả hai sai"], paired) if measured else pending
    if measured:
        common_wrong = len(set.intersection(*wrong_sets))
        values["PAIRED_TEXT"] = " ".join(paired_text) + f" Có **{common_wrong} ảnh cả ba mô hình cùng sai**. Kết quả tổng tốt hơn chưa có nghĩa CNN luôn đúng ở từng ảnh."
        baseline_order = list(baselines)
        values["PER_CLASS_COMPARISON_TABLE"] = table(["Mệnh giá", *baseline_order, "CNN"], [
            [number(row["denomination_vnd"]) + "đ",
             *[number(baselines[name]["metrics"]["per_class"][i]["f1"], 4) for name in baseline_order],
             number(row["f1"], 4)] for i, row in enumerate(metrics["per_class"])])
    values["CONCLUSION"] = (f"Trong bốn cấu hình đã thử, **{lock['candidate']['candidate_id']}** được chọn bằng dev và đạt trên test **Accuracy {number(metrics['accuracy'] * 100, 2)}%, Macro-F1 {number(metrics['macro_f1'], 6)}, MAE {number(metrics['mae_vnd'], 2)}đ/ảnh**. Kết luận chỉ áp dụng cho lượt chạy và bộ dữ liệu này."
                            if measured else "Phần phương pháp, kiến trúc, cấu hình và cách kiểm tra đã được định nghĩa. Chưa huấn luyện CNN nên chưa kết luận về độ chính xác, vai trò màu, lợi ích augmentation hoặc độ sâu. Các ô chờ sẽ được thay bằng số đo khi chạy xong chuỗi notebook.")
    weight_text = (f" Có {len(missing_weights)} checkpoint lớn không có trong bản dữ liệu hiện tại."
                   if missing_weights else " Bốn file best.pt đều có mặt và khớp hash đã khóa.")
    values["VERIFICATION_TEXT"] = (f"**Kiểm tra số liệu.** Dự đoán test tính lại khớp Accuracy, Macro-F1, MAE, tổng chênh lệch và ma trận nhầm lẫn; ba manifest CNN/KNN/SVM khớp từng hàng. Có {verified} file khóa được kiểm tra tại thời điểm dựng báo cáo." + weight_text + " Hash của khóa dùng trên test khớp khóa đã chọn. Kiểm tra xác nhận file và thứ tự ảnh, không chứng minh đã so sánh byte ảnh nguồn của mọi phiên huấn luyện. Hồ sơ và nguồn hình nằm trong `report_manifest.json`."
                                   if measured else "**Hồ sơ kiểm tra hiện có.** Hai baseline KNN/SVM đã được đối chiếu từ dự đoán CSV và JSON, cùng provenance nguồn. CNN chưa có output để xác minh hash checkpoint hoặc metric. Không điền số giả định vào các bảng.")
    text = (HERE / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", text):
        raise ValueError("Unfilled report token")
    destination = output_root / "BAO_CAO_THUC_NGHIEM_CNN.md"
    destination.write_text(text, encoding="utf-8")
    notes = HERE / "REFERENCE_NOTES.md"
    if notes.is_file() and notes.resolve() != (output_root / notes.name).resolve():
        shutil.copy2(notes, output_root / notes.name)
    record = {"status": "measured" if measured else "awaiting_cnn_run", "figures": figure_sources,
              "reference_provenance_sha256": digest(reference_root / "provenance.json"),
              "verified_lock_files": verified, "missing_large_checkpoints": missing_weights,
              "report_sha256": digest(destination)}
    (output_root / "report_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "src/CNN/results_224")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args()
    print(build_report(args.results, args.output, args.project))
