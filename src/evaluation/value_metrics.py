"""Shared classification metrics and denomination gaps for six VND classes."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASS_ORDER = (10_000, 20_000, 50_000, 100_000, 200_000, 500_000)


def evaluate_classification(y_true, y_pred):
    """Score hard predictions with actual VND values as labels."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    matrix = confusion_matrix(true, pred, labels=CLASS_ORDER)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=CLASS_ORDER, average=None, zero_division=0
    )
    predicted_count = matrix.sum(axis=0)

    # Precision, Recall và F1 thông thường cho từng mệnh giá.
    per_class = []
    for i, value in enumerate(CLASS_ORDER):
        tp = int(matrix[i, i])
        per_class.append({
            "denomination_vnd": int(value), "support": int(support[i]),
            "predicted_count": int(predicted_count[i]),
            "tp": tp, "fp": int(predicted_count[i] - tp),
            "fn": int(matrix[i, :].sum() - tp),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]),
        })

    # Mỗi dòng mô tả một cặp mệnh giá đã bị nhầm.
    error_pairs = []
    for i, true_value in enumerate(CLASS_ORDER):
        for j, predicted_value in enumerate(CLASS_ORDER):
            count = int(matrix[i, j])
            if i != j and count > 0:
                gap = abs(true_value - predicted_value)
                error_pairs.append({
                    "true_vnd": true_value, "predicted_vnd": predicted_value,
                    "count": count, "gap_vnd": gap,
                    "total_gap_vnd": count * gap,
                })

    total_gap = int(np.abs(true - pred).sum())
    return {
        "class_order_vnd": list(CLASS_ORDER),
        "n_samples": int(len(true)),
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(np.trace(matrix) / len(true)),
        "macro_f1": float(np.mean(f1)),
        "mae_vnd": total_gap / len(true),
        "total_gap_vnd": total_gap,
        "per_class": per_class,
        "error_pairs": error_pairs,
    }
