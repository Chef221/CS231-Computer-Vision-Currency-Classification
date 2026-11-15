"""Shared classification metrics and denomination gaps for six VND classes."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASS_ORDER = (10_000, 20_000, 50_000, 100_000, 200_000, 500_000)


def evaluate_classification(y_true, y_pred):
    """Score hard predictions with actual VND values as labels."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    n_samples = len(true)

    # Overall metrics
    matrix = confusion_matrix(true, pred, labels=CLASS_ORDER)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=CLASS_ORDER, zero_division=0
    )
    accuracy = float(np.trace(matrix) / n_samples)
    macro_f1 = float(f1.mean())

    gaps = np.abs(true - pred)
    total_gap = int(gaps.sum())
    mae_vnd = total_gap / n_samples

    # One row per denomination
    per_class = []
    for i, denomination in enumerate(CLASS_ORDER):
        tp = int(matrix[i, i])
        predicted_count = int(matrix[:, i].sum())
        true_count = int(matrix[i, :].sum())

        per_class.append({
            "denomination_vnd": denomination,
            "support": int(support[i]),
            "predicted_count": predicted_count,
            "tp": tp,
            "fp": predicted_count - tp,
            "fn": true_count - tp,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        })

    # One row per observed mistake, with the true and predicted labels in order
    error_pairs = []
    for i, true_vnd in enumerate(CLASS_ORDER):
        for j, predicted_vnd in enumerate(CLASS_ORDER):
            count = int(matrix[i, j])
            if i == j or count == 0:
                continue

            gap = abs(true_vnd - predicted_vnd)
            error_pairs.append({
                "true_vnd": true_vnd,
                "predicted_vnd": predicted_vnd,
                "count": count,
                "gap_vnd": gap,
                "total_gap_vnd": count * gap,
            })

    return {
        "class_order_vnd": list(CLASS_ORDER),
        "n_samples": n_samples,
        "confusion_matrix": matrix.tolist(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mae_vnd": mae_vnd,
        "total_gap_vnd": total_gap,
        "per_class": per_class,
        "error_pairs": error_pairs,
    }
