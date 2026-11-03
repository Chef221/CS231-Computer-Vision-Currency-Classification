"""MacroValueF1 and diagnostic metrics for six VND denominations."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASS_ORDER = (10_000, 20_000, 50_000, 100_000, 200_000, 500_000)


def evaluate_classification(y_true, y_pred):
    """Score hard predictions with actual VND values as labels."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(true, pred, labels=CLASS_ORDER)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=CLASS_ORDER, average=None, zero_division=0
    )
    predicted_count = matrix.sum(axis=0)

    # Công thức value metrics được tính riêng cho từng mệnh giá.
    per_class = []
    for i, value in enumerate(CLASS_ORDER):
        tp = int(matrix[i, i])
        correct_value = tp * value
        incoming_gap = sum(
            int(matrix[j, i]) * abs(CLASS_ORDER[j] - value) for j in range(6)
        )
        outgoing_gap = sum(
            int(matrix[i, j]) * abs(value - CLASS_ORDER[j]) for j in range(6)
        )
        p_den = correct_value + incoming_gap
        r_den = correct_value + outgoing_gap
        f_den = 2 * correct_value + incoming_gap + outgoing_gap
        value_p = correct_value / p_den if p_den else 0.0
        value_r = correct_value / r_den if r_den else 0.0
        value_f = 2 * correct_value / f_den if f_den else 0.0
        per_class.append({
            "denomination_vnd": int(value), "support": int(support[i]),
            "predicted_count": int(predicted_count[i]),
            "tp": tp, "fp": int(predicted_count[i] - tp),
            "fn": int(matrix[i, :].sum() - tp),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "correct_value_vnd": int(correct_value),
            "incoming_gap_vnd": int(incoming_gap),
            "outgoing_gap_vnd": int(outgoing_gap),
            "value_precision": float(value_p), "value_recall": float(value_r),
            "value_f1": float(value_f),
        })

    # Các phần còn lại lập điểm tổng hợp và kết quả JSON.
    return {
        "class_order_vnd": list(CLASS_ORDER),
        "n_samples": int(len(true)),
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(np.trace(matrix) / len(true)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "macro_value_precision": float(np.mean([row["value_precision"] for row in per_class])),
        "macro_value_recall": float(np.mean([row["value_recall"] for row in per_class])),
        "macro_value_f1": float(np.mean([row["value_f1"] for row in per_class])),
        "per_class": per_class,
    }
