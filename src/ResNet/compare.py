import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")

from src.ResNet.dataset import BanknoteDataset, get_transforms, DENOMINATIONS, IDX_TO_CLASS
from src.ResNet.model import create_resnet18
from src.evaluation.value_metrics import evaluate_classification
from cost_metrics import (
    default_cost_matrix,
    increasing_importance_matrix,
    bayes_predict,
    evaluate as evaluate_cost,
    sample_costs,
    bootstrap_ci,
    CRITICAL
)

def run():
    print("=" * 80)
    print("      THỰC NGHIỆM ĐỐI SÁNH: KHÔNG DÙNG COST-SENSITIVE VS COST-SENSITIVE (1 CHIỀU & 2 CHIỀU)")
    print("=" * 80)

    # 1. Load Data & Model
    data_dir = Path("DATA")
    _, eval_transform = get_transforms(224)
    test_dataset = BanknoteDataset(root_dir=data_dir, split="test", transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = create_resnet18(num_classes=6, pretrained=False)
    checkpoint_path = Path("experiments/kaggle/output/best_resnet18.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    y_true_idx = []
    y_true_vnd = []
    probs = []

    with torch.no_grad():
        for images, targets, denoms, _ in test_loader:
            outputs = model(images)
            p = torch.softmax(outputs, dim=1).numpy()
            probs.append(p)
            y_true_idx.extend(targets.numpy().tolist())
            y_true_vnd.extend(denoms.numpy().tolist())

    P = np.vstack(probs)
    y_idx = np.array(y_true_idx)
    y_vnd = np.array(y_true_vnd)

    # 2. Định nghĩa các quy tắc ra quyết định
    # Cấu hình 1: Không dùng Cost-Sensitive (Standard Argmax)
    pred_argmax_idx = np.argmax(P, axis=1)
    pred_argmax_vnd = [IDX_TO_CLASS[i] for i in pred_argmax_idx]

    # Cấu hình 2: Cost-Sensitive 1 chiều (symmetric=False)
    # Phạt theo mệnh giá tờ tiền THẬT v_thật (Bối cảnh ATM / Máy thối tiền / Kho quỹ)
    C_1way = increasing_importance_matrix(alpha=0.5, symmetric=False)
    pred_bayes_1way_idx = bayes_predict(P, C_1way)
    pred_bayes_1way_vnd = [IDX_TO_CLASS[i] for i in pred_bayes_1way_idx]

    # Cấu hình 3: Cost-Sensitive 2 chiều (symmetric=True)
    # Phạt theo max(v_thật, v_đoán) (Bối cảnh Self-checkout / Cây thanh toán 2 chiều vừa thu vừa thối)
    C_2way = increasing_importance_matrix(alpha=0.5, symmetric=True)
    pred_bayes_2way_idx = bayes_predict(P, C_2way)
    pred_bayes_2way_vnd = [IDX_TO_CLASS[i] for i in pred_bayes_2way_idx]

    # 3. Đánh giá bằng Value Metrics (Metric chuẩn trong Git)
    m_argmax = evaluate_classification(y_vnd, pred_argmax_vnd)
    m_1way = evaluate_classification(y_vnd, pred_bayes_1way_vnd)
    m_2way = evaluate_classification(y_vnd, pred_bayes_2way_vnd)

    # 4. Đánh giá bằng Cost Metrics (Độ đo chuyên sâu kinh tế)
    cost_argmax_1w = evaluate_cost(y_idx, pred_argmax_idx, C_1way, critical=CRITICAL)
    cost_bayes_1w  = evaluate_cost(y_idx, pred_bayes_1way_idx, C_1way, critical=CRITICAL)
    
    cost_argmax_2w = evaluate_cost(y_idx, pred_argmax_idx, C_2way, critical=CRITICAL)
    cost_bayes_2w  = evaluate_cost(y_idx, pred_bayes_2way_idx, C_2way, critical=CRITICAL)

    # Tạo bảng tổng hợp
    summary_data = [
        {
            "Phương pháp": "1. Không dùng Cost-Sensitive (Argmax)",
            "Accuracy": f"{m_argmax['accuracy']:.4f} ({m_argmax['accuracy']*100:.2f}%)",
            "Macro-F1": f"{m_argmax['macro_f1']:.4f}",
            "MAE (VNĐ)": f"{m_argmax['mae_vnd']:,.1f} đ",
            "Tổng sai lệch (VNĐ)": f"{m_argmax['total_gap_vnd']:,} đ",
            "Avg Cost (1 chiều)": f"{cost_argmax_1w['avg_cost']:.4f}",
            "Avg Cost (2 chiều)": f"{cost_argmax_2w['avg_cost']:.4f}",
            "Critical Recall (100k-500k)": f"{cost_argmax_1w['critical_recall']:.4f}"
        },
        {
            "Phương pháp": "2. Cost-Sensitive 1 chiều (Bayes 1-way)",
            "Accuracy": f"{m_1way['accuracy']:.4f} ({m_1way['accuracy']*100:.2f}%)",
            "Macro-F1": f"{m_1way['macro_f1']:.4f}",
            "MAE (VNĐ)": f"{m_1way['mae_vnd']:,.1f} đ",
            "Tổng sai lệch (VNĐ)": f"{m_1way['total_gap_vnd']:,} đ",
            "Avg Cost (1 chiều)": f"{cost_bayes_1w['avg_cost']:.4f}",
            "Avg Cost (2 chiều)": f"{cost_bayes_1w['avg_cost']:.4f}",
            "Critical Recall (100k-500k)": f"{cost_bayes_1w['critical_recall']:.4f}"
        },
        {
            "Phương pháp": "3. Cost-Sensitive 2 chiều (Bayes 2-way)",
            "Accuracy": f"{m_2way['accuracy']:.4f} ({m_2way['accuracy']*100:.2f}%)",
            "Macro-F1": f"{m_2way['macro_f1']:.4f}",
            "MAE (VNĐ)": f"{m_2way['mae_vnd']:,.1f} đ",
            "Tổng sai lệch (VNĐ)": f"{m_2way['total_gap_vnd']:,} đ",
            "Avg Cost (1 chiều)": f"{cost_bayes_2w['avg_cost']:.4f}",
            "Avg Cost (2 chiều)": f"{cost_bayes_2w['avg_cost']:.4f}",
            "Critical Recall (100k-500k)": f"{cost_bayes_2w['critical_recall']:.4f}"
        }
    ]

    print("\n--- BẢNG SO SÁNH TỔNG HỢP (THEO CẢ METRIC GIT VÀ COST METRICS) ---")
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Kiểm tra các ca thay đổi nhãn
    diff_1 = np.where(pred_argmax_idx != pred_bayes_1way_idx)[0]
    diff_2 = np.where(pred_argmax_idx != pred_bayes_2way_idx)[0]
    print(f"\nSố ảnh đổi quyết định từ Argmax -> Bayes 1 chiều: {len(diff_1)} ảnh")
    print(f"Số ảnh đổi quyết định từ Argmax -> Bayes 2 chiều: {len(diff_2)} ảnh")

    if len(diff_1) > 0:
        print("\nChi tiết các ảnh thay đổi bởi Bayes 1 chiều:")
        for idx in diff_1:
            true_v = y_vnd[idx]
            arg_v = pred_argmax_vnd[idx]
            bay_v = pred_bayes_1way_vnd[idx]
            probs_str = ", ".join([f"{v//1000}k:{P[idx, i]:.3f}" for i, v in enumerate(DENOMINATIONS)])
            print(f"  Ảnh #{idx} | Thật: {true_v:,} đ | Argmax: {arg_v:,} đ -> Bayes: {bay_v:,} đ | P: [{probs_str}]")

    if len(diff_2) > 0:
        print("\nChi tiết các ảnh thay đổi bởi Bayes 2 chiều:")
        for idx in diff_2:
            true_v = y_vnd[idx]
            arg_v = pred_argmax_vnd[idx]
            bay_v = pred_bayes_2way_vnd[idx]
            probs_str = ", ".join([f"{v//1000}k:{P[idx, i]:.3f}" for i, v in enumerate(DENOMINATIONS)])
            print(f"  Ảnh #{idx} | Thật: {true_v:,} đ | Argmax: {arg_v:,} đ -> Bayes: {bay_v:,} đ | P: [{probs_str}]")

    # Lưu kết quả đối sánh
    out_path = Path("experiments/kaggle/output/comparison_cost_sensitive.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_data,
            "argmax_metrics": m_argmax,
            "bayes_1way_metrics": m_1way,
            "bayes_2way_metrics": m_2way
        }, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu kết quả đối sánh vào: {out_path}")

if __name__ == "__main__":
    run()
