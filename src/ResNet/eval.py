import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.value_metrics import CLASS_ORDER, evaluate_classification
from src.ResNet.dataset import BanknoteDataset, get_transforms, DENOMINATIONS, IDX_TO_CLASS
from src.ResNet.model import create_resnet18

def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet-18 on the test split.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/kaggle/output"))
    args = parser.parse_args()
    print("=" * 60)
    print("   CS231 - RESNET-18 TEST SET EVALUATION (OFFICIAL)")
    print("=" * 60)

    # 1. Load Dataset
    data_dir = args.data_dir
    _, eval_transform = get_transforms(224)
    test_dataset = BanknoteDataset(root_dir=data_dir, split="test", transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Tổng số mẫu tập Test: {len(test_dataset)} ảnh")

    # 2. Load Model
    model = create_resnet18(num_classes=6, pretrained=False)
    checkpoint_path = args.checkpoint or Path("experiments/kaggle/output/best_resnet18.pth")
    if args.checkpoint is None and not checkpoint_path.exists():
        checkpoint_path = Path("web_demo/backend/weights/best_resnet18_baseline.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    print(f"Checkpoint: {checkpoint_path}")

    # 3. Inference
    y_true_vnd = []
    y_pred_vnd = []
    relative_paths = []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets, denoms, paths) in enumerate(test_loader):
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).numpy()

            y_true_vnd.extend(denoms.numpy().tolist())
            y_pred_vnd.extend([IDX_TO_CLASS[p] for p in preds])
            relative_paths.extend(paths)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"  Đã dự đoán: {len(y_true_vnd)}/{len(test_dataset)} ảnh...")

    elapsed = time.time() - start_time
    print(f"Hoàn thành inference trong {elapsed:.2f}s ({len(test_dataset)/elapsed:.1f} FPS)")

    # 4. Compute Metrics
    metrics = evaluate_classification(y_true_vnd, y_pred_vnd)

    print("\n" + "=" * 60)
    print("   KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC (METRICS STANDARD)")
    print("=" * 60)
    print(f"  • Số lượng mẫu   : {metrics['n_samples']} ảnh")
    print(f"  • Accuracy        : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Macro-F1        : {metrics['macro_f1']:.4f}")
    print(f"  • MAE theo VNĐ    : {metrics['mae_vnd']:,.1f} VNĐ / ảnh")
    print(f"  • Tổng sai lệch   : {metrics['total_gap_vnd']:,} VNĐ")
    print("=" * 60)

    # Chi tiết từng lớp
    print("\nChi tiết từng mệnh giá (Per-Class Performance):")
    pc_data = []
    for pc in metrics["per_class"]:
        pc_data.append({
            "Mệnh giá": f"{pc['denomination_vnd']:,} đ",
            "Precision": f"{pc['precision']:.4f}",
            "Recall": f"{pc['recall']:.4f}",
            "F1-score": f"{pc['f1']:.4f}",
            "Số lượng": pc["support"]
        })
    print(pd.DataFrame(pc_data).to_string(index=False))

    # Bảng cặp nhầm lẫn
    print("\nBảng các cặp nhầm lẫn nguy hiểm nhất (Error Pairs):")
    error_pairs = metrics["error_pairs"]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(error_pairs) == 0:
        print("  Tuyệt vời! Không có ảnh nào bị dự đoán sai.")
    else:
        err_df = pd.DataFrame(error_pairs).sort_values(by="total_gap_vnd", ascending=False)
        print(err_df.to_string(index=False))
        err_df.to_csv(out_dir / "resnet_error_pairs.csv", index=False)
        print(f"Đã lưu: {out_dir / 'resnet_error_pairs.csv'}")

    # 5. Save outputs
    # JSON metrics
    metrics_path = out_dir / "resnet_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu: {metrics_path}")

    # Predictions CSV
    pred_df = pd.DataFrame({
        "relative_path": relative_paths,
        "y_true_vnd": y_true_vnd,
        "y_pred_vnd": y_pred_vnd
    })
    pred_csv_path = out_dir / "resnet_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Đã lưu: {pred_csv_path}")

    # 6. Plot Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    labels = [f"{v//1000}k" for v in CLASS_ORDER]

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ResNet-18 Confusion Matrix (Test Set - N=1553)", fontsize=13)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, fontsize=11)
    plt.yticks(tick_marks, labels, fontsize=11)
    plt.xlabel("Dự đoán", fontsize=11)
    plt.ylabel("Thực tế", fontsize=11)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color=color, fontsize=11)

    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Đã lưu: {cm_path}")

if __name__ == "__main__":
    main()
