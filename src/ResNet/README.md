# ResNet-18 và demo

Mô hình fine-tune ResNet-18 từ trọng số ImageNet để phân loại sáu mệnh giá. Baseline dùng cross-entropy, AdamW, resize RGB 224 × 224 và chuẩn hóa ImageNet. Augmentation áp dụng trên train. Quyết định của demo là argmax.

Trên 1.553 ảnh test, checkpoint đi kèm demo dự đoán đúng 1.548 ảnh, đạt Accuracy 99,68%, Macro-F1 0,996814 và MAE 978,75 đồng/ảnh. [Báo cáo chung](../../docs/RESULTS.md) · [Chỉ số từng lớp và cặp nhầm](results/metrics_test.json) · [Dự đoán từng ảnh](results/predictions_test.csv) · [Ma trận nhầm lẫn](results/confusion_matrix.png)

## Thực nghiệm

- [Notebook huấn luyện Kaggle](notebooks/train_resnet_kaggle.ipynb) chứa baseline và các cấu hình cost-sensitive.
- [Notebook đánh giá](notebooks/resnet18_evaluation.ipynb) chứa bảng, ma trận nhầm lẫn và ví dụ lỗi.
- [Checkpoint PyTorch](../../web_demo/backend/weights/best_resnet18_baseline.pth) dùng cho FastAPI và lệnh đánh giá dưới đây.
- [Mô hình ONNX](../../docs/models/resnet18_vnd.onnx) dùng cho bản demo tĩnh. Khi phát hành GitHub Pages, chọn thư mục `/docs` của nhánh chứa bản build.

Từ thư mục gốc repo, cài môi trường riêng rồi chạy đánh giá với dữ liệu M0 đã giải nén. Kết quả được ghi vào thư mục chỉ định.

```powershell
python -m pip install -r src/ResNet/requirements.txt
python -m src.ResNet.eval --data-dir DATA --checkpoint web_demo/backend/weights/best_resnet18_baseline.pth --output-dir output/resnet
```

## Chạy demo local

```powershell
python -m pip install -r web_demo/requirements.txt
python web_demo/backend/app.py
```

Mở [giao diện local](http://localhost:8080/) hoặc [Swagger API](http://localhost:8080/docs). Server dùng bản build trong `web_demo/frontend/dist` nếu có, rồi dùng `docs/index.html` khi thư mục build chưa có.

Để build giao diện từ mã nguồn, dùng Node.js 22.12+ hoặc một bản tương thích với phiên bản Vite đã khóa.

```powershell
npm ci --prefix web_demo/frontend
npm run build --prefix web_demo/frontend
```

Bản tĩnh dùng ONNX Runtime Web và nạp tài nguyên runtime từ CDN. Thư mục `docs` chứa giao diện đã build và mô hình ONNX.

[CSV gốc từ nhánh ResNet](notebooks/resnet_test_predictions.csv) ghi 8 lỗi. Bảng baseline dùng bộ dự đoán đánh giá lại từ checkpoint đi kèm demo, ghi 5 lỗi.
