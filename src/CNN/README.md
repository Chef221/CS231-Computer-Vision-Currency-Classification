# CNN TensorFlow — phân loại mệnh giá

Giữ cấu hình của lượt CNN cuối trong slide, viết bằng TensorFlow/Keras. Mỗi notebook train khai báo trực tiếp `models.Sequential()`, từng `model.add(...)` và `model.summary()`.

## Kết quả đã có

Mở [00_READ_RESULTS.ipynb](224/00_READ_RESULTS.ipynb) để xem bảng và hình ngay trên GitHub. Cấu hình chọn là **C04_DEEP_AUG, epoch 25**: đúng **1.512/1.553**, Accuracy **97,36%**, Macro-F1 **0,973945**, MAE **7.250,48 đồng/ảnh**.

Giữ bốn cấu hình để so sánh với baseline:

| Cấu hình | Đầu vào | Các khối Conv | Augmentation |
|---|---|---|---|
| C01_RGB | RGB | 32–64–128 | Không |
| C02_GRAY | Ảnh xám | 32–64–128 | Không |
| C03_RGB_AUG | RGB | 32–64–128 | Có |
| C04_DEEP_AUG | RGB | 32–64–128–256 | Có |

Mỗi khối: hai Conv3×3, mỗi Conv kèm BatchNorm và ReLU; sau đó MaxPool2×2 và SpatialDropout0.1. Cuối mạng: GAP → Dense128 ReLU → Dropout0.5 → Dense6 softmax. C04 có vector GAP 256 chiều; các mạng ba khối có vector GAP 128 chiều.

## Chạy các notebook

**01 → 02 → 03 → 04 → 05 → 06 → 07 → 08**:

1. Chuẩn bị train/dev, minh họa resize và augmentation.
2. Train RGB ba khối, không augmentation.
3. Train ảnh xám ba khối.
4. Train RGB ba khối có augmentation.
5. Train RGB bốn khối có augmentation.
6. Chọn checkpoint trên dev, xem feature maps và PCA vector GAP.
7. Nạp checkpoint đã chọn, đánh giá test.
8. Xem ảnh lỗi, Grad-CAM và xuất kết quả.

AdamW lr0.001, weight_decay0.0001, beta0.9/0.999, epsilon1e-8; batch64; clip gradient norm5; tối đa25 epoch; learning rate giảm theo cosine tới1e-5. Chọn theo Macro-F1 dev rồi MAE, patience6. Seed42. GPU dùng mixed precision khi train; dev/test float32.

Augmentation OpenCV chỉ trên train của C03/C04: xoay±12°, dịch±5%, scale0.95–1.05, viền xám127; contrast0.9–1.1 và cộng sáng±10% thang pixel. Thực hiện trước chia255. HSV/HOG chỉ minh họa để đối chiếu với KNN, không phải feature đầu vào CNN.

## Code nằm ở đâu?

- [model.py](model.py): các lớp, xây mạng, AdamW và cosine LR.
- [data.py](data.py): cấu hình, đọc ảnh theo batch, shuffle và augmentation.
- [pipeline.py](pipeline.py): train, chấm dev, lưu best.keras, chọn và test.
- [analysis.py](analysis.py): augmentation, feature maps, PCA và Grad-CAM.

Lớp `TorchBatchNormalization` được viết hoàn toàn bằng TensorFlow. Giữ lớp này để cách cập nhật thống kê giống lượt đã chạy; tên được giữ để nạp checkpoint `.keras` cũ của nhóm. **Không cần cài PyTorch.** Xem [giải thích CNN](GIAI_THICH_CNN.md) nếu cần trình bày trước lớp.

Bộ kết quả đã công bố nằm trong `results_224/`. Chạy lại ghi vào `runs/CNN` hoặc `/kaggle/working/CS231_CNN/results_224`. Bản học tập lưu best.keras và lịch sử, chạy lại hàm train sẽ bắt đầu từ epoch1. Không có phần chạy tiếp tự động phức tạp. Muốn giữ một lượt chạy, sao chép Output trước khi chạy lại.

File `.keras` không đưa vào Git vì kích thước lớn; trên máy hiện tại vẫn giữ model từ bộ kết quả nhóm gửi. Sau khi clone có thể đọc toàn bộ CSV/hình ngay, còn muốn dự đoán ảnh mới thì chạy notebook để tạo model hoặc gắn model đã lưu của nhóm. Kết quả chạy lại có thể khác do môi trường và tính ngẫu nhiên.

Cài riêng: `pip install -r src/CNN/requirements.txt`. Kaggle nên dùng TensorFlow có sẵn. Xem [hướng dẫn Kaggle](../../docs/REBUILD_KAGGLE.md).
