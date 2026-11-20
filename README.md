# Phân loại mệnh giá tiền polymer Việt Nam

Đồ án CS231 khảo sát bốn phương pháp phân loại ảnh chứa một tờ tiền polymer Việt Nam. Đầu ra là một trong sáu mệnh giá từ 10.000 đến 500.000 đồng.

- [Mở demo ResNet-18 trên trình duyệt](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)
- [Báo cáo tổng hợp của nhóm](docs/RESULTS.md)
- [Danh mục tài liệu và hướng dẫn thực nghiệm](docs/README.md)
- [Hướng dẫn chạy demo ResNet-18](src/ResNet/README.md)

## Kết quả trên 1.553 ảnh test

| Phương pháp | Cấu hình đánh giá | Số ảnh đúng | Accuracy | Macro-F1 | MAE (đồng/ảnh) |
|---|---|---:|---:|---:|---:|
| [KNN](src/KNN/README.md) | HSV12, k = 1, uniform, Euclidean | 1.187 | 76,43% | 0,768710 | 53.947,20 |
| [SVM](src/SVM/README.md) | HSV12 + HOG, RBF, C = 10, gamma = scale | 1.362 | 87,70% | 0,881102 | 32.891,18 |
| [CNN](src/CNN/README.md) | TensorFlow C04_DEEP_AUG, checkpoint epoch 25 | 1.512 | 97,36% | 0,973945 | 7.250,48 |
| [ResNet-18](src/ResNet/README.md) | ImageNet pretrained, fine-tuning baseline, argmax | 1.548 | 99,68% | 0,996814 | 978,75 |

Bốn hàng dùng cùng danh sách ảnh test. Báo cáo giải thích cách đọc kết quả, khác biệt giữa các quy trình huấn luyện và giới hạn của phép so sánh.

## Bộ dữ liệu

| Mệnh giá (đồng) | Train | Dev | Test | Tổng |
|---:|---:|---:|---:|---:|
| 10.000 | 2.299 | 315 | 281 | 2.895 |
| 20.000 | 1.916 | 238 | 224 | 2.378 |
| 50.000 | 1.646 | 195 | 223 | 2.064 |
| 100.000 | 1.918 | 250 | 247 | 2.415 |
| 200.000 | 2.525 | 309 | 317 | 3.151 |
| 500.000 | 2.190 | 257 | 261 | 2.708 |
| **Tổng** | **12.494** | **1.564** | **1.553** | **15.611** |

Ảnh PNG nằm tại `DATA/{train,dev,test}/{mệnh_giá}/`. [Metadata](DATA/metadata/dataset.csv) lưu đường dẫn nguồn, nhóm ảnh, tập và mệnh giá. M0 đọc nhãn, crop thêm lề 20% mỗi phía, loại trùng pixel trong cùng lớp và chia tập theo nhóm tên ảnh gốc, có phân tầng theo nguồn và mệnh giá. [Notebook M0](src/M0)

Bốn nguồn công khai trên Roboflow Universe gồm [Banknotes Recognition v2](https://universe.roboflow.com/money-recognition/banknotes-recognition/dataset/2), [Money v1](https://universe.roboflow.com/bao-vu/money-bx6u5-a5myk/dataset/1), [Money VN Regconization v1](https://universe.roboflow.com/blindtrix/money_vn_regconization-2-cumhs/dataset/1) và [Vietnamese Currency v6](https://universe.roboflow.com/first-qkz37/vietnamese-currency-lgi9i/dataset/6). Tài liệu kèm các bản tải ghi giấy phép CC BY 4.0.

## Xem và chạy

1. Clone nông để tránh tải lịch sử chứa các phiên bản dữ liệu lớn.

```powershell
git clone --depth 1 https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification.git
cd CS231-Computer-Vision-Currency-Classification
```

2. Để đọc kết quả, mở báo cáo hoặc notebook đã lưu. Để chạy thực nghiệm, tải **cả bốn ZIP** từ [release dữ liệu M0](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/tag/m0-dataset-2026-11-01) rồi giải nén chúng tại gốc repo.
3. Dùng Python 3.11 hoặc 3.12 và cài `requirements.txt` trong môi trường ảo cho M0/KNN/SVM/CNN. Với các notebook có `PROJECT_ROOT = Path(r"C:\CS231")`, sửa biến này theo đường dẫn checkout. Xem [hướng dẫn Kaggle cho SVM/CNN](docs/REBUILD_KAGGLE.md) và [môi trường riêng của ResNet](src/ResNet/README.md).
4. SVM/CNN lưu lần chạy local vào `runs/`. KNN có thể ghi đè đầu ra trong `src/KNN/results_224/` khi chạy lại, nên dùng checkout riêng nếu cần giữ bản công bố.

```powershell
python -m pip install -r requirements.txt
python -X utf8 -m unittest discover -s tests
```

Các mô hình dùng [chuẩn đánh giá chung](docs/METRICS_STANDARD.md). Truyền nhãn bằng giá trị VND vào `evaluate_classification` để tính Accuracy, Macro-F1 và MAE.
