# Phân loại mệnh giá tiền polymer Việt Nam

Đồ án môn **CS231 — Computer Vision**, gồm sáu mệnh giá: 10.000, 20.000, 50.000, 100.000, 200.000 và 500.000 đồng.

Nhóm dùng **KNN làm baseline**, so sánh với SVM dùng đặc trưng thủ công và CNN học trực tiếp từ ảnh. CNN viết bằng TensorFlow/Keras, huấn luyện từ đầu. Repository này tập trung vào KNN, SVM và CNN; phần ResNet/demo của thành viên khác chưa có trong mã nguồn này.

## Kết quả cuối của nhóm

Các số liệu dưới đây lấy từ kết quả thực nghiệm đã lưu, cùng bộ kết quả dùng trong slide cuối. Không phải số liệu chạy lại khi sắp xếp mã nguồn.

| Mô hình | Cấu hình chọn trên dev | Đúng / 1.553 ảnh test | Accuracy | Macro-F1 | MAE (đồng/ảnh) |
|---|---|---:|---:|---:|---:|
| KNN | HSV12, k=1, uniform, Euclidean | 1.187 | 76,43% | 0,768710 | 53.947,20 |
| SVM | HSV12 + HOG, RBF, C=10, gamma=scale | 1.362 | 87,70% | 0,881102 | 32.891,18 |
| CNN | C04, RGB, 4 khối, augmentation, epoch 25 | 1.512 | **97,36%** | **0,973945** | **7.250,48** |

Chọn cấu hình bằng **Macro-F1 dev cao nhất**, nếu bằng nhau thì MAE thấp nhất. Train để học, dev để chọn, test để đánh giá cấu hình đã chọn. Không gộp train và dev. Test từng được xem trong các lượt trước; cần dữ liệu mới để kiểm tra khả năng áp dụng ngoài bộ này.

## Bắt đầu đọc code

| Phần | Nơi bắt đầu | Nội dung |
|---|---|---|
| KNN | [Hướng dẫn KNN](src/KNN/README.md) | Baseline gốc, giữ nguyên code |
| SVM | [Hướng dẫn SVM](src/SVM/README.md) | Giống bố cục KNN: chuẩn bị → đặc trưng → chọn → test → lỗi |
| CNN | [Hướng dẫn CNN](src/CNN/README.md) | Chuẩn bị → xây mạng → train → chọn → test → lỗi |
| Kết quả có sẵn | [Tổng hợp kết quả](docs/RESULTS.md) | Bảng, ma trận nhầm lẫn và ảnh phân tích lỗi |
| Giải thích thuyết trình | [Slide và code](docs/SLIDE_CODE.md) | Mỗi nội dung trên slide được tạo ở đâu |
| Chạy trên Kaggle | [Hướng dẫn Kaggle](docs/REBUILD_KAGGLE.md) | Gắn code, ảnh và Output bước trước |

SVM và CNN có notebook **`00_READ_RESULTS.ipynb`** để xem kết quả đã có. Không cần tải ảnh hoặc huấn luyện lại để đọc bảng và hình trên GitHub.

```text
src/
├── M0/             # Chuẩn bị bộ ảnh ban đầu
├── KNN/            # Baseline gốc
├── SVM/            # 9 bước thực nghiệm + notebook xem kết quả
├── CNN/            # 8 bước thực nghiệm + notebook xem kết quả
├── experiments/    # Đường dẫn và các hàm vẽ dùng chung
├── evaluation/     # Tính Accuracy, Macro-F1, MAE và bảng lỗi
└── preprocessing.py

docs/               # Giải thích, kết quả và tài liệu tham khảo
runs/               # Kết quả chạy lại SVM/CNN; Git bỏ qua
```

`src/<mô hình>/results_224/` chứa bộ kết quả đã công bố. Với SVM/CNN, notebook mới ghi riêng vào `runs/` khi chạy local, nên không ghi đè bộ kết quả dùng trong slide. File model và ma trận đặc trưng lớn không đưa vào Git; có thể tạo lại bằng các notebook.

## Cài đặt và chạy

Dùng Python 3.11–3.12. Tại thư mục gốc dự án:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Đọc README của từng mô hình rồi chạy theo thứ tự từ 01. Các notebook SVM và CNN đã xóa log chạy cũ; ô xây CNN vẫn viết trực tiếp bằng `models.Sequential()`, `model.add(...)` và `model.summary()`.

Giữ baseline SVM gồm 5 biểu diễn × 7 cấu hình = 35 cấu hình. Giữ CNN C01_RGB, C02_GRAY, C03_RGB_AUG và C04_DEEP_AUG; không chỉ giữ cấu hình tốt nhất. Việc sắp xếp code không thay đổi lưới SVM, kiến trúc CNN, augmentation hay các thông số huấn luyện của lượt cuối.

## Dữ liệu

| Mệnh giá | Train | Dev | Test |
|---:|---:|---:|---:|
| 10.000 | 2.299 | 315 | 281 |
| 20.000 | 1.916 | 238 | 224 |
| 50.000 | 1.646 | 195 | 223 |
| 100.000 | 1.918 | 250 | 247 |
| 200.000 | 2.525 | 309 | 317 |
| 500.000 | 2.190 | 257 | 261 |
| **Tổng** | **12.494** | **1.564** | **1.553** |

Bộ ảnh có 15.611 PNG. Đặt ảnh theo `DATA/{train,dev,test}/{mệnh_giá}/*.png`. Ảnh gốc được giữ nguyên; các mô hình đọc và resize về 224×224 trong bộ nhớ.

Tải tại [GitHub Release m0-dataset-2026-11-01](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/tag/m0-dataset-2026-11-01), giải nén cả bốn phần vào cùng thư mục gốc repo:

- [Train phần 1](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-train-1-of-2.zip), [train phần 2](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-train-2-of-2.zip).
- [Dev](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-dev.zip), [test](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-test.zip).

`src/M0/00…05` lưu các bước đọc nhãn, crop có padding 20%, bỏ ảnh trùng pixel, chia tập và xuất ảnh. Muốn chạy lại M0 cần tải ảnh nguồn. Dữ liệu đã xử lý và metadata không thay đổi trong lần dọn code này.

Bốn nguồn Roboflow ghi giấy phép CC BY 4.0 trong tệp đi kèm:

- [Banknotes Recognition v2](https://universe.roboflow.com/money-recognition/banknotes-recognition/dataset/2).
- [Money v1](https://universe.roboflow.com/bao-vu/money-bx6u5-a5myk/dataset/1).
- [Money VN Regconization v1](https://universe.roboflow.com/blindtrix/money_vn_regconization-2-cumhs/dataset/1).
- [Vietnamese Currency v6](https://universe.roboflow.com/first-qkz37/vietnamese-currency-lgi9i/dataset/6).

## Kiểm tra và đóng gói

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/build_bundle.py --kind github
python tools/build_bundle.py --kind kaggle
```

Cách đưa lên GitHub: [hướng dẫn ngắn](docs/GITHUB.md).

Các kiểm tra dùng ảnh nhỏ tạm thời và đối chiếu các CSV đã công bố; không huấn luyện toàn bộ bộ dữ liệu. Xem [chuẩn đánh giá](docs/METRICS_STANDARD.md) để dùng đúng nhãn mệnh giá khi tính MAE.
