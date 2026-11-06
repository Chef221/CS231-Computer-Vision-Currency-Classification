# Phân loại mệnh giá tiền polymer Việt Nam

## Bộ dữ liệu

Bộ cuối gồm 15.611 ảnh PNG, chia theo thư mục lớp và tập dữ liệu:

| Mệnh giá | Train | Dev | Test | Tổng |
|---:|---:|---:|---:|---:|
| 10.000 | 2.299 | 315 | 281 | 2.895 |
| 20.000 | 1.916 | 238 | 224 | 2.378 |
| 50.000 | 1.646 | 195 | 223 | 2.064 |
| 100.000 | 1.918 | 250 | 247 | 2.415 |
| 200.000 | 2.525 | 309 | 317 | 3.151 |
| 500.000 | 2.190 | 257 | 261 | 2.708 |
| **Tổng** | **12.494** | **1.564** | **1.553** | **15.611** |

Thư mục có dạng `DATA/{train,dev,test}/{10000,20000,50000,100000,200000,500000}/`. Tên ảnh là `<mệnh_giá>_<số thứ tự sáu chữ số>.png`; số thứ tự tăng theo từng lớp trên toàn bộ dữ liệu. `DATA/metadata/dataset.csv` lưu nguồn, đường dẫn ảnh gốc và ảnh crop, nhóm ảnh, split, mệnh giá và đường dẫn kết quả tương đối `DATA`.

## Tải dự án và dữ liệu

Clone nông để lấy code và metadata hiện tại, không tải lịch sử cũ có chứa ảnh lớn:

```powershell
git clone --depth 1 https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification.git
cd CS231-Computer-Vision-Currency-Classification
```

Nếu đã có bản clone, dùng `git pull` sau khi lưu hoặc commit công việc local cần giữ; không dùng lệnh reset hoặc clean để cập nhật.

Tải bốn ZIP từ [GitHub Release m0-dataset-2026-11-01](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/tag/m0-dataset-2026-11-01). Mỗi ZIP chứa đường dẫn bắt đầu bằng `DATA/`; hai tệp train chứa các phần ảnh khác nhau. Tệp dev có thêm `DATA/metadata/dataset.csv`. `manifest.json` ghi kích thước, SHA-256 và nguồn của bộ dữ liệu.

- [data-train-1-of-2.zip](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-train-1-of-2.zip)
- [data-train-2-of-2.zip](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-train-2-of-2.zip)
- [data-dev.zip](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-dev.zip)
- [data-test.zip](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/data-test.zip)
- [manifest.json](https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification/releases/download/m0-dataset-2026-11-01/manifest.json)

```powershell
Expand-Archive .\data-train-1-of-2.zip -DestinationPath . -Force
Expand-Archive .\data-train-2-of-2.zip -DestinationPath . -Force
Expand-Archive .\data-dev.zip -DestinationPath . -Force
Expand-Archive .\data-test.zip -DestinationPath . -Force
```

Dùng train để huấn luyện, dev để chọn mô hình/cấu hình, và chỉ dùng test để đánh giá cuối. Resize ảnh trong pipeline của mô hình khi cần; ảnh PNG trong bộ dữ liệu được giữ nguyên độ phân giải.

## Quy trình M0

Cài thư viện và mở notebook trong `src/M0` theo thứ tự `00` đến `05`:

```powershell
pip install -r requirements.txt
```

1. `00_COUNT_RAW_DATA.ipynb`: đếm ảnh và nhãn nguồn.
2. `01_READ_LABELS.ipynb`: đọc nhãn, tổng hợp sáu mệnh giá và xem ví dụ.
3. `02_CROP_BANKNOTES.ipynb`: tạo một crop cho mỗi nhãn; vùng crop thêm padding 20% mỗi phía. Ảnh giữ nguyên kích thước, không augment.
4. `03_REMOVE_EXACT_DUPLICATES.ipynb`: bỏ bản trùng pixel hoàn toàn trong cùng mệnh giá; giữ ảnh nguồn.
5. `04_SPLIT_TRAIN_DEV_TEST.ipynb`: chia train/dev/test theo nhóm tên ảnh gốc với seed cố định.
6. `05_EXPORT_FINAL_DATA.ipynb`: sao chép và đặt tên ảnh cuối theo thư mục lớp.

Các ảnh nguồn chưa được đóng gói trong release. Muốn chạy lại các bước từ đầu, tải bốn bộ nguồn từ trang Roboflow bên dưới và đặt chúng theo cấu trúc mà `00_COUNT_RAW_DATA.ipynb` đọc dưới `DATA_RAW/`.

## Nguồn và ghi công

Bốn bộ nguồn ghi giấy phép CC BY 4.0 trong tệp `README.dataset.txt`/`README.roboflow.txt` đi kèm:

- [Banknotes Recognition v2](https://universe.roboflow.com/money-recognition/banknotes-recognition/dataset/2)
- [Money v1](https://universe.roboflow.com/bao-vu/money-bx6u5-a5myk/dataset/1)
- [Money_VN_Regconization v1](https://universe.roboflow.com/blindtrix/money_vn_regconization-2-cumhs/dataset/1)
- [Vietnamese Currency v6](https://universe.roboflow.com/first-qkz37/vietnamese-currency-lgi9i/dataset/6)

Bộ cuối là kết quả xử lý và kết hợp các nguồn trên: chỉ giữ sáu mệnh giá mục tiêu, crop theo nhãn, loại trùng pixel hoàn toàn trong cùng lớp, rồi chia lại theo nhóm. Việc ghi nguồn không hàm ý các tác giả nguồn bảo trợ dự án này.

## Chuẩn đánh giá

KNN, SVM, CNN và ResNet dùng **Macro-F1 làm tiêu chí chính trên dev**; nếu bằng nhau thì ưu tiên **MAE theo VND thấp hơn**. Kết quả cuối báo cáo Accuracy, Macro-F1, MAE, Precision/Recall/F1 từng lớp, ma trận nhầm lẫn và bảng các cặp lỗi.

Hướng dẫn import, mapping và lưu kết quả nằm tại [docs/METRICS_STANDARD.md](docs/METRICS_STANDARD.md). Công thức và ví dụ được giải thích trong [M2.2](docs/M2.2%20-%20Mô%20tả%20độ%20đo%20đánh%20giá.md). Truyền nhãn thật/dự đoán là giá trị VND thực tế, không phải chỉ số lớp:

```python
from src.evaluation.value_metrics import evaluate_classification

metrics = evaluate_classification(y_true_vnd, y_pred_vnd)
print("Macro-F1:", metrics["macro_f1"])
print("Accuracy:", metrics["accuracy"])
print("MAE (VND/ảnh):", metrics["mae_vnd"])
print(metrics["confusion_matrix"])
```

Giữ nguyên import cũ nhưng đổi chỗ đọc `macro_value_f1` sang `macro_f1`; các trường value metric cũ đã bỏ. Cách chuyển được ghi trong hướng dẫn dùng chung. Hàm đánh giá vẫn là một hàm tại [src/evaluation/value_metrics.py](src/evaluation/value_metrics.py).

Chạy kiểm tra bằng `python -m unittest discover -s tests -p test_value_metrics.py -v`.

## Đề cương thực nghiệm

[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) quy định ba phần: so sánh bốn mô hình, phân tích lỗi mệnh giá và thí nghiệm từ chối dự đoán ở bước mở rộng. Các nguồn và phần nhóm kế thừa được ghi trong [docs/REFERENCES.md](docs/REFERENCES.md).

So sánh chính luôn dùng toàn bộ test sau khi khóa cấu hình. Cơ chế từ chối hiện mới là đề cương, chưa được triển khai; kết quả trên phần được trả lời sau này phải báo cáo riêng cùng coverage.

## Thực nghiệm KNN

Notebook hiện tại nằm trong [src/KNN/224](src/KNN/224/README.md); thứ tự chạy và kết quả được hướng dẫn tại [README KNN](src/KNN/README.md). Lượt thực nghiệm dùng RGB224, không augmentation, năm biến thể thuộc ba họ feature: HSV12, HOG và kết hợp; PCA là biến thể của nhánh HOG. Tổng cộng 100 cấu hình được đánh giá trên dev với Euclidean.

Phương án khóa là HSV12, k=1, uniform. Trên 1.553 ảnh test hiện có, mô hình nhận đúng 1.187 ảnh: Accuracy 76,43%, Macro-F1 0,768710, MAE 53.947,20đ/ảnh. Test từng được đánh giá trong khảo sát trước, nên không gọi đây là holdout hoàn toàn chưa xem. Không trộn các kết quả cũ vào bảng RGB224.

[Báo cáo KNN](docs/KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md) trình bày từ tiền xử lý đến phân tích lỗi, với năm bảng và sáu hình từ dữ liệu thật. [Cấu hình khóa](src/KNN/results_224/selection/locked_knn_config.json), [kết quả test](src/KNN/results_224/test_final/TEST_REVIEW.md) và [phân tích lỗi](src/KNN/results_224/error_analysis/ERROR_ANALYSIS_REVIEW.md) được lưu riêng để đối chiếu. Các notebook/kết quả khảo sát cũ đã được dọn khỏi checkout hiện tại.

## Thực nghiệm SVM

[src/SVM](src/SVM/README.md) có chín notebook RGB224 theo cùng năm biến thể đặc trưng của KNN, chạy trên Kaggle và chọn theo 35 cấu hình SVC trên dev. Phương án khóa là HSV12+HOG với SVC RBF, `C=10`, `gamma='scale'`. Trên cùng 1.553 ảnh test, mô hình đúng 1.362 ảnh: Accuracy **87,70%**, Macro-F1 **0,881102**, MAE **32.891,18đ/ảnh**.

[Báo cáo SVM](docs/SVM_REPORT/BAO_CAO_THUC_NGHIEM_SVM.md) trình bày dữ liệu, đặc trưng, cấu hình, test, phân tích lỗi và so sánh với KNN. [Cấu hình khóa](src/SVM/results_224/selection/locked_svm_config.json), [kết quả test](src/SVM/results_224/test_final/TEST_REVIEW.md) và [phân tích lỗi](src/SVM/results_224/error_analysis/ERROR_ANALYSIS_REVIEW.md) giữ số liệu và bằng chứng riêng. NPY/PKL được giữ local/ngoài Git.

[Mục lục báo cáo](docs/README.md) tập hợp KNN, SVM và CNN; [hồ sơ rà soát](docs/MODEL_FORMAT_REVIEW.md) ghi kết quả đối chiếu code, notebook và số liệu theo cùng cấu trúc.

## Thực nghiệm CNN

[src/CNN](src/CNN/README.md) có tám notebook chạy trên Kaggle: chuẩn bị RGB224, huấn luyện bốn CNN từ đầu, chọn checkpoint trên dev, khóa trọng số, đánh giá test và phân tích lỗi. Bốn cấu hình khảo sát RGB, ảnh xám, augmentation và thêm một khối tích chập. Các notebook giữ cùng split và chuẩn Macro-F1/Accuracy/MAE theo VND của dự án, kèm learning curve, feature map, PCA của biểu diễn GAP và Grad-CAM.

Lượt CNN đã chạy trên Kaggle với Tesla T4. Phương án khóa là **C04_DEEP_AUG**, CNN RGB bốn khối 32–64–128–256 có augmentation trên train, checkpoint **epoch 24**. Trên cùng 1.553 ảnh test, mô hình đúng **1.507 ảnh**: Accuracy **97,04%**, Macro-F1 **0,969963**, MAE **9.104,96đ/ảnh**. CNN hơn SVM đã khóa 145 ảnh đúng và giảm MAE 72,32%; so sánh này đánh giá các pipeline hoàn chỉnh trên một split đã có lịch sử sử dụng.

[Báo cáo CNN](docs/CNN_REPORT/BAO_CAO_THUC_NGHIEM_CNN.md) trình bày bốn cấu hình dev, thời gian chạy thật, test, Grad-CAM và so sánh ghép cặp với KNN/SVM. [Cấu hình khóa](src/CNN/results_224/selection/locked_cnn_config.json), [kiểm tra test](src/CNN/results_224/test_final/TEST_REVIEW.md) và [phân tích lỗi](src/CNN/results_224/error_analysis/ERROR_ANALYSIS_REVIEW.md) được lưu riêng để đối chiếu. Thư mục [kết quả CNN](src/CNN/results_224/README.md) giữ CSV/JSON/hình; checkpoint và ma trận lớn được giữ local/ngoài Git.
