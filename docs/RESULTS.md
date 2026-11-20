# Báo cáo tổng hợp thực nghiệm CS231

## Bài toán và thiết kế đánh giá

Đầu vào là ảnh chứa một tờ tiền polymer Việt Nam thuộc sáu mệnh giá hỗ trợ. Nhóm khảo sát hai hướng. KNN và SVM dùng vector đặc trưng thiết kế trước. CNN và ResNet-18 học đặc trưng từ ảnh.

Bộ dữ liệu cuối gồm 15.611 ảnh từ bốn nguồn Roboflow Universe. Nhóm dùng 12.494 ảnh train, 1.564 ảnh dev và 1.553 ảnh test. Ảnh được resize về 224 × 224 khi đưa vào mô hình. Mỗi pipeline giữ quy trình đọc, chuẩn hóa và augmentation tương ứng. Bốn bộ dự đoán test đã được đối chiếu với cùng đường dẫn và nhãn trong [metadata](../DATA/metadata/dataset.csv).

Nhóm chọn cấu hình hoặc checkpoint trên dev, ưu tiên Macro-F1 cao và dùng MAE thấp để xử lý hòa theo quy tắc từng pipeline. Accuracy đo tỷ lệ ảnh đúng. Macro-F1 lấy trung bình F1 của sáu mệnh giá. MAE lấy trung bình độ lệch tuyệt đối giữa hai mệnh giá trên toàn bộ test. Ảnh đúng góp 0 đồng. [Chuẩn đánh giá](METRICS_STANDARD.md)

## Phương pháp

| Mô hình | Thiết kế và cấu hình đánh giá |
|---|---|
| KNN | So sánh năm biểu diễn HSV12, HOG, HOG-PCA và hai biến thể nối đặc trưng. Thử 20 cấu hình mỗi biểu diễn trên dev. HSV12, k = 1, uniform, Euclidean đứng đầu. [Báo cáo KNN](KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md) |
| SVM | Thử 35 cấu hình từ năm biểu diễn và bảy thiết lập SVC. HSV12 + HOG với RBF, C = 10, gamma = scale đứng đầu. [Báo cáo SVM](SVM_REPORT/BAO_CAO_THUC_NGHIEM_SVM.md) |
| CNN | So sánh bốn cấu hình TensorFlow với ảnh RGB hoặc xám, có hoặc không augmentation, và ba hoặc bốn khối tích chập. C04_DEEP_AUG dùng RGB, bốn khối, augmentation trên train và checkpoint epoch 25. [Báo cáo CNN](CNN_REPORT/BAO_CAO_THUC_NGHIEM_CNN.md) |
| ResNet-18 | Fine-tuning từ trọng số ImageNet với đầu ra sáu lớp. Baseline dùng cross-entropy, AdamW với learning rate 0,0001, augmentation trên train và chuẩn hóa ImageNet. Bảng dùng baseline và quyết định argmax. [Notebook huấn luyện](../src/ResNet/notebooks/train_resnet_kaggle.ipynb) · [Notebook đánh giá](../src/ResNet/notebooks/resnet18_evaluation.ipynb) |

## Kết quả trên cùng tập test

| Mô hình | Đúng | Sai | Accuracy | Macro-F1 | MAE (đồng/ảnh) | Bằng chứng |
|---|---:|---:|---:|---:|---:|---|
| KNN | 1.187 | 366 | 76,43% | 0,768710 | 53.947,20 | [JSON](../src/KNN/results_224/test_final/metrics_test.json) · [Dự đoán](../src/KNN/results_224/test_final/predictions_test.csv) |
| SVM | 1.362 | 191 | 87,70% | 0,881102 | 32.891,18 | [JSON](../src/SVM/results_224/test_final/metrics_test.json) · [Dự đoán](../src/SVM/results_224/test_final/predictions_test.csv) |
| CNN | 1.512 | 41 | 97,36% | 0,973945 | 7.250,48 | [JSON](../src/CNN/results_224/test_final/metrics_test.json) · [Dự đoán](../src/CNN/results_224/test_final/predictions_test.csv) |
| ResNet-18 | 1.548 | 5 | 99,68% | 0,996814 | 978,75 | [JSON](../src/ResNet/results/metrics_test.json) · [Dự đoán](../src/ResNet/results/predictions_test.csv) |

Số liệu được tính lại từ từng bộ dự đoán bằng hàm đánh giá chung. Hàng CNN dùng lượt TensorFlow C04. Hàng ResNet-18 được xác minh bằng checkpoint đi kèm demo và khớp kết quả đã lưu trong notebook.

## Phân tích và giới hạn

- ResNet-18 cho Accuracy và Macro-F1 cao nhất, đồng thời có MAE thấp nhất trên tập test này. Năm lỗi có tổng lệch 1.520.000 đồng. Lỗi lệch lớn nhất là ảnh 500.000 đồng được dự đoán thành 20.000 đồng. [Chi tiết ResNet-18](../src/ResNet/results/metrics_test.json)
- Khi đối chiếu từng ảnh giữa SVM và CNN, cả hai đúng 1.343 ảnh, chỉ CNN đúng 169 ảnh, chỉ SVM đúng 19 ảnh và cả hai sai 22 ảnh. Điểm tổng thể và mức độ trùng nhau của lỗi trả lời hai câu hỏi khác nhau.
- Kết quả so sánh các **pipeline đã chạy**. Kiến trúc, đặc trưng, augmentation và trọng số pretrained khác nhau. Thiết kế này chưa đo riêng mức đóng góp của từng yếu tố.
- Tập test có lịch sử được xem trong các vòng khảo sát. Kết quả mô tả một lần chia trên bốn nguồn hiện có. Cần dữ liệu ngoài nguồn để đánh giá khả năng áp dụng ở môi trường chụp mới.
- MAE đo chênh lệch mệnh giá danh nghĩa. Thiệt hại thực tế phụ thuộc vào cách dự đoán được dùng trong giao dịch. Các thí nghiệm cost-sensitive của ResNet là phần khảo sát riêng trong notebook huấn luyện.

## Demo và tái lập

Demo dùng ResNet-18 baseline. Giao diện hiển thị Top 1 hoặc Top 3, với argmax là quyết định nhận diện. Bảng trên đánh giá phân loại trên toàn bộ test. Phạm vi demo là ảnh chứa một tờ tiền thuộc sáu lớp. [Hướng dẫn chạy demo và đánh giá lại](../src/ResNet/README.md)

Để xem nhanh kết quả đã lưu, dùng notebook `00_READ_RESULTS.ipynb` của [SVM](../src/SVM/224/00_READ_RESULTS.ipynb) và [CNN](../src/CNN/224/00_READ_RESULTS.ipynb). Xem [tài liệu tham khảo](REFERENCES.md) và [hướng dẫn tái lập SVM/CNN trên Kaggle](REBUILD_KAGGLE.md).
