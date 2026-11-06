# Báo cáo thực nghiệm RGB224

Các báo cáo dưới đây dùng cùng sáu mệnh giá và split 12.494 train / 1.564 dev / 1.553 test. Mỗi mô hình được chọn theo Macro-F1 dev chưa làm tròn; hòa chính xác thì MAE thấp hơn. So sánh test dùng đúng ảnh, nhãn và thứ tự, theo hàm metric chung.

| Phần | Báo cáo | Đặc trưng / phương án đã khóa |
|---|---|---|
| KNN | [Báo cáo KNN](KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md) | HSV12; k=1, uniform, Euclidean |
| SVM | [Báo cáo SVM](SVM_REPORT/BAO_CAO_THUC_NGHIEM_SVM.md) | HSV12+HOG; SVC RBF C=10, gamma=scale |
| CNN | [Báo cáo CNN](CNN_REPORT/BAO_CAO_THUC_NGHIEM_CNN.md) | RGB, bốn khối 32–64–128–256, augmentation train; epoch 24 |

## Kết quả toàn test

| Mô hình | Đúng / 1.553 | Accuracy | Macro-F1 | MAE (đ/ảnh) |
|---|---:|---:|---:|---:|
| KNN | 1.187 | 76,43% | 0,768710 | 53.947,20 |
| SVM | 1.362 | 87,70% | 0,881102 | 32.891,18 |
| CNN | 1.507 | 97,04% | 0,969963 | 9.104,96 |

Đọc phần phương pháp và giới hạn cùng số liệu. CNN học đặc trưng và có augmentation ở cấu hình được chọn; KNN/SVM dùng đặc trưng thủ công. Bảng đánh giá các pipeline hoàn chỉnh trên một split đã có lịch sử được xem, không cô lập riêng hiệu quả thuật toán hoặc chứng minh tổng quát hóa ngoài nguồn.

## Cấu trúc trình bày

Cả ba phần đi từ mục tiêu/dữ liệu, tiền xử lý, mô hình và tham số, thí nghiệm từng biến thể, tổng hợp dev/khóa phương án, test, phân tích lỗi đến kết luận và nguồn tham khảo. README của từng thư mục ghi cách dựng hình/báo cáo và đối chiếu số liệu; giữ thư mục hình cạnh Markdown khi ghép báo cáo học phần. Phụ lục CSV giữ độ chính xác gốc.

- [Đề cương chung](EXPERIMENT_PLAN.md).
- [Chuẩn metric và mapping VND](METRICS_STANDARD.md).
- [Hồ sơ rà soát SVM/CNN theo cấu trúc KNN](MODEL_FORMAT_REVIEW.md).

Các thư mục báo cáo nằm trên từng nhánh tương ứng. Muốn thấy chúng trên GitHub, cần chọn đúng nhánh hoặc dùng nhánh đã tích hợp đầy đủ; nội dung của một nhánh không tự xuất hiện trên các nhánh khác.
