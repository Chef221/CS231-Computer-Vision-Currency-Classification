# SVM RGB224 — kiểm tra đánh giá test

Phương án khóa: **HSV12+HOG 7.812 chiều, SVC RBF C=10, gamma=scale**, fit 12.494 ảnh train; không gộp train+dev. Bộ phân loại có 10.332 support vectors, tương đương 82,70% số mẫu train.

| Tập | Số ảnh | Đúng | Accuracy | Macro-F1 | MAE (đ/ảnh) |
|---|---:|---:|---:|---:|---:|
| Dev | 1.564 | 1.374 | 87,85% | 0,881213 | 27.653,45 |
| Test | 1.553 | 1.362 | 87,70% | 0,881102 | 32.891,18 |

Test có **191 lỗi**, tổng chênh lệch danh nghĩa **51.080.000đ**. MAE = 51.080.000/1.553 = 32.891,178364…đ/ảnh, tính cả ảnh đúng góp 0. Dev/test gần nhau về tỷ lệ đúng và F1 nhưng test có MAE cao hơn 5.237,73đ/ảnh; chưa đủ bằng chứng quy nguyên nhân cho overfitting hay thay đổi phân bố.

## Kiểm tra số liệu

- Dự đoán test tính lại khớp Accuracy, Macro-F1, MAE, bảng từng lớp và ma trận JSON.
- 15 hash khóa khớp code, grid và cache đang có ở bản local.
- Khóa tại `selection/` trùng byte với khóa dùng trên test.
- Ba manifest train/dev/test của SVM khớp KNN theo đường dẫn, nhãn và thứ tự ảnh; bản baseline đóng gói cho CNN cũng trùng các file SVM này.
- `run_info.json` ghi `test_used_for_tuning=false`, đúng 12.494 train và 1.553 test.

Thời gian riêng notebook 07: **218,86 giây trích feature**, **1.032,81 giây fit**, **128,83 giây predict**. Đây là log một lượt, chưa phải thời gian toàn bộ notebook và không so tốc độ phần cứng với KNN/CNN. SVM dùng Python 3.13.18, scikit-learn 1.6.1, OpenCV 4.14.0 và NumPy 2.1.3.

Notebook 08 fit lại đúng SVC trên cache train để tạo điểm quyết định và xác nhận nhãn trùng CSV test. Việc này không chọn tham số mới từ test. Phiên kiểm tra tài liệu đọc file/hash và tính lại metric, không chạy lại lượt SVC trên toàn train.

## So với KNN

SVM đúng thêm ròng **175 ảnh**, tăng **11,27 điểm phần trăm Accuracy**, tăng **0,112392 Macro-F1** và giảm MAE **21.056,02đ/ảnh** so với KNN HSV12 đã khóa. So sánh mô tả trên cùng split không chứng minh ưu thế với mọi nguồn dữ liệu.

Nguồn: [metric](metrics_test.json), [dự đoán](predictions_test.csv), [từng lớp](per_class_test.csv), [khóa đã dùng](locked_svm_config_used.json), [thông tin lần chạy](run_info.json), [hồ sơ kiểm tra](../../../../docs/SVM_REPORT/review/validation.json).
