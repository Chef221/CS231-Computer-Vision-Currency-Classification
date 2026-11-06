# Kết quả CNN RGB224

Thư mục giữ output tích lũy của chuỗi notebook 01–08 đã chạy trên Kaggle. Phương án khóa là **C04_DEEP_AUG**, CNN RGB bốn khối 32–64–128–256 có augmentation train, checkpoint **epoch 24**. Trên 1.553 ảnh test, mô hình đúng **1.507 ảnh**, sai **46 ảnh**: Accuracy **97,04%**, Macro-F1 **0,969963**, MAE **9.104,96đ/ảnh**, tổng chênh lệch danh nghĩa **14.140.000đ**.

## Đọc kết quả

| Nội dung | File |
|---|---|
| So sánh bốn cấu hình dev | [DEV_COMPARISON.md](DEV_COMPARISON.md) |
| Cấu hình/checkpoint đã chọn | [locked_cnn_config.json](selection/locked_cnn_config.json) |
| Kiểm tra test | [TEST_REVIEW.md](test_final/TEST_REVIEW.md) |
| Phân tích lỗi và Grad-CAM | [ERROR_ANALYSIS_REVIEW.md](error_analysis/ERROR_ANALYSIS_REVIEW.md) |
| So sánh toàn test | [metrics_comparison_test.csv](comparison/metrics_comparison_test.csv) |
| So sánh đúng/sai từng ảnh | [paired_summary_test.csv](comparison/paired_summary_test.csv) |
| Báo cáo có hình | [BAO_CAO_THUC_NGHIEM_CNN.md](report/BAO_CAO_THUC_NGHIEM_CNN.md) |
| Hồ sơ kiểm tra khi nhập output | [validation.json](../../../docs/CNN_REPORT/review/validation.json) |

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `inputs` | Manifest train/dev, protocol, mapping, kiến trúc, ảnh minh họa |
| `candidates/C01_RGB` | CNN RGB cơ sở: 25 epoch, chọn epoch 25 |
| `candidates/C02_GRAY` | CNN ảnh xám: 25 epoch, chọn epoch 25 |
| `candidates/C03_RGB_AUG` | CNN RGB có augmentation: 25 epoch, chọn epoch 24 |
| `candidates/C04_DEEP_AUG` | CNN bốn khối có augmentation: 25 epoch, chọn epoch 24 |
| `selection` | Bảng bốn cấu hình, đủ 100 dòng epoch, cấu hình và hash khóa |
| `feature_analysis` | GAP train/dev, PCA fit train, feature map |
| `test_final` | Manifest, dự đoán, metric, ma trận nhầm lẫn và khóa đã dùng |
| `error_analysis` | 46 lỗi, cặp nhầm, 12 ví dụ, Grad-CAM và mẫu ghi chú |
| `comparison` | KNN/SVM trên cùng ba manifest và dự đoán ghép cặp |
| `report` | Báo cáo và hình dựng từ số liệu đã lưu |

41 hash trong cấu hình khóa khớp bản local khi nhập output; khóa dùng trên test trùng byte với khóa lựa chọn. Tất cả manifest train/dev/test khớp KNN và SVM theo đường dẫn, nhãn và thứ tự dòng. File nguồn tải về được giữ riêng, không ghi đè.

Checkpoint `.pt/.pth` và ma trận `.npy/.npz/.pkl` được giữ local/output Kaggle, bỏ qua bởi Git. Giữ cả bốn `best.pt` vì khóa kiểm tra các run; `last.pt` phục vụ resume. CSV/JSON/PNG đủ đọc báo cáo nhưng không thay trọng số khi cần suy luận, xem feature map hoặc Grad-CAM. Bản báo cáo trong `report/` đã được bổ sung khi nhập về repository; bản xuất ban đầu từ Kaggle vẫn còn trong gói tải về.
