# Kết quả SVM RGB224

Thư mục này giữ kết quả thực nghiệm SVM RGB224 đã chạy trên Kaggle. Các CSV, JSON và hình được lưu để kiểm tra số liệu; ma trận đặc trưng NPY và PCA pickle được giữ ngoài Git do dung lượng lớn. Phương án đã khóa là HSV12+HOG với SVC RBF, `C=10`, `gamma='scale'`; kết quả test có trong `test_final/metrics_test.json`.

| Thư mục | Nội dung được tạo |
|---|---|
| `inputs` | Manifest, kiểm tra ảnh, cấu hình thực nghiệm |
| `hsv12`, `hog`, `hog_pca` | Feature, grid dev, dự đoán và metric |
| `hsv12_hog`, `hsv12_hog_pca` | Hai biến thể kết hợp và kết quả dev |
| `selection` | Bảng 35 cấu hình, đối chứng cố định, cấu hình khóa |
| `test_final` | Dự đoán, metric, ma trận nhầm lẫn và thông tin lần chạy test |
| `error_analysis` | Bảng lỗi, điểm quyết định, ví dụ và ghi chú thị giác |

Muốn chạy lại notebook phụ thuộc cache, cần dùng đúng đầu ra đầy đủ từ các bước trước, bao gồm NPY và PKL. Các kết quả CSV/JSON/PNG trong Git đủ để đọc và kiểm tra bảng báo cáo, nhưng không thay thế cache cần cho bước fit hoặc phân tích lỗi.
