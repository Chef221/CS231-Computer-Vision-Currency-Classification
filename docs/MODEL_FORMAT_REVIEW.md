# KNN, SVM và CNN dùng chung những gì?

- Cùng sáu mệnh giá và split train/dev/test đã có.
- Cùng đọc ảnh bằng OpenCV RGB, resize224×224 INTER_LINEAR.
- Cùng chọn theo Macro-F1 dev; nếu bằng nhau thì MAE thấp hơn.
- Cùng chấm nhãn mệnh giá bằng `src/evaluation/value_metrics.py`.
- Cùng bố cục: chuẩn bị, khảo sát baseline/biến thể, chọn cấu hình, test, phân tích lỗi.

KNN và SVM dùng cùng định nghĩa HSV12/HOG, PCA chỉ fit trên train, ghép các nhánh đã L2 riêng. CNN nhận pixel, có augmentation ở train của C03/C04, tự học đặc trưng. Vì cách biểu diễn và chi phí huấn luyện khác nhau, bảng kết quả so sánh các phương pháp hoàn chỉnh, chưa tách riêng tác dụng của từng thay đổi.

Code KNN và M0 không thay đổi trong lần sắp xếp này. SVM/CNN giữ phương pháp baseline, rút gọn phần kiểm tra hash, stage và chạy tiếp để dễ học hơn. Bảng công bố là số liệu đã chạy của nhóm; xem [RESULTS.md](RESULTS.md).
