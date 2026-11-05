# KNN — thực nghiệm RGB224

Lộ trình: [chín bước chính thức](../../../docs/KNN_224_ROADMAP.md).

Mở `01_PREPARE_INPUT.ipynb`, sửa `PROJECT_ROOT` nếu checkout không ở `C:\CS231`, rồi chạy từ trên xuống. Notebook đọc train/dev, tạo manifest, kiểm tra ảnh đọc được và minh họa resize. Không trích feature hoặc chạy KNN ở bước này.

Đầu ra bước 1 nằm ở `src/KNN/results_224/inputs`: manifest train/dev, cấu hình, bảng đếm, bảng kiểm tra và ảnh minh họa. Chạy lại sẽ cập nhật các đầu ra của bước này, không ghi vào kết quả KNN cũ. CSV dùng đường dẫn tương đối theo DATA; giữ nguyên thứ tự dòng khi tạo feature.

Hàm dùng chung nằm ở `src.preprocessing.read_rgb224`. SVM/CNN/ResNet có thể import cùng hàm từ thư mục gốc repo để có đúng ảnh RGB uint8 224×224. Các phép trích feature/chuẩn hóa số học diễn ra sau đó. Không áp dụng thêm resize hoặc center crop nếu muốn giữ đầu vào hình học giống nhau.

Hàm đọc sẽ báo lỗi tại OpenCV nếu ảnh không đọc được, không âm thầm bỏ ảnh. Bước kiểm tra chỉ xác minh khả năng đọc, shape và dtype; không xác minh chất lượng nhãn hay thông tin nhận dạng còn đủ sau resize.

Code kiểm tra nội bộ nằm ngoài notebook để phần học tập giữ ngắn gọn. Kiến trúc và huấn luyện mô hình của các thành viên khác chưa được kiểm tra tại đây.

Sau khi chạy và kiểm tra notebook `06_SELECT_FINAL_KNN.ipynb`, mở `07_EVALUATE_TEST.ipynb` rồi Run All. Notebook 07 kiểm tra các file đã khóa, dùng cache train HSV12 và trích feature test từng ảnh với cùng pipeline. Chỉ đánh giá cấu hình đã chọn, không tìm tham số trên test hoặc gộp train+dev. Kết quả nằm trong `src/KNN/results_224/test_final`, gồm bảng độ đo, ma trận nhầm lẫn, dự đoán từng ảnh và bản sao cấu hình đã dùng. Chạy lại sẽ ghi đè kết quả trong thư mục này. Test đã được xem trong các thử nghiệm cũ, nên không gọi đây là holdout hoàn toàn chưa xem.

Sau khi kiểm tra bước test, chạy `08_ERROR_ANALYSIS.ipynb`. Notebook đối chiếu hàng xóm cho toàn bộ test, tổng hợp lỗi và chọn tối đa 12 ví dụ để xem query cạnh train được chọn và train đối chiếu. Hình, bảng và mẫu ghi chú nằm trong `src/KNN/results_224/error_analysis`. Các nguyên nhân thị giác chỉ được viết sau khi xem hình; notebook không tự suy ra nguyên nhân từ histogram. Không đổi mô hình theo lỗi test. Kế hoạch hình: [KNN_224_ERROR_ANALYSIS_FIGURE_PLAN.md](../../../docs/KNN_224_ERROR_ANALYSIS_FIGURE_PLAN.md).
