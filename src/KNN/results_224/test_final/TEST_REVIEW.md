# Kiểm tra kết quả test — KNN RGB224

Trạng thái: **PASS**, kiểm tra ngày 05/11/2026. Phương án đã khóa là HSV12 (1.728 chiều, L2), KNN k=1, uniform, Euclidean, brute; đầu vào RGB224 resize trực tiếp INTER_LINEAR. Fit trên 12.494 ảnh train, không gộp dev và không tìm tham số trên test.

| Độ đo | Dev | Test |
|---|---:|---:|
| Accuracy | 79,6036% | 76,4327% |
| Macro-F1 | 79,8307% | 76,8710% |
| MAE (VND/ảnh) | 46.585,68 | 53.947,20 |

Test có 1.553 ảnh: 1.187 đúng, 366 sai. Accuracy giảm 3,17 điểm phần trăm và Macro-F1 giảm 2,96 điểm phần trăm so với dev; MAE tăng 7.361,52đ/ảnh. Sự khác biệt này mô tả kết quả trên hai split, chưa đủ để kết luận nguyên nhân hoặc mức overfitting.

F1 cao nhất ở lớp 100.000đ (81,76%), tiếp theo là 20.000đ (81,53%). Hai lớp thấp nhất là 10.000đ (72,66%) và 200.000đ (72,93%). Cặp 10.000đ ↔ 200.000đ có 62 lỗi: 34 ảnh 10.000đ thành 200.000đ và 28 ảnh theo chiều ngược lại.

Lỗi xuất hiện nhiều và lỗi đóng góp chênh lệch lớn không nhất thiết là cùng một cặp. Chiều 20.000đ → 500.000đ có 19 lỗi, đóng góp 9.120.000đ vào tổng chênh lệch; chiều ngược lại có 18 lỗi, đóng góp 8.640.000đ. Tổng chênh lệch của toàn test là 83.780.000đ, chia 1.553 ảnh ra MAE. Đây là chênh lệch mệnh giá danh nghĩa, không phải thiệt hại tài chính thực tế.

## Bằng chứng kiểm tra

- Tất cả cell code đã thực thi, không có output lỗi được lưu.
- Bản sao file khóa khớp nguyên bản. Mã hash của file khóa, 12 file đầu vào đã khóa và hai module tiền xử lý/độ đo đều khớp.
- Đối chiếu đủ đường dẫn và nhãn 1.553 ảnh; tính lại toàn bộ độ đo, ma trận nhầm lẫn, bảng từng lớp và bảng cặp lỗi từ dự đoán đã lưu: khớp.
- Toàn bộ feature test có đúng shape/dtype, hữu hạn, không âm và chuẩn L2. Đọc lại 18 ảnh test (ba ảnh mỗi lớp) để trích HSV12: khớp chính xác cache.
- Chạy lại dự đoán cho toàn bộ 1.553 feature test với đúng train và cấu hình đã khóa: khớp toàn bộ dự đoán đã lưu. Không duyệt thêm tham số.

Chi tiết máy đọc được nằm trong `audit_results.json`. Không trích lại feature của toàn bộ test, không kiểm toán lại chất lượng nhãn hay tính độc lập của dataset. Test từng được xem trong các thử nghiệm trước, không phải holdout hoàn toàn chưa xem.

## Tiếp theo

Phân tích lỗi bằng ảnh thật và ảnh train láng giềng được KNN chọn. Dùng `predictions_test.csv` và cấu hình hiện tại; không thay tham số theo lỗi test. Việc giải thích lỗi do nền, ánh sáng, màu hay bố cục cần kiểm tra ảnh và histogram cụ thể, không suy ra chỉ từ ma trận nhầm lẫn.
