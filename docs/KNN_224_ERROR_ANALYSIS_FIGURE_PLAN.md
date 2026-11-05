# Hình phân tích lỗi KNN RGB224

Nguồn tái tạo: `src/KNN/224/08_ERROR_ANALYSIS.ipynb`. Đầu vào là cấu hình khóa, feature train/test, manifest và dự đoán test đã lưu. Chỉ xuất hình khi người dùng chạy notebook; không dùng ảnh sinh để làm bằng chứng.

## cases_page_01.png và các trang tiếp theo

- Loại: evidence/result; vai trò: failure analysis; trạng thái: ví dụ định tính gắn với số liệu chính xác.
- Thông điệp: hàng xóm trong không gian HSV12 quyết định nhãn của KNN k=1; ảnh train đúng mệnh giá có thể ở xa query hơn ảnh sai mệnh giá.
- Bố cục: tối đa bốn hàng mỗi trang, ba cột: query test, train được chọn, train đối chiếu. Ảnh hiển thị ở RGB224, nhãn và khoảng cách nằm phía trên.
- Backend: matplotlib, PNG 300 DPI. Không vẽ lại ma trận nhầm lẫn đã có ở notebook 07.
- Dữ liệu truy nguồn: `selected_cases.csv` và `all_test_neighbors.csv`; mã case liên kết với đường dẫn query và các ảnh train.
- Quy tắc chọn: ba cặp lỗi phổ biến theo hai chiều; ba chiều lỗi đóng góp tổng chênh lệch lớn nhất; một lỗi gần ranh giới; hai ảnh đúng đối chiếu. Loại trùng query, hòa theo đường dẫn. Tối đa 12 ví dụ.
- Giới hạn: ví dụ có chủ đích, không đại diện tần suất nguyên nhân trên toàn test. Khoảng cách không phải xác suất. Không kết luận nguyên nhân do nền, ánh sáng hoặc độ rõ trước khi xem ảnh và kiểm chứng.
- Dự phòng: bảng CSV lưu nhãn/đường dẫn/khoảng cách nếu chưa xuất được hình. Báo cáo cuối dùng khoảng hai hình tiêu biểu sau khi review thị giác, ghi rõ mã case.

## hsv_marginals_Cxx.png / .pdf

- Loại: evidence/result; vai trò: giải thích bổ sung cho hai ảnh sai và một ảnh đúng.
- Thông điệp: các ảnh có phân bố màu khác nhau hoặc tương tự ở từng kênh; biểu đồ hỗ trợ quan sát màu, không thay thế khoảng cách feature.
- Bố cục: ba panel H, S, V; mỗi panel có ba đường query, train được chọn, train đối chiếu; màu cố định xanh dương, đỏ cam, xanh lá.
- Backend: matplotlib, PNG 300 DPI và PDF vector; dữ liệu từ histogram chung 12×12×12 đã lưu, cộng theo hai trục rồi chia tổng.
- Giới hạn: phân bố biên không giữ tương quan ba kênh và không phải vector trực tiếp dùng tính Euclidean. Không suy ra hai vector joint giống nhau chỉ vì ba đường phân bố biên giống nhau.
- Dự phòng: ảnh ghép và khoảng cách vẫn đủ mô tả quyết định; không bắt buộc đưa biểu đồ histogram vào báo cáo nếu không giúp giải thích.

## Rà soát sau Run All

Kiểm tra nhãn, đường dẫn, khoảng cách, chất lượng hiển thị và mã case. Tách quan sát khỏi giả thuyết trong `visual_review_notes.csv`; ghi chú chưa có bằng chứng thay vì tự gán nguyên nhân. Không dùng lỗi test để thay cấu hình khóa trong thí nghiệm này.
