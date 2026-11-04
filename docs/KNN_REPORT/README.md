# Cách dùng phần báo cáo KNN

Đọc `BAO_CAO_THUC_NGHIEM_KNN.md`. Đây là phần nội dung để ghép vào báo cáo học phần, gồm sáu bảng ngắn và ba hình thật. Không phải nhật ký kỹ thuật hay bản sao nội dung trò chuyện.

- Giữ `KNN_FIGURES/` cạnh file Markdown để hiện hình. PNG xuất 300 dpi; confusion matrix có thêm SVG.
- Nhãn Bảng KNN-1…6 và Hình KNN-1…3 là nhãn tạm theo phần; khi ghép báo cáo tổng, đổi sang số chương/mục tương ứng. Giữ nguyên caption và số liệu.
- Bảng số dùng dấu phẩy thập phân. Nhãn khoảng cách trong ảnh dùng dấu chấm thập phân; caption và phần mô tả giải thích cùng một giá trị.
- Bốn tài liệu kỹ thuật tham khảo được dẫn cho API/công thức, không phải nguồn của kết quả thực nghiệm. Ghép lại số trích dẫn theo bibliography chung khi tổng hợp.
- Bản này trình bày RGB, HSV8, HSV12, Color Moments và HOG cell16 theo phạm vi tác giả chọn. Lịch sử CSV/notebook khác vẫn được giữ ở `src/KNN`, không xóa hoặc sửa số liệu để khớp báo cáo.
- Phần nhận xét nêu đúng giới hạn: các cấu hình KNN được chọn riêng trên dev; một split; Color Moments chưa chuẩn hóa thang đo; HOG chưa tối ưu; ba ví dụ lỗi không đại diện cho toàn test.

`build_figures.py` tạo lại hình từ DATA và CSV trong repo; không fit hoặc suy luận lại. `review/` là hồ sơ đối chiếu và kiểm tra nội bộ, không chép vào nội dung báo cáo.

Bài báo cáo nhóm phân loại trái cây được tham khảo ở các trang in 19–21, 40 và 44: cách tách cài đặt, bảng từng lớp, confusion matrix và nhận xét. Không kế thừa kết quả, điều kiện dữ liệu, PCA hay 5-fold CV của nhóm đó vì khác với quy trình hiện tại.
