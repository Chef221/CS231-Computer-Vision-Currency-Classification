# Chọn lại hai ví dụ HSV

Đã sàng lọc histogram HSV12 cache của 2.299 ảnh train 10.000đ và 2.190 ảnh train 500.000đ (tổng 4.489). Từ histogram L2, chia tổng để tái lập tần suất bin; dùng tâm bin ước lượng S/V và tỷ lệ Hue có trọng số saturation để tạo shortlist. Các thống kê này chỉ phục vụ tìm ví dụ, không phải bước tiền xử lý hay metric của mô hình.

Đã xem 24 ứng viên với RGB và H/S/V: 6 ảnh mỗi lớp ở vòng đầu và 12 ảnh 10.000đ ở vòng bổ sung. Vòng bổ sung ưu tiên nhóm ảnh tên số dưới 1.200 để tìm các ảnh tờ tiền trên nền đơn giản, vì nhiều ví dụ đầu có nền trái cây lấn át màu tờ tiền. Đây là lựa chọn thuận tiện để minh họa, không phải lấy mẫu đại diện.

Chốt `train/10000/10000_000896.png` và `train/500000/500000_001722.png`: ảnh có mệnh giá nhận ra được, Hue của vùng tiền khác nhau dễ thấy, mặt phẳng S của vùng tiền 10.000đ sáng hơn rõ, còn V toàn cảnh ở ảnh 500.000đ tối hơn. Hai cảnh có nền/ánh sáng khác nhau, nên không quy toàn bộ khác biệt cho mệnh giá. Hình chỉ giải thích vai trò kênh và cách các ảnh có thể khác nhau trong HSV.

Giữ nguyên phép đọc RGB224, chuyển đổi HSV, bảng màu và thang giá trị chung; không chỉnh S/V, tăng sáng, làm nét hoặc đổi thang riêng từng ảnh. Giá trị S/V thực đo toàn ảnh nằm trong `selected_examples.json`; chúng bao gồm nền, không được diễn giải là thống kê riêng tờ tiền.

Các CSV `all_train_class_candidates.csv` và shortlist lưu số ước lượng từ bin; các contact sheet ghi số xấp xỉ. Chỉ `selected_examples.json` ghi S/V thực đo sau đọc lại hai ảnh.

Không đổi DATA, vector đã dùng, mô hình hay kết quả thực nghiệm.
