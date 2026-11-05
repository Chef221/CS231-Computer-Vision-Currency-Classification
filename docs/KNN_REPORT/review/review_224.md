# Rà soát báo cáo KNN RGB224

Phạm vi: phần báo cáo học phần bằng Markdown, không phải bài báo hội nghị đầy đủ hoặc bản Word đã dàn trang. Bản chính: `../BAO_CAO_THUC_NGHIEM_KNN.md`. Bản trước RGB224 đã đưa vào ZIP dự phòng ngoài repository, xem `cleanup_legacy.json`.

## Cấu trúc và lập luận

1. Mục tiêu và điều kiện chung: giải thích split, phạm vi ba họ/năm biến thể, lịch sử test và khác biệt với khảo sát cũ.
2. Tiền xử lý: BGR/RGB, direct resize, nội suy, dtype, L2 và giới hạn của từng thao tác. Hình trước/sau resize.
3. KNN: Euclidean, hai cách bỏ phiếu, lưới 20/biến thể, tiêu chí và thứ tự phá hòa. Không gọi 100 cấu hình là 100 lượt độc lập.
4. Feature: mỗi nhánh có cách tạo vector, số chiều, kết quả và nhận xét. HOG có ảnh minh họa, PCA có số chiều/phương sai thực tế, fusion có đối chứng cố định k=1.
5. Chốt phương án: một bảng năm biến thể, không lặp bằng biểu đồ cột. Không trộn kết quả native/Bray–Curtis cũ.
6. Test: bảng dev/test, bảng từng lớp, ma trận nhầm lẫn.
7. Lỗi: tách tần suất và tổng chênh lệch, giải thích bốn ví dụ bằng ảnh và khoảng cách thật.
8. Kết luận và giới hạn: không xác nhận causal mechanism, độc lập dataset hay kết quả mô hình của thành viên khác khi chưa có evidence.

## Những nhận xét được phép giữ

- HSV12 đạt kết quả dev tốt nhất: bằng chứng trực tiếp từ bảng chọn cấu hình.
- PCA+L2 cải thiện nhánh HOG ngay ở cấu hình k=1, uniform cố định: bằng chứng trực tiếp; không quy toàn bộ cải thiện chỉ cho giảm chiều.
- Ghép không trọng số kém HSV12 ở cả best-by-feature và cấu hình tham chiếu: bằng chứng trực tiếp, không giấu kết quả âm.
- HOG bỏ màu, histogram không giữ vị trí: tính chất biểu diễn. Giải thích vì sao các tính chất đó có thể liên quan kết quả phải giữ mức giả thuyết.
- C04/C09 có train sai gần hơn train đúng lớp: bằng chứng trực tiếp, khoảng cách không phải xác suất.
- Nền/ánh sáng có thể góp phần: giả thuyết từ quan sát, chưa có masking/ablation để chứng minh.
- C10/C11 gần ranh giới: mô tả hình học sau khi biết nhãn; không thêm reject policy hoặc metric mới.

## Kiểm tra đã chạy

`verify_report_224.py` đối chiếu các hàng bảng, các số dẫn lại, PCA, khoảng cách case, source/figure hash, sáu link ảnh và cấu trúc công thức. Kết quả chi tiết nằm trong `verification_224.json` và được cập nhật sau bản sửa cuối. Đã xem trực tiếp cả sáu hình xuất mới, không thấy nhãn bị cắt hoặc sai ảnh/cột. Nguồn kỹ thuật đã mở và đối chiếu qua tài liệu chính thức; ghi riêng tại `citation_verification_224.md`.

## Giới hạn còn lại

Chưa dàn trang Word/PDF hoặc kiểm tra rubric toàn báo cáo nhóm. Không chạy lại grid, không tạo model mới, không kiểm toán lại quan hệ nguồn ảnh train/test. Phần thị giác dựa trên 12 query được xem ở notebook 08, bốn query minh họa trong báo cáo; không suy rộng tỷ lệ nguyên nhân ra toàn test. Không commit/push.

Bổ sung Hình KNN-2: một ảnh train RGB224 và ba mặt phẳng H/S/V có thang đo, minh họa trực tiếp phép chuyển đổi. Caption phân biệt mặt phẳng kênh với histogram kết hợp; không hiển thị mảng HSV dưới dạng RGB. Số thứ tự hình phía sau được cập nhật.

Hình HSV đã bổ sung hàng 500.000đ dưới hàng 10.000đ, cùng thang H/S/V; caption và provenance ghi rõ hai ảnh train. Không đổi feature hoặc kết quả mô hình.

Dọn theo yêu cầu tác giả: bỏ sáu caption riêng và footer HSV; nội dung cần thiết chuyển vào thân bài. Notebook/kết quả/hình cũ được sao lưu, kiểm tra hash rồi xóa; giữ nguyên nhánh RGB224 và DATA. Các ghi chú về caption trước đó trong hồ sơ này là lịch sử biên tập.
