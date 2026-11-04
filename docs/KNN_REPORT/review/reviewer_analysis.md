# Tự rà soát phần báo cáo KNN

## Phạm vi

Bản viết là một phần thực nghiệm trong báo cáo học phần, không phải bài báo khoa học hoàn chỉnh. Không tạo thông tin venue, novelty hoặc kết quả chưa chạy. Chỉ trình bày năm cấu hình tác giả chọn; không khẳng định đó là toàn bộ lịch sử thử nghiệm. Không xóa HSV16/HOG cell8 khỏi hồ sơ.

## Đối chiếu tiêu chí chấm

- Phương pháp dễ hiểu, khác biệt: Bảng KNN-1, cấu hình feature, cách chuẩn hóa, số chiều và giới hạn của từng cách; Bảng KNN-2 ghi lưới thực tế.
- Dataset/độ đo: mục 1 nêu train/dev/test, vai trò và quy tắc chọn; không lặp toàn bộ phần dataset/M2.2 đã có.
- Kết quả phương pháp: Bảng KNN-3 là dev, Bảng KNN-4/5 là cấu hình khóa trên test; không gọi grid search là cross-validation.
- Nhận xét/kết luận: giải thích các chênh lệch bằng số liệu, giữ giới hạn thống kê và đánh giá.
- Minh họa và hỏi đáp: ba hình thể hiện confusion matrix, lỗi ảnh rõ, lỗi hai lớp có khoảng cách sát nhau.
- Demo và so sánh đủ bốn mô hình thuộc báo cáo chung; phần KNN không tự đáp ứng toàn bộ rubric hay thay thế phần của thành viên khác.

## Ba lượt rà soát

1. Phương pháp: train-only fit; dev chọn; test không tune; k=1 vẫn giữ khi lấy năm hàng xóm; RGB histogram biên khác HSV kết hợp; Minkowski p=2 trùng Euclidean.
2. Số liệu: kiểm tra các CSV gốc và toàn bộ dự đoán test qua scripts hiện có; kiểm tra lại 15 khoảng cách từ PNG thật. Không tái chạy toàn bộ 840 cấu hình hoặc nearest-neighbor search đầy đủ.
3. Văn phong: giọng báo cáo học phần, xưng “nhóm”; bỏ lịch sử trao đổi, câu quảng cáo và khẳng định nguyên nhân chưa chứng minh. Không đưa code dài, learning curve giả, histogram dày hoặc hình trang trí.

## Các vấn đề đã xử lý trong bản cuối

- Không nói HSV tốt hơn chỉ do đổi không gian màu: histogram và chuẩn hóa cũng khác.
- Không nói mọi biến thể HOG/Color Moments đều kém.
- Không gọi HSV12 tối ưu tuyệt đối hoặc vượt SVM/CNN/ResNet.
- Không quy mọi lỗi ảnh rõ cho nền hoặc mọi lỗi cho nhòe.
- Không coi MAE/tổng chênh lệch là thiệt hại tài chính thật.
- Không gọi khoảng cách là độ tin cậy; trường hợp gần bằng nhau không phải hòa tuyệt đối.
- Không tuyên bố các ảnh test độc lập theo tờ tiền vật lý; chưa có CI/kiểm định hoặc nhiều split.

## Giới hạn còn giữ

Dữ liệu và code đã có cho phép xác minh nhất quán bảng số; chưa có đủ bằng chứng để định lượng tác động nền, nguồn, phiên chụp hoặc điều kiện ảnh. Không cần thêm thí nghiệm mới để mô tả đúng phạm vi hiện tại. Biên tập số chương, bibliography và dàn trang Word/LaTeX sẽ theo tài liệu tổng của nhóm.
