# Kiểm tra HOG RGB224 không PCA trên dev

Trạng thái: PASS_HOG_RGB224_AUDIT. Chi tiết kiểm tra nằm trong audit_results.json. Không chỉnh notebook hoặc tìm lại cấu hình trong lần audit này.

## Kết quả thực tế

| Nội dung | Giá trị |
|---|---:|
| Train / dev | 12.494 / 1.564 ảnh |
| Feature | HOG 6.084 chiều, float32, L2 toàn vector |
| Cấu hình thắng | k=1, uniform, Euclidean, brute |
| Đúng / sai | 611 / 953 |
| Accuracy | 0,3906649616368286 |
| Macro-F1 | 0,39074817533497597 |
| MAE | 116.234,01534526855 VND/ảnh |

Tổng chênh lệch danh nghĩa 181.790.000 VND không phải thiệt hại giao dịch. k=1 distance và k=2 distance hòa điểm với k=1 uniform; chọn theo thứ tự đã khai báo. Tăng k chưa cải thiện cấu hình tốt nhất trong lưới đã thử.

F1 cao nhất của HOG là 500.000đ (0,431211), thấp nhất 20.000đ (0,364261). Cặp nhầm hai chiều nhiều nhất: 10.000đ ↔ 20.000đ, 108 ảnh; sau đó 10.000đ ↔ 200.000đ, 85 ảnh; 20.000đ ↔ 200.000đ, 83 ảnh.

## Đối chiếu HSV12 cùng đầu vào RGB224 và phạm vi KNN

| Feature | Accuracy dev | Macro-F1 | MAE VND/ảnh |
|---|---:|---:|---:|
| HSV12 | 79,60% | 0,798307 | 46.585,68 |
| HOG không PCA | 39,07% | 0,390748 | 116.234,02 |

Hai cấu hình thắng đều k=1 uniform Euclidean. HOG đúng ít hơn 634 ảnh, Accuracy thấp hơn 40,54 điểm phần trăm. Trong thiết kế này, biểu diễn màu HSV12 phù hợp hơn HOG xám. Không khẳng định mọi cấu hình HOG đều kém hoặc HOG không thể hữu ích khi kết hợp.

HOG giữ gradient và bố trí cục bộ, không giữ màu hay đọc chữ số như OCR. Hình train minh họa cho thấy cả viền tiền, bàn tay và các đường trên áo đều xuất hiện trong HOG. Sự thay đổi góc quay/vị trí và đường nét nền là những giả thuyết ảnh hưởng khoảng cách; chưa có thí nghiệm tách riêng để xác nhận nguyên nhân chi phối 953 lỗi. Số chiều lớn không tự chứng minh là nguyên nhân và PCA chưa chắc cải thiện.

## Phạm vi kiểm tra

- Đúng 20 tổ hợp k/weights/Euclidean, thứ tự xếp hạng và quy tắc phá hòa.
- Tính lại toàn bộ metric từ CSV dự đoán dev; khớp JSON, confusion matrix, bảng lớp và cấu hình thắng.
- Hai manifest khớp bước 1; nhãn NPY khớp từng dòng.
- Quét toàn bộ feature train/dev: shape/dtype đúng, hữu hạn, không âm, bản L2 khớp bản before_l2 sau normalize và độ dài gần 1.
- Trích lại descriptor từ 18 PNG dev (ba ảnh đầu theo manifest của mỗi lớp), tất cả khớp bản before_l2. Dự đoán lại 18 ảnh bằng toàn bộ feature train đã lưu, tất cả khớp CSV. Không chạy lại toàn bộ suy luận hoặc 19 cấu hình còn lại.
- Notebook đã chạy mọi code cell, không có traceback được lưu. Chênh lệch MAE CSV/JSON 1,46e-11 VND chỉ là biểu diễn số thực, đã đối chiếu với sai số cho phép 1e-8 VND.

## Sẵn sàng cho bước PCA

Đã có đủ X_train_hog_before_l2.npy, X_dev_hog_before_l2.npy, X_train.npy, X_dev.npy, nhãn và manifest. File before_l2 vẫn có L2-Hys theo block. Trung vị norm toàn vector trước L2 là 13, phù hợp với 169 block có norm gần 1; không phải lỗi chuẩn hóa.

Bước kế tiếp fit PCA chỉ trên bản HOG train trước L2 toàn vector, giữ 95% phương sai; dùng PCA đó transform dev rồi L2. Giữ cùng 20 cấu hình KNN để đối chứng. Chưa có log thời gian chuẩn hóa cách đo cho lượt HOG này, nên chưa thể kết luận về mức tăng tốc PCA; cần phép đo thời gian cùng điều kiện nếu muốn so chi phí. Chưa fit PCA hoặc đọc test trong audit này.
