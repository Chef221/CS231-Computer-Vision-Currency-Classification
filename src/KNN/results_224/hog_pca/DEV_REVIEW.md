# Kiểm tra HOG + PCA95 + L2 trên dev

Trạng thái: PASS_HOG_PCA95_AUDIT. Evidence và giới hạn nằm trong audit_results.json. Không fit lại PCA hoặc thay cấu hình trong audit này.

## Kết quả

PCA từ 6.084 xuống 1.418 chiều, giảm 76,69% số chiều. Giữ 95,005643% phương sai train; nếu bỏ thành phần cuối chỉ còn 94,997561%, nên 1.418 là số thành phần nhỏ nhất đạt ngưỡng theo mô hình đã lưu. 95% phương sai không phải 95% Accuracy.

| Phương án | k | weights | Đúng/1.564 | Accuracy | Macro-F1 | MAE VND/ảnh |
|---|---:|---|---:|---:|---:|---:|
| HOG không PCA | 1 | uniform | 611 | 39,07% | 0,390748 | 116.234,02 |
| HOG + PCA95 + L2, tham chiếu cố định | 1 | uniform | 680 | 43,48% | 0,431541 | 110.249,36 |
| HOG + PCA95 + L2, tốt nhất trên dev | 5 | distance | 694 | 44,37% | 0,437934 | 106.898,98 |
| HSV12, tốt nhất trên dev | 1 | uniform | 1.245 | 79,60% | 0,798307 | 46.585,68 |

Mọi hàng đều dùng RGB224 bilinear, KNN Euclidean, cùng train/dev. Nhánh PCA tốt nhất đúng thêm 83 ảnh so với HOG không PCA, Accuracy tăng 5,31 điểm phần trăm, MAE giảm 9.335,04 VND/ảnh. Khi đối chiếu hai cấu hình thắng từng ảnh, 253 ảnh sai trước đây thành đúng, nhưng 170 ảnh đúng thành sai; 700 ảnh sai ở cả hai. Còn 870 ảnh sai; tổng chênh lệch danh nghĩa 167.190.000 VND không phải tổn thất giao dịch.

Giữ k=1 uniform cho cả hai nhánh HOG, PCA rồi L2 vẫn đúng thêm 69 ảnh, tăng 4,41 điểm phần trăm. Như vậy cải thiện không chỉ đến từ việc cấu hình thắng đổi sang k=5 distance. Tuy nhiên đây là tác dụng của cả nhánh PCA + L2 (bao gồm trừ trung bình, chiếu và chuẩn hóa), chưa tách riêng ảnh hưởng số chiều hoặc chứng minh PCA luôn loại được nhiễu.

Nhánh PCA vẫn thấp hơn rõ rệt HSV12. Chưa kết luận HOG không hữu ích trong kết hợp; hai biến thể fusion vẫn được giữ theo kế hoạch. F1 tốt nhất của nhánh PCA là 500.000đ (0,508475), thấp nhất 100.000đ (0,379592). Không phải mọi lớp đều cải thiện so với HOG gốc.

## Các kiểm tra đã đạt

- Đối tượng PCA có 12.494 mẫu fit, 6.084 feature nguồn, solver full, ngưỡng 0,95 và không whitening. Mean khớp HOG train; source notebook chỉ gọi fit trên train.
- Manifest/nhãn train/dev khớp HOG nguồn; ma trận lần lượt 12.494×1.418 và 1.564×1.418, float32, hữu hạn và L2 gần 1.
- Transform lại 18 dòng train và 18 dòng dev từ HOG trước L2: khớp feature đã lưu trong sai số số thực cho phép. Không transform lại toàn bộ ma trận.
- Tính lại toàn bộ metric từ dự đoán dev; JSON, CSV cấu hình thắng, confusion matrix, bảng lớp và hai bảng đối chứng khớp nhau.
- Đúng 20 tổ hợp KNN, xếp hạng và phá hòa đúng. Dự đoán lại 18 dòng dev bằng k=5 distance trên toàn feature train: khớp CSV.
- Mọi code cell đã chạy, không có traceback được lưu. Không rerun các cấu hình còn lại và không đọc test.

## Thời gian và giới hạn

Log notebook: fit PCA 116,87 giây; transform train + L2 1,87 giây; transform dev + L2 0,28 giây; toàn lưới KNN kèm metric 13,64 giây; predict riêng cấu hình thắng trên dev 0,39 giây. Đây là một lượt đo trên máy người dùng, không phải benchmark nhiều lần. HOG không PCA chưa có thời gian đo tương ứng nên chưa công bố hệ số tăng tốc. Giảm số chiều chưa đồng nghĩa tăng tốc đúng cùng tỷ lệ; PCA còn có chi phí fit/transform.

Đã lưu đủ PCA, feature sau PCA/L2, nhãn và manifest để dùng ở nhánh kết hợp. Bước kế tiếp chạy cả HSV12 + HOG gốc và HSV12 + HOG-PCA; PCA ở nhánh ghép được giữ trên HOG, không fit lại trên vector HSV+HOG. HSV12 trước đó chưa xuất ma trận feature nên cần xuất/trích lại đúng feature HSV12 trước khi ghép; không cần tìm lại tham số HSV12.
