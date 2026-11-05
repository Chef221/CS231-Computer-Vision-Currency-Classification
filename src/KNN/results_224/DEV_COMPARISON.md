# So sánh dev của năm biến thể KNN RGB224

Sau khi người dùng chạy 05A và 05B, đã kiểm tra các đầu ra lưu của cả hai nhánh. Kết quả: PASS_FUSION_AUDIT. Chưa đánh giá test trong thiết kế RGB224 này, chưa sửa tham số theo các kết quả vừa xem.

## Kết quả tốt nhất mỗi biến thể

| Biểu diễn | Chiều | k | weights | Đúng/1.564 | Accuracy | Macro-F1 | MAE VND/ảnh |
|---|---:|---:|---|---:|---:|---:|---:|
| HSV12 | 1.728 | 1 | uniform | 1.245 | 79,60% | 0,798307 | 46.585,68 |
| HSV12 + HOG | 7.812 | 6 | distance | 1.141 | 72,95% | 0,731995 | 62.071,61 |
| HSV12 + HOG-PCA | 3.146 | 6 | distance | 1.112 | 71,10% | 0,713416 | 66.086,96 |
| HOG-PCA + L2 | 1.418 | 5 | distance | 694 | 44,37% | 0,437934 | 106.898,98 |
| HOG không PCA | 6.084 | 1 | uniform | 611 | 39,07% | 0,390748 | 116.234,02 |

Mỗi biến thể có 20 cấu hình k=1..10, uniform/distance, Euclidean, brute. Tổng 100 cấu hình, không phải 100 phép lặp độc lập. Thứ tự này xếp theo Macro-F1; không tự chọn theo Accuracy hoặc MAE. Ví dụ ở HSV12+HOG, k=4 distance có Accuracy và MAE tốt hơn k=6 distance, nhưng Macro-F1 thấp hơn nên không được chọn.

## Đối chiếu khi cố định bộ phân loại

Với k=1, uniform, Euclidean, HSV12 đạt Accuracy 79,60%, Macro-F1 0,798307; HSV12+HOG đạt 72,63%, 0,726608; HSV12+HOG-PCA đạt 69,95%, 0,701336. Như vậy kết hợp không cải thiện màu trong cả so sánh cấu hình tham chiếu và so sánh cấu hình tốt nhất từng nhánh; không thể quy kết quả này chỉ cho việc k thắng khác nhau.

HSV12+HOG đúng ít hơn HSV12 104 ảnh (giảm 6,65 điểm phần trăm), MAE cao hơn 15.485,93 VND/ảnh. HSV12+HOG-PCA đúng ít hơn HSV12 133 ảnh (giảm 8,50 điểm phần trăm), MAE cao hơn 19.501,28 VND/ảnh. Đây là so sánh các cấu hình thắng trên cùng dev.

## Nhận xét cần giữ đúng phạm vi

PCA cải thiện HOG khi dùng riêng trong lần thử, nhưng ghép HSV12 với HOG-PCA lại kém ghép HSV12 với HOG gốc. Điều này không mâu thuẫn: kết quả dùng riêng không bảo đảm một biểu diễn bổ sung tốt khi kết hợp với biểu diễn khác.

Mỗi khối được L2 riêng, sau đó nối với hệ số 1. Khi dùng Euclidean, bình phương khoảng cách ghép bằng tổng bình phương khoảng cách của hai khối. Cùng chuẩn L2 không bảo đảm hai khối chứa lượng thông tin phân lớp bằng nhau hoặc đóng góp khoảng cách bằng nhau trên mỗi cặp ảnh. Thêm HOG thay đổi cách xếp hàng xóm; cách ghép cố định đang thử chưa mang lại lợi ích so với HSV12. Chưa có đối chứng tách riêng tác động của góc quay, nền, che khuất hoặc tối ưu trọng số; không kết luận mọi cách kết hợp màu/đường nét đều kém.

Không mở rộng thêm trọng số fusion, số cell hoặc khoảng cách để tìm kết quả đẹp trong lượt này. Theo quy tắc đã chốt, ứng viên thắng dev là HSV12 RGB224 L2, k=1 uniform Euclidean. Bước tiếp theo là ghi cấu hình khóa và chuẩn bị đánh giá test; chưa có file khóa mới hoặc kết quả test mới được tạo bởi audit này. Test cũ của dự án đã có lịch sử sử dụng, phải công khai khi báo cáo.

## Kiểm tra đã hoàn thành

- Quét toàn bộ feature train/dev của hai nhánh ghép: shape/dtype đúng; từng giá trị ở khối HSV khớp cache HSV12, khối HOG khớp HOG hoặc HOG-PCA đúng ảnh; không có phép chuẩn hóa sau ghép ngoài thiết kế. Norm vector ghép gần căn 2, hữu hạn.
- Manifest và nhãn khớp bước 1 và nguồn HOG; kiểm tra đúng thứ tự cả hai nhánh.
- Cache HSV12 đủ 12.494 train và 1.564 dev, 1.728 chiều float32, không âm và L2 gần 1. Trích lại 18 ảnh dev (ba ảnh đầu của mỗi lớp theo manifest), khớp cache. Chưa trích lại toàn bộ HSV từ PNG trong audit này.
- Tính lại toàn bộ metric từ CSV dự đoán từng ảnh; đối chiếu JSON, confusion matrix, bảng lớp, summary, bảng đối chiếu và bảng cấu hình tham chiếu.
- Đúng 20 tổ hợp và phá hòa cho mỗi nhánh. Cả hai thắng ở k=6 distance. Dự đoán lại 18 ảnh dev mỗi nhánh bằng toàn feature train đã lưu, khớp CSV; không rerun toàn bộ suy luận hoặc các cấu hình khác.
- Hai notebook đã chạy các code cell, không có traceback được lưu. Không fit lại PCA, không đọc test, không thay notebook.

Evidence chi tiết: `hsv12_hog/audit_results.json`, `hsv12_hog_pca/audit_results.json`; bảng số tổng hợp: `all_features_dev_summary.csv`. Kết quả HSV12/HOG/HOG-PCA riêng đã được audit trước đó; bảng này tổng hợp từ các CSV đã lưu.

## Dữ liệu đã sẵn sàng cho bước sau

Cache HSV12 hiện nằm trong thư mục `hsv12/features`, có cả feature, nhãn và manifest; hai nhánh ghép cũng có đủ các file tương ứng. Điều này bổ sung đầu ra feature còn thiếu của notebook HSV12 viết theo mẫu cũ. Các file kết quả phân loại HSV12 trước đó không bị ghi đè bởi bước ghép.
