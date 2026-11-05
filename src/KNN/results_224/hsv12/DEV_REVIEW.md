# Kiểm tra kết quả HSV12 RGB224 trên dev

Đã kiểm tra notebook đã chạy đủ các code cell, không lưu traceback; lưới đúng 20 cấu hình và thứ tự chọn đúng. CSV ma trận nhầm lẫn, bảng từng lớp, bảng lưới và cấu hình thắng nhất quán. Thứ tự đọc thư mục train/dev khớp manifest bước 1. Không chạy lại trích feature hoặc dự đoán trong đợt kiểm tra này.

## Kết quả

- Cấu hình thắng: k=1, uniform, Euclidean, brute.
- Train 12.494 ảnh; dev 1.564 ảnh; RGB224 bilinear; HSV12 L2, 1.728 chiều float32.
- Đúng 1.245; sai 319; Accuracy 0,7960358056265985; Macro-F1 0,7983069780027404.
- MAE 46.585,67774936061 VND/ảnh; tổng chênh lệch 72.860.000 VND, không phải thiệt hại giao dịch.
- k=1 distance và k=2 distance hòa cả ba độ đo tổng hợp với k=1 uniform. Chọn uniform theo thứ tự đã khai báo; bằng các độ đo không chứng minh dự đoán từng ảnh giống nhau.
- k=1 uniform/distance bình thường có cùng quyết định vì chỉ có một hàng xóm. Với k=2 distance, khi hai nhãn khác nhau và khoảng cách khác nhau thì hàng xóm gần hơn có phiếu lớn hơn; khi cùng nhãn cũng không đổi quyết định. Không suy từ ba hàng hòa điểm rằng mọi bộ dữ liệu đều có kết quả giống nhau.

## Nhận xét

- F1 cao nhất: 20.000đ = 0,857143. F1 thấp nhất: 200.000đ = 0,760976; kế tiếp 500.000đ = 0,764259.
- Cặp nhầm hai chiều nhiều nhất: 10.000đ ↔ 200.000đ, 61 ảnh (27 và 34 theo hai chiều).
- Cặp tiếp theo: 200.000đ ↔ 500.000đ, 42 ảnh (24 và 18).
- Độ đo tốt nhất của distance cao hơn uniform với từng k=2..10 trong lưới này; chưa làm distance thắng cấu hình k=1 uniform. Tăng k chưa cải thiện điểm tốt nhất trên dev này, không kết luận k lớn luôn kém trên mọi bài toán.

## Đối chiếu lịch sử

| Phương án | Accuracy dev | Macro-F1 dev | MAE VND/ảnh |
|---|---:|---:|---:|
| HSV12 ảnh gốc, k=1 uniform Euclidean | 79,6675% | 0,798920 | 45.492,33 |
| HSV12 RGB224, k=1 uniform Euclidean | 79,6036% | 0,798307 | 46.585,68 |
| HSV12 ảnh gốc, k=1 uniform Bray–Curtis | 89,3862% | 0,898057 | 23.554,99 |

Giữ Euclidean, kết quả mới giảm một ảnh đúng (0,06394 điểm phần trăm) và MAE tăng 1.093,35 VND/ảnh. Chênh lệch 89,39% → 79,60% đồng thời đổi cách resize và khoảng cách, không thể quy hoàn toàn cho resize. Bảng hiện có cũng chưa cung cấp kết quả Bray–Curtis trên RGB224 để kết luận tương tác giữa hai yếu tố. Việc giữ Euclidean trong thiết kế mới là giới hạn phạm vi để dễ giải thích, không phải tuyên bố nó tối ưu cho HSV.

## Giới hạn và đầu ra chưa có

Bản notebook viết theo mẫu cũ chỉ lưu CSV lưới, cấu hình thắng, confusion matrix và bảng từng lớp. Chưa xuất ma trận feature, manifest kèm feature, dự đoán từng ảnh hoặc thời gian chạy. Do đó kiểm tra hiện tại xác nhận tính nhất quán của số liệu tổng hợp, không xác minh lại từng dự đoán từ PNG. Các đầu ra này còn cần bổ sung khi bàn giao SVM/kết hợp; không bắt buộc chạy lại toàn bộ lưới nếu kernel còn các biến cần lưu.

Evidence: `audit_saved_results.json` chứa các kiểm tra và hash CSV tại thời điểm audit. Notebook chưa đánh giá test mới và chưa có kết quả HOG/fusion.
