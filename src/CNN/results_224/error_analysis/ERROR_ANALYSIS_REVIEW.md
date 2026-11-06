# CNN RGB224 — phân tích lỗi sau khóa

Phân tích từ cùng 1.553 dự đoán test của C04_DEEP_AUG epoch 24. Có **46 lỗi**, tổng chênh lệch **14.140.000đ**. Không chỉnh cấu hình theo lỗi test.

## Cặp nhầm và độ lớn lỗi

| Thật → dự đoán | Số lỗi | Chênh mỗi lỗi | Tổng chênh |
|---|---:|---:|---:|
| 20.000đ → 500.000đ | 14 | 480.000đ | 6.720.000đ |
| 500.000đ → 100.000đ | 4 | 400.000đ | 1.600.000đ |
| 500.000đ → 200.000đ | 4 | 300.000đ | 1.200.000đ |
| 500.000đ → 20.000đ | 2 | 480.000đ | 960.000đ |
| 100.000đ → 500.000đ | 2 | 400.000đ | 800.000đ |

Cặp 20.000đ ↔ 500.000đ gộp hai chiều có **16 lỗi**, tổng chênh **7.680.000đ**, bằng **54,31%** toàn bộ tổng chênh lệch. Cặp này còn 37 lỗi ở KNN và 34 ở SVM; CNN giảm số lỗi nhưng chưa giải quyết hết nhầm mệnh giá cách xa nhau. Tổng số lỗi và tổng chênh trong bảng cặp khớp metric test; MAE là chênh danh nghĩa, không phải thiệt hại giao dịch.

## Điểm dự đoán và Grad-CAM

- **C08:** thật 50.000đ, dự đoán 10.000đ với softmax **0,998898**. Ảnh sai vẫn có điểm rất cao. Hình thấy tay cầm tờ tiền trước nền nhiều trái cây; không xác nhận nền là nguyên nhân gây sai.
- **C09:** thật 500.000đ, dự đoán 20.000đ; softmax hai lớp đầu là **0,288079** và **0,283999** (lớp thứ hai 100.000đ), gap **0,004080**. Điểm lớp thật chỉ **0,042403**; điểm đầu sát nhau chưa có nghĩa mô hình gần chọn lớp đúng.
- **C10:** ảnh 10.000đ dự đoán đúng, softmax **0,328291**, gap **0,028763**. Điểm thấp không đồng nghĩa nhãn sai. Chưa có ngưỡng từ chối được chọn trên dev hoặc thử trên test trong lượt này.
- **C11:** ảnh 10.000đ dự đoán đúng; softmax lưu float32 bằng 1,0 dù còn các điểm lớp khác rất nhỏ. Không diễn giải đây là xác suất đúng tuyệt đối.

Grad-CAM dùng gradient logit ở ReLU thứ hai của khối cuối, trước pooling. Hai map cho lớp dự đoán/lớp thật chuẩn hóa cực đại riêng; không so độ lớn tuyệt đối của màu nóng. Cả 24 map của 12 ảnh đều được ghi nhận không bằng 0. Map chỉ là gợi ý vùng đáp ứng, chưa chứng minh CNN đọc chữ số, hiểu mệnh giá hoặc có quan hệ nhân quả với nền/ánh sáng.

## So sánh ghép cặp

| Baseline | Cả hai đúng | Chỉ CNN đúng | Chỉ baseline đúng | Cả hai sai |
|---|---:|---:|---:|---:|
| KNN | 1.168 | 339 | 19 | 27 |
| SVM | 1.339 | 168 | 23 | 23 |

Có **18 ảnh cả ba mô hình cùng sai**. So sánh trên cùng các đường dẫn và nhãn; CNN không chiếm ưu thế ở mọi ảnh riêng lẻ dù tốt hơn về metric tổng.

Nguồn: [toàn bộ lỗi](all_errors.csv), [cặp theo tổng chênh](error_pairs_cost.csv), [cặp gộp hai chiều](error_pairs_bidirectional.csv), [12 ví dụ](selected_cases.csv), [kiểm tra Grad-CAM](gradcam_checks.csv). CSV ghi chú gốc [visual_review_notes.csv](visual_review_notes.csv) còn trống; các ghi chú đã xem ảnh khi viết báo cáo nằm riêng tại [hồ sơ trực quan](../../../../docs/CNN_REPORT/review/VISUAL_REVIEW.md).
