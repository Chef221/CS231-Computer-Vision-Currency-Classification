# SVM RGB224 — phân tích lỗi sau khóa

Phân tích dùng cùng 1.553 dự đoán của HSV12+HOG, RBF C=10, gamma=scale. Có **191 lỗi**, tổng chênh lệch **51.080.000đ**. Điểm quyết định được tính sau khi fit lại cùng SVC trên cache train và xác nhận dự đoán trùng kết quả đã lưu.

## Cặp nhầm thường gặp

Bảng sắp theo số lỗi gộp hai chiều, không theo tổng chênh lệch VND.

| Cặp mệnh giá | Hai chiều | Số lỗi | Tổng chênh |
|---|---:|---:|---:|
| 10.000đ ↔ 200.000đ | 25 + 18 | 43 | 8.170.000đ |
| 20.000đ ↔ 500.000đ | 25 + 9 | 34 | 16.320.000đ |
| 200.000đ ↔ 500.000đ | 7 + 15 | 22 | 6.600.000đ |
| 50.000đ ↔ 200.000đ | 13 + 6 | 19 | 2.850.000đ |
| 10.000đ ↔ 500.000đ | 11 + 5 | 16 | 7.840.000đ |

Cặp 10k↔200k có nhiều lỗi nhất, nhưng **20k↔500k** đóng góp tổng chênh lớn nhất: **16.320.000đ**, chiếm **31,95%** tổng chênh của SVM. MAE đo chênh danh nghĩa, không đo thiệt hại giao dịch.

## Ví dụ và cách diễn giải

- **C09:** thật 10k, dự đoán 200k. Một số chữ số có thể quan sát trên ảnh nhưng SVM nhận histogram/HOG, không có cơ chế đọc chữ số trực tiếp như người xem.
- **C10:** thật 500k, dự đoán 200k, chênh hai điểm đầu **0,003890**. Hai lớp đứng đầu là 200k và 10k, nên điểm sát nhau chưa có nghĩa mô hình gần chọn lớp thật.
- **C11:** ảnh 500k dự đoán đúng dù chênh hai điểm đầu chỉ **0,036052**.

`decision_function` không phải xác suất đã hiệu chỉnh hoặc khoảng cách vật lý tới ảnh train. Không suy nguyên nhân nền, độ sáng, góc chụp hay độ nhòe chỉ từ score. Mẫu `visual_review_notes.csv` của lượt này còn trống; chưa có nhãn điều kiện trên toàn test để công bố tỷ lệ lỗi theo từng điều kiện.

## So sánh ghép cặp với KNN

| Trạng thái | Số ảnh |
|---|---:|
| Cả hai đúng | 1.124 |
| Chỉ SVM đúng | 238 |
| Chỉ KNN đúng | 63 |
| Cả hai sai | 128 |

SVM cải thiện số đúng tổng thể nhưng vẫn làm sai 63 ảnh KNN đúng. Không dùng các ví dụ test để thay đổi lưới hoặc chọn lại SVC.

Nguồn: [toàn bộ điểm](all_test_decisions.csv), [toàn bộ lỗi](all_errors.csv), [cặp gộp hai chiều](error_pairs_bidirectional.csv), [cặp theo tổng chênh](error_pairs_cost.csv), [12 ví dụ](selected_cases.csv), [hồ sơ phân tích](analysis_record.json).
