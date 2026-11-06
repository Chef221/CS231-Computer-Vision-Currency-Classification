# SVM RGB224 — so sánh dev và khóa phương án

Năm biến thể dùng cùng 12.494 ảnh train và 1.564 ảnh dev, với cùng reader RGB224 của KNN. Mỗi biến thể thử bảy cấu hình SVC: linear C=0,1/1/10 và RBF C=1/10 kết hợp gamma=scale/10. Tổng cộng 35 cấu hình, không cross-validation.

| Biến thể | Số chiều | Đúng / 1.564 | Accuracy | Macro-F1 | MAE (đ/ảnh) |
|---|---:|---:|---:|---:|---:|
| HSV12 | 1.728 | 1.367 | 87,40% | 0,879544 | 29.705,88 |
| HOG | 6.084 | 811 | 51,85% | 0,506556 | 100.409,21 |
| HOG-PCA | 1.418 | 824 | 52,69% | 0,514872 | 99.757,03 |
| **HSV12+HOG** | **7.812** | **1.374** | **87,85%** | **0,881213** | **27.653,45** |
| HSV12+HOG-PCA | 3.146 | 1.320 | 84,40% | 0,846732 | 36.809,46 |

Tất cả hàng tốt nhất chọn RBF, C=10, gamma=scale. Mức gamma số thực của `scale` phụ thuộc dữ liệu từng biểu diễn; không mặc định cùng gamma thực giữa các hàng.

## Nhận xét

HSV12 là biến thể đơn lẻ mạnh nhất. Ghép HOG gốc tăng thêm **7 ảnh đúng** và **0,001669 Macro-F1**, giảm MAE **2.052,43đ/ảnh** so với HSV12; độ chênh nhỏ chưa chứng minh phép ghép luôn tốt hơn ở nguồn khác.

HOG-PCA giữ 1.418 thành phần, 95,0056% phương sai HOG train; PCA được fit trước L2 toàn vector và chỉ trên train. F1 HOG-PCA cao hơn HOG **0,008316**, nhưng ghép HSV12+HOG-PCA lại thấp hơn ghép HOG gốc **0,034481**. Giữ phần lớn phương sai không bảo đảm giữ mọi tín hiệu phân loại.

Các nhánh có HSV đạt kết quả cao hơn HOG đơn lẻ trong lưới hiện tại. Chưa có thí nghiệm tách riêng màu tiền, nền, góc chụp và ánh sáng để gán nguyên nhân. Không coi nhiều chiều tự nó là lý do HOG kém.

## Cấu hình khóa và bằng chứng

Phương án **HSV12+HOG → SVC RBF, C=10, gamma=scale** được chọn bằng F1 dev chưa làm tròn; hòa thì MAE thấp hơn, vẫn hòa giữ thứ tự biến thể/cấu hình. C=10 là mức lớn nhất trong lưới, chưa phải tối ưu toàn cục. Hai khối feature được L2 riêng và nối, không chuẩn hóa lại sau nối.

- [Toàn bộ 35 cấu hình](selection/all_35_configurations_dev.csv).
- [Hàng tốt nhất từng biến thể](selection/best_by_feature_dev.csv).
- [Đối chứng linear C=1](selection/reference_dev.csv).
- [Khóa và 15 hash](selection/locked_svm_config.json).

Test chỉ đánh giá phương án khóa, fit train không gộp dev. Các CSV và cache phải khớp manifest trước khi chạy lại.
