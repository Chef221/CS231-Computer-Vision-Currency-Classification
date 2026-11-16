# Kết quả cuối dùng trong slide

Train12494 / dev1564 / test1553 ảnh. Chọn cấu hình bằng dev, rồi chấm test.

| Mô hình | Đúng | Sai | Accuracy | Macro-F1 | MAE (đ/ảnh) |
|---|---|---|---|---|---|
| KNN | 1187 | 366 | 76.43% | 0.768710 | 53,947.20 |
| SVM | 1362 | 191 | 87.70% | 0.881102 | 32,891.18 |
| CNN | 1512 | 41 | 97.36% | 0.973945 | 7,250.48 |

- KNN: HSV12, k=1, uniform, Euclidean. [Báo cáo gốc](KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md).
- SVM: HSV12 + HOG, RBF, C=10, gamma=scale. [Bảng và hình](../src/SVM/results_224/README.md).
- CNN: C04_DEEP_AUG, epoch25. [Bảng và hình](../src/CNN/results_224/README.md).

CNN TensorFlow mới đúng1512 ảnh; không dùng lại mốc1507 ảnh của PyTorch hoặc1393 ảnh của bản TensorFlow trước. Các mốc cũ được giữ trong bản sao lưu trên máy.

So sánh SVM/CNN theo cùng đường dẫn từng ảnh test: cả hai đúng1343; chỉ CNN đúng169; chỉ SVM đúng19; cả hai sai22. Dù CNN tốt hơn tổng thể, vẫn có ảnh SVM nhận đúng mà CNN nhận sai. Một split và một seed chưa đủ chứng minh kết quả trên ảnh mới; test đã có lịch sử được xem.

## Đọc số liệu đúng cách

- Macro-F1 là trung bình F1 của sáu lớp.
- MAE tính chênh lệch mệnh giá tuyệt đối trên toàn bộ ảnh, kể cả ảnh đúng góp0; đơn vị đồng/ảnh.
- Dev và test là hai tập khác nhau nên số liệu có thể khác dù dùng cùng checkpoint.
- Code được trình bày lại để dễ học, giữ phương pháp và cấu hình baseline. Chưa huấn luyện lại toàn bộ dữ liệu sau khi dọn; bảng này là số liệu đã lưu của nhóm.

## Xem mà không train

Mở `00_READ_RESULTS.ipynb` trong SVM/CNN. GitHub hiển thị sẵn bảng/hình từ các ô Markdown. CSV/JSON và hình không phụ thuộc model nhị phân. Muốn tự dự đoán ảnh mới cần checkpoint/model hoặc chạy lại notebook.
