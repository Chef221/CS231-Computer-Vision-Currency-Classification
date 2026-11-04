# Kế hoạch hình cho phần báo cáo KNN

Ba hình đều là evidence/result, tạo bằng matplotlib từ CSV và PNG thật, không dùng ảnh sinh.

| Hình | Thông điệp | Nguồn | Bố cục / backend | Rủi ro và cách xử lý |
| --- | --- | --- | --- | --- |
| KNN-1 | Đọc số ảnh đúng và các cặp nhầm trên toàn test | hsv12_test/hsv12_confusion_matrix_test.csv | Heatmap 6×6, có số đếm; matplotlib; PNG 300 dpi và SVG | Ghi rõ hàng thật, cột dự đoán, đơn vị nghìn đồng |
| KNN-2 | Ảnh rõ có thể gần histogram của lớp sai hơn lớp đúng | neighbor_analysis/five_nearest_neighbors.csv, 10000_002541 và train rank1/rank3 | Ba ảnh: query, sai gần nhất, đúng lớp gần nhất trong top5; matplotlib PNG 300 dpi | Không kết luận nền gây lỗi; ghi đúng hạng và khoảng cách |
| KNN-3 | k=1 chọn sai khi hai nhãn có khoảng cách gần bằng nhau | 500000_000096 và train rank1/rank2 | Ba ảnh, khoảng cách 9 chữ số thập phân; matplotlib PNG 300 dpi | Không gọi là hòa tuyệt đối, xác suất hoặc lỗi tính toán |

Không thêm biểu đồ cột lặp bảng dev, learning curve không có dữ liệu, t-SNE/PCA hoặc histogram 1.728 bin khó đọc. Ảnh nhòe vẫn được nhận xét bằng số hàng xóm, không thêm một hình lặp ý. Caption dùng nhãn KNN-1…3 để dễ đánh số lại khi ghép báo cáo tổng.
