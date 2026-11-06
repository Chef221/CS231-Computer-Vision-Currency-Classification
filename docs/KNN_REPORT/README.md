# Báo cáo KNN — lượt thực nghiệm RGB224

Đọc [BAO_CAO_THUC_NGHIEM_KNN.md](BAO_CAO_THUC_NGHIEM_KNN.md). Đây là phần nội dung để ghép vào báo cáo học phần, trình bày từ tiền xử lý đến phân tích lỗi của lượt RGB224: HSV12, HOG, HOG-PCA và hai biến thể kết hợp. Có năm bảng và sáu hình, dùng số liệu thật từ `src/KNN/results_224`.

## Hình được đặt ở đâu?

| Vị trí trong nội dung | Hình | Vai trò |
|---|---|---|
| Tiền xử lý | KNN-1 | Ảnh train trước/sau resize |
| Histogram HSV12 | KNN-2 | RGB224 và ba mặt phẳng H, S, V của hai ảnh train 10.000đ và 500.000đ |
| HOG | KNN-3 | RGB → ảnh xám → trực quan HOG |
| Đánh giá test | KNN-4 | Ma trận nhầm lẫn 6×6 |
| Phân tích lỗi | KNN-5 | C04 và C09: sai dù có dấu hiệu nhận dạng nhìn thấy |
| Phân tích ranh giới | KNN-6 | C10 sai và C11 đúng, cùng có chênh lệch khoảng cách nhỏ |

Không thêm biểu đồ cột lặp số liệu trong bảng. Histogram biên và 12 query đầy đủ vẫn nằm trong evidence của notebook 08; không cần đưa hết vào phần chính.

## Cách dùng

- Giữ `KNN224_FIGURES/` cạnh Markdown để hiện hình. Ảnh PNG 300 DPI; ma trận nhầm lẫn có thêm PDF vector.
- Số liệu dùng dấu phẩy thập phân; khoảng cách trong ảnh dùng dấu chấm. Hai cách ghi biểu diễn cùng giá trị.
- Nhãn Hình/Bảng KNN là nhãn theo phần, cần đánh số lại khi ghép báo cáo tổng. Giữ nội dung giải thích và giới hạn diễn giải trong thân bài.
- Sau mỗi feature có nhận xét. Các kết quả đo được được phân biệt với giả thuyết về nguyên nhân; không diễn đạt “HOG yếu vì nhiều chiều” hoặc “nền chắc chắn gây lỗi” khi chưa kiểm chứng.
- Bốn nguồn tham khảo là tài liệu chính thức về thuật toán/API, không phải nguồn số liệu của nhóm. Ghép số trích dẫn vào bibliography chung khi tổng hợp.
- Notebook, kết quả và hình lượt cũ đã được dọn khỏi checkout. Bản sao lưu ngoài repository được ghi trong `review/cleanup_legacy.json`. Chỉ dùng `build_figures_224.py` cho bản báo cáo này.

## Tái tạo và kiểm tra

Chạy từ repository root:

```powershell
py -3 docs/KNN_REPORT/build_figures_224.py
py -3 docs/KNN_REPORT/review/verify_report_224.py
```

Script tạo hình đọc DATA và các bảng đã có, không fit KNN/PCA hay chạy lại grid. Script kiểm tra đối chiếu các bảng và số liệu với CSV/JSON gốc; hồ sơ nội bộ nằm trong `review/`. Không đưa hồ sơ kiểm tra vào nội dung báo cáo nộp.

Báo cáo công khai lịch sử test đã được xem, giới hạn một split và phạm vi năm biến thể. Phần này ghi lại thực nghiệm KNN; đối chiếu SVM/CNN hiện được lưu trong [hồ sơ chung](../MODEL_FORMAT_REVIEW.md) và [mục lục báo cáo](../README.md). Kết quả ResNet chưa được kiểm tra trong các báo cáo này.

Theo lựa chọn trình bày của tác giả, caption riêng dưới sáu hình và dòng chú thích ở đáy hình HSV đã được bỏ. Nhãn nội bộ trong hình, thang giá trị và phần giải thích khoa học trong thân bài được giữ.
