# Báo cáo SVM — lượt thực nghiệm RGB224

Đọc [BAO_CAO_THUC_NGHIEM_SVM.md](BAO_CAO_THUC_NGHIEM_SVM.md). Đây là phần nội dung để ghép vào báo cáo học phần, trình bày dữ liệu, tiền xử lý, năm biến thể đặc trưng, 35 cấu hình SVC trên dev, phương án khóa, kết quả test, phân tích lỗi và so sánh với KNN trên cùng split. Báo cáo dùng số liệu thật từ `src/SVM/results_224` và `src/KNN/results_224`; không dùng số đo giả định.

## Hình được đặt ở đâu?

| Vị trí trong nội dung | Hình | Vai trò |
|---|---|---|
| Tiền xử lý | SVM-1 | Ví dụ ảnh nguồn và ảnh RGB224 theo sáu lớp |
| HOG | SVM-2 | Ảnh RGB, ảnh xám và trực quan HOG |
| HOG-PCA | SVM-3 | Phương sai tích lũy trên train |
| Đánh giá test | SVM-4 | Ma trận nhầm lẫn SVM 6×6 |
| Phân tích lỗi | SVM-5 | Bốn ví dụ C09–C12 và điểm quyết định SVM |
| So sánh KNN | SVM-6 | Ma trận nhầm lẫn KNN trên cùng test |

Các hình SVM-1 đến SVM-5 là bản sao từ output notebook SVM; SVM-6 là hình tham chiếu từ báo cáo KNN. [Provenance](SVM224_FIGURES/provenance.json) ghi đường dẫn nguồn và SHA-256 của từng hình. `PHU_LUC_35_CAU_HINH_SVM.csv` là bảng gốc đầy đủ; các bảng trong Markdown làm tròn để dễ đọc.

## Cách dùng

- Giữ `SVM224_FIGURES/` cạnh Markdown để hiện hình. Nhãn Bảng SVM và Hình SVM là nhãn theo phần; có thể đánh số lại khi ghép vào báo cáo tổng.
- Đọc kết quả dev/test cùng [cấu hình khóa SVM](../../src/SVM/results_224/selection/locked_svm_config.json), [metric test SVM](../../src/SVM/results_224/test_final/metrics_test.json) và [báo cáo KNN](../KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md). So sánh chính dùng đúng 1.553 ảnh test và cùng thứ tự nhãn.
- Số liệu tách biệt với giả thuyết về nguyên nhân thị giác. Các điểm `decision_function` chưa được hiệu chỉnh thành xác suất; `visual_review_notes.csv` của SVM hiện chưa có nhận xét thị giác được xác minh.
- Không đưa ma trận NPY và PCA pickle vào Git. Chúng cần cho việc chạy lại notebook phụ thuộc cache, còn Markdown, CSV/JSON và hình đủ để đọc, đối chiếu kết quả đã lưu.
- Các trang đọc nhanh [dev](../../src/SVM/results_224/DEV_COMPARISON.md), [test](../../src/SVM/results_224/test_final/TEST_REVIEW.md) và [lỗi](../../src/SVM/results_224/error_analysis/ERROR_ANALYSIS_REVIEW.md) có cùng vai trò như phần KNN/CNN. [Mục lục báo cáo](../README.md) dẫn tới cả ba mô hình.

## Tái tạo hình

Chạy từ repository root:

```bash
python docs/SVM_REPORT/build_figures_224.py
python docs/SVM_REPORT/review/verify_results_224.py
```

Script chỉ chép sáu hình đã được tạo bởi notebook/báo cáo KNN và ghi SHA-256; không fit lại SVC/PCA hoặc chạy lại grid. Báo cáo công khai giới hạn một split, lịch sử test từng được xem và phạm vi lưới cấu hình đã thử.

Script kiểm tra đối chiếu đủ 35 cấu hình, quy tắc lựa chọn, dự đoán dev/test, metric từng lớp, ma trận, MAE, 15 hash khóa và nguồn của sáu hình. [validation.json](review/validation.json) giữ kết quả kiểm tra bản local; nếu thiếu cache sau khi clone, file này ghi rõ cache vắng mặt khi chạy lại. Script không chạy lại huấn luyện/test. Hồ sơ rà soát cấu trúc chung nằm tại [MODEL_FORMAT_REVIEW.md](../MODEL_FORMAT_REVIEW.md).
