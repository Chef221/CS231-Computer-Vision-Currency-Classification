# SVM — phân loại tiền polymer Việt Nam

Nhánh này dùng cùng DATA, RGB224 và năm biến thể đặc trưng của [KNN](../KNN/README.md): HSV12, HOG, HOG-PCA, HSV12+HOG và HSV12+HOG-PCA. Bộ phân loại là `sklearn.svm.SVC`. Các notebook giữ cách trình bày từng bước, bảng dev, cấu hình khóa, đánh giá test và phân tích lỗi như nhánh KNN. Kết quả thật đã được lưu trong `results_224`: phương án khóa là HSV12+HOG với SVC RBF, `C=10`, `gamma='scale'`; trên test đạt Accuracy 87,70%, Macro-F1 0,881102 và MAE 32.891,18đ/ảnh.

## Thứ tự notebook

| File trong `224/` | Công việc |
|---|---|
| `01_PREPARE_INPUT.ipynb` | Kiểm tra train/dev, thống nhất RGB224 và ghi cấu hình |
| `02_FEATURE_HSV12.ipynb` | Histogram HSV12, lưu ma trận và tìm SVC trên dev |
| `03_FEATURE_HOG.ipynb` | HOG, lưu vector trước/sau L2 và tìm SVC trên dev |
| `04_FEATURE_HOG_PCA.ipynb` | Fit PCA trên HOG train, transform dev, tìm SVC |
| `05A_FEATURE_HSV12_HOG.ipynb` | Nối hai khối HSV12 và HOG đã L2 |
| `05B_FEATURE_HSV12_HOG_PCA.ipynb` | Nối HSV12 và HOG-PCA đã L2 |
| `06_SELECT_FINAL_SVM.ipynb` | Tổng hợp 35 cấu hình dev và khóa phương án |
| `07_EVALUATE_TEST.ipynb` | Trích đúng feature đã khóa, fit train và chấm test |
| `08_ERROR_ANALYSIS.ipynb` | Tổng hợp lỗi, điểm quyết định SVM và xem ví dụ |

Đọc [hướng dẫn chi tiết](224/README.md) trước khi chạy. Trên Kaggle, gắn Dataset code và DATA vào mỗi notebook; từ notebook 02 trở đi gắn thêm đúng một Notebook Output của bước liền trước. Các ô tìm đường dẫn và chép kết quả đã có sẵn ở đầu notebook. `pipeline.py` chứa các hàm dùng chung để giữ phép trích feature, thứ tự ảnh, cách chấm và grid nhất quán giữa notebook; phần Markdown giải thích phương pháp và kết quả cho mọi người đọc.

[Báo cáo thực nghiệm SVM](../../docs/SVM_REPORT/BAO_CAO_THUC_NGHIEM_SVM.md) trình bày số liệu và nhận xét theo cùng phong cách báo cáo KNN, kèm hình, phụ lục 35 cấu hình và so sánh trên từng ảnh test.

## Thiết kế cố định

- Sáu nhãn VND và split 12.494/1.564/1.553 của dự án. Không augmentation; OpenCV BGR → RGB, resize trực tiếp 224×224 bằng `INTER_LINEAR`.
- HSV12 là histogram kết hợp 12×12×12, L2. HOG là ảnh xám, 9 hướng, cell 16×16, block 2×2, L2-Hys rồi L2 toàn vector. PCA fit HOG train trước L2, giữ ít nhất 95% phương sai (`full`, không whitening), sau đó L2. Hai nhánh kết hợp nối các khối đã L2, không thêm trọng số hay chuẩn hóa sau nối.
- Mỗi biến thể thử 7 cấu hình SVC: linear với `C ∈ {0.1, 1, 10}`; RBF với `C ∈ {1, 10}` và `gamma ∈ {scale, 10}`. Tổng cộng 35 cấu hình. `class_weight=None`, `cache_size=512` MB, `decision_function_shape='ovr'`, `break_ties=True`, không bật xác suất. `gamma` không ảnh hưởng kernel linear. Lưới này cố ý gọn để chạy trên Kaggle CPU; không tuyên bố đã tối ưu tuyệt đối.
- Fit SVC và PCA chỉ trên train. Chọn Macro-F1 dev cao nhất, hòa thì MAE VND thấp hơn, vẫn hòa thì giữ thứ tự biến thể và tham số đã khai báo. Test chỉ dùng một cấu hình đã khóa; không gộp train+dev.
- Bảng tham chiếu `linear, C=1` cho cả năm biến thể giúp đọc chênh lệch feature khi cố định SVC. Số chiều HOG-PCA là đầu ra tính từ train, không điền trước.

## Kết quả và giới hạn

Kết quả CSV, JSON, NPY, PCA pickle và hình đã được ghi vào `results_224/{inputs,hsv12,hog,hog_pca,hsv12_hog,hsv12_hog_pca,selection,test_final,error_analysis}`. NPY/PKL bị Git bỏ qua do dung lượng lớn. Để chạy lại notebook phụ thuộc cache, cần có đúng cache và file khóa từ các bước trước.

Phân tích lỗi SVM dùng điểm `decision_function` chưa hiệu chỉnh và ảnh minh họa. Điểm đó không phải xác suất. Ảnh train gần theo Euclidean, nếu bổ sung, cũng không phải lời giải thích trực tiếp cho quyết định SVM; notebook 08 không gán nguyên nhân thị giác khi chưa xem ảnh. Test của dự án đã từng được xem trong khảo sát cũ, nên không coi là holdout hoàn toàn mới. Nhánh này chưa xác nhận kết quả hay code của CNN/ResNet.
