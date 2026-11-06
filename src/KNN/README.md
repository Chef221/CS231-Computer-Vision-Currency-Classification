# KNN — phân loại tiền polymer Việt Nam

Nhánh hiện tại dùng RGB224, HSV12, HOG, HOG-PCA và hai biến thể kết hợp. Các notebook và kết quả khảo sát cũ đã được dọn khỏi checkout; lịch sử test từng được xem vẫn được ghi trong báo cáo.

## Cấu trúc

```text
src/KNN/
├── 224/              # 9 notebook, chạy theo thứ tự bên dưới
├── results_224/      # Feature, manifest, kết quả và bằng chứng hiện tại
├── README.md
└── requirements.txt
```

## Thứ tự notebook

| File trong `224/` | Công việc |
|---|---|
| `01_PREPARE_INPUT.ipynb` | Kiểm tra train/dev và thống nhất RGB224 |
| `02_FEATURE_HSV12.ipynb` | HSV12 + 20 cấu hình KNN trên dev |
| `03_FEATURE_HOG.ipynb` | HOG + 20 cấu hình KNN trên dev |
| `04_FEATURE_HOG_PCA.ipynb` | Fit PCA trên train, đánh giá HOG-PCA |
| `05A_FEATURE_HSV12_HOG.ipynb` | Ghép HSV12 + HOG; tạo cache HSV dùng chung |
| `05B_FEATURE_HSV12_HOG_PCA.ipynb` | Ghép HSV12 + HOG-PCA |
| `06_SELECT_FINAL_KNN.ipynb` | Tổng hợp 100 cấu hình dev và khóa phương án |
| `07_EVALUATE_TEST.ipynb` | Đánh giá đúng phương án khóa trên test |
| `08_ERROR_ANALYSIS.ipynb` | Truy vết hàng xóm, tổng hợp lỗi và xem ảnh |

Không có notebook riêng cho bước báo cáo: bản viết nằm trong [docs/KNN_REPORT](../../docs/KNN_REPORT/README.md).

## Chạy trên máy thành viên

Tải DATA theo [README dự án](../../README.md), giữ nguyên split. Cài thư viện:

```powershell
py -3 -m pip install -r src/KNN/requirements.txt
```

Sửa `PROJECT_ROOT` trong notebook theo checkout trên máy, rồi Run All theo thứ tự. Đóng/restart kernel trước khi chuyển notebook, đặc biệt trước PCA. Notebook 02 giữ ảnh RGB224 trong RAM theo cách code hiện tại; các nhánh HOG/fusion trích ảnh lần lượt. CPU đủ dùng, không cần CUDA. PCA full có thể tốn RAM/thời gian.

Notebook có thể ghi đè kết quả khi chạy lại. Nếu chỉ muốn xem số liệu, đọc báo cáo và CSV thay vì quét lại. Không thay cấu hình dựa trên lỗi test. Test đã có lịch sử sử dụng trong khảo sát trước, không phải holdout hoàn toàn chưa xem.

## Đọc kết quả

- [So sánh dev](results_224/DEV_COMPARISON.md) và [bảng năm biến thể](results_224/selection/best_by_feature_dev.csv).
- [Cấu hình đã khóa](results_224/selection/locked_knn_config.json): HSV12, k=1, uniform, Euclidean.
- [Kiểm tra test](results_224/test_final/TEST_REVIEW.md): 1.187/1.553 ảnh đúng; Accuracy 76,43%, Macro-F1 0,768710, MAE 53.947,20đ/ảnh.
- [Phân tích lỗi](results_224/error_analysis/ERROR_ANALYSIS_REVIEW.md), kèm bảng hàng xóm và ảnh.
- [Báo cáo học phần](../../docs/KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md), dùng sáu hình thật.

Giữ các file NPY, PCA pickle và manifest trong `results_224`: các notebook sau dùng lại chúng, file khóa cũng ghi hash một số file. Không xóa cache chỉ vì đã có CSV; muốn chạy lại từ đầu phải thực hiện đúng thứ tự và xác minh lại trước test. Quy tắc biểu diễn SVM và đầu vào RGB224/mapping của CNN đã được đối chiếu với KNN trong [hồ sơ rà soát chung](../../docs/MODEL_FORMAT_REVIEW.md). CNN học biểu diễn riêng; việc khớp manifest và cấu hình không khẳng định các ma trận feature của các lượt chạy trùng từng byte.

Các file `.npy` và `.pkl` được giữ local và bỏ qua bởi Git vì tổng kích thước lớn. Sau khi clone, đọc báo cáo/CSV/hình được ngay; muốn chạy notebook phụ thuộc feature, cần chạy các bước trước theo thứ tự để tạo lại cache. Nếu hash không khớp cấu hình đã khóa, dừng để kiểm tra phiên bản thư viện và dữ liệu, không tự ghi đè file khóa để bỏ qua khác biệt.
