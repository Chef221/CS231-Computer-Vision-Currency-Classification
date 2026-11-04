# Thực nghiệm KNN

Báo cáo chi tiết: [EXPERIMENTS.md](EXPERIMENTS.md). Bảng tổng hợp: [knn_all_features_dev.csv](results/knn_all_features_dev.csv). Đây là kết quả trên **dev**, chưa có kết quả test.

## Notebook

| File | Nội dung |
| --- | --- |
| `01_INSPECT_TRAIN_IMAGE.ipynb` | Khảo sát ảnh train và kích thước |
| `02_FEATURE_RGB_HISTOGRAM.ipynb` | RGB histogram 768 chiều |
| `03_FEATURE_HSV_HISTOGRAM.ipynb` | HSV 8×8×8, 512 chiều |
| `04_FEATURE_COLOR_MOMENTS.ipynb` | Mean, std, skewness RGB, 9 chiều |
| `05_FEATURE_HOG.ipynb` | HOG 128×128, cell16, 1.764 chiều |
| `06_FEATURE_HOG_CELL8.ipynb` | HOG 128×128, cell8, 8.100 chiều |
| `08_FEATURE_HSV_HISTOGRAM_12BINS.ipynb` | HSV 12×12×12, 1.728 chiều |
| `09_FEATURE_HSV_HISTOGRAM_16BINS.ipynb` | HSV 16×16×16, 4.096 chiều |

Giữ số notebook theo lịch sử thực nghiệm; không có notebook 07 trong gói hiện tại. Mỗi notebook feature có lưới 120 cấu hình KNN; chọn bằng Macro-F1 dev, phá hòa bằng MAE_VND rồi thứ tự thử. Không lấy số cấu hình làm số lần lặp độc lập.

## Chạy trên máy thành viên

1. Tải DATA theo hướng dẫn ở [README gốc](../../README.md). Giữ nguyên train/dev/test.
2. Cài môi trường Python và thư viện:

```powershell
py -3 -m pip install -r src/KNN/requirements.txt
```

3. Mở notebook muốn chạy. Đổi `PROJECT_ROOT = Path(r"C:\CS231")` thành đường dẫn checkout trên máy mình. Notebook RGB và notebook 01 còn các đường dẫn tuyệt đối trong cell; sửa chúng về DATA của checkout tương ứng.
4. Đặt working directory của kernel là `src/KNN`: các notebook RGB/HSV lưu vào `Path("results")`. HOG/Color Moments dùng đường dẫn results theo PROJECT_ROOT. Không chạy đồng thời nhiều notebook giữ ảnh trong RAM; restart kernel cũ trước khi đổi notebook. HOG đã đọc từng ảnh; các notebook màu hiện giữ ảnh gốc trong RAM và có thể cần nhiều bộ nhớ.
5. Chạy cell từ trên xuống. Nếu chỉ muốn đọc kết quả, không cần chạy lại tìm cấu hình: xem báo cáo và CSV có sẵn. Chưa chạy test trong các notebook kết quả này.

Môi trường dùng để **kiểm tra kết quả** được ghi trong [audit_results_verification.json](results/audit_results_verification.json). requirements chỉ liệt kê thư viện cần thiết, không cam kết đã khôi phục chính xác phiên bản của mọi lượt chạy gốc. CPU đủ để chạy các thí nghiệm, không cần CUDA; cấu hình nhiều chiều có thể mất nhiều thời gian.

## Kiểm tra các kết quả đã lưu

Từ thư mục gốc repo:

```powershell
py -3 src/KNN/audit_results.py
```

Lệnh này không đọc ảnh DATA, không training; kiểm tra CSV rồi tạo lại báo cáo, bảng tổng hợp, bảng lỗi có trọng số chênh lệch VND, hình so sánh và JSON bằng chứng/hash. Nó dùng assertions cho kiểm tra nội bộ, không phải code notebook thực nghiệm. Kiểm tra nhất quán bảng số không thay thế chạy lại dự đoán từ ảnh.

CSV gốc: `*_knn_grid_dev.csv`, `*_best_dev.csv` (RGB không có summary gốc), `*_confusion_matrix_dev.csv`, `*_per_class_dev.csv` nếu có. CSV bổ sung `*_error_costs_dev.csv` được dựng từ confusion matrix: nhãn thật, dự đoán, số lỗi, chênh lệch mỗi lỗi, tổng chênh lệch. Không phải tổn thất tài chính thực tế.

Ứng viên hiện được chọn trên dev: **HSV12 + KNN k=1, uniform, Bray–Curtis**. Kết quả test là bước tiếp theo sau khi khóa cấu hình, không phải một phần của gói hiện tại.
