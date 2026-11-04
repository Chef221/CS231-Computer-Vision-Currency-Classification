# Thực nghiệm KNN

Báo cáo chi tiết: [EXPERIMENTS.md](EXPERIMENTS.md). Bảng tổng hợp: [knn_all_features_dev.csv](results/knn_all_features_dev.csv). Báo cáo trên ghi giai đoạn **dev**; kết quả cuối nằm trong [TEST_RESULTS.md](TEST_RESULTS.md), kèm dự đoán từng ảnh trong [results/hsv12_test](results/hsv12_test).

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
| `10_EVALUATE_HSV12_TEST.ipynb` | Fit cấu hình HSV12 đã khóa trên train, đánh giá toàn test |
| `11_ANALYZE_KNN_NEIGHBORS.ipynb` | Truy vết ba lỗi test, xem năm hàng xóm; quyết định chính thức vẫn k=1 |

Giữ số notebook theo lịch sử thực nghiệm; không có notebook 07 trong gói hiện tại. Bảy notebook khảo sát feature có lưới 120 cấu hình KNN mỗi notebook; notebook 10 không tìm tham số; chọn bằng Macro-F1 dev, phá hòa bằng MAE_VND rồi thứ tự thử. Không lấy số cấu hình làm số lần lặp độc lập.

## Chạy trên máy thành viên

1. Tải DATA theo hướng dẫn ở [README gốc](../../README.md). Giữ nguyên train/dev/test.
2. Cài môi trường Python và thư viện:

```powershell
py -3 -m pip install -r src/KNN/requirements.txt
```

3. Mở notebook muốn chạy. Đổi `PROJECT_ROOT = Path(r"C:\CS231")` thành đường dẫn checkout trên máy mình. Notebook RGB và notebook 01 còn các đường dẫn tuyệt đối trong cell; sửa chúng về DATA của checkout tương ứng.
4. Đặt working directory của kernel là `src/KNN`: các notebook RGB/HSV lưu vào `Path("results")`. HOG/Color Moments dùng đường dẫn results theo PROJECT_ROOT. Không chạy đồng thời nhiều notebook giữ ảnh trong RAM; restart kernel cũ trước khi đổi notebook. HOG đã đọc từng ảnh; các notebook màu hiện giữ ảnh gốc trong RAM và có thể cần nhiều bộ nhớ.
5. Chạy cell từ trên xuống. Nếu chỉ muốn đọc kết quả, không cần chạy lại tìm cấu hình: xem báo cáo và CSV có sẵn. Notebook 10 đã chạy test; chỉ cần đọc CSV nếu xem lại kết quả, không cần chạy lại để tìm tham số.

Môi trường dùng để **kiểm tra kết quả** được ghi trong [audit_results_verification.json](results/audit_results_verification.json). requirements chỉ liệt kê thư viện cần thiết, không cam kết đã khôi phục chính xác phiên bản của mọi lượt chạy gốc. CPU đủ để chạy các thí nghiệm, không cần CUDA; cấu hình nhiều chiều có thể mất nhiều thời gian.

## Kiểm tra các kết quả đã lưu

Từ thư mục gốc repo:

```powershell
py -3 src/KNN/audit_results.py
```

Lệnh này không đọc ảnh DATA, không training; kiểm tra CSV rồi tạo lại báo cáo, bảng tổng hợp, bảng lỗi có trọng số chênh lệch VND, hình so sánh và JSON bằng chứng/hash. Nó dùng assertions cho kiểm tra nội bộ, không phải code notebook thực nghiệm. Kiểm tra nhất quán bảng số không thay thế chạy lại dự đoán từ ảnh.

CSV gốc: `*_knn_grid_dev.csv`, `*_best_dev.csv` (RGB không có summary gốc), `*_confusion_matrix_dev.csv`, `*_per_class_dev.csv` nếu có. CSV bổ sung `*_error_costs_dev.csv` được dựng từ confusion matrix: nhãn thật, dự đoán, số lỗi, chênh lệch mỗi lỗi, tổng chênh lệch. Không phải tổn thất tài chính thực tế.

Cấu hình đã khóa: **HSV12 + KNN k=1, uniform, Bray–Curtis**. Test đã chạy: **1.354/1.553 ảnh đúng**, Accuracy **87,19%**, Macro-F1 **0,874772**, MAE **31.461,69 VND/ảnh**. Không chỉnh lại mô hình dựa trên test.

Kiểm tra kết quả test (cần DATA/test để đối chiếu đủ đường dẫn; không đọc nội dung ảnh hoặc suy luận lại):

```powershell
py -3 src/KNN/audit_test_results.py
```

Lệnh này tạo lại TEST_RESULTS.md và các bảng đối chiếu từ dự đoán đã lưu, xác minh summary/JSON/confusion matrix/bảng lỗi. Notebook 10 lưu riêng cấu hình khóa, dự đoán từng ảnh, chỉ số và hình lỗi trong `results/hsv12_test/`.

## Phần báo cáo học phần và phân tích hàng xóm

[Báo cáo KNN](../../docs/KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md) trình bày năm cấu hình được chọn đưa vào báo cáo, kết quả test và phân tích lỗi, kèm ba hình minh họa. Các thí nghiệm còn lại vẫn được giữ trong hồ sơ đầy đủ.

Notebook 11 xuất bảng và hình vào [results/neighbor_analysis](results/neighbor_analysis). Cả ba dự đoán truy vết khớp kết quả test đã lưu. Năm hàng xóm chỉ để phân tích, không chọn lại k theo test. Script [build_figures.py](../../docs/KNN_REPORT/build_figures.py) dựng hình báo cáo và kiểm tra lại khoảng cách của 15 cặp ảnh, cần DATA trên máy; không fit hoặc thay đổi mô hình.
