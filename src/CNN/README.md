# CNN — phân loại tiền polymer Việt Nam

Nhánh này dùng CNN tự xây dựng, huấn luyện từ đầu trên cùng sáu mệnh giá và split RGB224 của KNN/SVM. Mạng tự học đặc trưng từ pixel, không ghép histogram/HOG vào đầu vào. Lượt Kaggle đã hoàn tất: phương án khóa là **C04_DEEP_AUG, epoch 24**, đạt **Accuracy test 97,04%, Macro-F1 0,969963, MAE 9.104,96đ/ảnh**, đúng 1.507/1.553 ảnh. Cả bốn cấu hình được huấn luyện đủ 25 epoch trên Tesla T4; test dùng checkpoint đã chọn, không fit lại.

## Cấu trúc

```text
src/CNN/
├── 224/                  # 8 notebook, chạy theo thứ tự
├── results_224/          # Manifest, lịch sử, checkpoint, dev/test và hình
├── reference_results/    # Bản sao có hash của baseline KNN/SVM
├── model.py              # Các khối Conv và khởi tạo trọng số
├── data.py               # Mapping, Dataset, augmentation và protocol
├── pipeline.py           # Huấn luyện, chọn checkpoint, khóa và test
├── analysis.py           # Feature map, PCA biểu diễn, Grad-CAM và so sánh
├── runtime.py            # Khôi phục Notebook Output trên Kaggle
├── README.md
└── requirements.txt
```

## Thứ tự notebook

| File trong `224/` | Công việc |
|---|---|
| `01_PREPARE_INPUT.ipynb` | Kiểm tra train/dev, RGB224, minh họa augmentation, ghi bốn cấu hình |
| `02_TRAIN_RGB.ipynb` | CNN ba khối nhận RGB, không augmentation |
| `03_TRAIN_GRAY.ipynb` | Cùng số khối, đầu vào xám để khảo sát màu |
| `04_TRAIN_RGB_AUG.ipynb` | CNN RGB ba khối, thêm augmentation trên train |
| `05_TRAIN_DEEP_AUG.ipynb` | CNN RGB bốn khối, cùng augmentation |
| `06_SELECT_FINAL_CNN.ipynb` | So sánh dev, khóa trọng số; xem feature map và PCA của vector GAP |
| `07_EVALUATE_TEST.ipynb` | Nạp đúng checkpoint đã khóa, đánh giá toàn test |
| `08_ERROR_ANALYSIS.ipynb` | Cặp nhầm, Grad-CAM, so KNN/SVM, tạo báo cáo và ZIP kết quả |

Đọc [hướng dẫn Kaggle](224/README.md). Các ô tìm code/DATA và chép output bước trước đã có sẵn. Notebook 01–06 không đọc ảnh test. Sau khi khóa, notebook 07 dùng chính `best.pt` đã chọn; không huấn luyện lại hay gộp train+dev.

## Kiến trúc và thiết kế cố định

- Mỗi khối: hai lần Conv3×3–BatchNorm–ReLU, rồi MaxPool2×2 và Dropout2d(0,1). Ba khối có 32–64–128 kênh; biến thể sâu thêm khối 256. GAP tạo vector 128 hoặc 256 chiều, rồi Dense128–ReLU–Dropout(0,5)–Dense6 logits. Tổng tham số: 304.742 cho RGB cơ sở, 304.166 cho ảnh xám, 1.206.886 cho mạng sâu. Không pretrained, không residual block.
- Đọc bằng cùng `src.preprocessing.read_rgb224` như KNN/SVM; sau đó chia pixel cho 255. Ánh xạ chỉ số lớp theo mệnh giá tăng dần, đổi về VND trước khi tính metric.
- Bốn cấu hình được khai báo trước. Seed 42, batch 64, AdamW lr=0,001, weight decay=0,0001; cosine learning rate xuống 0,00001; tối đa 25 epoch, patience 6. Conv khởi tạo Kaiming, Dense Xavier, BatchNorm scale 1/bias 0. Không class weight và label smoothing.
- Mỗi checkpoint được chọn bằng Macro-F1 dev chưa làm tròn; hòa chính xác thì MAE thấp hơn, vẫn hòa giữ epoch sớm hơn. So cấu hình bằng cùng quy tắc, ưu tiên thứ tự C01…C04 trước epoch khi vẫn hòa. Test không chọn lại cấu hình.
- Augmentation chỉ ở C03/C04: xoay ±12°, dịch ±5%, scale 0,95–1,05, sáng/contrast nhẹ; không lật hoặc đổi hue/saturation. PCA hai chiều ở notebook 06 chỉ trực quan vector GAP sau khi đã khóa, không là đầu vào phân loại.
- Feature map và Grad-CAM phục vụ quan sát; softmax chưa hiệu chỉnh. Không tự gán nguyên nhân lỗi từ heatmap và không triển khai cơ chế từ chối trong bảng test chính.

## Đọc và giữ kết quả

Thư mục [results_224](results_224/README.md) đã nhận đầy đủ output notebook 08. Các kết quả đọc nhanh:

- [Bốn cấu hình dev](results_224/DEV_COMPARISON.md): C04 đạt Macro-F1 dev 0,964559; C03 chỉ nhỉnh hơn C01 0,000308 Macro-F1 nhưng MAE cao hơn 684,14đ/ảnh.
- [Cấu hình đã khóa](results_224/selection/locked_cnn_config.json): RGB, bốn khối, augmentation train, vector GAP 256 chiều, checkpoint epoch 24.
- [Kiểm tra test](results_224/test_final/TEST_REVIEW.md): 1.507 ảnh đúng, 46 ảnh sai, tổng chênh lệch 14.140.000đ.
- [Phân tích lỗi](results_224/error_analysis/ERROR_ANALYSIS_REVIEW.md): cặp 20.000đ ↔ 500.000đ còn 16 lỗi, chiếm 54,31% tổng chênh lệch VND.
- [So sánh KNN/SVM](results_224/comparison/metrics_comparison_test.csv) và [ghép cặp từng ảnh](results_224/comparison/paired_summary_test.csv): CNN đúng thêm ròng 320 ảnh so với KNN, 145 ảnh so với SVM; vẫn có ảnh baseline đúng mà CNN sai.

Giữ `best.pt`, `last.pt`, manifest và file khóa trong output Kaggle để chạy bước sau. Bản local đã có trọng số; `.pt/.pth` và ma trận lớn bị Git bỏ qua. Sau khi clone cần khôi phục output đầy đủ hoặc chạy lại các bước trước để có checkpoint, không chỉ dựa vào CSV/hình.

[Báo cáo CNN](../../docs/CNN_REPORT/BAO_CAO_THUC_NGHIEM_CNN.md) dùng bố cục như báo cáo KNN/SVM, với kiến trúc, nhận xét từng cấu hình, bảng dev/test, thời gian đo thật, phân tích lỗi và so sánh. Bản trong `docs/CNN_REPORT` và bản tại `results_224/report/` được dựng từ cùng output đã kiểm tra; có phụ lục đủ 100 dòng lịch sử epoch và nguồn hình theo hash.

Các thư viện/hardware được ghi trong từng run và khi chấm test. Một GPU Kaggle được dùng nếu có; CPU phù hợp kiểm tra nhỏ nhưng huấn luyện đầy đủ sẽ chậm. Chỉ có một seed/split, test đã từng được xem, nên chưa suy kết quả thành độ tin cậy ngoài thực tế.

## Kiểm tra code

Sau khi cài `requirements.txt` của CNN, chạy từ repository root:

```bash
python -m unittest discover -s tests -p test_cnn_pipeline.py -v
```

Bộ kiểm tra dùng ảnh và protocol nhỏ tạo riêng trong thư mục tạm để xác minh mapping VND, resume sau gián đoạn, chọn checkpoint, phát hiện thay đổi hash, đọc lại test, Grad-CAM và dựng báo cáo. Các số từ fixture không được ghi vào `results_224` hoặc báo cáo chính thức; kiểm tra CPU không thay một lượt huấn luyện đầy đủ trên Kaggle GPU.

[Hồ sơ rà soát chung](../../docs/MODEL_FORMAT_REVIEW.md) đối chiếu SVM/CNN theo cấu trúc KNN; [mục lục báo cáo](../../docs/README.md) dẫn tới đầy đủ ba phần.
