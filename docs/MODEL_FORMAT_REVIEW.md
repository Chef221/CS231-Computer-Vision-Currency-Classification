# Rà soát SVM và CNN theo cấu trúc KNN

## Phạm vi và trạng thái nhánh

Rà soát ngày 06/11/2026 trên bản local tích hợp KNN, SVM và CNN. Báo cáo SVM đã có trong commit `6d87bc4` của nhánh `SVM`; checkout `CNN` tại `0ba58d5` chưa có vì xuất phát từ `main` sau lần hoàn tác merge SVM. Phần SVM được lấy lại từ commit đó vào working tree `CNN` để đối chiếu chung. Rà soát này không đổi nhánh, tạo commit, push hoặc thay đổi `main` trên GitHub.

Kết luận: SVM và CNN theo cùng cấu trúc thực nghiệm, cách đọc kết quả và phong cách báo cáo KNN. Khác biệt về đặc trưng, số notebook và cách lưu mô hình phản ánh phương pháp của từng mô hình. Các số liệu được đối chiếu với output đã lưu; không huấn luyện hoặc chấm lại toàn bộ test trong lần rà soát này.

## Đối chiếu cấu trúc

| Hạng mục của KNN | SVM | CNN |
|---|---|---|
| Thư mục mô hình riêng trong `src/` | `src/SVM/` | `src/CNN/` |
| Notebook RGB224 trong `224/`, Markdown giải thích cho độc giả chung | 9 file, có hai nhánh 05A/05B | 8 file |
| Chuẩn bị → khảo sát → chọn dev → test → phân tích lỗi | 5 biến thể đặc trưng, 35 cấu hình SVC | 4 cấu hình mạng, 100 dòng lịch sử epoch |
| README mô hình và hướng dẫn chạy notebook | Có; ghi cách gắn DATA, code và Output Kaggle | Có; ghi thêm GPU, resume và checkpoint |
| `results_224/inputs`, thư mục thí nghiệm, `selection`, `test_final`, `error_analysis` | Đủ manifest, grid, khóa, CSV/JSON và hình | Đủ manifest, history, khóa, CSV/JSON, hình và checkpoint local |
| Trang đọc nhanh dev/test/lỗi | Đã bổ sung theo vai trò như KNN | Đã có |
| Báo cáo trong `docs/`, README, hình cạnh Markdown | `SVM_REPORT`: 10 bảng đánh số, 6 hình, phụ lục đủ 35 cấu hình | `CNN_REPORT`: 11 bảng đánh số, 9 hình, phụ lục đủ 100 epoch |
| Nguồn hình và đối chiếu kết quả | SHA-256, phụ lục CSV, script kiểm tra độc lập | SHA-256, CSV cấu hình/lịch sử, script dựng và kiểm tra báo cáo |

SVM có thêm một notebook vì khảo sát cả ghép HSV12+HOG và HSV12+HOG-PCA. CNN không chép cách ghép đặc trưng thủ công: ảnh RGB/xám đi vào mạng để học biểu diễn. Cấu trúc báo cáo được giữ chung, còn phần kiến trúc, checkpoint, learning curve, feature map và Grad-CAM được bổ sung cho CNN.

## Dữ liệu, chọn phương án và metric

| Nội dung | Kết quả đối chiếu |
|---|---|
| Sáu nhãn | 10.000, 20.000, 50.000, 100.000, 200.000, 500.000 VND |
| Split | 12.494 train / 1.564 dev / 1.553 test; ba manifest trùng đường dẫn tương đối, nhãn và thứ tự với KNN |
| Đọc ảnh | Cùng `src.preprocessing.read_rgb224`; CNN chuyển pixel sang tensor và chia 255 sau bước này |
| Đặc trưng SVM | HSV12 1.728 chiều; HOG 6.084; HOG-PCA 1.418; hai bản ghép 7.812 và 3.146 chiều. PCA fit train; dev/test chỉ transform |
| Đặc trưng CNN | Ba hoặc bốn khối Conv; GAP 128 hoặc 256 chiều. Augmentation chỉ trên train ở C03/C04; PCA hai chiều chỉ trực quan sau khi khóa |
| Quy tắc lựa chọn | Macro-F1 dev chưa làm tròn cao nhất; hòa chính xác thì MAE thấp hơn; vẫn hòa giữ thứ tự đã khai báo/epoch sớm hơn theo protocol |
| Dữ liệu fit | Train; không gộp train+dev trước khi chấm test |
| Metric | Dùng module chung, đổi dự đoán về VND trước khi tính MAE. Macro-F1 lấy trung bình trên đủ sáu lớp; MAE tính trên toàn split, kể cả ảnh đúng |

SVM khóa cấu hình, feature và mã xử lý qua 15 hash. Notebook 07 fit SVC trên train bằng cấu hình đã khóa; notebook 08 fit lại cùng cấu hình để lấy điểm quyết định và bắt buộc dự đoán khớp CSV test đã lưu. SVM không lưu checkpoint bộ phân loại. CNN khóa 41 file, nạp `best.pt` của C04 ở epoch 24 để chấm test; đọc kết quả test đã có thay vì huấn luyện lại. Đây là hai cơ chế khác nhau, được mô tả rõ trong README và báo cáo.

## Kết quả đã xác minh

| Mô hình | Đúng / 1.553 | Accuracy | Macro-F1 | MAE (đ/ảnh) | Tổng chênh lệch (đ) |
|---|---:|---:|---:|---:|---:|
| KNN | 1.187 | 76,43% | 0,768710 | 53.947,20 | 83.780.000 |
| SVM | 1.362 | 87,70% | 0,881102 | 32.891,18 | 51.080.000 |
| CNN | 1.507 | 97,04% | 0,969963 | 9.104,96 | 14.140.000 |

- **SVM:** 15/15 hash khóa khớp tại thời điểm kiểm tra, đủ 35 cấu hình dev và 35 dòng phụ lục; tính lại metric, từng lớp, confusion matrix và MAE từ dự đoán. Sáu hình khớp nguồn. Xem [validation.json](SVM_REPORT/review/validation.json).
- **CNN:** 41/41 hash khóa khớp, 24 file được đánh dấu hoàn tất và 14 file baseline khớp; bốn cấu hình × 25 epoch. Đối chiếu metric, khóa, lịch sử, ghép cặp baseline và nguồn hình. Xem [validation.json](CNN_REPORT/review/validation.json) và [manifest báo cáo](CNN_REPORT/report_manifest.json).
- **Cache SVM:** kiểm tra shape, dtype và toàn bộ nhãn của 10 ma trận train/dev; lấy 16 dòng cố định mỗi split để kiểm tra hữu hạn, L2 và thứ tự ghép đặc trưng. Đây là kiểm tra mẫu trên ma trận; ma trận được chọn còn được kiểm tra SHA-256 toàn file theo khóa. Xem [cache_validation.json](SVM_REPORT/review/cache_validation.json).
- **Code và notebook:** toàn bộ 24 test đã qua trên CPU: 5 SVM, 7 tích hợp CNN, 10 metric và 2 tiền xử lý chung. Test dùng dữ liệu nhỏ riêng trong thư mục tạm, không thay kết quả Kaggle. Cả 17 notebook hợp lệ, có ID cell duy nhất, cú pháp ô code đúng và không lưu output cũ; xem [notebook_validation.json](SVM_REPORT/review/notebook_validation.json). Đã kiểm tra liên kết Markdown trong các README/báo cáo liên quan.

Không coi bằng nhau về tên feature là đã chứng minh ma trận đặc trưng KNN/SVM giống từng byte: HOG/PCA và thư viện thực thi có thể khác. Manifest trùng nhau cũng không chứng minh byte ảnh nguồn giữa các phiên gốc trùng nhau. So sánh trong bảng là so sánh các pipeline đã chạy trên cùng danh sách ảnh; chưa cô lập riêng hiệu quả thuật toán hay augmentation. Giữ giới hạn một seed/split và lịch sử test đã được xem trong các báo cáo.

## Các chỉnh sửa từ lần rà soát

1. Bổ sung SVM vào checkout hiện tại, tạo [mục lục báo cáo](README.md), nối lại các liên kết và thay ghi chú cũ cho rằng SVM/CNN chưa được kiểm tra.
2. Thêm trang đọc nhanh dev, test và lỗi cho SVM, script kiểm tra số liệu/hình/phụ lục và hồ sơ đối chiếu.
3. Thêm ID hợp lệ cho 128 cell của 9 notebook SVM theo nbformat. Sửa bảng lỗi rỗng ở SVM 07/08 để vẫn giữ tên cột khi dự đoán đúng toàn bộ; có test thực thi các ô liên quan. Xem [notebook_compatibility.json](SVM_REPORT/review/notebook_compatibility.json).
4. Làm rõ cách fit lại SVC ở bước 08, cách giữ output và việc SVM không tự tạo ZIP như CNN. Bổ sung ignore cho cache NPY/PKL lớn; các file này vẫn cần lưu trong output thực nghiệm đầy đủ ngoài Git.

Không thay mã feature/mô hình đã khóa, cấu hình, CSV dự đoán hoặc kết quả metric của lượt Kaggle. Các điểm quyết định SVM và softmax CNN chưa hiệu chỉnh; hình đặc trưng/Grad-CAM là quan sát mô tả, không chứng minh nguyên nhân nhầm lẫn. ResNet không nằm trong phạm vi rà soát này.

## Lệnh kiểm tra lại

Chạy tại repository root sau khi cài requirements của SVM/CNN:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python docs/SVM_REPORT/review/verify_results_224.py
python docs/CNN_REPORT/review/verify_results_224.py
```

Các script kiểm tra báo cáo dùng thư viện chuẩn Python. Khi clone không có cache/checkpoint lớn, script ghi rõ file vắng mặt; chỉ file thực sự có mới được xác minh hash. Hồ sơ JSON hiện tại phản ánh bản local đã khôi phục đủ cache và checkpoint, không thay cam kết về nội dung Git.

Đọc nội dung báo cáo: [KNN](KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md), [SVM](SVM_REPORT/BAO_CAO_THUC_NGHIEM_SVM.md), [CNN](CNN_REPORT/BAO_CAO_THUC_NGHIEM_CNN.md).
