# Chạy SVM/CNN trên Kaggle

## 1. Chuẩn bị

Tại gốc repo chạy:

```bash
python tools/build_bundle.py --kind kaggle
```

Upload `output/CS231_KAGGLE_CODE.zip` thành Dataset code. Dataset sau giải nén phải có `src/` ngay dưới gốc. Gắn cùng một version Dataset code vào tất cả notebook. Gắn thêm Dataset ảnh có `DATA/train`, `DATA/dev`, `DATA/test` hoặc trực tiếp ba thư mục train/dev/test.

Import notebook từ `src/SVM/224` hoặc `src/CNN/224`. Notebook00 chỉ xem kết quả đã có trên GitHub, không nằm trong bộ chạy Kaggle.

## 2. Chạy lần lượt

- SVM: **01 → 02 → 03 → 04 → 05A → 05B → 06 → 07 → 08**. CPU đủ dùng.
- CNN: **01 → 02 → 03 → 04 → 05 → 06 → 07 → 08**. Bật GPU cho các bước train và suy luận; dùng một GPU mặc định.

Mỗi bước chọn **Save Version → Save & Run All**. Khi thành công, gắn **Output bước đó** vào bước tiếp theo. Mỗi Output cần giữ toàn bộ kết quả các bước trước. Notebook sẽ sao chép các file cần dùng sang `/kaggle/working` rồi ghi tiếp.

Đầu ra:

```text
/kaggle/working/CS231_SVM/results_224/
/kaggle/working/CS231_CNN/results_224/
```

Khi nhiều bộ code/ảnh/Output cùng được gắn, bỏ comment và sửa các biến ở ô đầu notebook:

```python
os.environ['CS231_PROJECT_ROOT'] = '/kaggle/input/TEN-DATASET-CODE'
os.environ['CS231_DATA_ROOT'] = '/kaggle/input/TEN-DATASET-ANH/DATA'
os.environ['CS231_CNN_PREVIOUS_RESULTS'] = '/kaggle/input/.../CS231_CNN/results_224'
# Với SVM đổi CNN thành SVM.
```

Không gắn kết quả lịch sử để chạy tiếp bộ notebook đã được sắp xếp lại. Bộ này bắt đầu từ01. Bảng/hình công bố trong repo chỉ để tham khảo và không bị ghi đè khi chạy.

## 3. Thư viện và bộ nhớ

Kaggle nên dùng TensorFlow có sẵn, không cần cài lại CUDA. Ô import in phiên bản TensorFlow, GPU và kiểu tính toán. Nếu thiếu thư viện phụ thì cài theo requirements của thuật toán rồi restart session.

SVM giữ nguyên HSV/HOG/PCA như KNN. PCA full SVD có thể tốn thời gian/RAM; không mở nhiều lượt fit cùng lúc. CNN batch64 đọc ảnh theo batch, không cache toàn bộ ảnh float. AMP khi train GPU, dev/test float32. Bản học tập lưu best.keras, nhưng không có chức năng chạy tiếp epoch sau gián đoạn; chạy lại hàm train bắt đầu từepoch1.

## 4. Lấy kết quả

Bước08 xuất `CS231_SVM_RESULTS.zip` hoặc `CS231_CNN_RESULTS.zip` cạnh thư mục kết quả. ZIP gồm model/checkpoint, bảng và hình đã tạo trong chuỗi đó. Tải Output version thành công để giữ lại trước khi chạy lượt khác.

Chọn cấu hình và checkpoint chỉ theo dev. Không chỉnh theo test để cố khớp một con số trong slide. Môi trường/thư viện và tính ngẫu nhiên có thể làm kết quả chạy lại khác.
