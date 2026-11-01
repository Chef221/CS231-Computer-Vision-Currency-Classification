# Phân loại mệnh giá tiền giấy Việt Nam

## Bộ dữ liệu

Bộ cuối gồm 15.611 ảnh PNG, chia theo thư mục lớp và tập dữ liệu:

| Mệnh giá | Train | Dev | Test | Tổng |
|---:|---:|---:|---:|---:|
| 10.000 | 2.299 | 315 | 281 | 2.895 |
| 20.000 | 1.916 | 238 | 224 | 2.378 |
| 50.000 | 1.646 | 195 | 223 | 2.064 |
| 100.000 | 1.918 | 250 | 247 | 2.415 |
| 200.000 | 2.525 | 309 | 317 | 3.151 |
| 500.000 | 2.190 | 257 | 261 | 2.708 |
| **Tổng** | **12.494** | **1.564** | **1.553** | **15.611** |

Thư mục có dạng `DATA/{train,dev,test}/{10000,20000,50000,100000,200000,500000}/`. Tên ảnh là `<mệnh_giá>_<số thứ tự sáu chữ số>.png`; số thứ tự tăng theo từng lớp trên toàn bộ dữ liệu. `DATA/metadata/dataset.csv` lưu nguồn, đường dẫn ảnh gốc và ảnh crop, nhóm ảnh, split, mệnh giá và đường dẫn kết quả tương đối `DATA`.

## Tải dự án và dữ liệu

Clone nông để lấy code và metadata hiện tại, không tải lịch sử cũ có chứa ảnh lớn:

```powershell
git clone --depth 1 https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification.git
cd CS231-Computer-Vision-Currency-Classification
```

Nếu đã có bản clone, dùng `git pull` sau khi lưu hoặc commit công việc local cần giữ; không dùng lệnh reset hoặc clean để cập nhật.

Mở GitHub Release `m0-dataset-2026-11-01`, tải đủ bốn tệp và giải nén tất cả vào thư mục gốc dự án. Mỗi tệp chứa đường dẫn bắt đầu bằng `DATA/`; hai tệp train chứa các phần ảnh khác nhau. Tệp dev có thêm `DATA/metadata/dataset.csv`.

```powershell
Expand-Archive .\data-train-1-of-2.zip -DestinationPath . -Force
Expand-Archive .\data-train-2-of-2.zip -DestinationPath . -Force
Expand-Archive .\data-dev.zip -DestinationPath . -Force
Expand-Archive .\data-test.zip -DestinationPath . -Force
```

Dùng train để huấn luyện, dev để chọn mô hình/cấu hình, và chỉ dùng test để đánh giá cuối. Resize ảnh trong pipeline của mô hình khi cần; ảnh PNG trong bộ dữ liệu được giữ nguyên độ phân giải.

## Quy trình M0

Cài thư viện và mở notebook trong `src/M0` theo thứ tự `00` đến `05`:

```powershell
pip install -r requirements.txt
```

1. `00_COUNT_RAW_DATA.ipynb`: đếm ảnh và nhãn nguồn.
2. `01_READ_LABELS.ipynb`: đọc nhãn, tổng hợp sáu mệnh giá và xem ví dụ.
3. `02_CROP_BANKNOTES.ipynb`: tạo một crop cho mỗi nhãn; vùng crop thêm padding 20% mỗi phía. Ảnh giữ nguyên kích thước, không augment.
4. `03_REMOVE_EXACT_DUPLICATES.ipynb`: bỏ bản trùng pixel hoàn toàn trong cùng mệnh giá; giữ ảnh nguồn.
5. `04_SPLIT_TRAIN_DEV_TEST.ipynb`: chia train/dev/test theo nhóm tên ảnh gốc với seed cố định.
6. `05_EXPORT_FINAL_DATA.ipynb`: sao chép và đặt tên ảnh cuối theo thư mục lớp.

Các ảnh nguồn chưa được đóng gói trong release. Muốn chạy lại các bước từ đầu, tải bốn bộ nguồn từ trang Roboflow bên dưới và đặt chúng theo cấu trúc mà `00_COUNT_RAW_DATA.ipynb` đọc dưới `DATA_RAW/`.

## Nguồn và ghi công

Bốn bộ nguồn ghi giấy phép CC BY 4.0 trong tệp `README.dataset.txt`/`README.roboflow.txt` đi kèm:

- [Banknotes Recognition v2](https://universe.roboflow.com/money-recognition/banknotes-recognition/dataset/2)
- [Money v1](https://universe.roboflow.com/bao-vu/money-bx6u5-a5myk/dataset/1)
- [Money_VN_Regconization v1](https://universe.roboflow.com/blindtrix/money_vn_regconization-2-cumhs/dataset/1)
- [Vietnamese Currency v6](https://universe.roboflow.com/first-qkz37/vietnamese-currency-lgi9i/dataset/6)

Bộ cuối là kết quả xử lý và kết hợp các nguồn trên: chỉ giữ sáu mệnh giá mục tiêu, crop theo nhãn, loại trùng pixel hoàn toàn trong cùng lớp, rồi chia lại theo nhóm. Việc ghi nguồn không hàm ý các tác giả nguồn bảo trợ dự án này.
