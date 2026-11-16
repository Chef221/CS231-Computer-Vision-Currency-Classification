# SVM — phân loại mệnh giá

Lấy cách trích đặc trưng và bố cục notebook KNN làm gốc. Giữ **5 biểu diễn × 7 cấu hình SVC**, tổng 35 cấu hình. Không augmentation.

## Đọc trước khi chạy

Mở [00_READ_RESULTS.ipynb](224/00_READ_RESULTS.ipynb) để xem ngay kết quả đã có. Cấu hình cuối: **HSV12 + HOG, SVC RBF, C=10, gamma=scale**. Test đúng **1.362/1.553**, Accuracy **87,70%**, Macro-F1 **0,881102**, MAE **32.891,18 đồng/ảnh**.

`results_224/` là bộ kết quả đã chạy, dùng trong slide. Chạy lại ghi vào `runs/SVM` trên máy cá nhân hoặc `/kaggle/working/CS231_SVM/results_224` trên Kaggle.

## Thứ tự học và chạy

| Bước | Công việc |
|---|---|
| 01 | Đọc train/dev, kiểm tra ảnh và minh họa RGB/HSV/HOG |
| 02 | HSV12: histogram kết hợp 12×12×12 = 1728 chiều, chuẩn hóa L2 |
| 03 | HOG xám: 6084 chiều, chuẩn hóa L2 |
| 04 | Fit PCA trên HOG raw của train, giữ 95% phương sai, sau đó L2 |
| 05A | Ghép HSV12 và HOG, mỗi nhánh đã L2 riêng |
| 05B | Ghép HSV12 và HOG-PCA, mỗi nhánh đã L2 riêng |
| 06 | So sánh dev, giữ bảng baseline linear C=1 và chọn cấu hình cuối |
| 07 | Fit SVC đã chọn trên train, lưu model rồi dự đoán test |
| 08 | Phân tích lỗi, xem điểm SVM và ảnh train để đối chiếu |

Lưới khảo sát: linear C={0.1, 1, 10}; RBF C={1, 10} × gamma={scale, 10}. Chọn Macro-F1 dev cao nhất, nếu bằng nhau thì MAE thấp nhất, sau đó thứ tự khai báo. Không dùng test để chọn lại.

PCA chỉ fit trên train; dev/test chỉ transform. Nối hai nhánh đã L2 riêng, không L2 lại vector sau nối. Trong scikit-learn, SVC học các cặp lớp; `decision_function_shape='ovr'` chỉ đổi dạng điểm trả về. Điểm này không phải xác suất.

## Code nằm ở đâu?

- [pipeline.py](pipeline.py): trích HSV/HOG, lưu/đọc feature và khảo sát bảy SVC.
- [224/](224/README.md): các bước chính viết trực tiếp trong notebook, gồm fit PCA, chọn cấu hình và fit SVC.
- `src/experiments/figures.py`: vẽ ảnh, histogram, ma trận và ảnh train đối chiếu.

Các ảnh train gần trong bước 08 giống cách xem ảnh của KNN, nhưng **không tham gia quyết định của SVM**. Chỉ nạp model/PCA do nhóm tự tạo.

Cài riêng: `pip install -r src/SVM/requirements.txt`. Xem [hướng dẫn Kaggle](../../docs/REBUILD_KAGGLE.md) nếu chạy trên Kaggle.
