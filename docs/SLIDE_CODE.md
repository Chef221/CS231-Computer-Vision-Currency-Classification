# Từ slide tới code

Theo thứ tự phần SVM/CNN trong bản slide cuối của nhóm. Tên khái niệm dùng giống KNN, ưu tiên đọc các ô notebook trước rồi mới xem hàm phụ.

| Nội dung trên slide | Notebook / file | Phần cần đọc |
|---|---|---|
| SVM: biểu diễn ảnh | [02](../src/SVM/224/02_FEATURE_HSV12.ipynb), [03](../src/SVM/224/03_FEATURE_HOG.ipynb) | Hàm `hsv12`, `hog_raw`, `l2` trong pipeline.py |
| SVM: HOG-PCA | [04](../src/SVM/224/04_FEATURE_HOG_PCA.ipynb) | `PCA(0.95, svd_solver='full')`, fit train, transform dev |
| SVM: ghép hai nhánh | [05A](../src/SVM/224/05A_FEATURE_HSV12_HOG.ipynb), [05B](../src/SVM/224/05B_FEATURE_HSV12_HOG_PCA.ipynb) | `np.concatenate`, không chuẩn hóa lại sau nối |
| SVM:7 cấu hình/biểu diễn | [pipeline.py](../src/SVM/pipeline.py) | `candidates` và `run_dev_grid`: tạo SVC, fit train, predict dev |
| SVM: chọn cấu hình | [06](../src/SVM/224/06_SELECT_FINAL_SVM.ipynb) | Tổng hợp35 cấu hình; F1 cao rồi MAE thấp; baseline linear C=1 |
| SVM: kết quả test | [07](../src/SVM/224/07_EVALUATE_TEST.ipynb) | Fit lại cấu hình đã chọn trên train; save model.joblib, predict test |
| SVM: ảnh lỗi | [08](../src/SVM/224/08_ERROR_ANALYSIS.ipynb) | Điểm SVM, bảng cặp nhầm và ảnh train để đối chiếu |
| CNN:4 cấu hình | [data.py](../src/CNN/data.py) | `CANDIDATES`, `TRAINING` |
| CNN: các lớp và số tham số | [02](../src/CNN/224/02_TRAIN_RGB.ipynb) đến [05](../src/CNN/224/05_TRAIN_DEEP_AUG.ipynb) | Ô `model = models.Sequential`, từng `model.add`, `model.summary()` |
| CNN: augmentation | [data.py](../src/CNN/data.py) | `augment_rgb`, gọi trong dataset train trước chia255 |
| CNN: AdamW, cosine | [model.py](../src/CNN/model.py) | `compile_model`, `cosine_learning_rate` |
| CNN: train/dev và checkpoint | [pipeline.py](../src/CNN/pipeline.py) | `train_candidate`, `DevCheckpoint`, `evaluate_dev` |
| CNN: chọn C04 epoch25 | [06](../src/CNN/224/06_SELECT_FINAL_CNN.ipynb) | `select_and_lock` đọc summary từng cấu hình |
| CNN: chấm checkpoint trên test | [07](../src/CNN/224/07_EVALUATE_TEST.ipynb) | Nạp best.keras, không train thêm |
| CNN: Grad-CAM | [08](../src/CNN/224/08_ERROR_ANALYSIS.ipynb), [analysis.py](../src/CNN/analysis.py) | `gradcam`, gradient của điểm trước softmax tại Conv cuối |
| Accuracy, Macro-F1, MAE | [value_metrics.py](../src/evaluation/value_metrics.py) | Nhãn đưa vào là mệnh giá bằng đồng, không phải chỉ số0…5 |

## Ba điểm dễ nhầm

1. C01 có GAP128 chiều; C04 có GAP256 chiều. Dense phía sau đều có128 neuron.
2. SVM bước07 fit trên train với tham số đã chọn; CNN bước07 nạp trọng số checkpoint. Cả hai đều không học từ test.
3. Số97.36% của CNN là test. Dev của cùng C04 epoch25 là96.36%; hai tập khác nhau nên số liệu khác nhau.

Bảng/hình có sẵn ở [kết quả tổng hợp](RESULTS.md). Xem thêm [giải thích CNN](../src/CNN/GIAI_THICH_CNN.md).
