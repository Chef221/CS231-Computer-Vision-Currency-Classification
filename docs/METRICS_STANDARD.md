# Chuẩn đánh giá dùng chung

KNN, SVM, CNN và ResNet dùng cùng hàm [evaluate_classification](../src/evaluation/value_metrics.py). Bộ tiêu chí chính thức gồm **Macro-F1, Accuracy, MAE theo VND, Precision/Recall/F1 từng lớp và ma trận nhầm lẫn**.

## 1. Chọn cấu hình và báo cáo kết quả

1. Fit mô hình và các bước học tham số, như scaler hoặc PCA, bằng **train**. Dev chỉ dùng để chọn cấu hình/checkpoint.
2. Chọn **Macro-F1 cao nhất trên dev**, so sánh số chưa làm tròn.
3. Nếu Macro-F1 bằng nhau chính xác, chọn **MAE theo VND thấp hơn**. Nếu vẫn bằng nhau, giữ cấu hình đứng trước trong danh sách đã khai báo trước khi xem dev.
4. Khóa cấu hình, pipeline và mapping lớp; sau đó đánh giá trên **test**. Không dùng test để chọn feature, siêu tham số, checkpoint hoặc thiết kế metric. Không tự refit train+dev trong quy trình này.
5. Báo cáo đầy đủ các tiêu chí. Mô hình có Macro-F1 cao hơn có thể có MAE cao hơn; khi đó phải trình bày đánh đổi, không kết luận nó tốt hơn về mọi mặt.

CNN và ResNet áp dụng cùng quy tắc cho việc chọn checkpoint. Accuracy và F1 được lưu dạng số trong [0,1]; MAE có đơn vị VND/ảnh. Làm tròn chỉ khi trình bày.

## 2. Đầu vào và ý nghĩa các độ đo

Nhãn phải là **mệnh giá VND**, theo thứ tự cố định:

```text
[10000, 20000, 50000, 100000, 200000, 500000]
```

Hàm giả định hai vector một chiều, cùng độ dài, có ít nhất một ảnh và khớp thứ tự từng ảnh. Nhãn là số nguyên hoặc chuỗi số nguyên thuộc sáu giá trị trên. Không truyền nhãn thiếu, số thực hay chỉ số lớp `0..5`. Code tập trung vào phép tính, không có bộ kiểm tra đầu vào riêng.

| Kết quả | Ý nghĩa | Chiều tốt hơn |
|---|---|---|
| `macro_f1` | Trung bình đều F1 của đủ sáu lớp | Cao hơn |
| `accuracy` | Số ảnh đúng chia tổng số ảnh | Cao hơn |
| `mae_vnd` | Tổng chênh lệch tuyệt đối chia tổng số ảnh, kể cả ảnh đúng góp 0 | Thấp hơn |
| `per_class` | Support, TP, FP, FN, Precision, Recall, F1 từng lớp | Đọc theo từng lớp |
| `confusion_matrix` | Bảng đếm 6×6; hàng là nhãn thật, cột là dự đoán | Xem các cặp nhầm |
| `error_pairs` | Các cặp nhầm có xuất hiện, số lần và chênh lệch VND | Phân tích lỗi |
| `total_gap_vnd` | Tổng chênh lệch tuyệt đối, bằng MAE nhân số ảnh | Dùng kiểm tra phép tính |

Luôn giữ đủ sáu lớp trong ma trận và Macro-F1. Precision/Recall/F1 có mẫu số 0 được quy ước bằng 0. Vì vậy, ví dụ chỉ có một lớp và đúng hết vẫn có Macro-F1 bằng 1/6; Accuracy bằng 1 và MAE bằng 0. Dev/test chính thức có đủ sáu lớp.

**MAE là sai lệch danh nghĩa, không phải thiệt hại giao dịch.** Công thức và ví dụ chi tiết nằm trong [M2.2](M2.2%20-%20Mô%20tả%20độ%20đo%20đánh%20giá.md).

## 3. Gọi hàm

Chạy Python/notebook với thư mục gốc repository là working directory. Cài thư viện bằng `pip install -r requirements.txt`.

### KNN và SVM

Nếu mô hình đã dùng nhãn VND:

```python
from src.evaluation.value_metrics import evaluate_classification

y_pred_vnd = model.predict(X_dev)
metrics = evaluate_classification(y_dev_vnd, y_pred_vnd)

print("Macro-F1:", metrics["macro_f1"])
print("Accuracy:", metrics["accuracy"])
print("MAE (VND/ảnh):", metrics["mae_vnd"])
```

Nếu working directory đang là `src/KNN`, có thể thêm root trước khi import:

```python
from pathlib import Path
import sys

project_root = Path.cwd().parents[1]
sys.path.insert(0, str(project_root))
```

Đoạn thêm đường dẫn này chỉ dành cho working directory `src/KNN`; không dùng nguyên xi khi đang ở thư mục khác.

### CNN và ResNet

Nếu mô hình dùng chỉ số lớp, đổi **cả nhãn thật và dự đoán** về VND bằng mapping thật. Ví dụ sau giả định đã có `all_targets` và `all_logits` của **toàn bộ dev hoặc test**, cùng thứ tự:

```python
from src.evaluation.value_metrics import evaluate_classification

idx_to_vnd = {
    index: int(class_name)
    for class_name, index in class_to_idx.items()
}
true_indices = all_targets.detach().cpu().tolist()
pred_indices = all_logits.detach().argmax(dim=1).cpu().tolist()

y_true_vnd = [idx_to_vnd[index] for index in true_indices]
y_pred_vnd = [idx_to_vnd[index] for index in pred_indices]
metrics = evaluate_classification(y_true_vnd, y_pred_vnd)
```

`class_to_idx` phải lấy từ bộ nạp dữ liệu của mô hình. Thứ tự tên thư mục theo chuỗi có thể khác thứ tự mệnh giá theo số; không tự gán lớp 0 là 10000, lớp 1 là 20000.

Ghép các batch rồi gọi hàm một lần trên toàn split. Không lấy trung bình Macro-F1 của từng batch.

## 4. Xem và lưu kết quả

```python
import json
from pathlib import Path
import pandas as pd

print(pd.DataFrame(metrics["per_class"]))
print(pd.DataFrame(metrics["confusion_matrix"],
                   index=metrics["class_order_vnd"],
                   columns=metrics["class_order_vnd"]))
print(pd.DataFrame(metrics["error_pairs"]))

output = Path("results/knn")
output.mkdir(parents=True, exist_ok=True)
(output / "metrics_dev.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

Đổi tên thư mục và split tương ứng với lần chạy. Cùng với JSON, lưu cấu hình và CSV dự đoán từng ảnh:

| Cột CSV | Nội dung |
|---|---|
| `relative_path` | Đường dẫn ảnh tương đối theo thư mục DATA |
| `y_true_vnd` | Mệnh giá thật |
| `y_pred_vnd` | Mệnh giá dự đoán |

Các dòng phải khớp đúng ảnh và thứ tự. `error_pairs` rỗng nếu không có lỗi; mỗi dòng có `true_vnd`, `predicted_vnd`, `count`, `gap_vnd`, `total_gap_vnd`. Có thể tìm cặp xuất hiện nhiều hoặc đóng góp tổng chênh lệch lớn từ bảng này.

## 5. Ví dụ kiểm tra bằng tay

Đây là bảy dự đoán **giả lập**, không phải kết quả thực nghiệm:

```python
from src.evaluation.value_metrics import evaluate_classification

y_true = [10000, 10000, 20000, 50000, 100000, 200000, 500000]
y_pred = [10000, 20000, 20000, 50000, 100000, 200000, 200000]
metrics = evaluate_classification(y_true, y_pred)
```

Kết quả cần nhận được:

- Accuracy = 5/7 ≈ 0,714286.
- Macro-F1 = 2/3 ≈ 0,666667.
- Tổng chênh lệch = 10000 + 300000 = 310000 VND.
- MAE = 310000/7 ≈ 44285,71 VND/ảnh.

Kiểm tra phép tính bằng:

```powershell
py -3 -m unittest discover -s tests -p test_value_metrics.py -v
```

## 6. Chuyển từ cách chấm cũ

Giữ nguyên import `evaluate_classification`. Code cũ đọc `metrics["macro_value_f1"]` phải chuyển sang `metrics["macro_f1"]` khi chọn cấu hình; dùng thêm `metrics["mae_vnd"]` để phân tích chênh lệch.

Các trường `macro_value_precision`, `macro_value_recall`, `macro_value_f1`, `weighted_f1` và các trường value theo lớp đã được bỏ. Không giữ alias cho công thức cũ vì sẽ làm sai ý nghĩa kết quả. RMSE và F1 theo chi phí không nằm trong bộ tiêu chí chính thức.

Kết quả cũ và tài liệu cũ dùng MacroValueF1 không đại diện cho chuẩn hiện tại. Nếu còn CSV nhãn thật/dự đoán, có thể chấm lại bằng hàm mới mà không cần huấn luyện lại.

## 7. Thí nghiệm mở rộng

Cơ chế từ chối là phần đề xuất trong [đề cương thực nghiệm](EXPERIMENT_PLAN.md), chưa được triển khai trong hàm đánh giá. So sánh chính giữa bốn mô hình luôn dùng toàn bộ test. Nếu thực hiện từ chối, kết quả trên phần được trả lời phải trình bày riêng, kèm coverage; không gán ảnh bị từ chối thành sai lệch 0.
