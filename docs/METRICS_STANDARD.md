# Chuẩn đánh giá phân loại theo mệnh giá

Tài liệu này chốt **cách chấm và cách dùng thống nhất** cho KNN, SVM, CNN và ResNet. Đây là chuẩn đánh giá nội bộ cho bài toán sáu lớp; nó không thay đổi thuật toán huấn luyện, loss hay dữ liệu. Mã nguồn duy nhất là [`src/evaluation/value_metrics.py`](../src/evaluation/value_metrics.py), triển khai bằng một hàm `evaluate_classification`.

## 1. Nhãn và thứ tự lớp

Metric nhận nhãn là **giá trị mệnh giá bằng VND**, theo đúng thứ tự cố định:

```text
[10000, 20000, 50000, 100000, 200000, 500000]
```

**Tiền điều kiện:** truyền hai vector một chiều, cùng độ dài và có ít nhất một phần tử; mọi nhãn phải là một trong sáu giá trị VND ở trên. Hàm nhận nhãn số hoặc chuỗi số thập phân; không truyền chỉ số lớp `0..5`, số thực, giá trị thiếu hay nhãn khác. Nếu mô hình trả chỉ số lớp, ánh xạ cả nhãn thật và nhãn dự đoán về mệnh giá VND bằng đúng mapping của mô hình trước khi chấm. Không suy ra mapping từ thứ tự từ điển của tên thư mục: ví dụ thứ tự chuỗi có thể là `10000, 100000, 20000, ...`.

Ma trận nhầm lẫn luôn có 6 hàng và 6 cột theo thứ tự trên, kể cả lớp không xuất hiện trong mẫu đang chấm. Hàng là nhãn thật, cột là nhãn dự đoán.

## 2. Độ đo được báo cáo

### Metric chính: MacroValueF1

MacroValueF1 là trung bình cộng sáu F1 theo giá trị mệnh giá. Với mỗi lớp $c$, đặt $v_c$ là mệnh giá VND và $M_{ab}$ là số mẫu thật lớp $a$, dự đoán lớp $b$:

$$
T_c=M_{cc}v_c,\qquad
E^{in}_c=\sum_{a\ne c}M_{ac}|v_a-v_c|,\qquad
E^{out}_c=\sum_{b\ne c}M_{cb}|v_c-v_b|.
$$

$$
P^{value}_c=\frac{T_c}{T_c+E^{in}_c},\quad
R^{value}_c=\frac{T_c}{T_c+E^{out}_c},\quad
F^{value}_c=\frac{2T_c}{2T_c+E^{in}_c+E^{out}_c},\quad
\mathrm{MacroValueF1}=\frac{1}{6}\sum_{c=1}^{6}F^{value}_c.
$$

Nếu mẫu số bằng 0, metric tương ứng bằng 0. Luôn lấy trung bình đủ sáu lớp; lớp không có mẫu thật và không có dự đoán góp 0. `MacroValueF1` nằm trong `[0,1]`, càng cao càng tốt. Các giá trị trung gian $T$ và $E$ có đơn vị VND; tỉ số và điểm tổng hợp không có đơn vị.

Đây là **metric custom của dự án**, không phải MAE, expected financial loss, xác suất đúng, hay công thức được gán cho Elkan/Delgado. Nó cân nhắc khoảng chênh lệch VND theo từng lớp nhưng vẫn có giới hạn của một điểm tổng hợp; luôn xem thêm confusion matrix và các metric chẩn đoán.

### Metric phụ và chẩn đoán

- **Macro-F1 truyền thống** là metric phụ dùng để phân tích và phá hòa cấu hình; trung bình F1 sáu lớp, lớp vắng mặt nhận 0.
- **Accuracy**, **Weighted-F1** và precision/recall/F1 từng lớp được báo cáo để đối chiếu. Weighted-F1 dùng support của nhãn thật, không dùng giá trị tiền.
- **Confusion matrix** là bảng đếm 6×6 cố định để truy cặp nhầm. Các metric từng lớp của cả hai bộ được trả trong kết quả hàm.
- Không báo cáo `MAE_VND` làm metric mặc định/tiêu chí chọn cấu hình trong chuẩn này. Không suy diễn điểm thành thiệt hại giao dịch hoặc khẳng định mô hình “an toàn tài chính”.

## 3. Chọn cấu hình và dùng tập dữ liệu

1. Giữ nguyên các split đã chốt. Train dùng để fit; dev dùng để chọn cấu hình và checkpoint; test chỉ dùng một lần sau khi đóng băng lựa chọn.
2. Trong từng phương pháp, chọn cấu hình theo **MacroValueF1 trên dev, so sánh giá trị chưa làm tròn**. CNN và ResNet cũng dùng tiêu chí này để chọn checkpoint.
3. Nếu MacroValueF1 bằng nhau chính xác, so Macro-F1 truyền thống chưa làm tròn. Nếu vẫn hòa, chọn candidate xuất hiện trước trong thứ tự đã khai báo trước khi xem dev. Ghi lại thứ tự đó. Không tạo điểm ghép tùy ý.
4. Sau khi khóa cấu hình, báo cáo cùng bộ metric và ma trận nhầm lẫn trên test cho KNN, SVM, CNN và ResNet. Không điều chỉnh cấu hình bằng test và không tự refit train+dev sau khi đã chốt.
5. Nếu các metric cho thấy đánh đổi, trình bày chúng song song; không khẳng định một model “thắng” nếu chỉ một metric cải thiện.

Metric được tính từ nhãn thật và nhãn dự đoán cứng. Xác suất, calibration, cost-sensitive loss, thresholding và quyết định hậu xử lý không thuộc tài liệu này.

Tính metric **một lần trên toàn bộ dự đoán của split**: ghép nhãn thật và dự đoán của mọi batch trong dev hoặc test rồi gọi hàm. Không lấy trung bình MacroValueF1/F1 của từng batch; macro F1 không cộng gộp tuyến tính theo batch.

## 4. Dùng chung hàm Python

### 4.1. KNN và SVM

KNN/SVM có thể truyền trực tiếp nhãn thật và dự đoán nếu chúng đã là giá trị VND:

```python
from src.evaluation.value_metrics import evaluate_classification

metrics = evaluate_classification(y_true_vnd, y_pred_vnd)
```

### 4.2. CNN và ResNet

Ví dụ này giả định `target_batches` và `logit_batches` đã được tích lũy theo cùng thứ tự cho **toàn bộ split**. `class_to_idx` là mapping thật của `ImageFolder`; chuyển cả target lẫn argmax về VND trước khi chấm.

```python
import torch

from src.evaluation.value_metrics import evaluate_classification

all_targets = torch.cat(target_batches, dim=0)
all_logits = torch.cat(logit_batches, dim=0)
idx_to_vnd = {
    index: int(class_name)
    for class_name, index in class_to_idx.items()
}
y_true_vnd = [idx_to_vnd[int(index)] for index in all_targets.cpu()]
y_pred_indices = all_logits.detach().argmax(dim=1).cpu().numpy()
y_pred_vnd = [idx_to_vnd[int(index)] for index in y_pred_indices]
metrics = evaluate_classification(y_true_vnd, y_pred_vnd)
```

### 4.3. Lưu kết quả chung

Sau khi `y_true_vnd`, `y_pred_vnd`, `metrics`, đường dẫn ảnh và cấu hình đã được tạo cho một split, lưu một dòng prediction cho mỗi ảnh cùng JSON metric/config để nhóm có thể kiểm tra và tính lại:

```python
import json
import pandas as pd

pd.DataFrame({
    "final_path_relative": test_final_paths_relative,
    "y_true_vnd": y_true_vnd,
    "y_pred_vnd": y_pred_vnd,
}).to_csv("predictions_test.csv", index=False)
run_record = {
    "split": "test",
    "n_samples": metrics["n_samples"],
    "class_order_vnd": metrics["class_order_vnd"],
    "configuration": selected_configuration,
    "metrics": metrics,
}
with open("metrics_test.json", "w", encoding="utf-8") as f:
    json.dump(run_record, f, ensure_ascii=False, allow_nan=False, indent=2)
```

`test_final_paths_relative` và `selected_configuration` là placeholder cần thay bằng giá trị của lần chạy. `class_order_vnd` là thứ tự nhãn chuẩn của confusion matrix; nếu mô hình dùng encoded class indices, lưu mapping index→VND thực tế riêng trong JSON để provenance rõ ràng. Khi nhãn đã là VND, không cần tạo `idx_to_vnd`.

Trong hàm, phần công thức nằm ở vòng lặp sáu lớp; các đoạn còn lại kiểm tra nhãn và lập kết quả báo cáo. Người dùng chỉ gọi hàm, không cần viết lại mã chấm.

Đầu ra là dictionary chỉ chứa kiểu JSON-serializable, có `class_order_vnd`, số mẫu, confusion matrix, metric tổng hợp và sáu mục `per_class`. Lưu giá trị đầy đủ để xếp hạng; làm tròn sáu chữ số chỉ để hiển thị. Với CNN/ResNet cần kiểm tra mapping class-index→mệnh giá trước khi gọi hàm.

Nên khởi động notebook từ thư mục gốc project để import trực tiếp. Nếu kernel đang ở `src/KNN`, có thể thêm project root như sau:

```python
from pathlib import Path
import sys

project_root = Path.cwd().parents[1]
sys.path.insert(0, str(project_root))
```

Sau đó import từ `src.evaluation.value_metrics`; không cần cài package riêng.

Chạy kiểm tra từ project root bằng `python -m unittest discover -s tests -p test_value_metrics.py -v`. Khi nộp kết quả một lần chạy, lưu cấu hình, split, số mẫu, thứ tự/mapping lớp, `metrics_*.json` và `predictions_*.csv` gồm `final_path_relative`, `y_true_vnd`, `y_pred_vnd`. Không cần lưu xác suất.

## 5. Ví dụ số nhỏ

Giả sử nhãn thật `[10000, 10000, 20000]` và dự đoán `[10000, 20000, 20000]`. Với lớp 10.000: $T=10{,}000$, $E^{in}=0$, $E^{out}=10{,}000$, nên $P^{value}=1$, $R^{value}=0.5$, $F^{value}=2/3$. Với lớp 20.000: $T=20{,}000$, $E^{in}=10{,}000$, $E^{out}=0$, nên $P^{value}=2/3$, $R^{value}=1$, $F^{value}=0.8$. Bốn lớp còn lại không có true positive hay error mass nên mỗi lớp góp 0; MacroValueF1 $=(2/3+0.8)/6\approx0.244444$.

**Giới hạn:** MacroValueF1 là quy ước dự án, không phải kết quả thực nghiệm hay chuẩn phổ quát. Nó không mô hình hóa quy trình thanh toán, chiều dòng tiền hoặc hậu quả vận hành. Điểm số không thay thế việc đọc từng cặp nhầm trong confusion matrix.
