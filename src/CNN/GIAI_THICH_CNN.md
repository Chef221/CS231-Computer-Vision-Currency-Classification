# Các phần chính trong CNN

- **Conv2D:** học các bộ lọc dùng chung trên ảnh. Padding same giữ chiều cao/rộng khi stride1.
- **BatchNorm:** điều chỉnh giá trị từng kênh để hỗ trợ huấn luyện. Mỗi kênh có gamma/beta được học và hai thống kê dùng lúc dự đoán.
- **ReLU:** giữ số dương, đổi số âm thành0.
- **MaxPool:** lấy giá trị lớn nhất trong mỗi vùng2×2; làm chiều cao/rộng giảm một nửa.
- **SpatialDropout:** khi train, bỏ ngẫu nhiên một số bản đồ đặc trưng; lúc dự đoán thì tắt.
- **GAP:** lấy trung bình mỗi bản đồ, tạo một số cho mỗi kênh. Không có trọng số.
- **Dense128:** kết hợp vector GAP thành128 giá trị mới, rồi qua ReLU.
- **Dropout0.5:** khi train bỏ ngẫu nhiên một phần giá trị trong vector; số chiều không đổi.
- **Dense6 + softmax:** tạo sáu xác suất, chọn lớp có giá trị cao nhất.

## Ví dụ C01

```text
224×224×3
→ Khối32  → 112×112×32
→ Khối64  →  56×56×64
→ Khối128 →  28×28×128
→ GAP:128 → Dense128 → Dropout → Dense6 + softmax
```

C04 thêm khối256: sau pooling là14×14×256, nên GAP có256 chiều, sau đó Dense vẫn có128 chiều.

## Cách đếm tham số

Conv không bias: `3 × 3 × kênh_vào × kênh_ra`. Dense có bias: `số_đầu_vào × số_neuron + số_neuron`. BatchNorm có tổng `4 × số_kênh`, trong đó chỉ một nửa là tham số được học qua optimizer.

| Cấu hình | Trainable | Thống kê BatchNorm | Tổng |
|---|---:|---:|---:|
| C01/C03 | 304742 | 896 | 305638 |
| C02 | 304166 | 896 | 305062 |
| C04 | 1206886 | 1920 | 1208806 |

Optimizer còn lưu các giá trị để cập nhật trọng số; chúng không tính vào số tham số kiến trúc trên slide.

## Vì sao có lớp BatchNorm riêng?

Đây là phần được giữ từ baseline đã chạy. Mean/variance dùng để chuẩn hóa batch và variance dùng để cập nhật thống kê có mẫu số khác nhau. `TorchBatchNormalization` làm đúng cách cập nhật đã dùng trong lượt kết quả mới. Chỉ cần hiểu vai trò của BatchNorm trước; không cần sửa lớp này để chạy notebook. Thay bằng mặc định Keras có thể làm kết quả huấn luyện thay đổi.

## Khi xem learning curves và Grad-CAM

Loss/Accuracy train có dropout/augmentation; dev thì không. Dev dùng chọn checkpoint, test dùng chấm checkpoint đã chọn, nên số liệu hai tập có thể khác nhau. Grad-CAM chỉ minh họa vùng đáp ứng theo lớp, chưa chứng minh nguyên nhân một ảnh bị nhận sai.
