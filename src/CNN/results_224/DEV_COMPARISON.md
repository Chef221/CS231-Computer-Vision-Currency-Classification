# CNN RGB224 — so sánh dev và khóa phương án

Bốn cấu hình cùng dùng 12.494 ảnh train và 1.564 ảnh dev; seed 42, AdamW lr=0,001, batch 64, cosine schedule, tối đa 25 epoch và patience 6. Cả bốn lượt chạy đủ 25 epoch. Không đọc ảnh test để chọn checkpoint hoặc cấu hình.

| Cấu hình | Đầu vào / kiến trúc | Epoch chọn | Đúng / 1.564 | Accuracy | Macro-F1 | MAE (đ/ảnh) |
|---|---|---:|---:|---:|---:|---:|
| C01_RGB | RGB, 3 khối, không augmentation | 25 | 1.463 | 93,54% | 0,935554 | 17.794,12 |
| C02_GRAY | Xám, 3 khối, không augmentation | 25 | 625 | 39,96% | 0,352394 | 141.585,68 |
| C03_RGB_AUG | RGB, 3 khối, augmentation train | 24 | 1.463 | 93,54% | 0,935862 | 18.478,26 |
| **C04_DEEP_AUG** | **RGB, 4 khối, augmentation train** | **24** | **1.508** | **96,42%** | **0,964559** | **9.584,40** |

## Nhận xét

C02 mất thông tin màu và còn đạt train Accuracy online 36,71% tại epoch 25. Kết quả thấp ở cả train/dev cho thấy cấu hình/ngân sách hiện tại chưa học tốt với đầu vào xám; chưa đủ để kết luận mọi CNN xám đều yếu. Conv đầu, khởi tạo và quá trình tối ưu cũng thay đổi khi dùng một kênh. Checkpoint tốt nhất ở cuối lượt chạy chưa chứng minh đã hội tụ.

C03 có cùng số ảnh đúng với C01. Macro-F1 tăng **0,000308** nhưng MAE tăng **684,14đ/ảnh**; không gọi augmentation là cải thiện trên mọi tiêu chí. Quy tắc chọn dùng F1 chưa làm tròn, MAE chỉ phá hòa chính xác.

C04 đúng thêm **45 ảnh** so với C03, Macro-F1 tăng **0,028697**, MAE giảm **8.893,86đ/ảnh**. Cấu hình này tăng đồng thời độ sâu, tham số và số chiều GAP (128 → 256), nên chưa cô lập riêng tác dụng độ sâu.

## Khóa cấu hình

Chọn checkpoint trong từng run bằng Macro-F1 cao nhất, hòa thì MAE thấp hơn, vẫn hòa giữ epoch sớm hơn. So bốn cấu hình theo cùng cặp metric rồi thứ tự C01…C04.

Phương án khóa: **C04_DEEP_AUG, epoch 24**, 1.206.886 tham số, GAP 256 chiều. Epoch cuối 25 có Macro-F1 0,963115 và MAE 10.326,09đ/ảnh, thấp hơn checkpoint được chọn về F1 và cao hơn về MAE. Notebook test dùng `best.pt` epoch 24, không dùng `last.pt` hoặc refit train+dev.

- [Bảng gốc bốn cấu hình](selection/best_by_candidate_dev.csv).
- [Đủ 100 dòng lịch sử epoch](selection/all_epochs_dev.csv).
- [Cấu hình khóa và 41 hash](selection/locked_cnn_config.json).

Thời gian train+dev từ log lần lượt là 42,47; 38,98; 43,89; 43,39 phút trên Tesla T4. Tổng trước làm tròn 168,72 phút, chưa tính chuẩn bị, vẽ hình, test hoặc phân tích.
