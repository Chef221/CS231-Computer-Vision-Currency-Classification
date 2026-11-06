# CNN RGB224 — kiểm tra đánh giá test

Phương án: **C04_DEEP_AUG**, RGB224, bốn khối 32–64–128–256, GAP256 và classifier Dense128–Dropout–Dense6. Checkpoint chọn ở **epoch 24**, nạp từ `candidates/C04_DEEP_AUG/best.pt`; không huấn luyện lại và không gộp train+dev.

| Tập | Số ảnh | Đúng | Accuracy | Macro-F1 | MAE (đ/ảnh) |
|---|---:|---:|---:|---:|---:|
| Dev | 1.564 | 1.508 | 96,42% | 0,964559 | 9.584,40 |
| Test | 1.553 | 1.507 | 97,04% | 0,969963 | 9.104,96 |

Test có **46 ảnh sai** và tổng chênh lệch danh nghĩa **14.140.000đ**. MAE = 14.140.000/1.553 = 9.104,9581455…đ/ảnh, tính cả ảnh đúng góp 0. Test cao hơn dev 0,62 điểm phần trăm Accuracy và 0,005404 Macro-F1; không suy từ một split rằng khả năng tổng quát hóa luôn tốt hơn.

## Ma trận nhầm lẫn

Hàng là nhãn thật; cột là dự đoán. Đơn vị nhãn trong bảng là nghìn đồng.

| Thật \ dự đoán | 10k | 20k | 50k | 100k | 200k | 500k |
|---|---:|---:|---:|---:|---:|---:|
| 10k | 280 | 0 | 1 | 0 | 0 | 0 |
| 20k | 0 | 210 | 0 | 0 | 0 | 14 |
| 50k | 4 | 1 | 216 | 0 | 2 | 0 |
| 100k | 1 | 0 | 0 | 244 | 0 | 2 |
| 200k | 3 | 1 | 4 | 1 | 307 | 1 |
| 500k | 1 | 2 | 0 | 4 | 4 | 250 |

Lớp 100.000đ có F1 cao nhất (0,9839), lớp 500.000đ thấp nhất (0,9470). Recall 20.000đ là 0,9375: cả 14 ảnh bị bỏ sót của lớp này đều bị dự đoán thành 500.000đ.

## Đối chiếu đã thực hiện

- Tính lại Accuracy, Macro-F1, MAE, từng lớp và ma trận từ đủ 1.553 dòng dự đoán, khớp JSON.
- Đường dẫn/nhãn/thứ tự test khớp manifest; ba manifest train/dev/test khớp KNN và SVM.
- Khóa dùng trên test trùng byte với khóa ở bước 06; 41 hash khóa khớp bản local, gồm bốn `best.pt` và code.
- Checkpoint nạp đúng epoch 24 và hash `983063cc4c304487d5aca0625d08744ec557860806bc479b2dec82f70319b918`.
- `run_info.json` ghi `refit=false`, `test_used_for_tuning=false`, 1.553 ảnh và 83,19 giây suy luận kèm đọc ảnh trên `cuda:0`, Tesla T4.

Đây là kiểm tra file, metadata và số liệu đã lưu; không chạy lại suy luận GPU. Split test đã được xem trong khảo sát trước của dự án. Softmax chưa hiệu chỉnh, không dùng làm xác suất đúng được bảo đảm.

Nguồn: [metrics_test.json](metrics_test.json), [predictions_test.csv](predictions_test.csv), [run_info.json](run_info.json), [khóa đã dùng](locked_cnn_config_used.json), [bảng từng lớp](per_class_test.csv).
