# Báo cáo CNN — lượt thực nghiệm RGB224

Đọc [BAO_CAO_THUC_NGHIEM_CNN.md](BAO_CAO_THUC_NGHIEM_CNN.md). Bố cục đi theo báo cáo KNN/SVM: thiết lập, tiền xử lý, kiến trúc/cấu hình, nhận xét từng biến thể, chọn dev, test, lỗi và kết luận; bổ sung so sánh ghép cặp KNN/SVM. Báo cáo có 11 bảng đánh số, chín hình (một SVG kiến trúc và tám PNG) và phụ lục lịch sử đủ 100 epoch từ bốn cấu hình.

Phương án khóa là **C04_DEEP_AUG, epoch 24**. Kết quả test: **1.507/1.553 ảnh đúng**, Accuracy **97,04%**, Macro-F1 **0,969963**, MAE **9.104,96đ/ảnh**. Số liệu lấy từ output notebook 08; tính lại từ CSV và đối chiếu JSON, checkpoint, khóa và các bảng tham chiếu trước khi viết báo cáo.

## Hình được đặt ở đâu?

| Vị trí trong nội dung | Hình | Vai trò |
|---|---|---|
| Kiến trúc | `CNN_ARCHITECTURE.svg` | Sơ đồ thiết kế cố định, không phụ thuộc huấn luyện |
| Tiền xử lý | CNN-1 | Ảnh train nguồn và RGB224 |
| Augmentation | CNN-2 | Ví dụ biến đổi chỉ áp dụng lên train |
| Thí nghiệm | CNN-3 | Learning curve của cấu hình được chọn bằng dev |
| Biểu diễn | CNN-4 | Vector GAP dev chiếu lên PCA fit train |
| Feature map | CNN-5 | Sáu kênh đáp ứng cao ở khối đầu và cuối |
| Đánh giá test | CNN-6 | Ma trận nhầm lẫn 6×6 |
| Phân tích lỗi | CNN-7 | C05–C08: chênh lệch lớn và ảnh sai có softmax cao |
| Phân tích điểm dự đoán | CNN-8 | C09–C12: hai điểm sát nhau, ảnh đúng và sai |

Tám PNG là bản sao ảnh notebook, không vẽ số liệu giả. [report_manifest.json](report_manifest.json) ghi nguồn và SHA-256; các hình CNN-7/8 lấy từ trang ví dụ 02/03 của notebook 08. Grad-CAM và PCA là trực quan mô tả, không thay metric hoặc chứng minh nguyên nhân lỗi. [Bảng bốn cấu hình](BANG_CAU_HINH_DEV.csv) và [lịch sử epoch đầy đủ](PHU_LUC_LICH_SU_EPOCH_CNN.csv) giữ số chưa làm tròn.

## Tái tạo và kiểm tra

Chạy từ repository root:

```bash
python docs/CNN_REPORT/build_report_224.py
python docs/CNN_REPORT/review/verify_results_224.py
```

Có thể chỉ rõ output tải từ Kaggle:

```bash
python docs/CNN_REPORT/build_report_224.py --results /duong/dan/results_224 --output /duong/dan/bao_cao
```

Script dùng thư viện chuẩn Python, không fit CNN/PCA. Khi có test, script kiểm tra hash, khóa đã dùng, tính lại Accuracy/Macro-F1/MAE/ma trận từ dự đoán, đối chiếu manifest KNN/SVM và sinh bảng. `report_manifest.json` ghi nguồn hình, hash và trạng thái kiểm tra. Nếu checkpoint bị bỏ khỏi bản Git, báo cáo ghi rõ số checkpoint không có; không tự nhận đã kiểm tra được file vắng mặt.

`REPORT_TEMPLATE.md` là nguồn cấu trúc; sửa phần phương pháp ở template rồi dựng lại Markdown. Các nhận xét dev, test, cặp lỗi và so sánh được tính từ file gốc để giữ đồng bộ khi tái tạo. [REFERENCE_NOTES.md](REFERENCE_NOTES.md) ghi ý tưởng kế thừa từ tài liệu tham khảo. Nhãn Bảng CNN có thể đánh số lại khi ghép báo cáo chung; giữ `CNN224_FIGURES/` cạnh Markdown.

Hồ sơ nội bộ nằm trong `review/`: [kết quả đối chiếu](review/validation.json), [nguồn nhập output](review/import_record.json) và [ghi chú xem ảnh](review/VISUAL_REVIEW.md). Không đưa hồ sơ nội bộ vào thân báo cáo nộp. CSV ghi chú gốc của notebook được giữ nguyên; các quan sát bổ sung tách rõ điều nhìn thấy và điều chưa kiểm chứng. Phần báo cáo công khai giới hạn một seed/split, lịch sử test đã được xem và khác biệt pipeline/thư viện giữa ba mô hình.

[Mục lục báo cáo](../README.md) dẫn tới cả ba mô hình. [Hồ sơ rà soát SVM/CNN theo cấu trúc KNN](../MODEL_FORMAT_REVIEW.md) ghi phạm vi kiểm tra code, notebook, dữ liệu, khóa và các khác biệt phương pháp cần giữ.
