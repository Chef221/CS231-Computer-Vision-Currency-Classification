# Nguồn ý tưởng cho thiết kế CNN RGB224

Tài liệu tham khảo do nhóm cung cấp là “Hệ thống Phân loại Trái cây Tại Quầy Thu Ngân”, bản PDF 57 trang. Tên file nguồn: `Đồ-án-CV-HỆ-THỐNG-PHÂN-LOẠI-TRÁI-CÂY-TẠI-QUẦY-THU-NGÂN-1.pdf`; SHA-256: `4dab1457535cfd4e4df2684c619a3a4354f519d590533b2de51444fa074ff99b`. PDF được dùng để tham khảo cách tổ chức phương pháp và trình bày, không phải dữ liệu hay kết quả của dự án tiền polymer. Repository không phân phối lại PDF.

## Phần ý tưởng được áp dụng

| Vị trí nguồn | Ý tưởng | Cách áp dụng trong dự án |
|---|---|---|
| Trang 11–12 | Resize, chia pixel cho 255, augmentation trên train | Dùng reader RGB224 chung, chia 255; augmentation có nhánh đối chứng |
| Trang 27–29 | CNN ba khối 32–64–128, BatchNorm, pooling, dropout và GAP | Tự xây CNN PyTorch, hai Conv mỗi khối, thêm nhánh bốn khối để khảo sát |
| Trang 29–30 | Loss phân loại, optimizer, checkpoint và dừng sớm | CrossEntropyLoss, AdamW; chọn và dừng theo Macro-F1/MAE dev của chuẩn nhóm |
| Trang 47–48 | Learning curve, ma trận nhầm lẫn và xem lỗi | Lưu curve, ma trận, cặp nhầm VND, ảnh đối chứng và Grad-CAM |

## Điều chỉnh theo bài toán tiền polymer

Sáu nhãn và split của dự án tiền thay cho năm lớp trái cây. Mạng được huấn luyện từ đầu, không lấy trọng số hoặc kết quả trong PDF. Không chuyển quy tắc lựa chọn theo Accuracy/val_loss của tài liệu thành quy tắc của nhóm; checkpoint và cấu hình phải theo Macro-F1, hòa thì MAE thấp hơn.

Thiết kế mặc định giữ RGB224 của KNN/SVM. Cấu hình ảnh xám khảo sát tác động của việc bỏ màu; không dùng HSV/HOG làm đầu vào CNN. Hai cấu hình augmentation không lật ảnh và không đổi hue/saturation, để tránh thêm một phép thay đổi hình chữ số hoặc màu mệnh giá chưa được khảo sát riêng. Biên affine điền xám có thể tạo dấu hiệu tổng hợp; tác dụng thực tế phải đọc trên dev, không được giả định trước là cải thiện.

Dropout là regularization trên kích hoạt, không phải phép augmentation ảnh. MaxPool giảm kích thước không gian; lớp pooling không có trọng số học được, nên không diễn đạt rằng bản thân pooling cắt số tham số Conv trước đó. BatchNorm còn có tham số học được và thống kê chạy; lúc eval dùng thống kê đã học trên train. Những giải thích trong báo cáo CNN bám đúng các module đã triển khai.

PCA hai chiều của vector GAP chỉ giúp xem biểu diễn đã học, không thay bộ phân loại. Grad-CAM được bổ sung từ nguồn chính thức Selvaraju và cộng sự (ICCV 2017), không gán cho PDF trái cây. Các nhận xét nhân quả, con số Accuracy/F1 và tỷ lệ lỗi của trái cây không được chuyển sang kết quả tiền polymer.

## Nguồn kỹ thuật

- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
- [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html), [ghi chú tái lập](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- [Grad-CAM, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
