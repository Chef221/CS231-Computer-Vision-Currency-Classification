# Đề cương thực nghiệm

Đây là kế hoạch cho KNN, SVM, CNN và ResNet. Tài liệu chưa chứa kết quả huấn luyện. Cách tính và chọn cấu hình được chốt tại [METRICS_STANDARD.md](METRICS_STANDARD.md); giải thích công thức nằm trong [M2.2](M2.2%20-%20Mô%20tả%20độ%20đo%20đánh%20giá.md).

## 1. Câu hỏi và thí nghiệm

| Mã | Câu hỏi | Cách thực hiện | Kết quả cần có |
|---|---|---|---|
| E1 — chính | Bốn phương pháp phân loại sáu mệnh giá tốt đến đâu? | Fit trên train, chọn cấu hình/checkpoint trên dev, đánh giá cấu hình đã khóa trên test | Accuracy, Macro-F1, MAE, bảng từng lớp, ma trận nhầm lẫn |
| E2 — chính | Những cặp nào bị nhầm và chênh lệch VND lớn đến đâu? | Phân tích dự đoán đã lưu của E1 | Bảng cặp nhầm, số lần, chênh lệch mỗi lần và tổng chênh lệch; ảnh lỗi minh họa |
| E3 — mở rộng | Từ chối một phần đầu vào có giảm câu trả lời sai không? | Sau baseline, thử ngưỡng trên dev của một mô hình đã chọn, khóa ngưỡng rồi đánh giá test | Coverage, tỷ lệ đúng/sai/từ chối trên toàn bộ đầu vào; Accuracy và MAE trên phần được trả lời |

E1 và E2 là phạm vi phải hoàn thành. E3 chưa được triển khai trong mã đánh giá hiện tại; chỉ làm khi đã có baseline và đã ghi rõ quy tắc chọn ngưỡng. Kết quả E3 được trình bày riêng, không thay bảng so sánh E1.

## 2. Những quy tắc giữ cố định

- Dùng nguyên DATA và các split đã chốt: 12494 train, 1564 dev, 1553 test; sáu lớp mệnh giá như chuẩn đánh giá.
- Mỗi ảnh được dự đoán đúng một nhãn VND trong E1. Không bỏ ảnh test khó hoặc chuyển chúng sang lớp khác.
- Mọi bước cần học tham số, như scaler, PCA hoặc mô hình phân loại, chỉ fit bằng train. Các biến đổi áp lên dev/test dùng tham số đã học.
- Trước khi khảo sát dev, mỗi thành viên ghi danh sách cấu hình và thứ tự xét. Chọn Macro-F1 dev cao nhất; hòa chính xác thì chọn MAE thấp hơn; vẫn hòa thì giữ cấu hình đứng trước.
- Với CNN/ResNet, áp dụng cùng quy tắc cho checkpoint. Ghi kiến trúc, pretrained weights nếu có, cách xử lý ảnh, augmentation chỉ dùng khi huấn luyện, seed và số epoch.
- Khóa cấu hình, pipeline và mapping trước khi chấm test. Không thay feature hoặc tham số sau khi đọc lỗi test; không tự refit train+dev.
- So sánh mô hình trên cùng toàn bộ test và cùng hàm đánh giá. Không bắt buộc bốn phương pháp dùng cùng loại đặc trưng hoặc cùng kích thước ảnh; ghi rõ khác biệt.
- Các thành viên tự viết phần mô hình. Mã tính metric chỉ dùng chung một bản để tránh sai lệch.

Các con số trên mô tả bộ dữ liệu hiện tại, không chứng minh phân bố ảnh bằng phân bố sử dụng tiền trong thực tế. Metadata có nhóm ảnh để chia split, nhưng chưa đủ để khẳng định độc lập theo mọi tờ tiền vật lý hoặc phiên chụp.

## 3. E1 — So sánh bốn phương pháp

Nhóm phân công một thành viên làm KNN, một thành viên làm SVM và CNN, một thành viên làm ResNet. Ghi tên người phụ trách trong hồ sơ bàn giao kết quả.

Mỗi phương pháp lưu hai phần: các lần thử trên dev và kết quả cuối của cấu hình được chọn. KNN tiếp tục theo các milestone học và viết code; đề cương này không triển khai thay notebook của thành viên.

### Bảng cấu hình dev

Mỗi dòng tương ứng một cấu hình hoặc checkpoint:

| Cột | Nội dung |
|---|---|
| candidate_id | Mã cấu hình, kèm thứ tự đã khai báo |
| model | KNN, SVM, CNN hoặc ResNet |
| features / preprocessing | Đặc trưng và cách xử lý ảnh |
| parameters | Tham số mô hình, seed và thông tin huấn luyện liên quan |
| macro_f1_dev | Số chưa làm tròn để chọn cấu hình |
| mae_vnd_dev | Tiêu chí phá hòa |
| accuracy_dev | Thông tin đối chiếu |

Bảng không chứa số giả định. Khi có lần chạy thật mới bổ sung kết quả. Nếu mở rộng danh sách cấu hình sau khi đã xem dev, ghi lại việc mở rộng; không dùng test để quyết định.

### Hồ sơ kết quả cuối

Mỗi mô hình lưu tối thiểu:

1. Cấu hình và pipeline đã chọn, mapping lớp, seed và các thư viện chính.
2. JSON metric do hàm dùng chung tạo ra.
3. CSV từng ảnh với relative_path, y_true_vnd, y_pred_vnd.
4. Bảng Precision/Recall/F1/support từng lớp và ma trận nhầm lẫn theo cùng thứ tự mệnh giá.
5. Thông tin phần cứng, thời gian chạy nếu có đo. Không so tốc độ như một kết luận chung khi phần cứng hoặc cách đo khác nhau.

Bảng tổng kết E1 có các cột: mô hình, đặc trưng/pipeline, Accuracy test, Macro-F1 test, MAE test (VND/ảnh). Các giá trị trình bày được làm tròn; file JSON giữ giá trị gốc. Không gộp các độ đo thành một điểm mới.

## 4. E2 — Phân tích lỗi mệnh giá

Dùng CSV nhãn và bảng error_pairs của cùng kết quả E1, không tạo tập đánh giá mới.

- Xem cả số lượng lỗi và chênh lệch VND. Hai chiều nhầm giữ thành hai dòng riêng.
- Tạo một bảng các cặp nhầm có số lần xuất hiện, chênh lệch mỗi lần và tổng chênh lệch. Có thể sắp theo số lần hoặc tổng chênh lệch, nhưng phải ghi rõ tiêu chí.
- Chọn ảnh minh họa từ các cặp thường gặp và các cặp chênh lệch lớn; ghi đường dẫn, nhãn thật, dự đoán. Không dùng vài ảnh đẹp để thay cho số liệu toàn test.
- Khi nói lỗi liên quan đến ánh sáng, che khuất, góc chụp hoặc đặc trưng, phân biệt quan sát ảnh với kết luận đã kiểm chứng. Chưa có nhãn điều kiện ảnh thì không tự công bố tỷ lệ lỗi theo từng điều kiện.
- Tổng chênh lệch trong bảng bằng total_gap_vnd; chia n_samples phải ra mae_vnd.
- Nếu Macro-F1 tốt hơn nhưng MAE xấu hơn, trình bày đánh đổi. Không diễn giải chênh lệch danh nghĩa thành số tiền tổn thất thực tế.

Đây là bước phân tích kết quả, không phải bước sửa nhãn hoặc sửa DATA.

## 5. E3 — Đề xuất cơ chế từ chối, chưa triển khai

### Ý tưởng và nguồn tham khảo

Cơ chế ngưỡng và việc báo cáo đồng thời kết quả sai/từ chối có cơ sở từ nghiên cứu nhận diện tiền [R2]. BankNote-Net cũng nêu mục tiêu Precision ở nhóm dự đoán confidence cao khi triển khai [R3]. Tuy nhiên, thuật toán và ngưỡng của họ không tự áp dụng nguyên cho KNN của nhóm.

Phép thử đầu tiên có thể dùng tỷ lệ phiếu của lớp thắng trong KNN bỏ phiếu đều. Đây là lựa chọn triển khai của nhóm, không phải thuật toán khoảng cách tới class centroid của [R2]. Nếu k=1, điểm tỷ lệ phiếu luôn bằng 1 nên cơ chế ngưỡng này không phân biệt được đầu vào. Điểm cao vẫn có thể đi kèm dự đoán sai.

### Quy trình dự kiến

1. Hoàn tất baseline, chọn một mô hình để nghiên cứu cơ chế từ chối.
2. Xác định rõ điểm dùng để quyết định trả lời và danh sách ngưỡng. Không áp cùng một ngưỡng số cho các mô hình có cách tính điểm khác nhau.
3. Trước khi chọn ngưỡng, ghi quy tắc đánh đổi, chẳng hạn mức coverage tối thiểu. Giá trị cụ thể chưa chốt ở đề cương này; sẽ quyết định cho E3 và ghi lại trước khi khảo sát ngưỡng trên dev.
4. Khảo sát trên dev, chọn ngưỡng bằng quy tắc đã ghi; sau đó khóa mô hình, điểm và ngưỡng.
5. Đánh giá test, lưu trạng thái answered/rejected riêng với nhãn dự đoán gốc; không tạo mệnh giá thứ bảy.
6. Kiểm tra các lỗi chênh lệch lớn còn được trả lời. Không mặc định tăng ngưỡng sẽ loại mọi lỗi.

### Kết quả phải báo cáo

Gọi N là tổng ảnh, A là số ảnh được trả lời, C là số trả lời đúng, W là số trả lời sai, R là số bị từ chối:

- C + W + R = N; A = C + W.
- Tỷ lệ trả lời đúng = C/N; trả lời sai = W/N; từ chối = R/N.
- Coverage = A/N.
- Accuracy trên phần được trả lời = C/A.
- MAE trên phần được trả lời = tổng chênh lệch của các ảnh được trả lời chia A.
- Xem thêm tỷ lệ từ chối theo lớp để tránh che việc một mệnh giá bị từ chối nhiều.

Nếu A=0, Accuracy và MAE trên phần được trả lời là không xác định; ghi N/A, không gán 0 và không gọi hàm đánh giá trên vector rỗng. Không tính ảnh bị từ chối là ảnh đúng hoặc có chênh lệch 0.

Biểu đồ coverage–error và bảng sai/từ chối phản ánh đánh đổi khi chọn ngưỡng [R4]. Đề xuất này không triển khai bảo đảm rủi ro thống kê của bài báo đó. Với ảnh tĩnh, chỉ kiểm chứng được cơ chế từ chối; muốn kết luận chụp lại giúp ích phải thu thập thí nghiệm chụp lại thực tế.

## 6. Nguồn tham khảo và phần nhóm kế thừa

| Nguồn | Nội dung có thể kế thừa | Giới hạn khi áp dụng |
|---|---|---|
| [R1] Hasanuzzaman và cộng sự, 2012 | Kiểm tra nhận diện sai trong điều kiện chụp khác nhau, ảnh có tiền/ảnh nền, ngưỡng học từ training | Bài của họ có nhận biết ảnh không có tiền; E1 của nhóm chỉ xét sáu lớp đã biết |
| [R2] Pham và cộng sự, 2016 | Dùng độ phân biệt lớp và ngưỡng, báo cáo riêng lỗi và từ chối; phân tích ảnh sai | Họ dùng PCA/class centroid và ảnh cảm biến dòng, không phải KNN trên crop RGB |
| [R3] Oviedo và cộng sự, 2022 | Đánh giá phân loại và mục tiêu Precision khi confidence cao trong triển khai | Không sao chép ngưỡng 0,99 hay coi đó là xác suất đúng được bảo đảm |
| [R4] Geifman và El-Yaniv, 2017 | Đánh giá sự đánh đổi giữa lỗi và coverage | Không nhận bảo đảm lý thuyết của bài khi chỉ thử ngưỡng thông thường |

Các tài liệu này cung cấp ý tưởng phương pháp và thiết kế thí nghiệm. MAE theo VND là phần nhóm áp dụng để phân tích chênh lệch mệnh giá; không gán công thức đó hoặc kết quả trên DATA của nhóm cho các bài báo. Bộ tiêu chí chính thức vẫn là chuẩn đã chốt.

Thông tin xuất bản, vị trí bằng chứng và đường dẫn nằm trong [REFERENCES.md](REFERENCES.md). File kết quả và hình thực nghiệm sẽ được bổ sung sau những lần chạy thật.
