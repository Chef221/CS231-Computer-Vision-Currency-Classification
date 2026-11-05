# Lộ trình chính thức nhánh KNN: RGB224, HSV12, HOG và PCA

Chốt ngày 2026-11-05 theo yêu cầu hoàn thiện lộ trình của người dùng. Tài liệu này là lộ trình hiện tại; bản nháp cũ đã được dọn khỏi checkout. Bước 01–08 đã chạy và được kiểm tra. Báo cáo ở bước 09 đã viết tại [báo cáo KNN](KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md); việc kiểm tra/bàn giao code SVM vẫn chưa hoàn tất. Không coi cấu hình đã chọn là tối ưu tuyệt đối.

## Mục tiêu và cách làm việc

Người dùng code KNN; Codex hướng dẫn, giải thích và kiểm tra từng milestone khi được yêu cầu. SVM/CNN/ResNet do thành viên khác thực hiện. Gói đầu vào chung được bàn giao để nhóm áp dụng; không coi ba phương pháp kia đã đồng bộ khi chưa kiểm tra code của họ.

Hoàn thành ba họ biểu diễn: HSV12, HOG, HSV12 + HOG. PCA là khảo sát giảm chiều trong họ HOG, không mở thêm họ feature khác. Kết quả thực tế quyết định PCA/fusion có ích hay không. Không đặt điều kiện phải đạt một ngưỡng Accuracy tùy ý hoặc mô hình sau phải tốt hơn mô hình trước.

## Điều kiện cố định

- Giữ DATA/split: 12.494 train, 1.564 dev, 1.553 test; sáu nhãn VND.
- Không augmentation trong thí nghiệm chính; không đổi nhãn hoặc lọc thêm ảnh.
- Đọc ảnh bằng OpenCV; BGR → RGB; resize trực tiếp 224×224 bằng INTER_LINEAR; giữ uint8 trước trích feature. Không crop, thêm viền hoặc xử lý tăng cường ảnh khác.
- Đọc từng ảnh, ghi feature float32 rồi bỏ ảnh khỏi bộ nhớ. Lưu feature và manifest để dùng lại; không giữ toàn bộ ảnh gốc hoặc trích lại cho mỗi bộ tham số.
- Mọi phép học tham số chỉ fit train. Dev chọn cấu hình; không tự fit lại train+dev.
- Chọn theo Macro-F1 dev chưa làm tròn; hòa chính xác thì MAE_VND thấp hơn; vẫn hòa thì ứng viên đứng trước theo thứ tự bên dưới.
- Dùng hàm metric chung; lưu cấu hình, dự đoán từng ảnh và đầy đủ Accuracy/Macro-F1/MAE/bảng lớp/confusion matrix.
- Kết quả cũ được lưu như lịch sử. Kết quả trên ảnh gốc/HOG128 không đại diện cho RGB224.
- Test hiện tại đã được đánh giá và phân tích trong khảo sát cũ. Lần chạy mới phải công khai lịch sử đó, không mô tả test là hoàn toàn chưa xem.

## Khóa biểu diễn và PCA

| Thành phần | Cấu hình |
|---|---|
| HSV12 | Histogram kết hợp H,S,V, 12 bin mỗi trục, miền OpenCV H [0,180), S/V [0,256); L2 toàn vector; 1.728 chiều |
| HOG trước giảm chiều | Xám từ RGB224; 9 hướng; cell16×16; block2×2; L2-Hys; transform_sqrt=False; 6.084 chiều |
| HOG không PCA | Chuẩn hóa L2 toàn vector HOG 6.084 chiều |
| HOG có PCA | Fit PCA trên HOG train trước chuẩn hóa L2 toàn vector; n_components=0.95, svd_solver=full, whiten=False; transform bằng PCA đó rồi chuẩn hóa L2 từng vector |
| Kết hợp | Nối HSV12 đã L2 với HOG đã L2, hoặc với HOG-PCA đã L2. Giữ nguyên thứ tự khối, không thêm trọng số hay chuẩn hóa sau nối |

PCA giữ 95% phương sai là quy tắc thiết kế được khai báo trước, không phải bảo đảm giữ 95% khả năng phân loại. Số chiều sau PCA được tính từ train, ký hiệu d; không chọn trước d=128/256 và không thử nhiều ngưỡng phương sai trong vòng này. Solver full hỗ trợ trực tiếp ngưỡng phương sai, dễ giải thích; bước fit có thể tốn thời gian và RAM, chỉ thực hiện một lần. Đo chi phí fit/transform riêng, không coi PCA miễn phí. Nếu không chạy được an toàn, ghi hạn chế và sửa kế hoạch trước lượt khảo sát tiếp, không âm thầm thay solver/ngưỡng rồi dùng chung tên cấu hình.

PCA được đặt trên HOG, không trên HSV hoặc vector ghép. Đây là quyết định kiểm soát câu hỏi giảm chiều đường nét trong khi giữ nguyên biểu diễn màu. Dùng cùng HOG-PCA khi đứng riêng và khi kết hợp. L2 sau PCA có thể làm thay đổi hình học khoảng cách; nhánh được đánh giá là toàn bộ phép xử lý PCA rồi L2, không khẳng định tác dụng chỉ đến từ giảm chiều.

Không thêm StandardScaler, whitening, PCA trên HSV hay tối ưu trọng số ghép trong vòng này. Để so sánh bộ phân loại, SVM cần dùng đúng cùng các ma trận đầu vào tương ứng; việc thêm scaler/PCA riêng sẽ tạo pipeline khác và phải công khai.

## Khóa phạm vi tìm tham số

Thiết kế mới dùng **Euclidean cho tất cả nhánh**, algorithm=brute. Lý do: có một khoảng cách thống nhất cho vector gốc và PCA có thể âm. Không dùng Bray–Curtis trên tọa độ PCA rồi diễn giải như histogram không âm. Đây là phạm vi nghiên cứu, không phải tuyên bố Euclidean tối ưu cho mọi biểu diễn. Kết quả HSV12/Bray–Curtis cũ vẫn được lưu là khảo sát trước.

Mỗi biến thể chạy k=1..10, lần lượt tăng dần; với mỗi k, duyệt uniform rồi distance: 20 cấu hình. Thứ tự biến thể cố định:

1. HSV12: 1.728 chiều.
2. HOG không PCA: 6.084 chiều.
3. HOG có PCA: d chiều.
4. HSV12 + HOG không PCA: 7.812 chiều.
5. HSV12 + HOG có PCA: 1.728+d chiều.

Tổng **100 cấu hình dev**, không phải 100 lần lặp độc lập. Con số này thay thế đề xuất 40 cấu hình/feature ở bản nháp. Vẫn chỉ có ba họ feature, nhưng cần hiển thị năm biến thể trong bảng dev để không che việc đã khảo sát PCA.

Giữ thêm bảng tham chiếu k=1, uniform, Euclidean cho cả năm biến thể (nằm sẵn trong 100 cấu hình, không chạy thêm). Bảng này cố định bộ phân loại khi phân tích biểu diễn. Bảng tốt nhất từng biến thể cho biết kết quả trong cùng phạm vi tìm tham số; không quy toàn bộ chênh lệch của bảng này cho feature khi k/weights thắng khác nhau.

## Chín bước và sản phẩm

### 01 — Chuẩn bị đầu vào chung và ví dụ chạy được

Phụ thuộc: không.

- Lập manifest train/dev có thứ tự cố định, đường dẫn và nhãn VND; kiểm tra số ảnh/lớp.
- Chạy hàm tiền xử lý trên ảnh train, minh họa gốc → RGB224 và giải thích dtype/kênh màu/nội suy.
- Chốt thư mục đầu ra mới; giữ nguyên bằng chứng cũ.
- Ghi cấu hình ở tài liệu này và bàn giao quy tắc xử lý hình học cho nhóm.

Hoàn thành khi có ví dụ chạy được, manifest khớp DATA và người dùng giải thích được vì sao 224/bilinear. Không cần viết framework hoặc in bảng kỹ thuật dài.

### 02 — Hoàn thành HSV12 + KNN trên dev

Phụ thuộc: 01.

Trích và lưu feature train/dev, chạy 20 cấu hình, lưu toàn bộ bảng tìm tham số và dự đoán của cấu hình thắng. Người dùng hiểu histogram ba chiều, L2, khoảng cách và hai cách bỏ phiếu. Kiểm tra từng dòng vector khớp manifest. Không dùng test.

### 03 — Hoàn thành HOG không PCA + KNN trên dev

Phụ thuộc: 01; có thể học sau 02 nhưng không cần kết quả 02 để chạy.

Trích HOG train/dev trước L2 toàn vector để bước 04 dùng cùng dữ liệu nguồn; nhánh này L2 toàn vector rồi chạy 20 cấu hình. Có ảnh xám/HOG minh họa và giải thích công thức 13×13×2×2×9=6.084. Đo một cấu hình trước khi quét đủ lưới để biết chi phí, không dùng phép đo ấy để tự đổi thiết kế.

### 04 — Hoàn thành HOG-PCA và đối chứng với HOG gốc

Phụ thuộc: 03.

Fit PCA chỉ trên HOG train, transform train/dev, L2 và chạy 20 cấu hình. Lưu đối tượng PCA, số chiều d, phương sai tích lũy, thời gian fit/transform/dự đoán cùng phạm vi đo, kết quả và dự đoán dev. So sánh với 03 ở cùng cấu hình tham chiếu và ở cấu hình tốt nhất. Không chọn nhánh thắng để bỏ qua biến thể fusion còn lại ở bước 05.

Hoàn thành khi người dùng hiểu PCA học gì, vì sao không dùng nhãn, vì sao phương sai không phải Accuracy và cách giảm chiều có thể làm mất thông tin phân lớp. Nhận xét là về PCA+L2 đã triển khai, không chỉ riêng số chiều.

### 05 — Hoàn thành hai biến thể kết hợp HSV12 + HOG

Phụ thuộc: 02, 03 và 04.

Đối chiếu từng đường dẫn/nhãn trước khi nối hai khối; tạo cả hai biến thể không PCA và có PCA. Mỗi biến thể chạy 20 cấu hình, tổng 40. Lưu kết quả kể cả khi kết hợp kém. Không tối ưu thêm trọng số HSV/HOG. Người dùng giải thích được hai loại thông tin bổ sung gì và vì sao vector dài hơn có thể không tốt hơn.

### 06 — So sánh dev và khóa KNN cuối

Phụ thuộc: 02, 03, 04 và 05.

Tổng hợp đủ 100 cấu hình; chọn một cấu hình cuối theo quy tắc Macro-F1 → MAE → thứ tự ứng viên. Lập bảng năm biến thể, ghi rõ ba họ feature và PCA có/không. Khóa cả ảnh, feature, chuẩn hóa, PCA nếu có, k, weights, metric, mapping, thứ tự dữ liệu và phiên bản môi trường. Không chọn fusion hoặc PCA chỉ vì muốn kết quả đi theo kỳ vọng. Không tự refit train+dev.

### 07 — Đánh giá KNN đã khóa trên test hiện có

Phụ thuộc: 06.

Dùng đúng mô hình/pipeline đã khóa để dự đoán toàn bộ 1.553 ảnh. Với PCA, chỉ transform bằng đối tượng đã fit, không fit lại. Lưu nhãn thật/dự đoán theo từng ảnh; tính metric bằng hàm chung. Đối chiếu số đúng, support, tổng confusion matrix và MAE từ tổng chênh lệch. Đây là đánh giá lại trên test đã có lịch sử sử dụng, không phải kiểm chứng mới hoàn toàn độc lập với quá trình khảo sát cũ.

### 08 — Phân tích lỗi từ dự đoán cuối

Phụ thuộc: 07.

Xem cặp nhầm thường gặp và cặp đóng góp nhiều tổng chênh lệch VND; chọn ví dụ theo tiêu chí ghi rõ. Hiển thị query và đủ k hàng xóm quyết định, nhãn, khoảng cách, trọng số và tổng phiếu. Các hàng xóm thêm ngoài k chỉ để tham khảo. Dùng đúng không gian feature/PCA của mô hình cuối khi tìm hàng xóm. Phân biệt quan sát và giả thuyết; không dùng lỗi test để quay lại thay cấu hình trong lượt báo cáo này.

### 09 — Hoàn thiện báo cáo và bàn giao SVM

Phụ thuộc: 08 để nghiệm thu toàn bộ; có thể viết phương pháp và bàn giao train/dev sớm hơn.

Hoàn thiện notebook chạy lại được, kết quả truy nguồn và báo cáo theo ba họ feature, có bảng phụ đối chứng PCA. Hình tập trung vào pipeline, minh họa feature, confusion matrix và một hoặc hai hình lỗi có hàng xóm. Ghi lịch sử khảo sát cũ, giới hạn resize và việc test đã được xem.

Bàn giao cho SVM các ma trận train/dev tương ứng năm biến thể, manifest, nhãn, đối tượng PCA đã fit và quy tắc chuẩn hóa. Bàn giao test theo bước đánh giá cuối của nhóm. Cập nhật các tài liệu cũ còn mâu thuẫn về kích thước ảnh/augmentation khi thực hiện bước này. Hồ sơ sẵn sàng review/push khi được yêu cầu; việc chốt lộ trình hiện tại không tự chạy code hay push.

## Trình tự và tiêu chí hoàn thành

01 → (02 và 03) → 04 → 05 → 06 → 07 → 08 → 09. Khi học và code một mình, đi lần lượt từ 01 đến 09.

Nhánh hoàn thành khi có năm biến thể dev minh bạch trong ba họ feature, một cấu hình cuối, kết quả test có dự đoán từng ảnh, phân tích lỗi, báo cáo và gói bàn giao. Nếu PCA hoặc fusion không cải thiện, vẫn hoàn thành và báo cáo đúng kết quả.

Ticket theo dõi cục bộ tại `.scratch/knn-224/issues`, mỗi bước một file. Các ticket phản ánh kế hoạch lúc khởi tạo; trạng thái cập nhật được ghi ở đầu tài liệu này. Các ticket không tự động giao agent chạy code.

Nguồn: [chuẩn metric](METRICS_STANDARD.md), [audit HOG/PCA](HOG_PCA_LITERATURE_AUDIT.md), [báo cáo KNN hiện tại](KNN_REPORT/BAO_CAO_THUC_NGHIEM_KNN.md). Cấu hình PCA dựa trên API [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html); ngưỡng 95% là lựa chọn của thiết kế này, không gán cho một paper cụ thể.
