# Thực nghiệm phân loại tiền polymer Việt Nam bằng CNN

{{STATUS}}

{{OVERVIEW}}

## 1. Mục tiêu và thiết lập thực nghiệm

Phần này đánh giá một CNN huấn luyện từ đầu trên cùng bộ dữ liệu đã dùng cho KNN và SVM. Mạng nhận pixel ảnh và tự học bộ lọc cùng bộ phân loại; không nhận histogram HSV12 hoặc HOG làm đầu vào. Bốn thí nghiệm khảo sát thông tin màu, augmentation và việc thêm một khối tích chập. Mục tiêu là chọn cấu hình và checkpoint trên dev, khóa trọng số, đánh giá test và phân tích các cặp nhầm mệnh giá.

Dữ liệu gồm sáu mệnh giá 10.000đ, 20.000đ, 50.000đ, 100.000đ, 200.000đ và 500.000đ, tổ chức theo `DATA/{train,dev,test}/{mệnh_giá}/*.png`. Train có **{{TRAIN_N}} ảnh**, dev có **{{DEV_N}} ảnh**, test có **{{TEST_N}} ảnh**. Nguồn và quy trình M0 được mô tả tại README dự án: crop theo nhãn, loại trùng pixel hoàn toàn trong cùng lớp và chia tập theo nhóm ảnh gốc. Quy trình này chưa chứng minh đã loại hết ảnh gần trùng hoặc cùng tờ tiền vật lý xuyên split.

{{DATA_TABLE}}

Train dùng cập nhật trọng số; dev dùng chọn checkpoint và một trong bốn cấu hình; test dùng đánh giá cấu hình đã khóa. Không gộp train+dev. Test đã từng được xem trong các thực nghiệm trước của dự án, nên không gọi đây là holdout hoàn toàn mới. Mỗi cấu hình dùng một seed; chưa có ước lượng độ bất định qua nhiều lần chạy.

## 2. Tiền xử lý ảnh đầu vào

Ảnh được đọc bằng `src.preprocessing.read_rgb224`: OpenCV đổi BGR thành RGB và resize trực tiếp 224×224 bằng `INTER_LINEAR`. Đây là cùng quy tắc hình học của KNN/SVM RGB224. Resize trực tiếp giữ nội dung crop nhưng có thể đổi tỷ lệ hình học. Ảnh nguồn không bị ghi đè. CNN chuyển ảnh thành tensor CHW float32 và chia 255 để đưa pixel về [0,1]; không áp chuẩn hóa ImageNet vì không dùng trọng số pretrained, không fit scaler và không L2 descriptor như KNN/SVM.

{{INPUT_FIGURE}}

Biến thể ảnh xám chuyển RGB224 thành một kênh bằng OpenCV. Hai nhánh có augmentation chỉ biến đổi train: xoay trong ±12°, dịch chuyển tối đa 5% mỗi chiều, scale 0,95–1,05; thay đổi độ tương phản 0,9–1,1 quanh 127,5 và cộng độ sáng trong ±10% thang 255. Phần biên phát sinh được điền xám RGB(127,127,127). Không lật ảnh, không đổi hue/saturation. Đây là một gói biến đổi được khai báo trước, không phải kiểm định riêng từng phép biến đổi. Dev/test chỉ dùng resize và chia 255.

{{AUGMENTATION_FIGURE}}

Label encoding dùng thứ tự mệnh giá theo số `[10000,20000,50000,100000,200000,500000]`, ánh xạ sang chỉ số 0…5 khi tính loss. Trước khi gọi hàm đánh giá chung, cả nhãn thật và dự đoán được đổi về VND. Không lấy thứ tự tên thư mục theo chuỗi làm thứ tự mệnh giá.

## 3. Kiến trúc và cách CNN học đặc trưng

Mạng cơ sở có ba khối với 32, 64 và 128 kênh. Mỗi khối gồm hai lần `Conv3×3 → BatchNorm → ReLU`, sau đó `MaxPool2×2 → Dropout2d(0,1)`. Conv dùng stride 1, padding 1, không bias vì đã có BatchNorm. Biến thể sâu thêm khối 256 kênh. Sau các khối là global average pooling (GAP), Dense128, ReLU, Dropout(0,5) và Dense6 trả về logits.

**Bảng CNN-1. Kích thước không gian của các khối.**

| Thành phần | CNN ba khối | CNN bốn khối |
|---|---|---|
| Đầu vào | 3×224×224; nhánh xám 1×224×224 | 3×224×224 |
| Khối 1 | 32×112×112 | 32×112×112 |
| Khối 2 | 64×56×56 | 64×56×56 |
| Khối 3 | 128×28×28 | 128×28×28 |
| Khối 4 | — | 256×14×14 |
| GAP | 128 chiều | 256 chiều |
| Bộ phân loại | Dense128 → Dropout0,5 → Dense6 | Dense128 → Dropout0,5 → Dense6 |
| Tổng tham số (RGB) | 304.742 | 1.206.886 |

Nhánh ảnh xám có 304.166 tham số; chênh 576 trọng số ở Conv đầu do số kênh đầu vào giảm từ ba xuống một. Đây là số tham số tính từ kiến trúc, không phải kết quả huấn luyện.

![Kiến trúc CNN và vị trí biểu diễn GAP](CNN224_FIGURES/CNN_ARCHITECTURE.svg)

Conv học các phép đáp ứng cục bộ; ReLU tạo phi tuyến; pooling giảm kích thước feature map; GAP lấy trung bình không gian của mỗi kênh để tạo vector. BatchNorm dùng thống kê mini-batch khi train và thống kê đã tích lũy khi eval. Dropout tắt ngẫu nhiên kích hoạt trong train và được tắt khi đánh giá. Không có bảo đảm rằng một kênh cụ thể chỉ mã hóa màu, cạnh hay chữ số; việc gán ý nghĩa cần bằng chứng từ trực quan và đối chứng.

Trọng số Conv khởi tạo Kaiming normal (`fan_out`, ReLU); Dense dùng Xavier uniform, bias bằng 0; BatchNorm có scale 1 và bias 0. Các lựa chọn này là khởi tạo, không phải trọng số học từ ảnh ngoài dự án. Tổng tham số và shape thực tế được notebook 01 tính từ module và lưu trong `inputs/architecture.csv`.

**Bảng CNN-2. Thiết lập huấn luyện cố định.**

| Thành phần | Cấu hình |
|---|---|
| Seed | 42 cho mỗi thí nghiệm; seed theo epoch để hỗ trợ resume |
| Optimizer | AdamW, lr=0,001; betas=(0,9; 0,999); weight decay=0,0001 |
| Loss | CrossEntropyLoss trên logits; không class weight, không label smoothing |
| Batch / số epoch tối đa | 64 / 25 |
| Learning rate | CosineAnnealingLR theo epoch, eta_min=0,00001 |
| Dừng sớm | 6 epoch liên tiếp không cải thiện cặp (Macro-F1 dev, MAE dev) |
| Gradient / số học | Clip norm=5; CUDA dùng AMP; CPU dùng float32 |
| DataLoader | 2 workers; shuffle train, giữ thứ tự dev/test |
| Pretrained / refit | Không pretrained; không refit train+dev |

Cross-entropy học phân loại sáu lớp, không tối ưu trực tiếp MAE VND. MAE được dùng khi đánh giá và phá hòa trên dev. CuDNN benchmark và TF32 được tắt; thuật toán deterministic được yêu cầu ở chế độ cảnh báo. Seed không bảo đảm kết quả giống từng bit giữa phiên bản thư viện, loại GPU hoặc thiết bị khác nhau. Notebook dùng một GPU nếu có; không cộng hai GPU T4 thành một GPU lớn.

Batch 64 là điểm khởi đầu để cân bằng bộ nhớ GPU và thống kê BatchNorm. Learning rate 0,001 phù hợp làm mốc khảo sát AdamW; weight decay 0,0001 và dropout tạo regularization ở mức khai báo cố định. GAP giữ đầu phân loại gọn hơn việc Flatten toàn bộ lưới 28×28×128. Cosine schedule không cần chọn tốc độ giảm từ test; giới hạn 25 epoch và patience 6 khống chế ngân sách mỗi lượt. Các giá trị này dễ đối chiếu giữa bốn cấu hình, chưa được coi là bộ tham số tối ưu hoặc bảo đảm đã hội tụ.

## 4. Thí nghiệm từng biểu diễn và nhận xét

**Bảng CNN-3. Bốn cấu hình được khai báo trước khi khảo sát dev.**

| Thứ tự | Mã | Đầu vào | Khối | Augmentation train | Câu hỏi |
|---:|---|---|---|---|---|
| 1 | C01_RGB | RGB | 32–64–128 | Không | Mốc CNN học trực tiếp từ pixel |
| 2 | C02_GRAY | Ảnh xám | 32–64–128 | Không | Ảnh hưởng của việc bỏ thông tin màu |
| 3 | C03_RGB_AUG | RGB | 32–64–128 | Có | Ảnh hưởng của gói augmentation |
| 4 | C04_DEEP_AUG | RGB | 32–64–128–256 | Có | Ảnh hưởng của thêm một khối và dung lượng mạng |

C02 đổi cả biểu diễn đầu vào và số trọng số ở Conv đầu, nên không cô lập hoàn toàn màu sắc. C03 thay đổi nhiều phép augmentation cùng lúc. C04 đồng thời tăng tham số và giảm kích thước không gian sâu nhất. Cùng seed không đồng nghĩa mọi trọng số khởi tạo của các kiến trúc là bản sao của nhau. Chỉ báo cáo tác động của các cấu hình cụ thể trong phạm vi khảo sát.

{{CANDIDATE_OBSERVATIONS}}

Learning curve ghi train loss/Accuracy trong các mini-batch đang cập nhật trọng số, với dropout và augmentation nếu có; dev được chấm bằng model.eval() sau mỗi epoch. Vì cách tính khác nhau, khoảng cách hai đường không phải phép đo thuần túy duy nhất của overfitting. Mỗi epoch chấm toàn bộ dev một lần, không lấy trung bình Macro-F1 theo batch.

{{LEARNING_FIGURE}}

Sau khi khóa mô hình, notebook 06 lưu vector GAP của train/dev ở chế độ eval. PCA hai chiều được fit trên toàn train, rồi transform dev để trực quan. PCA này không được đưa vào bộ phân loại, không tham gia chọn checkpoint và không giống thí nghiệm HOG-PCA của KNN/SVM. Khoảng chồng lấn hay phân cụm trong hai trục không thay thế metric ở toàn bộ không gian đặc trưng.

{{PCA_TEXT}}

{{EMBEDDING_FIGURE}}

Feature map được minh họa ở khối đầu và khối cuối trên một ảnh train cố định; mỗi khối chọn sáu kênh có đáp ứng trung bình lớn nhất, không chọn theo độ đẹp. Thang màu từng kênh được tự co giãn nên không dùng để so độ lớn tuyệt đối giữa kênh.

{{FEATURE_MAP_FIGURE}}

## 5. Tổng hợp dev và khóa phương án cuối

Chọn checkpoint mỗi cấu hình bằng Macro-F1 dev chưa làm tròn cao nhất, hòa chính xác thì MAE thấp hơn, vẫn hòa giữ epoch sớm hơn. Dừng sớm dùng cùng cặp tiêu chí. Khi so bốn cấu hình, ưu tiên Macro-F1 rồi MAE, sau đó thứ tự C01…C04 và epoch. Đây là một split dev và tối đa 100 lượt epoch được quan sát, không phải cross-validation hay 100 cấu hình độc lập.

**Bảng CNN-4. Checkpoint tốt nhất trên dev của mỗi cấu hình.**

{{DEV_TABLE}}

{{SELECTION_TEXT}}

Notebook 06 ghi hash cho protocol, manifest, lịch sử, dự đoán dev, bảng tổng hợp, checkpoint và mã xử lý. Notebook 07 kiểm tra khóa và nạp chính trọng số đã chọn; không khởi tạo lại để huấn luyện một lượt mới. `last.pt` phục vụ tiếp tục một lượt bị ngắt ở ranh giới epoch; `best.pt` là checkpoint được chọn. Một lượt đã hoàn tất sẽ được kiểm tra hash và đọc lại thay vì train lại khi Run All.

**Bảng CNN-5. Thời gian train/dev cộng từ lịch sử epoch.**

{{RUNTIME_TABLE}}

{{RUNTIME_TEXT}}

{{HISTORY_LINK}}

## 6. Đánh giá phương án đã khóa trên test

**Bảng CNN-6. Kết quả dev và test của cấu hình đã khóa.**

{{TEST_TABLE}}

{{TEST_TEXT}}

**Bảng CNN-7. Precision, Recall và F1 theo mệnh giá trên test.**

{{PER_CLASS_TABLE}}

{{PER_CLASS_TEXT}}

{{CONFUSION_FIGURE}}

Độ đo dùng cùng `src.evaluation.value_metrics.evaluate_classification` với KNN/SVM. Macro-F1 là trung bình F1 của sáu lớp. MAE = `Σ |mệnh_giá_thật − mệnh_giá_dự_đoán| / N`, tính trên toàn bộ ảnh; ảnh đúng góp 0. MAE có đơn vị đồng/ảnh và mô tả chênh lệch danh nghĩa. Softmax chỉ dùng mô tả điểm của mô hình, chưa được hiệu chỉnh thành xác suất đúng; không dùng ngưỡng để bỏ ảnh khó trong bảng test chính.

## 7. Phân tích lỗi và chênh lệch mệnh giá

**Bảng CNN-8. Các cặp nhầm đóng góp tổng chênh lệch lớn nhất, giữ hai chiều riêng.**

{{ERROR_TABLE}}

{{ERROR_SUMMARY}}

Notebook 08 chọn tối đa 12 ảnh không trùng: từ các chiều nhầm phổ biến, các chiều gây tổng chênh lệch lớn, ảnh sai có softmax cao hoặc hai điểm đầu sát nhau, và ảnh đúng đối chứng. Quy tắc chọn có thể kiểm tra trong `selected_cases.csv`; không dùng vài ví dụ thay số liệu toàn test. File `visual_review_notes.csv` dành cho quan sát, giả thuyết và điều chưa xác minh, được để trống đến khi người đọc thực sự xem ảnh.

Grad-CAM lấy gradient logit của lớp mục tiêu theo feature map ở ReLU thứ hai của khối cuối, trước MaxPool. Trọng số kênh là gradient trung bình theo không gian; tổng có trọng số được ReLU và co giãn về [0,1] riêng cho từng ảnh và lớp. Các map bằng 0 được giữ và ghi vào `gradcam_checks.csv`. So sánh cả lớp dự đoán và lớp thật để quan sát sự khác biệt, không coi heatmap là chứng minh mô hình đọc đúng chữ số hoặc nền là nguyên nhân gây lỗi. Màu nóng của hai map đã chuẩn hóa riêng không so sánh được độ lớn tuyệt đối.

{{ERROR_FIGURE}}

{{CASE_TEXT}}

{{BOUNDARY_FIGURE}}

## 8. So sánh với KNN và SVM trên cùng dữ liệu

KNN RGB224 đã khóa dùng HSV12, k=1, uniform, Euclidean. SVM RGB224 đã khóa dùng HSV12+HOG, RBF C=10, gamma=scale. Bản sao CSV/JSON tham chiếu và SHA-256 nằm tại `src/CNN/reference_results`; nguồn commit được ghi riêng, không lấy số từ báo cáo trái cây. Trước khi so sánh, chương trình đối chiếu đường dẫn, nhãn và thứ tự ảnh trên cả ba manifest.

**Bảng CNN-9. Kết quả toàn bộ test của các phương án đã chọn bằng dev.**

{{COMPARISON_TABLE}}

{{COMPARISON_TEXT}}

**Bảng CNN-10. So sánh đúng/sai ghép cặp từng ảnh.**

{{PAIRED_TABLE}}

{{PAIRED_TEXT}}

**Bảng CNN-11. F1 test theo mệnh giá của ba phương án đã khóa.**

{{PER_CLASS_COMPARISON_TABLE}}

CNN học cả đặc trưng lẫn bộ phân loại; hai cấu hình có thể dùng augmentation. KNN/SVM dùng đặc trưng thủ công và không augmentation. Vì vậy, bảng so sánh đánh giá các pipeline hoàn chỉnh, chưa tách riêng hiệu quả của thuật toán, biểu diễn hay augmentation. Khác biệt phần cứng/thư viện và cách đo không cho phép suy luận tốc độ chung từ thời gian một lượt chạy.

## 9. Kết luận và giới hạn

{{CONCLUSION}}

Phạm vi hiện tại gồm một split, một seed, bốn cấu hình CNN gọn và tối đa 25 epoch mỗi cấu hình. Không có bảo đảm đây là kiến trúc hay learning rate tối ưu. Test đã có lịch sử được xem; metadata chưa bảo đảm độc lập theo mọi tờ tiền hoặc phiên chụp. Bài toán chỉ phân loại crop của sáu mệnh giá đã biết, chưa kiểm tra phát hiện tiền trong ảnh toàn cảnh, ảnh ngoài tập hoặc cơ chế từ chối dự đoán. Phân tích trực quan nêu dấu hiệu có thể quan sát, chưa định lượng nguyên nhân nếu chưa có nhãn điều kiện hoặc ablation riêng.

Một hướng mở rộng là chạy nhiều seed, tạo test độc lập theo nguồn hoặc phiên chụp và đánh dấu điều kiện ảnh để kiểm tra các giả thuyết về lỗi. Nếu tăng ngân sách epoch hoặc khảo sát augmentation khác, cần lưu thành lần thực nghiệm mới và chọn lại trên dev; giữ nguyên bản kết quả đã khóa ở đây.

{{VERIFICATION_TEXT}}

## Tài liệu kỹ thuật tham khảo

1. Dữ liệu, chuẩn metric và quy trình nhóm: `README.md`, `docs/EXPERIMENT_PLAN.md`, `docs/METRICS_STANDARD.md`, `src/evaluation/value_metrics.py`.
2. Ý tưởng CNN ba khối, BatchNorm, dropout và GAP từ tài liệu tham khảo “Hệ thống Phân loại Trái cây Tại Quầy Thu Ngân”, trang 27–30. Phần kế thừa và điều chỉnh được ghi trong `REFERENCE_NOTES.md`; không kế thừa kết quả, split hoặc nhãn trái cây.
3. [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html).
4. Selvaraju và cộng sự, ICCV 2017. [Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
5. Kết quả của dự án: `src/KNN/results_224`, nhánh SVM commit `6d87bc4`, và `src/CNN/results_224`. File CSV/JSON giữ số chưa làm tròn; Markdown dùng dấu phẩy thập phân và dấu chấm phân cách hàng nghìn.
