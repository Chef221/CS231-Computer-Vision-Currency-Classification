# Audit tài liệu: HOG có cần PCA trước khi phân loại?

Ngày đọc: 2026-11-05. Đây là rà soát có mục tiêu bằng nguồn gốc, không phải systematic review hay thống kê tỷ lệ paper dùng PCA. Chưa chạy thí nghiệm PCA trên DATA CS231.

## Bằng chứng đã kiểm tra

| Công trình | Vị trí kiểm tra | Kết quả đọc |
|---|---|---|
| Dalal & Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005, DOI 10.1109/CVPR.2005.177 | Hình 1; mục 3 và 6 | Pipeline HOG chính đưa vector ghép vào SVM, không có PCA. PCA-SIFT là phương án khác được so sánh. |
| Lai & Teoh, *An Efficient Method of HOG Feature Extraction Using Selective Histogram Bin and PCA Feature Reduction*, AECE 16(4), 101–108, 2016, DOI 10.4316/AECE.2016.04016 | Toàn văn, III.F, IV.C và bảng VII | Dùng thông tin hệ số PCA để chọn chiều gốc; không đồng nhất với phép chiếu sklearn.PCA. Feature sửa đổi 2.041 → chọn 1.500; HOG gốc đối chứng 3.780. Abstract nói linear SVM nhưng phần thực nghiệm nói RBF SVM; chỉ mô tả chung là SVM. |
| Narasimhaiah & Rangarajan, *Recognition of compound characters in Kannada language*, IJECE 12(6), 6103–6113, 2022, DOI 10.11591/ijece.v12i6.pp6103-6113 | Mục 3–4.1, bảng 5 và 13 | HOG → PCA → phân loại, có KNN; quy tắc 85% phương sai. Bảng 13 có trường hợp 1.296 → 94 chiều. Không coi bảng này là đối chứng chứng minh PCA tốt hơn không PCA. |
| Kaur & Priyadarshni, *Serial Number Recognition in Banknotes Using HoG Feature Extraction and KNN Classification*, IOSR-JCE 18(6-V), 41–49, 2016, DOI 10.9790/0661-1806054149 | Mục III; hình 9–10; mục IV | Pipeline trình bày HOG → KNN, không có bước PCA. Chọn cell để vector là 324 chiều. Đây là nhận diện ký tự số serial, không phải phân loại mệnh giá cả ảnh. |

Trong cấu hình HOG chuẩn của Dalal–Triggs, số chiều 3.780 tính từ 7×15 block × 36. Bài Zhu và cộng sự CVPR 2006 xác nhận trực tiếp số này khi mô tả phương pháp Dalal–Triggs trong mục 3. Đây là ví dụ rõ ràng rằng hàng nghìn chiều có thể được dùng trực tiếp với SVM.

## Ý nghĩa đối với CS231

Không có quy tắc HOG bắt buộc đi kèm PCA. Các hướng được thấy gồm dùng trực tiếp, giảm số chiều bằng cấu hình HOG, chiếu PCA, hoặc chọn feature bằng thông tin PCA. Chúng không phải cùng một thuật toán.

Với KNN, nên đánh giá PCA như một cách thay đổi biểu diễn và chi phí tính khoảng cách, không chỉ như giải pháp chữa thiếu RAM. Số chiều cao và số ảnh train khiến thử giảm chiều có lý do; lợi ích phân loại vẫn phải đo trên dev.

Đề xuất điều chỉnh bản nháp lộ trình: đặt trước một thí nghiệm HOG không PCA đối chiếu HOG có PCA; giữ cùng KNN và khoảng cách Euclidean để phân tích riêng phép giảm chiều. Báo cáo số chiều, phương sai giữ lại, Macro-F1, MAE và thời gian cùng cách đo. Không bắt buộc đưa toàn bộ khảo sát phụ vào bảng ba họ feature chính.

PCA chỉ fit train, dùng cùng đối tượng để transform dev/test và bàn giao SVM. Nếu áp dụng vào nhánh kết hợp phải ghi rõ PCA được đặt trên HOG hay trên cả vector nối; đó là hai thiết kế khác nhau. Đầu ra PCA có thể âm; không dùng nguyên lưới Bray–Curtis của histogram không âm cho nhánh này. Giữ 95% phương sai, nếu được chọn làm quy tắc, là lựa chọn thiết kế chứ không phải bảo đảm giữ 95% khả năng phân loại.

Đây là đề xuất dựa trên tài liệu; chưa sửa bản nháp lộ trình, notebook hoặc cấu hình thực nghiệm. Không chuyển số chiều hay điểm của bài khác thành thông số tối ưu hoặc kỳ vọng điểm cho CS231.

## Nguồn gốc

1. [Dalal–Triggs: bản tác giả, toàn văn](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).
2. [Zhu et al.: Fast Human Detection Using a Cascade of Histograms of Oriented Gradients, CVPR 2006, mục 3](https://lear.inrialpes.fr/people/triggs/student/vj/zhu-cvpr06.pdf).
3. [Lai–Teoh: trang nhà xuất bản](https://www.aece.ro/abstractplus.php?article=16&number=4&year=2016); [toàn văn](https://www.aece.ro/displaypdf.php?article=16&number=4&year=2016). Trình đọc web không hỗ trợ content-type của link tải; đã tải vào bộ nhớ và đọc PDF 8 trang bằng pypdf, kiểm tra các mục nêu trên.
4. [Narasimhaiah–Rangarajan: toàn văn nhà xuất bản](https://ijece.iaescore.com/index.php/IJECE/article/download/26219/16052).
5. [Kaur–Priyadarshni: toàn văn nhà xuất bản](https://www.iosrjournals.org/iosr-jce/papers/Vol18-issue6/Version-5/H1806054149.pdf).
