# Tài liệu tham khảo cho đề cương thực nghiệm

Bảng dưới ghi nguồn và phần thực sự được tham khảo. Không dùng các bài này để khẳng định một ngưỡng, một công thức custom hoặc độ chính xác đã được xác nhận cho DATA của nhóm.

| Mã | Tài liệu | Vị trí bằng chứng và phạm vi |
|---|---|---|
| R1 | F. M. Hasanuzzaman, X. Yang, Y. Tian. **Robust and Effective Component-Based Banknote Recognition for the Blind**. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 42(6), 1021–1030, 2012. [DOI](https://doi.org/10.1109/TSMCC.2011.2178120), [bản toàn văn](https://pmc.ncbi.nlm.nih.gov/articles/PMC3363999/) | Mục III-C, V–VI: ngưỡng học từ training, ảnh có tiền/ảnh nền và thử nghiệm với người khiếm thị. Không có độ đo chênh lệch VND của nhóm. |
| R2 | T. D. Pham và cộng sự. **Efficient Banknote Recognition Based on Selection of Discriminative Regions with One-Dimensional Visible-Light Line Sensor**. Sensors, 16(3), 328, 2016. [DOI](https://doi.org/10.3390/s16030328), [bản toàn văn](https://pmc.ncbi.nlm.nih.gov/articles/PMC4813903/) | Mục 2.4.2, Bảng 4, Hình 10–11: khoảng cách lớp gần nhất và lớp gần thứ hai, error/rejection rates, ảnh sai và từ chối. Không phải phương pháp bỏ phiếu KNN. |
| R3 | F. Oviedo và cộng sự. **BankNote-Net: Open dataset for assistive universal currency recognition**. arXiv:2204.03738, 2022. [bản ghi arXiv](https://arxiv.org/abs/2204.03738), [toàn văn HTML](https://arxiv.org/html/2204.03738) | Mục 3.2, Hình 3 và mục 4: Accuracy/Precision/Recall; mục tiêu Precision với confidence từ 0,99 trong một biến thể triển khai. Không bảo đảm confidence đó đúng cho mô hình khác. |
| R4 | Y. Geifman, R. El-Yaniv. **Selective Classification for Deep Neural Networks**. Advances in Neural Information Processing Systems 30, 2017. [bài tại NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) | Khung selective classification: đánh đổi lỗi với coverage. Đề cương CS231 không tự kế thừa bảo đảm rủi ro lý thuyết của bài báo. |

R3 được dẫn theo bản preprint arXiv đã đọc, không tự gán một hội nghị xuất bản. R4 có bản hội nghị và bản arXiv tương ứng.

Thông tin DOI/arXiv được đối chiếu bằng nguồn Crossref/arXiv. Hồ sơ gồm [các phát biểu và bằng chứng](references/citation_requests.json) cùng [kết quả xác minh metadata](references/citation_lock.json). Xác minh metadata không thay cho việc đọc nội dung; vị trí hỗ trợ và giới hạn của từng phát biểu được ghi riêng trong bảng và hồ sơ.

Các nguồn phần mềm cho công thức nằm ở cuối [M2.2](M2.2%20-%20Mô%20tả%20độ%20đo%20đánh%20giá.md): tài liệu chính thức scikit-learn về Precision/Recall/F1, confusion matrix và MAE. Mã nguồn dự án tính các độ đo từ nhãn thật/dự đoán; chưa có kết quả huấn luyện được tạo trong lần cập nhật tài liệu này.
