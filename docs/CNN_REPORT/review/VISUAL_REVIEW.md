# Ghi chú đối chiếu hình CNN RGB224

Hồ sơ nội bộ cho báo cáo, dựa trên ba trang `cases_page_01/02/03.png`, learning curve, PCA dev và feature map của output notebook 08. Khóa liên quan: `9bccf375a9df5ceaafffa71b25f98c4aa5819ac8846fe15332b98ff6bac93edb`. Nhãn và điểm số đối chiếu từ `selected_cases.csv`; không suy nhãn thật chỉ từ ảnh. CSV ghi chú gốc của notebook được giữ nguyên để bảo toàn output tải về.

| Ví dụ | Quan sát trực tiếp | Điều chưa được xác minh |
|---|---|---|
| C01 | Tờ tiền dựng gần dọc, có tay ở mép; nhiều chấm tối và vùng chữ/họa tiết chưa rõ ở mức hiển thị | Chấm tối, độ nét hoặc nền có gây nhầm 20k→500k hay không |
| C02 | Tờ tiền nghiêng, tay xuất hiện ở vùng trên và mép trái; CAM hai lớp có các điểm đáp ứng khác nhau | Kênh/vùng nào quyết định lỗi 50k→10k; chưa làm occlusion test |
| C03 | Có thể quan sát cụm chữ số 200000; CAM dự đoán không chỉ tập trung ở cụm số | CNN có đọc/không đọc số; nhìn thấy con số không chứng minh cơ chế quyết định |
| C04 | Tờ tiền bị gấp và tay che một phần; có nền phía sau | Ảnh hưởng riêng của nếp gấp, tay hoặc nền tới lỗi |
| C05 | Hai tay giữ ảnh tiền có nhiều vùng sáng và chấm tối | Độ sáng hoặc nhiễu là nguyên nhân gây nhầm 20k→500k |
| C06 | Tờ tiền bị gấp, bàn tay chiếm một phần vùng ảnh; CAM lớp thật và dự đoán khác nhau | Việc thay phần bị che có làm đổi nhãn không |
| C07 | Toàn ảnh có tông vàng rõ; cụm số 500000 vẫn xuất hiện | Nguồn của tông màu và tác động nhân quả tới dự đoán 200k |
| C08 | Tay giữ tờ tiền trước nền nhiều trái cây; softmax sai rất cao được xác nhận từ CSV | Mô hình dùng trái cây/nền thay cho dấu hiệu tiền hay không |
| C09 | Tay che một phần tờ tiền; nền có nhiều vật và màu; hai CAM phân bố ở nhiều vùng | Nền hoặc che khuất có làm điểm lớp thật thấp hay không |
| C10 | Tờ tiền nghiêng; được dự đoán đúng với điểm softmax thấp và gap nhỏ | Có thể chọn ngưỡng từ chối chung hay không; chưa chọn ngưỡng trên dev |
| C11 | Tờ tiền tương đối mở, tay ở mép; dự đoán đúng và điểm float32 lớp thắng bằng 1 | Độ tin cậy khi gặp ảnh mới hoặc ngoài nguồn |
| C12 | Tay che một phần tờ tiền, ánh sáng và màu có biến thiên; nhầm 10k→50k | Tác dụng riêng của màu, che khuất hoặc điều kiện chụp |

Learning curve C02 cho thấy train/dev còn thấp và dev Macro-F1 còn tăng về cuối 25 epoch. Điều đó không phải bằng chứng của kiểu overfitting có train rất cao nhưng dev thấp; có thể còn giới hạn tối ưu/ngân sách, chưa kiểm định bằng lượt chạy dài hơn. C04 có dev loss thấp hơn train loss online, nhưng dropout/augmentation và thời điểm tính hai đường khác nhau nên không kết luận dữ liệu bị rò rỉ chỉ từ quan sát này.

PCA dev cho thấy các lớp 20k và 500k có nhiều điểm chồng lấn trong hai trục; tổng hai trục chỉ giữ 53,88% phương sai train. Không suy rằng hai lớp không thể phân biệt trong 256 chiều. Feature map đầu có nhiều đáp ứng theo mép/vùng ảnh, khối cuối 14×14 có mẫu rộng và thô hơn; chưa gán một ý nghĩa ngữ nghĩa duy nhất cho từng kênh hoặc coi sáu kênh được chọn đại diện cho toàn bộ mạng.

Các ảnh trong báo cáo giữ nguyên byte từ output, không tăng nét, chỉnh màu hoặc sửa heatmap. Báo cáo chính dùng trang ví dụ 02 và 03 để thể hiện lỗi có điểm cao và trường hợp sát điểm. Việc kiểm tra thị giác này chưa cung cấp nhãn điều kiện cho toàn bộ test và chưa xác lập tỷ lệ lỗi theo ánh sáng, nền hoặc mức che khuất.
