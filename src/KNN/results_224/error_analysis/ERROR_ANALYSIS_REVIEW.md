# Kiểm tra notebook 08 và nhận xét phân tích lỗi

Kiểm tra ngày 05/11/2026. **PASS về tính nhất quán kỹ thuật.** Đã xem ba trang ảnh gồm 12 query và các ảnh train đối chiếu, cùng ba hình histogram. Không xem bằng mắt toàn bộ 366 lỗi.

## 1. Phạm vi đã xác minh

- Tất cả cell code đã chạy, không có output lỗi được lưu; các hash đầu vào và file khóa khớp.
- Toàn bộ 1.553 dòng liên kết đúng dự đoán test; bảng lỗi có đúng 366 dòng.
- Tính lại khoảng cách nhỏ nhất tới từng lớp trên toàn bộ train cho từng query: hàng xóm quyết định, hàng xóm đúng lớp và hàng xóm khác lớp đều đạt khoảng cách tối thiểu tương ứng. Kiểm tra lại đường dẫn, nhãn, margin và cờ gần hòa: khớp.
- Tính lại các bảng từng lớp, cặp lỗi hai chiều, cặp đóng góp chênh lệch và quy tắc chọn 12 ví dụ: khớp. Không có query bị chọn trùng đường dẫn.
- Đọc lại 12 ảnh test và 21 ảnh train khác nhau xuất hiện trong hình để trích RGB224 → HSV12 → L2: khớp chính xác feature đã lưu.

Evidence máy đọc được: `audit_results.json`. Kiểm tra trên không phải kiểm toán lại nhãn hoặc tính độc lập của dataset.

## 2. Những trường hợp có giá trị giải thích

| Case | Quan sát ảnh và kết quả | Điều có thể kết luận |
|---|---|---|
| C01 | Query 10.000đ tối, có nhiều đốm; ảnh 200.000đ được chọn cũng có tông tối/nâu. Ảnh đúng lớp đối chiếu sáng hơn rõ rệt. Khoảng cách sai 0,65365, đúng lớp 0,91669. | Trong không gian feature hiện tại, ví dụ khác lớp gần hơn. Ánh sáng/nền là giả thuyết phù hợp với quan sát, chưa được cô lập để chứng minh. |
| C02 | Query 200.000đ trên bàn phím tối, số 200000 còn nhìn được; train 10.000đ được chọn có nền đen. Khoảng cách sai 0,79122, đúng lớp 1,04515. | Thông tin chữ số nhìn thấy không được histogram đọc như ký tự. Chưa định lượng riêng ảnh hưởng của nền. |
| C03 | Query 20.000đ và train 500.000đ được chọn đều có tờ tiền xanh nhạt trên bề mặt sáng; ảnh đúng lớp đối chiếu có bàn tay và bối cảnh khác. Khoảng cách 0,49756 so với 0,91765. | Màu toàn ảnh có thể tương tự giữa hai mệnh giá; khoảng cách joint HSV12 xác nhận lựa chọn sai này. |
| C04 | Query 500.000đ và train 20.000đ được chọn đều có nền xanh lớn; khoảng cách 0,10337 so với 0,28476 tới train đúng lớp. | Đây là ví dụ khoảng cách nhỏ vẫn đi cùng nhãn sai. Không được đổi khoảng cách thành xác suất hay coi khoảng cách nhỏ là bảo đảm đúng. |
| C05 | Query 200.000đ trên nền đan, còn train 500.000đ được chọn bị nhòe; query vẫn có số mệnh giá đọc được. Khoảng cách 0,62685 so với 0,86884. | KNN so histogram toàn ảnh, không ưu tiên ảnh train nhìn rõ hơn theo đánh giá con người. |
| C06 | Query 500.000đ có tay và nền nhiều chi tiết; train 200.000đ được chọn cũng có bàn tay chiếm vùng đáng kể. Khoảng cách 0,72586 so với 0,97322. | Bàn tay/nền là thành phần được histogram đếm cùng tờ tiền. Chưa chứng minh riêng thành phần nào gây lỗi. |
| C07–C08 | C07 rất giống C03 về cách chụp và màu; C08 rất giống C04. Mỗi query có đường dẫn riêng, và từng cặp dùng chung train sai gần nhất. | Hai quy tắc chọn ví dụ có thể lấy các ảnh tương tự. Không coi chúng là bốn cơ chế lỗi độc lập; báo cáo chỉ cần một ảnh mỗi nhóm tương tự. |
| C09 | Query 10.000đ hiển thị số 10000 và bố cục tờ tiền, nhưng train gần nhất là tờ 500.000đ gấp trên bề mặt xanh/xám. Khoảng cách 0,50741 so với 0,88216. | Ví dụ phù hợp để trả lời “ảnh khá rõ sao vẫn sai”: sự tương đồng theo HSV12 khác sự tương đồng theo chữ số và bố cục mà con người sử dụng. |
| C10 | Query 100.000đ bị gọi thành 20.000đ. Khoảng cách sai 0,56349629, đúng lớp 0,56369638; chênh 0,00020009. | Là lỗi có chênh lệch hai nhóm lớp nhỏ nhất trong 366 lỗi theo quy tắc chọn. Không phải hòa chính xác; chưa thử thay đổi ảnh nên không kết luận chắc chắn một nhiễu nhỏ sẽ đảo nhãn. |
| C11 | Query 200.000đ khá nhòe nhưng dự đoán đúng; hai khoảng cách là 0,79974383 và 0,79995364, margin dương 0,00020981. | Ảnh nhòe vẫn có thể được phân loại đúng. Không thể giải thích mọi lỗi chỉ bằng độ mờ. |
| C12 | Query 10.000đ và train được chọn rất giống về bố cục, màu và nền; khoảng cách đúng lớp 0,06472, khác lớp 0,98190. | Minh họa một quyết định đúng có khoảng cách tách biệt lớn. Chưa xác minh nguồn gốc/quan hệ hai ảnh; không dùng để khẳng định khả năng tổng quát trên cảnh mới. |

## 3. Ý nghĩa của histogram minh họa

C01 cho thấy phân bố độ sáng của query gần ảnh được chọn hơn ảnh đối chiếu sáng mạnh. C02 cho thấy query và ảnh được chọn cùng có đỉnh Hue ở vùng bin thấp. C11 cũng có các phân bố biên tương tự ở một số vùng dù hai lớp cạnh tranh gần nhau về khoảng cách.

Các hình chỉ vẽ phân bố biên H/S/V sau khi cộng histogram chung theo hai trục. Chúng giúp đọc đặc điểm màu, không trực tiếp biểu diễn Euclidean trên 1.728 chiều. Không suy ra thứ tự khoảng cách joint chỉ bằng độ giống của ba đường biên.

## 4. Cách viết kết luận có thể bảo vệ trước giảng viên

“KNN k=1 chọn nhãn của ảnh train có histogram HSV12 gần ảnh đầu vào nhất. Histogram tóm tắt phân bố màu của toàn ảnh, không mã hóa trực tiếp chữ số hoặc vị trí họa tiết. Vì vậy, một ảnh có mệnh giá đọc được vẫn có thể nằm gần ảnh khác mệnh giá hơn trong không gian đặc trưng. Các ví dụ C04 và C09 minh họa hạn chế này. Nền và ánh sáng là những yếu tố có thể góp phần làm thay đổi histogram, nhưng phân tích hiện tại chưa cô lập để định lượng tác động riêng của từng yếu tố.”

Không báo tỷ lệ lỗi “do nền”, “do mờ” hoặc “do ánh sáng”: chưa có nhãn nguyên nhân cho toàn test hoặc thí nghiệm đối chứng nguyên nhân. Không gọi margin là độ tin cậy đã hiệu chỉnh. Không thay cấu hình theo các lỗi vừa xem.

## 5. Đề xuất hình cho báo cáo

Ưu tiên C04 và C09 để giải thích hạn chế màu toàn ảnh và trường hợp mệnh giá đọc được vẫn sai. Thêm C10 cùng C11 nếu cần minh họa hai quyết định sát ranh giới ở hai phía. C12 chỉ làm đối chiếu phụ, kèm giới hạn về hai ảnh rất giống nhau. Không cần đưa đồng thời C03/C07 hoặc C04/C08 vào báo cáo; giữ cả 12 trong evidence để minh bạch lựa chọn.

Bước phân tích đã đủ bằng chứng kỹ thuật và nhận xét trên các ví dụ để chuyển sang tổng hợp báo cáo. Kết luận về tính độc lập của train/test và nguyên nhân thị giác ở quy mô toàn dataset vẫn nằm ngoài phạm vi xác minh này.
