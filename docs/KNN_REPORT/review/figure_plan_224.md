# Kế hoạch hình cho báo cáo KNN RGB224

Đây là phần báo cáo học phần bằng Markdown, không phải bản thảo hội nghị trọn vẹn. Số liệu lấy từ `results_224`; kết quả cũ được lưu riêng. Không tạo ảnh sinh cho bằng chứng.

| Hình | Vị trí | Thông điệp và bố cục | Nguồn | Backend / dự phòng |
|---|---|---|---|---|
| KNN-1 | Tiền xử lý | Hai ảnh train: ảnh trước resize và RGB224 cạnh nhau; thể hiện thay đổi tỷ lệ trực tiếp | Manifest train, PNG DATA; ghi đường dẫn trong provenance | matplotlib, PNG 300 DPI; có mô tả chữ nếu hình chưa tải |
| KNN-2 | Histogram HSV12 | Hai ảnh train 10.000đ và 500.000đ, mỗi hàng gồm RGB224 và ba mặt phẳng H, S, V, có thang giá trị; giải thích pixel đi vào bin kết hợp | train/10000/10000_000896.png và train/500000/500000_001722.png, RGB224 và cv2.COLOR_RGB2HSV | matplotlib, PNG 300 DPI; minh họa trực tiếp dữ liệu, không phải sơ đồ sinh |
| KNN-3 | Đặc trưng HOG | Một ảnh train RGB, ảnh xám và ảnh trực quan HOG cùng cấu hình; cho thấy thông tin màu không còn trong nhánh HOG xám | Hàm HOG notebook 03, cùng PNG train | matplotlib; PNG 300 DPI, ghi tham số; ảnh HOG không phải vector đầu vào trực tiếp |
| KNN-4 | Đánh giá test | Ma trận nhầm lẫn 6×6; hàng thật, cột dự đoán, số ảnh trong ô | confusion_matrix_test.csv | matplotlib; PNG 300 DPI, PDF vector; bảng độ đo là dự phòng |
| KNN-5 | Phân tích lỗi | Hai hàng C04 và C09: query / train được chọn / train đúng lớp gần nhất; có nhãn và khoảng cách | selected_cases.csv, feature và manifest, PNG DATA | matplotlib, PNG 300 DPI; CSV lưu đầy đủ trường hợp |
| KNN-6 | Phân tích lỗi sát ranh giới | Hai hàng C10 và C11: quyết định sai/đúng có chênh lệch khoảng cách nhỏ; ảnh đối chiếu đúng lớp khi sai, khác lớp khi đúng | Như hình KNN-5 | matplotlib, PNG 300 DPI; bảng khoảng cách là dự phòng |

Tất cả là evidence/result figures, không phải concept diagram. Tái tạo bằng `build_figures_224.py`; không fit PCA hoặc KNN. Màu xanh dương cho colormap; nhãn khoảng cách không đổi sang xác suất. Ghi provenance và hash nguồn. Không cần biểu đồ cột lặp lại bảng năm biến thể hoặc biểu đồ histogram biên trong phần chính.

Rủi ro diễn giải: 12 query có chủ đích, có vài query trông rất giống nhau; không đại diện tần suất nguyên nhân. Chọn bốn case để minh họa các lập luận khác nhau, công khai danh sách trong CSV. Giả thuyết nền/ánh sáng chưa được kiểm chứng bằng đối chứng. C12 không làm ví dụ chính vì rất giống ảnh train và chưa xác minh nguồn gốc.

Hình HSV là trực quan giá trị đo trên ảnh train thật; S/V hiển thị xám 0–255, H hiển thị bản đồ màu tuần hoàn 0–179 kèm colorbar. Không hiển thị mảng HSV như RGB. Ba mặt phẳng không phải ba feature riêng: thuật toán vẫn đếm bin kết hợp H,S,V và L2 toàn vector.

Chọn lại ví dụ HSV sau rà 4.489 histogram train thuộc hai lớp và xem 24 ứng viên RGB/H/S/V. Chi tiết trong `hsv_example_audit/decision.md`. Giữ thang tuyệt đối H [0,179], S/V [0,255], không tăng tương phản từng ảnh. Đây là lựa chọn minh họa, không phải đánh giá phân biệt hai lớp hoặc thay đổi mô hình.
