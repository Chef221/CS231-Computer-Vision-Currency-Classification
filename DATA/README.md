# Bộ dữ liệu cuối

Bộ dữ liệu gồm 15.611 ảnh PNG của sáu mệnh giá, được chia thành 12.494 ảnh train, 1.564 ảnh dev và 1.553 ảnh test. Mỗi ảnh nằm trong thư mục split/mệnh giá; tên file là `<mệnh_giá>_<số thứ tự sáu chữ số>.png`.

Tải đủ bốn ZIP trong GitHub Release `m0-dataset-2026-11-01` và giải nén vào thư mục gốc dự án. `metadata/dataset.csv` nằm trong ZIP dev; đường dẫn trong CSV tương đối với `DATA`.

Ảnh là crop PNG không resize. Quy trình crop thêm padding 20% mỗi phía; loại trùng pixel hoàn toàn trong cùng lớp; chia theo nhóm tên ảnh nguồn. Xem `README.md` để biết bảng số lượng, thứ tự notebook và ghi công bốn nguồn CC BY 4.0.
