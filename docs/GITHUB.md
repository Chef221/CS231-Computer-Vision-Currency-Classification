# Đưa mã nguồn lên GitHub

Repository đã giữ code, notebook, bảng số liệu và hình cần đọc. `.gitignore` bỏ ảnh DATA, model lớn, ma trận đặc trưng, môi trường và bản sao lưu riêng.

## Dùng Git

Tại thư mục gốc dự án, xem thay đổi trước:

```bash
git status --short
git diff --stat
```

Khi đã kiểm tra nội dung, tự tạo commit và push lên nhánh của nhóm:

```bash
git add README.md requirements.txt .gitignore docs src tests tools
git commit -m "Simplify SVM and CNN code and add final results"
git push
```

Phần lớn file bị xóa là báo cáo/hình trùng, tài liệu kiểm tra cũ và kết quả CNN cũ đã được thay bằng bộ TensorFlow cuối. Bản sao trước khi dọn vẫn nằm trong `.local_backup/github_cleanup_2026-11-16` trên máy, được Git bỏ qua. Code M0/KNN và dữ liệu không thay đổi.

## Nếu chuyển source bằng ZIP

```bash
python tools/build_bundle.py --kind github
```

Gói `output/CS231_GITHUB_CODE.zip` có cấu trúc repo và kết quả nhỏ. Giải nén trước khi thêm các thư mục vào repository; không chỉ upload một ZIP nếu muốn người khác đọc code và notebook trực tiếp trên GitHub. Gói không chứa `.git`, ảnh DATA, checkpoint hoặc feature cache.

Muốn chạy lại sau khi clone, tải bộ DATA và làm theo README từng thuật toán. File model lớn giữ trong Output Kaggle hoặc một nơi lưu riêng của nhóm; không có model vẫn xem được bảng và hình đã tích hợp.
