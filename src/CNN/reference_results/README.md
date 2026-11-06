# Baseline RGB224 dùng để đối chiếu CNN

Thư mục giữ bản sao metric, dự đoán test, manifest và cấu hình KNN/SVM đã chạy trước đó. Đây là bằng chứng tham chiếu; không phải đầu ra CNN và không dùng để chọn checkpoint CNN.

- KNN: commit `fabe1b4`, HSV12, k=1, uniform, Euclidean.
- SVM: commit `6d87bc4`, HSV12+HOG, RBF C=10, gamma=scale.

`provenance.json` ghi đầy đủ commit nguồn, đường dẫn gốc và SHA-256. Byte của CSV/JSON được giữ nguyên khi sao chép. Notebook 08 kiểm tra hash, tính lại metric từ dự đoán và đối chiếu ba manifest trước khi so sánh. Các file này đi cùng Dataset code trên Kaggle; không cần gắn Notebook Output KNN/SVM riêng.

Baseline chỉ có một split và môi trường huấn luyện khác CNN. Nếu dùng bộ DATA hoặc baseline mới, phải tạo một bộ tham chiếu mới có nguồn và hash tương ứng, không đổi số trong JSON thủ công.
