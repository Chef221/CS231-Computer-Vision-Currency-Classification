# Đối chiếu tài liệu kỹ thuật — 05/11/2026

Tất cả mục dưới đây là tài liệu phần mềm chính thức, trạng thái **software-doc**, không được mô tả như bằng chứng từ bài báo thực nghiệm trên dataset của nhóm. Không có trích dẫn bài báo mới cần khóa DOI trong phần báo cáo này.

| Mục | Nguồn đã mở | Mệnh đề được hỗ trợ | Quan hệ / giới hạn |
|---|---|---|---|
| [1] | https://scikit-learn.org/1.6/modules/neighbors.html | KNN phân loại theo hàng xóm; uniform/distance và tìm trực tiếp | software; phiên bản 1.6.1 tương ứng môi trường; không chứng minh 20 cấu hình là tối ưu |
| [2] | https://docs.opencv.org/4.12.0/d6/dc7/group__imgproc__hist.html | API `calcHist` tính histogram theo kênh/bin/ranges | software; 12 bin, L2 và cách nối do code dự án quy định |
| [3] | https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog | HOG, orientations/cell/block/L2-Hys/transform_sqrt | software; trang được đọc là 0.26.0; cấu hình cụ thể của nhóm lấy từ notebook 03 |
| [4] | https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html | PCA fit phép chiếu; ngưỡng phương sai với full solver; whitening | software; không nói 95% phương sai là 95% Accuracy; số chiều 1.418 từ evidence thực tế |

Không dùng các tài liệu API làm dẫn chứng cho “HSV hơn HOG”, nguyên nhân lỗi nền, ý nghĩa thống kê hoặc khả năng tổng quát. Những nhận định đó được giới hạn theo dữ liệu và ghi rõ mức chắc chắn trong bản thảo.
