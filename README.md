# Nhận Diện & Phân Loại Mệnh Giá Tiền Polymer Việt Nam (ResNet-18)

[![LIVE WEB DEMO](https://img.shields.io/badge/%F0%9F%9A%80%20LIVE%20WEB%20DEMO-VISIT%20ONLINE-00b894?style=for-the-badge&logo=googlechrome&logoColor=white)](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)
[![API DOCUMENTATION](https://img.shields.io/badge/%F0%9F%93%96%20API%20DOCUMENTATION-OPEN%20SWAGGER-6c5ce7?style=for-the-badge&logo=fastapi&logoColor=white)](http://localhost:8080/docs)

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20ResNet--18-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Dataset 15.611 ảnh](https://img.shields.io/badge/Dataset-15.611%20ảnh-blue?style=flat-square)
![Accuracy 99.68%](https://img.shields.io/badge/Accuracy-99.68%25-brightgreen?style=flat-square)
![GitHub Pages Active](https://img.shields.io/badge/GitHub%20Pages-Active-2ea44f?style=flat-square&logo=github&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-orange?style=flat-square)

> 🌟 **TRẢI NGHIỆM DỰ ÁN TRỰC TIẾP TRÊN TRÌNH DUYỆT (KHÔNG CẦN CÀI ĐẶT / TẢI VỀ):**  
> 🔗 **Web Demo Nhận diện Tiền Polymer (ONNX Runtime Web Client-side):** [https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)  
> 📓 **Notebook Đánh giá Hoàn chỉnh (Kèm biểu đồ ma trận nhầm lẫn & phân tích 5 ảnh lỗi):** [src/ResNet/notebooks/resnet18_evaluation.ipynb](src/ResNet/notebooks/resnet18_evaluation.ipynb)

Dự án: **Phân loại tự động 6 mệnh giá tiền giấy polymer Việt Nam (10k, 20k, 50k, 100k, 200k, 500k)** sử dụng mạng nơ-ron tích chập học phần dư **ResNet-18** (Transfer Learning từ ImageNet) và triển khai hệ thống demo nhận diện thời gian thực trên nền web. Đồ án môn học **CS231 — Thị giác Máy tính** (Trường Đại học Công nghệ Thông tin — ĐHQG-HCM).

---

## 📊 Kết Quả Thực Nghiệm Trên Tập Test Độc Lập (1.553 ảnh)

Mô hình đạt độ chính xác ấn tượng **99,68%**, chỉ nhầm lẫn đúng **5 ảnh** trên tổng số 1.553 ảnh kiểm thử thực tế.

### 1. Bảng Tổng Hợp Chỉ Số Toàn Cục
| Chỉ số đánh giá | Giá trị đạt được | Ghi chú kỹ thuật |
| :--- | :---: | :--- |
| **Tổng số ảnh Test** | **1.553** | Tập kiểm thử độc lập hoàn toàn với quá trình huấn luyện |
| **Dự đoán chính xác** | **1.548 / 1.553** | Tỷ lệ nhận diện đúng vượt trội |
| **Accuracy** | **99,68%** | So với SVM (92,9%) và CNN baseline (97,4%) |
| **Macro-F1 Score** | **0,9968** | Cân bằng hoàn hảo trên toàn bộ 6 mệnh giá |
| **Sai số tiền tệ (MAE)** | **978,75 đ/ảnh** | Mức độ rủi ro tài chính cực kỳ thấp |

### 2. Chi Tiết Hiệu Năng Theo Từng Mệnh Giá (Confusion Matrix Analysis)
| Mệnh giá | Số lượng test | Nhận diện đúng | Precision | Recall | F1-score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10.000đ** | 281 | 281 / 281 | 99,65% | **100,00%** | **99,82%** |
| **20.000đ** | 224 | 224 / 224 | 99,56% | **100,00%** | **99,78%** |
| **50.000đ** | 223 | 222 / 223 | **100,00%** | 99,55% | **99,78%** |
| **100.000đ** | 247 | 247 / 247 | 99,60% | **100,00%** | **99,80%** |
| **200.000đ** | 317 | 316 / 317 | 99,68% | 99,68% | **99,68%** |
| **500.000đ** | 261 | 258 / 261 | 99,61% | 98,85% | **99,23%** |
| **Toàn bộ tập Test** | **1.553** | **1.548 / 1.553** | **99,68%** | **99,68%** | **99,68%** |

*Đặc biệt: 4/6 mệnh giá (10k, 20k, 100k, 50k Precision) đạt độ nhạy và độ đặc hiệu tuyệt đối 100%.*

---

## 🏗️ Cấu Trúc Mã Nguồn Nhánh `ResNet-+-demo`

```text
├── src/ResNet/                             # Mã nguồn huấn luyện & đánh giá ResNet-18
│   ├── model.py                            # Kiến trúc ResNet-18 Transfer Learning (FC 512 -> 6)
│   ├── dataset.py                          # DataLoader, Resize 224x224 & Augmentation (ColorJitter hue=0)
│   ├── eval.py                             # Script tính toán ma trận nhầm lẫn & chỉ số tài chính MAE
│   ├── compare.py                          # So sánh chiến lược ra quyết định Argmax vs Bayes Risk
│   ├── cost_metrics.py                     # Định nghĩa ma trận chi phí sai số tiền tệ (Cost Matrix)
│   ├── requirements.txt                    # Thư viện môi trường Python
│   └── notebooks/
│       ├── resnet18_evaluation.ipynb       # Notebook phân tích trực quan kết quả thực nghiệm
│       └── train_resnet_kaggle.ipynb       # Notebook quy trình huấn luyện trên Kaggle GPU T4
│
├── web_demo/                               # Ứng dụng Full-Stack Web Demo
│   ├── backend/
│   │   ├── app.py                          # Server FastAPI nhận diện ảnh qua REST API
│   │   └── weights/best_resnet18_baseline.pth  # Trọng số mô hình PyTorch (42,7 MB)
│   ├── frontend/                           # Giao diện hiện đại (React + Vite + Tailwind/CSS)
│   └── Dockerfile                          # Đóng gói container triển khai Production
│
└── docs/                                   # Bản build tĩnh phục vụ GitHub Pages (Serverless)
    ├── index.html                          # Giao diện chạy ONNX Runtime Web trực tiếp trên client
    └── models/resnet18_vnd.onnx            # Mô hình ResNet-18 định dạng ONNX (42,6 MB)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Local

### 1. Tải nhánh và cài đặt môi trường
```bash
# Clone riêng nhánh ResNet + Web Demo
git clone -b ResNet-+-demo --single-branch https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification.git
cd CS231-Computer-Vision-Currency-Classification

# Cài đặt thư viện phụ thuộc
pip install -r src/ResNet/requirements.txt
```

### 2. Khởi chạy Web Demo trên máy tính (FastAPI + React UI)
```bash
python web_demo/backend/app.py
```
* Mở trình duyệt tại: **[http://localhost:8080](http://localhost:8080)**
* Tài liệu Swagger API: **[http://localhost:8080/docs](http://localhost:8080/docs)**

---

## 🍴 Hướng Dẫn Fork & Bật GitHub Pages Trên Repo Cá Nhân

Để các thành viên nhóm hoặc người xem có thể lưu trữ và tự động kích hoạt Web Demo trên tên miền GitHub cá nhân của mình:

1. **Fork Repository:**
   * Bấm nút **Fork** ở góc trên cùng bên phải giao diện GitHub.
   * Chọn tài khoản cá nhân của bạn (ví dụ: `ThienPhucNg-1410`).
   * Bỏ chọn *"Copy the main branch only"* để fork toàn bộ các nhánh (trong đó có `ResNet-+-demo`).

2. **Kích hoạt GitHub Pages (Web Demo chạy trực tiếp):**
   * Truy cập repository vừa fork trên tài khoản của bạn: `Settings` $\rightarrow$ `Pages`.
   * Tại mục **Build and deployment**:
     * **Source:** Chọn `Deploy from a branch`.
     * **Branch:** Chọn `ResNet-+-demo` và chọn thư mục `/docs`.
   * Bấm **Save**.
   * Chờ khoảng 1–2 phút, web demo của bạn sẽ hoạt động tại địa chỉ:  
     `https://<username>.github.io/<tên-repo>/`

---

## 👨‍💻 Tác Giả & Ghi Công

* **Môn học:** CS231 — Thị giác Máy tính (Computer Vision)
* **Giảng viên hướng dẫn:** Bộ môn Thị giác máy tính & Đa phương tiện — UIT
* **Thành viên phụ trách ResNet & Web Demo:** Nguyễn Thiện Phúc
* **Giấy phép:** [MIT License](LICENSE)
