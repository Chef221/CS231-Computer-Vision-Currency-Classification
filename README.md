# 🇻🇳 ResNet-18 & Web Demo — Phân Loại Mệnh Giá Tiền Polymer Việt Nam

[![Live Web Demo](https://img.shields.io/badge/%F0%9F%9A%80%20LIVE%20WEB%20DEMO-VISIT%20ONLINE-00b894?style=for-the-badge&logo=googlechrome&logoColor=white)](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)
[![Evaluation Notebook](https://img.shields.io/badge/%F0%9F%93%93%20EVALUATION%20NOTEBOOK-OPEN%20VIEWER-6c5ce7?style=for-the-badge&logo=jupyter&logoColor=white)](src/ResNet/notebooks/resnet18_evaluation.ipynb)

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-ee4c2c?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-99.68%25-44cc11?style=flat-square)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Active-2ea44f?style=flat-square)
![ONNX Web](https://img.shields.io/badge/ONNX%20Runtime-In--Browser%20AI-005CED?style=flat-square)

> 🌟 **TRẢI NGHIỆM DỰ ÁN TRỰC TIẾP TRÊN TRÌNH DUYỆT (KHÔNG CẦN CÀI ĐẶT / TẢI VỀ):**  
> 🔗 **Web Demo Nhận diện Tiền Polymer (ResNet-18 ONNX):** [https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)  
> 📓 **Notebook đã chạy hoàn chỉnh (kèm biểu đồ & phân tích lỗi):** [src/ResNet/notebooks/resnet18_evaluation.ipynb](src/ResNet/notebooks/resnet18_evaluation.ipynb)

Nhánh **`ResNet-+-demo`** chứa toàn bộ mã nguồn huấn luyện/đánh giá mô hình học sâu **ResNet-18** (Transfer Learning từ ImageNet) và ứng dụng **Web Demo nhận diện 6 mệnh giá tiền polymer Việt Nam** (10.000đ, 20.000đ, 50.000đ, 100.000đ, 200.000đ, 500.000đ) thuộc đồ án môn học **CS231 — Thị giác Máy tính (UIT)**.

---

## 📊 Kết Quả Đánh Giá Trên Tập Test (1.553 ảnh độc lập)

**Cấu hình khóa:** ResNet-18 (Pretrained ImageNet, **11,18 triệu tham số**), AdamW, `LR = 1e-4`, ảnh đầu vào `224x224` (không CenterCrop), `ColorJitter` với `hue = 0`.

### 1. Chỉ số tổng thể
| Nội dung | Kết quả |
| :--- | :---: |
| **Số ảnh test** | **1.553** |
| **Dự đoán đúng** | **1.548** *(chỉ sai 5 ảnh)* |
| **Accuracy** | **99,68%** |
| **Macro-F1** | **0,9968** |
| **MAE (đ/ảnh)** | **978,75** |

### 2. Chi tiết hiệu năng từng mệnh giá
| Mệnh giá | Precision | Recall | F1-score | Đúng / Tổng |
| :--- | :---: | :---: | :---: | :---: |
| **10.000đ** | 99,65% | **100,0%** | 99,82% | 281 / 281 |
| **20.000đ** | 99,56% | **100,0%** | 99,78% | 224 / 224 |
| **50.000đ** | **100,0%** | 99,55% | 99,78% | 222 / 223 |
| **100.000đ** | 99,60% | **100,0%** | 99,80% | 247 / 247 |
| **200.000đ** | 99,68% | 99,68% | 99,68% | 316 / 317 |
| **500.000đ** | 99,61% | 98,85% | 99,23% | 258 / 261 |
| **Trung bình** | **99,68%** | **99,68%** | **99,68%** | **1.548 / 1.553** |

---

## 🏗️ Cấu Trúc Nội Dung Nhánh `ResNet-+-demo`

```text
├── src/ResNet/                             # Mã nguồn cốt lõi mô hình ResNet-18
│   ├── model.py                            # Kiến trúc ResNet-18 Transfer Learning (6 lớp)
│   ├── dataset.py                          # Dataset loader & Data Augmentation (hue=0)
│   ├── eval.py                             # Script đánh giá trên tập test
│   ├── compare.py                          # So sánh chiến lược Argmax vs Bayes Decider
│   ├── cost_metrics.py                     # Ma trận chi phí đối xứng & đánh giá rủi ro
│   ├── requirements.txt                    # Thư viện Python cần thiết
│   └── notebooks/
│       ├── resnet18_evaluation.ipynb       # Notebook đã chạy đầy đủ output & Confusion Matrix
│       └── train_resnet_kaggle.ipynb       # Notebook huấn luyện trên GPU Kaggle
│
├── web_demo/                               # Ứng dụng Web Demo (Full-Stack)
│   ├── backend/
│   │   ├── app.py                          # Server FastAPI nhận diện thời gian thực
│   │   └── weights/best_resnet18_baseline.pth  # Checkpoint PyTorch (42,7 MB)
│   ├── frontend/                           # Mã nguồn giao diện React + Vite + TypeScript
│   └── Dockerfile                          # Cấu hình Docker triển khai Cloud (nếu cần)
│
└── docs/                                   # Bản build tĩnh host trực tiếp trên GitHub Pages
    ├── index.html                          # Giao diện Web Demo chạy ONNX Runtime Web
    └── models/resnet18_vnd.onnx            # Mô hình ResNet-18 ONNX chạy ngay trên trình duyệt
```

---

## 🚀 Hướng Dẫn Sử Dụng & Khởi Chạy Local

### 1. Cài đặt môi trường Python
```bash
git clone -b ResNet-+-demo --single-branch https://github.com/Chef221/CS231-Computer-Vision-Currency-Classification.git
cd CS231-Computer-Vision-Currency-Classification
pip install -r src/ResNet/requirements.txt
```

### 2. Khởi chạy Web Demo trên máy cá nhân (1 câu lệnh)
```bash
python web_demo/backend/app.py
```
* **Giao diện Web:** [http://localhost:8080](http://localhost:8080)
* **Tài liệu API (Swagger UI):** [http://localhost:8080/docs](http://localhost:8080/docs)

### 3. Xem Notebook kết quả thực nghiệm
```bash
jupyter notebook src/ResNet/notebooks/resnet18_evaluation.ipynb
```
