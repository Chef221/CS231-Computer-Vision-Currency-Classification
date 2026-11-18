# 🇻🇳 ResNet-18 — Phân Loại Mệnh Giá Tiền Polymer Việt Nam & Web Demo

[![Live Web Demo](https://img.shields.io/badge/%F0%9F%9A%80%20LIVE%20WEB%20DEMO-VISIT%20ONLINE-00b894?style=for-the-badge&logo=googlechrome&logoColor=white)](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)
[![Evaluation Notebook](https://img.shields.io/badge/%F0%9F%93%93%20EVALUATION%20NOTEBOOK-OPEN%20RESULTS-6c5ce7?style=for-the-badge&logo=jupyter&logoColor=white)](./notebooks/resnet18_evaluation.ipynb)

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-ee4c2c?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-99.68%25-44cc11?style=flat-square)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Active-2ea44f?style=flat-square)
![ONNX Web](https://img.shields.io/badge/ONNX%20Runtime-In--Browser%20AI-005CED?style=flat-square)

> 🌟 **TRẢI NGHIỆM DỰ ÁN TRỰC TIẾP TRÊN TRÌNH DUYỆT (KHÔNG CẦN CÀI ĐẶT / TẢI VỀ):**  
> 🔗 **Web Demo Nhận diện Tiền Polymer (ResNet-18 chạy trực tiếp trên trình duyệt):** [https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/](https://thienphucng-1410.github.io/CS231-Computer-Vision-Currency-Classification/)  
> 📓 **Notebook đã chạy hoàn chỉnh (kèm biểu đồ & phân tích lỗi):** [notebooks/resnet18_evaluation.ipynb](./notebooks/resnet18_evaluation.ipynb)

---

## 📊 Kết Quả Đánh Giá Trên Tập Test Độc Lập (1.553 ảnh)

**Cấu hình khóa:** ResNet-18 (Pretrained ImageNet, **11,18 triệu** tham số), AdamW, `LR = 1e-4`, đầu vào `224x224` (không CenterCrop), `ColorJitter` với `hue = 0`.

### 1. Bảng Tổng Hợp
| Nội dung | Kết quả |
| :--- | :---: |
| **Số ảnh test** | **1.553** |
| **Dự đoán đúng** | **1.548** *(chỉ sai 5 ảnh)* |
| **Accuracy** | **99,68%** |
| **Macro-F1** | **0,9968** |
| **MAE (đ/ảnh)** | **978,75** |

### 2. Chi Tiết Từng Mệnh Giá
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

## 🏗️ Cấu Trúc Mã Nguồn & Web Demo

```text
├── src/ResNet/
│   ├── README.md                          # Tài liệu phần ResNet-18 & Web Demo
│   ├── model.py                           # Kiến trúc ResNet-18 Transfer Learning (6 lớp)
│   ├── dataset.py                         # Dataset loader & Data Augmentation (hue=0)
│   ├── eval.py                            # Script đánh giá tập test
│   ├── compare.py                         # Đối sánh các chiến lược ra quyết định
│   ├── cost_metrics.py                    # Ma trận chi phí và hàm đánh giá rủi ro
│   └── notebooks/
│       ├── resnet18_evaluation.ipynb      # Notebook đã chạy đầy đủ output & Confusion Matrix
│       └── train_resnet_kaggle.ipynb      # Notebook huấn luyện trên Kaggle GPU
│
├── web_demo/
│   ├── backend/
│   │   ├── app.py                         # Server FastAPI nhận diện thời gian thực
│   │   └── weights/best_resnet18_baseline.pth  # Checkpoint PyTorch (42.7 MB)
│   ├── frontend/                          # Mã nguồn giao diện React + Vite + TypeScript
│   └── Dockerfile                         # Hỗ trợ triển khai Docker nếu cần
│
└── docs/                                  # Bản build tĩnh cho GitHub Pages
    ├── index.html                         # Trang Web Demo chạy ONNX Runtime Web
    └── models/resnet18_vnd.onnx           # Mô hình ResNet-18 ONNX chạy trực tiếp trên trình duyệt
```

---

## 🚀 Cách Chạy Local Trên Máy Cá Nhân

```bash
# 1. Cài đặt thư viện
pip install -r src/ResNet/requirements.txt

# 2. Khởi chạy Web Demo (FastAPI + Giao diện React tích hợp sẵn)
python web_demo/backend/app.py
```
Truy cập trình duyệt tại: **[http://localhost:8080](http://localhost:8080)** (Swagger API tại `/docs`).
