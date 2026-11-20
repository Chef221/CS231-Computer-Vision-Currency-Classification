import os
import io
import time
import base64
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 1. Danh sách mệnh giá chuẩn của dự án CS231
DENOMINATIONS = [10000, 20000, 50000, 100000, 200000, 500000]

DENOMINATION_NAMES = {
    10000: "10.000 VNĐ",
    20000: "20.000 VNĐ",
    50000: "50.000 VNĐ",
    100000: "100.000 VNĐ",
    200000: "200.000 VNĐ",
    500000: "500.000 VNĐ",
}

# 2. Ma trận chi phí đối xứng chuẩn hóa C_2D (Elkan 2001)
C_2D = np.array([
    [0.00, 0.44, 0.69, 0.98, 1.38, 2.19],
    [0.44, 0.00, 0.69, 0.98, 1.38, 2.19],
    [0.69, 0.69, 0.00, 0.98, 1.38, 2.19],
    [0.98, 0.98, 0.98, 0.00, 1.38, 2.19],
    [1.38, 1.38, 1.38, 1.38, 0.00, 2.19],
    [2.19, 2.19, 2.19, 2.19, 2.19, 0.00]
], dtype=np.float32)

# 3. Tiền xử lý chuẩn thực nghiệm (Resize trực tiếp, tuyệt đối không CenterCrop)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Tải mô hình ResNet-18 Checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(DENOMINATIONS))
    return model

model = create_model().to(device)

# Tự động tìm checkpoint thông minh
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

POSSIBLE_CHECKPOINTS = [
    os.path.join(BASE_DIR, "weights", "best_resnet18_baseline.pth"),
    os.path.join(PROJECT_DIR, "experiments", "kaggle", "output", "best_resnet18_baseline.pth"),
    os.path.join(BASE_DIR, "best_resnet18_baseline.pth"),
]

CHECKPOINT_PATH = next((p for p in POSSIBLE_CHECKPOINTS if os.path.exists(p)), POSSIBLE_CHECKPOINTS[0])

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    print(f"[*] Da load thanh cong checkpoint: {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"ResNet checkpoint not found: {CHECKPOINT_PATH}")

model.eval()

# 5. Khởi tạo FastAPI App
app = FastAPI(
    title="AI Nhận Diện Tiền Polymer Việt Nam - ResNet-18 Baseline",
    description="API phân loại 6 mệnh giá tiền Polymer bằng mô hình ResNet-18 Baseline (Độ chính xác 99.68%)",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model": "ResNet-18 Baseline (Standard 224×224)",
        "accuracy": "99.68%",
        "classes": DENOMINATIONS,
        "checkpoint_loaded": os.path.exists(CHECKPOINT_PATH)
    }

@app.post("/api/predict")
async def predict(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    strategy: str = Form("argmax")
):
    start_time = time.perf_counter()

    # Đọc ảnh từ file upload hoặc chuỗi base64
    if image is not None:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    elif image_base64 is not None:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        decoded = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(decoded)).convert("RGB")
    else:
        return {"error": "Vui lòng cung cấp file ảnh hoặc base64 image!"}

    # Tiền xử lý & Forward pass
    tensor = eval_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    argmax_idx = int(np.argmax(probs))
    predicted_denom = DENOMINATIONS[argmax_idx]

    probabilities_list = []
    for i, denom in enumerate(DENOMINATIONS):
        probabilities_list.append({
            "denomination": denom,
            "label": str(denom),
            "name": DENOMINATION_NAMES[denom],
            "probability": float(probs[i]),
            "expectedRisk": 0.0,
            "isPredicted": i == argmax_idx
        })

    end_time = time.perf_counter()
    processing_time_ms = int((end_time - start_time) * 1000)

    return {
        "predictedLabel": predicted_denom,
        "predictedName": DENOMINATION_NAMES[predicted_denom],
        "strategyUsed": "argmax",
        "confidence": float(probs[argmax_idx]),
        "expectedRisk": 0.0,
        "probabilities": probabilities_list,
        "decisionComparison": {
            "argmaxChoice": DENOMINATIONS[argmax_idx],
            "bayesChoice": DENOMINATIONS[argmax_idx],
            "isIdentical": True,
            "differenceNote": "Mô hình chuẩn ResNet-18 Baseline (Argmax)"
        },
        "processingTimeMs": processing_time_ms,
        "modelInfo": "ResNet-18 Baseline (Acc 99.68%)"
    }

# Phục vụ ảnh mẫu thực tế từ thư mục samples
SAMPLES_DIR = os.path.join(PROJECT_DIR, "frontend", "public", "samples")
if os.path.exists(SAMPLES_DIR):
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

TEST_CASES_DIR = os.path.join(PROJECT_DIR, "frontend", "public", "test_cases")
if os.path.exists(TEST_CASES_DIR):
    app.mount("/test_cases", StaticFiles(directory=TEST_CASES_DIR), name="test_cases")

# Phục vụ static web (ưu tiên frontend/dist, tự động fallback sang docs/ ở thư mục gốc repo)
REPO_ROOT = os.path.dirname(PROJECT_DIR)
DIST_DIR = os.path.join(PROJECT_DIR, "frontend", "dist")
if not os.path.exists(DIST_DIR):
    DIST_DIR = os.path.join(REPO_ROOT, "docs")
if os.path.exists(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        static_root = Path(DIST_DIR).resolve()
        file_path = (static_root / full_path).resolve()
        if not file_path.is_relative_to(static_root):
            raise HTTPException(status_code=404)
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print("\n" + "="*70)
    print("  AI NHAN DIEN TIEN POLYMER VIET NAM - BACKEND & WEB DEMO")
    print(f"  Host: http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
