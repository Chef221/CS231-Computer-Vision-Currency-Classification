FROM python:3.11-slim

WORKDIR /app

# Cài PyTorch bản CPU (nhẹ, tải và build nhanh trên Cloud)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart Pillow numpy

# Chép mã nguồn Backend, Trọng số mô hình & Giao diện Web đã build
COPY backend/ ./backend/
COPY frontend/dist/ ./frontend/dist/
COPY frontend/public/ ./frontend/public/

# Hugging Face Spaces mặc định chạy cổng 7860 (Render tự động truyền biến môi trường PORT)
ENV PORT=7860
EXPOSE 7860

CMD ["python", "backend/app.py"]
