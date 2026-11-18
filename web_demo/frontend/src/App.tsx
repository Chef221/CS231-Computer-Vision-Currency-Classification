import React, { useState } from "react";
import { Header } from "./components/Header";
import { ImageUploader } from "./components/ImageUploader";
import { PredictionStrategy } from "./components/PredictionStrategy";
import { PredictionResult } from "./components/PredictionResult";
import { CameraCapture } from "./components/CameraCapture";
import { InfoModal } from "./components/InfoModal";
import type {
  PredictionResponse,
  OutputStrategy,
} from "./types/prediction";
import {
  predictBanknote,
} from "./services/predictionService";

export const App: React.FC = () => {
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [outputStrategy, setOutputStrategy] = useState<OutputStrategy>("top3");
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState<boolean>(false);

  // Xử lý thay đổi ảnh
  const handleImageChange = (dataUrl: string | null, file: File | null) => {
    setCurrentImage(dataUrl);
    setImageFile(file);
    setPredictionResult(null);
  };

  // Xử lý chụp ảnh từ Camera
  const handleCameraCapture = (base64Image: string) => {
    setCurrentImage(base64Image);
    setImageFile(null);
    setPredictionResult(null);
  };

  // Xử lý quét liên tục thời gian thực qua Live Camera Stream
  const handleLiveFramePredict = async (frameDataUrl: string) => {
    try {
      const response = await predictBanknote({
        imageBase64: frameDataUrl,
        outputStrategy,
        deciderMode: "argmax",
      });
      setPredictionResult(response);
      setCurrentImage(frameDataUrl);
    } catch (err) {
      console.error("Lỗi quét live camera frame:", err);
    }
  };

  // Thực hiện nhận diện
  const handlePredict = async () => {
    if (!currentImage) return;

    setIsLoading(true);
    try {
      const response = await predictBanknote({
        imageFile: imageFile || undefined,
        imageBase64: currentImage,
        outputStrategy,
        deciderMode: "argmax",
      });
      setPredictionResult(response);
    } catch (error) {
      console.error("Lỗi khi thực hiện nhận diện:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Chuyển đổi chiến lược hiển thị (Top 1 vs Top 3)
  const handleSelectOutputStrategy = (strategy: OutputStrategy) => {
    setOutputStrategy(strategy);
    if (predictionResult) {
      setPredictionResult({
        ...predictionResult,
        outputStrategy: strategy,
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#f1f5f9] flex flex-col selection:bg-blue-500 selection:text-white">
      {/* 1. Header chính khớp mockup */}
      <Header />

      {/* 2. Nội dung Dashboard chính */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {/* Bố cục 2 khu vực: Bên trái ~58% Input - Bên phải ~42% Prediction */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* A. KHU VỰC BÊN TRÁI: INPUT IMAGE AREA */}
          <section className="lg:col-span-7 flex flex-col gap-6">
            <ImageUploader
              currentImage={currentImage}
              imageFile={imageFile}
              onImageChange={handleImageChange}
              onPredict={handlePredict}
              isLoading={isLoading}
              onOpenCam={() => setIsCameraOpen(true)}
              onLiveFramePredict={handleLiveFramePredict}
            />
          </section>

          {/* B. KHU VỰC BÊN PHẢI: PREDICTION AREA */}
          <section className="lg:col-span-5 flex flex-col gap-6">
            {/* Card 1: Cách hiển thị */}
            <PredictionStrategy
              outputStrategy={outputStrategy}
              onSelectOutputStrategy={handleSelectOutputStrategy}
              disabled={isLoading}
            />

            {/* Card 2: Kết quả nhận diện */}
            <PredictionResult
              result={predictionResult}
              isLoading={isLoading}
              hasImage={!!currentImage}
              currentImage={currentImage}
              outputStrategy={outputStrategy}
            />
          </section>
        </div>
      </main>

      {/* 3. Footer */}
      <footer className="bg-white border-t border-slate-200/80 py-3 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>© 2026 AI Nhận Diện Tiền Polymer Việt Nam — CS231</p>
          <p className="flex items-center gap-2">
            <span>ResNet-18 Backbone</span>
            <span>•</span>
            <span>Cost-Sensitive 2 chiều</span>
          </p>
        </div>
      </footer>

      {/* 4. Modals */}
      <CameraCapture
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCameraCapture}
      />

      <InfoModal
        isOpen={isInfoModalOpen}
        onClose={() => setIsInfoModalOpen(false)}
      />
    </div>
  );
};

export default App;
