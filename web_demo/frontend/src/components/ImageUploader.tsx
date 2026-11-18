import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  FolderOpen,
  Trash2,
  RefreshCw,
  Sparkles,
  AlertCircle,
  Play,
  Loader2,
  Square,
  SwitchCamera,
  ScanLine,
} from "lucide-react";
import { SAMPLE_BANKNOTES } from "../utils/sampleBanknotes";
import type { SampleBanknote, TestCaseCategory } from "../utils/sampleBanknotes";

interface ImageUploaderProps {
  currentImage: string | null;
  imageFile: File | null;
  onImageChange: (dataUrl: string | null, file: File | null) => void;
  onPredict: () => void;
  isLoading: boolean;
  onOpenCam?: () => void;
  onLiveFramePredict?: (frameDataUrl: string) => Promise<void>;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  currentImage,
  imageFile,
  onImageChange,
  onPredict,
  isLoading,
  onLiveFramePredict,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<TestCaseCategory>("all");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Trạng thái Camera trực tiếp (Live Stream Scanner)
  const [isLiveMode, setIsLiveMode] = useState<boolean>(false);
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isProcessingLiveRef = useRef<boolean>(false);

  // Khởi động Camera trực tiếp
  const startLiveCamera = useCallback(async () => {
    try {
      setErrorMessage(null);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      setIsLiveMode(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera error:", err);
      setErrorMessage(
        "Không thể mở webcam. Vui lòng cấp quyền truy cập camera trong trình duyệt hoặc sử dụng tính năng tải ảnh từ thiết bị."
      );
      setIsLiveMode(false);
    }
  }, [facingMode]);

  // Dừng Camera trực tiếp
  const stopLiveCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsLiveMode(false);
  };

  // Đổi camera trước/sau
  const toggleFacingMode = () => {
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
  };

  // Tự động gán stream vào video ref khi isLiveMode bật
  useEffect(() => {
    if (isLiveMode && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [isLiveMode]);

  // Vòng lặp nhận diện thời gian thực qua Live Camera (2 FPS không nghẽn luồng)
  // Tự động cắt theo tỷ lệ chuẩn của Target Box (2.28:1) để loại bỏ nhiễu nền
  useEffect(() => {
    if (!isLiveMode) return;

    const intervalId = setInterval(async () => {
      if (!videoRef.current || isProcessingLiveRef.current || !onLiveFramePredict) return;
      const video = videoRef.current;
      if (video.readyState < 2) return; // Chưa sẵn sàng khung hình

      try {
        const vW = video.videoWidth || 1280;
        const vH = video.videoHeight || 720;

        // Tỷ lệ chuẩn Polymer trung bình: 152mm / 65mm ≈ 2.28
        const POLYMER_ASPECT = 2.28;

        // Vùng target box chiếm khoảng 82% chiều ngang video
        let boxW = vW * 0.82;
        let boxH = boxW / POLYMER_ASPECT;

        // Nếu chiều cao vượt quá 80% video thì giới hạn lại theo chiều cao
        if (boxH > vH * 0.80) {
          boxH = vH * 0.80;
          boxW = boxH * POLYMER_ASPECT;
        }

        // Tọa độ cắt căn giữa video
        const cropX = Math.max(0, (vW - boxW) / 2);
        const cropY = Math.max(0, (vH - boxH) / 2);

        // Tạo canvas kích thước chuẩn (~456x200 px) tối ưu truyền tải mạng và inference
        const canvas = document.createElement("canvas");
        canvas.width = 456;
        canvas.height = Math.round(456 / POLYMER_ASPECT);
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Cắt đúng vùng tờ tiền bên trong Target Box
        ctx.drawImage(
          video,
          cropX,
          cropY,
          boxW,
          boxH,
          0,
          0,
          canvas.width,
          canvas.height
        );
        const dataUrl = canvas.toDataURL("image/jpeg", 0.75);

        isProcessingLiveRef.current = true;
        await onLiveFramePredict(dataUrl);
      } catch (e) {
        // Bỏ qua lỗi rớt khung hình mạng tạm thời
      } finally {
        isProcessingLiveRef.current = false;
      }
    }, 500);

    return () => clearInterval(intervalId);
  }, [isLiveMode, onLiveFramePredict]);

  // Dọn dẹp stream khi unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const validateAndProcessFile = (file: File) => {
    setErrorMessage(null);
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      setErrorMessage("Định dạng file không hỗ trợ! Vui lòng chỉ tải lên ảnh JPG, PNG hoặc WEBP.");
      return;
    }

    const maxSize = 15 * 1024 * 1024;
    if (file.size > maxSize) {
      setErrorMessage("Kích thước file quá lớn (> 15MB). Vui lòng chọn file nhẹ hơn.");
      return;
    }

    if (isLiveMode) {
      stopLiveCamera();
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setSelectedSampleId(null);
      onImageChange(dataUrl, file);
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleClearImage = () => {
    setSelectedSampleId(null);
    onImageChange(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSelectSample = async (sample: SampleBanknote) => {
    if (isLiveMode) {
      stopLiveCamera();
    }
    setSelectedSampleId(sample.id);
    setErrorMessage(null);
    try {
      const res = await fetch(sample.dataUrl);
      const blob = await res.blob();
      const file = new File([blob], `${sample.denomination || "sample"}.png`, {
        type: blob.type || "image/png",
      });
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        onImageChange(dataUrl, file);
      };
      reader.readAsDataURL(blob);
    } catch {
      onImageChange(sample.dataUrl, null);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Khung Card Upload / Camera Scanner chính */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative rounded-3xl border-2 border-dashed transition-all flex flex-col items-center justify-center min-h-[460px] sm:min-h-[520px] shadow-xs overflow-hidden ${
          isLiveMode ? "p-2 sm:p-4 bg-slate-900 border-blue-400" : "p-6 sm:p-10 bg-[#f8faff] hover:border-blue-300"
        } ${
          isDragging ? "border-blue-500 bg-blue-50/60 scale-[0.99]" : "border-blue-200/90"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/jpg"
          className="hidden"
          onChange={handleFileInputChange}
        />

        {/* Thông báo lỗi nếu có */}
        {errorMessage && (
          <div className="w-full mb-4 p-3 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2 z-20">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* TRƯỜNG HỢP 1: ĐANG BẬT LIVE CAMERA SCANNER */}
        {isLiveMode ? (
          <div className="w-full h-full flex flex-col items-center relative">
            {/* Viewport Video Live chiếm trọn chiều rộng */}
            <div className="relative w-full aspect-16/10 sm:aspect-16/9 min-h-[420px] sm:min-h-[480px] rounded-2xl overflow-hidden bg-slate-950 border-2 border-blue-500 shadow-xl flex items-center justify-center">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />

              {/* Lớp phủ HUD Scan thời gian thực */}
              <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-4 sm:p-5 z-10">
                {/* Header HUD */}
                <div className="flex items-center justify-between">
                  <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-white text-xs font-bold border border-red-500/40 shadow-sm">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping"></span>
                    <span>LIVE SCANNER</span>
                  </div>
                  <span className="text-xs text-white/90 bg-black/50 px-2.5 py-1 rounded-lg font-mono backdrop-blur-xs">
                    2 FPS • Tự Động Quét
                  </span>
                </div>

                {/* Khung ngắm chuẩn kích thước & tỷ lệ Polymer Việt Nam (152 × 65 mm ~ 2.28:1) */}
                <div className="w-[92%] sm:w-[86%] max-w-[620px] aspect-[2.28/1] mx-auto relative rounded-2xl border-2 border-emerald-400 shadow-[0_0_25px_rgba(52,211,153,0.25)] bg-transparent overflow-hidden select-none pointer-events-none">
                  {/* 4 Góc định vị phát sáng chuẩn tờ tiền */}
                  <div className="absolute -top-1.5 -left-1.5 w-7 h-7 border-t-3 border-l-3 border-emerald-400 rounded-tl-xl shadow-[0_0_8px_#34d399]"></div>
                  <div className="absolute -top-1.5 -right-1.5 w-7 h-7 border-t-3 border-r-3 border-emerald-400 rounded-tr-xl shadow-[0_0_8px_#34d399]"></div>
                  <div className="absolute -bottom-1.5 -left-1.5 w-7 h-7 border-b-3 border-l-3 border-emerald-400 rounded-bl-xl shadow-[0_0_8px_#34d399]"></div>
                  <div className="absolute -bottom-1.5 -right-1.5 w-7 h-7 border-b-3 border-r-3 border-emerald-400 rounded-br-xl shadow-[0_0_8px_#34d399]"></div>

                  {/* Tâm ngắm chữ thập trung tâm mờ nhẹ */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="relative w-8 h-8 flex items-center justify-center opacity-30">
                      <div className="w-6 h-px bg-emerald-400"></div>
                      <div className="h-6 w-px bg-emerald-400 absolute"></div>
                      <div className="w-2.5 h-2.5 rounded-full border border-emerald-400 absolute"></div>
                    </div>
                  </div>

                  {/* Tia laser quét ngang */}
                  <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-emerald-400/80 to-transparent shadow-[0_0_12px_#34d399] animate-pulse"></div>
                </div>

                {/* Khoảng trống đệm dưới khung để cân bằng */}
                <div className="h-4"></div>
              </div>

              {/* Nút thao tác nhanh trên Live Video */}
              <div className="absolute top-3 right-3 flex items-center gap-2 z-20">
                <button
                  type="button"
                  onClick={toggleFacingMode}
                  className="bg-black/60 hover:bg-black/80 text-white p-2.5 rounded-xl backdrop-blur-md transition-colors shadow-xs"
                  title="Chuyển camera trước/sau"
                >
                  <SwitchCamera className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Thanh nút bấm điều khiển Live Camera */}
            <div className="flex items-center gap-3 mt-4">
              <button
                type="button"
                onClick={stopLiveCamera}
                className="px-6 py-2.5 rounded-2xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-md transition-all flex items-center gap-2"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Dừng Camera (Quay lại)</span>
              </button>
            </div>
          </div>
        ) : !currentImage ? (
          /* TRƯỜNG HỢP 2: CHƯA CÓ ẢNH VÀ CHƯA BẬT CAMERA */
          <div className="flex flex-col items-center text-center max-w-md w-full">
            {/* Biểu tượng Camera tròn màu xanh nhạt với tia phát sáng */}
            <div className="relative mb-5">
              <div className="w-24 h-24 rounded-full bg-[#dbeafe] flex items-center justify-center text-[#1d4ed8]">
                <div className="w-13 h-10 bg-[#1d4ed8] rounded-xl flex items-center justify-center relative shadow-sm">
                  <div className="absolute -top-1.5 left-3 w-4 h-1.5 bg-[#1d4ed8] rounded-t-sm"></div>
                  <div className="w-5 h-5 rounded-full border-2 border-white bg-[#1d4ed8]"></div>
                </div>
              </div>

              <div className="absolute -top-1 -right-2 text-[#1d4ed8]">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2L13.5 7.5L19 9L13.5 10.5L12 16L10.5 10.5L5 9L10.5 7.5L12 2Z" />
                </svg>
              </div>
            </div>

            {/* Tiêu đề & phụ đề */}
            <h2 className="text-xl sm:text-2xl font-extrabold text-[#0f172a] tracking-tight mb-1.5">
              Tải ảnh lên hoặc quét camera trực tiếp
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 font-medium mb-6">
              Hỗ trợ định dạng: JPG, PNG, WEBP hoặc đưa tiền trước webcam
            </p>

            {/* Dòng ngăn cách: ──── hoặc ──── */}
            <div className="w-64 flex items-center justify-center gap-3 mb-6 text-xs text-slate-400">
              <div className="flex-1 h-px bg-slate-200"></div>
              <span>hoặc</span>
              <div className="flex-1 h-px bg-slate-200"></div>
            </div>

            {/* Hai nút hành động: Camera trực tiếp & Chọn ảnh */}
            <div className="flex flex-wrap items-center justify-center gap-3 w-full">
              <button
                type="button"
                onClick={startLiveCamera}
                className="px-6 py-2.5 rounded-2xl bg-[#2563eb] hover:bg-blue-700 text-white text-sm font-semibold shadow-sm transition-all flex items-center gap-2 group"
              >
                <ScanLine className="w-4 h-4 group-hover:scale-110 transition-transform" />
                <span>Camera trực tiếp</span>
                <span className="px-1.5 py-0.5 rounded-full bg-red-500 text-[9px] font-extrabold text-white animate-pulse">
                  LIVE
                </span>
              </button>

              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-2.5 rounded-2xl bg-white hover:bg-slate-50 text-[#2563eb] border border-blue-400 text-sm font-semibold shadow-xs transition-all flex items-center gap-2"
              >
                <FolderOpen className="w-4 h-4" />
                <span>Chọn ảnh</span>
              </button>
            </div>
          </div>
        ) : (
          /* TRƯỜNG HỢP 3: ĐÃ CHỌN ẢNH TĨNH */
          <div className="w-full flex flex-col items-center">
            {/* Ảnh xem trước to rõ */}
            <div className="relative w-full min-h-[380px] max-h-[480px] sm:max-h-[540px] rounded-2xl overflow-hidden bg-slate-900 border border-slate-200 shadow-md flex items-center justify-center">
              <img
                src={currentImage}
                alt="Tiền polymer xem trước"
                className="w-full h-full max-h-[480px] sm:max-h-[540px] object-contain"
              />

              {/* Nút thao tác trên góc ảnh */}
              <div className="absolute top-3 right-3 flex items-center gap-2">
                <button
                  onClick={startLiveCamera}
                  type="button"
                  className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-3 py-1.5 rounded-xl flex items-center gap-1.5 transition-all shadow-xs"
                  title="Mở camera trực tiếp"
                >
                  <ScanLine className="w-3.5 h-3.5" />
                  <span>Quét Live</span>
                </button>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  type="button"
                  className="bg-black/60 hover:bg-black/80 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded-xl flex items-center gap-1.5 transition-all"
                  title="Chọn ảnh khác từ máy"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Đổi ảnh</span>
                </button>

                <button
                  onClick={handleClearImage}
                  type="button"
                  className="bg-rose-600/80 hover:bg-rose-700 text-white text-xs px-3 py-1.5 rounded-xl flex items-center gap-1.5 transition-all"
                  title="Xóa ảnh hiện tại"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Xóa</span>
                </button>
              </div>

              {imageFile && (
                <div className="absolute bottom-3 left-3 bg-black/60 backdrop-blur-md text-slate-200 text-xs px-3 py-1 rounded-xl">
                  {imageFile.name} ({(imageFile.size / 1024).toFixed(0)} KB)
                </div>
              )}
            </div>

            {/* Nút bấm Nhận diện sau khi có ảnh */}
            <button
              onClick={onPredict}
              disabled={isLoading}
              type="button"
              className="mt-5 w-full sm:w-auto px-8 py-3 rounded-2xl bg-[#2563eb] hover:bg-blue-700 text-white font-bold text-sm shadow-md shadow-blue-500/20 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Đang nhận diện...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Bắt Đầu Nhận Diện</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Bộ sưu tập các kịch bản thử nghiệm thực tế: Gần nhau, Khó, Dễ, Ngoại cảnh */}
      <div className="bg-white rounded-3xl p-5 border border-slate-200/90 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-3.5">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800">
                Kịch bản thử nghiệm thực tế
              </h3>
              <p className="text-[11px] text-slate-400">
                Ảnh chuẩn, mệnh giá dễ nhầm, ảnh khó và dữ liệu ngoại cảnh
              </p>
            </div>
          </div>

          {/* Bộ lọc phân loại kịch bản */}
          <div className="flex items-center gap-1 overflow-x-auto pb-1 sm:pb-0 text-xs">
            {[
              { key: "all", label: "Tất cả (8)" },
              { key: "easy", label: "Ảnh dễ" },
              { key: "confusable", label: "Gần nhau" },
              { key: "hard", label: "Ảnh khó" },
              { key: "wild", label: "Ngoại cảnh/OOD" },
            ].map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setSelectedCategory(tab.key as TestCaseCategory)}
                className={`px-2.5 py-1 rounded-lg font-medium transition-colors shrink-0 ${
                  selectedCategory === tab.key
                    ? "bg-blue-600 text-white shadow-xs"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Danh sách các card kịch bản */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {SAMPLE_BANKNOTES.filter(
            (s) => selectedCategory === "all" || s.category === selectedCategory
          ).map((sample) => {
            const isSelected = selectedSampleId === sample.id;
            return (
              <button
                key={sample.id}
                onClick={() => handleSelectSample(sample)}
                type="button"
                className={`p-2 rounded-2xl text-left border transition-all flex flex-col justify-between group ${
                  isSelected
                    ? "border-blue-600 bg-blue-50/90 ring-2 ring-blue-500/20 shadow-xs"
                    : "border-slate-200/80 bg-slate-50/70 hover:bg-white hover:border-blue-300 hover:shadow-xs"
                }`}
              >
                {/* Thumbnail với Badge phân loại */}
                <div className="w-full aspect-2.1/1 rounded-xl overflow-hidden bg-slate-900 mb-2 border border-slate-200/60 relative">
                  <img
                    src={sample.dataUrl}
                    alt={sample.name}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <span
                    className={`absolute top-1 left-1 px-1.5 py-0.5 rounded text-[9px] font-bold text-white shadow-xs ${
                      sample.category === "easy"
                        ? "bg-emerald-600"
                        : sample.category === "confusable"
                        ? "bg-amber-600"
                        : sample.category === "hard"
                        ? "bg-rose-600"
                        : "bg-purple-600"
                    }`}
                  >
                    {sample.categoryLabel}
                  </span>
                </div>

                {/* Tiêu đề & Tag chi tiết */}
                <div>
                  <span className="font-bold text-xs text-slate-800 block truncate group-hover:text-blue-600 transition-colors">
                    {sample.name}
                  </span>
                  <span className="text-[10px] text-blue-600 font-semibold block truncate">
                    {sample.tag}
                  </span>
                  <p className="text-[10px] text-slate-400 line-clamp-2 mt-0.5 leading-tight">
                    {sample.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
