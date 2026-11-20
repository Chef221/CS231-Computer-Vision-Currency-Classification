import React, { useRef, useState, useEffect, useCallback } from "react";
import { Camera, RefreshCw, X, Check, SwitchCamera, AlertCircle } from "lucide-react";

interface CameraCaptureProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (base64Image: string) => void;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({
  isOpen,
  onClose,
  onCapture,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isFlashing, setIsFlashing] = useState<boolean>(false);

  // Khởi động Camera stream
  const startCamera = useCallback(async () => {
    try {
      setCameraError(null);
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error("Lỗi truy cập Camera:", err);
      setCameraError(
        "Không thể truy cập camera. Vui lòng cấp quyền truy cập webcam trên trình duyệt hoặc sử dụng tính năng tải ảnh từ máy."
      );
    }
  }, [facingMode]);

  useEffect(() => {
    if (isOpen) {
      startCamera();
    } else {
      // Dọn dẹp stream khi đóng modal
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        setStream(null);
      }
      setCapturedImage(null);
    }
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isOpen, startCamera]);

  // Chuyển đổi camera trước/sau
  const toggleFacingMode = () => {
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
  };

  // Chụp ảnh từ video frame
  const handleTakePhoto = () => {
    if (!videoRef.current) return;

    setIsFlashing(true);
    setTimeout(() => setIsFlashing(false), 200);

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/png");
      setCapturedImage(dataUrl);
    }
  };

  // Xác nhận ảnh
  const handleConfirmPhoto = () => {
    if (capturedImage) {
      onCapture(capturedImage);
      onClose();
    }
  };

  // Chụp lại
  const handleRetake = () => {
    setCapturedImage(null);
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm animate-fadeIn">
      <div className="bg-white rounded-3xl overflow-hidden max-w-xl w-full shadow-2xl border border-slate-200 flex flex-col">
        {/* Header Modal */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-blue-600" />
            <h3 className="font-bold text-slate-800 text-base">
              {capturedImage ? "Kiểm tra ảnh chụp" : "Chụp ảnh tờ tiền Polymer"}
            </h3>
          </div>
          <button
            onClick={onClose}
            type="button"
            className="p-1.5 rounded-full text-slate-400 hover:text-slate-600 hover:bg-slate-200/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Viewfinder / Preview Body */}
        <div className="relative bg-black aspect-4/3 sm:aspect-16/9 flex items-center justify-center overflow-hidden">
          {cameraError ? (
            <div className="p-6 text-center text-white max-w-md">
              <AlertCircle className="w-12 h-12 text-rose-500 mx-auto mb-3" />
              <p className="text-sm text-slate-200 mb-4">{cameraError}</p>
              <button
                onClick={startCamera}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-xl inline-flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Thử lại
              </button>
            </div>
          ) : capturedImage ? (
            <img
              src={capturedImage}
              alt="Ảnh chụp tiền polymer"
              className="w-full h-full object-contain"
            />
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />

              {/* Khung hướng dẫn căn chỉnh tờ tiền (tỷ lệ chuẩn ~ 2.25:1) */}
              <div className="absolute inset-0 pointer-events-none flex items-center justify-center p-6">
                <div className="w-[85%] aspect-2.25/1 border-2 border-dashed border-white/80 rounded-2xl shadow-[0_0_0_9999px_rgba(0,0,0,0.45)] relative flex items-center justify-center">
                  {/* 4 góc định vị */}
                  <div className="absolute top-0 left-0 w-4 h-4 border-t-3 border-l-3 border-blue-400 -mt-1 -ml-1"></div>
                  <div className="absolute top-0 right-0 w-4 h-4 border-t-3 border-r-3 border-blue-400 -mt-1 -mr-1"></div>
                  <div className="absolute bottom-0 left-0 w-4 h-4 border-b-3 border-l-3 border-blue-400 -mb-1 -ml-1"></div>
                  <div className="absolute bottom-0 right-0 w-4 h-4 border-b-3 border-r-3 border-blue-400 -mb-1 -mr-1"></div>
                  <span className="text-xs font-medium text-white/90 bg-black/60 px-2.5 py-1 rounded-full backdrop-blur-xs">
                    Đặt trọn vẹn tờ tiền vào khung
                  </span>
                </div>
              </div>

              {/* Nút lật camera nếu có */}
              <button
                onClick={toggleFacingMode}
                type="button"
                className="absolute top-4 right-4 p-2.5 rounded-full bg-black/50 text-white hover:bg-black/70 backdrop-blur-md transition-colors"
                title="Đổi camera trước/sau"
              >
                <SwitchCamera className="w-5 h-5" />
              </button>
            </>
          )}

          {/* Flash Effect khi bấm chụp */}
          {isFlashing && (
            <div className="absolute inset-0 bg-white opacity-80 pointer-events-none transition-opacity duration-150"></div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 bg-slate-50 flex items-center justify-between border-t border-slate-100">
          <p className="text-xs text-slate-500">
            {capturedImage
              ? "Bạn có hài lòng với ảnh chụp này?"
              : "Khuyên dùng: Đủ ánh sáng, không bị gấp mép hay lóa đèn"}
          </p>

          <div className="flex items-center gap-2.5">
            {capturedImage ? (
              <>
                <button
                  onClick={handleRetake}
                  type="button"
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-700 bg-white border border-slate-300 hover:bg-slate-100 transition-colors flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Chụp lại
                </button>
                <button
                  onClick={handleConfirmPhoto}
                  type="button"
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-sm transition-colors flex items-center gap-1.5"
                >
                  <Check className="w-3.5 h-3.5" /> Sử dụng ảnh
                </button>
              </>
            ) : (
              <button
                onClick={handleTakePhoto}
                disabled={!!cameraError}
                type="button"
                className="px-5 py-2.5 rounded-xl text-xs font-bold text-white bg-blue-600 hover:bg-blue-700 shadow-md shadow-blue-500/20 disabled:opacity-50 transition-all flex items-center gap-2"
              >
                <Camera className="w-4 h-4" /> Bấm Chụp Ngay
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
