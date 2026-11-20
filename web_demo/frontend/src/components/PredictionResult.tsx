import React from "react";
import type { PredictionResponse, OutputStrategy } from "../types/prediction";
import { CheckCircle2, BarChart2, Loader2, Inbox, Sparkles } from "lucide-react";

// Bảng ánh xạ hình ảnh chuẩn của từng mệnh giá tiền polymer
const STANDARD_DENOM_IMAGES: Record<number, string> = {
  10000: "./samples/10000_web.jpg",
  20000: "./samples/20000_web.jpg",
  50000: "./samples/50000_web.jpg",
  100000: "./samples/100000_web.jpg",
  200000: "./samples/200000_web.jpg",
  500000: "./samples/500000_web.jpg",
};

interface PredictionResultProps {
  result: PredictionResponse | null;
  isLoading: boolean;
  hasImage: boolean;
  currentImage: string | null;
  outputStrategy: OutputStrategy;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  result,
  isLoading,
  hasImage,
  outputStrategy,
}) => {
  return (
    <div className="bg-white rounded-3xl p-6 shadow-xs border border-slate-200/90 flex flex-col justify-between">
      <div>
        {/* Header: Circle xanh lá với dấu checkmark + Tiêu đề Kết quả nhận diện */}
        <div className="flex items-center gap-2.5 mb-5">
          <div className="w-6 h-6 rounded-full bg-[#16a34a] text-white flex items-center justify-center shrink-0 shadow-xs">
            <CheckCircle2 className="w-4 h-4 fill-current stroke-white text-[#16a34a]" />
          </div>
          <h3 className="font-bold text-[#0f172a] text-base">
            Kết quả nhận diện
          </h3>
        </div>

        {/* Trạng thái 1: Đang Loading */}
        {isLoading && (
          <div className="py-12 text-center">
            <Loader2 className="w-10 h-10 animate-spin text-[#2563eb] mx-auto mb-3" />
            <h4 className="font-bold text-[#0f172a] text-sm mb-1">
              Đang phân tích hình ảnh...
            </h4>
            <p className="text-xs text-slate-400">
              Mô hình ResNet-18 đang tính toán vector xác suất
            </p>
          </div>
        )}

        {/* Trạng thái 2: Chưa có kết quả (Empty State) */}
        {!isLoading && !result && (
          <div className="py-10 text-center border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/50">
            <div className="w-10 h-10 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-2.5">
              <Inbox className="w-5 h-5 stroke-[1.8]" />
            </div>
            <h4 className="font-bold text-slate-700 text-sm mb-1">
              Chưa có kết quả nhận diện
            </h4>
            <p className="text-xs text-slate-400 max-w-xs mx-auto">
              {hasImage
                ? "Nhấn nút 'Bắt Đầu Nhận Diện' ở bên trái để xem kết quả."
                : "Vui lòng tải ảnh lên hoặc chọn ảnh mẫu để bắt đầu."}
            </p>
          </div>
        )}

        {/* Trạng thái 3: Đã có kết quả nhận diện — Khớp 100% hình mẫu mockup */}
        {!isLoading && result && (
          <div className="space-y-6">
            {/* Hàng trên: Ảnh tờ tiền bên trái + Mệnh giá dự đoán bên phải */}
            <div className="flex items-center gap-4 sm:gap-6">
              {/* Hình ảnh chuẩn của mệnh giá dự đoán */}
              <div className="w-36 sm:w-44 aspect-2.1/1 rounded-xl overflow-hidden bg-slate-100 border border-slate-200/80 shadow-xs shrink-0 relative group">
                <img
                  src={
                    STANDARD_DENOM_IMAGES[result.predictedLabel] ||
                    "./samples/500000_web.jpg"
                  }
                  alt={`Mẫu chuẩn ${result.predictedName}`}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-1 right-1 bg-black/60 backdrop-blur-xs text-[9px] text-white px-1.5 py-0.5 rounded font-medium">
                  Mẫu chuẩn
                </div>
              </div>

              {/* Mệnh giá dự đoán & Badge độ tin cậy */}
              <div className="flex-1">
                <span className="text-xs text-slate-500 font-medium block">
                  Mệnh giá dự đoán
                </span>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-[#0f172a] tracking-tight mt-0.5">
                  {result.predictedName}
                </h2>

                {/* Badge Độ tin cậy màu xanh lá đúng mockup */}
                <div className="mt-2 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#dcfce7] text-[#16a34a] text-xs font-bold border border-green-200">
                  <BarChart2 className="w-3.5 h-3.5 stroke-[2.5]" />
                  <span>Độ tin cậy: {(result.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Hàng dưới: Xác suất các mệnh giá (Top 3) */}
            {outputStrategy === "top3" ? (
              <div className="pt-4 border-t border-slate-100">
                {/* Tiêu đề mục Top 3 */}
                <h4 className="text-sm font-bold text-[#0f172a] flex items-center gap-1.5 mb-3.5">
                  <BarChart2 className="w-4 h-4 text-[#2563eb]" />
                  <span>Xác suất các mệnh giá (Top 3)</span>
                </h4>

                {/* Danh sách 3 thanh tiến trình ngang */}
                <div className="space-y-3">
                  {result.top3Probabilities.map((item, idx) => {
                    const percent = (item.probability * 100).toFixed(1);
                    const isTop1 = idx === 0;

                    return (
                      <div
                        key={item.denomination}
                        className="flex items-center justify-between gap-3 text-xs"
                      >
                        {/* Tên mệnh giá */}
                        <span className="font-semibold text-slate-700 w-28 shrink-0">
                          {item.name}
                        </span>

                        {/* Thanh Bar màu xanh dương */}
                        <div className="flex-1 bg-slate-100 h-2.5 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ease-out ${
                              isTop1 ? "bg-[#2563eb]" : "bg-[#60a5fa]"
                            }`}
                            style={{
                              width: `${Math.max(item.probability * 100, 1.5)}%`,
                            }}
                          ></div>
                        </div>

                        {/* Số phần trăm */}
                        <span className="font-mono font-bold text-slate-800 w-12 text-right shrink-0">
                          {percent}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              /* Nếu chọn Chiến lược 1: Phân loại trực tiếp */
              <div className="pt-4 border-t border-slate-100">
                <div className="p-3 rounded-2xl bg-blue-50/60 border border-blue-200 text-xs text-blue-900 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-[#2563eb] shrink-0" />
                  <span>
                    Chế độ phân loại trực tiếp: Trả về duy nhất 1 nhãn mệnh giá có độ tin cậy cao nhất (<strong>{result.predictedName}</strong>).
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer nhỏ ghi chú */}
      {result && (
        <div className="mt-5 pt-3 border-t border-slate-100 text-[11px] text-slate-400 flex items-center justify-between">
          <span className="font-medium text-slate-500">Mô hình: ResNet-18 Baseline (Độ chính xác: 99.68%)</span>
          <span className="font-mono">Thời gian xử lý: {result.processingTimeMs} ms</span>
        </div>
      )}
    </div>
  );
};
