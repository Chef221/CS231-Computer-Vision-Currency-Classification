import React from "react";
import type { OutputStrategy } from "../types/prediction";
import { Settings, HelpCircle } from "lucide-react";

interface PredictionStrategyProps {
  outputStrategy: OutputStrategy;
  onSelectOutputStrategy: (strategy: OutputStrategy) => void;
  disabled?: boolean;
}

export const PredictionStrategy: React.FC<PredictionStrategyProps> = ({
  outputStrategy,
  onSelectOutputStrategy,
  disabled = false,
}) => {
  return (
    <div className="bg-white rounded-3xl p-6 shadow-xs border border-slate-200/90">
      {/* Tiêu đề Card: Icon bánh răng xanh + Tiêu đề + Icon (?) */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <Settings className="w-5 h-5 text-[#2563eb]" />
          <h3 className="font-bold text-[#0f172a] text-base">
            Cách hiển thị
          </h3>
        </div>

        <button
          type="button"
          className="text-slate-400 hover:text-slate-600 transition-colors"
          title="Chọn giữa phân loại trực tiếp 1 nhãn hoặc hiển thị top 3 xác suất cao nhất"
        >
          <HelpCircle className="w-5 h-5 stroke-[1.8]" />
        </button>
      </div>

      {/* 2 Lựa chọn Radio Buttons khớp hoàn toàn ảnh mẫu */}
      <div className="space-y-4">
        {/* Lựa chọn 1: Phân loại trực tiếp */}
        <label
          onClick={() => !disabled && onSelectOutputStrategy("top1")}
          className={`flex items-start gap-3 cursor-pointer select-none ${
            disabled ? "opacity-60 cursor-not-allowed" : ""
          }`}
        >
          {/* Custom Radio Button */}
          <div className="mt-0.5 shrink-0">
            <div
              className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                outputStrategy === "top1"
                  ? "border-[#2563eb] bg-white"
                  : "border-slate-300 bg-white"
              }`}
            >
              {outputStrategy === "top1" && (
                <div className="w-2.5 h-2.5 rounded-full bg-[#2563eb]"></div>
              )}
            </div>
          </div>

          <div>
            <h4 className="font-bold text-sm text-[#0f172a]">
              Phân loại trực tiếp
            </h4>
            <p className="text-xs text-slate-500 mt-0.5">
              Dự đoán mệnh giá duy nhất (10k – 500k)
            </p>
          </div>
        </label>

        {/* Lựa chọn 2: Hiển thị Top-K */}
        <label
          onClick={() => !disabled && onSelectOutputStrategy("top3")}
          className={`flex items-start gap-3 cursor-pointer select-none ${
            disabled ? "opacity-60 cursor-not-allowed" : ""
          }`}
        >
          {/* Custom Radio Button */}
          <div className="mt-0.5 shrink-0">
            <div
              className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                outputStrategy === "top3"
                  ? "border-[#2563eb] bg-white"
                  : "border-slate-300 bg-white"
              }`}
            >
              {outputStrategy === "top3" && (
                <div className="w-2.5 h-2.5 rounded-full bg-[#2563eb]"></div>
              )}
            </div>
          </div>

          <div>
            <h4 className="font-bold text-sm text-[#0f172a]">
              Hiển thị top K
            </h4>
            <p className="text-xs text-slate-500 mt-0.5">
              Trả về top 3 mệnh giá có xác suất cao nhất
            </p>
          </div>
        </label>
      </div>
    </div>
  );
};
