import React, { useState } from "react";
import { DENOMINATION_CONFIG } from "../types/prediction";
import type { ClassProbability } from "../types/prediction";
import { ArrowUpDown, Check } from "lucide-react";

interface ProbabilityChartProps {
  probabilities: ClassProbability[];
  showRiskScore?: boolean;
}

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  probabilities,
  showRiskScore = true,
}) => {
  const [sortBy, setSortBy] = useState<"prob_desc" | "denom_asc">("prob_desc");

  // Xử lý sắp xếp
  const sortedProbabilities = [...probabilities].sort((a, b) => {
    if (sortBy === "prob_desc") {
      return b.probability - a.probability;
    }
    return a.denomination - b.denomination;
  });

  return (
    <div className="mt-4 pt-4 border-t border-slate-100">
      {/* Tiêu đề & Nút sắp xếp */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Phân bố xác suất 6 mệnh giá
        </h4>
        <button
          onClick={() =>
            setSortBy((prev) => (prev === "prob_desc" ? "denom_asc" : "prob_desc"))
          }
          type="button"
          className="text-[11px] font-medium text-slate-600 hover:text-blue-600 flex items-center gap-1 bg-slate-100 hover:bg-slate-200/70 px-2 py-1 rounded-lg transition-colors"
        >
          <ArrowUpDown className="w-3 h-3" />
          <span>
            {sortBy === "prob_desc" ? "Xác suất giảm dần" : "Thứ tự mệnh giá"}
          </span>
        </button>
      </div>

      {/* Bảng danh sách & Progress bars */}
      <div className="space-y-2.5">
        {sortedProbabilities.map((item, index) => {
          const config = DENOMINATION_CONFIG[item.denomination];
          const percent = (item.probability * 100).toFixed(2);
          const isTop = item.isPredicted;

          return (
            <div
              key={item.denomination}
              className={`p-2.5 rounded-2xl border transition-all ${
                isTop
                  ? "bg-slate-50/90 border-blue-400 shadow-xs ring-1 ring-blue-500/20"
                  : "bg-white border-slate-100 hover:border-slate-200"
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-500 text-[10px] font-bold flex items-center justify-center shrink-0">
                    {index + 1}
                  </span>
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: config.color }}
                  ></span>
                  <span
                    className={`font-bold ${
                      isTop ? "text-slate-900" : "text-slate-700"
                    }`}
                  >
                    {item.name}
                  </span>
                  {isTop && (
                    <span className="inline-flex items-center gap-0.5 text-[10px] font-semibold text-emerald-700 bg-emerald-100 px-1.5 py-0.2 rounded-md">
                      <Check className="w-2.5 h-2.5" /> Được chọn
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  {showRiskScore && (
                    <span
                      className="text-[11px] text-slate-500 font-mono hidden sm:inline"
                      title="Rủi ro tài chính kỳ vọng (Risk) = P · C_2D"
                    >
                      Risk:{" "}
                      <span className="font-semibold text-slate-700">
                        {item.expectedRisk.toFixed(2)}
                      </span>
                    </span>
                  )}
                  <span
                    className={`font-mono text-xs font-bold ${
                      isTop ? "text-blue-600" : "text-slate-800"
                    }`}
                  >
                    {percent}%
                  </span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500 ease-out"
                  style={{
                    width: `${Math.max(item.probability * 100, 1.5)}%`,
                    backgroundColor: config.color,
                  }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
