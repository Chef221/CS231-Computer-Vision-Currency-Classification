import React from "react";
import { X, ShieldCheck, Cpu, Database, Award } from "lucide-react";
import { C_2D } from "../services/predictionService";
import { DENOMINATIONS } from "../types/prediction";

interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const InfoModal: React.FC<InfoModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm animate-fadeIn">
      <div className="bg-white rounded-3xl overflow-hidden max-w-3xl w-full shadow-2xl border border-slate-200 flex flex-col max-h-[90vh]">
        {/* Header Modal */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/70">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-blue-100 text-blue-600 flex items-center justify-center font-bold">
              <Award className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 text-base">
                Kiến trúc Mô hình & Cơ sở Lý thuyết (CS231)
              </h3>
              <p className="text-xs text-slate-500">
                Phân loại tiền polymer Việt Nam bằng ResNet-18 & Cost-Sensitive Learning
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            type="button"
            className="p-1.5 rounded-full text-slate-400 hover:text-slate-600 hover:bg-slate-200/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs text-slate-700 leading-relaxed">
          {/* Card 1: Tổng quan đồ án */}
          <div className="p-4 rounded-2xl bg-blue-50/60 border border-blue-200/80">
            <h4 className="font-bold text-sm text-blue-900 mb-2 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-600" /> 1. Kiến trúc Mạng Nơ-ron ResNet-18
            </h4>
            <p className="mb-2">
              Mô hình sử dụng <strong>ResNet-18 ImageNet Transfer Learning</strong> với 11.18 triệu tham số, tinh chỉnh toàn bộ trọng số bằng thuật toán AdamW (lr $10^{-4}$, weight decay $0.01$).
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-blue-200/60 text-slate-800">
              <div className="bg-white p-2 rounded-xl border border-blue-100">
                <span className="text-[10px] text-slate-400 block">Độ phân giải chuẩn</span>
                <span className="font-bold text-xs">224 × 224 px</span>
              </div>
              <div className="bg-white p-2 rounded-xl border border-blue-100">
                <span className="text-[10px] text-slate-400 block">Biến đổi Augmentation</span>
                <span className="font-bold text-xs text-blue-700">hue = 0.0 (Bỏ hẳn)</span>
              </div>
              <div className="bg-white p-2 rounded-xl border border-blue-100">
                <span className="text-[10px] text-slate-400 block">Độ trễ GPU Tesla T4</span>
                <span className="font-bold text-xs text-emerald-700">~2.8 ms (350+ FPS)</span>
              </div>
              <div className="bg-white p-2 rounded-xl border border-blue-100">
                <span className="text-[10px] text-slate-400 block">Kích thước Checkpoint</span>
                <span className="font-bold text-xs">44.8 MB (.pth)</span>
              </div>
            </div>
          </div>

          {/* Card 2: Ma trận chi phí đối xứng C_2D */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-sm text-slate-900 mb-2 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-600" /> 2. Ma trận Chi phí Đối xứng C_2D (Elkan 2001)
            </h4>
            <p className="mb-3">
              Quy tắc phạt: C_ij = sqrt(max(v_i, v_j) / 10.000) kết hợp ngưỡng sàn an toàn 0.20 và chuẩn hóa kỳ vọng Elkan (C_bar = 1.0):
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-center border-collapse font-mono text-[11px]">
                <thead>
                  <tr className="bg-slate-200 text-slate-700">
                    <th className="p-1.5 border border-slate-300">Thật \ Đoán</th>
                    {DENOMINATIONS.map((d) => (
                      <th key={d} className="p-1.5 border border-slate-300">
                        {d >= 1000 ? `${d / 1000}k` : d}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {C_2D.map((row, rIdx) => (
                    <tr key={rIdx} className={rIdx % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                      <td className="p-1.5 font-bold border border-slate-300 bg-slate-100">
                        {DENOMINATIONS[rIdx] >= 1000
                          ? `${DENOMINATIONS[rIdx] / 1000}k`
                          : DENOMINATIONS[rIdx]}
                      </td>
                      {row.map((val, cIdx) => (
                        <td
                          key={cIdx}
                          className={`p-1.5 border border-slate-300 ${
                            rIdx === cIdx
                              ? "bg-emerald-50 text-emerald-800 font-bold"
                              : cIdx === 5 || rIdx === 5
                              ? "bg-rose-50/70 text-rose-800 font-semibold"
                              : "text-slate-700"
                          }`}
                        >
                          {val.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] text-slate-500 mt-2 italic">
              * Ô đường chéo chính = 0.00 (đoán đúng không phạt). Cột/hàng 500k bị phạt nặng nhất ($2.19$), răn đe mô hình tuyệt đối không đoán bừa lên tờ 500.000 VNĐ.
            </p>
          </div>

          {/* Card 3: So sánh Baseline vs CS 2 chiều */}
          <div className="p-4 rounded-2xl bg-amber-50/60 border border-amber-200/80">
            <h4 className="font-bold text-sm text-amber-900 mb-2 flex items-center gap-2">
              <Database className="w-4 h-4 text-amber-600" /> 3. Kết Quả Thực Nghiệm Trên Tập Test (1.553 ảnh)
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-slate-800">
              <div className="bg-white p-3 rounded-xl border border-amber-200">
                <span className="font-bold text-xs text-slate-900 block mb-1">
                  Baseline (Argmax - 0-1 Loss)
                </span>
                <ul className="space-y-1 text-[11px]">
                  <li>• Accuracy: <strong>99.68%</strong> (sai 5 ảnh)</li>
                  <li>• Đoán thừa tiền: <strong>300.000 VNĐ</strong> ⚠️</li>
                  <li className="text-rose-700 font-medium">• Một ảnh 200.000 đồng bị dự đoán thành 500.000 đồng</li>
                </ul>
              </div>
              <div className="bg-white p-3 rounded-xl border border-emerald-300 bg-emerald-50/30">
                <span className="font-bold text-xs text-emerald-900 block mb-1">
                  Thử nghiệm Cost-Sensitive 2 chiều (Bayes Decider)
                </span>
                <ul className="space-y-1 text-[11px]">
                  <li>• Accuracy: <strong>99.48%</strong> (sai 8 ảnh)</li>
                  <li className="text-emerald-700 font-bold">• Đoán thừa lên 500k: <strong>0 VNĐ</strong> 🛡️</li>
                  <li>• Precision tờ 500k: <strong>100.00%</strong></li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Modal */}
        <div className="px-6 py-3.5 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
          <span className="text-[11px] text-slate-500">
            Đồ án Thị giác Máy tính (CS231) — Nhận diện Tiền Polymer
          </span>
          <button
            onClick={onClose}
            type="button"
            className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 transition-colors"
          >
            Đóng cửa sổ
          </button>
        </div>
      </div>
    </div>
  );
};
