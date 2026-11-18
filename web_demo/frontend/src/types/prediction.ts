// Định nghĩa các mệnh giá tiền polymer Việt Nam
export type Denomination = 10000 | 20000 | 50000 | 100000 | 200000 | 500000;

export const DENOMINATIONS: Denomination[] = [10000, 20000, 50000, 100000, 200000, 500000];

export const DENOMINATION_CONFIG: Record<
  Denomination,
  {
    name: string;
    label: string;
    color: string;
    textColor: string;
    description: string;
  }
> = {
  10000: {
    name: "10.000 VNĐ",
    label: "10000",
    color: "#D97706",
    textColor: "text-amber-800",
    description: "Màu nâu vàng trên nền xanh lục nhẹ, hoa văn mỏ dầu Bạch Hổ",
  },
  20000: {
    name: "20.000 VNĐ",
    label: "20000",
    color: "#0284C7",
    textColor: "text-sky-800",
    description: "Màu xanh lam đậm, chùa Cầu Hội An",
  },
  50000: {
    name: "50.000 VNĐ",
    label: "50000",
    color: "#DB2777",
    textColor: "text-pink-800",
    description: "Màu hồng tím sẫm, phong cảnh Nghênh Lương Đình - Huế",
  },
  100000: {
    name: "100.000 VNĐ",
    label: "100000",
    color: "#16A34A",
    textColor: "text-emerald-800",
    description: "Màu xanh lá cây ánh vàng, Văn Miếu - Quốc Tử Giám",
  },
  200000: {
    name: "200.000 VNĐ",
    label: "200000",
    color: "#EA580C",
    textColor: "text-orange-900",
    description: "Màu đỏ nâu hạt dẻ, phong cảnh hòn Đỉnh Hương (Vịnh Hạ Long)",
  },
  500000: {
    name: "500.000 VNĐ",
    label: "500000",
    color: "#2563EB",
    textColor: "text-blue-900",
    description: "Màu xanh lơ tím sẫm, phong cảnh làng Sen - Nam Đàn",
  },
};

// Hai chiến lược hiển thị đầu ra theo mockup UI:
// - top1: Phân loại trực tiếp (chỉ trả về 1 mệnh giá duy nhất)
// - top3: Phân loại TOP-K (trả về top 3 mệnh giá có xác suất cao nhất)
export type OutputStrategy = "top1" | "top3";

// Cơ chế tính toán nhãn
export type DeciderMode = "bayes" | "argmax";

// Chi tiết xác suất từng mệnh giá
export interface ClassProbability {
  denomination: Denomination;
  label: string;
  name: string;
  probability: number; // 0.0 -> 1.0
  expectedRisk: number; // Giá trị rủi ro kỳ vọng theo C_2D
  isPredicted: boolean;
}

// Kết quả dự đoán tổng thể
export interface PredictionResponse {
  predictedLabel: Denomination;
  predictedName: string;
  outputStrategy: OutputStrategy;
  deciderMode: DeciderMode;
  confidence: number; // 0.0 -> 1.0
  expectedRisk?: number;
  probabilities: ClassProbability[];
  top3Probabilities: ClassProbability[];
  decisionComparison: {
    argmaxChoice: Denomination;
    bayesChoice: Denomination;
    isIdentical: boolean;
    differenceNote?: string;
  };
  processingTimeMs: number;
  financialSafety: {
    safeForATM: boolean;
    overEstimationRisk: "Tuyệt đối an toàn (0đ)" | "Thấp" | "Cảnh báo";
    note: string;
  };
}

export interface PredictionRequest {
  imageFile?: File;
  imageBase64?: string;
  outputStrategy?: OutputStrategy;
  deciderMode?: DeciderMode;
}
