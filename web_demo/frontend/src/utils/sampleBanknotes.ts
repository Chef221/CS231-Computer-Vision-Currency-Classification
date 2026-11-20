import type { Denomination } from "../types/prediction";

export type TestCaseCategory = "all" | "easy" | "confusable" | "hard" | "wild";

export interface SampleBanknote {
  id: string;
  name: string;
  denomination?: Denomination;
  category: "easy" | "confusable" | "hard" | "wild";
  categoryLabel: string;
  tag: string;
  color: string;
  description: string;
  dataUrl: string;
}

export const SAMPLE_BANKNOTES: SampleBanknote[] = [
  {
    id: "case_easy_500k",
    name: "500k Chuẩn",
    denomination: 500000,
    category: "easy",
    categoryLabel: "Ảnh Dễ",
    tag: "Độ tin cậy ~100%",
    color: "#2563EB",
    description: "Ảnh phẳng, ánh sáng chuẩn, hoa văn sắc nét",
    dataUrl: "./test_cases/case_easy_500k.jpg",
  },
  {
    id: "case_easy_100k",
    name: "100k Chuẩn",
    denomination: 100000,
    category: "easy",
    categoryLabel: "Ảnh Dễ",
    tag: "Độ tin cậy ~100%",
    color: "#16A34A",
    description: "Màu xanh lá cây chuẩn, không phản quang",
    dataUrl: "./test_cases/case_easy_100k.jpg",
  },
  {
    id: "case_confusable_10k_100k",
    name: "10k vs 100k",
    denomination: 10000,
    category: "confusable",
    categoryLabel: "Gần Nhau",
    tag: "Ám xanh rêu",
    color: "#D97706",
    description: "Tờ 10k vàng nâu nhưng bị ám sắc xanh dễ nhầm 100k",
    dataUrl: "./test_cases/case_confusable_10k_100k.png",
  },
  {
    id: "case_confusable_500k_20k",
    name: "500k vs 20k",
    denomination: 500000,
    category: "confusable",
    categoryLabel: "Gần Nhau",
    tag: "Xanh lơ / Lam",
    color: "#0284C7",
    description: "Tờ 500k thiếu sáng, sắc lam ngọc dễ nhầm 20k",
    dataUrl: "./test_cases/case_confusable_500k_20k.png",
  },
  {
    id: "case_hard_glare",
    name: "Lóa sáng & Nhàu",
    denomination: 500000,
    category: "hard",
    categoryLabel: "Ảnh Khó",
    tag: "Lóa sáng polymer",
    color: "#DC2626",
    description: "Mặt nhựa polymer phản quang trắng, mất chi tiết mép",
    dataUrl: "./test_cases/case_hard_glare.png",
  },
  {
    id: "case_hard_perspective",
    name: "Nghiêng & Gấp mép",
    denomination: 500000,
    category: "hard",
    categoryLabel: "Ảnh Khó",
    tag: "Xiên góc 3D",
    color: "#EA580C",
    description: "Chụp chéo góc, bóng đổ và nếp gập ngang thân",
    dataUrl: "./test_cases/case_hard_perspective.png",
  },
  {
    id: "case_wild_hue_hand",
    name: "Cầm tay ngoài trời",
    denomination: 50000,
    category: "wild",
    categoryLabel: "Thực Tế",
    tag: "Bầu trời xanh lấn át",
    color: "#8B5CF6",
    description: "Tờ 50k trên tay nhưng trời xanh chiếm 80% hậu cảnh",
    dataUrl: "./test_cases/case_wild_hue_hand.jpg",
  },
  {
    id: "case_wild_lottery",
    name: "Tờ vé số",
    category: "wild",
    categoryLabel: "Dữ Liệu Lạ",
    tag: "OOD • Dãy số đỏ",
    color: "#EC4899",
    description: "Vật thể lạ ngoài tiền: số đỏ kích hoạt nơ-ron 200k",
    dataUrl: "./test_cases/case_wild_lottery.jpg",
  },
];
