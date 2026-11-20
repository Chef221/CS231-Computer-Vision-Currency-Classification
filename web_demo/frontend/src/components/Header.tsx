import React from "react";

interface HeaderProps {
  isMockMode?: boolean;
  onToggleMockMode?: () => void;
  onOpenInfoModal?: () => void;
}

export const Header: React.FC<HeaderProps> = () => {
  return (
    <header className="relative bg-white border-b border-slate-200/80 overflow-hidden shadow-xs">
      {/* Hình nền mờ (Watermark) mô phỏng tờ 500.000đ ở góc phải đúng như mockup */}
      <div className="absolute right-0 top-0 bottom-0 w-80 md:w-96 pointer-events-none opacity-20 overflow-hidden flex items-center justify-end pr-4">
        <img
          src="./samples/500000_web.jpg"
          alt="Watermark 500.000đ"
          className="h-full object-cover rounded-xl filter grayscale contrast-125 mix-blend-multiply"
        />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        {/* Logo và Tiêu đề chuẩn hình mẫu */}
        <div className="flex items-center gap-4">
          {/* Logo biểu tượng tiền tệ: Hộp màu xanh đậm với viền tờ tiền và chữ ₫ */}
          <div className="w-14 h-10 rounded-xl bg-[#0b3b95] text-white flex items-center justify-center shadow-xs shrink-0 border border-blue-900">
            <div className="w-10 h-7 border border-white/80 rounded-md flex items-center justify-center relative">
              <div className="w-4 h-4 rounded-full border border-white/70 flex items-center justify-center font-serif text-sm font-bold">
                ₫
              </div>
            </div>
          </div>

          {/* Vạch ngăn cách dọc */}
          <div className="w-px h-9 bg-slate-300 hidden sm:block"></div>

          {/* Tiêu đề & Subtitle */}
          <div>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
              AI Nhận Diện Tiền Polymer Việt Nam
            </h1>
            <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
              Nhận diện mệnh giá từ 10.000 – 500.000 VNĐ
            </p>
          </div>
        </div>
      </div>
    </header>
  );
};
