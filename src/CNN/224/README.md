# CNN — notebook RGB224 trên Kaggle

Tám notebook đi theo [README CNN](../README.md). Bốn notebook 02–05 huấn luyện bốn cấu hình khác nhau; các notebook khác chuẩn bị, chọn mô hình, chấm test và phân tích.

## Gắn code và dữ liệu

Từ root repository có thể đóng gói Dataset code bằng:

```bash
python src/CNN/build_kaggle_bundle.py
```

File `src/CNN/CS231_CNN_CODE.zip` chứa đúng các đường dẫn cần cho notebook và bộ dựng báo cáo, không chứa ảnh DATA hoặc trọng số. Upload ZIP này để tạo Dataset code trên Kaggle; tạo lại ZIP nếu code thay đổi. Các file notebook để Import nằm riêng trong `src/CNN/224/`.

1. Tạo Dataset code chứa `src/CNN/`, `src/evaluation/`, `src/preprocessing.py` và `docs/CNN_REPORT/`. Giữ nguyên các đường dẫn tương đối này bên dưới thư mục gốc của Dataset. `src/CNN/reference_results/` phải đi cùng code để so sánh KNN/SVM; không cần upload lại NPY/PKL của hai mô hình đó.
2. Gắn Dataset code và Dataset ảnh chứa `DATA/{train,dev,test}/{mệnh_giá}/*.png` vào mỗi notebook bằng Add Input. Nếu Dataset ảnh chứa trực tiếp `train/dev/test`, bộ tìm đường dẫn cũng hỗ trợ.
3. Import notebook `.ipynb` cần chạy. Các ô đầu tự tìm code và DATA, kể cả mount `/kaggle/input/datasets/<tài_khoản>/<tên_dataset>`. Không tìm đệ quy toàn bộ thư mục ảnh. Nếu gắn nhiều bản code/DATA, đặt `CS231_PROJECT_ROOT` và `CS231_DATA_ROOT` để chọn đúng bản.
4. Ở 02–05 bật GPU trong Session options/Accelerator; 06–08 cũng nên bật GPU để trích đặc trưng và suy luận. Code tự dùng `cuda:0`, chỉ một GPU. Kaggle đã có PyTorch và các thư viện thông dụng; nếu thiếu, cài theo đường dẫn tuyệt đối đến `src/CNN/requirements.txt`. Giữ một môi trường thư viện xuyên chuỗi thực nghiệm khi có thể.

## Chạy và truyền output

Chạy `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`. Sau mỗi bước, dùng Save Version → Save & Run All; chờ version hoàn tất thành công rồi gắn **Output của đúng version đó** vào notebook kế tiếp qua Add Input → Notebooks. Notebook 01 không cần Notebook Output. Notebook 02–08 cần đúng một output CNN của bước liền trước, đã chứa toàn bộ kết quả tích lũy.

Với notebook 02–05, có thể dùng Save & Run All để bắt đầu lượt huấn luyện nền ngay sau khi gắn đủ input, thay vì train toàn bộ trong Draft rồi mới chạy lại để lưu version. Cơ chế đọc lại run hoàn tất chỉ có tác dụng khi thư mục kết quả của run đó đang hiện diện; một phiên mới chỉ có output của bước trước vẫn phải huấn luyện cấu hình hiện tại.

Các notebook đã có ô khôi phục output, kiểm tra `stage_completed.json` và artifact của bước trước. Không cần thêm ô chép thủ công. Ví dụ 04 yêu cầu output có stage `03` và `candidates/C02_GRAY/run_complete.json`, không chỉ một thư mục trùng tên. Có thể đặt `CS231_CNN_PREVIOUS_RESULTS` bằng đường dẫn tuyệt đối tới `results_224` được gắn nếu mount có cấu trúc khác.

Đầu ra Kaggle mặc định: `/kaggle/working/CS231_CNN/results_224`; local: `src/CNN/results_224`. Có thể đổi bằng `CS231_CNN_RESULTS_ROOT`. Mỗi notebook sẽ sao chép toàn bộ kết quả của bước trước vào vùng ghi được rồi mới xử lý.

## Checkpoint, chạy lại và thời gian

`best.pt` giữ trọng số của epoch được chọn theo dev; `last.pt` giữ thêm optimizer, scheduler, scaler và lịch sử để resume một lượt bị gián đoạn từ epoch tiếp theo đã lưu. Việc resume chỉ dùng được khi file `last.pt` còn trong phiên hoặc output đã được lưu. Một phiên Kaggle bị hủy trước khi lưu output có thể làm mất các file chưa được lưu.

Run All của một lượt đã hoàn tất sẽ kiểm tra hash rồi đọc lại kết quả thay vì train thêm. Đổi kiến trúc, epoch, augmentation, data manifest hoặc code cần một thư mục thực nghiệm mới và chạy lại từ 01; không sửa hash để vượt kiểm tra. Không lấy checkpoint cuối cùng thay cho checkpoint tốt nhất.

Notebook huấn luyện ghi thời gian từng epoch trong `history.csv`; thời gian thực phụ thuộc GPU, tốc độ đọc ảnh và early stopping. Lấy số giây epoch đầu đã hoàn tất để ước lượng phần còn lại, tối đa 25 epoch/cấu hình. CNN bốn khối có nhiều tham số hơn nhưng thời gian không suy trực tiếp từ số tham số. Tải báo cáo và bảng có thể làm trên CPU.

Lượt đã lưu chạy đủ 25 epoch/cấu hình trên `cuda:0`, Tesla T4, PyTorch 2.11.0+cu130. Thời gian dưới đây cộng từ các cột `train_seconds` và `dev_seconds`, không phải thời gian toàn bộ phiên notebook:

| Notebook | Train (phút) | Dev (phút) | Tổng (phút) |
|---|---:|---:|---:|
| 02 — C01_RGB | 37,71 | 4,75 | 42,47 |
| 03 — C02_GRAY | 34,56 | 4,41 | 38,98 |
| 04 — C03_RGB_AUG | 39,17 | 4,72 | 43,89 |
| 05 — C04_DEEP_AUG | 38,67 | 4,72 | 43,39 |

Tổng tính từ giây gốc trước làm tròn là **168,72 phút**, khoảng 2 giờ 49 phút, chưa gồm chuẩn bị và phân tích. Notebook 07 ghi **83,19 giây** suy luận test kèm đọc ảnh. Chưa có log đủ để công bố thời gian toàn notebook 01/06/08; không suy thời gian từ timestamp của các file tải về. Đây là số đo một lượt, không bảo đảm thời gian ở phiên hoặc phần cứng khác.

## Lấy kết quả cuối

Notebook 08 ghi báo cáo vào `results_224/report/` và tạo `CS231_CNN_RESULTS.zip` ở thư mục cha của `results_224`. Tải ZIP từ Output của version 08 đã hoàn tất. ZIP chứa toàn bộ thư mục kết quả tích lũy, gồm checkpoint để tái sử dụng. Giữ một bản đầy đủ ngoài Git.

Sau khi giải nén trên máy, chép `results_224` vào `src/CNN/results_224` và chạy từ root repository:

```bash
python docs/CNN_REPORT/build_report_224.py
```

Script đọc số đã lưu, kiểm tra đối chiếu và chép hình; không train lại CNN. Các Markdown trong notebook giải thích phương pháp và cách đọc kết quả cho toàn bộ độc giả. Nhận xét nguyên nhân thị giác cần được viết sau khi xem ảnh thực tế.
