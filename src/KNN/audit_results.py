"""Kiem tra CSV da luu va tao bao cao; khong training, khong doc anh DATA."""
import hashlib
import importlib.metadata
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.value_metrics import evaluate_classification

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
SPECS = [
    ('rgb', 'RGB histogram', 768, '02_FEATURE_RGB_HISTOGRAM.ipynb'),
    ('hsv', 'HSV8', 512, '03_FEATURE_HSV_HISTOGRAM.ipynb'),
    ('moments', 'Color Moments', 9, '04_FEATURE_COLOR_MOMENTS.ipynb'),
    ('hog', 'HOG cell16', 1764, '05_FEATURE_HOG.ipynb'),
    ('hog_cell8', 'HOG cell8', 8100, '06_FEATURE_HOG_CELL8.ipynb'),
    ('hsv12', 'HSV12', 1728, '08_FEATURE_HSV_HISTOGRAM_12BINS.ipynb'),
    ('hsv16', 'HSV16', 4096, '09_FEATURE_HSV_HISTOGRAM_16BINS.ipynb'),
]
METRICS = ['braycurtis', 'canberra', 'correlation', 'cosine', 'euclidean', 'minkowski']
ORDER = {config: i for i, config in enumerate(itertools.product(range(1, 11), ['uniform', 'distance'], METRICS))}
LABELS = [10000, 20000, 50000, 100000, 200000, 500000]


def table(frame):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f'{v:.6f}'
        return str(v)
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


records = []
analyses = []
for prefix, name, dim, notebook in SPECS:
    grid = pd.read_csv(RESULTS / f'{prefix}_knn_grid_dev.csv')
    cm = pd.read_csv(RESULTS / f'{prefix}_confusion_matrix_dev.csv', index_col=0)
    assert len(grid) == 120 and len(grid.drop_duplicates(['k', 'weights', 'metric'])) == 120
    configs = list(zip(grid.k, grid.weights, grid.metric))
    assert set(configs) == set(ORDER)
    expected = grid.assign(_order=[ORDER[c] for c in configs]).sort_values(
        ['macro_f1', 'mae_vnd', '_order'], ascending=[False, True, True]
    ).drop(columns='_order').reset_index(drop=True)
    pd.testing.assert_frame_equal(grid.reset_index(drop=True), expected)
    assert list(map(int, cm.index)) == LABELS and list(map(int, cm.columns)) == LABELS
    matrix = cm.to_numpy()
    assert np.isfinite(matrix).all() and (matrix >= 0).all() and np.equal(matrix, matrix.astype(int)).all()
    assert matrix.sum(axis=1).tolist() == [315, 238, 195, 250, 309, 257]
    true, pred = [], []
    for i, a in enumerate(LABELS):
        for j, b in enumerate(LABELS):
            true.extend([a] * int(matrix[i, j]))
            pred.extend([b] * int(matrix[i, j]))
    scores = evaluate_classification(true, pred)
    best = grid.iloc[0]
    for key in ['macro_f1', 'accuracy', 'mae_vnd']:
        assert np.isclose(scores[key], best[key], rtol=1e-12, atol=1e-9)
    summary_path = RESULTS / f'{prefix}_best_dev.csv'
    if summary_path.exists():
        summary = pd.read_csv(summary_path).iloc[0]
        assert int(summary.feature_dim) == dim
        for key in ['k', 'weights', 'metric']:
            assert summary[key] == best[key]
        for key in ['macro_f1', 'accuracy', 'mae_vnd']:
            assert np.isclose(scores[key], summary[key], rtol=1e-12, atol=1e-9)
    per_class = pd.DataFrame(scores['per_class'])
    old_per = RESULTS / f'{prefix}_per_class_dev.csv'
    if old_per.exists():
        old = pd.read_csv(old_per)
        for key in ['denomination_vnd', 'precision', 'recall', 'f1', 'support']:
            np.testing.assert_allclose(old[key], per_class[key], rtol=1e-12, atol=1e-12)
    pairs = pd.DataFrame(scores['error_pairs']).sort_values('total_gap_vnd', ascending=False, kind='stable')
    pairs.to_csv(RESULTS / f'{prefix}_error_costs_dev.csv', index=False)
    record = dict(feature=name, feature_dim=dim, k=int(best.k), weights=best.weights, metric=best.metric,
                  macro_f1=scores['macro_f1'], accuracy=scores['accuracy'], mae_vnd=scores['mae_vnd'],
                  correct=int(np.trace(matrix)), errors=int(1564-np.trace(matrix)), total_gap_vnd=scores['total_gap_vnd'])
    records.append(record)
    analyses.append((prefix, name, notebook, grid, cm, per_class, pairs, record))

all_results = pd.DataFrame(records)
all_results.to_csv(RESULTS / 'knn_all_features_dev.csv', index=False)
all_results[all_results.feature.isin(['HSV8', 'HSV12', 'HSV16'])].to_csv(RESULTS / 'hsv_bins_comparison_dev.csv', index=False)

parts = ['''# Nhật ký và phân tích thực nghiệm KNN

Ngày tổng hợp: 04/11/2026. Tài liệu này ghi nhận giai đoạn khảo sát **trên dev**. Kết quả test đã được bổ sung riêng tại [TEST_RESULTS.md](TEST_RESULTS.md); các câu mô tả “chưa đánh giá test” bên dưới chỉ trạng thái tại thời điểm khảo sát dev. Các bảng được tạo từ CSV đã lưu bằng `audit_results.py`; không chạy lại training hay dự đoán khi tổng hợp.

## 1. Câu hỏi và phạm vi

Nhóm khảo sát các biểu diễn màu và đường nét cho bài toán phân loại một ảnh crop vào sáu mệnh giá polymer Việt Nam: 10.000, 20.000, 50.000, 100.000, 200.000 và 500.000đ. Mỗi biểu diễn được chọn cấu hình KNN riêng trên cùng tập dev. Không phát hiện tiền giả, không cộng tổng tiền, không OCR và chưa triển khai từ chối dự đoán.

Đây là quá trình khảo sát tuần tự: thử RGB, HSV8, Color Moments, hai cấu hình HOG, rồi tăng bin HSV sau khi quan sát kết quả trước. Vì dev được dùng nhiều lần để điều chỉnh hướng thử nghiệm, kết quả dev có thể lạc quan so với dữ liệu chưa xem. Không coi đây là một thiết kế đã đăng ký toàn bộ từ trước.

## 2. Dữ liệu và quy trình chung

| Tập | Số ảnh | Vai trò trong các bảng này |
| --- | ---: | --- |
| Train | 12.494 | Fit KNN |
| Dev | 1.564 | Chọn feature/tham số và phân tích lỗi |
| Test | 1.553 | Chưa đánh giá; chỉ dùng sau khi khóa cấu hình |

Thứ tự lớp cố định: 10000, 20000, 50000, 100000, 200000, 500000. Support trên dev tương ứng là 315, 238, 195, 250, 309, 257. Ảnh và nhãn được đọc theo cùng thứ tự thư mục lớp và tên file; nhãn là giá trị VND, không phải chỉ số 0–5. OpenCV đọc BGR rồi chuyển sang RGB.

Dữ liệu đầu vào là PNG crop có padding từ M0. Trong thực nghiệm này không sửa DATA, không chia lại split, không augment, không PCA, không fit scaler, không gộp train+dev. M0 nhóm ảnh theo tên nguồn khi chia; điều đó không chứng minh mọi ảnh đều độc lập theo phiên chụp hoặc tờ tiền vật lý. Các kết quả ở đây không thay thế một cuộc kiểm toán leakage mới.

Notebook RGB có đọc và trích xuất feature test nhưng không dùng chúng trong các bảng kết quả/chọn cấu hình đang lưu. Các notebook sau chỉ đọc train/dev. Việc đã tạo feature test khác với đã đánh giá dự đoán test.

### 2.1. Lưới KNN

- `n_neighbors`: 1 đến 10.
- `weights`: `uniform`, `distance`.
- `metric`: `braycurtis`, `canberra`, `correlation`, `cosine`, `euclidean`, `minkowski`.
- `algorithm='brute'`; Minkowski mặc định `p=2`.

Mỗi biểu diễn có 10 × 2 × 6 = **120 cấu hình**, tổng cộng **840 dòng cấu hình** trong bảy bảng. Đây không phải 840 mô hình độc lập về hành vi: Euclidean và Minkowski p=2 tương đương; với k=1, hai cách bỏ phiếu thường cho cùng dự đoán. Không dùng các dòng lặp hành vi như bằng chứng lặp thí nghiệm hay ước lượng độ bất định.

Với `uniform`, hàng xóm bỏ phiếu như nhau; với `distance`, hàng xóm gần có ảnh hưởng lớn hơn. Bray–Curtis so tổng chênh lệch tuyệt đối với tổng độ lớn; Canberra cộng chênh lệch tương đối theo từng chiều; correlation so hình dạng vector sau trừ trung bình; cosine so góc; Euclidean đo khoảng cách thẳng. Color Moments có skewness âm, nên cách hiểu Bray–Curtis như so sánh histogram không âm không áp dụng trực tiếp cho feature đó.

Chọn bằng **Macro-F1 chưa làm tròn cao nhất**; nếu bằng nhau chọn **MAE_VND thấp nhất**; nếu vẫn bằng nhau chọn thứ tự `k → weights → metric` theo lưới trên. Các điểm hiển thị trong tài liệu được làm tròn, CSV giữ độ chính xác gốc. Đây là tìm kiếm tham số bằng vòng lặp train/dev, không phải k-fold cross-validation.

### 2.2. Độ đo

Dùng hàm chung `src/evaluation/value_metrics.py`: Accuracy là số đúng/N; Macro-F1 là trung bình F1 của đúng sáu lớp; MAE_VND = tổng |mệnh giá thật − dự đoán| / N, gồm cả ảnh đúng với sai lệch 0. Precision/Recall/F1 từng lớp dùng quy tắc chia cho 0 trả 0. Ma trận nhầm lẫn có hàng là nhãn thật, cột là dự đoán.

MAE đo chênh lệch mệnh giá danh nghĩa, **không phải thiệt hại tài chính thực tế**. Một độ đo tốt hơn không tự động làm mọi độ đo còn lại tốt hơn. Quy tắc chọn đã cố định ưu tiên Macro-F1, không đổi sang MAE sau khi nhìn thấy kết quả.

## 3. Các biểu diễn đã thử

| Biểu diễn | Cách tính | Số chiều |
| --- | --- | ---: |
| RGB histogram | 256 bin riêng mỗi kênh R/G/B; chuẩn hóa L2 riêng từng kênh rồi nối | 768 |
| HSV8 | Histogram kết hợp H/S/V, 8 bin mỗi trục; chuẩn hóa L2 cả vector | 512 |
| Color Moments | Nối mean R/G/B, std R/G/B và skewness R/G/B | 9 |
| HOG cell16 | Xám, resize 128×128, cell16×16, block2×2, 9 hướng, L2-Hys | 1.764 |
| HOG cell8 | Giữ HOG trên, chỉ đổi cell8×8 | 8.100 |
| HSV12 | Như HSV8, đổi 12 bin mỗi trục | 1.728 |
| HSV16 | Như HSV8, đổi 16 bin mỗi trục | 4.096 |

HSV dùng miền OpenCV H=[0,180), S/V=[0,256), số chiều bằng số bin H × S × V. Không resize các feature màu. Color Moments tính skewness bằng `mean(((image-mean)/(std+1e-6))**3)` theo từng kênh, không chuẩn hóa L2 hoặc standardize sau đó. Mean/std và skewness có thang đo khác nhau; kết quả của cấu hình này chưa đại diện cho Color Moments đã chuẩn hóa.

HOG dùng `transform_sqrt=False`, không dùng màu, không đọc chữ số như OCR. Resize vuông có thể làm đổi tỷ lệ ảnh; chưa căn thẳng tờ tiền hay chuẩn hóa góc quay. Cell16 tạo 7×7 block, cell8 tạo 15×15 block; mỗi block đóng góp 2×2×9 phần tử. Chuẩn hóa theo block khác với chuẩn hóa L2 toàn vector histogram.

## 4. Kết quả tổng hợp
''', table(all_results), '''
Từng feature được chọn KNN riêng, nên bảng so sánh các tổ hợp feature + cấu hình KNN tốt nhất trong phạm vi khảo sát; không phải tác động thuần của feature khi giữ cố định mọi tham số KNN. Những số liệu này thuộc đúng một split hiện tại, chưa có khoảng tin cậy hay kiểm định ý nghĩa thống kê.
''']

interpretations = {
'rgb': 'RGB đạt kết quả thấp hơn HSV8 trong thiết lập này. RGB dùng ba histogram biên riêng, còn HSV dùng histogram kết hợp; thí nghiệm đồng thời đổi không gian màu và kiểu histogram, vì vậy chưa thể quy toàn bộ cải thiện cho riêng việc chuyển RGB sang HSV. Histogram không giữ vị trí của màu. Chưa thử RGB histogram kết hợp hoặc các số bin RGB khác.',
'hsv': 'HSV8 tốt hơn RGB và Color Moments ở cả ba độ đo tổng hợp. Điều này cho thấy biểu diễn màu kết hợp này hữu ích trên dev hiện tại; không chứng minh màu là tín hiệu duy nhất hay mô hình đã hiểu chữ số. Số chiều 512 thấp hơn RGB 768 nhưng điểm tốt hơn: số chiều không quyết định chất lượng.',
'moments': 'Color Moments nén ảnh thành 9 thống kê và cho điểm thấp hơn RGB/HSV. Các thống kê này không giữ đầy đủ hình dạng phân bố màu và vị trí họa tiết. Đây là các giới hạn có thể giải thích kết quả, chưa được tách riêng bằng thí nghiệm nhân quả. Chưa thử scaler train-only nên không kết luận mọi biến thể Color Moments đều kém.',
'hog': 'HOG cell16 thấp hơn các feature màu đã thử. Việc bỏ màu, mất chi tiết sau resize và thay đổi vị trí/hướng của họa tiết là các giả thuyết giải thích hợp lý; chưa có thí nghiệm đủ để xác định nguyên nhân nào chi phối. Không được nói HOG không nhận diện được tiền nói chung.',
'hog_cell8': 'Giảm cell từ 16 xuống 8 tăng số chiều từ 1.764 lên 8.100 (khoảng 4,59 lần), nhưng không cải thiện ba độ đo trên dev. Cấu hình thắng đổi khoảng cách, nên đây là so sánh hai cấu hình feature đã được chọn KNN riêng. Chưa thử ảnh 256×256, giữ tỷ lệ thêm viền hay căn góc; không suy kết quả sang các phương án đó.',
'hsv12': 'HSV12 tăng độ phân giải histogram so với HSV8 và cải thiện cả Macro-F1, Accuracy lẫn MAE. Kết quả phù hợp với giả thuyết các bin chi tiết hơn giúp phân biệt màu trong dữ liệu này, nhưng chưa chứng minh cơ chế đó hoặc khả năng tổng quát hóa sang dữ liệu mới. HSV12 là ứng viên được chọn theo quy tắc dev hiện tại.',
'hsv16': 'HSV16 không tiếp tục tăng Macro-F1 so với HSV12 dù vector dài hơn. MAE lại thấp hơn: đây là đánh đổi giữa chất lượng phân loại và chênh lệch mệnh giá, không phải lỗi tính điểm. Không có dự đoán từng ảnh được lưu để đối chiếu trực tiếp các trường hợp chuyển đúng/sai giữa hai mô hình. Không nói hai mô hình chỉ khác nhau đúng hai ảnh; chỉ biết tổng số ảnh đúng chênh hai.',
}
parts.append('## 5. Phân tích từng thí nghiệm\n')
for number, (prefix, name, notebook, grid, cm, per, pairs, rec) in enumerate(analyses, 1):
    parts += [f'## 5.{number}. {name}\n\nNotebook: [{notebook}]({notebook}). Evidence: [grid](results/{prefix}_knn_grid_dev.csv), [confusion matrix](results/{prefix}_confusion_matrix_dev.csv), [chi phí từng cặp lỗi](results/{prefix}_error_costs_dev.csv).\n',
              f'Cấu hình được chọn: **k={rec["k"]}, weights={rec["weights"]}, metric={rec["metric"]}**. Đúng **{rec["correct"]}/1.564**, sai **{rec["errors"]}**. Macro-F1 **{rec["macro_f1"]:.6f}**, Accuracy **{100*rec["accuracy"]:.4f}%**, MAE **{rec["mae_vnd"]:.2f} VND/ảnh**. Tổng chênh lệch tuyệt đối là **{rec["total_gap_vnd"]:,} VND**; đây là tổng trên toàn dev, không phải khoản tiền tổn thất.\n',
              '### Cấu hình tốt nhất theo từng khoảng cách\n', table(grid.drop_duplicates('metric')[['metric','k','weights','macro_f1','mae_vnd','accuracy']]),
              '\n### Kết quả từng lớp\n', table(per[['denomination_vnd','support','precision','recall','f1']])]
    low = per.loc[per.f1.idxmin()]; high = per.loc[per.f1.idxmax()]
    parts += [f'\nLớp có F1 cao nhất là **{int(high.denomination_vnd):,}đ ({high.f1:.6f})**; thấp nhất là **{int(low.denomination_vnd):,}đ ({low.f1:.6f})**. Đây là xếp hạng theo F1, không đồng nhất với xếp hạng Recall.\n',
              '### Ma trận nhầm lẫn\n', table(cm.reset_index().rename(columns={cm.index.name or 'index':'true / predicted'})),
              '\n### Năm cặp lỗi có tổng chênh lệch lớn nhất\n', table(pairs.head(5))]
    top_count = pairs.sort_values('count',ascending=False,kind='stable').iloc[0]
    top_cost = pairs.iloc[0]
    parts += [f'\nCặp có nhiều lỗi nhất (một chiều) là {int(top_count.true_vnd):,} → {int(top_count.predicted_vnd):,}đ: **{int(top_count["count"])} ảnh**. Cặp đóng góp tổng chênh lệch lớn nhất là {int(top_cost.true_vnd):,} → {int(top_cost.predicted_vnd):,}đ: **{int(top_cost["count"])} ảnh**, tương ứng **{int(top_cost.total_gap_vnd):,} VND**. Tần suất sai và mức đóng góp MAE là hai góc nhìn khác nhau.\n\n', interpretations[prefix], '\n']

by_name = {r['feature']: r for r in records}
a,b,c = [by_name[name] for name in ['HSV8','HSV12','HSV16']]
h1,h2 = [by_name[name] for name in ['HOG cell16','HOG cell8']]
parts += [f'''## 6. Đối chiếu các thay đổi có chủ đích

### 6.1. Tăng bin HSV

HSV8 → HSV12: Macro-F1 tăng **{b['macro_f1']-a['macro_f1']:.6f}**, Accuracy tăng **{100*(b['accuracy']-a['accuracy']):.4f} điểm phần trăm**, số ảnh đúng tăng **{b['correct']-a['correct']}**, MAE giảm **{a['mae_vnd']-b['mae_vnd']:.2f} VND/ảnh**. Số chiều tăng 3,375 lần.

HSV12 → HSV16: Macro-F1 giảm **{b['macro_f1']-c['macro_f1']:.6f}**, Accuracy giảm **{100*(b['accuracy']-c['accuracy']):.4f} điểm phần trăm**, số ảnh đúng giảm **{b['correct']-c['correct']}**, nhưng MAE giảm **{b['mae_vnd']-c['mae_vnd']:.2f} VND/ảnh**. Số chiều tăng khoảng 2,37 lần. Chênh lệch điểm nhỏ; chưa có bằng chứng thống kê rằng HSV12 vượt trội trong tổng thể.

Theo quy tắc chính, **chọn HSV12 + KNN k=1, uniform, Bray–Curtis**, vì Macro-F1 cao nhất trong bảy tổ hợp đã khảo sát. MAE chỉ dùng phá hòa khi Macro-F1 bằng nhau, không thay thế tiêu chí chính ở bước này. HSV16 được giữ như một phương án có MAE thấp hơn, không giấu kết quả này.

### 6.2. Giảm cell HOG

Cell16 → cell8: Macro-F1 giảm **{h1['macro_f1']-h2['macro_f1']:.6f}**, số đúng giảm **{h1['correct']-h2['correct']}**, MAE tăng **{h2['mae_vnd']-h1['mae_vnd']:.2f} VND/ảnh**. Kết quả chưa cho thấy lợi ích từ lưới chi tiết hơn ở resize 128×128. Nhóm dừng mở rộng HOG ở đây, không tuyên bố đã tìm tối ưu HOG.

### 6.3. Bộ nhớ và thời gian

Không có log thời gian/RAM đỉnh đồng bộ cho bảy thí nghiệm, nên không báo cáo tốc độ đo được hoặc khẳng định cấu hình nào nhanh hơn bao nhiêu. Có thể tính riêng dung lượng một ma trận feature float32 của 12.494 ảnh train (chưa gồm ảnh, bản sao, ma trận khoảng cách và overhead):
''', table(pd.DataFrame([{'feature':r['feature'],'train_float32_MiB':12494*r['feature_dim']*4/1024**2} for r in records if r['feature']!='Color Moments'])), '''
Color Moments thực tế lưu float64: ma trận train 12.494×9 chiếm khoảng 0,858 MiB. Các con số trên là dung lượng lý thuyết của ma trận, không phải tổng RAM tiến trình.

Lần đầu HOG hết RAM ở bước giữ toàn bộ ảnh gốc. Notebook 05/06 đã chuyển sang giữ đường dẫn, đọc từng ảnh và ghi vector vào ma trận float32. Cấu hình HOG và kết quả feature không đổi theo cách đọc này; đã kiểm tra trên mẫu ảnh. RGB/HSV/Color Moments hiện vẫn giữ toàn bộ ảnh trong RAM. Restart kernel các notebook khác trước khi chạy; nếu vẫn thiếu RAM cần đổi cách đọc hoặc chạy máy có RAM phù hợp, không tự resize các feature màu vì sẽ đổi thí nghiệm.

## 7. Kiểm tra bằng chứng và khả năng chạy lại

`audit_results.py` kiểm tra 120 cấu hình duy nhất cho từng bảng, thứ tự chọn theo lưới, support dev, rồi dựng cặp nhãn từ confusion matrix và tính lại bằng hàm đánh giá chung. Các độ đo phải khớp dòng thắng trong grid, summary và bảng từng lớp (nếu đã có). CSV cặp lỗi theo hướng được tạo lại trực tiếp từ ma trận; `total_gap_vnd=count×gap_vnd`. Đây là kiểm tra tính nhất quán số học của file đã lưu, **không chứng minh từng dự đoán đã được tái tạo từ model**. Trong giai đoạn dev chưa lưu vector dự đoán gắn với tên ảnh hay model đã fit; không suy ra lỗi cụ thể của một ảnh từ bảng tổng hợp.

Các cell code của các notebook hiện có được kiểm tra cú pháp; giai đoạn dev gồm tám notebook, notebook test được bổ sung sau. Khi chuẩn bị chia sẻ, notebook RGB được sửa tối thiểu: thêm dấu đóng ngoặc trong cell lưu CSV và ghép `df_cosine` vào `df_results` trước khi chọn cấu hình, loại cấu hình lặp. Hai cell này đã xóa output cũ; không chạy lại thí nghiệm và không thay CSV kết quả gốc. CSV RGB hiện có đủ 120 cấu hình và kiểm tra nhất quán với confusion matrix. Các output khác trong notebook là lịch sử chạy trước sửa; nên dùng CSV đã kiểm tra làm evidence chính. Notebook RGB còn hard-code cấu hình thắng đã biết cho confusion matrix; nếu đổi grid thì phải cập nhật hoặc lấy trực tiếp dòng thắng.

Dữ liệu train/dev đã chốt được dùng một lần chia. Không có thí nghiệm nhiều split/seed, khoảng tin cậy, đánh giá theo nguồn/điều kiện chụp, hoặc kiểm tra ngoài miền trong đợt KNN này. Không có log đủ để gán chính xác phiên bản thư viện tại thời điểm từng notebook được chạy. `audit_results_verification.json` ghi phiên bản môi trường **lúc kiểm tra**, không coi đó là môi trường đã chứng thực cho mọi lượt chạy trước.

Chưa chạy lại toàn bộ notebook trong đợt tổng hợp này vì sẽ lặp lại 840 cấu hình đã có kết quả. Không có kết quả test, SVM/CNN/ResNet hoặc số liệu thời gian giả định trong tài liệu.

## 8. Bước tiếp theo

1. Dừng khảo sát bin ở ba cấu hình đã thử; không gọi HSV12 là tối ưu tuyệt đối.
2. Khóa HSV12, chuẩn hóa L2, k=1, uniform, Bray–Curtis và thứ tự lớp, trước khi dùng test.
3. Fit chỉ train và đánh giá test một lần theo bộ độ đo chung. Lưu `filename, y_true, y_pred` để truy vết ảnh lỗi; không chỉnh cấu hình theo test.
4. Phân tích ảnh sai thực tế cùng các cặp lỗi có tần suất cao và tổng chênh lệch lớn. Sau đó so sánh với thành viên làm SVM/CNN/ResNet trên cùng giao thức.

Việc đánh giá test là công việc kế tiếp, chưa được thực hiện trong lần viết tài liệu và push này.
''']
# Figure drawn only from verified result CSV values.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, key, title in zip(axes, ['macro_f1', 'accuracy', 'mae_vnd'],
                         ['Macro-F1 (higher is better)', 'Accuracy (higher is better)', 'MAE VND/image (lower is better)']):
    ax.barh(all_results.feature, all_results[key], color=['#8c8c8c', '#527da3', '#8c8c8c', '#8c8c8c', '#8c8c8c', '#27825b', '#527da3'])
    ax.set_title(title)
    ax.invert_yaxis()
    if key != 'mae_vnd':
        ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.2)
fig.suptitle('KNN: best configurations on DEV (N=1,564)')
fig.tight_layout()
fig.savefig(RESULTS/'knn_dev_comparison.png', dpi=160)
plt.close(fig)
parts.insert(3, '![So sánh kết quả KNN trên dev](results/knn_dev_comparison.png)\n')

(HERE/'EXPERIMENTS.md').write_text('\n\n'.join(parts).rstrip()+'\n',encoding='utf-8')
notebook_checks=[]
for p in sorted(HERE.glob('*.ipynb')):
    n=json.loads(p.read_text(encoding='utf-8'))
    for i,cell in enumerate(n['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'{p.name}:cell{i}','exec')
    notebook_checks.append(p.name)
files = sorted(RESULTS.glob('*.csv')) + sorted(HERE.glob('*.ipynb'))
verification = {
    'status':'PASS', 'audit_date':'2026-11-04', 'grid_rows':840,
    'n_dev':1564, 'features_checked':7, 'notebooks_syntax_checked':notebook_checks,
    'scope':'Arithmetic consistency of stored results only; no retraining or prediction rerun; no DATA/split modifications.',
    'audit_environment':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','opencv-python','scikit-image']},
    'sha256':{str(p.relative_to(HERE)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}
(RESULTS/'audit_results_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PASS: 7 result sets, 840 grid rows, all notebook syntax checks; dev report and evidence saved.')
