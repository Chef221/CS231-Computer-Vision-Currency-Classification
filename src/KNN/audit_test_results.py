"""Verify saved test predictions and write a report. No fitting or inference."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.value_metrics import evaluate_classification
HERE = Path(__file__).resolve().parent
R = HERE / 'results' / 'hsv12_test'

def table(df):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |', '| '+' | '.join(['---']*len(df.columns))+' |'] + ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

p = pd.read_csv(R/'hsv12_predictions_test.csv')
assert len(p)==1553 and p.image_path.is_unique
labels=[10000,20000,50000,100000,200000,500000]
expected=[f.relative_to(ROOT).as_posix() for label in labels for f in sorted((ROOT/'DATA'/'test'/str(label)).glob('*.png'))]
assert p.image_path.tolist()==expected
assert p.filename.tolist()==[Path(f).name for f in expected]
assert p.y_true.tolist()==[int(Path(f).parent.name) for f in expected]
assert set(p.y_pred).issubset(labels)
assert p.is_correct.dtype==bool
np.testing.assert_array_equal(p.is_correct,p.y_true==p.y_pred)
np.testing.assert_array_equal(p.gap_vnd,np.abs(p.y_true-p.y_pred))
scores=evaluate_classification(p.y_true,p.y_pred)
assert scores['class_order_vnd']==labels
s=pd.read_csv(R/'hsv12_summary_test.csv').iloc[0]
dev=pd.read_csv(HERE/'results'/'hsv12_best_dev.csv').iloc[0]
config=json.loads((R/'hsv12_locked_config.json').read_text(encoding='utf-8'))
assert config['k']==1 and config['weights']=='uniform' and config['metric']=='braycurtis'
assert config['feature_dim']==1728 and config['normalization']=='L2'
assert config['fit_split']=='train' and config['selection_split']=='dev' and config['evaluation_split']=='test'
for key in ['k','weights','metric','feature_dim']:assert s[key]==config[key] and s[key]==dev[key]
for key in ['accuracy','macro_f1','mae_vnd','n_samples','total_gap_vnd']:
    assert np.isclose(s[key],scores[key],rtol=1e-12,atol=1e-9)
assert json.loads((R/'hsv12_metrics_test.json').read_text(encoding='utf-8'))==scores
cm=pd.read_csv(R/'hsv12_confusion_matrix_test.csv',index_col=0)
assert list(map(int,cm.index))==labels and list(map(int,cm.columns))==labels
np.testing.assert_array_equal(cm,scores['confusion_matrix'])
assert cm.sum(axis=1).tolist()==[281,224,223,247,317,261]
per=pd.read_csv(R/'hsv12_per_class_test.csv')
pd.testing.assert_frame_equal(per,pd.DataFrame(scores['per_class']),check_exact=False,rtol=1e-12,atol=1e-12)
pairs=pd.DataFrame(scores['error_pairs']).sort_values('total_gap_vnd',ascending=False,kind='stable').reset_index(drop=True)
pd.testing.assert_frame_equal(pd.read_csv(R/'hsv12_error_pairs_test.csv'),pairs)
errors=p[~p.is_correct].sort_values('gap_vnd',ascending=False,kind='stable').reset_index(drop=True)
pd.testing.assert_frame_equal(pd.read_csv(R/'hsv12_error_images_test.csv'),errors)
n=json.loads((HERE/'10_EVALUATE_HSV12_TEST.ipynb').read_text(encoding='utf-8'))
for i,c in enumerate(n['cells']):
    if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
    assert not any(o.get('output_type')=='error' for o in c.get('outputs',[]))
comparison=pd.DataFrame([{'split':'dev','n':1564,**{k:dev[k] for k in ['accuracy','macro_f1','mae_vnd']}}, {'split':'test','n':1553,**{k:scores[k] for k in ['accuracy','macro_f1','mae_vnd']}}])
comparison.to_csv(R/'hsv12_dev_test_comparison.csv',index=False)
per_dev=pd.read_csv(HERE/'results'/'hsv12_per_class_dev.csv')
class_comparison=per[['denomination_vnd','f1']].rename(columns={'f1':'f1_test'}).merge(per_dev[['denomination_vnd','f1']].rename(columns={'f1':'f1_dev'}),on='denomination_vnd')
class_comparison['test_minus_dev']=class_comparison.f1_test-class_comparison.f1_dev
class_comparison.to_csv(R/'hsv12_per_class_dev_test.csv',index=False)
both=pairs.assign(a=pairs[['true_vnd','predicted_vnd']].min(axis=1),b=pairs[['true_vnd','predicted_vnd']].max(axis=1)).groupby(['a','b'],as_index=False)[['count','total_gap_vnd']].sum().sort_values('total_gap_vnd',ascending=False,kind='stable')
both['share_of_total_gap']=both.total_gap_vnd/scores['total_gap_vnd']
both.to_csv(R/'hsv12_bidirectional_error_pairs_test.csv',index=False)
mean_error=float(errors.gap_vnd.mean())
report=f'''# Đánh giá cuối KNN HSV12 trên test

Ngày tổng hợp: 04/11/2026. Nối tiếp [khảo sát dev](EXPERIMENTS.md), báo cáo này phân tích kết quả notebook [10_EVALUATE_HSV12_TEST.ipynb](10_EVALUATE_HSV12_TEST.ipynb). Mọi con số được tính lại từ [1.553 dự đoán theo tên ảnh](results/hsv12_test/hsv12_predictions_test.csv), không phải số liệu minh họa. Không chạy lại suy luận trong lần kiểm tra và không thay đổi cấu hình theo test.

## 1. Giao thức đã dùng

- Fit trên đúng 12.494 ảnh train, không gộp dev. Test gồm 1.553 ảnh, đánh giá toàn bộ, không lọc hay từ chối ảnh.
- Histogram HSV kết hợp 12×12×12, 1.728 chiều; miền H=[0,180), S/V=[0,256), chuẩn hóa L2. Không resize, PCA, augment hoặc standardize.
- KNN: k=1, uniform, Bray–Curtis, brute. Các giá trị khớp CSV cấu hình thắng đã chọn trên dev và [file cấu hình khóa](results/hsv12_test/hsv12_locked_config.json).
- Đọc ảnh lần lượt để tiết kiệm RAM; thứ tự tạo feature khớp đường dẫn khi lưu nhãn thật và dự đoán. Hàm chấm điểm chung giữ đúng sáu lớp.
- Dev được dùng khảo sát tuần tự feature/tham số trước đó. Không chọn lại theo kết quả test này. Không có log đủ để xác minh số lần người dùng thực thi notebook; đây là bộ kết quả test đã lưu hiện tại, không tuyên bố đã kiểm toán toàn bộ lịch sử chạy.

## 2. Kết quả tổng thể và đối chiếu dev

{table(comparison)}

Test đúng **{int(p.is_correct.sum())}/1.553 ảnh**, sai **{len(errors)} ảnh** ({100*len(errors)/len(p):.4f}%). Macro-F1 là **{scores['macro_f1']:.6f}**, Accuracy **{100*scores['accuracy']:.4f}%**, MAE **{scores['mae_vnd']:.2f} VND/ảnh**.

So với dev, Macro-F1 giảm **{dev.macro_f1-scores['macro_f1']:.6f}**, Accuracy giảm **{100*(dev.accuracy-scores['accuracy']):.4f} điểm phần trăm**, MAE tăng **{scores['mae_vnd']-dev.mae_vnd:.2f} VND/ảnh** (khoảng **{100*(scores['mae_vnd']/dev.mae_vnd-1):.2f}%**). Test có chất lượng thấp hơn dev ở cả ba độ đo. Tuy nhiên, không thể chỉ từ chênh lệch này kết luận overfitting là nguyên nhân duy nhất: tập ảnh và support từng lớp khác nhau, còn dev đã dùng chọn cấu hình qua nhiều thí nghiệm. Chưa có khoảng tin cậy hoặc kiểm định nên không gán ý nghĩa thống kê cho mức chênh lệch.

MAE trên mọi ảnh là {scores['mae_vnd']:.2f} VND; nếu chỉ xét {len(errors)} ảnh sai, chênh lệch tuyệt đối trung bình là **{mean_error:.2f} VND/ảnh sai**. Hai mẫu số khác nhau. Kiểm tra: tỷ lệ sai × chênh lệch trung bình trên ảnh sai = MAE toàn test. Tổng chênh lệch là **{scores['total_gap_vnd']:,} VND**, không phải khoản tiền đã bị mất trong giao dịch.

## 3. Kết quả từng mệnh giá

{table(per[['denomination_vnd','support','tp','fp','fn','precision','recall','f1']])}

- **50.000đ** có F1 cao nhất: **0,914943** và Precision **0,938679**. Trong 212 lần dự đoán 50.000đ, có 199 lần đúng; còn 24/223 ảnh 50.000đ thật bị bỏ sót.
- **200.000đ** có F1 thấp nhất: **0,838305**, với 50 ảnh thật bị dự đoán thành lớp khác và 53 ảnh lớp khác bị dự đoán thành 200.000đ. Đây là lớp có số ảnh sai nhiều nhất (50), đồng thời có support cao nhất (317); không dùng số lỗi tuyệt đối thay cho tỷ lệ.
- **20.000đ** có Recall cao nhất **0,897321**, nhưng Precision **0,855319**. Mô hình nhận đúng 201/224 ảnh 20.000đ; trong 235 lần dự đoán lớp này, có 34 lần sai.
- **500.000đ** đạt Recall **0,854406**, bỏ sót 38/261 ảnh. Trong đó 18 ảnh bị dự đoán thành 20.000đ; lỗi này quan trọng về chênh lệch dù không phải cặp có nhiều lỗi một chiều nhất.

### Chênh lệch F1 từng lớp so với dev

{table(class_comparison)}

Các lớp không suy giảm giống nhau: F1 của 20.000đ và 500.000đ giảm rõ hơn trong bảng, trong khi 200.000đ tăng nhẹ. Đây là mô tả số liệu trên hai tập khác nhau, chưa xác minh nguyên nhân theo nguồn hay điều kiện chụp. Không nói mọi lớp đều giảm hoặc dataset test đồng đều khó hơn ở mọi mệnh giá.

## 4. Ma trận nhầm lẫn

Hàng là nhãn thật, cột là dự đoán:

{table(cm.reset_index().rename(columns={cm.index.name or 'index':'true / predicted'}))}

![Confusion matrix test](results/hsv12_test/hsv12_confusion_matrix_test.png)

## 5. Mức độ sai lệch mệnh giá

### 5.1. Các cặp có tổng chênh lệch lớn nhất, giữ riêng hướng

{table(pairs.head(10))}

Cặp nhiều lỗi một chiều nhất là **10.000 → 200.000đ: 21 ảnh**, đóng góp **3.990.000 VND**. Nhưng cặp đóng góp tổng chênh lệch lớn nhất là **500.000 → 20.000đ: 18 ảnh × 480.000 = 8.640.000 VND**. Điều này minh họa lý do không chỉ xem Accuracy hoặc số lỗi: số lỗi ít hơn vẫn có thể gây sai lệch mệnh giá lớn hơn.

### 5.2. Gộp hai hướng để xác định các cặp cần chú ý

{table(both.head(6))}

Cặp **20.000 ↔ 500.000đ** có **33 lỗi** (18 và 15 theo hai hướng), chiếm **{100*33/len(errors):.2f}% số lỗi** nhưng **{100*15840000/scores['total_gap_vnd']:.2f}% tổng chênh lệch**. Cặp **10.000 ↔ 200.000đ** có nhiều lỗi hơn (**40**) nhưng tổng chênh lệch nhỏ hơn (**7.600.000 VND**). Đây là hai ưu tiên phân tích khác nhau: tần suất nhầm và mức sai lệch.

Lỗi có khoảng cách lớn nhất là **490.000 VND/ảnh**: 10.000 → 500.000 có 7 ảnh; chiều ngược lại có 3 ảnh. Hai hướng chịu cùng chênh lệch tuyệt đối theo MAE. MAE không mô hình hóa ai nhận/thối tiền hoặc trách nhiệm giao dịch, nên không diễn giải thành thiệt hại thực tế.

## 6. Quan sát ảnh lỗi thật

![Sáu ảnh lỗi có chênh lệch lớn nhất](results/hsv12_test/hsv12_largest_gap_errors_test.png)

Các ảnh trên được notebook chọn theo `gap_vnd` giảm dần, giữ thứ tự ổn định nếu hòa. Vì thứ tự đường dẫn bắt đầu từ lớp 10.000đ, sáu ảnh đều là 10.000 → 500.000đ. Đây là cách lấy ví dụ xác định trước trong code, không phải sáu ảnh đại diện ngẫu nhiên hay toàn bộ lỗi nghiêm trọng.

| Ảnh | Quan sát trực tiếp từ hình |
| --- | --- |
| `10000_001077.png` | Tờ tiền cong và chụp nghiêng, thấy nhiều nền sáng |
| `10000_001717.png` | Ảnh mờ, bàn tay che một phần, nền tối |
| `10000_001838.png` | Tiền cầm tay tương đối rõ, ngón tay che một phần và nền nhiều vật thể |
| `10000_002183.png` | Nhòe rõ, đường nét trên tờ tiền khó quan sát |
| `10000_002326.png` | Cầm tiền ngoài trời, có nền và một phần bàn tay trong crop |
| `10000_002541.png` | Tờ 10.000đ trên nền sáng, chữ và họa tiết vẫn khá rõ nhưng dự đoán thành 500.000đ |

Không thể quy mọi lỗi cho ảnh mờ: ví dụ cuối vẫn nhìn tương đối rõ. Histogram chỉ đếm phân bố màu và không đọc chữ số hoặc giữ vị trí họa tiết. Phần nền, bàn tay, ánh sáng hoặc cách chia bin có thể làm các ảnh khác mệnh giá có vector gần nhau. Đây là các giả thuyết phù hợp với cách biểu diễn, chưa được kiểm chứng bằng cách phân vùng tiền/nền hay xem hàng xóm gần nhất. Với k=1, dự đoán lấy nhãn của một hàng xóm gần nhất; chưa có khoảng cách/hàng xóm được lưu để nêu cụ thể ảnh train nào gây từng lỗi.

Không đo tỷ lệ mờ, che khuất hoặc nền trên toàn test vì chưa có nhãn điều kiện. Không dùng sáu ảnh này để sửa DATA, đổi cấu hình hoặc báo lại một điểm test mới như chưa từng xem test.

## 7. Kiểm tra đã thực hiện và giới hạn

`audit_test_results.py` kiểm tra:

1. Đủ 1.553 đường dẫn duy nhất, khớp toàn bộ PNG trong `DATA/test` theo đúng thứ tự sáu lớp; nhãn thật khớp thư mục, tên file khớp đường dẫn.
2. Dự đoán thuộc sáu lớp, `is_correct` và `gap_vnd` khớp từng dòng.
3. Tính lại độ đo từ dự đoán bằng hàm chung; khớp summary, JSON, confusion matrix, support, từng lớp, bảng cặp lỗi và danh sách ảnh sai.
4. Cấu hình JSON/summary khớp cấu hình đã chọn trên dev; notebook không lưu lỗi thực thi và code hợp lệ về cú pháp.

Đây là xác minh file dự đoán và code đã lưu, không chạy lại suy luận và không chứng minh toàn bộ quá trình thực thi qua lịch sử độc lập. Không fit lại, không sửa ảnh, nhãn hay split. [JSON kiểm tra](results/hsv12_test/hsv12_test_verification.json) ghi hash evidence. Không suy ra mức an toàn tài chính hoặc khả năng nhận diện tiền ngoài thực tế chỉ từ điểm test này. Chưa đánh giá nhiều split/seed, ảnh ngoài miền hay các mô hình của thành viên khác.

## 8. Kết luận và bước tiếp theo

HSV12 + KNN đã cho một kết quả cuối truy vết được trên test: **Accuracy 87,19%, Macro-F1 0,874772, MAE 31.461,69 VND/ảnh**. Chất lượng thấp hơn dev và vẫn có lỗi chênh lệch lớn. Macro-F1 mô tả khả năng phân loại cân bằng giữa lớp; MAE và bảng cặp lỗi cho thấy mức chênh lệch mệnh giá mà tỷ lệ đúng đơn lẻ không thể hiện.

Giữ cấu hình này làm kết quả KNN đã khóa. Bước tiếp theo là ghép vào bảng so sánh với SVM/CNN/ResNet khi các thành viên có kết quả cùng test và cùng hàm chấm điểm. Nếu phát triển thêm phương án sau khi đã xem test, cần ghi rõ đó là thử nghiệm tiếp nối, không dùng test hiện tại làm dev rồi vẫn tuyên bố là đánh giá chưa xem.
'''
(HERE/'TEST_RESULTS.md').write_text(report.rstrip()+'\n',encoding='utf-8')
verification={'status':'PASS','n_samples':len(p),'correct':int(p.is_correct.sum()),'errors':len(errors),'checks':'Paths match all test PNGs; labels, predictions, metrics, confusion matrix, per-class, error pairs, error images, config and notebook checked. No inference rerun.','metrics':{k:scores[k] for k in ['accuracy','macro_f1','mae_vnd']},'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(R.iterdir()) if f.is_file() and f.name!='hsv12_test_verification.json'}}
(R/'hsv12_test_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PASS:',len(p),'test predictions,',len(errors),'errors; report and verification saved.')
