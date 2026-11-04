"""Generate the three report figures and traceable tables. No model fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import braycurtis
import cv2
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
R=ROOT/'src'/'KNN'/'results'
F=HERE/'KNN_FIGURES'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
grid=pd.read_csv(R/'knn_all_features_dev.csv')
selected=grid.set_index('feature').loc[['RGB histogram','HSV8','HSV12','Color Moments','HOG cell16']].reset_index()
selected.to_csv(HERE/'review'/'selected_dev_results.csv',index=False)
cm=pd.read_csv(R/'hsv12_test'/'hsv12_confusion_matrix_test.csv',index_col=0)
fig,ax=plt.subplots(figsize=(6.7,5.5))
image=ax.imshow(cm.to_numpy(),cmap='Blues',vmin=0)
labels=['10','20','50','100','200','500']
ax.set_xticks(range(6),labels);ax.set_yticks(range(6),labels)
ax.set_xlabel('Mệnh giá dự đoán (nghìn đồng)');ax.set_ylabel('Mệnh giá thực tế (nghìn đồng)')
for i in range(6):
 for j in range(6):ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',color='white' if cm.iloc[i,j]>cm.to_numpy().max()/2 else 'black',fontsize=11)
fig.colorbar(image,ax=ax,label='Số ảnh',fraction=0.046,pad=0.04)
fig.tight_layout()
fig.savefig(F/'KNN_1_CONFUSION_MATRIX.png',dpi=300)
fig.savefig(F/'KNN_1_CONFUSION_MATRIX.svg')
svg_path = F/'KNN_1_CONFUSION_MATRIX.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
plt.close(fig)
neighbors=pd.read_csv(R/'neighbor_analysis'/'five_nearest_neighbors.csv')
cases=pd.read_csv(R/'neighbor_analysis'/'case_summary.csv')
original=pd.read_csv(R/'hsv12_test'/'hsv12_predictions_test.csv')
assert cases.matches_saved_prediction.all()

def feat(path):
 im=cv2.imread(str(ROOT/path));hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
 h=cv2.calcHist([hsv],[0,1,2],None,[12,12,12],[0,180,0,256,0,256])
 return cv2.normalize(h,h).flatten().astype(np.float64)
checks=[]
for _,case in cases.iterrows():
 row=original[original.filename==case.filename].iloc[0]
 assert case.y_pred==case.recomputed_prediction==row.y_pred and case.y_true==row.y_true
 q=feat(case.image_path)
 group=neighbors[neighbors.query_filename==case.filename]
 assert group['rank'].tolist()==[1,2,3,4,5]
 assert (np.diff(group.distance_braycurtis)>=0).all()
 for _,n in group.iterrows():
  assert int(Path(n.train_image_path).parent.name)==n.train_label_vnd
  d=braycurtis(q,feat(n.train_image_path))
  assert np.isclose(d,n.distance_braycurtis,atol=1e-7,rtol=1e-7)
  checks.append({'query':case.filename,'rank':int(n['rank']),'distance_csv':float(n.distance_braycurtis),'distance_recomputed_float64':float(d)})

def triptych(filename,correct_rank,destination,decimals):
 case=cases[cases.filename==filename].iloc[0]
 group=neighbors[neighbors.query_filename==filename].set_index('rank')
 wrong=group.loc[1];correct=group.loc[correct_rank]
 assert wrong.train_label_vnd!=case.y_true and correct.train_label_vnd==case.y_true
 fig,axes=plt.subplots(1,3,figsize=(10,4.1))
 paths=[case.image_path,wrong.train_image_path,correct.train_image_path]
 titles=[f'Ảnh test — nhãn {int(case.y_true):,}đ\nDự đoán: {int(case.y_pred):,}đ',
         f'Hàng xóm hạng 1 — {int(wrong.train_label_vnd):,}đ\nd = {wrong.distance_braycurtis:.{decimals}f}',
         f'Hàng xóm hạng {correct_rank} — {int(correct.train_label_vnd):,}đ\nd = {correct.distance_braycurtis:.{decimals}f}']
 for ax,path,title in zip(axes,paths,titles):
  with Image.open(ROOT/path) as im:ax.imshow(im.convert('RGB'))
  ax.set_title(title.replace(',','.'),fontsize=11,pad=10);ax.axis('off')
 fig.tight_layout(w_pad=1.2)
 fig.savefig(F/destination,dpi=300,facecolor='white')
 plt.close(fig)
triptych('10000_002541.png',3,'KNN_2_CLEAR_IMAGE_ERROR.png',6)
triptych('500000_000096.png',2,'KNN_3_NEAR_EQUAL_DISTANCES.png',9)
(HERE/'review'/'neighbor_distance_checks.json').write_text(json.dumps({'status':'PASS','scope':'15 stored query-neighbor pairs recomputed from original image histograms; no full nearest-neighbor search or training rerun','pairs':checks},indent=2)+'\n',encoding='utf-8')
files=[R/(prefix+'_knn_grid_dev.csv') for prefix in ['rgb','hsv','hsv12','moments','hog']]
files+=list((R/'hsv12_test').glob('*.csv'))+list((R/'neighbor_analysis').glob('*.csv'))
manifest={'scope':'Five configurations presented; original experiment archive retained unchanged. Figure source script reads only saved results and referenced images.','sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(HERE/'review'/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PASS: 3 report figures, 5-feature evidence table, 15 distance checks and source manifest.')
