from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path('outputs/paper')
RUNS={'CALCE-CS2_35':'calce_cs2_35','CALCE-CS2_37':'calce_cs2_37','NASA-B0005':'nasa_b0005','TJU-CY25_1':'tju_cy25_1'}

def best(rows):
 return {'mae':min(r['mae'] for r in rows),'rmse':min(r['rmse'] for r in rows),
         'r2':max(r['r2'] for r in rows),'AE':float(np.mean([r['AE'] for r in rows])),
         'RE':float(np.mean([r['RE'] for r in rows]))}

single=[];multi=[]
for dataset,folder in RUNS.items():
 base=ROOT/folder
 one=json.loads((base/'results.json').read_text(encoding='utf-8'))['folds']
 single.append({'dataset':dataset,**best(one)})
 ms=json.loads((base/'multistep_results.json').read_text(encoding='utf-8'))['metrics']
 for h in sorted({r['horizon'] for r in ms}):
  rows=[r for r in ms if r['horizon']==h]
  multi.append({'dataset':dataset,'horizon':h,'mae':min(r['mae'] for r in rows),
                'rmse':min(r['rmse'] for r in rows),'r2':max(r['r2'] for r in rows)})
payload={'aggregation':{'MAE/RMSE':'minimum over start points','R2':'maximum over start points','AE/RE':'mean over start points'},'single_step':single,'multi_step':multi}
(ROOT/'paper_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
for name,rows in [('single_step',single),('multi_step',multi)]:
 with (ROOT/f'{name}_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(payload,indent=2,ensure_ascii=False))
