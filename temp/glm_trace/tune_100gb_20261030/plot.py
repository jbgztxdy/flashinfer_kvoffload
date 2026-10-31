"""Draw the final trace comparison from audited samples, with explicit configuration labels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
p=Path(__file__).resolve().parent;root=p.parents[2]
rows=json.loads((p/'summary.json').read_text())['rows']
font=Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc');fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
records=[]
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,setting in zip(axes,('g100_l20000','g50_l20000')):
 selected=[]
 for m,b in ((1,512),(8,64),(8,128)):
  found=[r for r in rows if r['variant']=='final' and not r['confirmation'] and r['kind']=='lru' and r['setting']==setting and r['mtp']==m and r['requests']==b]
  if not found:continue
  selected.append(found[0])
 x=np.arange(len(selected));w=.19
 for i,(key,name,color) in enumerate((('local_us','纯device','#3978b7'),('memcpy_us','Memcpy＋attention','#e28542'),('original_ours_us','优化前固定配置','#8da18a'),('ours_us','整行路径＋推荐参数','#23866b'))):
  y=[r[key]/1000 for r in selected];ax.bar(x+(i-1.5)*w,y,w*.94,label=name,color=color)
  for xx,yy in zip(x+(i-1.5)*w,y):ax.text(xx,yy+.04,f'{yy:.2f}',ha='center',fontsize=8,color=color)
 for i,r in enumerate(selected):
  ax.plot([i-.43,i+.43],[r['reference_us']/1000]*2,'--',color='#24364b',label='max(计算,搬运)参考' if i==0 else None)
 ax.set_xticks(x,[f"MTP{r['mtp']}／{r['requests']}请求" for r in selected])
 ax.set_title(('100' if setting.startswith('g100') else '50')+' GB/s＋20 µs')
 ax.set_ylabel('完整窗口平均 GPU 耗时（ms）');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 if selected:ax.set_ylim(0,max(r['memcpy_us'] for r in selected)/1000*1.2)
 records+=selected
axes[0].legend(frameon=False,fontsize=8,ncol=2,loc='upper left')
fig.suptitle('GLM-5.3完整trace · staged · LRU4096槽/请求 · BF16 · A100-SXM4',fontsize=13)
assets=root/'docs/assets/glm53_100gb_optimization_20261030';assets.mkdir(parents=True,exist_ok=True)
fig.savefig(assets/'trace_optimization.png',dpi=180,bbox_inches='tight');plt.close(fig)
(p/'figure_data.json').write_text(json.dumps(records,indent=2)+'\n')
print('plotted',len(records),'source cells')
