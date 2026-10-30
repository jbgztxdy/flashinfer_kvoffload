"""Measured performance heatmaps; unavailable cells stay masked."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
meta=json.loads((DEST/"experiment.json").read_text())
rows=json.loads((DEST/"matrix.json").read_text())
names=[s["name"] for s in meta["settings"]]
out=ROOT/"docs/assets/glm53_kernel_matrix_20261029"
out.mkdir(parents=True,exist_ok=True)
for kind in ("nocache","lru"):
    fig,axes=plt.subplots(2,2,figsize=(18,17),layout="constrained")
    used=0
    for i,mtp in enumerate((1,8)):
        batches=meta["batches"][str(mtp)]
        sub=[r for r in rows if r["kind"]==kind and r["mtp"]==mtp
             and (kind=="lru" or r["host_fraction_target"]==.3)]
        used+=len(sub)
        for j,key in enumerate(("ours_to_memcpy","ours_to_local_mean")):
            ax=axes[i,j];data=np.full((len(batches),len(names)),np.nan)
            for r in sub:data[batches.index(r["requests"]),names.index(r["setting"])]=r[key]
            cm=plt.get_cmap("RdYlGn_r").copy();cm.set_bad("#eeeeee")
            im=ax.imshow(np.ma.masked_invalid(data),aspect="auto",cmap=cm,norm=LogNorm(.5,8))
            ax.set_xticks(range(len(names)),names,rotation=65,ha="right",fontsize=9)
            ax.set_yticks(range(len(batches)),batches)
            ax.set_ylabel("Requests")
            ax.set_title(f"MTP{mtp}: Ours / {'Memcpy' if j==0 else 'All-local mean'}")
            for (a,b),v in np.ndenumerate(data):
                if np.isfinite(v):ax.text(b,a,f"{v:.2f}",ha="center",va="center",fontsize=7,
                                         color="white" if v>3.5 else "black")
            cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.02,ticks=[.5,1,2,4,8]);cb.ax.set_yticklabels(["0.5","1","2","4","8"])
    title="Real trace + LRU, all 1024 positions" if kind=="lru" else "No LRU, target Host 30%, four trace groups"
    expected=len(names)*sum(len(v) for v in meta["batches"].values())
    fig.suptitle(f"{title} | {used}/{expected} measured | lower is better",fontsize=16)
    fig.savefig(out/f"{kind}.png",dpi=160)
    plt.close(fig)
print(out)
