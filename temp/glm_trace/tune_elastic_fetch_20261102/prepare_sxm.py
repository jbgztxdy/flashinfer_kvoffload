import shutil
from pathlib import Path
p=Path(__file__).resolve().parent;old=p.parent/'route_select_20261102'
shutil.copy2(old/'resources.py',p/'resources.py')
s=(old/'bench.py').read_text()
s=s.replace('VARIANT_INCLUDE=DEST.parent/"tune_qk_cursor_20261102/include_qk_cursor"','VARIANT_INCLUDE=DEST/("include_"+VARIANT)')
(p/'bench_sxm.py').write_text(s)
