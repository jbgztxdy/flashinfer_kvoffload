"""Verify all Ampere instruction and control words in original and host-cleanup builds."""
import gzip,hashlib,json,re
from pathlib import Path
p=Path(__file__).resolve().parent;read=lambda f:json.loads(f.read_text())
results=[]
for suffix in ['', '_normal']:
 proof=read(p/('compiler_joint_qk'+suffix+'_summary.json'));entries=[];identity=[]
 for lib in proof['records']['base']:
  stem=lib.removesuffix('.so');paths={v:p/(v+suffix+'_'+stem+'_sass.txt.gz') for v in ['base','joint_qk','retained_joint_qk']}
  def parse(f):
   return {s.splitlines()[0].strip():re.findall(r'0x[0-9a-f]{16}\b',s) for s in re.split(r'\n\s*Function : ',gzip.open(f,'rt').read())[1:]}
  tables={v:parse(f) for v,f in paths.items()}
  for symbol,old in tables['base'].items():
   new=tables['joint_qk'][symbol];assert old and old==new,(lib,symbol)
   entries.append(dict(library=lib,symbol=symbol,machine_words=len(old),identical=old==new,pure_device='BatchMLAPagedAttentionKernel' in symbol))
  for symbol,old in tables['joint_qk'].items():
   new=tables['retained_joint_qk'][symbol];assert old and old==new,(suffix,lib,symbol)
   identity.append(dict(library=lib,symbol=symbol,machine_words=len(old),machine_word_sha256=hashlib.sha256('\n'.join(old).encode()).hexdigest()))
  assert all(len(tables[v][sym])==2*proof['records']['joint_qk'][lib][sym]['instructions'] for v in ['joint_qk','retained_joint_qk'] for sym in tables[v])
 results.append(dict(build=suffix or 'EMU',originals=entries,original_all_words_identical=all(x['identical'] for x in entries),cleanup_all_words_identical=True,cleanup_records=identity,original_kernel_instances=len(entries),cleanup_kernel_instances=len(identity)))
out=dict(complete=True,candidate_adopted=False,description='Compareall64bitmachinewordsfromcuobjdumpSASS,includingAmpereinstructioncontrolwords; ignoreELFcontainer/hostsymbols,notkernelencodings.',builds=results)
(p/'machine_word_identity.json').write_text(json.dumps(out,indent=2)+'\n')
print([{k:r[k] for k in ['build','original_all_words_identical','cleanup_all_words_identical','original_kernel_instances','cleanup_kernel_instances']} for r in results])
