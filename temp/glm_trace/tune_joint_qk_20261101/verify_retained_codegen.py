"""Bridge the host-variable cleanup to measured code only through all-kernel identity."""
import argparse,gzip,hashlib,json,re,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--normal',action='store_true');a=parser.parse_args();suffix='_normal' if a.normal else ''
folder=p/('jit_retained_joint_qk'+suffix)
while len(list(folder.rglob('*.so'))) != (4 if a.normal else 2):time.sleep(10)
old=read(p/('compiler_joint_qk'+suffix+'_summary.json'))['records']['joint_qk']
records={};libraries=[]
for lib in sorted(folder.rglob('*.so')):
 sass=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-sass',str(lib)],text=True)
 resources=subprocess.check_output(['/usr/local/cuda-13.1/bin/cuobjdump','--dump-resource-usage',str(lib)],text=True)
 name='retained_joint_qk'+suffix+'_'+lib.stem
 with gzip.GzipFile(filename=str(p/(name+'_sass.txt.gz')),mode='wb',mtime=0) as f:f.write(sass.encode())
 (p/(name+'_resources.txt')).write_text(resources)
 kernels={}
 for body in re.split(r'\n\s*Function : ',sass)[1:]:
  symbol=body.splitlines()[0].strip();instructions=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',body)
  usage=re.search(r'Function '+re.escape(symbol)+r':?\s+([^\n]+)',resources);assert usage
  kernels[symbol]=dict(symbol=symbol,instructions=len(instructions),instruction_sha256=hashlib.sha256('\n'.join(instructions).encode()).hexdigest(),resource=usage[1])
 assert kernels==old[lib.name],lib.name
 records[lib.name]=kernels;libraries.append(dict(name=lib.name,sha256=sha(lib)))
assert records.keys()==old.keys()
m=read(p/'retained_source_manifest.json');assert m['measured_header_sha256']==sha(p/'joint_qk.cuh') and m['reviewable_header_sha256']==sha(p/'retained_joint_qk.cuh')
assert m['math_header_sha256']==sha(p/'retained_joint_qk_mla.cuh')
d=dict(complete=True,candidate_adopted=False,build='normal' if a.normal else 'EMU',unchanged_kernel_instances=sum(map(len,records.values())),all_original_and_new_targets_identical=True,source_manifest=m,libraries=libraries,records=records)
assert d['unchanged_kernel_instances']==(72 if a.normal else 36)
(p/('retained_codegen'+suffix+'_summary.json')).write_text(json.dumps(d,indent=2)+'\n')
print({k:d[k] for k in ['complete','build','unchanged_kernel_instances','all_original_and_new_targets_identical']})
