import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
r=subprocess.run([str(p/'protocol')],capture_output=True,text=True,timeout=30)
(p/'protocol_stdout.log').write_text(r.stdout);(p/'protocol_stderr.log').write_text(r.stderr)
assert r.returncode==0,r.stderr
d=json.loads(r.stdout);assert d['errors']==0
d['source_sha256']=hashlib.sha256((p/'protocol.cu').read_bytes()).hexdigest();d['binary_sha256']=hashlib.sha256((p/'protocol').read_bytes()).hexdigest()
(p/'protocol.json').write_text(json.dumps(d,indent=2)+'\n');print(r.stdout,flush=True)
