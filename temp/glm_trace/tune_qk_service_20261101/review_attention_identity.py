"""Prove numerical helper text is inherited from the tested QP control after renaming."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
sources={'control_qp':((p/'control_qp.cuh').read_text(),'qkpv_'),'qks':((p/'qks.cuh').read_text(),'qks_'),'qks_shared':((p/'qks_shared.cuh').read_text(),'qkss_')}
def function(source,name):
 pos=source.index('void '+name+'(');start=source.rfind('template <',0,pos);first=source.index('{',pos);depth=1;end=first+1
 while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
 return source[start:end]
records=[]
for name in ['write_o','load_query','load_q','load_kv','compute_qk','softmax_publish','compute_pv','consume']:
 values={v:function(s,prefix+name).replace(prefix,'attention_') for v,(s,prefix) in sources.items()}
 assert len(set(values.values()))==1,name
 records.append(dict(helper=name,normalized_sha256=hashlib.sha256(values['control_qp'].encode()).hexdigest()))
(p/'attention_identity_review.json').write_text(json.dumps(dict(complete=True,numerical_helpers_after_rename_identical=records,
 limitation='Does not claim schedule/storage/FP32 equivalence to productionJQ or CPE; QP already changes grouping. Produce/progress/lifetime/commonwork differences require actual CUDA gates.'),indent=2)+'\n')
print('NUMERICAL_HELPERS_IDENTICAL',len(records))
