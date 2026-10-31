"""Hardware counter comparison remains a separate synthetic diagnostic, not benchmark evidence."""
import csv,hashlib,io,json
from pathlib import Path
p=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_text())
state=read(p/'counters_state.json');assert len(state)==2 and all(x['exit_code']==0 for x in state)
rows={};captures=[]
for variant,marker in [('base','BatchMLAHostBatchKernel'),('joint_qk','BatchMLAJointQKKernel')]:
 text=(p/('counters_'+variant+'.csv')).read_text();start=text.index('"ID",');records=list(csv.DictReader(io.StringIO(text[start:])))
 assert len(records)==10 and len({r['Kernel Name'] for r in records})==1 and all(marker in r['Kernel Name'] for r in records)
 assert all(r['Block Size']=='(32, 4, 2)' and r['Grid Size']=='(1, 108, 1)' for r in records)
 parsed={r['Metric Name']:dict(unit=r['Metric Unit'],value=float(r['Metric Value'].replace(',',''))) for r in records};rows[variant]=parsed
 correct=read(p/('counters_'+variant+'_correctness.json'));assert correct['complete'] and correct['synthetic'] and not correct['trace_performance_claim'] and correct['max_abs_error']<.01
 assert correct['header_sha256']==sha(p/(variant+'.cuh')) and correct['source_sha256']==sha(p/'diagnose_counters.py')
 assert not correct['resources_start']['foreign'] and not correct['resources_end']['foreign']
 assert correct['checked_cache_slots']==128*2560
 captures.append(dict(variant=variant,kernel_name=records[0]['Kernel Name'],metrics=parsed,correctness_sha256=sha(p/('counters_'+variant+'_correctness.json')),csv_sha256=sha(p/('counters_'+variant+'.csv'))))
assert rows['base'].keys()==rows['joint_qk'].keys()
comparison=[dict(metric=k,unit=rows['base'][k]['unit'],base=rows['base'][k]['value'],joint_qk=rows['joint_qk'][k]['value']) for k in rows['base']]
result=dict(complete=True,synthetic=True,trace_performance_claim=False,reference_claim=False,captures=captures,comparison=comparison,
 limitation='Single1024x2048syntheticcachelaunchwith25%newfarrows,independentHBMonGPU3,NsightComputeapplicationreplayanddefaultcachecontrol. Warpstallpercentagesoverlap/arenotadditive. Cannotattributerealmixedtraceperformanceormainreferencegapfromthissample. EveryapplicationreplayreconstructsmutableLRU/epochstate;outputandall327680cacheslotsverified. InitialimportfailurewasbeforeGPUlaunchandpreserved.')
(p/'counters_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
