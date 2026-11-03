"""Read-only checks of fixed-model correctness, calibration, integration and provenance."""
from pathlib import Path
import hashlib,json,time
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
load=lambda name:json.loads((p/name).read_text())
cal,ver,integ,over=map(load,['calibration.json','verification.json','integration.json','overhead.json'])
assert cal['complete'] and integ['complete'] and len(cal['settings'])==15
assert all(x['bad']==x['early']==0 for x in ver['reservation_cases'])
assert len(ver['reservation_cases'])==32
assert all(x['bad']==0 for x in ver['arithmetic_cases']+ver['stage_cases'])
errors=[];lat_errors=[]
for case in cal['settings']:
 assert case['payload_ok']
 bw,lat=case['bandwidth_GBps'],case['latency_ns']
 if bw:
  assert all(abs((262144+13)*1152/t/1000/bw-1)<=.05 for t in case['samples_us'])
  errors.append(abs(case['measured_GBps']/bw-1))
 if lat:
  assert all(abs(t-lat)<=.1*lat+600 for t in case['dependent_read_samples_ns'])
  lat_errors.append(max(abs(t-lat) for t in case['dependent_read_samples_ns']))
for case in ver['payload_cases']:
 assert case['payload_ok']
 if case['GBps']:assert abs(case['mean_GBps']/case['GBps']-1)<=.05
for record in [cal,ver,integ,over]:assert record['resources_start']['foreign']==record['resources_end']['foreign']==[]
header=root/'include/flashinfer/attention/far_link_emu.cuh';h=sha(header)
assert h==sha(p/'include_v2/flashinfer/attention/far_link_emu.cuh')==ver['emulator_sha256']==over['emulator_sha256']
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==load('baseline.json')['include/flashinfer/attention/mla_host_batch.cuh']['sha256']
assert sha(root/'include/flashinfer/attention/mla.cuh')==load('baseline.json')['include/flashinfer/attention/mla.cuh']['sha256']
for record in load('build_v2.json'):assert record['exit_code']==0
libraries=[]
for mode in ['emu','normal']:
 for f in sorted((p/('jit_v2'+('_normal' if mode=='normal' else ''))).rglob('*.so')):
  libraries.append(dict(mode=mode,path=str(f.relative_to(p)),sha256=sha(f)))
assert len(libraries)==4
binding=load('binding.json');assert binding['complete'] and binding['payload_ok'] and binding['graph_replays']==2
assert load('binding_guard.json')['old_header_rejected_before_build']
assert binding['resources_start']['foreign']==binding['resources_end']['foreign']==[]
for rel,digest in binding['sources'].items():assert sha(root/rel)==digest
manifest=load('manifest.json')
for rel,digest in manifest['files'].items():assert sha(root/rel)==digest
result=dict(timestamp=time.strftime('%FT%T%z'),version=2,status='passed',model='Issue-time CAS max-plus,ceil(ns)service; latency excluded from serialization.',emulator_sha256=h,gather_sha256=sha(root/'temp/sim_link/emu_gather.cu'),binding_sha256=sha(root/'temp/sim_link/emu.py'),host_header_unchanged=True,math_header_unchanged=True,reservation_cases=32,reservation_trials=sum(x['trials'] for x in ver['reservation_cases']),arithmetic_cases=len(ver['arithmetic_cases']),stage_cases=len(ver['stage_cases']),real_payload_cases=len(ver['payload_cases']),standalone_calibration_settings=15,integration_tests=3,public_binding_graph_replays=2,mismatched_old_header_rejected=True,bandwidth_acceptance_relative=.05,latency_acceptance='abs(error)<=10%target+600ns forallpositiveL samples',max_standalone_mean_bandwidth_error=max(errors),max_positive_latency_error_ns=max(lat_errors),libraries=libraries,evidence=[dict(path=n,sha256=sha(p/n)) for n in ['manifest.json','build_v2.json','verification.json','calibration.json','integration.json','overhead.json','binding.json','binding_guard.json']],limits='GPU3 same-deviceHBM. No physicalNVLink/H800 or freshMLAperformance. Single active launch per module (CTA slots shared). Separate modules independent clocks. Packet rounding<1ns;50/100GBps exact10/20ps/B. Software overhead isnotfullMLA decomposition. Historicalraw retained,notrelabeled.')
(p/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['libraries','evidence']},indent=2))
