from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;root=p.parents[2];u=p.parent/'tune_unlimited_uniform_20261103'
assert not(p/'direct.cuh').exists()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();s=base
# Activate exactly the new direct math for emulator experiments, never fallback silently.
s=s.replace('#ifndef FLASHINFER_FAR_EMU\n// Fast normal-build BF16','// EMULATOR EXPERIMENT\n// Fast normal-build BF16').replace('#ifndef FLASHINFER_FAR_EMU\n    using DirectPVKernelTraits','// EMULATOR EXPERIMENT\n    using DirectPVKernelTraits').replace('#endif  // !FLASHINFER_FAR_EMU','')
a=s.index('void direct_pv_attend_joint_host_tiles(');b=s.index('template <typename KTraits, typename Params, bool ROW_MERGE>',a);x=s[a:b]
x=x.replace('    cp_async::wait_group<0>();\n    __syncthreads();','    cp_async::wait_group<0>();\n    FAR_EMU_WAIT(2,hi%2);\n    FAR_EMU_WAIT(2,lo%2);\n    __syncthreads();')
x=x.replace('  cp_async::wait_group<0>();\n  __syncthreads();','  cp_async::wait_group<0>();\n  FAR_EMU_WAIT_ALL(2);\n  __syncthreads();');s=s[:a]+x+s[b:]
values={'staged':base,'direct':s,'uniform':(p.parent/'tune_uniform_warps_20261103/candidate.cuh').read_text()}
for v,header in values.items():
 (p/(v+'.cuh')).write_text(header);shutil.copytree(root/'include',p/('include_'+v));(p/('include_'+v)/'flashinfer/attention/mla_host_batch.cuh').write_text(header);shutil.copy2(p/'batch.cuh',p/('include_'+v)/'flashinfer/attention/far_link_emu.cuh')
shutil.copytree(root/'include',p/'include_base');(p/'include_base/flashinfer/attention/mla_host_batch.cuh').write_text(base)
for name in ['compile.py','candidatepaths.py','experiment.json','plans.cpp']:shutil.copy2(u/name,p/name)
# Existing realtrace benchmark already performs output and fullcache payload verification.
s=(p.parent/'tune_uniform_warps_20261103/bench.py').read_text();s=s.replace("if VARIANT!='base':","if VARIANT=='uniform':").replace("if VARIANT!='base':assert","if VARIANT=='uniform':assert")
s=s.replace('w.host_stage=True;w.host_split=False','w.host_stage=VARIANT!=\'direct\';w.host_split=False').replace('staged=True,','staged=VARIANT!=\'direct\',')
s=s.replace('        result[\'passes\'].append(row);save(target,result)',"        if VARIANT=='direct':row['actual_far_read_bytes']=[n*1152 for n in row['reference_misses']]\n        result['passes'].append(row);save(target,result)")
s=s.replace("refs[t]=(w._host_batch_buffers[3]<=-(R*CAP+1)).sum()", "refs[t]=((w._host_batch_buffers[3]>=0)&((w._host_batch_buffers[3]&~(1<<30))>=near)).sum() if VARIANT=='direct' else (w._host_batch_buffers[3]<=-(R*CAP+1)).sum()")
(p/'bench.py').write_text(s)
for v in ['staged','direct']:
 s=(u/'test_clean.py').read_text().replace('test_clean_output','test_'+v+'_output')
 if v=='direct':s=s.replace('w.host_stage=staged','w.host_stage=False')
 (p/('test_'+v+'.py')).write_text(s)
(p/'test_uniform.py').write_text((p.parent/'tune_uniform_warps_20261103/test_uniform.py').read_text())
s=(p.parent/'tune_uniform_warps_20261103/run.py').read_text();s=s[:s.index("m=json.loads((p/'manifest.json')")]
s=s.replace("str(p/'jit_candidate') if variant=='candidate' else", "str(p/('jit_'+variant)) if variant!='base' else")
s+='''save()
for v in ['staged','uniform','direct']:
 run('correctness_'+v,v,['-m','pytest','-q','-s','-p','no:cacheprovider','-p','jitpaths','-p','candidatepaths',str(p/('test_'+v+'.py'))],True)
for setting,order in [('g100_l0',['base','staged','uniform','direct']),('g100_l20000',['direct','uniform','staged','base'])]:
 for v in order:
  run(v+'_'+setting,v,['-u',str(p/'bench.py'),'--kind','lru','--mtp','8','--requests','128','--positions','256','--passes','3','--settings',setting,'--fetch-ctas','24','--host-split','off'])
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
''';(p/'run_integration.py').write_text(s)
f=p/'.gitignore';f.write_text(f.read_text()+'include_*/\njit_*/\nplans.so\n')
m=json.loads((p/'manifest.json').read_text());m['integration_headers']={v:hashlib.sha256(header.encode()).hexdigest() for v,header in values.items()};m['production_host_sha256']=hashlib.sha256(base.encode()).hexdigest();m['integration_scope']='PrivateEMU builds; directgate enabled onlyexperiment and perpair stage waits added. All actualbytes/hooks retained; production untouched.';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
