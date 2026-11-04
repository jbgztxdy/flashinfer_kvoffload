from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'resources_gpu3.py').read_text().replace("'--id=3'","'--id=0,1'")
s=s.replace('r=next(csv.reader(io.StringIO(raw)))','rows=list(csv.reader(io.StringIO(raw)))\n    r=rows[0]')
s=s.replace("z[0].strip()==g['uuid']","z[0].strip() in [r[1].strip() for r in rows]")
s=s.replace("not foreign and g['used_mib']<512 and g['utilization']<10","not foreign and all(int(r[2])<512 and int(r[4])<10 for r in rows)")
(p/'resources_nvlink.py').write_text(s)
for v,src in [('retained','retained'),('clean','clean'),('next_packet_qk','next_packet_qk'),('deferred_packet_normal','deferred_packet_normal'),('base','production_service22_control')]:
 s=(p/('bench_'+src+'.py')).read_text()
 s=s.replace('from resources_gpu3 import','from resources_nvlink import').replace("=='1', NUMA_POLICY","=='0', NUMA_POLICY")
 s=s.replace('tune_general_20261031/gpu3_correctness.lock','kernel_report_20261029/gpu01.lock').replace('"--id=3"','"--id=0,1"')
 s=s.replace("0,'peer',0)","0,'peer',1)")
 s=s.replace("name='native_unlimited'","name='nvlink_native'").replace('emulated=True','emulated=False')
 s=s.replace("target=DEST/(VARIANT+", "target=DEST/('nvlink_20261104_'+VARIANT+")
 s=s.replace("config=dict(normal_build=True,","residency=kv.residency,config=dict(physical_nvlink=True,normal_build=True,")
 (p/('bench_nvlink_'+v+'.py')).write_text(s)
