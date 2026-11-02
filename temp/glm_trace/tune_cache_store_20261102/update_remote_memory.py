from pathlib import Path
import hashlib
p=Path('/WORK/PUBLIC/luyouy_work/rtx/MEMORY.md')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e3df236ed55997c74a3ffc8ec9572545488d7240a9392a3edaedf9c1d8c0c5cd'
entry='''

## MLA续接状态（2026-11-02）

- 用户明确偏好轻量迭代：“每次不用验证这么多，确认机制有效、在不断优化就行。”同范围机制只做定向正确性及少量关键同参新旧对照，不重复全套pytest/sanitizer/矩阵；单cooperative kernel、精确launch级LRU、真实缓存载荷/字节、计时/参考正确仍保持。
- 15:14只读核对：原1004362(host)和1004363(peer)均COMPLETED0；1040680(bulk)FAILED1，nvcc链接找不到libcudadevrt/libcudart_static，尚无bulk GPU测量。1089379依赖已满足，PENDING/Priority；1089380仍afterok依赖。watcher4160646自然结束，1174145仍活，不重复启动/提交。
- 两原H800作业为旧header29667eaf/四步旧trace/末半段steady口径，不含本机新优化；host旧大batch输旧Memcpy、peer大batch有收益，均保留，不能与当前assistant1024/正确Memcpy完整基线混比。14份原始JSON与日志已下载到本机flashinfer/temp/glm_trace/tune_cache_store_20261102/h800_archive并绑定manifest SHA。原排队源码/脚本未改，无新Slurm提交。
- 本机A100 GPU3/NUMA1已保留QK swizzle地址游标，生产SHA754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79；只从48c2推进游标，原unroll4/加载MMA/数学/LRU/字节/FAR保持。4项BF16定向载荷/Graph测试；完整1024x3反向同参100GB0/20us快1.492%/1.276%，50GB0.067%，正常同卡HBM0.183%、HostPCIe0.005%持平。GPU3同卡HBM不是NVLink，未新测FP16/SXM/H800，不外推全场景。两个独立cache streaming store、去未使用QPE copy候选未显著改善，已拒绝。所有本轮runner结束。生产/证据只在本机，未迁移覆盖H800排队树；整体100GB差距约24%–25%，目标未完成。本机HANDOFF最新入口及docs/mla_host_a100_followup_20261102.md为准。
'''
with p.open('a') as f:f.write(entry)
print('MEMORY_UPDATED',hashlib.sha256(p.read_bytes()).hexdigest())
