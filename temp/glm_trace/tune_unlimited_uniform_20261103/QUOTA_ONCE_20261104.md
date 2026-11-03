# 容量配额单次规划负结果

poolquota_v2的确定性local-first+spill不变，由CTA0 leader一次性构造全部列表，各owner独立cursor保留与原重复规划相同包序；随后一个cooperative grid barrier发布。每CTA仍q/q+1包、逐行轮转。零unique跳过旧row_slot_base；所有必要GPU规划、串行列表写入、grid等待、真实KV/cache/staging写回均计时，未新增kernel或CPU规划。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3coldpass3659.135997/3667.231999/3700.671986µs，均3675.679994µs，比重复planner3540.138672µs更慢3.83%，拒绝安装、不追加100GB模拟。跨轮非夹测，不精确归因gridbarrier或串行写表；减少重复规划没有实测净收益。两项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过，性能采样与全部最终cache载荷通过。header/实际加载库SHA核验见quota_once_audit.json。无新sanitizer/全矩阵/物理/EMU结论，生产81093保持，runner结束。
