# 同输入bank wavefront对照

正常GPU3 K2048/R64首次目标launch，pytest通过；与此前profile相同限制，不是正式热trace。direct冲突load255854、store1212697，wavefront load91476116、store3053269。clean本次：

```json
{
  "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "1142280",
  "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "1184900",
  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "93628546",
  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "2809863"
}
```

store冲突两路径接近；clean读冲突更多，但不能将全部short scoreboard归因bank conflict，wavefront包含必要多phase事务，不是请求数。准备/LRU/数学/output均在计数内，不作source因果归属。后续若改shared布局须定位具体指令，不能仅凭总数改swizzle。生产未改，raw/report完整保留。
