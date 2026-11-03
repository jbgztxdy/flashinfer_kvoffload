# 实际pool关联

GPU3正常clean真实32步一pass，事件外最后一步读remap和slot_pool。350495远端引用全部same pool；177643unique，slot_pool127次变化（128连续pool段），每pool56–3343unique。raw保存完整counts、源码/实际库SHA、数学采样/最终cache载荷检查。只最后一步，不能外推全部trace。

现有全局20页packet可按首slot所属pool分配，保持原ready索引，无需每页额外flag或取消去重。CPU模型：

```json
{
  "chunks": 8883,
  "cross_pool_rows_upper_bound": 1065,
  "unique_rows": 177643,
  "cross_pool_rows_fraction": 0.00599517008832321,
  "mechanism": "assign each existing global20-row packet to its first slot pool, preserving packet ready protocol; exact chunk ownership; only boundary rows cross pool",
  "limitation": "CPU model from final-step actual counts, not GPU scheduler or liveness proof"
}
```

下一完整实现须构造每CTA拥有packet列表并确保等待时能推进所有所拥有pool，不得只服务当前计算pool而产生边界等待环；pool负载差异大，需粗粒度平衡。尚无新调度GPU实现/收益，不重复此前独立均分页/行候选。
