# MTP 相邻 token 的 top-k 重叠

## 结论

同一请求中，MTP 8 个相邻 token 的 top-2048 选择高度重叠。按请求去重后，需要读取的 KV 行数约为不去重时的 30%；只看距离当前位置超过 4K 的「远端」行（更可能被卸载到 Host），去重后约为 41%。也就是说，远端 Host 流量可以减少约 2.4 倍。这个比例在 32K、64K、128K 上下文下基本不变（0.40～0.41）。独立随机选择的基线则从 0.71 升到 0.93，说明这里的重叠来自模型本身的注意力结构，而不是上下文短造成的偶然碰撞。

把这个结果代入批量基准（[mla_host_batch_baseline.md](mla_host_batch_baseline.md)）：

- 1024 行、20% 命中：Direct 实测 17.9 ms，全本地 2.83 ms。如果 Host 读取按请求去重，Host 部分按 0.41 缩放，估算约 9.0 ms，即全本地的 3.2×（去重前为 6.3×）。
- 按「额外开销不超过全本地的 20%」估算，去重后能承受的 Host 命中率约从 0.6% 提高到 1.5%。

所以去重值得做，而且收益稳定；但只靠它，仍远不足以支撑 20%～30% 的 Host 命中。另一个可用的事实是：每个 token 的选择中有 43%～47% 落在最近 4K 个位置内，这部分始终留在 HBM 即可。

## 方法

本地没有带 DSA 索引器的模型（GLM-5 / DeepSeek-V3.2），因此使用同为 MLA 结构的 DeepSeek-V2-Lite（`/mnt/data/mhy/model/deepseekv2-16b`）做代理：

- 每层、每个 query token，用该层真实的 softmax attention（16 个 head 取平均）对所有 ≤ 当前位置的 key 取 top-2048，作为这一层这个 token 的选择。DSA 索引器本来就是训练来逼近这个分布的，但两者并不相同。
- 每份文档在上下文后半段均匀取 32 组，每组 8 个连续位置，模拟 MTP 一次 verify 的 8 个 token。这里假设 8 个 draft 全部被接受，即使用真实文本 token；被拒绝的 draft 可能更不相似。
- 语料：Pile-CC 中的 3 本长书，外加拼接的 flashinfer `include/flashinfer/attention/*.cuh` 代码。代码的 tail loss 很低（0.05～0.43），内容重复度高，结论与书籍基本一致。书籍在 128K 时 tail loss 为 2.6～3.1，说明模型在这个长度下仍在正常读取文本。
- 模型权重 31 GB，GPU 3 上还有其他用户的 ANN 任务占用 3.5～6 GB 显存，因此脚本每次只把一层搬到 GPU 上执行，峰值显存为 7.4 GiB。

## 结果

数值为 4 份文档 × 27 层 × 32 组的中位数。「并集比例」= |8 个集合的并集| / (8 × 2048)，越小表示重叠越多；1/8 = 0.125 表示完全相同，1 表示完全不重叠。

| 上下文 | 2 token 并集比例 | 4 token | 8 token | 随机基线（8 token） | 远端行占比 | 远端 8 token 并集比例 | 随机远端基线 | 相邻 Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32K | 0.668 | 0.447 | 0.297 | 0.752 | 0.44 | 0.396 | 0.712 | 0.507 |
| 64K | 0.666 | 0.453 | 0.307 | 0.866 | 0.43 | 0.412 | 0.855 | 0.492 |
| 128K | 0.674 | 0.452 | 0.305 | 0.930 | 0.47 | 0.410 | 0.927 | 0.494 |

各组之间差异较大：远端 8 token 并集比例的 p10～p90 约为 0.29～0.55。各层之间也有差异，第 0～2 层约 0.28，第 3、7、19～21 层约 0.50～0.56。逐文档、逐层的数据见 [summary.md](../temp/mtp_overlap/summary.md) 和 [summary.json](../temp/mtp_overlap/summary.json)。

## 局限

- 代理模型是 16 头的 DeepSeek-V2-Lite，用的是真实 attention，而不是 GLM 5.2 的 DSA 索引器；需要用真实索引器的 top-k trace 复核。
- 只测到 128K；更长的上下文需要用更省显存的方式计算分数。
- 「远端 = 距离超过 4K」只是对 Host 放置策略的一个假设；实际的流量节省取决于哪些页被放到 Host。
- 9.0 ms 是按比例缩放得到的估算；带去重的 kernel 还没有实现和实测。

## 复现

```bash
for c in 32768 65536 131072; do
  TMPDIR=$PWD/temp /mnt/data/mhy-vllm-venv/bin/python temp/measure_mtp_topk_overlap.py \
    --device 3 --context $c --docs temp/mtp_overlap/docs/*.txt \
    --output temp/mtp_overlap/overlap_ctx$c.json
done
/mnt/data/mhy-vllm-venv/bin/python temp/summarize_mtp_topk_overlap.py temp/mtp_overlap/overlap_ctx*.json
```

每档上下文 4 份文档共约 2～6 分钟。`temp/mtp_overlap/docs/` 中的书籍文本由 Pile-CC 的 `part-00001` 第 138450、160310 行和 `part-00002` 第 119619 行导出。
