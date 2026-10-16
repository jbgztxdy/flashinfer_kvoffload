# 批量 MTP 稀疏 MLA：Host KV 直访 kernel

## 结论

新增 `BatchMLAHostBatchKernel`（`include/flashinfer/attention/mla_host_batch.cuh`），把原来只支持 1 个 query 的 Host 直访路径扩展到任意 batch：每行一个 query token，各自有自己的 top-k，支持 1～128 个请求 × MTP 8 个 token（8～1024 行），64 的倍数个 head。接口不变：`BatchMLAPagedAttentionWrapper.run(..., host_page_start=...)`；单 query、2048 页的形状仍走原来的单 batch kernel。

A100-PCIE-40GB（GPU 3）、BF16、top-k 2048 的测试结果：

- **MTP top-k（按实测的 token 间重叠回放）**：56 个格子中 **50 个**同时快于 Memcpy 和 Direct。有 Host 命中时耗时为 Memcpy 的 0.59～1.00、Direct 的 0.42～1.19。例如 1024 行、10% 命中：**4.05 ms**，Memcpy 6.26 ms，Direct 9.05 ms；30% 命中：**11.13 ms**，Memcpy 13.23 ms，Direct 25.99 ms。
- 没赢的 6 格全部集中在**极低命中率或最小 batch**，此时传输量极小、瓶颈是固定开销：8 行 0%（1.22×）、8 行 5%（1.19×）、32 行 0%（1.13×）、512 行 0%（1.08×）、1024 行 0%（1.01×）、8 行 25%（1.00×，持平）。
- **独立随机 top-k（悲观情形）**：token 之间几乎没有可去重的页（去重后仍剩 93%～99%），新 kernel 仍比 Memcpy 快（0.63～1.00），但对 Direct 只在 64 行以上、高命中率时快 2%～6%；小 batch 比 Direct 慢 10%～49%。

所有 case 都做了 FP32 抽样核对，最大绝对误差 0.0071；29 项单元测试通过；racecheck、synccheck、memcheck 均为 0 错误。

## 设计

一次 cooperative launch，全部在 GPU 上完成，不需要 CPU 参与：

1. **去重与布局**：每个 CTA 处理若干行，把 Host 条目排在行首、HBM 条目在后。Host 页按整个 batch 去重：第一个 CAS 到该页的行认领它，并记录（行号, 在本行认领中的序号）。每个 warp 每轮只做一次 `atomicAdd`。
2. **grid barrier 后编号**：每个 CTA 自己对各行的认领数做前缀和，得到按行顺序排列的 staging slot，并把 Host 条目改写成 `-(slot + 1)`。再一次 barrier 后，fetch CTA（默认 8 个）按 slot 顺序以 64 页为一块，用 `cp.async` 把 Host 页搬到 HBM staging，每块发布一个就绪标志；搬完后它们从列表末尾领取 attention item。
3. **attention item**：worker CTA 按静态轮转从列表前端领取 item（行, 64-head tile, KV 段），沿用原生 MLA 的 tile 循环，先算 HBM tile，只在含 Host 条目的 tile 前等待对应标志。大 batch 时每行一个 item；小 batch 时剩余的行拆成 HBM 段和细粒度的 Host 段（每段至少 64 条），最后由整个 grid 按 LSE 合并。
4. 超出 staging 容量的页保留原页号直接读 Host，结果仍然正确。每次 launch 用 epoch 区分标志，不需要清零。

## 调优中排除的方向和发现

- **shared memory 对齐**：kernel 自己的静态 `__shared__` 让 dynamic smem 从 592 字节处开始，`cp.async` 的目标与源地址在 128 字节上错位，每个请求被拆开、sysmem 扇区翻倍（fetch 从约 26 GB/s 掉到约 22 GB/s），HBM tile 加载也变慢。与原生 kernel 同一编译单元里的 `extern __shared__ smem` 是同一个符号，`__align__` 会被覆盖，因此改用独立名字 `host_batch_smem`。
- **寄存器溢出**：用 shared memory 队列动态领取 item 时，item 值不再是 warp-uniform，tile 循环里的 shared 偏移被溢出到栈，HBM 部分慢约 11%。改为由 blockIdx 和循环计数推出的静态轮转后，寄存器 241～252 个，零栈。
- **fetch 方式**：寄存器加载（`ld.global.cg`）只有约 15 GB/s，`cp.async` 双缓冲约 26 GB/s；fetch CTA 取 8～12 个效果相同。按行收集的稀疏 fetch 单元在去重后很稀疏，只有约 21 GB/s。
- **准备阶段**：逐条目认领、等待其他行认领的 slot 号，在 1 个请求的 8 行上要 20～30 µs；改为 barrier 后统一解析后约 13 µs（8 行）～37 µs（1024 行）。
- **Host 等待的放置**：把 Host 条目平均分到每个 KV 段，会让每个 item 都在等待 PCIe 时占住 SM（64 行 30%：974 → 1176 µs）；把 Host 部分集中在行首一个大段，则数据到达后还剩一个约 22 tile 的 item（尾巴约 100 µs）。最终方案是小 batch 时把 Host 部分单独拆成细粒度 item，排在 HBM item 之后。

## 零 Host 页的快速路径

0% 命中时 kernel 原本仍要付全套去重开销。两处优化：

1. **每行快速路径**：某行没有 Host 条目时，`remap` 直接拷贝 `kv_indices`（两者在该情况下本就相同），跳过 ballot 和跨 warp 前缀和。
2. **整批跳过**：`unique == 0` 时跳过 resolve 重写、第二次 grid barrier 和 fetch。`unique` 在每个 CTA 上算出的值相同，分支是 grid-uniform 的，cooperative barrier 不会失配。

0% 命中下的效果（µs）：8 行 57.8→54.2、16 行 71.5→66.7、32 行 131.5→127.4、128 行 452.0→429.9、1024 行 2962.7→2914.8。**所有规模都改善 2%～7%，无退化。**

实现时踩过一个坑，已记入代码注释：`fetcher` 这个变量同时承担「是否取数」和「从 item 列表哪一头遍历」两个职责。最初只按 `unique` 关掉它，导致后端 CTA 与 worker 用相同起点重复遍历并争用 claim（0% 命中、512 行：1514→1784 µs）。修法是把遍历方向固定按 `blockIdx.y` 划分，与 `unique` 解耦。

## 局限

- 0% Host 命中时仍有固定开销（8 行 +9.3 µs）。上述快速路径消除了 resolve、第二次 barrier 和 fetch，但 `prepare` 的计数阶段（必须扫一遍 `kv_indices` 才知道有没有 Host 页）、第一次 grid barrier、以及 tail 行的 split + merge 跳不掉。要进一步消除需要在 plan 阶段（Python/FFI 层）就判定该批无 Host 页。
- 独立随机 top-k 下小 batch 不如 Direct：没有可去重的页时，先写 HBM staging 再读回，比直接读多一次搬运。
- Memcpy 的 GPU 时间不含 CPU gather（8 行时约 0.3 ms，1024 行时约 24 ms）；实际系统中 gather 在关键路径上，新 kernel 不需要这一步。
- 只在 A100 上测试；64-tile 配置（H100 等 smem 更大的 GPU）只编译检查过，零溢出，未实测。
- host_slot 每个 Host 页占 4 字节；rows < 32768、每行 < 65536 页（由 launcher 检查）。

## 文件

| 文件 | 内容 |
|---|---|
| `include/flashinfer/attention/mla_host_batch.cuh` | kernel 与 launcher |
| `csrc/batch_mla_run.cu`、`csrc/batch_mla_binding.cu` | `run_host_batch` FFI |
| `flashinfer/mla/_core.py` | `run(host_page_start=...)` 分派、内部 buffer、`host_staging_rows` / `host_fetch_ctas` |
| `tests/attention/test_mla_host_batch.py` | 29 项正确性测试（重复 launch、索引更新、溢出、CUDA Graph、FP16/128 head） |
| `temp/bench_glm52_host_batch_kernel.py` | 三方对比（`--index-mode mtp/random`） |
| `temp/host_attention_batch_kernel/` | `mtp_v3.*`（最终）、`mtp.*`（零 Host 快速路径之前）、`random.*`、`attn.*`（纯计算）、sanitizer 日志 |
| `temp/bench_glm52_host_rounds.py` + `temp/host_attention_rounds/` | 跨 decode 轮持久缓存的模拟实测 |
| `docs/mla_host_pcie_roofline.md` | PCIe 瓶颈的量化分析与各优化方向的评估 |

复现：

```bash
export PYTHONPATH=. FLASHINFER_WORKSPACE_BASE=$PWD/temp/jit_host_batch_kernel FLASHINFER_CUDA_ARCH_LIST=8.0 TMPDIR=$PWD/temp
F="0 0.05 0.1 0.15 0.2 0.25 0.3"
/mnt/data/mhy-vllm-venv/bin/python temp/bench_glm52_host_batch_kernel.py --index-mode mtp --fractions $F \
  --output temp/host_attention_batch_kernel/mtp_v3.json
python3 temp/summarize_glm52_host_batch_kernel.py temp/host_attention_batch_kernel/mtp_v3.json
```
