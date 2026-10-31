# A100：两个概率tile联合发布、减少PV同步（2026-11-01）

候选拒绝。相对冻结JQ控制，100GB0/20µs分别慢0.182%/1.115%，50GB20µs仅快0.048%。没有证据支持进一步扩大完整窗口/跨形状/物理链路矩阵；停止此概率联合发布机制，不扫描tile、unroll、fetch或同步间隔。相对b6基线的正收益也保留，但不能将JQ已有改进计为本候选收益。

JQ已先算完两个32KV tile的QK并共同完成MDO重缩放，随后仍两次调用原生PV，各有一次KPE→P覆写保护和一次P发布barrier。本候选把两次P量化、rowsum及共享写入提前合并，用一个CTA barrier发布两份P，再按原hi-PV→hi-refill→lo-PV→lo-refill顺序消费。每对tile的4个原生PV barrier变为1个；原补数前barrier、奇数尾tile的原生PV和全部ready等待保留。原联合MDO barrier已经覆盖两个KPE stage的QK读取完成；两份P写入不同stage，和m_wg互不重叠；hi补数不修改lo的P。浮点在线softmax方程、P量化/rowsum表达式及各输出PV的MMA次序相同，仅把第二次d更新移到第一次PV之前，d和PV输出累加独立。此依赖论证不等于性能收益。

固定原64head/256线程、32KV/2stage、dynamic147968B/static2304B、108CTA/active1、fetch64行双buffer/f24、精确LRU/真实cache载荷。canonical mla.cuh、模拟器及原5304格矩阵完全不改。新helper从canonical原生P写入/PV消费表达式逐字提取，只实例化专属JointPVTraits；原34EMU/68normal kernel指令与资源精确相同，机器字和控制字也全部相同。新REG252/STACK0/LOCAL0；JQ控制为REG251/STACK0，资源数字不用于推断因果。

105项原正常BF16/FP16测试和4项K2048长队列/容量512及4096/淘汰/零miss/Graph/完整cache payload测试均通过，共109项新测试。实际正常BF16 Graph只有一个BatchMLAJointPVKernel，108CTA、block[32,4,2]、dynamic147968、active1，最大绝对误差0.004128。冻结JQ控制复用其原109测试和binary证据，reused_joint_control.json记录原件SHA，不算新增测试。

另行Compute Sanitizer诊断：正常BF16、432行、K97的实际probe，synccheck零错误；racecheck仅检查第一个匹配BatchMLAJointPVKernel的launch，零hazard、零错误、零warning。独立dispatch记录确认名称匹配。这不是完整trace或穷尽race证明，工具时间不计性能。首个synccheck在等待正确性时触发附加超时，尚未创建GPU context，仅结束已核对的自有collector/子进程，原log及stop JSON保留；第二次在测试就绪后使用launch-timeout=0并成功。短测首次因漏拷贝plans.cpp在源码预检退出，无有效计时；补齐与JQ逐字相同的plan源码、加入CPU输入存在检查后，只重跑尚未完成的短测，日志/state追加、全部原失败保留，没有重跑通过的测试。

GPU3 A100-PCIE、NUMA1、far为独立同卡HBM模拟，不是真实NVLink。MTP8/R128、LRU4096/请求、f24/off、256位置×3冷启动pass，JQ→候选→b6固定顺序；全部步/pass算术平均，两参考按原同硬件数据与公式。

| 链路 | b6基线 µs | JQ控制 µs | 联合发布P µs | 较JQ降低 | 较b6降低 | Memcpy µs | 纯device µs | 逐步max µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g100_l0 | 3554.389 | 3447.531 | 3453.792 | -0.182% | +2.830% | 6071.883 | 2556.960 | 2699.882 |
| g100_l20000 | 3581.429 | 3432.533 | 3470.795 | -1.115% | +3.089% | 6092.000 | 2556.960 | 2699.967 |
| g50_l20000 | 5284.021 | 5272.843 | 5270.304 | +0.048% | +0.260% | 8490.763 | 2556.960 | 4960.187 |

[完整审计](../temp/glm_trace/tune_joint_pv_20261101/performance_audit.json)含9格/6个对b6配对、1728计时launch/108有效segment，另列3个对JQ的三方比较；每步/每pass/命中/unique miss/guard/源码SHA/快照齐全，两参考无缺失，没有未完成raw。全部runner已terminal，候选未安装。JQ原SXM确认/保留1398在独立目录等待原门槛，不受本候选影响。主目标仍active，100GB完整trace距参考约25%–28%仍未解决。
