# Kernel内粗粒度pool配额候选

从CPU容量模型实现到完整单cooperative kernel：每CTA固定global roundrobin包数配额q或q+1，每pool整段优先分给其query行owner中剩余配额最大者，不能本地分配的尾段再全局spill。每次分配整段，不逐包atomic，所有CTA leader重复确定性规划并只写自己list。临时remain/spill放query尚未加载的shared区，不增加shared；所有必要规划、真实unique copy、stage/cache载荷、LRU、数学/ready协议都在计时内。重复规划成本明确，不作零成本假设。CPU镜像160随机case（含0行pool）exact quota和每包唯一通过；CPU旧真实末步约94.72%包可本地，不等于GPU完整trace实测比例。

首版忽略unique0时row_slot_base不更新，热路径读旧边界导致spill与零quota不一致、GPU检查停滞。只终止本任务pytest PID3718148，首版源/JIT/log/state保留；无性能数据，不能算正确性通过。v2在unique0完全跳过边界规划，独立新include/JIT重编。

v2两项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过。正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3coldpass3523.712017/3530.911982/3565.792017µs，均3540.138672µs，慢于clean约3128和等负载listcontrol3159.147，无采用信号，拒绝安装不测100GB。不能据端到端负差精确分摊重复planner、供需顺序、寄存器；严格均衡不足以保证收益。raw/header/实际加载库SHA重新核验见poolquota_v2_audit.json。无新sanitizer/完整矩阵/EMU/物理结论，生产81093不变，runner均结束。
