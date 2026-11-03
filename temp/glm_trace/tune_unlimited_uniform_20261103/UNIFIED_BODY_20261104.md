# 统一active/pure计算主体：拒绝

以clean为基线，ACTIVE/SERVICE模板bool变为每行开始时CTA一致的runtime active。paired QK/item QPE/jointMDO/PV各只保留一份主体，服务操作受runtime bool控制，pure_wait与active wait分支保留。所有屏障、数学、真实unique copy/cache/staging写入和LRU保持。与phase（分离主体以减少循环判断）方向相反，不是原样重跑phase。

实际加载库SHA核对后cuobjdump确认REG255/STACK8/static1152，6687静态指令、299MMA、75BAR；clean9336/598/91、STACK16。静态减少不等同动态执行量减少，runtime分支仍有成本。

正常GPU3同卡HBM BF16MTP8R128K2048LRU4096、256位置32步×3coldpass3169.919990/3158.944003/3197.472006µs，均3175.445333。近期clean3127.680µs，跨轮非夹测，无采用信号，拒绝不测100GB。两项K65/K2048数学/LSE/padding/QPEstride80/Graph/cache全有效载荷通过，性能采样及全部最终cache载荷通过。raw/source/实际库SHA与codegen独立保存；无新全suite/sanitizer/物理/EMU结论，生产81093保持。
