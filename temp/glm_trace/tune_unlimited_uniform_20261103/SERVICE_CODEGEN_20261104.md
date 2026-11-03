# 实际加载库代码生成核对

审计脚本audit_service_codegen.py先核对raw记录的实际加载.so SHA，再cuobjdump目标symbol，结果service_codegen_audit.json。

| normal目标 | REG | STACK | 目标内LDL/STL | 静态指令 | BAR | MMA |
|---|---:|---:|---:|---:|---:|---:|
| clean全CTA |255|16|10|9336|91|598|
| issue/stores noinline |255|0|0|8811|91|598|
| retained direct |254|0|0|5156|46|299|

noinline确实消除了目标kernel的栈访问，但没有降低REG或改善完整耗时，故不能继续将这处spill视为已证实主要瓶颈。统计noinline只覆盖主symbol，不包括callee，不能把8811当全程序动态指令减少。全CTA有ACTIVE=true/false两套数学主体，静态MMA/BAR接近direct两倍，不是每行动态执行两遍，不是因果性能证明。若下一机制试共享计算主体，须区别既有phase候选（分离active/pure已降低运行服务检查），保留运行时控制开销，不能只凭代码体积推断收益。

本轮CPU审计，无新GPU计时、生产修改或100GB结论。
