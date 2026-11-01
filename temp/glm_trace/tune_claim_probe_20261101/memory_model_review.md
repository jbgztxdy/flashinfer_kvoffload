# Claim-state reasoning

PTX将`.relaxed`列为strong memory operation；同GPU、同generic proxy、完全重叠的32-bit访问可满足其morally-strong条件，相应规则约束单副本原子性、RMW与逐地址一致性。[NVIDIA PTX ISA §8.7、8.10.3、8.10.5](https://docs.nvidia.com/cuda/parallel-thread-execution/)

以下是基于该规则与当前代码生命周期的实现推论，并非用少量测试代替完整内存模型证明：

- prepare期间每个host_slot从0经唯一成功CAS变1，再由该owner发布encoded(row,rank)；原fetch清零发生在prepare/resolve两个grid阶段之后。此阶段非零不会回到零。
- claim_probe使用`ld.relaxed.gpu.global.u32`，读到非零即可保留“已认领”结果；读到零仍执行原CAS，所以竞争时仍由CAS唯一选owner。不消费尚未发布的rank，resolve沿用原grid同步后再读。
- claim_store保留每次原CAS。只有成功owner以`st.relaxed.gpu.global.u32`发布encoded值；其他CAS比较0，对1/encoded均不写。发布不需要atomicExch的旧值，rank消费者仍在原grid同步后。
- 元数据为同GPU的32-bit对齐分配；不把这些操作当作Host或peer payload同步。跨launch复用沿用现有串行workspace/stream约定，原fetch必须清理所有unique（含overflow）后才进入下一launch。

独立typed kernel保留所有原定义，原36/72完整代码字、实际新launch、113项正常测试（含跨CTA重复/页集合切换/Graph重放/全cache载荷）及sanitizer检查分别作为编译与运行证据。SASS中的条件CAS/GPU强load或store需核对；动态操作节省和速度不能由静态数推断。
