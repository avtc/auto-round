# AutoRound 环境变量配置

[English](./environments.md) | 简体中文

本文档介绍 AutoRound 使用的环境变量及其配置说明。

## 概述

AutoRound 通过 `envs.py` 模块提供统一的环境变量管理系统，支持懒加载求值与程序化配置。

## 可用环境变量

### AR_LOG_LEVEL
- **描述**：控制 AutoRound 默认日志级别
- **默认值**：`"INFO"`
- **有效值**：`"TRACE"`、`"DEBUG"`、`"INFO"`、`"WARNING"`、`"ERROR"`、`"CRITICAL"`
- **用途**：通过设置该变量控制 AutoRound 的日志详细程度

```bash
export AR_LOG_LEVEL=DEBUG
```

### AR_ENABLE_COMPILE_PACKING
- **描述**：启用编译打包优化
- **默认值**：`False`（等价于 `"0"`）
- **有效值**：`"1"`、`"true"`、`"yes"`（不区分大小写）表示启用；其他值表示禁用
- **用途**：启用后可在将 FP4 张量打包为 `uint8` 时获得性能优化

```bash
export AR_ENABLE_COMPILE_PACKING=1
```

### AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE
- **描述**：让 fused NVFP4 权重投影共用一个 weight global scale。该设置作用于 `q_proj`/`k_proj`/`v_proj` 与 `gate_proj`/`up_proj`，以满足 vLLM fused kernel 的要求。
- **默认值**：`True`（等价于 `"1"`）
- **有效值**：`"0"`、`"false"`、`"no"` 或 `"off"`（不区分大小写）表示关闭共享；其他值表示启用。
- **用途**：仅当导出的运行时不要求 fused 投影使用统一 global scale 时关闭。

```bash
export AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE=0
```

### AR_USE_MODELSCOPE
- **描述**：控制是否使用 ModelScope 下载模型
- **默认值**：`False`
- **有效值**：`"1"`、`"true"`（不区分大小写）表示启用；其他值表示禁用
- **用途**：启用后将使用 ModelScope 替代 Hugging Face Hub 下载模型

```bash
export AR_USE_MODELSCOPE=true
```

### AR_WORK_SPACE
- **描述**：设置 AutoRound 操作的工作目录
- **默认值**：`"ar_work_space"`
- **用途**：指定 AutoRound 存储临时文件和输出结果的自定义目录

```bash
export AR_WORK_SPACE=/path/to/custom/workspace
```

### AR_ROTATION_FIX_LINEAR_ATTN
- **描述**：控制旋转变换（SpinQuant）在混合注意力模型上使用的线性注意力离线 R1 修复开关。启用时，`layer.linear_attn` 投影会吸收 R1，并将 `input_layernorm` 的 gamma 折叠到其中，以保持离线 R1 的数学等价性。默认关闭以保持旧行为。仅由旋转变换代码路径读取。
- **默认值**：`"0"`
- **有效值**：`"1"`、`"true"`、`"yes"`、`"on"`（不区分大小写）表示启用；其他任意值表示禁用
- **用法**：在混合线性注意力架构上运行离线 R1 旋转时启用

```bash
export AR_ROTATION_FIX_LINEAR_ATTN=1
```

### AR_DISABLE_OFFLOAD
- **描述**：强制禁用 `OffloadManager` 中的权重卸载功能。在开发和调试时可跳过所有卸载/重载开销。
- **默认值**：`False`（等价于 `"0"`）
- **有效值**：`"1"`、`"true"`、`"yes"`（不区分大小写）表示禁用卸载；其他值保持默认行为
- **用途**：设置后将完全绕过权重卸载

```bash
export AR_DISABLE_OFFLOAD=1
```

### AR_DISABLE_DATASET_SUBPROCESS
- **描述**：禁用子进程方式进行数据集预处理。默认情况下，AutoRound 使用子进程确保所有临时内存在进程退出后被操作系统回收。
- **默认值**：`False`
- **有效值**：`"1"`、`"true"`（不区分大小写）表示禁用子进程；其他值表示启用子进程
- **用途**：设置后数据集预处理将在主进程中运行

```bash
export AR_DISABLE_DATASET_SUBPROCESS=true
```

### AR_ACT_SCALE
- **描述**：只用于研究性质，控制激活量化时对激活值最小/最大值的缩放系数。小于 1.0 的值会缩小裁剪范围，有助于减小离群值的影响。
- **默认值**：`1.0`
- **有效值**：任意浮点数，如 `0.8`、`0.9`、`1.0`
- **用途**：调整激活裁剪范围

```bash
export AR_ACT_SCALE=0.9
```

### AR_ENABLE_ACT_MINMAX_TUNING 
- **描述**：只用于研究性质，使用激活量化中最小/最大缩放参数（`act_min_scale`、`act_max_scale`）的调优。启用后，这些缩放参数将固定为 1.0。
- **默认值**：`False`（等同于 `"0"`）
- **有效值**：`"1"`、`"true"`、`"yes"`（不区分大小写）表示禁用调优；其他值表示保持调优
- **用途**：禁用激活最小-最大缩放参数的调优

```bash
export AR_ENABLE_ACT_MINMAX_TUNING=1
```

### AR_SEARCH_SCALE_RATIO
- **描述**：控制 `auto_round.data_type.int.search_scales` 中对称 INT 量化 scale 搜索的范围比例。搜索上界为 `nmax * AR_SEARCH_SCALE_RATIO`，其中 `nmax = 2^(bits-1)`。值越小搜索范围越窄（更快，但可能漏掉较优解）；值越大搜索范围越广（更慢，对离群权重可能更准）。
- **默认值**：未设置 → 走内置默认值（`0.5`，即 `nmax/2`）。
- **有效值**：正浮点数，如 `0.25`、`0.5`、`0.75`、`1.0`
- **用途**：覆盖默认的 scale 搜索范围

```bash
export AR_SEARCH_SCALE_RATIO=0.75
```

### AR_DYNAMO_CACHE_SIZE_LIMIT
- **描述**：在开启 `torch.compile`（除 Windows 外默认开启）时，将 `torch._dynamo` 的 `cache_size_limit`、`accumulated_cache_size_limit` 与 `recompile_limit` 提升到的最小值。同一个被编译的量化函数会被 transformer block 内的所有 linear 层（q/k/v/o_proj、gate/up/down_proj 等）复用，但每层权重 shape 不同，按层的静态重编译会很快超过 dynamo 默认上限（8），导致打印告警并退回 eager。提高该上限可保留静态 shape 编译（性能最佳），仅增加缓存条目数。
- **默认值**：`16`
- **有效值**：正整数
- **用途**：当模型单个 block 内 linear 权重 shape 种类超过 16 时（较少见）可调大。

```bash
export AR_DYNAMO_CACHE_SIZE_LIMIT=32
```

### AR_MODEL_FREE_SHARD_PARALLELISM
- **描述**：控制 model-free 量化时同时处理的权重 shard 数量。增大该值可提高资源利用率，但会占用更多内存。
  - 自动策略（变量**未设置**时）：`shard_count // 4`，最大 **10**，最小 **1**。例如：8 个 shard → 2 个 worker；40 个 shard → 10 个 worker。
  - 实际并行数始终不超过 shard 总数。
- **默认值**：未设置 → 走自动策略（`shard_count // 4`，最大 10，最小 1）
- **有效值**：任意正整数，不限于特定值，如 `2`、`4`、`6`、`8`；不能整除 shard 数时会自动均匀分配，末批处理剩余 shard，结果正确
- **用途**：覆盖自动并行策略，手动指定并行数

```bash
export AR_MODEL_FREE_SHARD_PARALLELISM=4
```

### AR_AUTO_SCHEME_NSAMPLES
- **描述**：控制 AutoScheme 评分时使用的校准样本数默认值，仅在 `AutoScheme.nsamples` 未显式设置时生效。
- **默认值**：未设置 → 16
- **有效值**：任意正整数，如 `8`、`16`、`32`
- **用途**：覆盖 AutoScheme 的自动样本数选择

```bash
export AR_AUTO_SCHEME_NSAMPLES=1
```

### AR_AUTO_SCHEME_BATCH_SIZE
- **描述**：控制 AutoScheme 评分时使用的批大小默认值，仅在 `AutoScheme.batch_size` 未显式设置时生效。
- **默认值**：未设置 → 走内置启发式规则（低GPU内存模式为 8，普通模式为 1）
- **有效值**：任意正整数，如 `1`、`2`、`4`
- **用途**：覆盖 AutoScheme 的默认批大小

```bash
export AR_AUTO_SCHEME_BATCH_SIZE=1
```

### AR_AUTO_SCHEME_SEQLEN
- **描述**：控制 AutoScheme 评分时使用的校准序列长度默认值，仅在 `AutoScheme.seqlen` 未显式设置时生效。
- **默认值**：未设置 → 走内置启发式规则（MoE 模型为 128，其他为 256）
- **有效值**：任意正整数，如 `256`、`512`、`1024`
- **用途**：覆盖 AutoScheme 的默认序列长度（2-bit 方案通常在 `1024` 时效果更好）

```bash
export AR_AUTO_SCHEME_SEQLEN=1024
```

### AR_AUTO_SCHEME_NO_SERIAL_FALLBACK
- **Description**: Turn a parallel-scoring failure into a hard error instead of falling back to serial scoring. Useful when the serial pass is known to be unable to run (or would take workers-count times longer): completed schemes and batches are persisted in the per-scheme cache, so a rerun scores only the failed parts.
- **Default**: unset -> parallel scoring failure falls back to serial
- **Valid Values**: `1`, `true`, `yes`
- **Usage**: Set this to fail fast on parallel scoring errors

```bash
export AR_AUTO_SCHEME_NO_SERIAL_FALLBACK=1
```

### AR_AUTO_SCHEME_CACHE
- **描述**：存放可持久复用的 AutoScheme 单方案评分 JSON 文件。该目录独立于用于临时工作数据的 `AR_WORK_SPACE`。
- **默认值**：`~/.cache/auto_round`
- **有效值**：任意可写目录路径
- **用途**：将可复用的 AutoScheme 评分结果保存到其他缓存目录

```bash
export AR_AUTO_SCHEME_CACHE=/path/to/auto_scheme_cache
```

### AR_ENABLE_AUTO_SCHEME_PARALLEL
- **描述**：启用 AutoScheme 候选方案之间的多进程并行。可与 `AR_DISK_STREAM_MODEL=1` 同时使用；此时每个 worker 会构建独立的 meta 模型骨架并分别流式加载 block。当并发 worker 可能耗尽主机内存或显存时，请将其关闭。
- **默认值**：`"1"`（满足多进程要求时并行评分各方案）
- **有效值**：`"1"`、`"true"`、`"yes"`（不区分大小写）表示启用并行评分；其他值表示关闭
- **用途**：运行 AutoScheme 前将其设为 `0`，以强制串行评分候选方案

```bash
export AR_ENABLE_AUTO_SCHEME_PARALLEL=0
```

### AR_NEUQI_COARSE
- **描述**：NeUQI 搜索在精细加性细化之前探索的粗粒度（对数间隔）scale 候选数量。
- **默认值**：`"64"`
- **有效值**：任意正整数
- **用法**：调低可加速搜索，调高可进行更穷举的 scale 扫描

```bash
export AR_NEUQI_COARSE=64
```

### AR_NEUQI_FINE
- **描述**：NeUQI 搜索中每个粗粒度候选对应的精细（加性）scale 细化候选数量。
- **默认值**：`"32"`
- **有效值**：任意正整数
- **用法**：调低可加速搜索，调高可提升 scale 分辨率

```bash
export AR_NEUQI_FINE=32
```

### AR_NEUQI_BACKEND
- **描述**：NeUQI 零点扫描的后端链。`"auto"`（默认）在 CPU 上保持参考的分块 eager 扫描，在 CUDA 上优先用 `auto_round_extension.triton.neuqi_sweep` 中的手写 Triton 内核服务批量扫描（寄存器驻留：每个权重元素只加载一次，所有（候选，零点）损失都在寄存器中计算），Triton 不可用时回退到 `torch.compile` 融合扫描；`"triton"` 强制使用 Triton 内核；`"compile"` 在任意设备上强制使用 `torch.compile` 融合扫描；`"eager"` 始终使用参考扫描。所有后端在完全相同的候选网格上计算完全相同的损失（选择仅在近似并列时不同：少于约 0.1% 的组会翻转，RTX 3090 上实测最坏相对损失差约 5e-5，Triton 内核的并列特征与其完全一致），并消除大规模批量专家搜索中暴力网格的显存带宽瓶颈（200 万组的 RTX 3090 扫描实测：eager 12.3 秒 → 单候选融合 0.59 秒 → 编译批量 0.49 秒 → Triton 0.22 秒，约 57 倍）。每一级失败都会永久逐级回退：Triton → 编译批量 → 编译逐候选 → eager。Triton/编译批量阶段需要启用 `AR_NEUQI_BATCH`（默认启用）。
- **默认值**：`"auto"`
- **有效值**：`"auto"`、`"triton"`、`"compile"`、`"eager"`（无法识别的值按 `"eager"` 处理）
- **用法**：固定某一阶段做 A/B 对比，或彻底禁用编译

```bash
export AR_NEUQI_BACKEND=eager
```

### AR_NEUQI_LAYOUT
- **描述**：融合零点扫描表达式的内存布局。扫描需要对每个整数零点沿组轴归约平方误差，两个轴可互换布置：`"last"` 布置为 `[组数, 零点数, 组大小]`，在连续的末维上归约（Triton/CUDA 的经典融合形态；中间维归约在 RTX 3090 上实测不快于 eager）；`"mid"` 布置为 `[组数, 组大小, 零点数]`（在 eager TensorIterator 路径与编译后的 CPU 后端上更快）。`"auto"`（默认）按设备选择：CUDA 用 `"last"`，其余设备用 `"mid"`。两种布局计算的损失在 fp32 组内求和顺序之外完全一致（仅末位 ulp 并列）。
- **默认值**：`"auto"`
- **有效值**：`"auto"`、`"last"`、`"mid"`（无法识别的值按设备的 `"auto"` 规则处理）
- **用法**：在特定设备上对两种布局做 A/B 对比

```bash
export AR_NEUQI_LAYOUT=mid
```

### AR_PRESINQ_BACKEND
- **描述**：Pre-SINQ 变换的 Sinkhorn 循环后端。`"auto"`（默认）在 CUDA 上使用 `auto_round_extension.triton.presinq_sinkhorn` 中的手写 Triton 内核（每次迭代三个 fp64 内核 —— 权重分块只读取一次、两个 std 都在寄存器中计算，确定性的两阶段列归约，以及带容差的失衡追踪器），其他设备使用参考 eager 循环。RTX 3090 实测：比 std 复用后的 eager 循环快 2.1–2.8 倍（约为原始实现的 4.3 倍），fp64 ulp 级一致性（约 1e-15），并且 MoE 池化范数折叠（巨大的拼接消费者矩阵，如 Hunyuan-A13B 类层）可在 24 GiB 内运行而 eager 循环会 OOM。`"triton"` 强制使用 Triton 内核（任何失败都会永久回退到 torch 循环 —— 包括无 CUDA 时的强制尝试）；`"eager"` 始终使用 eager 循环；`"compile"` 强制 `torch.compile` 融合图（可选：RTX 3090 实测最好情况仅持平、大矩阵慢达 20%）。eager 循环本身与上一版本位级一致且快约 1.8 倍（每次迭代的 std 只计算一次并复用，跳过不需要的缩放后矩阵物化）。
- **默认值**：`"auto"`
- **有效值**：`"auto"`、`"triton"`、`"eager"`、`"compile"`（无法识别的值按 `"eager"` 处理）
- **用法**：固定后端做 A/B 对比，或在测量结果不同的架构上切换

```bash
export AR_PRESINQ_BACKEND=eager
```

### AR_STREAM_MEM_INVENTORY

- **类型**：布尔值（`1`/`true`/`yes` 启用；默认关闭）
- **描述**：流式量化诊断开关。启用后，zero-shot 循环每 16 个 block 输出一次按 GPU 的显存分解（`[stream-mem] ...`）：分配器视角（alloc/reserved）、按类别统计的张量（`block:<k>` 暂存 block 权重、`embeddings`、`nonblock:<...>` 初始化阶段创建的模块、`chain` 校准 fp/q 隐状态及 kwargs），以及 `other = alloc - tracked`（临时对象、打包缓冲、优化器状态）。用于查看主 GPU 上到底驻留了什么、为何占用如此之大。与 `AR_SCHEME_MEM_INVENTORY`（AutoScheme 评分池）互补。

### AR_DISABLE_TUNING_FANOUT

- **类型**：布尔值（`1`/`true`/`yes` 启用；默认关闭）
- **描述**：关闭 RTN/NeUQI zero-shot 路径中的多 GPU 逐模块调优扇出（即 `[OptRTN] tuning fan-out: ... across N GPUs` 轮询分发，它会把模块权重搬到各工作设备）。设置该环境变量后，所有 scale/zp 搜索都在主设备上串行执行。两种方式下逐模块结果完全一致；这是一个隔离/取证开关（单流 Triton 启动、无扇出线程），不是提速开关。显式传入 `parallel_tuning=True` 配置项时仍以配置为准。不影响 block-parallel tuning（BPT）或数据驱动 SignRound 调优，它们有各自的开关。

### AR_NEUQI_BATCH
- **描述**：全候选批量零点扫描。`"auto"`（默认）在 CUDA 上把每轮的全部粗/精 scale 候选折叠为每个组块一次融合内核调用（单候选融合后，剩余墙钟时间主要消耗在逐候选的调度与簿记启动上）；`"on"` 在任意可融合设备上强制批量扫描（用于 A/B 或测试）；`"off"` 保持逐候选循环。批量内核在内核内完成零点最小化，每次启动只输出 `[组数, 候选数]` 的最优损失与获胜零点。选择遵循与顺序扫描相同的首个最小值并列规则；在单候选融合（RTX 3090 实测 21 倍）之上，这是同一候选网格上的第二级加速。批量调用一旦失败（例如更大的符号中间量导致显存不足），进程将永久回退到逐候选融合扫描处理其余候选。
- **默认值**：`"auto"`
- **有效值**：`"auto"`、`"on"`、`"off"`（无法识别的值按 `"auto"` 规则处理）
- **用法**：固定为逐候选循环以对批量阶段做 A/B 对比

```bash
export AR_NEUQI_BATCH=off
```

### AR_NVFP4_E5M3_CACHE_HP_WEIGHT
- **描述**：控制 `NVFP4E5M3QuantLinear` 是否在首次前向后缓存解量化得到的高精度权重，而不是每次调用都从打包的 FP4 权重重新解量化。
- **默认值**：`False`（等价于 `"0"`）
- **有效值**：`"1"`、`"true"`、`"yes"`、`"on"`（不区分大小写）表示启用缓存；其他值表示禁用缓存
- **用途**：当重复推理吞吐比内存占用更重要时可启用。当前实现会在缓存高精度权重后释放 `weight_packed` 和 `weight_scale`，因此稳态内存占用会增大，且之后无法再切回打包存储。

```bash
export AR_NVFP4_E5M3_CACHE_HP_WEIGHT=1
```

### AR_DISK_STREAM_MODEL
- **描述**：启用后，`AutoRound(model=<path>, ...)` 会将模型构建为 meta 设备骨架，而不是先把整个 checkpoint 完全加载到 CPU 内存；随后按需从 checkpoint 的 safetensors 分片中流式加载每个解码器块的真实权重——在该块被使用前（校准、调优或 `AutoScheme` 敏感度评分）才实体化，用完后立即释放回 meta。这样峰值 CPU 内存基本保持平稳，而不会随 checkpoint 大小成比例增长。非块参数（embedding、`lm_head`、最终归一化层）体积通常较小，仍会一次性加载。文本模型的 AutoScheme 评分也支持与默认启用的并行评分组合使用；每个 worker 会流式加载自己的 block 副本。
- **默认值**：`False`
- **有效值**：`"1"`、`"true"`、`"yes"`(不区分大小写)表示启用；其他任何值表示禁用
- **用途**：用于量化体积超过可用 CPU 内存 + GPU 显存总和的 checkpoint。仅在 `model` 为字符串(本地目录)路径时生效，对已加载的模型对象无效。

```bash
export AR_DISK_STREAM_MODEL=1
```

### AR_POST_SCALE_REFIT
- **描述**: 调优结束后按组做最小二乘 scale 重拟合，整数网格与零点冻结。单步闭式解（冻结整数约束下的精确最优解）；存在 imatrix 时按 imatrix 加权；按构造保证加权 MSE 单调不增。可与任意 `AR_TUNE_RECIPE`（以及普通 minmax 初始化运行）组合；仅作用于标准 group size 的非对称 int 层（对称层保持原网格，记录一次日志）。
- **默认值**: `0`（关闭）
- **有效值**: `0`、`1`（也接受 `true`/`yes`）

```bash
export AR_POST_SCALE_REFIT=1
```

### AR_BIAS_CORRECT
- **描述**: 逐 Transformer block 的量化后偏置校正：在 block 残差流边界取 `b = 校准 token 均值(y_fp - y_q)`，吸收进该 block 喂入残差流的投影层（out_features == hidden 的最后一个 Linear/Conv1D；路由专家模块被降权，因为它们只在部分 token 上执行）。bias 缺失时自动创建——所有导出格式原生支持，vLLM 无需改动即可生效。qon 路径复用链式前向（零额外前向）；qoff 路径增加一次 no-grad 前向。仅限串行/流式——block-parallel worker 下硬报错（bias 不属于 worker result 文件）。可与任意 `AR_TUNE_RECIPE`、`AR_POST_SCALE_REFIT` 组合。
- **默认值**: `0`（关闭）
- **有效值**: `0`、`1`（也接受 `true`/`yes`）

```bash
export AR_BIAS_CORRECT=1
```

### AR_QOFF_NOISE
- **描述**: qoff（FP 参考链）调优解锁器。BPT/串行 qoff 调优针对 FP 参考输入优化每个 block，而部署的量化链根本不会产生这些输入——正是实测的部署失配回退。启用后，调优前向将逐通道量化噪声 `mean + std*eps`（上一个量化 block 的输出漂移统计量，`eps` 按 block 种子确定性生成）注入 FP 输入；缓存的 FP 输入绝不会被原地修改。block 0 跳过注入（其输入是 embedding）。守卫：qon 下（量化输入链已看到真实输入）、未设置 `AR_QOFF_NOISE_STATS`、统计文件缺失或宽度不匹配时均硬报错。
- **默认值**: `0`（关闭）
- **有效值**: `0`、`1`（也接受 `true`/`yes`）

```bash
# 1) 采集阶段（廉价，例如 --iters 0）：写出 block_<idx>.pt 统计
export AR_QOFF_NOISE_STATS=/mnt/bigdisk/qoff_stats
# 2) 调优阶段（qoff/BPT）：注入这些统计
export AR_QOFF_NOISE=1 AR_QOFF_NOISE_STATS=/mnt/bigdisk/qoff_stats
```

### AR_TOUCHUP_ITERS
- **描述**: BPT 之后的串行 qon 补调：当 block-parallel（qoff）运行完成所有 block 调优后，设置此环境的串行重跑会在真实量化链（qon）上对每个 block 补调 N 次迭代。每个 SignRound wrapper 以 BPT 调优得到的 (scale, zp) 对为锚点起始——恰好从并行运行停下的地方继续——舍入参数清零、边距仍可调；改进后的结果会覆盖该 block 的 result 文件，后续 apply/导出使用补调后的网格。运行签名包含补调次数：修改 N 会使过期的 resume 产物失效。守卫：仅限串行（worker 环境必须 unset）、要求量化输入链（qon）、要求 results 目录存在且每个 block 都有完整 result 文件。
- **默认值**: `0`（关闭）
- **有效值**: `0`（关闭），或小的正整数（通常 2–5）

```bash
# 1) BPT 阶段（结果在 AR_RESUME_DIR）
# 2) 量化链上的串行补调：
export AR_TOUCHUP_ITERS=5   # 同一个 AR_RESUME_DIR；去掉 --enable_block_parallel_tuning；启用 qon
```

### AR_TUNE_RECIPE
- **描述**: SignRound 调优路径（`--iters > 0`）的实验性初始化搜索配方。配方用搜索到的网格替换逐组 min/max 调优网格：`neuqi_*` 锚定联合（scale、整数零点）搜索（`neuqi_search_scale_zero`，imatrix 加权）；`opt_rtn_qon` 锚定对称 scale-clip 搜索。`neuqi_frozen_qon` 额外把 `min_scale`/`max_scale` 调优边距固定为 1.0（网格完全固定，仅调舍入）。`neuqi_fp` 与 `neuqi_qon` 的区别仅在于初始化 imatrix 来自哪条链（FP 参考链 vs 量化链）——流式 qon 下实时 imatrix 本身就是链一致的。配方适用于标准（非 tuple）group size 的 int 数据类型；不支持的布局保持 min/max 网格。
- **默认值**: 未设置（保持现状 min/max 初始化）
- **有效值**: `minmax_qon`（显式对照臂）、`neuqi_qon`、`neuqi_frozen_qon`、`neuqi_fp`（兼容 BPT/qoff）、`opt_rtn_qon`（仅对称）、`neuqi_it0`（零样本参考标记；需 `--iters 0`）
- **用法**: 在小模型上按 KLD 对比分量配方。`neuqi_*` 需要 `--asym`；`opt_rtn_qon` 需要对称。可与 `AR_POST_SCALE_REFIT`、`AR_BIAS_CORRECT` 组合。

```bash
export AR_TUNE_RECIPE=neuqi_qon   # + --iters 20 --asym --imatrix_enabled true
```

### AR_TUNE_ASYNC_LOSS
- **描述**： DDP 调优循环的无围栏最优参数选择。循环期间 loss 值完全不离开 GPU：每次迭代的 world 求和 loss 存入设备缓冲，与设备侧运行最小值比较（严格小于、平局先到先得，与立即读取的选择完全一致），布尔结果以受 CUDA 事件保护的 pinned 标志字节送达宿主。宿主每迭代轮询就绪标志（非阻塞），通过预存快照的指针交换完成晋升。围栏：循环结束时的有界 drain，以及当最旧标志滞后满一个迭代时的罕见有界等待（pending 上限为一个快照——与旧 delayed-loss 模式显存相同：主设备多一个调优参数槽，27B 稠密 block 约 1.3 GB）。`dynamic_max_gap > 0` 时保持立即读取（早停需要循环内数值）。
- **默认值**: `0`
- **有效值**: `1`, `0`
- **用法**: 按需开启全异步循环（移除每迭代最后一个宿主围栏，world=4 下约 15-19 ms/迭代）；额外快照槽在主设备占用约一份调优参数量大小的显存（27B 稠密 block 约 1.3 GB）。取代已移除的 AR_TUNE_DDP_DELAYED_LOSS 模式（该模式每迭代仍有一次围栏；异步路径严格更强）。

```bash
export AR_TUNE_ASYNC_LOSS=1   # 异步 loss 解析（设备端比较 + pinned 标志轮询）
```

### AR_TUNE_REPLICA_PACING
- **描述**： DDP 图化调优步的两轮派发：所有副本的 prepare（收集样本到静态缓冲）全部完成后（线程池闩锁即屏障），再统一触发 replay，使各 GPU 的前向/反向从锁步起点开始，而不是被 prepare 抖动（宿主 `.item()` 围栏、分页 H2D）拖出 34-73 ms 的相位漂移。配套的 prepare 改动：批次索引用宿主 int 列表（不再构造设备索引张量，select_batch 内无逐元素 `.item()` 同步），常驻 CPU 叶子经 `HostLeafPinner` 固定一次后异步拷贝。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 保持开启；仅在与早期版本做 A/B 对比时设 `0` 恢复单闭包 prepare+replay 派发。

### AR_TUNE_PREPARE_FENCE_FREE
- **描述**： 无围栏的图化 prepare：批次索引以宿主 int 列表传入（不再构造设备索引张量，select_batch 内无逐元素 `.item()` 同步），常驻 CPU 叶子首次复现后固定一次，使刷新拷贝成为异步 pinned H2D 而非分页暂存。一次性叶子（逐样本新拼接的 mask）始终原样透传不固定。设 `0` 可完整恢复旧的设备索引收集路径，便于 A/B 隔离测试。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 保持开启；仅在 A/B 测试中隔离该项改进的速度/显存影响时设 `0`。

```bash
export AR_TUNE_PREPARE_FENCE_FREE=0   # 旧收集路径（设备索引张量、分页 H2D）
```

### AR_TUNE_EXCH_OVERLAP
- **描述**： 将 DDP 梯度交换与反向传播尾部重叠。调优参数梯度被切分为约 12 个桶；每个桶的反向完成时（post-accumulate-grad 钩子在 eager 反向或图回放期间触发），由单一的交换线程在记录流上立即执行该桶的 sign-cast 交换（reduce-scather 均值 + int8 符号 all-gather），取代整个反向排空后的单次整体 allreduce。逐元素位一致（分桶不改变每个元素的折半树）；任何交换错误都会使运行停止。仅适用于纯 sign-SGD、fp32/bf16 传输且 EAGER 反向的路径（作为完成信号的 post-accumulate-grad 钩子不会在 CUDA 图回放时触发，因此 AR_TUNE_CUDA_GRAPHS=1 时自动回退整体交换）。桶停滞时会在有界超时后带诊断信息中止。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 保持开启；设 `0` 用于 A/B 隔离测试（恢复整体 `sync_grads`）。

```bash
export AR_TUNE_EXCH_OVERLAP=0   # 反向完成后的整体交换
```

### AR_TUNE_PROACTIVE_PREPARE
- **描述**： 图化 DDP 调优循环的下一迭代前瞻 prepare。整个 block 的洗牌调度在循环入口一次性物化（与逐迭代惰性抽取使用相同的 RNG 流），且第 i+1 次迭代的静态缓冲刷新在第 i 次迭代的 replay 轮返回后立即入队——此时 GPU 仍在执行第 i 次迭代。各设备的流 FIFO 为静态缓冲的写后读排序（它们仅被 replay i+1 读取），因此无需任何同步。设 `0` 恢复迭代开始时 prepare。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 保持开启；仅在 A/B 测试中隔离该项改进时设 `0`。

```bash
export AR_TUNE_PROACTIVE_PREPARE=0   # 迭代开始时 prepare（不前瞻）
```

### AR_TUNE_SERIAL_DELAYED_LOSS
- **描述**： 串行调优路径（无 DDP 的逐 block，以及 block 外的逐层）在设备端累积 loss（fp64，与旧的 Python 浮点求和相同的从左到右顺序），并仅在反向入队之后读取（逐层路径每 batch 一次，逐 block 路径在 batch 循环后一次），取代每个 batch 在 loss 与反向之间用 `loss.item()` 围栏阻塞宿主——GPU 不再因等待宿主唤醒并 enqueue 反向而在迭代中途空转。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 保持开启；仅在 A/B 隔离测试时设 `0`。

```bash
export AR_TUNE_SERIAL_DELAYED_LOSS=0   # 反向前逐 batch .item() 读取（旧行为）
```

### AR_TUNE_SERIAL_DEVICE_SCHEDULE
- **描述**： 串行调优循环（无 DDP）：整个 block 的洗牌调度在循环入口一次性物化为一块锁页主机张量加一个设备索引缓冲（与逐迭代惰性 `next_batch` 序列使用相同的 RNG 流），每次迭代的样本选择以设备端 `index_select` 执行，并为 Python 层列表分支提供主机索引行——消除宿主 RNG 状态机、可分页 H2D 以及逐元素张量索引的 D2H 同步。抽取与 gather 均为位一致。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 保持开启；仅在 A/B 隔离测试时设 `0`。

```bash
export AR_TUNE_SERIAL_DEVICE_SCHEDULE=0   # 惰性 next_batch + 主机列表索引（旧行为）
```

### AR_TUNE_CUDA_GRAPHS（串行调优路径）
- **描述**： 当启用 `AR_TUNE_CUDA_GRAPHS` 且调优循环以串行方式运行（单设备、无 DDP、列表输入且非 diffusion 路径、`gradient_accumulate_steps=1`、fence-free prepare 开启、未使用 `valid_token_mask`、非 LFQ block、loss 与调优设备一致）时，每个 block 的 前向+loss+反向 会在 2 个 eager 预热迭代后被捕获进 CUDA graph：每次迭代的样本 gather 刷新静态设备缓冲（`_copy_into_static`，形状/ dtype 漂移即停机），迭代主体变为一次 graph replay（预热迭代在捕获侧流上 eager 执行，使 autograd 节点诞生于捕获流）。梯度保持驻留（`set_to_none=False`），反向累积进捕获时固化的地址；优化器 step 与 best-params 快照保持 eager。当满足纯符号配置（无 scaler、momentum 为 0、无 dynamic 早停）且设备能容纳一份额外调优参数（带 VRAM 守卫）时，优化器 step 也会被捕获进 graph（整迭代捕获）：lr 从每组设备张量读取并逐迭代刷新（lr 随调度变化的情形保持精确），best-params 快照读取 graph 内的 PRE-STEP 暂存，从而位精确地保持（loss, step 前参数）配对。当后台 CUDA 线程（bg pack、checkpoint prefetch）仍活跃时捕获会推迟并记录一次日志。任何捕获/replay 失败都会记录错误并提示 `AR_TUNE_CUDA_GRAPHS=0`，随后停机。不符合条件或纯 CPU 的运行保持旧版串行 batch 循环。另见 DDP 变体在 `AR_TUNE_EXCH_OVERLAP` 下的约束（仅 eager 反向）。
- **默认值**: `1`
- **有效值**: `1`, `0`
- **用法**: 在 CUDA 上保持开启；在 A/B 测试中可关闭以隔离 graph 影响。

```bash
export AR_TUNE_CUDA_GRAPHS=0   # 禁用整体 graph 捕获（DDP replica step 与串行 step）
```

### AR_RESUME_DIR
- **描述**：设置为目录路径后，逐块调优循环会在每完成一个块后将进度写入该目录，并在针对同一目录的新一次运行中从第一个未完成的块继续——而不是在崩溃或被杀死后从第 0 块重新开始整个调优过程。
- **默认值**：未设置(不支持断点续跑)
- **有效值**：任意可写目录路径
- **用途**：用于大 checkpoint 的长时间量化任务，避免运行中途崩溃导致从头重跑的高昂代价。

```bash
export AR_RESUME_DIR=/path/to/resume/state
```

跨模式断点续跑：串行（`AR_DISK_STREAM_MODEL`）、流式（`--stream_quantization`）与块并行（BPT）三种执行路径共用同一组逐块清单（manifest）。任意一种模式中断后，可以用相同或不同的模式从断点继续：已完成块的权重分片会被直接采用（不会被覆盖，也不会重新量化）；若中断的是 BPT 运行（已完成调优但尚未打包的块），流式路径会直接套用其调优结果（scale/zp）并打包，而不是重新搜索。

## 使用示例

### 设置环境变量

#### 通过 Shell 命令
```bash
# 将日志级别设置为 DEBUG
export AR_LOG_LEVEL=DEBUG

# 启用编译打包
export AR_ENABLE_COMPILE_PACKING=1

# 使用 ModelScope 下载模型
export AR_USE_MODELSCOPE=true

# 设置自定义工作目录
export AR_WORK_SPACE=/tmp/autoround_workspace
```

#### 通过 Python 代码
```python
from auto_round.envs import set_config

# 同时配置多个环境变量
set_config(
    AR_LOG_LEVEL="DEBUG",
    AR_USE_MODELSCOPE=True,
    AR_ENABLE_COMPILE_PACKING=True,
    AR_WORK_SPACE="/tmp/autoround_workspace",
)
```

### 查看环境变量

#### 通过 Python 代码
```python
from auto_round import envs

# 访问环境变量（懒加载求值）
log_level = envs.AR_LOG_LEVEL
use_modelscope = envs.AR_USE_MODELSCOPE
enable_packing = envs.AR_ENABLE_COMPILE_PACKING
workspace = envs.AR_WORK_SPACE

print(f"日志级别: {log_level}")
print(f"使用 ModelScope: {use_modelscope}")
print(f"启用编译打包: {enable_packing}")
print(f"工作目录: {workspace}")
```

#### 检查变量是否显式设置
```python
from auto_round.envs import is_set

# 检查环境变量是否被显式设置
if is_set("AR_LOG_LEVEL"):
    print("AR_LOG_LEVEL 已被显式设置")
else:
    print("AR_LOG_LEVEL 正在使用默认值")
```

## 配置最佳实践

1. **开发环境**：设置 `AR_LOG_LEVEL=TRACE` 或 `AR_LOG_LEVEL=DEBUG` 以获取详细日志
2. **生产环境**：使用 `AR_LOG_LEVEL=WARNING` 或 `AR_LOG_LEVEL=ERROR` 减少日志噪声
3. **中国用户**：建议设置 `AR_USE_MODELSCOPE=true` 以获得更好的模型下载速度
4. **性能优化**：如有足够算力，可启用 `AR_ENABLE_COMPILE_PACKING=1`
5. **自定义工作目录**：将 `AR_WORK_SPACE` 设置为磁盘空间充足的目录

## 注意事项

- 环境变量采用懒加载方式，仅在首次访问时读取
- `set_config()` 函数提供了便捷的程序化多变量配置方式
- `AR_USE_MODELSCOPE` 的布尔值会自动转换为适当的字符串表示
- 所有环境变量名称区分大小写
- 通过 `set_config()` 所做的修改将影响当前进程及其子进程
