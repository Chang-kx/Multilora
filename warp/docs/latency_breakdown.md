# 请求延迟分解

服务端启动前设置环境变量（需要让 Ray worker 也继承它）：

```bash
cd adjustfuselora
FUSELORA_PROFILE=1 bash run_serve.sh
```

benchmark 原有参数保持不变，增加 `--request-output output/requests.csv`
（与 `--breakdown-output` 是同一个参数，旧命令仍兼容）。不指定时默认写入当前目录
`request_metrics.csv`，每次运行覆盖该文件。
也可以直接运行 `bash run_benchmark.sh`，默认生成每个 rate 的分项 CSV。
逐请求 CSV 的父目录会自动创建。已有 Ray 集群的 worker 可能没有继承新环境变量，需重启
对应集群或配置 Ray runtime_env。客户端收到分项数据后会自动打印汇总，不依赖读取日志。
未开启 profiling 时仍记录每个请求，缺失的阶段计时留空。旧服务端或 TGI
没有提供的指标也留空，不将缺失计时当作零。

## 指标口径

原有 `Latency per token` 保留，其含义是整轮 benchmark wall time / 总输出 token，
是吞吐量的倒数，包含 workload 的发请求间隔，不能拆成各请求延迟的平均值。
API 在流式和非流式响应中均返回实际 token 数、执行时间，以及可用的阶段计时。
benchmark 对 vLLM 使用流式请求，按 NUL 分隔的 JSON 帧解析，不依赖 HTTP chunk 边界。

| CSV 字段 | 口径 |
| --- | --- |
| request_id / model_id | benchmark 请求编号 / adapter 模型编号 |
| processing_time_sec | 各次被调度执行的区间之和，从 scheduler 选中请求到收到该次 worker 输出；包含输入准备、worker/RPC 等耗时，排除首次调度前以及两次执行之间的排队、抢占等待；不含输出解码和 HTTP 传输，不是纯 GPU kernel 时间 |
| ttft_sec | 服务端请求实际开始执行至 scheduler 收到首个新 token 的累计处理时间，包含 prefill、输入准备和 worker/RPC，排除首次调度前及执行区间之间的排队等待，不含 HTTP 传输和输出文本解码。首 token 解码为空字符串也计入；best-of/beam 时指首个候选 token，不保证属于最终选中序列。首次记录后固定，后续 decode 不累加；旧服务端/TGI 未返回此字段时留空 |
| prompt_tokens | 服务端实际 prompt token 数 |
| output_tokens | 返回序列的新生成 token 数，按 token IDs 计数，包含已生成的特殊 token；n=1 时为该序列的生成长度，不含被丢弃候选的 token |
| total_tokens | prompt_tokens + output_tokens |
| tokens_source | server 表示服务端返回实际输出 token 数；requested 表示旧服务端/TGI 未提供，回退到计划长度，prompt 长度也在服务端未提供时使用客户端计划长度 |
| latency_sec | 客户端端到端总耗时，包含排队和网络，与 processing_time_sec 分开保存 |
| profile_available | 是否收到同步 worker 阶段数据 |

processing_time_sec 使用同一 engine 内的单调时钟计算每次执行区间，迁移时保留累计值，
新 engine 重新设置执行起点，不比较不同进程的时钟绝对值。未开启 profiling 也可记录。
TTFT 使用服务端累计执行时间，与 processing_time_sec 使用相同计时口径，
流式和非流式 API 均返回；不使用客户端收到首帧的时间，也不比较客户端与服务端时钟。

## 本次实验的 adapter 秩记录

重启更新后的服务端再运行 benchmark，逐请求 CSV 会额外记录下面的字段。
这些数值取自服务端原生 LoRA 执行路径的权重分配配置，不使用 benchmark 客户端默认值：

| 字段 | 含义 | 默认 max_r=16 时 |
| --- | --- | --- |
| lora_rank | 配置的基础低秩维度 max_r | 16 |
| qkv_rank | 融合 QKV 矩阵实际分配的中间维度，3 × max_r | 48 |
| o_rank | attention 输出投影中间维度 | 16 |
| gate_up_rank | Llama 融合 gate/up 投影中间维度 | 16 |
| down_rank | Llama down 投影中间维度 | 16 |

字段表示矩阵分解维度，不是对权重执行 matrix_rank 得到的数值秩；
不依赖 `FUSELORA_PROFILE`。当前实现所有原生 LoRA adapter 使用同一个 max_r。
OPT 路径中后三个投影字段分别对应 out_proj、fc1、fc2。
PEFT、非 LoRA 路径和旧服务端未提供此元数据时，字段留空。

benchmark 同时生成 `<逐请求CSV文件名去掉扩展名>_adapter_ranks.csv`，例如：

```text
output/fig9_maf1_7b_exec_rate20_breakdown_adapter_ranks.csv
```

可通过 `--adapter-ranks-output output/adapter_ranks.csv` 自定义路径。
每行包含 model_id、上述秩字段、rank_source 和 request_count，按 model_id 排序。
rank_source=server 表示来自服务端；unavailable 表示没有可用数据。
同一 adapter 的相同秩配置合并计数，不同秩配置分别保留。
`--num-models` 范围内没有已完成请求的 adapter 也会列出，秩留空、request_count=0。
原有实验 CSV 不会自动补写，缺少服务端元数据时无法追溯确认该次实验的真实配置。

新增分项的汇总口径是 **mean(每个请求的阶段累计秒数 / 该请求输出 token 数)**，
以 ms/output-token 显示，同时给出 s/request 均值。它可以与已有的
`Average latency per output token` 对齐，不能与整轮 `Latency per token` 相加比较。
CSV 每行一个已完成请求，按 request_id 排序，含原始秒数和每输出 token 的毫秒数。
控制台同时打印处理时间和 TTFT 的均值、P90；阶段汇总仅统计有 profiling 数据的请求。

| CSV 阶段 | 实际测量范围 |
| --- | --- |
| adapter_grouping | 每层适配器 token 计数、CPU 贪心负载均衡分区，以及分区映射/mask 构建 |
| request_collection | worker 输入打包；LoRA 的 token/权重索引收集、矩阵布局转换、中间结果 gather、输出缓冲和 stream 创建 |
| model_compute | 完整 model forward 的剩余耗时，包含基础模型、LoRA 矩阵乘和采样；扣除上述嵌套阶段，不是纯 CUDA kernel 时间 |
| output_reorder | 多分区 LoRA 结果通过 `y[token_mask]` 写回原 token 位置；单分区无需此操作，记为 0 |
| adapter_adjustment | worker 内适配器换入换出/激活调整 |
| worker_other | execute_model 中未被以上阶段覆盖的部分，包括 KV cache 操作和 Python 控制逻辑 |
| unattributed | 客户端端到端请求延迟减去以上阶段累计值；包含网络、排队、调度、RPC、解码、输出 top-n 排序、迁移/调整间隔、重试等，**不是纯排队时间** |

同一请求多轮 prefill/decode 的阶段耗时随 SequenceGroup 累加，并随迁移保留。
一个 batch 的完整阶段耗时计入每个参与请求一次；beam 的多条 sequence 不重复累计。
这表示请求经历的 batch 时间，不是按 batch 大小平摊的设备成本。
因此跨请求累加会重复计算共享 batch 时间，不能拿它当作 GPU 总工作量。
服务端 TP 选取已有 worker 输出路径中的一个 rank 的计时，不能代表各 rank 最大值；不跨 rank 相加。
当前只支持 pipeline parallel size=1；启用 profiling 且 PP>1 会显式报错。
非 LoRA 执行模式会将整批中多个模型的顺序执行时间计入该批请求。

各阶段使用 perf_counter；在开始和结束时同步当前 CUDA 设备，支持多 stream 上的工作完成计时。
计时为互斥壁钟耗时：嵌套阶段从父阶段中扣除，避免把 gather 同时算进 model_compute。
新增客户端请求延迟也使用 perf_counter，wall-clock 时间戳仍用于原有 midpoint 功能。
残余项保持有符号，负值会警告，避免掩盖计时不一致。

**此模式会破坏多 stream 重叠，并加入大量同步开销，适合定位阶段成本，不代表正常并发运行时的性能占比。**
同步本身的等待和调用成本也会影响结果；阶段边界管理等剩余成本进入父阶段或残余项。
正式吞吐/延迟实验应关闭 `FUSELORA_PROFILE`，另跑一轮分析实验。
未开启时不执行 CUDA 同步、不附加分项数据；计时上下文仍有少量 Python 调用开销。
分组 stream 开始前增加了对调用 stream 的依赖，确保读取已生成的输入和 mask；
写回后仍由调用 stream 等待各分组 stream。
更精确地研究正常重叠下的 GPU kernel 开销应使用 CUDA timeline；重叠阶段的设备时间通常不可直接相加。
