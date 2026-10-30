# 基于请求工作负载的动态 LoRA 分组

`--exec-type 3` 的实际服务路径已启用动态分组，包括 LLaMA/OPT 使用的
`LoRaColumnParallelLinear`、`QKV_LoRaColumnParallelLinear` 和
`LoRaRowParallelLinear`。同一批次的所有投影共享一次分组结果。
`layers/lora.py` 中的普通 Linear/QKV_Linear 也使用同一计算实现。

## 参数

```bash
--lora-grouping-reference-requests 100 \
--lora-grouping-reference-adapters 20 \
--lora-grouping-alpha 0.8
```

默认 `C_full = B × N = 100 × 20 = 2000`，阈值 `T = alpha × C_full = 1600`。
参数 B、N 必须为正整数，`0 < alpha <= 1`。
N 是计算预算的参考适配器数量，与 `--num-models` 的已配置适配器总数独立；
`--num-groups` 仍表示原有的引擎分组，动态 LoRA 组数由每个引擎分别维护。
`run_serve.sh` 已显式填写上述默认参数。

调度器原有默认值 `--max-num-seqs 16` 保持不变：每批最多 16 个请求、
最多 16 个有请求的适配器，所以 `C(1) <= 256`，在默认阈值下会保持一组。
若要按 100 请求规模观察多组调整，可另外设置 `--max-num-seqs 100`；
实际批量仍受请求到达量、token 上限和显存容量限制。也可以降低 B、N 或 alpha
在较小批量下验证组数变化。

## 批次与计算量

一个批次是一次非空的 `Worker.execute_model` 调用，包含该次调度的
prefill/decode 请求。每个请求计数一次，不乘以 prompt token 数或 beam 数，
也不计 Tensor Core 补齐位置。空批次、初始化的显存探测及非 LoRA 执行均不更新策略。

按当前请求统计各适配器的负载，按请求数从高到低，将整个适配器分配给当前
请求总数最少的一组；同负载时按组内适配器数和组序号打破平局。
无请求的适配器不参与分组。候选组数 g 的计算量为：

```text
C(g) = Σ（组内请求数 × 组内适配器数）
```

例如 30 个请求分别使用 30 个适配器，`C(1)=900`、`C(2)=450`、`C(3)=300`。
默认阈值 1600 下会选 1 组；若将阈值设为 320，则第一批选择 3 组。

## 调整规则

- 第一批从 1 组向上遍历，选择第一个满足 `C(g) <= T` 的组数。
- 第二批复用上批组数：若 `C(g) > T`，加一组；否则当 `g > 1` 且
  `C(g-1) <= T` 时减一组，其余情况保持。
- 第三批起：若 `C(g) > T`，加一组；否则当 `C(g) < T` 且上批吞吐量
  小于上上批吞吐量时减一组，其余情况保持。`C(g) == T` 时保持。

第二批起每次策略调整最多一组。组数始终限制在 `[1, 当前有请求的适配器数]`；
当适配器数减少时，先将上批组数截到这个范围，再应用本批规则。
即使每个适配器单独分组仍超预算，也会以最大可用组数执行，记录
`within_budget=false`。

第三批起吞吐下降触发减组时，按规则不额外检查 `C(g-1)`，因此选出的组数
可能暂时超预算；下一批再根据自己的工作负载调整。工作负载突然增加时，
一次加一组也可能尚未达到阈值，日志同样会如实记录。

## 吞吐量和日志

```text
throughput = 本批请求数 / elapsed_sec
```

用 `time.perf_counter()` 计时，从批次分组选择开始，覆盖缓存操作、
适配器调整、输入准备和模型执行，到 CUDA 同步确认整批 GPU 工作完成为止。
批次开始前同步以排除上一批尚未完成的 GPU 工作；排队、HTTP 和网络时间不计入。
这是每次迭代处理请求的速率，单位 requests/s，生成多轮的同一请求会参与多批统计。

模型并行的多个 rank 通过 MAX 归约使用相同的最长执行耗时，确保下一批
选择一致；每个模型并行组仅 rank 0 输出一条 `lora_grouping {JSON}` 日志。
每条日志包含：

- `batch_index`、`num_requests`、`num_adapters`、`previous_groups`、`num_groups`；
- `c_full`、`threshold`、`reused_compute`（调整前）、`compute`（调整后）、`within_budget`；
- `elapsed_sec`、`throughput`、`previous_throughputs`、`reason`；
- `partitions` 和 `active_model_ids`：分组中的索引对应本批 `active_model_ids` 的位置。

完整记录写入服务日志，内存中仅保留最近两批吞吐量和最新记录。
`FUSELORA_PROFILE=1` 的细粒度同步分析会改变实际耗时；评估正常运行的
吞吐量时可设置 `FUSELORA_PROFILE=0`，动态分组和每批吞吐日志仍然启用。

## 验证

在项目根目录运行，不依赖 CUDA 的策略与集成测试：

```bash
python -m unittest discover -s adjustfuselora/tests -p test_dynamic_lora_grouping.py
python -m unittest discover -s adjustfuselora/tests -p test_latency_breakdown.py
python -m unittest discover -s tests -p test_serving_breakdown_variants.py
```

第一条还包含 CUDA 多流数值校验，在没有 GPU 的环境自动跳过。
