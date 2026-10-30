# Serving 请求开销统计

本方法已接入与 adjustfuselora 一致的逐请求指标和 CSV schema。
dlora 原有单 adapter merge/unmerge 行为保留，计入 adapter_adjustment_sec。

在仓库根目录启动：

```bash
FUSELORA_PROFILE=1 bash dlora/run_serve.sh
```

服务就绪后运行 `bash dlora/run_benchmark.sh`，结果写入 dlora/output。
完整字段定义、方法间阶段对应关系和测试方式见
[统一开销统计说明](../../docs/serving_latency_breakdown.md)。
