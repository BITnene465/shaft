# 2026-09-15 权重状态清理与 0.8B checkpoint 选择

## 范围与结果

用户明确授权清理 tanjingyuan/shaft 下所有模型版本的训练状态，仅保留部署文件。
已检查 worker0、worker1、worker2，未发现 train.py、torchrun 或 DeepSpeed launcher 训练进程。

- 处理 93 个 checkpoint，覆盖 qwen35vl-sft、qwen38vl-sft、qwen35vl-offline-kd。
- 删除 1262 个训练状态文件，逻辑大小 764652128384 字节，约 764.65 GB / 712.14 GiB。
- 删除 optimizer、scheduler、RNG、training_args、trainer_state，以及 checkpoint 内训练效率和提交状态侧文件。
- 保留所有 safetensors、分片索引、config、generation_config、processor、tokenizer 与 chat template。
- 保留 checkpoint 外的训练运行配置、W&B 日志、评测结果，不修改权限或所有者。
- 删除前后均校验 safetensors 张量头、张量形状可读性与完整分片索引映射；保留文件的 inode、大小、mtime 未变。
- 这是结构校验，不等同于逐字节重新读取全部权重或重新执行模型推理。
- 清理后这些 checkpoint 仅用于部署或重新开始训练，不能精确恢复原优化器、调度器与随机数状态。

审计真源（相对于 shaft 根目录）：

- `temp/checkpoint_cleanup_20260915/plan.json`
- `temp/checkpoint_cleanup_20260915/deleted.json`
- `temp/checkpoint_cleanup_20260915/summary.json`
- `temp/checkpoint_cleanup_20260915.py`

## v5.10 0.8B 的最佳 checkpoint

不能将不同评测协议的最佳 checkpoint 混为一谈。

| 用途 | checkpoint | real v1 | real v2 | 总分 |
| --- | --- | --- | --- | --- |
| GT-BBox 子属性 weighted | 18000 | 81.2023 | 77.7704 | 78.4869 |
| detection F1 | 24000 | 80.7621 | 82.0209 | 不合并为 weighted 子属性总分 |

子属性结论限于已有 12k、18k、24k 对比；detection 结论来自已有 6k 至 24k 的重算结果。

评测来源：

- `outputs/v510-attributes-20260910-test/all_models_rescore/hf_total/totals.csv`
- `outputs/v510-hf-eval-audit-20260910/recomputed/comparison.csv`

## 部署目标

18k 和 24k 的部署文件通过使用者终端 `/tmp` 逐文件中继到 bananadev。
仅传 7 个部署文件/checkpoint，不传训练状态；每个文件校验 SHA256 后删除中继副本。

- `/mnt/model-dev/banana/banana-v5.10-qwen35-0.8B-checkpoint-18000`
- `/mnt/model-dev/banana/banana-v5.10-qwen35-0.8B-checkpoint-24000`

传输是否完成以本次 transfer_v510_08b_20260915 的 status.json 与校验清单为准。
