# Shaft TODO

本文档只记录当前明确延期或需要额外硬件/环境验收的事项。已经实现的能力不再以 TODO 形式重复维护。

## 当前主线

- 稳定 Qwen3VL / Qwen3.5 / Qwen3.6 的 HF-first SFT、RLHF、checkpoint、resume、infer 与 export 主链。
- 保持配置、数据、模型、模板、算法、pipeline、training、infer、codec、metrics、export 的单一真源。

## 发布前仍需按需执行的验收

- 在最终冻结 SHA 上执行 CUDA Qwen release gates；CPU tiny-model 测试不能替代真实 CUDA kernel 验收。
- 如果要对外声明真实多机能力，执行双主机 NCCL/NIC/共享存储 canary；同机 Gloo 多 launcher 只验证
  rank、topology、checkpoint 和故障收敛契约。

## 明确暂缓

- 多图 sequence packing：单图 varlen/packing 已有执行骨架，但多图 media-segment 对齐、隔离与模型族
  correctness 需要独立设计和 GPU 验收。
- FSDP/DeepSpeed 下尚未开放的 planned sampler 组合：继续 fail closed，不以兼容开关伪装支持。
- 脱离 Hugging Face Trainer 的大规模训练内核重写：当前继续 HF-first，除非真实瓶颈和收益足以支撑独立立项。
- 重型离线 benchmark / Eval Bench 产品：已经从主线切除；未来如有真实需求，作为独立项目重新立项，
  不复制 Shaft 的 codec、metric 或 infer 真源。
- Web UI：当前不属于主线，不重新引入第二套训练或配置语义。

## 维护规则

- 新的延期项必须说明边界、原因和验收条件，不能只写功能名。
- 已完成项及时删除；事故与经验写入 `docs/development_log.md`，不要把 TODO 当开发日志。
