# Shaft TODO

本文档只记录当前明确延期或需要额外硬件/环境验收的事项。已经实现的能力不再以 TODO 形式重复维护。

## 当前主线

- 稳定 Qwen3VL / Qwen3.5 / Qwen3.6 dense 与 MoE 的 HF-first SFT、checkpoint、resume、infer 与 export 主链。
- 保持配置、数据、模型、模板、算法、pipeline、training、infer、codec、metrics、export 的单一真源。

## 发布前仍需按需执行的验收

- 在最终冻结 SHA 上执行 CUDA Qwen release gates；CPU tiny-model 测试不能替代真实 CUDA kernel 验收。
- 如果要对外声明真实多机能力，执行双主机 NCCL/NIC/共享存储 canary，并覆盖本地 HF artifact 的
  node-leader baseline/post-load identity consensus；同机 Gloo 多 launcher 只验证 rank、topology、
  checkpoint 和故障收敛契约，不能替代 NCCL default group + Gloo identity group 或真实共享存储证据。

## 明确暂缓

- Qwen3VL 30B-A3B MoE 的两卡 BF16 FSDP LoRA 短门禁已经通过，但目标数据长程收敛、吞吐调优和
  full-parameter SFT 容量仍未验收。两卡全参数 AdamW 不在当前资源边界内；DeepSpeed ZeRO-3 还缺模型专属
  routed-expert leaf-module contract，因此 Qwen3VL MoE 当前只声明 padded FSDP LoRA 路线。
- Qwen3.5/3.6 真实 MoE 发布权重的生产验收：tiny upstream architecture 已覆盖 router objective、DDP、
  FSDP LoRA、ZeRO-3 full、exact resume 与 HF/PEFT export，但还需要真实 35B 权重的目标硬件显存/吞吐、
  长程数值稳定性和目标数据收敛 canary。tiny gate 不能替代该容量结论。
- `DeepSpeed ZeRO-3 + PEFT target_parameters`：PEFT 0.18.1 不能对构造期 partition placeholder 注入
  direct-parameter wrapper，当前明确拒绝。只有上游提供稳定支持并完成 fresh/resume/export gate 后才重新开放；
  不在 Shaft 内复制或猴子补丁 PEFT 注入器。
- 尚未验收的真实规模/精度组合：Qwen3.5/3.6 dense 27B、Qwen3.5/3.6 MoE 35B、Qwen3VL-32B、varlen
  FP16 与 FSDP FP16。当前模型注册或 runtime allowlist 不能替代对应 checkpoint/hardware gate。
- adapter-on-adapter 再训练的 merge provenance：当前可从 PEFT adapter 精确初始化并继续训练，但若产物的
  declared base 已变为上一个 adapter，安全 merge 不会自动递归追溯 adapter chain。需要定义扁平化 base
  provenance 与双 adapter state 语义后再开放，当前不要依赖 `allow-unverified-base-model` 绕过。
- PEFT 显式 `target_parameters` 与 freeze override 的优先级尚未统一为稳定公共语义；当前仍按 freeze plan
  过滤 direct-parameter target。模型族默认 `[auto]` 路径已有测试，用户显式覆盖 freeze 的语义在定稿前
  保持保守，不声明与显式 `target_modules` 完全等价。
- 多图 sequence packing：单图 varlen/packing 已有执行骨架，但多图 media-segment 对齐、隔离与模型族
  correctness 需要独立设计和 GPU 验收；普通 padded SFT/DPO 与有序多图推理已经支持，不在此 TODO 内。
- FSDP/DeepSpeed 下尚未开放的 planned sampler 组合：继续 fail closed，不以兼容开关伪装支持。
- backend-native typed checkpoint 的后续泛化：planned SFT 已用
  `shaft_backend_checkpoint_commit.json` 绑定 planning generation、Trainer/scheduler/RNG 小状态与完整 native
  shard 路径/非零尺寸集合，并支持 run-root fallback；普通 unplanned FSDP/DeepSpeed 仍沿用后端协议。若要把
  同等 transaction 扩到全部算法/ordinary fixed，或对数百 GiB model/optimizer shard 增加内容 digest/DCP
  identity，需要先设计不会给每次保存增加第二次全量 I/O 的 backend-native 方案并补崩溃点验收。
- 脱离 Hugging Face Trainer 的大规模训练内核重写：当前继续 HF-first，除非真实瓶颈和收益足以支撑独立立项。
- 重型离线 benchmark / Eval Bench 产品：已经从主线切除；未来如有真实需求，作为独立项目重新立项，
  不复制 Shaft 的 codec、metric 或 infer 真源。
- Web UI：当前不属于主线，不重新引入第二套训练或配置语义。

## 维护规则

- 新的延期项必须说明边界、原因和验收条件，不能只写功能名。
- 已完成项及时删除；事故与经验写入 `docs/development_log.md`，不要把 TODO 当开发日志。
