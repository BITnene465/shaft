# Shaft 总 TODO

本文是仓库唯一的当前待办真源，只记录尚未实现、尚未验收或需要补齐 fail-closed 的事项。当前已经支持的
行为与限制以 `architecture.md`、`module_reference.md`、`config_reference.md` 和 `testing.md` 为准；历史事故与
阶段性实验只写入 `development_log.md`。

维护规则：

- 完成项立即从本文删除，并按需要写入正式参考文档或开发日志。
- 不为单个算法、模型、后端或 feature 新建平行 TODO。
- “代码可装配”“CPU contract 通过”“真实发布权重 gate 通过”和“长程生产验收”是四种不同证据，不得互相
  外推。
- 未验收组合必须先 fail closed；不能仅通过修改文档把它表述为支持。

最近一次全局审计：2026-08-14。

## P0：修正错误能力边界

### 1. DPO/GRPO 的 FSDP + PEFT exact resume

状态：未完成，当前不属于支持范围。

- 2026-08-14 审计确认：现有通用配置预检仍会接受 DPO/GRPO 的
  `LoRA + FSDP + full_state_dict + periodic checkpoint`，但恢复语义只在 SFT 验收。
- 在共享 restore contract 完成前，应在数据和模型加载前显式 fail closed，并补配置与 pipeline 负例。
- 实现后分别验证 DPO、GRPO 的两 rank fresh/resume、标准 adapter、optimizer、scheduler、RNG、Trainer state
  与损坏 artifact 负例。
- 不直接复用 SFT 的成功结论，也不在各 RL trainer 复制一套 adapter preload。

### 2. PPO 训练正确性与对外定位

状态：仅 debug smoke，不是生产能力。

- 公开文档不再把 PPO 命令与可运行的 SFT/DPO/GRPO 命令并列；默认示例保持安全关闭 debug 开关。
- 若继续生产化，value/critic role 必须在 optimizer plan 前完成装配，并证明全部预期参数获得梯度和更新。
- Shaft sample plan 必须成为唯一数据顺序真源，禁止上游 DataLoader 再次 shuffle。
- 接入具有不可变身份和加载校验的真实 reward model；`allow_untrained_reward_model` 只保留为显式 debug
  开关。
- 若支持多模态 PPO，补齐 image 到 policy/value/reward 的有序媒体合同；
  `allow_text_only_multimodal_ppo` 不能被视为多模态支持。
- full-parameter PPO、真实 RM、KL/reward/entropy/ratio telemetry、checkpoint/resume、最终 artifact reload 与
  真实多卡都需要独立 release gate。

### 3. Qwen3.5/3.6 MTP artifact 一致性

状态：未实现，当前不属于 Shaft 支持范围。

- 模型 config 声明 MTP 时，checkpoint tensor 必须区分 `absent / inherited / trained`；缺失或残缺权重不能
  静默导出为 enabled artifact。
- 若实现可训练 MTP，objective 必须由模型 policy 持有，并验证 next-N shift、监督 mask、保存恢复与
  speculative decode；模型专属逻辑不能进入通用 trainer/collator。

## P1：补齐训练域生产门禁

### 4. DPO release gate

- 完成真实 Qwen 单卡与 DDP forward/backward、fresh/resume、HF/PEFT export/reload。
- 对 `precompute_ref_log_probs`、`label_smoothing` 等参数验证真实训练语义；未消费参数必须提前拒绝。
- FSDP/DeepSpeed、planned batching、packing/varlen 只有专项门禁后才开放。

### 5. GRPO release gate

- 让 `data.max_length`、rollout cadence、grouped sampler 与 checkpoint safe boundary 使用同一合同。
- 完成真实 Qwen、DDP、final export/reload 和 online eval 验收。
- GRPO vLLM 的 server RNG/恢复状态未进入 checkpoint 前，继续禁止 periodic save/resume。
- 删除或正式接入未被主链消费的输入实现，避免第二套 collator contract。
- 补可靠的 loss eval / `eval_final_loss` 聚合；完成前只支持 online `eval_final_score`。

### 6. OPD 发布与容量专项

以下不是当前 OPD 主链 blocker，但不能从现有单图/tiny 证据外推：

- 发布权重的有序多图 vLLM rollout。
- 独立 Qwen HTTP teacher GPU 部署与吞吐/限流测试。
- 发布 Qwen 的 FSDP/DeepSpeed OPD 容量门禁。
- 大词表 full/chunk/top-k-tail 的显存和吞吐 A/B。
- 跨 step rollout buffer、PG-OPD/OPSD；引入 buffer 前先设计持久化 cursor/state。

### 7. 生产证据与 CI

- required CI 当前只覆盖 CPU framework 与 tiny/fake smoke；distributed、integration、GPU 和真实模型门禁不应
  被 required 绿灯替代。
- 为发布版本维护可审计的真实模型/GPU gate 结果，不把历史一次性通过自动解释为当前依赖版本仍通过。
- 在资源允许时增加非阻塞的定期 GPU/真实模型验证；是否升级为 required 应由稳定性和资源成本共同决定。

## P2：扩展 batching、并行与规模矩阵

### 8. Sharded planned batching

当前 FSDP/DeepSpeed planned SFT 只开放 bounded-cost fixed padded。待补：

- token-budget cardinality。
- length grouping 与 greedy-varlen packing。
- world-size elastic resume。
- DPO/GRPO/PPO planned batching。

每个组合都必须验证 draw 守恒、无二次分片、GA boundary、backend-native resume 和 GPU canary，不能只解除
normalize 限制。

### 9. 多图 sequence packing

- padded SFT/DPO 与有序多图推理已支持；varlen/greedy packing 仍只支持单图。
- 需要统一 media-segment 顺序、attention isolation、M-RoPE/position、pixel/grid slice，并补模型族 GPU
  correctness；通用 varlen builder 不硬编码 Qwen 字段。

### 10. 分布式、编译与发布规模

- 真实双主机 NCCL/NIC/共享存储 canary。
- tensor/context/sequence parallel。
- `torch.compile` 支持矩阵。
- 后端 checkpoint 的可扩展内容身份，避免为数百 GiB shard 增加第二次全量读取。
- Qwen3.5/3.6 MoE 35B 真实权重。
- Qwen3VL dense 32B 与 30B-A3B 的长程收敛、吞吐和 full-parameter 容量。
- varlen FP16、FSDP FP16、adapter-on-adapter provenance。
- PEFT `target_parameters` 与 freeze override 的长期稳定性。

## P3：按需求扩展框架覆盖面

这些项目不是当前 Qwen SFT 主线 blocker，只在有明确产品或研究需求时启动。

### 11. 推理与评估

- 当前推理 API 是单样本、同步 stage 编排且要求至少一张图片；若承担在线服务，再设计 batch、streaming、
  async queue、text-only request 与服务生命周期合同。
- 本地 HF `generate()` 当前不能安全抢占；只有找到可证明的取消边界后才开放 deadline/cancellation。
- 在线 eval 当前只支持单阶段；多阶段在线 eval 或独立离线 bench 需要先明确所有权，避免与 infer/codec/metric
  建立平行实现。

### 12. 模型、数据、插件与发布生态

- 新增非 Qwen 模型族的真实训练、推理和导出 gate，验证现有扩展接口确实不依赖 Qwen 私有语义。
- 按需求增加 HF Dataset、Parquet、WebDataset 或 streaming source；继续保持 sample schedule 与数据 identity
  的单一真源。
- 若允许影响训练轨迹的 stateful plugin，先设计版本化 `state_dict/load_state_dict` 与 exact-resume 合同。
- Hub 或第三方平台发布保持在独立发布层，不进入 HF/PEFT export 内核。

## 推荐顺序

1. 先让 DPO/GRPO FSDP+PEFT 未验证组合明确 fail closed。
2. 保持 PPO debug-only，除非明确决定投入真实 RM、optimizer/data-order correctness 和 release gate。
3. 完成 DPO release gate，再收口 GRPO。
4. 按真实训练需求选择 OPD 发布、多图 packing、sharded batching 或模型规模门禁。
5. 最后按产品需求决定推理服务化、非 Qwen 模型与新增数据源，避免同时扩张全部矩阵。
