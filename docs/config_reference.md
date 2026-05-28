# Shaft 配置参考

本文档描述 `RuntimeConfig` 的主要配置块和推荐使用方式。配置以 YAML 为主，CLI 只允许无歧义 override。

当前 `config` 已按职责拆分为多文件实现：

- `src/shaft/config/base.py`
- `src/shaft/config/model.py`
- `src/shaft/config/data.py`
- `src/shaft/config/training.py`
- `src/shaft/config/algorithm.py`
- `src/shaft/config/runtime.py`

`src/shaft/config/schema.py` 只作为配置类型的聚合出口，不再承载全部 dataclass 实现。

## 1. 顶层结构

训练主配置由 `RuntimeConfig` 组成：

- `experiment`
- `model`
- `data`
- `algorithm`
- `train`
- `eval`
- `rlhf`
- `plugins`
- `logging`
- `progress`

## 2. `experiment`

用途：实验元信息和输出目录。

关键字段：

- `name`
- `seed`
- `output_dir`
- `run_id`

约束：

- `run_id` 用于区分同一实验模板下的不同运行实例。
- `output_dir` 应视为当前运行的唯一产物目录。

## 3. `model`

用途：模型族、模型路径和微调方式。

关键字段：

- `model_type`
- `model_name_or_path`
- `template`
- `trust_remote_code`
- `attn_implementation`
- `torch_dtype`
- `finetune`

### `model.finetune`

关键字段：

- `mode`: `full | lora | dora | qlora`
- `freeze`
- `target_modules`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `lora_bias`
- `use_rslora`
- `qlora_load_in_4bit`
- `qlora_use_double_quant`
- `qlora_quant_type`
- `qlora_compute_dtype`

约束：

- `target_modules=["auto"]` 表示交给模型族 `peft policy` 自动解析。
- `freeze.groups` 当前只允许：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `freeze.regex` 与 `freeze.trainable_regex` 必须是合法正则。
- `init_from_checkpoint` 与 `resume_from_checkpoint` 的兼容矩阵由 `training/checkpointing.py` 统一校验。

### `model.finetune.freeze`

关键字段：

- `groups`
- `prefixes`
- `regex`
- `trainable_prefixes`
- `trainable_regex`

说明：

- `groups` 使用模型族声明的结构分组：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `groups` 的匹配采用“最具体前缀优先”。
  - 例如 `language_model=("model",)` 且 `vision_tower=("model.visual",)` 时，
    `model.visual.*` 会归到 `vision_tower`，不会被 `language_model` 误伤。
- `prefixes` / `regex` 用于冻结。
- `trainable_prefixes` / `trainable_regex` 用于显式解冻，优先级高于冻结规则。

执行语义：

- 训练时会先把上述配置解析为一份 `resolved finetune plan`，后续训练执行与 adapter 导入校验都消费这份计划。
- 训练启动后，CLI 会打印一份运行时 `resolved freeze summary`，并在输出目录写入：
  - `shaft_finetune_summary.json`
  - `shaft_optimizer_summary.json`
- `full`
  - 先默认全部可训练
  - 再应用冻结规则
  - 最后应用 `trainable override`
- `lora / dora / qlora`
  - 冻结规则主要作用于 `target_modules=["auto"] / ["all-linear"]` 的自动展开结果
  - 如果显式指定 `target_modules`，则保持显式配置权威
  - `trainable override` 会额外导出为 `modules_to_save`
    - 这里的前缀匹配以模块名为准，例如 `lm_head`、`model.visual.merger`
  - 这类 adapter checkpoint 仍然是 PEFT 目录；如果后续部署后端只接受 full HF model，需要先 merge

## 4. `data`

用途：数据 catalog、多数据源、mixing 与批处理行为。

关键字段：

- `catalog_path`
- `catalog_names`
- `datasets`
- `prompt_sampling`
- `mix_strategy`
- `mix_refresh`
- `num_workers`
- `pin_memory`
- `persistent_workers`
- `min_pixels`
- `max_pixels`
- `max_length`
- `add_eos_token`
- `shuffle`

### `data.datasets`

每个条目是一个 `DatasetSourceConfig`：

- `dataset_name`
- `source_type`
- `train_path`
- `val_path`
- `train_paths`
- `val_paths`
- `weight`
- `enabled`
- `use_for_eval`
- `offline_transforms`
- `online_transforms`
- `help`
- `tags`

约束：

- `catalog_path/catalog_names` 用于复用“命名数据集”。
- `catalog_path` 只表示“去哪个 catalog 文件里找”，**不会自动启用里面全部数据集**。
- 只有写进 `catalog_names` 的数据集才会被展开到最终的 `data.datasets`。
- `datasets` 用于当前 YAML 内联声明数据源。
- 实际进入 `ShaftDataCenter` 前，catalog 会先展开成标准 `datasets` 列表。
- `DatasetSourceConfig` 只描述配置输入；进入数据主链前，会先被解析成 `ShaftDatasetMeta`。
- `use_for_eval=false` 表示该数据集只参与训练 mixing，不参与验证集构建，也不要求提供 `val_path/val_paths`。
- 当 `eval.enabled=true` 时，至少要有一个 `enabled=true` 且 `use_for_eval=true` 的数据集。
- `mix_refresh` 当前支持：
  - `static`
  - `epoch_refresh`
- `static` 会在训练启动前构建一次 train sampler，整个 run 复用同一份混合索引。
- `epoch_refresh` 会在每个 epoch 通过 train sampler 重建训练集 mixing 索引；验证集仍保持静态 concat。
- `max_length` 是训练 batch 组装阶段的 token 长度上限，语义接近 Swift / LLaMA-Factory 的
  `max_length` / `cutoff_len`。当 `prefix_tokens + target_tokens + eos > max_length` 时，SFT 会按剩余
  token budget 截断 assistant target；被截断的 target 不会补 EOS，避免把半截 JSON 教成合法结束。
  训练前仍应按真实 processor 做长度审计，超限样本优先过滤、记录或拆 crop。

### `data.prompt_sampling`

用途：训练运行时对同一数据样本随机轮换等价 prompt，避免把固定 prompt 学成任务 one-hot 编码。

关键字段：

- `enabled`: 是否启用，默认 `false`。
- `train_only`: 是否只对 train dataset 生效，默认 `true`。推荐保持 `true`，让 val/eval 使用固定 canonical
  prompt。
- `seed`: prompt 采样种子；未设置时使用 `experiment.seed`。
- `pools`: 按 `dataset_name` 配置单个版本化 prompt pool YAML 文件。

示例：

```yaml
data:
  prompt_sampling:
    enabled: true
    train_only: true
    seed: 42
    pools:
      grounding_arrow: ../prompts/pools/grounding_arrow.v2.4.yaml
      point_arrow: ../prompts/pools/point_arrow.v2.4.yaml
```

约束：

- prompt pool 路径相对训练 YAML 所在目录解析；一个数据集只能指向一个 pool 文件。
- pool 只按 `dataset_name` 匹配，不能跨任务复用不同 label scope 的 prompt。
- 每个 pool YAML 必须包含 `metadata.id`、版本信息和非空 `prompts` 列表；每个 prompt variant 必须包含
  `id` 和 `user_prompt`。
- 启用后，所有 `enabled=true` 的训练数据集都必须有对应 pool；SFT 行里的 `user_prompt` 不再作为 prompt
  真源。
- 采样键包含 `seed + epoch + dataset_name + sample_id`，同一 epoch 内可复现，不同 epoch 可能切换。
- 当前实现只替换 `system_prompt/user_prompt` 形式的样本；如果样本已经提供多轮 `messages`，会跳过采样并在
  `extra.prompt_sampling_skipped` 里记录原因。

补充说明：

- 仓库内置的 `configs/data/example.yaml` 当前只作为示例 catalog，不应默认视为可直接训练的数据清单。
- 如果你的实验数据较少或不需要复用 catalog，直接使用 `data.datasets` 往往更直观。

## 5. `algorithm`

用途：选择训练算法与算法级参数。

关键字段：

- `name`: `sft | dpo | ppo | grpo`
- `params`

说明：

- `params` 只保留算法的轻量补充参数。
- DPO/PPO/GRPO 的结构化核心参数在 `rlhf` 块中。

## 6. `train`

用途：训练行为、保存策略和 resume/init 规则。

关键字段：

- `epochs`
- `max_steps`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `gradient_checkpointing`
- `learning_rate`
- `param_group_lrs`
- `no_decay_name_patterns`
- `optimizer_name`
- `scheduler_name`
- `loss_name`
- `loss_scale`
- `weight_decay`
- `warmup_ratio`
- `max_grad_norm`
- `bf16`
- `use_cpu`
- `logging_steps`
- `save_strategy`
- `save_steps`
- `save_total_limit`
- `ddp_find_unused_parameters`
- `report_to`
- `load_best_model_at_end`
- `save_final_model`
- `save_final_state`
- `init_from_checkpoint`
- `resume_from_checkpoint`
- `distributed`

说明：

- `train` 是 SFT 与 RLHF 共用的基础训练块。
- `optimizer_name/scheduler_name/loss_name` 走注册表。
- `distributed.strategy` 描述训练拓扑入口，当前支持：
  - `ddp`
  - `fsdp`
  - `deepspeed`
  默认是 `ddp`，表示继续使用 Hugging Face / torchrun 的常规 DDP 语义。
- `distributed.fsdp` 只维护 FSDP 配置语义，不直接启动进程；训练入口仍由 CLI / torchrun 负责。关键字段：
  - `sharding_strategy`: `full_shard | shard_grad_op | no_shard | hybrid_shard`
  - `auto_wrap_policy`: `none | transformer | size`
  - `transformer_layer_cls_to_wrap`: transformer auto-wrap 的层类名列表，默认 `["auto"]`
  - `min_num_params`: size auto-wrap 下限，必须大于等于 0
  - `activation_checkpointing`
  - `cpu_offload`
  - `use_orig_params`
  - `backward_prefetch`: `backward_pre | backward_post`，也可为空
  - `forward_prefetch`
  - `limit_all_gathers`
  - `state_dict_type`: `full_state_dict | local_state_dict | sharded_state_dict`
  - `sync_module_states`
- `distributed.fsdp.transformer_layer_cls_to_wrap=["auto"]` 会按模型族默认解析。Qwen3VL 当前解析为：
  - `Qwen3VLTextDecoderLayer`
  - `Qwen3VLVisionBlock`
- `distributed.deepspeed` 支持 `config_path` 或 inline `config`。当 `strategy=deepspeed` 时，两者至少要提供一个；
  Shaft 只负责保存和校验配置真源，不在 `config` 层展开 DeepSpeed 运行时细节。
- `configs/deepspeed/zero3_bf16.json` 是 ZeRO-3 bf16 示例配置，包含保存时 gather 16-bit 权重的设置，
  用于保持 `trainer.save_model()` 的 HF export 语义。
- 分片策略属于训练运行时；数据、template、task prompt 和 collator 不应该感知 FSDP/DeepSpeed。
- `gradient_checkpointing`
  - 打开后会把 `TrainingArguments.gradient_checkpointing` 设为 `true`
  - 并在模型装配阶段显式把训练态 `use_cache` 关闭
  - `qlora` 路径会同步传给 `prepare_model_for_kbit_training(..., use_gradient_checkpointing=...)`
- `param_group_lrs` 用于显式配置分组学习率。当前支持的键：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
  - `lora_params`
  - `modules_to_save`
- 没有写进 `param_group_lrs` 的组，回退到全局 `train.learning_rate`。
- `no_decay_name_patterns` 用于把额外参数名并入 `no_decay` 规则。
  - 匹配语义是“参数规范名后缀匹配”，例如：
    - `embed_tokens.weight`
    - `lm_head.weight`
  - 这条规则会叠加在默认 `no_decay` 规则之上；默认规则仍然包括：
    - `*.bias`
    - `ndim <= 1` 的参数
- 结构组与训练语义组是两层概念：
  - 结构组：
    - `language_model`
    - `vision_tower`
    - `aligner`
    - `generator`
  - 训练语义组：
    - `lora_params`
    - `modules_to_save`
- `full`
  - 主要按结构组分学习率。
- `lora / dora / qlora`
  - `lora_params` 和 `modules_to_save` 会优先于结构组命中。
  - 其余仍可训练的原始参数，再按结构组回退。
- `loss_scale` 控制哪些粗粒度区段参与 loss 计算，当前内置：
  - `default`: 监督所有 assistant 回答（包括多轮对话中的历史 assistant，以及当前 target/response）
  - `last_round`: 只监督最后一轮 assistant 回答（当前 target/response）
  - `all`: 同时监督 system/user/prefix 与 target/response
- 当前 `loss_scale` 的落点在 `template -> SFTCollator -> ShaftSFTTrainer -> training/loss.py`
  这条链上：
  - `template` 负责把多轮消息规范化为 supervision plan，并直接产出单样本 `labels / loss_scale / span`
  - `SFTCollator` 只负责 batch 级 processor 调用、padding 与张量装配
  - `training/loss.py` 负责真正的加权 next-token loss

## 7. `eval`

用途：评估开关、频率和 best model 选择。

关键字段：

- `enabled`
- `per_device_eval_batch_size`
- `eval_strategy`
- `eval_steps`
- `loss_metrics_enabled`
- `do_sample`
- `temperature`
- `max_new_tokens`
- `online_metrics_enabled`
- `datasets`
- `metric_for_best_model`
- `greater_is_better`

额外说明：

- `eval.eval_strategy: epoch` 时，可以配合 `eval.epoch_interval` 控制“每隔多少个 epoch 才进行一次 eval”。
- `train.save_strategy: epoch` 时，可以配合 `train.save_epoch_interval` 控制“每隔多少个 epoch 才保存一次 checkpoint”。
- 当 interval 不能整除总 epoch 时，训练最后一个 epoch 仍会强制执行一次对应的 eval / save，避免漏掉最终结果。

说明：

- dataset-policy eval 现在统一产出两类聚合结果：
  - `eval_final_loss`
  - `eval_final_score`
- `loss_metrics_enabled=true` 时，框架会按 dataset policy 分别计算 teacher-forced loss，并按同一套 `weight` 聚合为 `eval_final_loss`。
- `online_metrics_enabled=true` 时，框架会按同一套 dataset policy 计算生成式任务指标，并聚合为 `eval_final_score`。
- 当前在线 eval 支持 SFT 与 GRPO。GRPO 训练侧使用 `GRPODataset` 做 rollout，在线 eval 侧保留原始 SFT 样本结构并复用 `SFTCollator` 生成评估 prompt。

### 7.1 在线 eval 配置

当前版本已支持单阶段在线 eval，目标是：

- 单阶段在线 eval
- 多数据集
- 多任务
- 每个数据集只绑定一个 task
- 通过一套统一 policy 同时支持：
  - `eval_final_score`
  - `eval_final_loss`

当前 dataset 级 eval policy 包含：

- `prediction_codec`
- `target_adapter`
- `metrics`
- `primary_metric`
- `normalizer`
- `weight`

关键约束：

1. 一个 `dataset_name` 只能绑定一个 eval policy
2. 每个 dataset 只能有一个 `primary_metric`
3. 每个 dataset 的 `primary_metric` 必须归一化到 `[0, 1]`
4. `eval_final_score` 由各 dataset 的 normalized primary score 按权重加权求和得到
5. `eval_final_loss` 由各 dataset 的 teacher-forced loss 按同样的权重加权求和得到
6. dataset policy 只要求为 `use_for_eval=true` 的数据集配置；训练专用数据集不会进入这一套聚合

示意配置如下：

```yaml
eval:
  enabled: true
  eval_strategy: epoch
  loss_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  online_metrics_enabled: true
  datasets:
    det_dataset:
      prediction_codec: det_json
      target_adapter: det_annotation
      metrics:
        - name: parse_success
        - name: parse_partial_rate
        - name: det_f1
          params:
            iou_threshold: 0.5
      primary_metric: det_f1
      normalizer:
        type: identity
      weight: 0.6

    keypoint_dataset:
      prediction_codec: keypoint_json
      target_adapter: keypoint_annotation
      metrics:
        - name: parse_success
        - name: parse_partial_rate
        - name: keypoint_pck
          params:
            threshold: 0.1
      primary_metric: keypoint_pck
      normalizer:
        type: identity
      weight: 0.4
```

说明：

- 这部分当前已经可用，但实现边界仍限定在单阶段在线 eval。
- 当前内置 metric 包括 `parse_success`、`parse_partial_rate` 与 `exact_match`，结构化任务指标需要按扩展指南新增。
- 当前内置 target adapter 只有 `target_text` 与 `extra_field`。
- 当前 `normalizer.type` 只支持 `identity` 与 `range`。
- `prediction_codec`、`target_adapter`、`metric` 会在配置加载阶段校验是否已注册，避免第一次 eval 才报错。
- 启用在线 eval 时，框架会强制使用贪心评估。
- 当 `metric_for_best_model=eval_final_score` 时，要求 `online_metrics_enabled=true`，且 `greater_is_better` 会收敛为 `true`。
- 当 `metric_for_best_model=eval_final_loss` 时，要求 `loss_metrics_enabled=true`，且 `greater_is_better` 会收敛为 `false`。
- 若配置了 dataset policy 且仍保留旧式 `metric_for_best_model=eval_loss`，框架会自动收敛为：
  - 有 online eval 时使用 `eval_final_score`
  - 否则使用 `eval_final_loss`
- 启用 dataset-policy eval 时，`report_to` 会同时上报：
  - per-dataset loss
  - per-dataset metrics
  - per-dataset normalized score
  - `eval_final_loss`
  - `eval_final_score`
- 若某个 dataset 在本次 eval 中没有样本，框架会打 warning 并跳过该 dataset，不把它计入 `final_score`。
- 若希望配置语义更直观，仍建议在 YAML 中显式写出：
  - `metric_for_best_model: eval_final_score`
  - 或 `metric_for_best_model: eval_final_loss`
- codec 已经作为共享层供 `infer` 和在线 eval 共用。

详细设计见：

- [docs/online_eval_design.md](online_eval_design.md)

## 8. `rlhf`

用途：DPO/PPO/GRPO 的结构化专属参数。

### `rlhf.dpo`

- `beta`
- `label_smoothing`
- `loss_type`
- `precompute_ref_log_probs`
- `use_weighting`

### `rlhf.ppo`

- `cliprange`
- `cliprange_value`
- `kl_coef`
- `vf_coef`
- `gamma`
- `lam`
- `whiten_rewards`
- `response_length`
- `temperature`
- `num_ppo_epochs`
- `num_mini_batches`
- `local_rollout_forward_batch_size`
- `num_sample_generations`
- `stop_token`
- `value_model_mode`
- `reward_model_mode`
- `train_value_backbone`
- `allow_untrained_reward_model`
- `allow_text_only_multimodal_ppo`

说明：

- PPO 仍是受限能力，文档与实现均不应把它表述为已完成生产方案。

### `rlhf.grpo`

- `beta`
- `rollout`
- `vllm`
- `reward_functions`

### `rlhf.grpo.rollout`

- `num_generations`
- `num_generations_eval`
- `max_completion_length`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repetition_penalty`
- `generation_kwargs`
- `cache_implementation`
- `use_transformers_paged`

### `rlhf.grpo.vllm`

- `enabled`
- `mode`: `server | colocate`
- `model_impl`: `vllm | transformers`
- `enable_sleep_mode`
- `structured_outputs_regex`
- `server_base_url`
- `server_host`
- `server_port`
- `server_timeout`
- `group_port`
- `gpu_memory_utilization`
- `max_model_length`
- `tensor_parallel_size`

说明：

- `rollout` 是 GRPO 采样行为的真源；旧的 flat 字段如 `max_completion_length` 和 `use_vllm` 仍兼容，但新配置应写入 `rollout / vllm`。
- `vllm.mode=colocate` 表示 vLLM 与训练进程共享同一组 GPU，适合 smoke 或单机资源有限场景；长训更推荐 `server` 模式，把 rollout 服务和训练进程拆开。
- 对 VLM GRPO，`data.max_pixels` 会在 `GRPODataset` 层先应用到 PIL 图像；否则 TRL/vLLM 会绕过 SFT collator，按原始大图展开过多 multimodal tokens。
- `vllm.max_model_length` 必须覆盖实际 prompt multimodal tokens 与 `rollout.max_completion_length` 的总长度；`max_completion_length=1024` 只限制生成长度，不限制图像 prompt 长度。
- 当前 GRPO 复用 `jsonl_sft` 数据格式：
  - prompt 来自 `messages` 或 `system_prompt + user_prompt`
  - reward target 来自 `target_text`
- 当前内置 reward 通过 `reward_functions` 配置，支持：
  - `exact_match`
  - `parse_success`
  - `grounding_det_f1`
  - `grounding_iou`
- 每个 reward function 由以下字段描述：
  - `name`
  - `codec`
  - `weight`
  - `params`
- `codec` 复用共享 `codec` 注册表，当前可以直接使用 `json_any / json_object / json_list / text`
- Shaft 会自动解析 `steps_per_generation`，保证 TRL 的 `generation_batch_size` 与 `num_generations` 整除约束成立。
- 当前 GRPO 明确要求 `data.mix_refresh=static`。
  - 原因是 TRL GRPOTrainer 内部使用自己的 prompt-repeat sampler 来实现 grouped generation
  - 这与 Shaft 的 `epoch_refresh` train sampler 语义冲突

## 9. `plugins`

- `hooks`
- `interceptors`

用途：

- 为训练主链注入横切增强点。

## 10. `logging`

- `level`
- `fmt`
- `file_path`
- `rank_zero_only`

## 11. `progress`

- `enabled`
- `leave`
- `mininterval`

## 12. CLI override 原则

只允许无歧义字段通过 CLI 覆盖，例如：

- `run-id`
- `seed`
- `epochs`
- `lr`
- `resume-from`
- `init-from`

禁止：

- 用 CLI 直接拼装复杂 `datasets` 列表
- 用 CLI 覆盖多层嵌套且语义不清的配置对象
