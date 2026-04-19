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
- `mix_strategy`
- `mix_refresh`
- `num_workers`
- `pin_memory`
- `persistent_workers`
- `min_pixels`
- `max_pixels`
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
- `learning_rate`
- `param_group_lrs`
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

说明：

- `train` 是 SFT 与 RLHF 共用的基础训练块。
- `optimizer_name/scheduler_name/loss_name` 走注册表。
- `param_group_lrs` 用于显式配置分组学习率。当前支持的键：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
  - `lora_params`
  - `modules_to_save`
- 没有写进 `param_group_lrs` 的组，回退到全局 `train.learning_rate`。
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
- `do_sample`
- `temperature`
- `max_new_tokens`
- `online_metrics_enabled`
- `datasets`
- `metric_for_best_model`
- `greater_is_better`

说明：

- 当前训练链仍保留 `eval_loss` 作为基础监控指标。
- 当 `online_metrics_enabled=true` 时，SFT 训练会额外挂接单阶段在线 task metric。

### 7.1 在线 eval 配置

当前版本已支持单阶段在线 eval，目标是：

- 单阶段在线 eval
- 多数据集
- 多任务
- 每个数据集只绑定一个 task
- 通过一个 `eval_final_score` 做 best model 选择

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
5. 在线 eval policy 只要求为 `use_for_eval=true` 的数据集配置；训练专用数据集不会进入在线 eval

示意配置如下：

```yaml
eval:
  enabled: true
  eval_strategy: epoch
  metric_for_best_model: eval_final_score
  greater_is_better: true
  online_metrics_enabled: true
  datasets:
    det_dataset:
      prediction_codec: det_json
      target_adapter: det_annotation
      metrics:
        - name: parse_success
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
- 当前内置 metric 只有 `parse_success` 与 `exact_match`，结构化任务指标需要按扩展指南新增。
- 当前内置 target adapter 只有 `target_text` 与 `extra_field`。
- 当前 `normalizer.type` 只支持 `identity` 与 `range`。
- `prediction_codec`、`target_adapter`、`metric` 会在配置加载阶段校验是否已注册，避免第一次 eval 才报错。
- 启用在线 eval 时，框架会强制使用贪心评估，并把 `metric_for_best_model` 收敛到 `eval_final_score`。
- 启用在线 eval 时，`report_to` 只上报 `eval_loss` 与 `eval_final_score`；per-dataset 指标只写本地日志，不进入 wandb。
- 若某个 dataset 在本次 eval 中没有样本，框架会打 warning 并跳过该 dataset，不把它计入 `final_score`。
- 若希望配置语义更直观，仍建议在 YAML 中显式写出 `metric_for_best_model: eval_final_score` 和 `greater_is_better: true`。
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
- `num_generations`
- `num_generations_eval`
- `max_completion_length`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repetition_penalty`
- `use_vllm`
- `reward_functions`

说明：

- 当前 GRPO 复用 `jsonl_sft` 数据格式：
  - prompt 来自 `messages` 或 `system_prompt + user_prompt`
  - reward target 来自 `target_text`
- 当前内置 reward 通过 `reward_functions` 配置，支持：
  - `exact_match`
  - `parse_success`
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
