# Shaft 配置参考

本文档描述 `RuntimeConfig` 的主要配置块和推荐使用方式。配置以 YAML 为主，CLI 只允许无歧义 override。

当前 `config` 已按职责拆分为多文件实现：

- `src/shaft/config/base.py`
- `src/shaft/config/model.py`
- `src/shaft/config/data.py`
- `src/shaft/config/training.py`
- `src/shaft/config/algorithm.py`
- `src/shaft/config/generation_backend.py`
- `src/shaft/config/opd.py`
- `src/shaft/config/runtime.py`

`src/shaft/config/schema.py` 只作为配置类型的聚合出口，不再承载全部 dataclass 实现。

## 1. 顶层结构与阅读顺序

训练主配置由 `RuntimeConfig` 组成。它不是一棵“下层字段覆盖上层字段”的优先级树，而是多个职责域共同
组成一条执行流水线。阅读配置时应先判断字段属于哪个职责域，再看 normalize 阶段允许哪些跨域组合。

### 1.1 `RuntimeConfig` 配置树

```text
RuntimeConfig
├── experiment                         运行身份、随机种子与输出目录
│   └── name / run_id / seed / output_dir
├── model                              模型加载、模板与微调策略
│   ├── model_type / model_name_or_path / revision / cache_dir / local_files_only
│   ├── template / trust_remote_code / torch_dtype / attn_implementation
│   ├── experts_implementation / device_map
│   └── finetune
│       ├── mode / target_modules / target_parameters
│       ├── freeze.groups / freeze.prefixes / freeze.regex
│       ├── freeze.trainable_prefixes / freeze.trainable_regex
│       ├── lora_r / lora_alpha / lora_dropout / lora_bias / use_rslora
│       └── qlora_load_in_4bit / qlora_use_double_quant / qlora_quant_type / qlora_compute_dtype
├── algorithm                          训练目标选择
│   ├── name: sft | dpo | ppo | grpo | opd
│   └── params
│       └── auxiliary_loss_weights.<term_name>（仅 SFT，稀有覆写）
├── data                               从记录到 local microbatch
│   ├── catalog_path / catalog_names / datasets
│   ├── schedule
│   │   └── mixing / shuffle
│   ├── prompt_sources
│   │   └── <dataset>.path / apply_to / seed / formulation_sources
│   ├── batching
│   │   ├── grouping
│   │   ├── cardinality
│   │   ├── packing.mode
│   │   ├── layout
│   │   └── buffer_size / cost_cache_size / max_tokens_per_microbatch / resource_budgets
│   ├── num_workers / prefetch_factor / pin_memory / persistent_workers
│   ├── record_cache_dir / media_snapshot_id / image_cache_size
│   └── min_pixels / max_pixels / max_length / add_eos_token
├── train                              优化、更新、保存与分布式执行
│   ├── duration.unit / duration.value
│   ├── per_device_train_batch_size / gradient_accumulation_steps
│   ├── optimizer_name / scheduler_name / loss_name / loss_scale
│   ├── learning_rate / warmup_ratio / weight_decay / max_grad_norm
│   ├── scheduler_num_cycles / scheduler_power
│   ├── adam_beta1 / adam_beta2 / adam_epsilon
│   ├── bf16 / fp16 / gradient_checkpointing / full_determinism
│   ├── save_strategy / save_steps / save_epoch_interval / save_total_limit / save_only_model
│   ├── max_shard_size
│   ├── load_best_model_at_end / save_final_model / save_final_state
│   ├── init_from_checkpoint / resume_from_checkpoint
│   ├── efficiency
│   └── distributed
│       ├── strategy: ddp | fsdp | deepspeed
│       ├── ddp
│       │   └── static_graph
│       ├── fsdp
│       └── deepspeed
├── eval                               loss eval、生成评估与模型选择
│   ├── enabled / eval_strategy / epoch_interval / eval_steps
│   ├── per_device_eval_batch_size / min_pixels / max_pixels
│   ├── loss_metrics_enabled / online_metrics_enabled
│   ├── do_sample / temperature / max_new_tokens
│   ├── metric_for_best_model / greater_is_better
│   └── datasets.<name>
│       └── prediction_codec / target_adapter / target_adapter_params / metrics / primary_metric / normalizer / weight
├── rlhf                               DPO/PPO/GRPO 结构化专用参数
│   ├── dpo
│   ├── ppo
│   └── grpo
│       ├── rollout
│       ├── vllm
│       └── reward_functions
├── opd                                on-policy direct distillation 专用参数
│   ├── teacher
│   ├── rollout
│   └── objective
├── plugins                            hooks / interceptors
├── logging                            日志格式与 rank 策略
└── progress                           终端、plain 与持久化进度
```

树中的路径节点和 `/` 分隔项都对应真实 YAML key；冒号后是允许值，右侧中文是职责说明。这里不使用
`optimizer`、`checkpoint`、`eval.schedule` 等 strict loader 不接受的伪节点。

### 1.2 从 YAML 到运行时配置

```text
训练 YAML
  │
  ├── 校验 batching 四轴是否显式声明
  ├── 展开 catalog_names，并与 inline datasets 合并
  ├── 解析配置相对路径
  ├── 严格构造 dataclass；未知字段直接拒绝
  ├── 应用 dataclass 默认值
  └── normalize + 跨层能力校验
          │
          ▼
     resolved RuntimeConfig
```

CLI 只允许少量无歧义覆写；覆写后会再次 normalize。不存在任意深层 merge，也不会因为 YAML 中后出现某个
字段，就自动覆盖另一个职责域的语义。所有训练 YAML 都必须显式声明
`data.batching.grouping/cardinality/packing.mode/layout`，即使 dataclass 中存在默认值。

所有 dataclass boolean 字段统一经过严格 parser：接受 YAML boolean、`0/1` 以及
`true/false/yes/no/on/off`（不区分大小写）的字符串形式，拒绝其他含糊文本。禁止用 Python
`bool(value)` 解析外部配置；例如字符串 `"false"` 必须得到 `False`，不能按非空字符串变成 `True`。

### 1.3 训练数据的执行树

```text
catalog_names + inline datasets
        │  展开并合并；同名 dataset 直接报错，不做覆盖
        ▼
source load -> datasets[*].offline_transforms -> train/val record stores
        │
        ├── train
        │     ▼
        │   data.schedule: draw_id -> source + row
        │     ▼
        │   datasets[*].online_transforms
        │     ▼
        │   data.prompt_sources: formulation -> prompt variant -> prompt + target
        │     ▼
        │   grouping -> cardinality -> packing -> layout/collator
        │     ▼
        │   每个 rank 的 local microbatch
        │     │  × data-parallel world size
        │     ▼
        │   global microstep
        │     │  × train.gradient_accumulation_steps
        │     ▼
        │   optimizer step（PPO 例外，见下方 PPO 几何说明）
        │
        └── val/eval
              ▼
            datasets[*].online_transforms
              ▼
            PromptSource（对应 source 的 apply_to=all 时）
              ▼
            eval fixed padded batch
            B = eval.per_device_eval_batch_size
```

`schedule` 决定训练哪些 logical draws；dataset transforms 与 PromptSource 决定同一个 draw 如何呈现；
`batching` 决定 draws 如何组合执行。planned batching 可以在有界窗口内重排 draws，但不能额外丢失、复制
或改变 schedule 已产生的 `draw_id` multiset。weighted schedule 在 canonical draw 顺序中对每个 source
独立无放回；source 全部行
耗尽后才进入下一轮确定性置换。这里守恒的是 draw identity；batching 可以重排 draws，GRPO 也有算法要求的
grouped repeat，因此不能从最终 tensor 顺序反推 canonical source cycle。

offline transform 改变正式 record store，因此发生在 schedule 之前；online transform 和 PromptSource
只生成 draw view。planned 路径会在 cost planning 时重放同一份 planning-safe view，worker 真正取样时再按
相同 draw context 确定性执行，并不会提前物化全量 transformed samples。train 的 schedule/mixing/
grouping/packing 不作用于 eval；`data.datasets[*].weight` 只控制 train schedule，
`eval.datasets.<name>.weight` 只控制指标聚合。

### 1.4 数据层级职责矩阵

配置树描述“字段放在哪里”，执行树描述“运行时先后顺序”。下面的矩阵描述每一层究竟有权改变什么；后层
消费前层产物，但不会覆盖前层语义：

| 层级 | 输入 -> 输出 | 可以改变 | 不得改变 |
|---|---|---|---|
| `catalog / datasets` | 配置 -> source snapshots | source 集合、路径、权重、train/eval 资格 | batch 大小、训练顺序 |
| `schedule` | `draw_id` -> `SampleRef` | source mixing、source 内 row 顺序 | prompt 内容、batch 成员关系 |
| `datasets[*].online_transforms` | `SampleRef` -> logical sample view | dataset 声明的 planning-safe 在线视图 | `draw_id`、source 权重、row identity |
| `prompt_sources` | aligned offline targets -> selected prompt/target view | formulation source、静态权重、prompt variant | `draw_id`、dataset mixing、业务 target 构造 |
| `batching.grouping` | draws -> 有界重排后的 draws | 候选分组和局部顺序 | draw multiset、prompt variant |
| `batching.cardinality` | draws -> local batch membership | 每卡本 microstep 的 physical-pack 数 | pack 内 segment 组合、tensor layout |
| `batching.packing` | logical samples -> physical packs | 多个完整 segment 是否共享一个 pack | 样本内容、segment 内 token 顺序 |
| `batching.layout` | physical packs -> model tensors | padded/varlen tensor 与 attention 表示 | sampling、pack 数、优化步数 |
| `train` | local microbatches -> optimizer updates | DP/GA、loss、optimizer、scheduler、checkpoint | 数据 source/mixing 语义 |

因此不存在通用的“下层覆盖上层”规则。真正的优先关系只有少数显式入口：YAML 先完成一次加载与
normalize，CLI 的白名单 override 再写回对应字段并二次 normalize；`algorithm.name` 选择唯一算法主链；
`normalize_runtime_config()` 对跨层组合做 fail-closed 校验。模型 execution policy 只能声明某个
layout/backend 是否可执行，不能把不支持的组合悄悄降级。

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
- `revision`
- `cache_dir`
- `local_files_only`
- `template`
- `trust_remote_code`
- `attn_implementation`
- `experts_implementation`
- `torch_dtype`
- `device_map`
- `finetune`

`model.experts_implementation` 只对已解析为 MoE 的 Qwen VL profile 生效。`null` 或 `auto` 解析为
Shaft 为该 profile 声明的确定性默认值；当前 Qwen3VL 与 Qwen3.5/3.6 MoE 的默认值均为
`grouped_mm`。Shaft 会把解析值显式传给 Transformers 并校验载入后的 root/text config，后端不可用时
直接失败，不允许静默回退到 `eager`。请求值与解析值均进入 exact-resume 语义，model-plan fingerprint
也绑定 profile 默认值。dense Qwen 或其它未声明 expert backend 的 profile 配置非空值会明确拒绝。

Qwen3VL MoE 的推荐执行字段与 PEFT 边界：

```yaml
model:
  model_type: qwen3vl
  model_name_or_path: models/Qwen3-VL-30B-A3B-Instruct
  torch_dtype: bfloat16
  experts_implementation: grouped_mm
  finetune:
    mode: lora
    target_modules: [auto]
    target_parameters: [auto]
    lora_dropout: 0.0
```

`model.device_map` 只用于 HF 本地推理装载。Shaft 训练在构造 `TrainingArguments`、加载模型权重之前拒绝
非空值；训练设备放置必须由 `train.distributed.strategy` 的 DDP/FSDP/DeepSpeed 负责，避免 torchrun
多个 rank 把完整模型装到同一张卡，或在 35B 权重已经物化后才由 Accelerate 晚失败。

### `model.finetune`

关键字段：

- `mode`: `full | lora | dora | qlora`
- `freeze`
- `target_modules`
- `target_parameters`
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

- `model_type=qwen3vl` 同时适用于 HF `qwen3_vl` dense 和 `qwen3_vl_moe`。dense 真实训练证据覆盖
  2B/4B；MoE 的 `Qwen3-VL-30B-A3B-Instruct` 已通过两卡 BF16 FSDP LoRA 短门禁，包括真实图片、router
  objective、标准 adapter exact resume、backend-native optimizer state 和标准 PEFT reload。它不代表两卡
  支持 30B 全参数 AdamW，
  也不替代长程收敛验收；dense 32B 仍未做生产 gate。
- `model_type=qwen35vl` / `qwen36vl` 适用于 Qwen3.5 / Qwen3.6 新一代 VLM。两者共享
  同一套 loader、processor policy 和模板默认值；`qwen36vl` 是为了让训练配置保留 3.6 口径。
- Qwen3.5 / Qwen3.6 训练需要安装支持 `qwen3_5`/`qwen3_5_moe` 架构的 Transformers。当前
  `qwen35vl` meta 会在运行前检查 `transformers>=5.10.1` 以及
  `transformers.models.qwen3_5` 模块是否存在；MoE profile 还要求 `transformers.models.qwen3_5_moe`。
  MoE padded SFT 已实现 router auxiliary objective、full/LoRA 保存恢复和 HF/PEFT 导出。Qwen3.6-27B
  dense 已有真实权重短程训练证据；MoE 证据仍主要来自 tiny upstream architecture 的 CPU 与 CUDA gate，
  不能解释为真实 35B MoE 权重的生产容量。完整验收条件见 `docs/TODO.md`。
- **MTP 当前不在 Shaft 支持范围内。** Qwen3.5/3.6 上游 artifact 可能携带 `mtp.*` speculative-draft
  权重，但当前 Transformers 标准 Qwen3.5/3.6 model class 不实例化这些模块，并把它们作为 ignored
  unexpected keys。Shaft 因而不加载、不训练、不恢复、不导出 MTP，也没有 `mtp_loss` 或 MTP
  speculative-serving 配置。Shaft full SFT 产物只包含标准 target model；PEFT adapter 也不声明对基座 MTP
  的兼容性。`config.text_config.mtp_num_hidden_layers > 0` 只是上游 config 残留，不能作为 checkpoint 包含
  MTP 的证据。该限制不影响标准逐 token autoregressive 推理和任务质量，但 Shaft 产物不得启用 vLLM/SGLang
  MTP speculative decoding；未来若开发该能力，必须先补模型装配、objective、artifact 完整性和部署门禁，
  不能只在导出时复制 `mtp.*` 文件。
- Qwen3.5/3.6 dense/MoE 的 `layout=varlen` 只开放 CUDA + DDP + bf16/fp16，且要求
  `flash-attn`、`flash-linear-attention` 与 `causal-conv1d`。这是 hybrid full/linear attention 的完整
  segment-isolation contract，不能只安装 FlashAttention 后强行开启。Qwen3.6 在 Transformers 5.10.1
  中复用 `qwen3_5` architecture；`qwen36vl` 是产品版本 alias，不是另一套 HF forward。现有 varlen
  dense/MoE gates 使用 BF16；FP16 当前仅是 runtime allowlist，尚未完成 varlen 专项验收。
- 仓库基础依赖要求 `transformers>=5.10.1,<6`；当前验证过的 lock 口径固定为
  `transformers==5.10.1`。旧的 `>=4.57.6` 声明与 checkpoint/runtime 实现不一致，已删除，不能把未测试的
  4.x 环境当成支持面。当前已接通 Qwen3.5 / Qwen3.6 dense/MoE 的 HF 本地训练与推理接口；Qwen3.6-27B
  dense 有短程真实权重训练记录，MoE 的完整 backend 矩阵仍限于 tiny upstream architecture。
  `qwen-next` extra 用于进一步精确固定新一代 Qwen 口径；业务 vLLM
  推理镜像使用同一份
  `uv.lock`，当前标准为 `vllm==0.19.1` + `transformers==5.10.1`。对本地 HF 训练环境，
  推荐执行：

  ```bash
  uv sync --extra dev --extra train --extra distributed --extra qwen-next --extra gpu
  ```

  Qwen3.5/3.6 MoE 的证据状态必须按下表解释：

  | 组合 | 状态 | 当前证据 |
  | --- | --- | --- |
  | DDP + full/LoRA + padded；DDP + full varlen | tiny validated | upstream tiny MoE、真实 processor、fresh/resume/export |
  | FSDP + LoRA `target_parameters` + padded | tiny validated | 2-rank CUDA、router/expert/ordinary LoRA 更新、backend-native exact resume |
  | DeepSpeed ZeRO-3 + full + padded | tiny validated | 2-rank CUDA、router/各 expert 更新、ZeRO shard exact resume、HF reload |
  | 真实 35B MoE 权重长训练 | wired, not production-validated | 尚缺目标硬件容量、吞吐、长程数值和目标集收敛 gate |
  | ZeRO-3 + PEFT `target_parameters` | rejected | PEFT 0.18.1 无法对构造期 empty shard 注入 parameter wrapper |
  | MoE QLoRA；预量化 FP8 artifact 训练 | rejected | fused experts 未被 bitsandbytes 覆盖；FP8 artifact 仅供推理 |

  `tiny validated` 只证明框架合同和上游架构行为，不等同于真实发布权重的生产验收。

  业务推理环境不要自行拼装依赖版本，应使用 `docker/inference/` 中的推理镜像或用同一份
  `uv.lock` 构建。推理效果对 prompt、pixel budget、generation 参数和 JSON 解析都敏感，
  不能只对齐模型权重；镜像构建和 `shaft-contract-smoke` 验收见
  `docker/inference/README.md`。
- `qwen35vl` / `qwen36vl` 默认使用 `template=qwen35vl`，该模板会在 generation prompt 中关闭
  thinking，避免结构化 JSON 任务无意训练或生成 `<think>` 内容。确实需要 CoT 数据时，显式设置
  `model.template: qwen35vl_thinking`。
- `data.min_pixels/max_pixels` 是否以及如何传给 processor 由模型的 `ProcessorPolicy` 唯一决定；
  `qwen_vl` 使用 `images_kwargs`，通用/identity policy 默认不假设 processor 支持 pixel budget。
  新模型不得在 collator、template 或 pipeline 中再维护一份转发开关。
- 从 Qwen3-VL 切换到 Qwen3.6-VL 训练时，核心差异应只落在模型字段，例如：
  `model_type: qwen36vl`、`model_name_or_path: models/Qwen3.6-27B`、必要时把
  `train.distributed.strategy` 切到 `fsdp` 或 `deepspeed`。`data`、`algorithm`、SFT target
  格式和 Qwen3-VL 主链保持一致。
- Shaft 会对本地 `config.json` 的 HF `model_type` 做早期校验：`qwen3vl` 期望
  `qwen3_vl` 或 `qwen3_vl_moe`，`qwen35vl` / `qwen36vl` 期望 `qwen3_5` 或
  `qwen3_5_moe`。这能在模型加载前
  发现 `model.model_type` 与权重目录不匹配的问题。
- 同一本地 config 还会解析为 `ResolvedModelDescriptor`，按 `hf_model_type/architectures` 选择 dense/MoE
  variant profile。非本地 HF repo 会按 `revision`、`cache_dir` 和 `local_files_only` 读取 cache/Hub config；
  目录名仅作为已知 catalog 的离线 hint。无法取得 config 且名称又不在已知 catalog 的多变体模型会
  fail closed，避免未知 MoE 静默使用 dense FSDP layer policy。`local_files_only=true` 禁止网络回退，但仍
  允许使用指定 cache；loader 与 descriptor resolver 消费同一组字段。
- `ResolvedModelPlan` 在加载前一次性决定 configured/effective artifact、base/adapter/full-checkpoint init kind、
  descriptor、variant adapter 和 fingerprint。`init_from_checkpoint` 指向 full HF checkpoint 时，该 checkpoint
  是模型 architecture 真源；指向 PEFT adapter 时，architecture 仍来自 `model_name_or_path` 的 base model。
- PEFT adapter 会解析为 `ResolvedAdapterInit`，并绑定 canonical config、声明的 base artifact 和权重 manifest。
  adapter base 必须能证明与当前 resolved model profile/config 等价；加载时比较当前运行时 PEFT canonical
  target/modules-to-save 及全部 state key/shape。PEFT 把完整模块路径保存为等价后缀集合时，不会把同一
  `target_modules: [auto]` adapter 误判为不兼容。
- `configs/train/qwen36_sft_27b_fsdp_example.yaml` 是最小 SFT/FSDP+LoRA 训练示例；其中
  `transformer_layer_cls_to_wrap: ["auto"]` 会按 `qwen36vl` 模型族解析为 Qwen3.5/3.6 的 dense
  decoder 与 vision block 类名。当前 Qwen3.6 / Transformers 5.10 / PyTorch 2.10 组合下，
  `distributed.fsdp.activation_checkpointing` 默认关闭，保留 `train.gradient_checkpointing` 走模型侧
  checkpointing；FSDP activation wrapper 在 Qwen3.6 linear-attention 层上会触发 recompute tensor
  数量不一致。
- 8x80GB 上 Qwen3.6-27B full-parameter FSDP + AdamW 会在 optimizer step 触达显存上限；默认示例
  使用 LoRA。full fine-tune 应使用 DeepSpeed ZeRO-3、CPU offload、低精度/8-bit optimizer 或更多显存资源。
- 对本地 HF sharded checkpoint，Shaft 会在模型装配前读取 `model.safetensors.index.json` 或
  `pytorch_model.bin.index.json`，确认索引引用的 shard 文件都已存在。半下载目录会在进入
  `from_pretrained` 前直接报出缺失 shard，避免把下载不完整误判为模型架构或训练配置问题。
- `target_modules=["auto"]` 表示交给模型族 `peft policy` 自动解析。
- `target_parameters` 用于直接对没有独立 module 的 2-D/3-D fused 参数应用 PEFT LoRA。空列表表示不启用；
  `["auto"]` 必须由模型 profile 提供默认值，否则启动前报错。Qwen3VL MoE 与 Qwen3.5/3.6 MoE 默认展开 routed-expert
  `gate_up_proj/down_proj` 与 router `gate.weight`，并可与 `target_modules=["auto"]` 同时使用。
- `target_parameters` 只适用于 adapter mode，要求 `peft>=0.18.1`、`lora_dropout=0`，不支持 DoRA；
  Qwen3.5/3.6 MoE 同时拒绝 QLoRA，因为 bitsandbytes 不会量化这些 fused 3-D expert 参数。resolved 参数名
  进入 finetune/adapter signature，init、resume 与 export 会做一致性校验。
- Qwen3.5/3.6 的预量化 FP8 artifact 在 Shaft 中是 inference-only；dense 和 MoE 训练都必须从未预量化
  base checkpoint 启动（发布权重通常为 BF16），不能靠路径或未知 dtype 静默退化。FP16 AMP full
  finetune 仍按精度合同以 FP32 参数加载。
- `freeze.groups` 当前只允许：
  - `language_model`
  - `vision_tower`
  - `aligner`
  - `generator`
- `freeze.regex` 与 `freeze.trainable_regex` 必须是合法正则。
- `init_from_checkpoint` 与 `resume_from_checkpoint` 各自的 artifact/state 兼容性由
  `training/checkpointing.py` 统一校验；两种启动语义应二选一，当前配置层不会替用户自动选择。

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
- `schedule`
- `transforms`
- `batching`
- `num_workers`
- `prefetch_factor`
- `pin_memory`
- `persistent_workers`
- `record_cache_dir`
- `image_cache_size`
- `min_pixels`
- `max_pixels`
- `max_length`
- `add_eos_token`

### `data.batching`

`data.batching` 不负责选择数据源，也不是一个单独的“batch size”字段。它用四个正交轴描述 logical samples
如何组成每个 rank 的 local microbatch：

```text
已由 schedule 选中的 logical draws
        │
        ▼
grouping
        │  none / length / bounded_cost
        │  决定候选 draws 是否以及如何分组、重排
        ▼
packing + cardinality
        │  logical samples -> physical packs
        │  cardinality 使用 train.per_device_train_batch_size 作为数量 B
        ▼
layout
        │  padded / varlen
        ▼
local microbatch tensor
```

四轴的职责如下：

| 轴 | 回答的问题 | 当前取值 |
|---|---|---|
| `grouping` | 已选 draws 如何在允许的窗口内分组、排序 | `none / length / bounded_cost` |
| `cardinality` | 每个 rank、每个 microstep 有几个 physical packs | `fixed / token_budget` |
| `packing.mode` | 一个 physical pack 内包含几条 logical samples | `none / greedy` |
| `layout` | pack 最终如何表示为 tensor/attention | `padded / varlen` |

`train.per_device_train_batch_size` 不是第五个 batching 轴，而是 cardinality 使用的数值参数：

- `cardinality=fixed`：每个 rank 的完整 local microstep 产生
  `B=per_device_train_batch_size` 个 physical packs。普通 `grouping=none + duration=epochs` 的最后一个
  DataLoader 尾批沿用 HF 行为，可能少于 `B`。
- `cardinality=token_budget`：`B` 是硬上限，每个 rank 实际产生 `1..B` 个 physical packs。
- `packing=none`：一个 physical pack 恰好是一条 logical sample。
- `packing=greedy`：一个 physical pack 可以包含多条 logical samples，此时 `B` 不等于样本数。

所有训练 YAML 必须显式写 grouping、cardinality、packing 和 layout。保留普通固定 batch 时：

```yaml
data:
  batching:
    grouping: none
    cardinality: fixed
    packing:
      mode: none
    layout: padded
```

启用有界成本感知批次时：

```yaml
data:
  media_snapshot_id: example-media-v1
  batching:
    grouping: bounded_cost
    cardinality: token_budget
    packing:
      mode: none
    layout: padded
    buffer_size: 64
    cost_cache_size: 65536
    max_tokens_per_microbatch: 10000
    resource_budgets:
      vision_patches: 16384
```

启用 Qwen3VL whole-sample packing 时：

```yaml
data:
  max_length: 8192
  media_snapshot_id: my-media-v1
  batching:
    grouping: length
    cardinality: fixed
    packing:
      mode: greedy
    layout: varlen
    buffer_size: 64
    cost_cache_size: 65536
    resource_budgets:
      vision_patches: 16384
```

当前可执行组合矩阵如下；不在表中的组合会明确失败，不会静默退回另一种模式：

| grouping | cardinality | packing | layout | 语义与附加边界 |
|---|---|---|---|---|
| `none` | `fixed` | `none` | `padded` | SFT/DPO/GRPO 的普通 fixed 路径；PPO 也只接受该四轴，但几何由 TRL 接管 |
| `length` | `fixed` | `none` | `padded` | 按长度分组，仍使用普通 padding |
| `length` | `fixed` | `none` | `varlen` | 不 packing，但使用 segment-aware varlen tensor |
| `length` | `fixed` | `greedy` | `varlen` | whole-sample packing，多条 logical samples 可进入一个 pack |
| `bounded_cost` | `fixed` | `none` | `padded` | 文本/视觉成本分组，每卡严格 `B` 个 packs |
| `bounded_cost` | `token_budget` | `none` | `padded` | 在 local token/resource budget 内动态选择 `1..B` 个 packs |

选择配置时可以按下面的决策树理解；它描述当前实现能力，不是自动推断逻辑，YAML 仍需显式写出四个轴：

```text
是否需要改变普通 HF fixed padded batch？
├── 否
│   └── none + fixed + none + padded
└── 是
    ├── 只想让相近长度样本靠近，仍接受 padding
    │   └── length + fixed + none + padded
    ├── 想使用 varlen，但暂不合并样本
    │   └── length + fixed + none + varlen
    ├── 想把多条完整样本装入 physical pack
    │   └── length + fixed + greedy + varlen
    └── 想同时按文本与视觉成本做有界分组
        ├── 每卡 pack 数必须固定
        │   └── bounded_cost + fixed + none + padded
        └── 允许预算决定每卡实际 pack 数
            └── bounded_cost + token_budget + none + padded
```

四轴可组合并不等于所有算法/backend 都已执行验收；当前能力边界如下：

| 路径 | 四轴 | 算法 | duration | distributed |
|---|---|---|---|---|
| 普通 fixed | `none/fixed/none/padded` | SFT/DPO/GRPO；PPO 使用独立几何 | `steps / epochs` | DDP/FSDP/DeepSpeed 可装配，重型验收按模型另行声明 |
| length padded | `length/fixed/none/padded` | SFT | `steps` | DDP |
| length varlen | `length/fixed/none或greedy/varlen` | SFT | `steps` | DDP |
| bounded cost fixed | `bounded_cost/fixed/none/padded` | SFT | `steps` | DDP；FSDP `full_shard + full_state_dict`；DeepSpeed ZeRO-3 |
| bounded cost token budget | `bounded_cost/token_budget/none/padded` | SFT | `steps` | DDP |

`length` 与 `bounded_cost` 统称 planned batching，均要求 `media_snapshot_id`、planning-safe transforms
和提供 exact image-cost policy 的模型。padded 路径可估算并执行单样本有序多图；varlen/greedy packing
仍只支持单图。`length` 要求 `data.max_length`；`bounded_cost` 要求
`max_tokens_per_microbatch`；`greedy` 还要求 `length + varlen + vision_patches guard`。varlen 当前支持
Qwen3VL 与 HF `qwen3_5`（Qwen3.5/Qwen3.6）image SFT；其它模型族以及 greedy+padded、
greedy+token-budget、bounded-cost+greedy 会 fail closed。这里的“可装配”不等于所有模型规模和 topology
都已完成生产级 E2E。

PPO 几何是明确例外：`per_device_train_batch_size=B` 仍是 inner update microbatch 大小，但 TRL rollout
DataLoader 每 rank 一次取得 `B × gradient_accumulation_steps` 条，再受 `rlhf.ppo.num_mini_batches` 与
`num_ppo_epochs` 控制 inner optimizer updates。因此下文 `B × world_size × GA` 的 physical-pack/optimizer
公式只适用于 SFT/DPO/GRPO，不得套到 PPO。

- 旧 `data.batching.strategy`、`cost_aware`、`dynamic_cost_aware`、
  `fixed_guard`、`planning_window`、`cost_plan_cache_dir`、`rank_balance` 和
  `train.optimizer_batch` 已删除；出现时按未知配置字段拒绝，不提供隐式迁移。
- `bounded_cost` 当前只支持 SFT 和 `train.duration.unit=steps`。DDP 支持 `fixed` 与 `token_budget`；
  FSDP/DeepSpeed 首版只开放 `fixed + none + padded`，并分别硬性要求
  `full_shard + full_state_dict` 与 ZeRO stage 3。sharded backend 继续拒绝 `token_budget`、`length`、
  greedy packing、varlen、其它 FSDP sharding/state-dict 组合和 ZeRO stage 0/1/2，不会静默退回普通 batch。
- `buffer_size` 是 planner 中最多常驻的轻量 `SampleRef + SampleCost` 数量。`fixed` 时必须至少等于
  `DP world size × per_device_train_batch_size`；`token_budget` 时至少等于 DP world size，实际配置应保留
  足够 lookahead 以找到相近成本样本。它不是全训练 horizon，也不会导致启动时扫描 `steps * samples`。
- `media_snapshot_id` 是 length/bounded planned 路径必填的不可变媒体快照 id。JSONL/Arrow record fingerprint 不会为了
  startup 安全而扫描全部外部图片；若图片集合、内容或尺寸可能改变，必须先生成新快照并更换该 id。
- `cost_cache_size` 同时限制 prompt-variant sample-cost LRU 与 canonical image-header LRU；`0` 表示
  禁用缓存。缓存不包含解码图像、processor tensor 或完整文本 token tensor。
- `train.per_device_train_batch_size` 是唯一 local physical-pack count 配置。`cardinality=fixed` 时每个 rank
  的完整 microstep 生成该数量的 pack；普通 epoch 尾批是上述 HF 兼容例外。
  SFT 使用跨 rank 的真实有效 token denominator，因此在 rank step 数相同的时候允许最后一个同步 step
  出现不等长 local batch；DPO 的 TRL loss/metric 需要各 rank cardinality 相同，分布式 epoch 尾部若不能
  被 `per_device_train_batch_size * world_size` 整除会在模型加载前 fail closed，不会复制或丢弃样本。
  `cardinality=token_budget` 时它是每个 rank 的硬上限，实际数量为
  `[1, per_device_train_batch_size]`。不存在第二个 max-samples 开关。
- `packing=none` 时一个 physical pack 恰好是一条 logical sample；`packing=greedy` 时一个 pack 可含多条
  logical segments，因此训练日志分别报告 `logical_segments`、`physical_packs` 和 `segments_per_pack`，不能
  用 tensor 的 batch 维或 pack count 反推 logical sample count。
- bounded-cost 的 `max_tokens_per_microbatch` 是硬安全上限，计算方式为
  `local sample count * processor 后最长 LLM sequence`，不是原始字符数。length 路径不重复配置该字段，
  local token cap 固定派生为 `per_device_train_batch_size * data.max_length`。
- `resource_budgets.vision_patches` 是 local batch 内所有图片 pre-merge vision patch 的可选总上限，用于避免多个大图
  被合到同一 batch。它必须能容纳 processor pixel budget 允许的最大单样本；例如 Qwen patch-size 16、
  `max_pixels=4,000,000` 的配置应至少使用 16,384。单样本超限会在该 draw 首次进入 buffer 时明确失败。
- 每个完整 global microstep 固定输出 W 个非空 local microbatch。完整 optimizer frame 中，`fixed` 的
  physical pack count 是 `B × W × GA`；`token_budget` 的范围是 `[W × GA, B × W × GA]`。普通 epoch
  尾批继续遵循 HF DataLoader 行为。SFT Trainer 对一个 optimizer frame 内
  真实 `labels/loss_scale` 求跨 GA/DP 的 global denominator，因此 local cardinality 不同不会改变 token
  权重，LR/scheduler 仍以 optimizer step 为单位。

#### 通用 token-budget 配置展开示例

上面的 `weighted + bounded_cost + token_budget` 示例可以展开为：

```text
weighted source schedule
        │
        ▼
PromptSource / online transform
        │
        ▼
bounded_cost，lookahead buffer=64
        │
        ▼
token_budget，B=per_device_train_batch_size=2
        │
        ▼
packing=none：1 pack = 1 logical sample
        │
        ▼
layout=padded
```

使用 8 个 data-parallel rank、`gradient_accumulation_steps=4` 时：

```text
每卡每 microstep       1..2 packs = 1..2 logical samples
每个 global microstep  8..16 logical samples
每个 optimizer step    32..64 logical samples
```

这里的几个限制不是相互覆盖关系：

| 字段 | 限制对象 |
|---|---|
| `data.max_length=10000` | 单条 logical sample 的 processor 后 `prefix + target + EOS` 严格上限 |
| `max_tokens_per_microbatch=10000` | 每卡整个 padded local microbatch 的 aggregate token 成本 |
| `data.max_pixels=4000000` | 单张图进入模型 processor 的像素预算 |
| `resource_budgets.vision_patches=16384` | 每卡整个 local microbatch 的视觉 patch 总预算 |
| `buffer_size=64` | planner 的轻量候选 draw 窗口，不是 batch size |
| `cost_cache_size=65536` | sample cost/image-header LRU 容量，不是训练计划长度 |

- planner 强制消费最老 draw，并从有界 buffer 中选择成本相近的样本。`fixed` 模式贪心快路径失败时使用
  有节点上限的 deterministic exact fallback；`token_budget` 模式以单样本可行为底线，在预算内尽量填充
  到上限。完整 planning frame 还会按 rank 累计成本重新分配 batches。因此不会饿死长样本，也不会改变
  mixing draw multiset。weighted mixing 在 bounded 模式要求 `data.schedule.shuffle=true`。
- 多 rank startup 对第一个 buffer 的 cost/plan digest 做一致性校验，并把该 preflight plan 交给正式 sampler
  复用。首个 forward 前会原子规划完整 GA frame，
  cost call 上界是 `buffer + (GA - 1) * W * per_device_train_batch_size`，而不是完整 duration；运行中最坏 host 预取内存还
  需计入 `num_workers * prefetch_factor`。
- 所有 rank 在 immutable snapshot contract 下独立重放同一个 canonical global microbatch stream。
  DDP 保持原有全局扁平 BatchSampler，由 Accelerate 在 `split_batches=false/even_batches=false` 下执行唯一一次
  rank 分片；FSDP/DeepSpeed 则由同一 `ShaftPlannedBatchSampler` 确定性选择本 rank 的 local batch，Trainer
  直接使用原始 DataLoader，禁止 Accelerate/后端再次分片或补齐。两条路径的 sampler 内都不做 collective，
  避免 worker 预取速度不同造成死锁；首 buffer 漂移和 startup 单 rank 错误会先做 all-rank 聚合再退出。
- duration-independent spec、已提交 global microstep、FIFO buffer 及实际累计成本，作为
  `ShaftBatchPlanningCallback` stateful payload 写入 checkpoint 的 HF `trainer_state.json`。只保存已完成
  optimizer step 对应 snapshot，不保存预取推进后的 live cursor。planned 状态写入 training 层通用
  `shaft_checkpoint_commit.json` 的 `batch_planning` extension，不再单独发布 completion marker。
- resume 会验证 source/media snapshot/mixing/prompt/tokenizer/processor/template、world size、buffer、budgets，
  以及 duration/GA/optimizer/scheduler exact-resume contract；随后从 committed state 继续并禁用 HF 二次
  data skip。persistent workers 使用 DataLoader 专用 generator，不改变模型 RNG。
- run root 的 `shaft_batching_run_metadata.json` 保存 resolved 策略、`local_pack_count_range`、
  `global_pack_count[_range]`、`optimizer_pack_count[_range]`、DP/GA、pixel budget、source weights、media
  snapshot id、buffer/cache/caps、versioned canonical
  `batch_contract`、`batch_contract_fingerprint`、完整 `train_input_contract` 和 bounded 路径的
  `planner_spec_fingerprint`；启用 W&B 时同一 payload 写入 `shaft_batching` run config。
  `train_input_contract` 绑定 data execution、train Dataset 的 source/runtime method identity 与 Pillow
  runtime、model/processor、tokenizer artifact 及 wrapper/backend package version、template、
  collator/input-policy、pixel/token limits；SFT 还绑定 sequence-execution fingerprint。`cost_cache_size`
  是性能审计字段，不参与 exact-resume fingerprint。
- 只要启用可恢复 checkpoint（`save_strategy != no` 且 `save_only_model=false`），训练输入契约就必须完整：
  active online transform 必须显式版本化，
  `media_snapshot_id` 必须标识 immutable 媒体快照，Dataset/Pillow runtime 必须可稳定识别，tokenizer 必须
  提供完整 backend/声明式 artifact identity，input builder 必须声明 `SHAFT_INPUT_POLICY_VERSION`。未知的
  component/config state 不会退化成“只记录类型”，而会把契约标为 incomplete。data-side identity 与旧
  checkpoint 的 input-contract payload 在大模型加载前预检；完整 processor/model identity 在装配后做第二阶段
  校验。否则 fresh startup 直接失败；关闭 checkpoint 时可以运行，但 metadata 与日志会明确标为不可 exact
  resume。
- checkpoint storage protocol 由 `train.distributed.strategy` 显式选择。SFT、DPO、GRPO 的 DDP/native-HF
  路径使用 `committed_manifest`：`ShaftCheckpointCommitMixin` 在 super save 前撤销旧 commit 并暂缓
  rotation；模型/adapter、Trainer、optimizer、scheduler、每-rank RNG 保存后，先要求所有 rank 的普通
  `on_save` callback 拓扑与顺序完全相同，再逐 callback 汇聚 rank-local 异常。拓扑差异在任何 callback
  执行前 fail closed；全部成功后才进入独立 commit phase，原子提交 manifest 并执行
  rotation；direct-path 和 run-root resume 只接受通过 manifest 校验的 checkpoint。FSDP/DeepSpeed 使用
  `backend_native`。普通 fixed FSDP/DeepSpeed 仍沿用后端保存、发现、校验和 rotation；planned sharded SFT
  额外使用 `shaft_backend_checkpoint_commit.json` 做 prepared -> committed generation 事务：marker 绑定
  backend、step、world size、Trainer/scheduler/RNG 等小状态内容身份、完整 native shard 路径/非零尺寸集合，
  以及 planning state/spec/batch/resume contract。大模型/optimizer shard 不额外全量重读计算 SHA256，字节内容
  仍由 backend-native loader 校验；marker 保证该完整 shard 集合与 planning generation 不可拆分。
  direct-path 和 run-root resume 只接受 marker 与 backend artifact 同时有效的 generation，并跳过 pending、
  torn、stale 或 generation 不匹配目录。FSDP+PEFT exact resume 当前只对 SFT 验收；DPO/GRPO 不属于支持
  范围，且通用配置预检尚可能接受该组合，因此不能将“配置可加载”解释为可恢复。PPO 不接入任一 resumable
  协议，要求 `save_strategy: no` 且仍禁止 resume；这不影响 `save_final_model` 的 `best` 导出或 root final state。
- 所有 checkpoint 都在 stateful callback 中保存同一 canonical batch contract。exact resume 改变四轴、
  `per_device_train_batch_size`、DP world size 或 GA 会在模型加载前失败；旧 checkpoint 若没有该 callback，
  只能作为 `init_from_checkpoint` 权重来源启动新 schedule。planned commit extension 还会交叉验证该
  callback 与 planner callback 的 spec fingerprint、batch geometry 和 GA。只有旧
  `shaft_batch_planning_complete.json`、没有通用 commit manifest 的 checkpoint 不再允许 exact resume。

exact resume 的启动校验按同一顺序分段收敛：`runtime/config -> cheap resume -> model plan -> data -> model ->
metadata -> trainer inputs -> trainer construction`。每个纯本地、可能失败的 builder 先收集所有 rank 的状态与
fingerprint；只有全 rank 成功后，才直接进入 model loader、metadata broadcast、efficiency all-reduce 或
Trainer/Accelerator constructor 等 collective-owning 边界，owning API 本身不再套 status envelope。显式
`train.use_cpu=true` 时早期 process group 固定选择 Gloo，即使节点仍可见 CUDA。readiness 阶段的 rank-local
失败会让所有 rank 得到同一错误，而不是让其它 rank 停在下一次 collective。
`ShaftTrainingResumeContract` v2 直接组合 `train_input_contract_fingerprint` 与
`data_execution_fingerprint`，因此只消费根契约也不会漏掉 data→tensor 漂移。
`shaft_checkpoint_commit.json` v2 对目录中记录的全部 artifact 保存 size + SHA256，并在发布 marker 前逐文件
fsync。manifest 还显式保存 exact-bool `requires_grad_scaler`：该值来自保存时真实的
`accelerator.scaler is not None`，为 true 时 `scaler.pt` 是 required artifact，为 false 时 bf16/fp32 等无
scaler checkpoint 可以合法提交；不得通过 precision 名称反推。manifest、trainer state、batch metadata、
GRPO cadence state、shard index 与 resume-consumed efficiency snapshot/transaction 都使用同一 strict JSON
loader，duplicate key、`NaN` 和 `Infinity` 会 fail closed；efficiency telemetry 损坏仍按既有语义降级为
partial coverage，不改变 optimizer trajectory。v1 marker 或任一同尺寸内容篡改不能 exact resume，但目录仍可
通过 `init_from_checkpoint` 只加载权重并启动新 schedule。
resolver 会把本次恢复固定为一个类型化 generation token：run root 从新到旧找到首个有效 checkpoint 后停止，
后续 preflight、planned state 和 Trainer 入口复用同一 content identity，不重复 hash 大 artifact；train 前只
重读小 marker 并核对 stat guard，同时对 generation fingerprint 做全 rank consensus。

`packing` 决定多个逻辑序列是否组合，`layout` 决定最终 tensor/attention 表示。Qwen varlen 把计划好的
logical rows 展平为 `[1,total_tokens]`，不传普通 2D attention mask；每段首 label/loss weight 清零，模型
adapter 在 host 侧构造 `[4,1,total_tokens]` positions，并校验每段 image grid/pixel slice。Qwen3.5/3.6
hybrid model 还生成 `seq_idx/cu_seq_lens/max_length`，因此 CUDA runtime execution path 同时要求
FlashAttention 2、flash-linear-attention、causal-conv1d、bf16/fp16 与 DDP；当前 release gate 使用 BF16，
FP16 仍只是 runtime allowlist。CPU 只保留 Qwen3VL eager/SDPA correctness oracle。
FSDP、DeepSpeed、torch.compile、varlen 下的单样本多图和视频仍明确拒绝；普通 padded 路径的有序多图
不依赖这套 segment-isolation contract。

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
- `data.schedule.mixing` 当前支持：
  - `concat`：覆盖全部有效行；开启 `shuffle` 时每轮使用无额外索引内存的可复现置换。
  - `weighted`：把 `DatasetSourceConfig.weight` 解析为固定配额 ticket block。常见简单权重在约 4K tickets
    中保持精确比例；不超过 64 个 source 时会完整搜索 `source_count..16384` 的 Hamilton block，选择最大
    相对误差最小的解。更大 catalog 先检查最多 32 个 denominator-derived 候选；未命中时先用 quota
    可接受区间筛出仍可能满足 5% 合同的 block，再用 O(source count) Hamilton rank predicate 完整验证，
    只对最终合法解物化 quota。因此不会因为 fast path 漏掉可表示权重，也不会对每个 block 都排序整个
    大 catalog。每个 source 的相对误差必须不超过 5%；极小正权重
    若无法获得 quota，或 16K 内所有解仍超过误差界，会带 source/target/resolved quota 明确报错。一个
    seed-specific base block 只物化一次。rotation 每 256 blocks 用 keyed SplitMix64 更换 phase，group 内
    使用 block/full-cycle coprime 的短 counter step；global-draw slope 与 `1..64` world size 均互质，因此
    稀有 source 的有限 block 前缀具有确定性 rank-count discrepancy 上界，而不是只依赖随机 hash 的期望
    公平。source 内使用 keyed Feistel
    permutation 无放回取行，耗尽后才开始下一轮。它是低方差 stratified stream，不是 IID multinomial
    有放回抽样。fixed step 使用有限 plan adapter，planned step 直接消费
    horizon-independent schedule。该 rotation 的有限前缀 rank-balance 证明覆盖
    `data_world_size=1..64`；`weighted + shuffle=true` 在更大 data world 上启动即 fail closed，不能把
    “通常较均匀”冒充已证明的 rank 公平。需要超过 64 个 data-parallel rank 时应使用 `concat`，或先升级
    sampling contract 与性质证明。
- `data.schedule.shuffle` 选择 schedule 自身的确定性置换策略：`concat` 置换整个 concat cycle；
  `weighted` 置换 ticket source order，并对每个 source 使用独立的无放回 row cycle。它不是 planned
  grouping 的窗口重排开关。`weighted + shuffle=false` 当前只开放 finite fixed-plan adapter，planned
  batching 仍明确拒绝；该限制来自 planned API 尚未开放 unshuffled schedule，并非再次按有限 horizon
  舍入 quota。unshuffled v3 复用同一正 quota/5%/16K 表示合同，把 exact quota occurrence midpoint 做
  确定性低差异 merge，所有 epoch 连续消费同一 ticket stream；source 内 row 按实际累计 occurrence 顺序
  推进并在耗尽后回绕。它不随机打乱 source 或 row，也不会永久饿死小于单轮一个样本的正权重 source。
  `sample_stream_fingerprint` 不包含 finite horizon，`sample_execution_fingerprint`/plan fingerprint 仍包含
  `num_samples`；前者用于同流比较，后者用于 exact resume。
- SFT/DPO/GRPO 的 resumable 路径不再增加第二个随机顺序源。受限且不可恢复的 PPO 是例外：当前 TRL
  DataLoader 会再 shuffle finite-plan positions，因此完整 plan 的 multiset/coverage 保留，但不承诺
  canonical prefix 顺序或 exact resume。
- `weight=0` 会禁用该 source 的 train split 加载与抽样；若 `use_for_eval=true`，val split 仍可参与评估。
- `num_workers` 是每个 rank 的 worker 数。例如 8 rank × 4 worker 会产生 32 个读取进程。
- `prefetch_factor` 是每个 worker 预取 batch 数，仅在 `num_workers>0` 时传给 HF DataLoader。
- JSONL 首次加载时会规范化到 source snapshot 指纹化的 Arrow cache；`record_cache_dir` 可覆盖默认的
  `~/.cache/shaft/records`。后续 rank/worker 使用只读 mmap，不再各自保留完整 Python record list。
- SFT JSONL 可使用顶层 `prompt_args` 保存 prompt 模板参数。它必须是 JSON object，是 Arrow record 的正式
  JSON 字段，不会进入 `extra`；提供完整 `messages` 的行不能同时提供非空 `prompt_args`。`prompt_args`
  不能替代非空 `target_text`，也不能承载在线 target 生成逻辑。
- SFT/DPO 单图行可使用 `image_path`（兼容 `image`）；多图行使用非空有序列表 `images`，三个字段只能
  选择一个。图片顺序是训练接口契约：显式 `messages` 中的 `type: image` 占位符必须逐个对应且数量相等；
  没有 `messages` 时模板会按 `images` 顺序自动生成占位符。通用 record 的 `user_prompt` 默认 `""`，任务
  prompt 应来自数据、prompt pool 或显式调用参数。
- PromptSource 在首次构建 Arrow cache 时按对应 pool prompt schema 校验每条 SFT record，随后校验各
  formulation store 的行数和 identity 对齐；validation fingerprint 进入 cache key。后续 mmap 命中表示该
  source snapshot 已在同一 prompt schema 下通过验证，不会先解码图片再在 worker 中发现参数错误。
- `image_cache_size` 是每个 worker 的解码后 PIL 图像 LRU 容量，默认 `0`（关闭）。多 rank/worker
  环境应按总内存预算谨慎开启。
- `max_length` 是 processor 后完整训练序列的严格上限，覆盖 multimodal prefix、assistant target 和 EOS，
  语义接近 Swift / LLaMA-Factory 的 `max_length` / `cutoff_len`。当 prefix 已达到上限时，框架只从模板声明的
  消息正文 span 中按时间顺序删除较早的普通文本 token，同时保留 chat message 边界、generation suffix、
  special token 和有序 media token，并为非空监督 target 保留至少一个可训练 token。cost estimator 与实际
  collator 复用同一截断 plan。被截断的 target 不补 EOS，避免把半截 JSON 教成合法结束；如果上限连结构和
  media 都无法安全容纳，则在 planning/collation 阶段明确失败，应提高 `max_length` 或降低像素预算。

### `data.schedule`

`schedule` 只决定逻辑样本流，不决定 batch 大小或 padding：

```yaml
data:
  schedule:
    mixing: weighted
    shuffle: true
```

同一个 `draw_id`、seed 和 source snapshot 必须解析到同一个 source/row。batching 可以在有界窗口内重组
已经选中的 draws，但不能改变 schedule 的 multiset。

### `data.prompt_sources`

用途：独立管理 task formulation 的离线 target sources，并为每个 logical draw 完成两层在线选择：

- task formulation：决定问什么、监督什么；数量、命名和合法组合完全由 pool 人工声明，
  `A / A+B / A+B+C` 是常见嵌套示例；
- prompt variant：同一 formulation 内轮换语义等价的措辞。

每个 formulation 的 `target_text` 必须已经存在于自己的标准 SFT JSONL。PromptSource 只选择答案，不从
`prompt_args`、full target 或模板生成答案。未配置的 dataset 直接使用普通 materialized 数据。

配置示例：

```yaml
data:
  datasets:
    - dataset_name: reconstruction
      use_for_eval: false
  prompt_sources:
    reconstruction:
      path: ../prompts/reconstruction.formulations.yaml
      apply_to: train       # train | all
      seed: 42              # 省略时继承 experiment.seed
      formulation_sources:
        a: {train_path: ../data/reconstruction/sft/formulations/a/train.jsonl}
        ab: {train_path: ../data/reconstruction/sft/formulations/ab/train.jsonl}
        abc: {train_path: ../data/reconstruction/sft/formulations/abc/train.jsonl}
```

formulation pool 示例：

```yaml
metadata: {id: shaft.reconstruction.formulations.v1, version: v1}
arguments:
  proposal_bbox_2d: {type: bbox_2d_0_999}
formulations:
  - id: a
    sampling_weight: 1.0
    prompts:
      - id: direct
        system_prompt: Return compact JSON only.
        user_prompt_template: Reconstruct A near {{ proposal_bbox_2d | json }}.
  - id: ab
    sampling_weight: 2.0
    prompts:
      - id: direct
        user_prompt_template: Reconstruct A and B near {{ proposal_bbox_2d | json }}.
  - id: abc
    sampling_weight: 4.0
    prompts:
      - id: direct
        user_prompt_template: Reconstruct A, B, and C near {{ proposal_bbox_2d | json }}.
```

每个 source 文件内仍是一行一个离线 target。例如 A 与 ABC 文件中的对齐行分别为：

```json
{"image_path":"images/a.png","sample_id":"a-1","prompt_args":{"proposal_bbox_2d":[1,2,300,400]},"target_text":"{\"A\":{\"x\":1}}"}
```

```json
{"image_path":"images/a.png","sample_id":"a-1","prompt_args":{"proposal_bbox_2d":[1,2,300,400]},"target_text":"{\"A\":{\"x\":1},\"B\":[2,3],\"C\":true}"}
```

现有顶层 `prompts` pool 是正式简写：它会编译成一个名为 `default`、使用 materialized target 的
formulation，因此当前 prompt 轮换自然属于 PromptSource，而不是另一条兼容链路。

约束：

- pool 与 formulation source 路径都相对训练 YAML 解析。`formulation_sources` id 必须与 pool exact match。
- formulation 模式禁止 dataset 顶层 `train_path`；`apply_to=train` 的 eval 走顶层 materialized val，
  `apply_to=all` 则要求每个 formulation source 都有对齐 val。
- `formulations` 与顶层 `prompts` 只能二选一。formulation 只声明 prompts；`target_template`、`target` 和
  source 行内多 target mapping 都会被拒绝。
- pool 级 `arguments` 只约束 prompt renderer；每行 `prompt_args` 不能承担 target 组合真值。
- 动态语法只有 `{{ name }}` 与 `{{ name | json }}`；不支持 Jinja、属性访问、表达式或任意代码。类型支持
  `string/enum/integer/float/boolean/json/bbox_2d_0_999`，所有参数必须齐全且不能多出 schema。
- pool 模式禁止 materialized `messages/system_prompt/user_prompt`。每个 formulation source 的
  `target_text` 必须非空；各 source 行数和 identity 字段完全对齐，仅 target 可以不同。
- 每个 draw 始终按 formulation 的静态 `sampling_weight` 执行 weighted categorical sampling，不是
  round-robin；短前缀不保证严格比例。`sampling_weight` 必须有限、非负，且 pool 至少有一个正权重。
  PromptSource 不提供按 epoch、step 或 draw 改权重的 curriculum。
- 框架不自动生成属性幂集或推断 `A -> A+B -> A+B+C` 的依赖；这些 formulation 必须逐项人工声明。
- formulation 和 prompt variant 使用独立的确定性 hash 随机域。增加 prompt wording 不会改变 formulation
  分布；planning/runtime、多 worker、DP rank 和 exact resume 对同一 draw 的结果一致。
- PromptSource 审计集中写入 `extra.prompt_source`，包含 pool/formulation/variant、全局 draw、实际权重
  以及 prompt/target/arguments 的 SHA256，不再维护散落的 `runtime_prompt_*` 字段。
- execution fingerprint 绑定 sample stream、全部 formulation record/media snapshots、pool prompt schema、
  static weights、seed 与 renderer。任一合同变化后旧 checkpoint 只能作为 `init_from_checkpoint` 启动新 run。

完整 pool schema、合法 row 模式、选择算法与验收门禁见
[data.md](data.md)。

补充说明：

- 仓库内置的 `configs/data/example.yaml` 当前只作为示例 catalog，不应默认视为可直接训练的数据清单。
- 如果你的实验数据较少或不需要复用 catalog，直接使用 `data.datasets` 往往更直观。

## 5. `algorithm`

用途：选择训练算法与算法级参数。

关键字段：

- `name`: `sft | dpo | ppo | grpo | opd`
- `params.auxiliary_loss_weights`（仅 SFT）

```text
algorithm.name
├── sft
│   ├── ShaftSFTPipeline
│   ├── jsonl_sft
│   ├── train.loss_name / train.loss_scale
│   └── params.auxiliary_loss_weights.<term_name>
├── dpo
│   ├── RL training domain / DPORuntime
│   ├── jsonl_dpo
│   └── rlhf.dpo
├── ppo
│   ├── RL training domain / PPORuntime
│   ├── jsonl_ppo（当前 text-only）
│   ├── rlhf.ppo
│   └── 不支持 periodic checkpoint / resume
├── grpo
│   ├── RL training domain / GRPORuntime
│   ├── jsonl_sft -> GRPODataset
│   ├── rlhf.grpo
│   └── grouped generation repeat
└── opd
    ├── OPD training domain / ShaftOPDPipeline
    ├── jsonl_opd（prompt-only + 有序 images）
    ├── opd.teacher / opd.rollout / opd.objective
    └── 本地 student rollout + 冻结 teacher direct loss
```

说明：

- resolved `RuntimeConfig` 中的 `algorithm.name` 是唯一算法选择器。中央入口通过 training-domain registry
  分派到 `sft / rl / opd`，没有算法名 `if/elif`；DPO/PPO/GRPO 再由 RL runtime registry 解析。
- `load_config()` 会先按 YAML 中的算法完整 normalize；CLI 子命令或 `--algorithm` 随后才写回该字段并再次
  normalize。因此 CLI 不能“修好”一份在原 YAML algorithm 下已经非法的配置，YAML 与命令必须先自洽。
- pipeline 只消费选中的 `rlhf.<name>`，但 normalize 当前仍校验 DPO/PPO/GRPO 三个子块；未选中的块不会
  改变 trainer，却也不能包含非法值。最清楚的写法是只显式填写当前算法的子块。
- 内置 SFT 当前只接受一个 `params` 项：`auxiliary_loss_weights`。它是很少需要使用的 run-level override；
  默认应省略。term name 会做 `strip + lower`，未知 params、未知 term、非数值、布尔值、非有限值或负数
  均 fail closed。显式空 map 会被规范化为省略。
- `auxiliary_loss_weights.<term_name>` **替换**模型 policy/checkpoint 提供的默认 coefficient，不是乘数；
  未列出的 term 继续使用模型默认值。以 Qwen3.5/3.6 MoE 为例：

  ```yaml
  algorithm:
    name: sft
    params:
      auxiliary_loss_weights:
        router_aux_loss: 0.002
  ```

  对应关系为
  `L = L_CE + w_effective * L_router`，其中配置存在时 `w_effective` 等于 override，否则等于模型
  `router_aux_loss_coef`。override 不修改 HF checkpoint 的 `config.json`。配置为 `0` 只把该项对总 loss 的
  加权贡献置零；为了保留 raw 诊断，模型仍会请求 router logits。
- raw `eval_aux/router_global_balance` 与 CE-only `eval_loss` 不受 override 影响；
  `aux/router_aux_loss_weighted` 和 `eval_aux/router_global_balance_weighted` 使用同一个 effective
  coefficient。eval metric 通过模型 policy 的显式 `coefficient_key` 关联训练 term，不按 metric 名猜测。
- normalized `algorithm.params` 进入 exact-resume contract。改变 override 后不能 resume 旧 schedule；应启动
  新训练，或将旧权重作为 `train.init_from_checkpoint` 使用。
- DPO/PPO/GRPO/OPD 当前都要求 `algorithm.params` 为空；未消费字段 fail closed。RL 的结构化参数位于
  `rlhf.<name>`，OPD 参数位于独立的 `opd` 节点。
- `rlhf.enabled` 是当前 schema 中保留的兼容字段，不参与算法选择，也不会替代
  `algorithm.name`。新配置不应依赖或显式设置它。

## 6. `train`

用途：训练行为、保存策略和 resume/init 规则。

关键字段：

- `duration`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `gradient_checkpointing`
- `learning_rate`
- `param_group_lrs`
- `no_decay_name_patterns`
- `optimizer_name`
- `scheduler_name`
- `scheduler_num_cycles`
- `scheduler_power`
- `loss_name`
- `loss_scale`
- `adam_beta1`
- `adam_beta2`
- `adam_epsilon`
- `weight_decay`
- `warmup_ratio`
- `lr_scheduler_type`
- `max_grad_norm`
- `bf16`
- `fp16`
- `use_cpu`
- `full_determinism`
- `logging_steps`
- `save_strategy`
- `save_epoch_interval`
- `save_steps`
- `save_total_limit`
- `save_only_model`
- `max_shard_size`
- `ddp_find_unused_parameters`
- `report_to`
- `load_best_model_at_end`
- `save_final_model`
- `save_final_state`
- `init_from_checkpoint`
- `resume_from_checkpoint`
- `efficiency`
- `distributed`

### 跨层字段关系

下列字段名称相近，但控制的是不同阶段；除表中明确说明的情况外，它们不是覆盖关系：

| 字段 | 作用阶段 | 当前关系 |
|---|---|---|
| `model.torch_dtype` | 模型权重加载 dtype | `train.bf16` / `train.fp16` 是互斥的 Trainer AMP 执行精度；不由框架静默改写模型加载 dtype，用户应按模型和硬件显式选择 |
| `data.max_length` | 单条 logical sample 的 processor 后完整序列严格上限 | 同时约束 prefix/target/EOS；batching 的 aggregate token budget 约束整个 local microbatch，不能替代它 |
| `data.max_pixels` | 单张输入图的 processor 像素预算 | `batching.resource_budgets.vision_patches` 约束整个 local microbatch 的视觉成本 |
| `train.per_device_train_batch_size` | 每卡 microstep 的 physical-pack 数量 `B` | 由 `data.batching.cardinality` 决定 `B` 是精确数量还是上限；world size 来自启动器，不写在该字段中 |
| `train.gradient_checkpointing` | 模型侧 gradient checkpointing | FSDP 且 `distributed.fsdp.activation_checkpointing=true` 时，Shaft 关闭 Trainer 模型侧开关，由 FSDP activation wrapper 单独负责，避免双重 checkpointing |
| `train.scheduler_name` | Shaft 自定义 scheduler 的执行真源 | 默认 `auto` 时从兼容字段 `lr_scheduler_type` 解析；显式设置后以 `scheduler_name` 为准 |
| `train.save_only_model` | periodic `checkpoint-*` 的内容语义 | `false` 保存可 exact-resume 的完整训练态；`true` 只发布标准 HF/PEFT 模型态，允许部署或 `init_from_checkpoint`，禁止 resume |
| `train.max_shard_size` | full HF 权重文件的分片上限 | 默认 `4GB`；接受正整数 byte，或 HF 支持的 `KB / MB / GB / TB` 字符串；不改变 checkpoint 的训练态语义 |
| `train.init_from_checkpoint` | 只加载权重/adapter，启动一个新 schedule | `resume_from_checkpoint` 恢复 Trainer、optimizer、scheduler、RNG 与可恢复的数据计划状态；两种语义应二选一 |

推荐在新 YAML 中显式写 `scheduler_name`。`lr_scheduler_type` 目前仍会传入 HF
`TrainingArguments` 并参与部分兼容性元数据；若两者都显式出现，应保持相同值，避免日志或第三方 hook
看到与 Shaft 实际 scheduler 不同的 HF 字段。

`train.bf16` 与 `train.fp16` 互斥；默认是 `bf16=true, fp16=false`。FP16 只允许可用 CUDA 设备，CPU
配置会在构建 Trainer 前失败。`train.fp16=true` 表示 Trainer AMP/GradScaler 执行精度，不等于把可训练
参数直接加载成 FP16；配置层明确拒绝 `model.torch_dtype=float16 + train.fp16=true`，模型装配后还会检查
实际 trainable floating parameters，因此 `torch_dtype=auto` 也不能从 FP16 artifact 绕过防线。full fine-tune
推荐 `model.torch_dtype=float32`。FP16 GradScaler 跳过 overflow 时，非有限 `grad_norm` 不写入 checkpoint，
改记有限的 `grad_norm_overflow=1`。precision 会进入 exact-resume contract；从 BF16/FP32 切到 FP16 必须
启动新训练或使用 `init_from_checkpoint`，不能伪装成同一条 exact resume。当前 FP16 release contract 覆盖
padded Qwen3VL-2B DDP LoRA；varlen FP16 和 FSDP FP16 尚待专项 CUDA canary。仓库 DeepSpeed preset 仍是 BF16-only，
`FP16 + strategy=deepspeed` 会 fail closed。

`init_from_checkpoint` 与 `resume_from_checkpoint` 严格互斥，schema、CLI override 和 pipeline preflight
都会 fail closed。若要继续旧训练，使用 resume；若只复用旧权重并重新开始数据流、optimizer 和
scheduler，使用 init。

保存与恢复边界：

- `max_shard_size=4GB` 是 full HF 模型权重的默认分片上限，适用于 periodic checkpoint 与 `<output_dir>/best`
  的 Trainer 保存路径。单个 tensor 本身超过上限时，HF 会把它单独放入一个更大的 shard；该字段控制新产物
  的分片大小，不承诺复刻输入模型原有的 shard 数量或边界。PEFT adapter 继续使用 PEFT 原生保存格式。
- `save_only_model=false` 是默认值：periodic `checkpoint-*` 保留模型、optimizer、scheduler、scaler（若有）、
  每 rank RNG、Trainer state 与 Shaft 可恢复 callback state，用于 exact resume。
- `save_only_model=true` 只对 SFT 开放。periodic `checkpoint-*` 仍是可由 HF/PEFT 直接加载的标准模型目录，
  并保留少量 `trainer_state.json` 等审计元数据；它不会保留 optimizer、scheduler、scaler、RNG 或
  FSDP/DeepSpeed native training state。成功发布后会写
  `shaft_model_only_checkpoint.json`，该 marker 只描述提交状态，不改变 HF/PEFT 权重布局。
  这类目录可部署，也可传给 `init_from_checkpoint` 开启新 schedule，但不能传给
  `resume_from_checkpoint`；direct path 会明确拒绝，run root resolver 会跳过它。
- `save_only_model=true` 要求 `save_strategy=steps|epoch`，且不能同时配置 `resume_from_checkpoint`。
  FSDP 要求 `state_dict_type=full_state_dict`；DeepSpeed ZeRO-3 要求
  `stage3_gather_16bit_weights_on_model_save=true`；两种 sharded backend 都要求
  `load_best_model_at_end=false`。未验收算法或无法形成完整 HF/PEFT 权重的组合会在加载数据和模型前失败。
- `save_final_model=true` 把部署用 HF/PEFT 导出写入 `<output_dir>/best`。
- `save_final_state=true` 把最终 `trainer_state.json` 保留在 run 根目录；finetune/optimizer summary 也属于
  run metadata，root layout 清理不得删除这些文件。
- 分布式结束阶段先收敛保存意图，再让可能拥有 FSDP/DeepSpeed collective 的 `trainer.save_model()` 在 status
  envelope 外执行。所有 rank 返回后，rank-zero HF/PEFT layout 校验、`save_state()` 与 root prune 进入统一
  local-finalization status convergence；任一 rank 的本地 I/O 异常会成为所有 rank 的同一错误，结束阶段不再
  依赖一个可能永远到不了的裸 barrier。
- root `trainer_state.json` 只用于保留最终指标，不等于包含 optimizer/RNG 的可恢复 checkpoint。
  `resume_from_checkpoint` 指向 run 根目录时，checkpoint 子目录始终优先于 root final state：
  `committed_manifest` 路径选择最新通过 manifest 校验的 checkpoint，并跳过未提交或 artifact 校验失败的
  目录；`backend_native` 路径沿用 HF/backend 的 latest-checkpoint 发现与兼容性检查，不附加通用 torn/atomic
  承诺。`best` 仍是部署导出，不作为训练恢复点。
- SFT/RL pipeline 总是在 dataset、base model 与 PEFT adapter 装配前使用 `experiment.seed` 初始化 Python/
  NumPy/PyTorch 随机状态，保证 full/PEFT fresh 初始化不依赖 Trainer 创建时机。
- `full_determinism=true` 还会在上述早期阶段调用 HF `enable_full_determinism`，并透传
  `TrainingArguments.full_determinism`，启用 PyTorch deterministic algorithms、确定性 cuBLAS/CuDNN，
  以及支持该能力的 FlashAttention deterministic backward。它用于 bitwise CUDA resume/fresh
  reproduction 验收，通常会降低吞吐；默认关闭。若默认关闭，planning/data/optimizer 状态仍可精确恢复，
  但非确定性 CUDA kernel 可能让两次运行产生正常的微小数值差异。
- 对三个及以上 DDP rank，`full_determinism` 本身不会保存 reducer 首轮 bucket-rebuild 状态。连续训练的后续
  iteration 与恢复后新建 DDP 实例的首个 iteration 可能采用不同 bucket 顺序，浮点归约因结合顺序不同而出现
  ULP 级差异。需要 bitwise DDP fresh/resume 验收时，应同时设置
  `train.distributed.ddp.static_graph=true`；动态图或跨 step 改变 used/unused 参数集合的训练不能冒充该契约。

### `train.efficiency`

```yaml
train:
  efficiency:
    enabled: true
    device_timing: auto  # auto | off
    persist: true
```

- 默认打开；它不改变 planner、loss 或 checkpoint 正确性语义。SFT 输出
  `shaft_training_efficiency.json`，统计 batching/supervision/data/optimizer 指标；OPD 使用独立的
  `shaft_opd_telemetry.json`，统计 rollout、student/teacher score、objective、distribution 与远端传输。
  两个协议共享 `enabled/device_timing/persist` 开关，但不共享 frame schema 或指标解释。
- `device_timing=auto` 在 CUDA 上从 optimizer frame 的第一次 `training_step` 到 optimizer 完成记录一对延迟
  解析的 events，覆盖该 frame 的 forward/backward 与 optimizer device timeline，只在 logging/final window
  同步；每个 committed frame、每个 rank 都必须有完整 event coverage。`off` 仍保留 iterator acquire、
  batch denominator prepare、training-step 与 optimizer host wall time。
- 周期指标在 `Trainer.log()` 进入 W&B/console 前合并；最终版本化真源是 run root 的
  `shaft_training_efficiency.json`。除 useful token throughput 外，summary 还记录 logical-segment/vision-patch
  throughput、critical-path p50/p95，以及各 rank 中最大的 CUDA peak allocated/reserved memory。显存窗口在
  HF `on_train_begin` 建立，即 model/optimizer 与 resume state 装配完成之后；不混入模型加载瞬时峰值。
  checkpoint 保存 rank-local 历史峰值，exact resume 最终取历史与当前窗口 MAX；旧/缺失历史或窗口不可用时
  JSON 为 `null`、比较器显示 `n/a`，不得静默当成 0。`reserved` 反映 allocator 缓存，不等同于模型实际持有的
  `allocated` memory。`[batch-plan-summary]` 只表示 planner producer，不等于执行吞吐。
- checkpoint 保存每 rank 的可选 `shaft_training_efficiency_rank<N>.json`。任一 rank 的 snapshot 缺失、损坏、
  span 或 contract 不一致时，所有 rank 都丢弃旧 telemetry history；这不阻止模型/optimizer exact resume，
  但最终 summary 会标记 `complete_history=false`。全套 snapshot 有效时从 checkpoint step 延续且不重复累计。
  Trainer 在覆盖 model checkpoint 前先写入本次 generation 的 `pending/revoked` transaction；revoke、rank
  snapshot write、manifest commit 与最终 transaction commit 都执行 all-rank 状态汇合。`persist=false` 也会
  先把 transaction 标为 revoked，再清除同名 checkpoint 的旧 snapshot set，避免 I/O 失败或后续 run 误恢复
  另一 generation。
- 使用 `python scripts/compare_efficiency.py RUN_A RUN_B ...` 比较 committed throughput、padding、
  segments/pack、logical-segment length 分布和 rank skew。工具默认校验模型、数据/source fingerprint、
  logical draw stream、DP/GA、优化器和 committed step span，只允许 batch/sequence contract 作为实验轴变化；
  默认还要求 committed logical workload 完全一致，适合 padded/varlen layout A/B。packing 使同一 step
  span 消费更多 logical work 时可使用 `--allow-workload-variation` 做 capacity comparison：它只放宽
  segment/token/vision/mass，仍强制 identity、step span、update count、microbatch/physical-pack count 与
  telemetry coverage 一致，并分别展示 useful/supervised token、segment、vision-patch rate；该模式不是等工作量
  speedup。
  exact-resume 继续另行绑定包含有限 plan horizon 的 execution fingerprint。`--allow-incompatible`
  仅供明确接受非公平结果的诊断。fixed path 中未版本化的 online transform 或缺失 `media_snapshot_id`
  不阻止训练，但会令 source identity 标为 incomplete，默认不能进入公平多 run 比较。
- 比较器只接受当前 v3 root summary；v2 采用旧显存边界且缺少 stream contract，不做自动迁移，也不能进入
  公平 A/B。缺失显存显示 `n/a` 仅指合法 v3 summary/snapshot 中显式的 `null`，不代表兼容旧 root schema。

### `train.duration`

训练时长只有一个真源：

```yaml
train:
  duration:
    unit: steps
    value: 10000
```

- `unit=steps` 是主路径，`value` 必须为正整数。Shaft 会把它映射到 HF `max_steps`。fixed batch 使用有限
  SamplePlan；planned（`length / bounded_cost`）路径直接从 horizon-independent schedule 惰性取数，不按
  duration 物化完整 draw index。
- `unit=epochs` 用于有限数据兼容，`value` 可为正浮点数。Shaft 会映射到 HF
  `num_train_epochs`；一个 epoch 的 plan 长度默认为所有有效 source 行数之和。
- YAML 不再同时维护 `epochs` 与 `max_steps`。CLI 仍提供互斥的 `--epochs` / `--max-steps` 便捷覆写。

说明：

- `train` 是 SFT、RL 与 OPD 共用的基础运行块；算法专属语义分别位于 `rlhf.*` 与 `opd.*`。
- `optimizer_name/scheduler_name/loss_name` 走注册表。
- `distributed.strategy` 描述训练拓扑入口，当前支持：
  - `ddp`
  - `fsdp`
  - `deepspeed`
  默认是 `ddp`，表示继续使用 Hugging Face / torchrun 的常规 DDP 语义。
- `distributed.ddp.static_graph` 默认是 `false`。开启后透传 HF
  `TrainingArguments.ddp_static_graph`，声明每个训练 iteration 使用相同的参数图和稳定的 used/unused 参数集合；
  当前只对 `strategy=ddp + algorithm=sft` 开放，其它 strategy/algorithm 会在配置阶段拒绝。该字段属于 DDP
  reducer 执行语义，并进入 exact-resume contract；它不是 batching、packing 或数据字段。
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
- 当前 `strategy=fsdp` 强制 `use_orig_params=true`。Shaft 的 resolved optimizer plan 以参数名
  绑定 decay、结构组和分组学习率；HF FSDP1 会在模型 wrap 后延迟创建 optimizer，Shaft 会针对 wrapped
  model 重建同一语义 plan，并验证 fingerprint 后才使用新的 `Parameter` 对象。`use_orig_params=false`
  会把参数替换为 `FlatParameter`，当前没有可靠的 name/group remap，因此在 config normalize 和
  `TrainingArguments` 构造阶段都会 fail closed，而不是等到训练开始后产生错误参数组。
- FSDP + `lora/dora/qlora` 当前还强制 `state_dict_type=full_state_dict` 与
  `load_best_model_at_end=false`。完整标准 PEFT adapter 是 exact-resume 的模型状态真源；backend-native
  optimizer、scheduler、scaler、每-rank RNG 继续由 FSDP 恢复。Transformers/Accelerate 5.10.1/1.13
  的 adapter-only native FSDP model 文件只包含 rank-local DTensor，Shaft 明确跳过该模型加载路径。
  若需要按最佳指标选择 FSDP PEFT checkpoint，应训练结束后从对应 checkpoint 的标准 adapter 做独立导出；
  在完成 wrapped-model scatter 门禁前不能打开 `load_best_model_at_end`。
- `distributed.fsdp.transformer_layer_cls_to_wrap=["auto"]` 会按模型族默认解析。Qwen3VL 当前解析为：
  - `Qwen3VLTextDecoderLayer`
  - `Qwen3VLVisionBlock`
- Qwen3.5 / Qwen3.6 dense 默认解析为：
  - `Qwen3_5DecoderLayer`
  - `Qwen3_5VisionBlock`
- Qwen3.5 / Qwen3.6 MoE profile 解析为：
  - `Qwen3_5MoeDecoderLayer`
  - `Qwen3_5MoeVisionBlock`
- Qwen3.5/3.6 dense 与 MoE 当前都要求 `distributed.fsdp.activation_checkpointing=false`；使用
  `train.gradient_checkpointing` 走模型侧重计算。该限制由 model sharding policy 校验，不应写进通用 pipeline。
- `distributed.deepspeed` 支持 `config_path` 或 inline `config`。当 `strategy=deepspeed` 时，两者至少要提供一个；
  `config_path` 的相对路径按训练 YAML 所在目录解析。Shaft 只负责保存和校验配置真源，不在
  `config` 层展开 DeepSpeed 运行时细节。
- 当前 Shaft 仍由自定义 optimizer/scheduler 持有参数分组学习率语义；DeepSpeed 配置如果包含
  `optimizer`/`scheduler` 块会在加载阶段报错。应交给 HF Trainer 将 Shaft optimizer 接入 DeepSpeed。
- `strategy=deepspeed` 时，pipeline 会先构建 `TrainingArguments`，再执行模型 `from_pretrained`。
  这是 ZeRO-3 大模型训练的必要顺序：HF 会在 `TrainingArguments` 初始化阶段建立 DeepSpeed
  runtime config，让模型加载阶段能感知 ZeRO-3 分片语义。
- PEFT 0.18.1 的 direct-parameter LoRA wrapper 会直接读取 parameter tensor shape；ZeRO-3 构造期参数是
  `shape=(0,)` 的 partition placeholder，真实形状只在 `ds_shape`。Shaft 自己的 plan 能识别该元数据，但
  不能安全替代 PEFT 注入器，因此 `ZeRO-3 + model.finetune.target_parameters` 会在加载前明确拒绝。
  Qwen3.5/3.6 MoE 的推荐组合是 FSDP + LoRA，或 ZeRO-3 + full finetune。
- `strategy` 不是 `deepspeed` 时，Shaft 会清理 HF/Accelerate 进程级 DeepSpeed 状态，避免
  在测试或长驻进程里先运行 DeepSpeed 后污染后续 DDP/FSDP 训练。
- `configs/deepspeed/zero1_bf16.json`、`zero2_bf16.json`、`zero3_bf16.json` 分别是 ZeRO-1/2/3
  bf16 示例配置；ZeRO-3 示例包含保存时 gather 16-bit 权重的设置，用于保持 `trainer.save_model()`
  的 HF export 语义。当前没有 FP16 DeepSpeed preset，因此 `train.fp16=true` 与 DeepSpeed 组合不属于
  已支持接口。
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
- 没有写进 `param_group_lrs` 的组，回退到全局 `train.learning_rate`。
- 该语义对 `full / lora / dora / qlora` 完全一致：先把 FSDP/PEFT 运行时参数名规范为稳定路径，
  再使用模型 adapter 的 `ModelModuleGroups` 最长边界前缀确定结构组。LoRA A/B、DoRA magnitude 和
  `modules_to_save` 包装不会改变参数所属的模型结构。
- `lora_params` 与 `modules_to_save` 不是合法 LR key。前者只是 PEFT 参数角色，后者仍是 adapter
  装配/保存/导出合同；二者都不参与 optimizer 学习率分组。
- 显式写入的结构组必须至少命中一个 trainable parameter，否则 optimizer 创建前报错。正式模型的所有
  trainable parameter 也必须能被 module group metadata 覆盖，不能静默落入 `default`。
- `no_decay_name_patterns` 用于把额外参数名并入 `no_decay` 规则。
  - 匹配语义是“参数规范名后缀匹配”，例如：
    - `embed_tokens.weight`
    - `lm_head.weight`
  - 这条规则会叠加在默认 `no_decay` 规则之上；默认规则仍然包括：
    - `*.bias`
    - `ndim <= 1` 的参数
- optimizer 只认识上述四个结构组；finetune mode 只决定 `requires_grad` 集合，不改变 LR 解析规则。
- `loss_scale` 控制哪些粗粒度区段参与 loss 计算，当前内置：
  - `default`: 监督所有 assistant 回答（包括多轮对话中的历史 assistant，以及当前 target/response）
  - `last_round`: 只监督最后一轮 assistant 回答（当前 target/response）
  - `all`: 同时监督 system/user/prefix 与 target/response
- 当前 `loss_scale` 的落点在 `template -> SFTCollator -> ShaftSFTTrainer -> training/loss.py`
  这条链上：
  - `template` 负责把多轮消息规范化为 rendered-token supervision plan，并产出单样本
    `labels / loss_scale / span`
  - `SFTCollator` 只执行一次 batch 级多模态 processor 调用、padding 与张量装配
  - `ShaftModelAdapter -> ProcessorPolicy` 负责将 canonical rendered-token span 精确投影到 processor
    展开后的 token layout；缺少模型专用映射时直接报错，不做近似对齐或 partial-image fallback
  - `training/loss.py` 负责真正的加权 next-token loss

## 7. `eval`

用途：评估开关、频率和 best model 选择。

关键字段：

- `enabled`
- `per_device_eval_batch_size`
- `eval_strategy`
- `eval_steps`
- `min_pixels`
- `max_pixels`
- `loss_metrics_enabled`
- `do_sample`
- `temperature`
- `max_new_tokens`
- `online_metrics_enabled`
- `datasets`
- `metric_for_best_model`
- `greater_is_better`

额外说明：

- `eval.eval_strategy: epoch` 时，SFT 可以配合 `eval.epoch_interval` 控制“每隔多少个 epoch 才进行一次
  eval”。`train.save_strategy: epoch` 时，SFT 可以配合 `train.save_epoch_interval` 控制保存间隔。
- 上述两个 interval callback 当前只在 SFT pipeline 安装；DPO/GRPO 尚未消费，RL 配置不要依赖它们。
- 当 interval 不能整除总 epoch 时，训练最后一个 epoch 仍会强制执行一次对应的 eval / save，避免漏掉最终结果。

控制关系：

```text
eval.enabled
├── false
│   ├── eval_strategy -> no
│   └── train.load_best_model_at_end 必须为 false
└── true
    ├── 至少一个 enabled + use_for_eval source，并提供 val split
    ├── eval_strategy / eval_steps / epoch_interval 决定何时调用
    ├── loss_metrics_enabled 控制 dataset-policy loss 聚合
    ├── online_metrics_enabled 控制生成式指标分支
    └── metric_for_best_model 选择已有指标，不会自动启用计算分支

train.load_best_model_at_end=true
├── eval.enabled=true
├── train.save_strategy != no
├── eval.eval_strategy != no
├── save_strategy 与 eval_strategy 一致
├── steps 模式：save_steps % eval_steps == 0
└── FSDP + PEFT 当前不支持，必须保持 false
```

说明：

- dataset-policy eval 定义两类聚合结果，但算法支持范围不同：SFT 同时支持 `eval_final_loss` 与
  `eval_final_score`，DPO 只接 loss 聚合，GRPO 当前只接 online `eval_final_score`。
- `loss_metrics_enabled=true` 控制 named dataset-policy 的 teacher-forced loss 聚合，并按 policy `weight`
  形成 `eval_final_loss`；它不是关闭 HF 基础 eval 调用的全局开关。
- `online_metrics_enabled=true` 时，框架会按同一套 dataset policy 计算生成式任务指标，并聚合为 `eval_final_score`。
- 当前在线 eval 支持 SFT 与 GRPO。GRPO 训练侧使用 `GRPODataset` 做 rollout，在线 eval 侧保留原始 SFT 样本结构并复用 `SFTCollator` 生成评估 prompt。
- `ShaftGRPOTrainer` 尚无可靠的 loss eval / `eval_final_loss` 聚合；当
  `algorithm.name=grpo` 且 `eval.enabled=true` 时必须显式设置 `eval.loss_metrics_enabled=false`。需要评估时
  配置 `online_metrics_enabled=true` 与 `metric_for_best_model=eval_final_score`，normalize 会对误开的 loss
  eval fail closed。
- eval pixel budget 按字段独立解析，优先级为
  `eval.datasets.<name>.min_pixels/max_pixels -> eval.min_pixels/max_pixels -> data.min_pixels/max_pixels`。
  省略新增字段时仍回退到旧的 `data.*` 行为；在同时运行 loss 与 online eval 的 SFT 路径中，两者复用
  同一个 resolved `EvalInputPolicy`，不会使用不同分辨率。
- DPO 的 train/eval 使用独立 collator，分别消费 `data.*` 和 resolved eval budget。当前 PPO 是 text-only，
  显式 eval pixel budget 会在 normalize 阶段报错。
- processor padding 不由 pipeline 写裸字符串：模型 `ProcessorInputPolicy` 声明 training 为 right padding、
  generation 为 left padding；启动时会输出一条紧凑的 `[eval-input]` 摘要。
- 缺失或未知的 `dataset_name` 按 eval default budget 处理；已知 dataset 使用其 resolved override。同一
  processor batch 可以混合最终预算相同的样本，最终预算不同则明确失败。

### 7.1 在线 eval 配置

当前版本已支持单阶段在线 eval，目标是：

- 单阶段在线 eval
- 多数据集
- 多任务
- 每个数据集只绑定一个 task
- 通过一套统一 policy 定义：
  - `eval_final_score`
  - `eval_final_loss`

当前 dataset 级 eval policy 包含：

- `min_pixels`
- `max_pixels`
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
5. 在支持 loss eval 的 SFT/DPO 路径，`eval_final_loss` 由各 dataset 的 teacher-forced loss 按同样的
   权重加权求和；GRPO 当前不产出该指标
6. dataset policy 只要求为 `use_for_eval=true` 的数据集配置；训练专用数据集不会进入这一套聚合

下面是同时启用 loss 与 online eval 的 SFT 示意配置：

```yaml
eval:
  enabled: true
  eval_strategy: epoch
  min_pixels: 200704
  max_pixels: 2000000
  loss_metrics_enabled: true
  metric_for_best_model: eval_final_score
  greater_is_better: true
  online_metrics_enabled: true
  datasets:
    det_dataset:
      max_pixels: 4000000
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
- 上一条不适用于 GRPO；GRPO 启用 eval 时 `loss_metrics_enabled=true` 会直接报错，只允许 online
  `eval_final_score`。
- 若配置了 dataset policy 且仍保留旧式 `metric_for_best_model=eval_loss`，框架会自动收敛为：
  - 有 online eval 时使用 `eval_final_score`
  - 否则使用 `eval_final_loss`
- SFT 同时启用两条评估链时，`report_to` 会上报：
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

选择关系：

```text
algorithm.name
├── sft   -> 不消费 rlhf.* 作为训练算法参数
├── dpo   -> rlhf.dpo
├── ppo   -> rlhf.ppo
└── grpo  -> rlhf.grpo
             ├── rollout
             ├── vllm
             └── reward_functions
```

`rlhf` 不是第二个算法开关。只由 `algorithm.name` 选择当前算法；未选中的子块即使因 dataclass 默认值
存在，也不会变成当前训练算法。

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
- PPO 不支持 periodic checkpoint、best-checkpoint selection 或 resume；配置必须使用
  `train.save_strategy: no` 和 `train.load_best_model_at_end: false`。这不影响训练结束后的 final model 导出。
- 当前 PPO rollout 明确是 text-only：`jsonl_ppo` 的 `image_path` 可省略，即使提供也不会在
  `PPODataset` 中打开/解码；messages 中的 image chunk 会在 PPO collator 中移除。
- `PPOCollator` 的类级默认输入模式是 `generation`：TRL 虽从训练 dataloader 取得 query，但下一步直接执行
  decoder-only rollout，因此异长 prompt 必须遵循模型 `ProcessorInputPolicy` 的 generation padding（默认
  left），不能继承普通 loss collator 的 training/right 语义。

### `rlhf.grpo`

- `beta`
- `rollout`
- `vllm`
- `reward_functions`

旧版 GRPO 的 `num_generations`、`max_completion_length`、`temperature`、`use_vllm` 等平铺字段仍作为
兼容 alias 接受；若平铺与嵌套同时存在，normalize 当前以平铺 alias 覆盖 nested 值。新配置应统一写入
`rlhf.grpo.rollout` 或 `rlhf.grpo.vllm`，不要同时维护两份值。

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
- 只有 GRPO resolved config 的 `vllm.enabled=true` 才执行 TRL/vLLM runtime compatibility preflight。Shaft 从
  当前已安装 TRL 的 `extra == "vllm"` requirement 解析当前 platform 唯一有效版本区间，并校验已安装 vLLM；
  requirement 缺失/歧义/畸形、vLLM 未安装或版本越界都会在 data/model 前 fail closed。当前标准环境为
  `trl==1.9.2` 与 `vllm==0.19.1`；TRL 声明的 vLLM 区间为 `>=0.17.0,<=0.25.1`，因此可以通过版本 gate。
  普通 HF rollout、SFT/DPO/PPO 以及独立 infer/vLLM 服务不经过这条 gate。
- 版本兼容 gate 与 vLLM rollout RNG checkpointability 是两条独立防线：即使版本兼容，外部 sampled-rollout
  RNG 尚未持久化时仍要求 `train.save_strategy=no` 且禁止 resume；关闭 checkpoint 不能绕过版本不兼容。
- 对 VLM GRPO，TRL/vLLM 会绕过 SFT collator，因此 RL pipeline 把 model-owned
  `ProcessorPolicy.prepare_rollout_image()` 注入 `GRPODataset`。Qwen policy 使用 `data.min_pixels/max_pixels`
  做像素预算；通用 dataset 不理解 Qwen factor/aspect-ratio，后续模型族必须实现自己的 policy。
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
- GRPO 通过 `ShaftGroupedSampleSampler` 保留 TRL grouped-generation 的 mini-repeat/repeat-count
  结构，同时按 HF epoch 设置确定性 plan cycle；因此 grouped prompt 一致且多 epoch resume 可复现。
  `data.schedule.shuffle` / `ShaftSamplePlan` 是唯一的数据顺序真源，Shaft 会显式设置 TRL
  `shuffle_dataset=false`；grouped sampler 只扩展算法要求的重复，不再对 plan position 做第二次洗牌。
  resolved mini-repeat、group batch、iteration count 与 steps-per-iteration 共同组成 grouped execution
  contract，并合入 checkpoint sample-execution fingerprint；任一 cadence 漂移都会在模型加载前拒绝 resume。
- TRL 的 generation buffer 以 local training microstep 计数，不能用 `global_step * GA` 近似。Shaft 从 grouped
  contract 解析每 rank 的真实 epoch microstep 数 `L`，并用
  `microstep(g) = floor(g / K) * L + min((g mod K) * GA, L)`、
  `K = ceil(L / GA)` 校验每个实际 save target 和 resume checkpoint。完整 grouped epoch 保证
  `L % (steps_per_generation * num_iterations) == 0`，所以完整 epoch save 可恢复；step-bounded 或小数 epoch
  在部分 epoch 结束时仍必须落在完整 generation-reuse boundary，否则启动阶段 fail closed。
- `vllm.enabled=true` 时，TRL/Shaft 当前不持久化 vLLM engine/server 的采样 RNG。该模式只允许
  `train.save_strategy=no` 且禁止 `resume_from_checkpoint`；最终模型导出仍可使用。恢复支持要等 rollout
  使用可持久化 RNG，或按 canonical draw 派生并绑定 per-request seed 后再开放。

## 9. `opd`

用途：fully on-policy direct-loss distillation 的 teacher、student rollout 与分布目标。

```text
model                              student artifact 与 finetune plan
opd
├── teacher                        冻结 teacher role
│   ├── provider: hf_local | http
│   ├── model_type / model_name_or_path / revision / cache_dir
│   ├── template / local_files_only / trust_remote_code
│   ├── attn_implementation / torch_dtype
│   └── remote                     provider=http 时使用
│       ├── endpoint / artifact_fingerprint / api_key_env
│       ├── request_timeout_seconds
│       └── max_request_bytes / max_response_bytes
├── rollout                        student generation
│   ├── backend: hf_local | vllm
│   ├── max_new_tokens / do_sample / temperature
│   ├── top_p / top_k / min_p / repetition_penalty
│   └── vllm                       共享 vLLM topology 配置
└── objective                      completion-token distribution loss
    ├── mode: full_vocab | topk_tail
    ├── divergence: forward_kl | reverse_kl | jsd
    ├── temperature / top_k
    └── token_chunk_size
```

约束与语义：

- `data.source_type` 必须是 `jsonl_opd`；记录只包含 prompt，不允许 trailing assistant target。
- teacher/student 由独立的 OPD input ABI 门禁判断，不要求完整 artifact、产品 alias 或原始 chat template
  相同。ABI 必须证明完整 token→ID、special token ID、实际 logits vocabulary 维度、多模态 processor/input
  schema，以及 teacher `forward` 对 student scoring tensor 字段的兼容性；任一项无法证明或不一致都会在训练前
  明确失败。类似 Qwen3.5/3.6/3.8 的产品版本在已有 adapter 能构建 artifact、且共享这套实际输入 ABI 时，
  可以作为不同 alias 组合使用，因为 teacher 消费的是 student 已生成的同一批 tensor。teacher 仍是独立
  module，全部参数冻结，不进入 optimizer、student checkpoint 或最终 export。
- `teacher.provider=hf_local` 加载独立冻结 HF teacher；`provider=http` 只解析远端 immutable identity，不在训练
  进程加载 teacher。`artifact_fingerprint` 必须是 64 位 SHA-256；API key 只从 `api_key_env` 读取。
- `rollout.backend=hf_local` 使用本地 student generate；`backend=vllm` 包装 TRL `VLLMGeneration`，支持
  `server / colocate`。server 模式通常先运行 `trl vllm-serve`，并配置独立 `server_port/group_port`。
- 多模态 vLLM 请求同时维护未展开 generation prompt 和本地 processor 展开后的 scoring prompt；返回 prompt
  必须逐 token 对齐 scoring 真源，禁止重复展开 `<image_pad>`。
- `data.max_length` 是 prompt + completion 严格上限，collator 会为 `rollout.max_new_tokens` 预留空间。
- batching 只允许 `grouping=none + cardinality=fixed + packing=none + layout=padded`；支持 DDP、FSDP 和
  DeepSpeed，仍不支持 packing/varlen 或 eval。
- divergence 必须显式配置；`full_vocab` 使用完整词表，`token_chunk_size` 沿 completion-position 轴降低
  objective 中间张量峰值。`topk_tail` 要求 `top_k>0`，其 loss 是 K 个显式 token 加一个剩余 mass bucket 的
  coarse-grained divergence；不是把未返回 token 当零。
- checkpoint contract 分别绑定不可变 teacher artifact fingerprint 与统一的
  `teacher_student_input_abi_fingerprint`，后者在 local/HTTP teacher 下都只表示实际蒸馏输入 ABI，不混入
  provider 或 teacher artifact 身份。旧的 artifact-equality 输入 fingerprint 不会被迁移为新语义，旧 OPD
  checkpoint 在 exact resume 时会 fail closed。
  contract 同时绑定 rollout/objective、数据执行身份与 RNG。
  vLLM 请求 seed 由 run seed、model version 与 request IDs 派生；每个 model version 只同步一次权重。新增
  execution component 只有声明 exact-resume 能力后才允许 periodic checkpoint/resume。
- `train.efficiency.enabled=true` 启用 OPD 独立 optimizer-frame telemetry，输出
  `shaft_opd_telemetry.json` 与 checkpoint 内 `shaft_opd_telemetry_rank<N>.json`。wall phase 与
  `device_*` CUDA event 字段语义分离；`device_timing=off` 只关闭 CUDA event，不关闭 wall telemetry。
- teacher service 入口为 `python scripts/serve_opd_teacher.py --config ...`。HTTP v2 identity 发布与 local
  teacher 相同结构的 OPD input ABI；request/response 使用版本化 safetensors envelope，按 `max_*_bytes`
  有界读取，且 body 必须匹配 `Idempotency-Key`。v1 identity 无法表达 forward/input ABI，客户端会明确拒绝。

## 10. `plugins`

- `hooks`
- `interceptors`

用途：

- 为训练主链注入横切增强点。

## 11. `logging`

- `level`
- `fmt`
- `file_path`
- `rank_zero_only`
  - 默认 `true`，表示所有 structured log 严格只由 global rank 0 输出，包括 WARNING/ERROR；分布式主链的
    rank-local 失败依靠同步 failure envelope 或 torchrun traceback，不允许普通 warning 与 rank-0 活动行
    竞争共享终端
  - 调试时若设为 `false`，text/JSON 每行都会包含 rank；多节点/多卡且配置了 `file_path` 时自动写入
    `<stem>.rank<N><suffix>`，避免各 rank 并发覆盖同一个文件。共享终端仍可能交错，因此建议同时把
    progress 设为 `plain` 或 `off`
- Shaft 的 `INFO` 只保留 Transformers/Hugging Face Hub 的 `WARNING/ERROR`，避免默认打印完整
  model/processor config；两者都移除独立 stderr handler 并经 Shaft 的 rank/progress-aware handler 输出一次。
  显式 `logging.level=DEBUG` 才恢复上游详细日志。该规则不改变 Shaft 自身的 INFO 生命周期日志。

## 12. `progress`

- `enabled`
- `display`
  - `auto`：TTY 使用单行交互显示，重定向日志、CI 和非交互子进程使用稀疏文本状态
  - `interactive`：强制单行原地刷新
  - `plain`：强制输出无 ANSI、无 `\r` 的稀疏状态行
  - `off`：不创建终端或文本 sink；若 `persist=true`，仍保存结构化快照
- `width`
  - 单行终端显示的最大物理列宽，默认 `72`，配置最小值 `40`；运行时还会跟随真实 terminal resize 向下
    收缩，避免窄窗口换行。CJK、组合字符、常见 emoji grapheme 和 ANSI style 按真实 terminal cell 处理；
    小于 40 列时只移除完整字段，不会把 `current/total` 截成半个值
  - interactive 使用高对比度 `━╸─` bar，72 列依次保留 task、精确百分比、bar、current/total、速度、ETA
    和最有用的指标；40 列先舍弃 ETA/metric，但保留进度和速度。慢 iteration 显示 `s/it`，快 iteration
    显示 `it/s`；其它任务显示 `sample/s`、`batch/s` 等带单位 rate。正数亚秒 ETA 显示 `<1s`，未知 total
    使用自动刷新的 spinner。未完成状态不会四舍五入成 `100%`
  - 真 TTY 自动使用克制的 label/bar/metric 颜色和 ANSI erase-line。`NO_COLOR`、`CLICOLOR=0` 关闭 SGR
    颜色但保留安全清行；`TERM=dumb` 或重定向输出关闭 ANSI terminal control。严格 ASCII stream 自动降级为
    `=>-` bar，plain 模式始终无 ANSI、无 `\r`
- `refresh_interval`
  - TTY 最小刷新间隔，默认 `0.5` 秒，必须为有限正数
- `log_interval`
  - 非 TTY 普通 update 的最小输出间隔，默认 `30.0` 秒，必须为有限正数；阶段开始/完成/失败不受节流，
    且不会被 `logging.level` 静默屏蔽
- `leave_completed`
  - 是否保留普通子阶段的完成行，默认 `false`；训练主任务仍输出一条最终摘要
- `persist`
  - 是否原子更新 `<output_dir>/shaft_progress.json`，默认 `true`

训练、loss eval、online eval 和 data/model startup 复用同一个 progress manager。bounded cost 在训练
DataLoader 内按需完成，不再创建独立的全量 startup 进度任务。终端只显示
当前前台阶段；进入 eval 时临时替换 train 行，结束后恢复 train。结构化快照保留完整任务树，供持久化
状态审计和外部诊断工具读取；外部消费者应读取 JSON，而不是解析日志来推导进度。Shaft 同时关闭
Transformers 与 Hugging Face Hub 的原生进度条；
新增长任务必须向统一 manager 发布，不能平行创建 tqdm。嵌套 eval 的 wall time 不计入恢复后的 train
step rate；失败/取消阶段会强制留下摘要。manager 对并发 advance/close 提供有序状态语义。
显示指标按 `loss -> useful token throughput -> grad norm -> LR` 排序，宽度不足时从右侧低优先级字段开始
省略；状态与 JSON snapshot 中仍保留完整 metric，不因终端宽度丢失。

10,000-step 任务的 interactive 形态示例：

```text
train 0.25% ╸───────── 25/10k 6.54s/it eta 18h07m loss 7.9 lr 2.5–5e-7
```

`lr` 为所有 optimizer param groups 的当前 min–max range；组间相同则只显示一个值。`logging_steps` 到达前
没有 loss 属于训练日志策略，但速度、ETA 和精确进度从第一步起可见；efficiency callback 首次提交后会优先
显示 `tok …/s`，此时窄行可能省略 LR。

## 13. CLI override 原则

只允许无歧义字段通过 CLI 覆盖，例如：

- `run-id`
- `seed`
- `max-steps`
- `epochs`
- `lr`
- `resume-from`
- `init-from`

禁止：

- 用 CLI 直接拼装复杂 `datasets` 列表
- 用 CLI 覆盖多层嵌套且语义不清的配置对象

`--max-steps` 与 `--epochs` 互斥；任一参数都会完整替换 `train.duration` 的 unit/value。
