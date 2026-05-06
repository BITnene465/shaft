# Shaft 开发日志

本文档记录已经暴露过的工程问题、指标误判、修复方式和后续防线。目标是让重复犯错的问题进入仓库真源，而不是停留在单次聊天或临时排障记录里。

## 维护规则

- 当线上/离线 eval 指标异常、训练语义被误判、或者同类 bug 第二次出现时，必须补一条日志。
- 每条日志至少包含：现象、根因、影响范围、修复、回归测试、后续防线。
- 如果问题涉及评估标准，必须明确区分“模型能力问题”和“eval/codec/metric 误判”。
- 日志不是待办列表；待实现事项可以同步到 `docs/todo.md`，但根因和经验必须留在这里。

## 2026-04-28: online eval 左 padding completion 切片污染 keypoint JSON 解析

### 现象

`grounding_row_bucket` 训练第一次在线 eval 中，`keypoint_arrow` 指标明显异常：

- `parse_success=0.5125`
- `keypoint_pck=0.3881`

从任务难度和已有数据质量看，keypoint 不应该弱到这个程度，尤其 parse success 不应该只有一半。

### 根因

在线 eval 的 prompt collator 使用 `left padding`，但 decoder-only 生成结果切 completion 时使用了每条样本的 `attention_mask.sum()`：

```python
completion_ids = row[prompt_length:]
```

对于左 padding batch，HF decoder-only `generate()` 返回的是：

```text
[padded_input_ids, generated_completion_ids]
```

completion 的起点应该是 batch padded input width，也就是 `input_ids.shape[1]`，不是每条样本自己的非 pad token 数。短样本用 `attention_mask.sum()` 会切早，把 prompt 尾部一起解码成 prediction。

keypoint prompt 里包含枚举列表：

```text
["solid", "dashed"]
["straight", "rounded", "curved"]
```

当 prompt 尾部混进 prediction 后，`json_object` codec 会先看到 `[`，解析成 JSON list，再因为期望 object 报 `json_type_error`。这会把本来可能合法的模型输出记为 parse failure。

### 影响范围

- 主要影响 generation-based online eval。
- 只要满足以下条件就有风险：
  - decoder-only 模型
  - `left padding`
  - 按 `attention_mask.sum()` 切 generated row
  - prompt 尾部含 JSON-like 片段，且 prediction codec 对 JSON 顶层类型敏感
- 对 `keypoint_arrow` 影响尤其大，因为 prompt 中有 JSON-style 枚举列表。

### 修复

- `ShaftOnlineEvalRunner` 对 decoder-only 输出统一按 `prepared["input_ids"].shape[1]` 切 completion。
- encoder-decoder 模型仍按生成输出本身解码，不追加 input prefix 假设。
- 新增回归测试覆盖左 padding 下 prompt 尾部包含 list、prediction codec 要求 `json_object` 的场景。

### 同步发现的 metric 标准问题

`keypoint_arrow` 的 `keypoints_2d` 使用 0-1000 bin 坐标，但 `keypoint_pck` 曾用图片宽高作为 5% 容差尺度。对于几十像素的小 crop，容差会被压到几格，明显偏严。

修复为：

- `keypoint_pck` 默认使用 `normalized_1000` 坐标尺度。
- 配置中显式写入：

```yaml
- name: keypoint_pck
  params:
    coordinate_space: normalized_1000
    num_bins: 1000
```

如未来评估像素坐标 keypoint，需要显式设置 `coordinate_space: image`。

### 回归测试

- `tests/test_online_eval.py::test_online_eval_runner_slices_left_padded_decoder_prompts_at_input_width`
- `tests/test_online_eval.py::test_keypoint_pck_uses_normalized_coordinate_scale_by_default`

本次验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_online_eval.py
.venv/bin/ruff check src/shaft/training/online_eval.py src/shaft/metrics/builtin.py tests/test_online_eval.py
.venv/bin/python -m compileall src/shaft/training/online_eval.py src/shaft/metrics/builtin.py tests/test_online_eval.py
```

### 后续防线

- 所有 generation eval / infer 路径都要明确 completion slice invariant：
  - decoder-only: completion starts at padded input width
  - encoder-decoder: decode generated sequence directly
- 不允许在 left padding generation 路径用 `attention_mask.sum()` 作为 completion 起点。
- 新增结构化任务 prompt 时，如果 prompt 含 JSON 示例、枚举列表或 schema 片段，必须额外检查 codec 是否可能被 prompt 泄漏污染。
- 指标中坐标尺度必须显式化：`bbox_2d` / `keypoints_2d` 如果是 0-1000 bin，metric 不得默认退回图片像素尺度。

## 2026-04-29: 单进程多卡触发 DataParallel 破坏 Qwen3VL 视觉张量对齐

### 现象

从 `checkpoint-23640` resume 训练时，第一步 forward 在 Qwen3VL visual tower 中报错：

```text
RuntimeError: The size of tensor a (1106) must match the size of tensor b (1676) at non-singleton dimension 0
```

调用栈中出现：

```text
torch.nn.parallel.data_parallel.py
```

这说明当前训练不是 DDP，而是单进程可见多张 CUDA 卡后被 Hugging Face Trainer 包成了 PyTorch `DataParallel`。

### 根因

Qwen3VL 的多模态 batch 中：

- `pixel_values` 是所有图片 patch 拼接后的变长张量，第 0 维是 patch 数。
- `image_grid_thw` 是按图片计数的网格元数据，第 0 维是图片数。

PyTorch `DataParallel` 会按第 0 维独立切分每个 tensor。它不知道 `pixel_values` 与 `image_grid_thw` 之间的语义对应关系，于是会把 patch 张量和 grid 元数据切到不一致的 shard。进入 visual tower 后，patch embedding 的长度与根据 `image_grid_thw` 生成的位置 embedding 长度不一致，最终在：

```python
hidden_states = hidden_states + pos_embeds
```

处报维度不匹配。

### 影响范围

- 影响所有 Qwen3VL 类 decoder-only 多模态训练路径，只要满足：
  - 单进程启动
  - 多张 CUDA 卡对进程可见
  - 没有用 `torchrun` / DDP
- 与 checkpoint 本身无关，也不是 `lm_head.weight` missing warning 的直接原因。
- `per_device_train_batch_size` warning 不是这次维度错误的根因；真正触发点是 `DataParallel` 对多模态变长视觉张量的错误切分。

### 修复

新增训练 topology guard：

- 当 CUDA 可用、可见 GPU 数量大于 1、且没有分布式启动环境变量时，训练启动阶段直接报错。
- 报错信息明确提示：
  - 单卡：使用 `CUDA_VISIBLE_DEVICES=<id> python scripts/train.py ...`
  - 多卡：使用 `torchrun` / DDP
- guard 放在模型加载前，避免先加载大模型再在第一步训练时炸。

### 回归测试

- `tests/test_pipeline_sft.py::test_training_topology_rejects_single_process_data_parallel`
- `tests/test_pipeline_sft.py::test_training_topology_allows_distributed_launch`

### 后续防线

- 多模态训练不允许依赖 PyTorch `DataParallel`。
- 任何训练入口只要可能看到多张 CUDA 卡，都必须显式区分：
  - 单卡单进程
  - DDP 多进程
  - 非法的单进程多卡
- 如果未来新增模型族，其视觉输入中存在按 patch 展平、按图片记录 metadata 的结构，也必须继承这条 topology 约束。

## 2026-04-29: DDP online eval 显示口径和样本去重必须与单卡一致

### 现象

cuda1/cuda2 的 DDP smoke 训练和 online eval 能跑通，但 progress bar 显示为：

```text
online_eval 1/1 batch
```

同一份 val 在单卡上显示为：

```text
online_eval 2/2 batch
```

这说明 DDP 下显示的是 rank0 本地 dataloader 进度，而不是全局 eval 进度。进一步检查发现，online eval 会 all-gather 各 rank 预测再聚合 metric，但没有对 DistributedSampler padding 可能带来的重复样本去重。

### 根因

- 显示层：progress bar 只在 rank0 创建，total 取 rank0 本地 dataloader 的 batch 数。
- metric 层：DDP all-gather 后直接聚合全部 entries。如果 eval 样本数不能被 world size 整除，分布式 sampler 可能 padding 重复样本，重复项会进入平均指标。

### 影响范围

- DDP online eval 的最终 metric 主路径是全局 all-gather 聚合，方向正确。
- 当样本数能被 world size 整除时，metric 与单卡一致。
- 当样本数不能被 world size 整除时，若 sampler padding 重复样本，metric 可能被重复样本轻微影响。
- progress bar 在 DDP 下不是单卡同口径，会低估全局 eval 总量。

### 修复

- `ShaftOnlineEvalRunner.aggregate_samples()` 聚合前按 `(dataset_name, sample_id, image_path)` 去重。
- DDP progress bar 改为全局 sample 口径：
  - total 使用 dataloader dataset 的全局长度。
  - rank0 每个 batch 按 `local_batch_size * world_size` 更新，并 cap 到 total，避免 padding batch 超出总量。

### 回归测试

- `tests/test_online_eval.py::test_online_eval_runner_deduplicates_gathered_samples_before_metrics`

### 后续防线

- DDP eval 的 metric 聚合必须以全局唯一样本为准，不得让 sampler padding 改变指标。
- DDP eval 的显示口径必须明确是全局样本进度，或在文案中显式标注为 local rank 进度。

## 2026-04-30: GRPO/vLLM 绕过 SFT collator 导致图像 token 预算失效

### 现象

在 cuda1 上尝试单卡 `vllm.mode=colocate` 的 GRPO smoke 时，vLLM 能加载并完成 CUDA graph 初始化，但第一步 rollout 在输入校验阶段失败：

```text
The decoder prompt (length 12324) is longer than the maximum model length of 8192.
```

此前 `vllm.max_model_length=4096` 时也出现过同类错误，某个样本的 prompt 长度已经达到 6205 tokens。

### 根因

SFT/DPO/PPO collator 会通过 `model_adapter.build_processor_inputs(..., min_pixels, max_pixels)` 把 `data.max_pixels` 传给 processor。

GRPO 使用 TRL `GRPOTrainer`，不走 Shaft 的 SFT collator。`GRPODataset` 之前直接返回原始 PIL 图像，TRL/vLLM 会按自己的 VLM 路径处理图像，导致 `data.max_pixels=262144` 没有生效。高分辨率图像被展开成过多 multimodal tokens，最终超过 vLLM context。

这不是模型能力问题，也不是 reward/metric 问题，而是 GRPO 数据适配层没有继承 Shaft 图像 token 预算语义。

### 影响范围

- 影响 VLM GRPO，尤其是 `use_vllm=true` 的 rollout。
- 非 vLLM GRPO 也会受到影响，因为 TRL 的 VLM prompt/forward 处理同样绕过 Shaft collator。
- SFT 主链不受影响，SFT collator 已显式传入 `min_pixels / max_pixels`。

### 修复

- `GRPODataset` 新增 `min_pixels / max_pixels` 参数。
- `ShaftRLHFPipeline` 构建 GRPO dataset 时传入 `config.data.min_pixels / max_pixels`。
- `GRPODataset` 在样本进入 TRL 前按像素预算调整 PIL 图像，避免原始大图撑爆 multimodal token 数。
- GRPO 配置结构同步改为：
  - `rlhf.grpo.rollout`
  - `rlhf.grpo.vllm`
  并保留旧 flat 字段作为兼容入口。

### 回归测试

- `tests/test_pipeline_rlhf.py::test_grpo_dataset_applies_image_pixel_budget`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_uses_sft_dataset_for_grpo`
- `tests/test_config_loader.py::test_load_config_supports_grpo_reward_config`
- `tests/test_training_modules.py::test_build_trl_grpo_config_from_training_args`

### 后续防线

- 新增 RLHF/VLM 路径时，必须确认是否经过 Shaft collator；如果不经过，图像 token 预算要在 dataset adapter 或算法 adapter 层显式落地。
- 不能只调大 `vllm.max_model_length` 来掩盖图像预算失效；必须先确认 `data.max_pixels` 对实际 rollout prompt 生效。
- `rollout.max_completion_length` 只限制生成长度，不能替代 prompt multimodal token 控制。

## 2026-04-30: GRPO/vLLM colocate sleep mode 触发每步磁盘重载 checkpoint

### 现象

启动 `grounding_grpo_vllm_colocate_g8_bs32_1024` 后，训练能正常进入 step，但每个 train step 前都会反复出现：

```text
Loading safetensors checkpoint shards: 0/2
Loading safetensors checkpoint shards: 2/2
```

这和预期不一致。GRPO 中 vLLM rollout 副本确实需要随 policy 动态更新，但正常应从训练进程内存中的当前参数同步到 vLLM，而不是每步从磁盘 checkpoint 重新加载 safetensors。

### 根因

配置中开启了：

```yaml
rlhf:
  grpo:
    vllm:
      enable_sleep_mode: true
```

当前 TRL/vLLM colocate 路径中：

- `sync_weights()` 会把训练中的 policy 参数同步到 vLLM 副本。
- `generate()` 在 `enable_sleep_mode=true` 时会唤醒 vLLM，并调用 `collective_rpc("reload_weights")`。
- 在当前 `vLLM 0.19.0` 环境下，这个 `reload_weights` 会触发从磁盘 checkpoint shard 重新加载权重。

因此日志中每个 step 的 safetensors reload 不是正常的 policy 内存同步，而是 sleep/wake 机制引入的额外磁盘重载。

### 影响范围

- 影响 GRPO `vllm.mode=colocate` 且 `enable_sleep_mode=true` 的训练。
- 性能上会显著拖慢 step，因为每步多了一次 checkpoint shard 读取和加载。
- 语义上存在风险：如果 `reload_weights` 从初始 checkpoint 重载，可能覆盖刚通过 `sync_weights()` 同步到 vLLM 的当前 policy 权重，使 rollout 退回旧权重。
- `enable_sleep_mode=false` 时，vLLM 副本常驻显存，不触发这类 sleep/wake reload 路径。

### 修复

- 将 `configs/train/train_grpo_4b_grounding.yaml` 中的：

```yaml
enable_sleep_mode: true
```

改为：

```yaml
enable_sleep_mode: false
```

关闭后，vLLM 推理副本常驻显存。只要不 OOM，就优先使用这一设置，保证 rollout 权重同步语义和训练速度都更稳定。

### 回归测试

本问题主要通过训练日志验证：

- 正常现象：vLLM 初始化阶段加载 checkpoint。
- 异常现象：每个 train step 都出现 `Loading safetensors checkpoint shards`。
- 修复后应重新启动同一训练命令，确认 step 间不再反复磁盘加载 safetensors。

### 后续防线

- GRPO/vLLM 的权重同步必须区分两种语义：
  - 正确：optimizer step 后从训练进程当前参数同步到 vLLM。
  - 错误：每步从磁盘 checkpoint 重新加载 vLLM 权重。
- 开启 vLLM `sleep mode` 前必须先做多 step canary，确认不会反复触发 safetensors reload。
- 如果关闭 sleep mode 后 OOM，优先考虑降低 `gpu_memory_utilization`、`max_model_length`、`max_completion_length`，或改用独立 vLLM server/单独 GPU rollout，而不是接受每步磁盘重载。

## 2026-04-30: GRPO reward wrapper 导致 W&B per-reward 指标不可读

### 现象

检查 GRPO 监控项时发现，多个 reward function 传给 TRL 后函数名都叫 `_reward_func`。TRL 使用 `reward_func.__name__` 作为 W&B metric key，因此多个 reward 会写入同一类指标：

```text
rewards/_reward_func/mean
rewards/_reward_func/std
```

同时，reward weight 被提前乘在 wrapper 返回值里，导致 per-reward mean/std 是加权后的数值。例如 `parse_success` 权重为 `0.05` 时，W&B 中该项最高只能到 `0.05`，不能直接看作 parse success rate。

### 根因

`build_grpo_reward_functions()` 为每个 reward 创建闭包，但没有设置可区分的 `__name__`。并且 reward 权重被内联进闭包返回值，而不是交给 TRL 原生的 `reward_weights`。

### 影响范围

- 影响 GRPO W&B 监控可读性。
- 不影响总 reward 的数学结果，但会让 per-reward 监控误导：
  - 无法区分 `parse_success` 和 `grounding_iou`
  - 无法从 per-reward mean 直接读出原始 parse rate / IoU reward

### 修复

- 每个 GRPO reward wrapper 设置稳定名称：
  - `grpo_reward_parse_success`
  - `grpo_reward_grounding_iou`
- reward function 返回原始 reward。
- `build_trl_grpo_config()` 将配置中的权重传给 TRL `reward_weights`，由 TRL 聚合总 reward。

### 回归测试

- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_exact_match_and_parse_success`
- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_grounding_iou`
- `tests/test_training_modules.py::test_build_trl_grpo_config_from_training_args`

### 后续防线

- 新增 reward function 时，W&B key 必须稳定且可区分。
- per-reward 指标应记录原始 reward；权重应放在聚合层，避免监控值被缩放后难以解释。

## 2026-04-30: DDP 训练时 Shaft summary 元数据并发写入失败

### 现象

使用两卡启动 GRPO/vLLM colocate 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train.py rlhf ...
```

vLLM 初始化成功，但训练创建 optimizer 时 rank0 报错：

```text
FileNotFoundError: shaft_optimizer_summary.tmp -> shaft_optimizer_summary.json
```

### 根因

`ShaftOptimizerMixin.create_optimizer()` 在每个 DDP rank 上都会调用
`write_resolved_optimizer_summary()`。该函数使用固定临时路径
`shaft_optimizer_summary.tmp` 后再 `replace()` 到正式 json。多个 rank 同时写同一个
tmp 文件时会产生竞争：一个 rank 已经把 tmp replace 掉，另一个 rank 再 replace
同一路径时就会找不到文件。

同类风险也存在于 `shaft_finetune_summary.json` 写入。
训练结束阶段的 `ensure_hf_export_layout()` 和 `prune_root_output_layout()` 也是
Shaft 自己的 run 级文件操作，不能在所有 rank 上重复执行。

这不是模型能力问题，也不是 vLLM rollout 问题，而是训练元数据落盘没有遵守
DDP single-writer 语义。

### 影响范围

- 影响所有 DDP 训练路径，包括 SFT 和 RLHF。
- 单卡训练不受影响。
- 小规模 DDP smoke 可能不稳定复现，因为 rank 间时序足够错开时不会撞到同一个 tmp 文件。

### 修复

- `shaft_optimizer_summary.json` 只在 rank0 写入和记录启动日志。
- `shaft_finetune_summary.json` 只在 rank0 写入和记录启动日志。
- final export layout 校验与 root output prune 只在 rank0 执行。
- 非 rank0 仍正常创建 optimizer 和训练，只跳过 run 级 summary 落盘。

### 回归测试

- `tests/test_training_modules.py::test_optimizer_summary_is_written_only_on_rank_zero`
- `tests/test_pipeline_sft.py::test_run_sft_rank_nonzero_skips_run_level_file_ops`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_rank_nonzero_skips_run_level_file_ops`
- `tests/test_smoke_distributed.py::test_torchrun_train_eval_smoke`

### 后续防线

- DDP 下 run 级元数据必须是 single-writer，优先 rank0 写入。
- 如果未来确实需要多 rank 分别写文件，文件名必须包含 rank 或使用独立子目录，不能共享固定 tmp 路径。
- 多卡 smoke 不应只验证 forward/eval，也要覆盖 optimizer 创建和 run-level metadata 写入路径。

## 2026-04-30: GRPO reward 误把 partial JSON 当作完整有效输出

### 现象

检查 grounding GRPO reward 设计时发现，模型如果只输出一个未闭合 JSON 起始符：

```text
[
```

`json_list` codec 会将其修复为 partial `[]`。在空目标 hard negative 样本上，
该输出可以同时拿到 `parse_success=1.0` 和 `grounding_iou=1.0`，形成格式层面的
reward hacking 空间。

### 根因

GRPO reward 使用 codec 的 `decoded.valid` 判断解析成功，但没有区分
`decoded.partial`。而 JSON codec 的 lenient repair 是为诊断和容错 eval 服务的，
不应在训练 reward 中等价为完整正确输出。

### 影响范围

- 影响 GRPO 中使用 JSON codec 的 reward：
  - `parse_success`
  - `exact_match`
  - `grounding_iou`
- 对 grounding hard negative 样本尤其敏感，因为空目标 `[]` 是合法答案。
- 这是 reward 语义偏差，不是模型能力问题，也不是 eval metric 的误判。

### 修复

- GRPO reward 只接受 `decoded.valid and not decoded.partial` 的完整解析结果。
- partial decode 在 `parse_success` 中计为 0。
- partial decode 在 `exact_match` 和 `grounding_iou` 中直接计为 0，避免修复后的
  `[]`、`{}` 与目标偶然匹配。

### 回归测试

- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_exact_match_and_parse_success`
- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_grounding_iou`

### 后续防线

- 训练 reward 应比离线诊断 codec 更严格；partial repair 可以用于观测，
  不应默认给正 reward。
- Grounding GRPO 监控需要单独记录 `parse_partial_rate`、`pred_empty_rate`、
  `target_empty_rate`、`positive_pred_empty_rate`，避免只看总 reward。
- 空目标 hard negative 应按 bucket 单独监控，防止模型通过过度输出空数组提高局部 reward。

## 2026-05-01: DDP 正常结束后仍打印 process group 清理 warning

### 现象

两卡 GRPO 训练完整跑到 `global_step=1526`，最终 checkpoint 与 `best/` 均正常保存，
但退出阶段打印：

```text
barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
WARNING: destroy_process_group() was not called before program exit
```

### 根因

Shaft 在训练结束阶段会调用 distributed barrier，但没有为 NCCL barrier 显式传入当前
CUDA device id。训练 CLI 退出时也没有显式调用 `torch.distributed.destroy_process_group()`，
因此 PyTorch 在进程退出阶段提示 process group 未主动销毁。

### 影响范围

- 影响 torchrun/DDP 启动的训练退出日志可读性。
- 不影响本次训练结果；本次运行已经成功完成并保存最终模型。
- 该 warning 本身不是 NCCL 通信失败。真正通信失败通常会伴随 timeout、rank 非零退出、
  `ChildFailedError` 或 barrier hang。

### 修复

- `barrier_if_distributed()` 在 NCCL backend 下显式传入当前 CUDA device id。
- 新增 `destroy_process_group_if_initialized()`，仅在 distributed 已初始化时执行销毁。
- 训练 CLI 在 `finally` 中调用销毁 helper，确保成功和异常退出都会清理 process group。

### 回归测试

- `tests/test_distributed_runtime.py::test_barrier_if_distributed_noop_without_dist`
- `tests/test_distributed_runtime.py::test_barrier_if_distributed_passes_nccl_device_ids`
- `tests/test_distributed_runtime.py::test_destroy_process_group_if_initialized_calls_dist_destroy`

### 后续防线

- 新增分布式收尾逻辑必须走 `shaft.utils.distributed`，避免各 pipeline 自己直接操作
  `torch.distributed`。
- torchrun/DDP smoke 除了检查训练完成，也应关注退出阶段是否还有 NCCL/process group 清理 warning。

## 2026-05-03: Grounding GRPO 离线最优点早于最终 checkpoint

### 现象

两卡 grounding GRPO 训练完成后，对 SFT baseline 与 GRPO checkpoint 做离线 grounding eval。
结果显示 `checkpoint-1200` 的 grounding macro F1 略高于 SFT，但最终 `checkpoint-1526`
和 `best/` 回落，且 `best/` 与 final 指标一致。

### 根因

当前 GRPO 配置未开启 online eval，`load_best_model_at_end=false`，因此 `best/` 不是按
validation `det_f1` 选择的 best，而是训练结束后的 final save。同时旧 reward 主要由
`grounding_iou` 驱动，和 offline `det_f1` 并不完全一致；`beta=0.0` 也缺少对 SFT policy
的 KL 约束，容易在 SFT 已较强时出现后段过优化。

这是训练选择与 reward 对齐问题，不是 offline eval/codec/metric 的误判。

### 影响范围

- 影响 grounding GRPO checkpoint 选择；不能默认使用最终 `best/`。
- 影响 reward 优化方向；单看 IoU reward 可能掩盖漏检、重复框和 precision/recall 退化。
- 影响训练监控；没有 online eval 时无法及时发现真实 validation F1 的回落。

### 修复

- GRPO pipeline 支持 `eval.online_metrics_enabled=true`，复用 SFT 的
  `ShaftOnlineEvalRunner` 做 generation-based validation。
- GRPO 配置开启 grounding online eval，使用与 SFT 相同的 grounding eval policy 与
  `max_new_tokens=2048`。
- 新增 `grounding_det_f1` reward，使训练 reward 直接对齐 offline `det_f1` 的
  IoU-threshold matching 语义。
- 新一版 GRPO 配置降低学习率、增加 `beta=0.02`，并把 `grounding_iou` 降为辅助 reward。

### 回归测试

- `tests/test_config_loader.py::test_load_config_supports_grpo_online_eval_dataset_policies`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_wires_grpo_online_eval_runner_with_named_eval_datasets`
- `tests/test_training_modules.py::test_build_grpo_reward_functions_supports_grounding_det_f1`

### 后续防线

- GRPO 长训必须开启 online eval，并让 `save_steps` 与 `eval_steps` 对齐。
- grounding reward 应至少包含 parse 完整性、det F1 和 IoU 辅助项，避免只优化局部框 IoU。
- 若 `best/` 与 final 指标一致，需要确认是否真的启用了 `load_best_model_at_end` 和
  `metric_for_best_model=eval_final_score`。

## 2026-05-03: Grounding GRPO v2 首步长时间无进度

### 现象

启动两卡 GRPO v2 online eval 训练后，日志停在 `train: 1/1526` 附近。诊断时 rank0
在 GPU0 上持续 100% 利用率，rank1 占用 GPU1 约 85GB 但 GPU 利用率为 0%，输出目录没有
checkpoint 写入。

### 根因

本轮把 `rlhf.grpo.beta` 从 0 调到 0.02 后，TRL GRPO 会创建 reference model 用于 KL。
当前 TRL 的 `create_model_from_path()` 在未显式传 `dtype` 时默认按 `float32` 加载 reference
model；这与主模型 bf16 训练精度不一致，导致显存占用接近卡容量并明显拖慢首步。

同时训练 rollout 的 `max_completion_length=2048` 会让每个 generation batch 采样更长 completion。
这应与 online eval 的 `eval.max_new_tokens=2048` 区分；eval 需要和 SFT 对齐，训练 rollout
不应默认使用同样长的上限。

### 影响范围

- 影响 `beta > 0` 的 GRPO full-finetune 训练，尤其是 colocate vLLM 场景。
- 不影响 `beta=0` 的旧 run，因为旧配置不会创建 reference model。
- 不影响 SFT online eval；这是 GRPO ref model 初始化精度与 rollout 长度组合导致的训练性能问题。

### 修复

- GRPO 的 TRL config 装配会根据 `TrainingArguments` 精度设置 `model_init_kwargs.dtype`：
  - `bf16=true` 时传 `dtype=bfloat16`
  - `fp16=true` 时传 `dtype=float16`
- 训练 rollout 的 `max_completion_length` 调回 1024。
- 保留 online eval 的 `eval.max_new_tokens=2048`，继续与 SFT 的生成评估口径对齐。

### 回归测试

- `tests/test_training_modules.py::test_build_trl_grpo_config_sets_bf16_model_init_kwargs`
- `tests/test_training_modules.py::test_build_trl_grpo_config_from_training_args`

### 后续防线

- 以后只要 GRPO 开启 `beta > 0`，启动日志里 reference model 不应再出现
  `default dtype torch.float32`。
- 训练 rollout 长度和 eval 生成长度必须分开审查；为了监控能力可以让 eval 更长，但训练采样
  应先按吞吐和 reward 可用性选上限。

## 2026-05-03: GRPO online eval 在 step 200 触发 rollout prepare 报错

### 现象

GRPO v2 训练推进到 step 200 后触发 online eval，两个 rank 同时报错：
`TypeError: string indices must be integers, not 'str'`。堆栈显示
`ShaftOnlineEvalRunner.collect_samples()` 调用 `trainer._prepare_inputs(batch)` 后进入了
TRL `GRPOTrainer._prepare_inputs()`，该函数继续调用 `_generate_and_score_completions()` 并按
`x["prompt"]` 读取输入。

### 根因

SFT online eval 里 `trainer._prepare_inputs()` 只做标准 HF batch 设备搬运；但 GRPOTrainer
覆写了同名方法，把它变成训练 rollout 的 generation/scoring 入口，要求输入是 GRPO 样本列表。
online eval 使用的是 `SFTCollator` 产出的模型输入 dict，两者语义不兼容。

这是 online eval 与 GRPO trainer 方法名复用导致的 trainer 接口误用，不是数据、codec 或 metric
本身的问题。

### 影响范围

- 影响 `eval.online_metrics_enabled=true` 的 GRPO 训练。
- 不影响 SFT online eval。
- 不影响 GRPO step 200 前的训练；报错发生在 `_maybe_log_save_evaluate()` 的 eval 阶段。

### 修复

- `ShaftOnlineEvalRunner` 优先调用 trainer 的 `prepare_online_eval_inputs()` hook。
- `ShaftGRPOTrainer.prepare_online_eval_inputs()` 显式调用 HF `Trainer._prepare_inputs()`，只做标准
  batch 准备，绕开 TRL GRPO 的 rollout `_prepare_inputs()`。
- 没有该 hook 的 trainer 继续走原有 `_prepare_inputs()`，保持 SFT 行为不变。

### 回归测试

- `tests/test_online_eval.py::test_online_eval_runner_uses_online_prepare_hook`
- `tests/test_pipeline_rlhf.py::test_run_rlhf_wires_grpo_online_eval_runner_with_named_eval_datasets`

### 后续防线

- online eval runner 不应直接假设所有 trainer 的 `_prepare_inputs()` 都是 HF 原始语义。
- 接入新 trainer 时，如果它覆写了 `_prepare_inputs()`、`prediction_step()` 或 dataloader 行为，
  必须显式确认 online eval 的 batch 准备路径不会触发训练专用逻辑。

## 2026-05-05: raw arrow/layout 标注语义归一与噪声清理

本次把 `data/raw_arrow/json` 统一为 `label=arrow + bbox + linestrip` schema；旧
`c0-c7` bbox 标签的单/双头、直/曲、实/虚信息进入 `subattr`，新增 connector 数据缺少
单/双头标注，因此保持 `arrow_type=unknown`。`data/raw_layout/json` 删除了 41 个零宽或零高
噪声实例，并复查同图内同 label、同 bbox 重复数为 0。

两个 raw 数据目录的当前状态和后续注意事项以各自 README 为维护入口：
`data/raw_arrow/README.md`、`data/raw_layout/README.md`。

## 2026-05-06: raw_layout/raw_arrow 合并为统一 raw_data 真源

### 现象

原始 layout 和 arrow 标注分散在两套 raw 目录中，同一张图片可能有 layout 层、arrow 层或只存在
未标注库存图片。继续维护两套 raw 目录会让 split、preview、补标状态和后续派生数据生成出现多处
状态源。

### 根因

`raw_layout` 和 `raw_arrow` 最初服务不同任务，目录结构和 split 独立；但后续多任务训练与补标流程
需要以图片为中心管理 layer 覆盖状态。任务级 raw 目录不能表达“该图只标了 arrow、layout 未标”或
“该图暂未进入任何标注层”的统一状态。

### 影响范围

- raw 真源切换为 `data/raw_data`；旧 `raw_layout` / `raw_arrow` 不再作为新维护入口。
- 训练派生数据仍按任务读取：arrow 使用 arrow layer，layout 使用 layout layer。
- `data/` 被 Git 忽略，实际 raw 数据通过共享目录同步；仓库中维护的是数据管理规则和生成代码。

### 修复

- 合并已有 JSON 标注为 instance-centric `shaft.raw_data.v1`：
  - `annotation.layers` 记录覆盖层，固定按 `layout`、`arrow` 顺序。
  - `annotation.status` 记录每个 layer 的流程状态。
  - layout instance 保持 `label + bbox + extra`。
  - arrow instance 保持 `label + bbox + linestrip + subattr + extra`。
- 将 layout image-only 库存也写入 `raw_data`，使用
  `annotation.layers=[]`、`annotation.status={}`、`instances=[]`，作为未来补标库存，
  不作为任何任务负样本。
- 按任务生成 split：`arrow_train/val` 和 `layout_train/val`；当前只从已标注 layer 中划分。
- Preview 改为按 label 生成：`icon`、`image`、`shape`、`arrow`，并复用
  `shaft.metrics.visualization` 的统一绘制风格。
- `shaft-data-manager` skill 收口为统一 raw_data 维护入口，原数据增强 skill 并入该 skill。

### 回归测试

- 校验 `raw_data/json` 与 `raw_data/images` 一一对应。
- 校验旧 layout/arrow instances 与合并后对应 layer 的 instances 完全一致。
- 校验 split 中每个 stem 都存在 JSON 和图片。
- 校验 preview 数量与每个 label 出现的 JSON 数一致。

### 后续防线

- 不再从旧 `raw_layout` / `raw_arrow` 目录启动新数据维护任务。
- 缺失 layer 不能当作负样本；只有 completed layer 且无 instance 才能表示人工确认负样本。
- raw split 和 preview 只是辅助状态，不替代 `annotation.layers` / `annotation.status`。

## 2026-05-06: SFT arrow/layout/keypoint v2 训练 step 547 OOM 引发 NCCL timeout

### 现象

`train_sft_4b_grounding.yaml` 从旧 SFT best 初始化后，2 卡 full SFT 在 step 547 附近失败。
rank1 在 `cross_entropy` 分配约 10.94 GiB 时 CUDA OOM；rank0 随后卡在 allreduce，30 分钟后
NCCL watchdog 报 `WorkNCCL(ALLREDUCE) timeout` 并终止进程组。

### 根因

首要根因是单卡 micro-batch 过大：`per_device_train_batch_size=4` 配合 4B full finetune、
`max_pixels=1048576` 和长 target 时，loss/logits 计算峰值显存超过 83GB 卡余量。NCCL timeout
是某个 rank OOM 后其它 rank 继续等待 collective 的连带结果，不是通信链路先异常。

日志里的 `Num examples = 22,196` 是当前 sharded mixed sampler 的单 rank 长度；全局 mix 仍按
catalog 权重生成约 44,392 条/epoch，不是数据源只剩 keypoint。

### 影响范围

- 影响当前 4B full SFT v2 配置下的训练稳定性。
- 不表示旧 SFT best checkpoint 损坏。
- 不表示 DDP/NCCL 本身不可用；OOM 后的 allreduce timeout 是预期连锁失败。

### 修复

- 将训练 micro-batch 从 `per_device_train_batch_size=4` 降到 `2`。
- 将 `gradient_accumulation_steps` 从 `2` 提到 `4`，保持 global batch size 仍为 16。
- 将 online eval batch size 从 `8` 降到 `2`，避免 epoch 2 生成式 eval 再次触发显存峰值。
- 建议重跑时设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`，降低显存碎片导致的边界 OOM 风险。

### 回归测试

- 配置加载检查：确认 batch/accum/eval batch 字段解析为新值。
- 训练重跑时以前 600 step 为 canary，若通过原失败点，说明本轮 OOM 风险已降低。

### 后续防线

- full finetune + multimodal large pixel budget 不应默认使用 per-device train batch 4。
- 看到 NCCL timeout 时先检查其它 rank 是否 OOM、异常退出或数据读取失败，再判断通信问题。
- 训练日志里的 `Num examples` 在 sharded custom sampler 场景下可能是单 rank 视角，数据 mix
  应用 sampler/global quota 复核。
