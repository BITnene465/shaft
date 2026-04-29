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
