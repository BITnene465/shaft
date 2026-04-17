# Shaft 脚本使用说明

本文档说明仓库 `scripts/` 目录下的正式脚本如何使用。

范围约束：
- 只覆盖正式脚本入口
- 覆盖 `scripts/tasks/` 下的任务脚本
- **不覆盖 `scripts/tmp/`**；`tmp` 目录视为临时实验区，不属于稳定接口

## 1. 设计原则

`scripts/*.py` 的定位是**薄入口**：
- CLI 解析与命令编排放在 `src/shaft/cli`
- `scripts/*.py` 只负责调用对应 CLI 主入口

当前唯一例外是：
- `scripts/tasks/convert_grounding_structured_to_sft.py`

它属于明确的任务数据准备脚本，不是训练内核入口。

## 2. 顶层脚本

### `scripts/train.py`

用途：
- 统一训练入口
- 当前支持 `sft` 与 `rlhf` 子命令

常用形式：

```bash
python scripts/train.py sft --config configs/train/train_sft_4b.yaml
python scripts/train.py rlhf --config configs/train/train_dpo_4b.yaml
```

兼容写法：

```bash
python scripts/train.py --config configs/train/train_sft_4b.yaml
```

说明：
- 如果直接传 `--config`，当前默认走 `sft`
- 真正的命令定义在 `src/shaft/cli`

### `scripts/infer.py`

用途：
- 运行可配置的多阶段推理 pipeline

常用形式：

```bash
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml \
  --image path/to/image.png
```

带初始上下文：

```bash
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml \
  --image path/to/image.png \
  --inputs '{"document_id":"demo-001"}'
```

说明：
- `--inputs` 是 JSON 字符串
- 输出会打印为 JSON

### `scripts/export.py`

用途：
- HF 兼容导出工具
- checkpoint 布局检查
- PEFT adapter 合并

子命令：
- `inspect`
- `validate`
- `merge-peft`

示例：

```bash
python scripts/export.py inspect --path outputs/run_x/checkpoint-100
```

```bash
python scripts/export.py validate \
  --path outputs/run_x/checkpoint-100 \
  --finetune-mode lora \
  --model-type qwen3vl
```

```bash
python scripts/export.py merge-peft \
  --model-type qwen3vl \
  --adapter-path outputs/run_x/checkpoint-100 \
  --output-dir outputs/run_x/merged
```

### `scripts/web.py`

用途：
- 启动面向工程师/科研人员的 Web UI

常用形式：

```bash
python scripts/web.py
```

指定 host / port：

```bash
python scripts/web.py --host 0.0.0.0 --port 7861
```

指定默认训练配置：

```bash
python scripts/web.py --base-config configs/train/train_sft_4b.yaml
```

说明：
- 默认端口不固定；省略 `--port` 时由 Web UI 服务自动选择空闲端口
- `Ctrl-C` 视为正常退出

## 3. `scripts/tasks/`

`scripts/tasks/` 用于**任务级数据准备或转换脚本**。

这类脚本可以：
- 读写数据文件
- 生成训练前产物
- 服务具体业务任务

但不应：
- 承载训练内核语义
- 复制一套新的训练 CLI
- 替代 `src/shaft/cli`

### `scripts/tasks/convert_grounding_structured_to_sft.py`

用途：
- 把 `structured/*.jsonl` 的 grounding 结构化 GT 转成当前框架可训练的 `jsonl_sft`

输入要求：
- 结构化 GT 至少包含：
  - `sample_id`
  - `image_path`
  - `image_width`
  - `image_height`
  - `instances`
- `instances` 中每个元素至少有：
  - `label`
  - `bbox`

输出字段：
- `image_path`
- `sample_id`
- `dataset_name`
- `system_prompt`
- `user_prompt`
- `target_text`
- `extra`

关键行为：
- `bbox` 会量化到 `1000` bins，输出为 `bbox_2d`
- `target_text` 是纯 JSON array
- 排序规则：
  - 先按 `bbox_area / image_area` 的 **log 尺度分桶**
  - 当前默认 `bucket_base = 1.5`
  - 同桶内按 `(y1, x1, y2, x2, label)` 排序
- prompt 从 YAML 配置文件读取

常用形式：

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow/structured/train.jsonl \
  --output data/grounding_arrow/sft/train.jsonl \
  --dataset-name grounding_arrow
```

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow/structured/val.jsonl \
  --output data/grounding_arrow/sft/val.jsonl \
  --dataset-name grounding_arrow
```

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow_syn/structured/train.jsonl \
  --output data/grounding_arrow_syn/sft/train.jsonl \
  --dataset-name grounding_arrow_syn
```

常用参数：
- `--prompt-config`
  - 默认：`configs/prompts/grounding_arrow.yaml`
- `--num-bins`
  - 默认：`1000`
- `--area-bucket-base`
  - 默认：`1.5`
- `--no-readme`
  - 跳过输出目录下的 `README.md`

示例：

```bash
python scripts/tasks/convert_grounding_structured_to_sft.py \
  --input data/grounding_arrow/structured/train.jsonl \
  --output data/grounding_arrow/sft/train.jsonl \
  --dataset-name grounding_arrow \
  --prompt-config configs/prompts/grounding_arrow.yaml \
  --num-bins 1000 \
  --area-bucket-base 1.5
```

原子写入约束：
- JSONL 输出使用临时文件写完后 `os.replace`
- README 也使用原子替换
- 目的是避免训练进程读到半成品

## 4. 维护规则

新增脚本时，至少需要同步更新本文件，说明：
- 脚本用途
- 输入输出
- 关键参数
- 示例命令

如果脚本只是一次性临时实验，不应写进这里，而应留在 `scripts/tmp/`。
