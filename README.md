# shaft（重构中）

当前仓库正在做“训练框架从头重构”：

- 训练内核统一为 SFT/RL 算法抽象（先落地 SFT）。
- 训练与保存强依赖 Hugging Face 标准能力（Trainer/TrainingArguments）。
- 旧实现已迁移到 `old/` 目录归档。

## 快速开始

```bash
uv venv --python 3.11 --prompt shaft
source .venv/bin/activate
uv pip install -e .
python scripts/train.py sft --config configs/train/train_sft_4b.yaml
```

按用途安装扩展依赖（不改变 `transformers==4.57.6`）：

```bash
# 训练基础（HF 生态）
uv pip install -e ".[train]"

# GPU 训练增强（FlashAttention / 8bit）
uv pip install -e ".[train,gpu]"

# 强化学习（TRL）
uv pip install -e ".[train,rlhf]"

# 部署（vLLM）
uv pip install -e ".[serve]"
```

统一训练入口在 `scripts/train.py`，任务命令在 `src/shaft/cli` 层实现：

```bash
python scripts/train.py sft --config configs/train/train_sft_4b.yaml
python scripts/train.py rlhf --config configs/train/train_sft_4b.yaml --algorithm dpo
```

推理支持引擎与多阶段流水线编排：

```bash
python scripts/infer.py --config configs/infer/pipeline_smoke.yaml --image /path/to/image.png
```

可选 hooks（训练时触发）：

```yaml
plugins:
  hooks: ["log_on_save"]
```

## 新架构（进行中）

- `src/shaft/config`：强类型配置与加载。
- `src/shaft/data`：数据源、样本级 mixing、SFT dataset/collator。
- `src/shaft/model`：HF 模型/processor/tokenizer 构建。
- `src/shaft/algorithms`：`sft/dpo/ppo` 算法注册与入口（后两者暂为占位）。
- `src/shaft/pipeline`：训练流水线（分阶段组装 model/data/algorithm/trainer）。
- `src/shaft/infer`：`InferEngine` 与 `InferPipeline`，支持单/多模型的多阶段推理编排。
- `src/shaft/plugins`：注册表与 Hook 拦截机制。

## 说明

- 本阶段先保证“最小可训练闭环 + 测试驱动重构”。
- 当前模型注册仅启用 `qwen3vl`，其它模型后续按“每模型一实现文件”扩展。
- 结构化任务语义评估会在后续以离线评估模块接入。
- 新训练数据格式推荐使用 `messages`（尾部 assistant 作为监督目标）；兼容 legacy `target_text`。

## 测试约定

测试默认执行“快速回归测试”（排除耗时/重依赖测试）：

```bash
pytest -q
```

集成测试通过 marker 管理：

- `integration`：包含模型加载、真实推理链路等耗时用例（默认跳过）。
- `manual`：仅在人工可控环境执行的重型验证。

本地常用命令：

```bash
pytest -q -m integration   # 只跑集成测试（一般需要本地模型与资源）
pytest -q -m manual        # 只跑手工验证测试
pytest -q -m "integration or manual"  # 临时跑所有重型验证
```

新增集成/手工用例约定：

- 需有明确 `skip` 保护（如本地模型不存在时跳过）。
- 失败路径与输出信息应明确（例如缺少模型权重、适配器未注册）。
- 正式流水线默认仍执行 `pytest -q`（即跳过集成级用例）。

更完整测试规范见：[docs/testing.md](docs/testing.md)
