# shaft（重构中）

当前仓库正在做“训练框架从头重构”：

- 训练内核统一为 SFT/RL 算法抽象（SFT/DPO/PPO）。
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
python scripts/train.py rlhf --config configs/train/train_dpo_4b.yaml --algorithm dpo
python scripts/train.py rlhf --config configs/train/train_ppo_4b.yaml --algorithm ppo
```

推理支持引擎与多阶段流水线编排：

```bash
python scripts/infer.py --config configs/infer/pipeline_smoke.yaml --image /path/to/image.png
python scripts/infer.py --config configs/infer/pipeline_vllm.yaml --image /path/to/image.png
```

`infer` 支持 stage 级 `codec`、`max_retries`、`fail_fast`，并在输出 `__trace__` 中记录每阶段尝试历史与耗时。  
`pixel budget` 支持 stage 级运行时覆盖：`min_pixels/max_pixels` 可在不同 stage 配不同值。  
`json_*` codec 默认使用容错解析：会优先严格 JSON，失败后尝试抽取可解析片段并做截断修复（适配长输出被截断场景）。
当前 backend：

- `hf_local`：本地 HF 模型直接推理。
- `vllm_openai`：调用 vLLM OpenAI 兼容接口（`endpoint + /v1/chat/completions`），stage 级 `min_pixels/max_pixels` 会透传为 `mm_processor_kwargs`。

可选 hooks（训练时触发）：

```yaml
plugins:
  hooks: ["log_on_save"]
```

## 新架构（进行中）

- `src/shaft/config`：强类型配置与加载。
- `src/shaft/data`：数据源、样本级 mixing、SFT/DPO/PPO dataset/collator。
- `src/shaft/model`：HF 模型/processor/tokenizer 构建。
- `src/shaft/algorithms`：`sft/dpo/ppo` 算法注册与入口。
- `src/shaft/pipeline`：训练流水线（`shaft_train`/`shaft_rlhf` 分流装配）。
- `src/shaft/infer`：`InferEngine` 与 `InferPipeline`，支持单/多模型的多阶段推理编排。
- `src/shaft/plugins`：注册表与 Hook 拦截机制。

## 说明

- 本阶段先保证“最小可训练闭环 + 测试驱动重构”。
- 当前模型注册仅启用 `qwen3vl`，其它模型后续按“每模型一实现文件”扩展。
- 结构化任务语义评估会在后续以离线评估模块接入。
- 新训练数据格式推荐使用 `messages`（尾部 assistant 作为监督目标）；兼容 legacy `target_text`。
- `jsonl_sft/jsonl_dpo/jsonl_ppo` 都支持行级聚合报错（会汇总坏样本行号与原因，而不是只报第一条）。
- DPO/PPO 已统一切换到 TRL 训练内核；Shaft 负责配置映射、数据形态与流水线编排。
- `jsonl_ppo` 当前使用 query-only 数据格式（样本提供 prompt，不再提供离线 `reward` 字段）。
- PPO 默认有两条安全保护：随机奖励头默认禁用（需显式开启 `allow_untrained_reward_model`），多模态模型默认禁用 text-only PPO 路径（需显式开启 `allow_text_only_multimodal_ppo`，仅建议 smoke/debug）。
- PPO 当前只支持 `lora/dora/qlora`，并默认使用 `value_model_mode=shared_backbone` + `reward_model_mode=adapter_disabled_policy` 以降低显存。
- PPO 暂停项与后续恢复条件见：[docs/ppo_todo.md](docs/ppo_todo.md)。

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
