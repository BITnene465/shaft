# Shaft

面向工程师与研究者的多模态训练与推理框架，基于 Hugging Face 生态构建。当前主线是 **Qwen 多模态 SFT**，同时提供推理编排、模型导出与实验性蒸馏能力。

[文档索引](docs/README.md) · [配置参考](docs/config_reference.md) · [使用指南](docs/usage_guide.md) · [训练评测记录](docs/training_evaluation/README.md)

## 能力概览

| 方向 | 当前能力 | 使用边界 |
| --- | --- | --- |
| SFT | Qwen 多模态全参数与 PEFT 微调，数据混合、分组与批次规划 | 模型、精度、packing 和分布式后端需符合配置门禁 |
| 训练优化 | 梯度检查点、Liger 算子、结构化参数组、多种 loss 归一化 | 不同模型与后端的支持范围不同，不能直接外推显存容量 |
| 推理 | 本地 HF、vLLM OpenAI 兼容后端，单阶段与多阶段编排 | 公共 API 为同步单样本接口，不提供原生异步队列或 serving 层 |
| 导出 | HF / PEFT 校验、adapter 合并、独立控制导出精度 | 仅模型权重的 checkpoint 不能精确断点续训 |
| 蒸馏与 RL | Offline KD、OPD；实验性 DPO / GRPO | 按专项文档使用；PPO 仅用于 debug smoke |

完整支持范围见[使用指南](docs/usage_guide.md)和[待办与验收缺口](docs/TODO.md)。

## 安装

使用 **Python 3.11**。GPU 训练前需确认驱动、CUDA toolkit 与依赖兼容。

```bash
git clone https://github.com/BITnene465/shaft.git
cd shaft
uv venv --python 3.11 --prompt shaft
source .venv/bin/activate
uv pip install -e ".[train]"
```

按需选择额外依赖，不必全部安装：

| 用途 | 命令 |
| --- | --- |
| Liger CE、RMSNorm、SwiGLU | `uv pip install -e ".[train,fused-ce]"` |
| FlashAttention、FLA、causal-conv1d | `uv pip install -e ".[train,gpu-kernels]"` |
| bitsandbytes | `uv pip install -e ".[train,gpu]"` |
| DeepSpeed | `uv pip install -e ".[train,distributed]"` |
| RLHF | `uv pip install -e ".[train,rlhf]"` |
| vLLM 部署 | `uv pip install -e ".[serve]"` |
| 开发与测试 | `uv pip install -e ".[train,dev]"` |

安装 Liger **不等于开启算子**，需在配置中显式设置对应开关。CUDA 扩展可能需要编译。

如果 CUDA 不在默认路径，可参考 [`.shaft.env.example`](.shaft.env.example) 创建不提交到 Git 的 `.shaft.env`，填写 `CUDA_HOME`。多卡 Triton/FLA 训练建议将 `SHAFT_TRITON_CACHE_ROOT` 指向节点本地盘，避免共享盘上的编译缓存竞争。已有 shell 环境变量不会被该文件覆盖。

## 开始训练

仓库配置是**需要填写路径的示例**，不包含模型和训练数据。运行前先配置：

1. `model.model_name_or_path`：模型位置，以及与之匹配的模型类型和模板。
2. `data`：数据集路径、像素预算、长度和显式 batching 配置。
3. `train`：训练步数、batch、学习率、精度与保存策略。
4. `experiment.output_dir`：本次运行独立的输出目录。

[`configs/train/sft_4b.yaml`](configs/train/sft_4b.yaml) 是 Qwen3-VL 示例，不是 Qwen3.5 配置；其中启用了 FlashAttention 2，需先安装对应依赖，或在适用的配置中改用 SDPA。

```bash
# 单进程入口；确认配置与显存容量后执行
python scripts/train.py sft --config /path/to/train.yaml

# 单机八卡；GPU 数量、每卡 batch 和累积步数由实验配置共同决定
torchrun --standalone --nproc_per_node=8 \
  scripts/train.py sft --config /path/to/train.yaml
```

先做短程预检，再启动长训练。增加 DDP 卡数不会自动分摊每张卡的参数和优化器状态。

### 关键配置

以下为配置片段，需合并到完整训练 YAML：

```yaml
model:
  torch_dtype: float32
train:
  bf16: true
  loss_normalization: microbatch_token
  liger:
    fused_linear_ce: true
    rms_norm: true
    swiglu: true
  save_only_model: true
  export_dtype: float32
```

- **加载、训练、保存精度分开控制**：`model.torch_dtype`、`train.bf16/fp16`、`train.export_dtype` 各自独立。上例以 FP32 参数进行 BF16 混合精度训练，保存 FP32 部署权重。
- **Loss 归一化显式选择**：支持 `global_token`、`rank_token`、`microbatch_token`，默认是 `global_token`。三者并非等价，见[归一化说明](docs/sft_loss_normalization.md)。
- **Liger 有适用范围**：上述开关用于已支持的 dense Qwen3.5 VL SFT 路径，详见[配置参考](docs/config_reference.md#trainliger)。
- **保存权重不等于保存训练状态**：`save_only_model: true` 不保存 optimizer、scheduler 和 RNG，不能精确续训。导出 dtype 转换也有后端限制，详见[配置参考](docs/config_reference.md)。

## 推理与导出

```bash
# 推理：先填写推理配置中的模型与后端信息
python scripts/infer.py \
  --config configs/infer/pipeline_smoke.yaml --image /path/to/image.png

# 检查 checkpoint
python scripts/export.py inspect --path /path/to/checkpoint

# 合并 PEFT adapter；模型类型应与实际模型匹配
python scripts/export.py merge-peft \
  --model-type qwen3vl \
  --adapter-path /path/to/adapter \
  --base-model /path/to/base_model \
  --output-dir /path/to/merged_model
```

多图、重试、超时和多阶段编排见[推理文档](docs/infer.md)；adapter 来源校验与合并限制见[导出文档](docs/export.md)。Offline KD、OPD 和 RL 的命令见[使用指南](docs/usage_guide.md)。

## 文档导航

| 需要做什么 | 阅读入口 |
| --- | --- |
| 查配置字段、合法组合与默认值 | [配置参考](docs/config_reference.md) |
| 准备数据、模板和 prompt | [数据合同](docs/data.md) |
| 理解 batching、packing 和恢复规则 | [批次规划](docs/training_batch_planning_design.md) |
| 核对训练与评测语义 | [框架对照](docs/training_framework_comparison.md) · [在线评测](docs/online_eval_design.md) |
| 阅读历史模型结果与分析 | [训练评测总览](docs/training_evaluation/README.md) |
| 开发、扩展和运行测试 | [架构](docs/architecture.md) · [扩展指南](docs/extension_guide.md) · [测试规范](docs/testing.md) |
| 查全部文档与已知限制 | [文档索引](docs/README.md) · [TODO](docs/TODO.md) |

## 开发验证

```bash
uv run pytest -q tests --suite smoke
uv run pytest -q tests --suite numerics
```

新实现位于 `src/shaft`，`scripts/` 只保留薄入口，旧代码归档在 `old/`。当前能力以参考文档和测试门禁为准，[开发日志](docs/development_log.md)用于追溯变化，不作为支持矩阵。
