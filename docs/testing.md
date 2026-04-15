# Shaft 测试规范

本文档定义当前仓库的测试层级、执行方式和模块责任矩阵。

## 1. 测试目标

- 保证主干训练链可回归。
- 保证推理、导出、配置等关键模块有单测兜底。
- 把重型测试与日常快速回归分层管理。

## 2. 测试层级

### 2.1 单元测试

特点：

- 不依赖大模型
- 不依赖外部服务
- 聚焦单模块行为

典型覆盖：

- config 校验
- registry 行为
- data source 解析
- mixing / transform
- optimizer / scheduler / loss 注册
- export 校验逻辑

### 2.2 Smoke 测试

特点：

- 跑一条最短主链
- 使用轻量模型或最小数据
- 验证组件装配是通的

典型覆盖：

- SFT 主链
- RLHF 主链
- 分布式最小链路

### 2.3 Integration 测试

特点：

- 加载真实模型或真实推理后端
- 运行真实推理链
- 默认不进入快速回归

### 2.4 Manual 测试

特点：

- 人工触发
- 重型、耗时、依赖环境

## 3. 推荐执行命令

### 日常快速回归

```bash
pytest -q
```

### 只跑 integration

```bash
pytest -q -m integration
```

### 只跑 manual

```bash
pytest -q -m manual
```

## 4. 按模块划分的最低测试责任

| 模块 | 至少需要的测试 |
| --- | --- |
| `config` | schema / loader / normalize |
| `data` | source / dataset / collator / mixing / data center |
| `model` | model registry / template registry / builder |
| `pipeline` | SFT / RLHF pipeline smoke |
| `training` | optimizer / scheduler / loss / checkpointing |
| `codec` | shared codec / JSON repair / partial parse |
| `metrics` | online eval metric registry / aggregator |
| `infer` | loader / pipeline / CLI |
| `export` | inspect / validate / merge-peft |
| `cli` | 命令解析与 override |

## 5. 变更类型与必跑清单

### 新增配置字段

- 对应 `config` 测试
- 至少一条消费该字段的 smoke

### 新增数据源或 mixing 规则

- `tests/test_data_sources.py`
- `tests/test_data_center.py`
- `tests/test_mixing.py`
- 如涉及 batch 结构，再跑 `tests/test_collator.py`

### 新增模型族或模板

- `tests/test_model_registry.py`
- `tests/test_template_registry.py`
- 必要时补最短加载 smoke

### 新增算法或训练编排变化

- `tests/test_pipeline_sft.py` 或 `tests/test_pipeline_rlhf.py`
- 对应算法单测
- 必要时补 smoke

### 新增推理能力

- `tests/test_infer_loader.py`
- `tests/test_infer_pipeline.py`
- `tests/test_infer_cli.py`
- 若涉及真实模型或后端，再补 integration/manual

### 新增共享 codec 或在线 eval 能力

- `tests/test_codec.py`
- `tests/test_online_eval.py`
- `tests/test_config_loader.py`
- 如改动 SFT 装配链，再跑 `tests/test_pipeline_sft.py` 与 `tests/test_training_modules.py`

### 新增导出能力

- `tests/test_export_tools.py`
- `tests/test_export_cli.py`
- 必要时加 checkpointing 测试

## 6. 标记规则

- `integration`: 真实模型加载、真实推理链路、依赖服务
- `manual`: 人工触发的重型验证

要求：

- 重型用例必须支持 `skip`
- skip 原因必须清晰，例如模型不存在、GPU 不可用、服务未启动

## 7. 与文档的联动要求

当测试边界变化时，需要同步更新：

- `README.md`
- `docs/architecture.md`
- `docs/testing.md`

## 8. 当前特别说明

- PPO 仍是受限能力，现阶段只维持 smoke 级测试，不作为完整生产验收能力。
- 真实 `Qwen3VL` 的推理 integration 可以长期保留，但不应默认进入快速回归。
