# 部署目录说明（给后端）

`deploy/` 仅保留线上推理必需代码，不依赖训练/评估链路。

## 1. 目录作用

后端主要接入：

- `deploy/arrow/config.yaml`
- `deploy/arrow/config.py`
- `deploy/arrow/decode.py`
- `deploy/arrow/pipeline.py`

其中：

- `decode.py` 提供两阶段解码函数：
  - `decode_stage1_output(...)`
  - `decode_stage2_output(...)`
- `pipeline.py` 提供两阶段编排：
  - Stage1 请求与解析
  - crop 构建
  - Stage2 请求与解析
  - 全局坐标回写

## 2. 快速使用

### 2.1 读取配置

```python
from deploy.arrow.config import load_arrow_config
config = load_arrow_config()
```

### 2.2 构建两阶段 pipeline

```python
from deploy.arrow.pipeline import ArrowTwoStagePipeline

pipeline = ArrowTwoStagePipeline(
    base_url="http://127.0.0.1:8001/v1",
    config=config,
)
```

### 2.3 执行推理

```python
result = pipeline.predict_two_stage("/path/to/image.png")
print(result.to_dict())
```

## 3. 模型组织

支持一个 base model + 多个 LoRA route，也支持单 route 混合模型。当前默认是单 route：

- `arrow_mixed_4b`（stage1/stage2 共用同一 route）

如需切回双 route，只需把 `config.yaml` 中 `stage1.route` 和 `stage2.route` 分别改成目标 route。

## 4. vLLM 启动示例

```bash
vllm serve deployment/qwen3vl/arrow/base_model \
  --port 8001 \
  --enable-lora \
  --lora-modules grounding_arrow=deployment/qwen3vl/arrow/adapters/grounding_arrow \
                 keypoint_sequence_arrow=deployment/qwen3vl/arrow/adapters/keypoint_sequence_arrow
```

建议按显存设置：

- `--gpu-memory-utilization`
- `--max-model-len`

注意：

- 单请求只会命中一个 LoRA route。
- `pixel budget` 由 `deploy/arrow/config.yaml` 的 `stage1/2.{min_pixels,max_pixels}` 控制，
  运行时按请求下发到 vLLM（无需在 `vllm serve` 固化 `--mm-processor-kwargs`）。
- vLLM 当前版本对 DoRA 支持可能受限，若 adapter 为 DoRA 需先验证兼容性。

## 5. 推荐解码默认

默认建议贪心解码：

- `do_sample: false`
- `temperature: 0.0`
- `top_p: 1.0`

如需采样，再在 `deploy/arrow/config.yaml` 显式开启。

## 6. 交付边界

算法侧交付：

- base model 目录
- LoRA adapter 目录（一个或多个）
- `deploy/` 协议与编排代码

后端无需接入：

- 训练代码
- 数据准备代码
- 离线评估代码
