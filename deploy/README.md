# 部署交接

`deploy/` 只放后端接模型所需的协议和运行时代码，不依赖 `src/` 的训练、数据准备和评估代码。

## 怎么用

### 1. 读配置

```python
from deploy.arrow.config import load_arrow_config

config = load_arrow_config()
```

`deploy/arrow/config.yaml` 统一管理：

- prompt
- route 名称
- token 上限
- `do_sample` / `temperature` / `top_p`
- `padding_ratio`
- 量化 bins

默认推荐贪心解码：

- `do_sample: false`
- `temperature: 0.0`
- `top_p: 1.0`

### 2. 构建 pipeline

```python
from deploy.arrow.pipeline import ArrowTwoStagePipeline

pipeline = ArrowTwoStagePipeline(
    base_url="http://127.0.0.1:8001/v1",
    config=config,
)
```

### 3. 做两阶段推理

```python
result = pipeline.predict_two_stage(image_path)
print(result.to_dict())
```

## 代码入口

后端主要使用这三个文件：

- [`deploy/arrow/config.yaml`](./arrow/config.yaml)
- [`deploy/arrow/decode.py`](./arrow/decode.py)
- [`deploy/arrow/pipeline.py`](./arrow/pipeline.py)

`deploy/arrow/decode.py`：

- `decode_stage1_output(...)`
- `decode_stage2_output(...)`

`deploy/arrow/pipeline.py`：

- `ArrowVLLMClient`
- `ArrowTwoStagePipeline`
- `build_padded_crop(...)`

这套 runtime 负责：

- 图像读取
- Stage1 请求与解码
- crop 预处理
- Stage2 请求与解码
- 全局坐标回写
- 最终结果拼装

## 模型组织

这份代码不强依赖“两个不同模型”。Stage1 和 Stage2 可以：

- 用两个 LoRA 路由
- 用同一个 LoRA 路由
- 以后统一成一个模型路由

当前默认路由名是：

```text
grounding_arrow
keypoint_sequence_arrow
```

如果后期只想用一个模型，把 `deploy/arrow/config.yaml` 里的两个 `route` 改成同一个值即可。

## vLLM 启动示例

```bash
vllm serve deployment/qwen3vl/arrow/base_model \
  --enable-lora \
  --lora-modules grounding_arrow=deployment/qwen3vl/arrow/adapters/grounding_arrow \
                    keypoint_sequence_arrow=deployment/qwen3vl/arrow/adapters/keypoint_sequence_arrow
```

单个请求只会使用一个 LoRA 路由。

## 两阶段编排

1. Stage1 整图 grounding
2. 后端根据 bbox 裁 crop
3. Stage2 对 crop 做 keypoint 预测
4. 后端合并结果

## vLLM 说明

- 支持多个 LoRA 路由
- 单个请求只用一个 LoRA
- 支持 Qwen3-VL 的 pixel budget 思路
- tower / connector LoRA 仍是实验性能力

如果动态 LoRA 不稳定，就退回到：

- merge 成 dense model
- 再部署到 vLLM

## 后端边界

后端需要我们提供：

- 基模权重目录
- 两个 LoRA checkpoint
- deployment bundle
- [`deploy/arrow/config.yaml`](./arrow/config.yaml)
- [`deploy/arrow/decode.py`](./arrow/decode.py)
- [`deploy/arrow/pipeline.py`](./arrow/pipeline.py)

后端不需要：

- 训练代码
- 数据准备代码
- 评估代码
- 实验脚本

## Bundle 结构

```text
deployment/qwen3vl/arrow/
  base_model/
  adapters/
    grounding_arrow/
    keypoint_sequence_arrow/
  manifests/
    adapters.json
```

## 联调说明

两阶段在线推理默认使用贪心解码。需要采样时，再在 `deploy/arrow/config.yaml` 里显式开启 `do_sample` 并调整 `temperature` / `top_p`。
