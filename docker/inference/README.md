# Shaft Inference Docker Image

这个目录用于构建业务推理镜像。目标是让业务环境和当前 Shaft 验证环境对齐，避免因为
vLLM、Transformers、prompt、图像预处理或采样参数漂移导致检测效果不一致。

## 当前推理标准环境

以当前 `uv.lock` 和本机已验证 `.venv` 为准。推理镜像只安装业务推理所需 extra，
不安装训练专用依赖：

- `torch==2.10.0`
- `torchvision==0.25.0`
- `transformers==5.10.1`
- `vllm==0.19.1`
- `flash-attn==2.8.3`
- `flash-linear-attention==0.5.0`
- `causal-conv1d==1.6.2.post1`
- `numpy==2.2.6`

业务推理只需要镜像内的 vLLM + Shaft 推理契约工具，不需要训练数据、训练输出或 raw data。

## 构建

在仓库根目录执行：

```bash
docker build \
  -f docker/inference/Dockerfile \
  -t shaft-infer:qwen3vl-v4.1 \
  .
```

如需替换 CUDA 基础镜像：

```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 \
  -f docker/inference/Dockerfile \
  -t shaft-infer:qwen3vl-v4.1 \
  .
```

## 启动 vLLM

示例：四卡启动 `v4.1 checkpoint-8000`。

```bash
docker run --rm -it \
  --gpus '"device=0,1,2,3"' \
  -p 8000:8000 \
  -v /root/workspace/shaft/outputs/qwen3vl-sft/4b/banana-v4.1/checkpoint-8000:/models/banana-v4.1-checkpoint-8000:ro \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e MODEL_PATH=/models/banana-v4.1-checkpoint-8000 \
  -e SERVED_MODEL_NAME=banana_v4_1_step8000 \
  -e TENSOR_PARALLEL_SIZE=4 \
  -e MAX_MODEL_LEN=32768 \
  -e GPU_MEMORY_UTILIZATION=0.80 \
  -e MAX_NUM_SEQS=16 \
  shaft-infer:qwen3vl-v4.1
```

默认入口是 `shaft-start-vllm`，本质上启动：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --generation-config vllm
```

## 推理契约 smoke

业务不要直接拿原图裸调 vLLM。必须先按 Shaft 的 Qwen pixel budget 做 smart resize，
再使用同一份 prompt pool 和同一组生成参数。

容器内提供 `shaft-contract-smoke`，用于验证业务环境是否和 Shaft 输出契约一致：

```bash
shaft-contract-smoke \
  --endpoint http://127.0.0.1:8000/v1 \
  --model banana_v4_1_step8000 \
  --image /work/test.jpg \
  --prompt-path configs/prompts/pools/grounding_arrow.v2.4.yaml \
  --prompt-id main \
  --min-pixels 200704 \
  --max-pixels 2000000 \
  --max-tokens 4096 \
  --temperature 0 \
  --top-p 1 \
  --output /work/smoke_grounding_arrow.json
```

输出会记录：

- 模型名和 endpoint
- prompt path / prompt id / prompt hash
- pixel budget 前后尺寸
- generation 参数
- finish reason
- raw model output
- shared `json_any` codec 的解析状态和 parsed payload
- token usage

这个 JSON 可以和 Shaft 本机产物逐项 diff，尤其要确认 prompt hash、pixel budget、
generation 参数、finish reason 和 parser 状态一致。

## 必须对齐的推理参数

主检测任务建议：

- prompt：使用 `configs/prompts/pools/*.yaml` 中的 `main`
- `temperature=0`
- `top_p=1`
- `max_tokens=4096`
- `min_pixels=200704`
- 默认 `max_pixels=1000000`
- 精细定位或预标注场景可用 `max_pixels=2000000`

不要依赖 vLLM 自己处理业务 pixel budget；`min_pixels/max_pixels` 是 Shaft 发送请求前的
图像重采样契约。

## 常见问题

### 为什么只统一 vLLM 镜像还不够？

因为 vLLM 只负责模型服务。业务侧如果 prompt、图像 resize、采样参数或 JSON 解析方式不同，
仍然不是同一次推理。这个镜像提供 `shaft-contract-smoke`，就是为了把这些契约显式化。

### 模型要不要打进镜像？

默认不要。模型通过 volume 挂载，便于切 checkpoint，也避免镜像过大。生产环境可以按运维策略
做带模型的派生镜像，但必须保留同一套启动参数和 smoke 验收。

### 如何判断环境对齐？

至少满足：

1. `uv.lock` 与镜像构建使用的 lock 一致。
2. `shaft-contract-smoke` 记录的 prompt hash 一致。
3. pixel budget 输出尺寸一致。
4. generation 参数一致。
5. 同一图片的 raw output 或 parsed output 在可接受范围内一致。
