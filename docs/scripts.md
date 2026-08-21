# Shaft 脚本使用说明

本文档说明仓库 `scripts/` 目录下的正式脚本如何使用。

范围约束：
- 覆盖顶层正式入口，以及 `scripts/tasks/` 与框架之间的职责边界
- 具体项目的数据版本、业务合同和构建基线与对应 task 脚本放在一起维护
- **不覆盖 `scripts/tmp/`**；`tmp` 目录视为临时实验区，不属于稳定接口

## 1. 设计原则

`scripts/*.py` 的定位是**薄入口**：
- CLI 解析与命令编排放在 `src/shaft/cli`
- `scripts/*.py` 只负责调用对应 CLI 主入口

`scripts/tasks/*.py` 是明确的离线数据准备入口，不是训练内核 CLI。

## 2. 顶层脚本

### `scripts/train.py`

用途：
- 统一训练入口
- 当前训练入口严格对应三个 domain：`sft`、`rl` 与 `opd`；不保留第二套 `rlhf` CLI

常用形式：

```bash
python scripts/train.py sft --config configs/train/sft_4b.yaml
python scripts/train.py rl --config configs/train/dpo_4b.yaml --algorithm dpo
python scripts/train.py opd --config /path/to/opd.yaml
```

兼容写法：

```bash
python scripts/train.py --config configs/train/sft_4b.yaml
```

说明：
- 如果直接传 `--config`，当前默认走 `sft`
- 真正的命令定义在 `src/shaft/cli`
- 入口会在导入训练栈前读取仓库根目录的可选 `.shaft.env`；可从 `.shaft.env.example` 复制，用于设置本机
  `CUDA_HOME` 等变量。文件中的值不会覆盖启动 shell 已显式设置的同名变量。
- 配置 `SHAFT_TRITON_CACHE_ROOT` 时必须使用节点本地绝对路径；入口会派生
  `<root>/<torchrun-run-id>/rank-<local-rank>` 并在 Triton 导入前写入 `TRITON_CACHE_DIR`。显式设置
  `TRITON_CACHE_DIR` 可覆盖该派生规则。

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

### `scripts/serve_opd_teacher.py`

用途：启动一个 immutable local-HF OPD teacher 的版本化 HTTP 服务。服务端配置必须选择
`opd.teacher.provider=hf_local`；训练端使用另一份 `provider=http` 配置并固定服务发布的 artifact SHA-256。

```bash
SHAFT_TEACHER_TOKEN='...' \
python scripts/serve_opd_teacher.py \
  --config /path/to/opd_teacher_server.yaml \
  --host 0.0.0.0 \
  --port 8100 \
  --api-key-env SHAFT_TEACHER_TOKEN
```

训练端对应配置：

```yaml
opd:
  teacher:
    provider: http
    model_type: qwen3vl
    remote:
      endpoint: http://teacher-host:8100
      artifact_fingerprint: <64-char-sha256-from-server-identity>
      api_key_env: SHAFT_TEACHER_TOKEN
```

服务只暴露 `/v1/identity` 与 `/v1/score`，请求/响应使用有界 safetensors envelope。服务端和训练端从环境
变量读取 token，不应把密钥写入 YAML。

OPD 使用独立 vLLM server rollout 时，先启动与 student base artifact 相同的 TRL 服务，再在训练配置设置
`opd.rollout.backend=vllm` 和匹配的 `opd.rollout.vllm.server_port/group_port`：

```bash
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model /path/to/student \
  --host 127.0.0.1 \
  --port 8000
```

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
  --model-type qwen3vl \
  --model-name-or-path org/model \
  --revision release-v2 \
  --cache-dir /path/to/hf-cache \
  --local-files-only true
```

```bash
python scripts/export.py merge-peft \
  --model-type qwen3vl \
  --adapter-path outputs/run_x/checkpoint-100 \
  --base-model org/model \
  --output-dir outputs/run_x/merged \
  --revision release-v2 \
  --cache-dir /path/to/hf-cache \
  --local-files-only true
```

`validate` 与 `merge-peft` 使用和训练相同的 HF locator 语义：`revision` 固定 Hub 版本，`cache-dir`
选择缓存根目录，`local-files-only=true` 禁止联网。adapter 的 base artifact 解析和模型 variant 选择均由
统一 `ResolvedModelPlan` 完成，CLI 不根据目录名猜测 dense/MoE。
`merge-peft` 默认要求 adapter 携带 Shaft 训练 metadata，并验证其中 base-model plan fingerprint。若导入的是
没有 Shaft provenance 的第三方 PEFT adapter，必须先独立核对 base，再显式增加
`--allow-unverified-base-model true`；该 escape hatch 不会关闭当前 base artifact 的完整 SHA256 校验。

### `scripts/compare_efficiency.py`

用途：比较多个已完成 run 的 committed `shaft_training_efficiency.json`，不负责启动实验。

```bash
python scripts/compare_efficiency.py outputs/fixed outputs/packed
python scripts/compare_efficiency.py --json outputs/fixed outputs/packed
```

默认要求模型 plan、数据/source、draw schedule、software/hardware、DP/GA、优化器、step span 与实际 committed
workload 一致，只允许 batch/sequence contract 作为实验轴变化。`--allow-incompatible` 仅用于明确接受非公平
条件的诊断结果，不能用于形成性能结论。


## 3. `scripts/tasks/`

`scripts/tasks/` 承载具体项目的可复现离线数据准备与转换，不是 Shaft 框架入口，也不代表框架能力。
任务脚本不得承载训练循环；其数据版本、构建命令、业务字段和完整性基线应与脚本放在一起维护，不进入
框架模块参考。具体任务说明由对应 task 目录内的 README 维护。

历史 weak-label 构建器 `build_drawio_shape_from_weak_labels.py` 只消费调用方显式指定的本地 job，不为
ignored `subTasks/` 路径提供默认值：

```bash
python scripts/tasks/build_drawio_shape_from_weak_labels.py \
  --weak-job-dir /path/to/weak-label-job \
  --output-root /path/to/output \
  --clean
```

`--weak-job-dir` 必须包含 `weak_labels.json`，可选 `job_manifest.json` 用于记录来源 job id。输入与 prompt
会在 `--clean` 删除旧输出之前检查；本地 subtask 目录及其临时测试不进入 Git。


## 4. 维护规则

新增脚本时，至少需要同步更新本文件，说明：
- 脚本用途
- 输入输出
- 关键参数
- 示例命令

如果脚本只是一次性临时实验，不应写进这里，而应留在 `scripts/tmp/`。
