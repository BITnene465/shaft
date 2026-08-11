# Shaft 测试与 CI 规范

本文档是 Shaft 测试分层、执行入口和 GitHub 门禁的真源。测试必须证明稳定的框架行为；
suite membership、资源需求和 GitHub required check 是三类不同语义，不能混在一组 marker 中。

## 1. 核心原则

- 默认测试只保护 Shaft 框架主链的稳定功能目标。
- 优先断言 public API、配置、CLI、artifact、数据语义和跨模块装配结果。
- 不长期保留只锁私有 helper、调用顺序、mock 次数、字段排列或 production 常量镜像的测试。
- CPU 功能契约、CPU smoke、distributed runtime、GPU、真实模型、临时 task 和 review visual
  分开执行。
- CI 通过明确的文件 suite 选择测试，不能先 import 整个目录再靠 `pytest -m` 排除可选依赖。
- 新测试文件必须显式分类；当前存在的测试文件未分类或重复分类时，pytest 在 collection 前直接失败。

## 2. Suite 真源

Framework suite membership 的唯一真源是 `tests/conftest.py` 中的 `_SUITE_FILES`：

| Suite | 用途 | 默认/required CI |
| --- | --- | --- |
| `framework` | config/data/model/template/pipeline/training/infer/eval/export CPU 功能契约 | 是 |
| `smoke` | tiny/fake SFT、DPO/PPO/GRPO、online eval 最短主链，以及轻量 rank-zero console 契约 | 是 |
| `distributed` | DeepSpeed/FSDP/torchrun runtime contract，含同机双 agent 多节点 Gloo | 否，独立 workflow |
| `integration` | 真实模型、真实推理服务或仓库 fixture 主链 | 否 |
| `gpu` | CUDA kernel、FlashAttention 等 GPU runtime | 否 |
| `task` | 临时数据构建、迁移与 review task | 否 |
| `visual` | render/overlay/dashboard 观感检查 | 否 |

Suite 负责“运行哪批文件”。pytest marker 仅补充描述 `unit/component/contract/smoke/integration/manual`
等测试属性，不再决定 required gate membership。

终端进度能力的 smoke 必须使用真实 PTY，并重放 carriage return / erase-line
后的最终屏幕，证明刷新过程中只有一个活动进度行，不会残留历史 frame。多进程
console 契约至少需要一条 CPU/Gloo 2-rank 用例进入 required smoke，确保只有 rank 0
创建 interactive sink，非零 rank 的日志不泄漏到公共 console。

## 3. 推荐命令

默认 framework 回归；`tests/conftest.py` 未指定 suite 时默认选择 `framework`：

```bash
uv run pytest -q
uv run pytest -q tests --suite framework
```

CPU 主链 smoke：

```bash
uv run pytest -q tests --suite smoke
```

Distributed、integration 和 GPU：

```bash
uv run pytest -q tests --suite distributed
uv run pytest -q tests --suite integration
uv run pytest -q tests --suite gpu
```

Task 与 visual：

```bash
uv run pytest -q tests --suite task
uv run pytest -q tests --suite visual
```

显式指定单个测试文件或使用 `-m` 时仍允许直接 collection，便于 focused 调试：

```bash
uv run pytest -q tests/test_collator.py
uv run pytest -q -m integration
```

静态与 lock 检查：

```bash
uv lock --check
uv run --locked ruff check src/shaft tests
uv run --locked python -m compileall -q src/shaft tests
```

## 4. GitHub Actions

### 4.1 Required framework gate

`.github/workflows/framework-ci.yml` 对所有 PR、`main` push 和手动触发运行：

- `uv lock --check`，拒绝过期 lockfile。
- 使用 `uv sync --locked --extra dev --extra train --extra rlhf` 创建不含 DeepSpeed 的 core 环境。
- 运行 ruff、compileall 和 workflow YAML parse。
- 构建 wheel，验证标准 Python package artifact。
- 运行 `framework` 与 CPU `smoke` suites。
- 保存 JUnit XML artifacts，所有 job 都有明确 timeout。
- 最终 `required` job 汇总内部 required jobs。

Branch protection 只绑定稳定 context：

```text
framework-ci / required
```

内部 job 可拆分或改名，但不得随意改变这个外部门禁名称。

Required workflow 不使用 PR path filter。GitHub 在整个 required workflow 因 path filter 未触发时，
可能让 required check 长期 pending。非 required workflow 才按路径过滤。

### 4.2 Distributed/runtime

`.github/workflows/framework-runtime.yml` 是非 required 的 focused workflow：

- 安装 `distributed` extra。
- 运行 `--suite distributed`。
- 覆盖 DeepSpeed 配置激活、global state 清理、TRL 参数传递和 CPU torchrun canary。
- 用两个独立 static-rendezvous launcher 模拟 `2 nodes x 2 processes`，覆盖 global/local/group rank 轴、
  bounded SFT committed checkpoint 与 `full_determinism + DDP static_graph` bitwise exact resume；该测试不
  声称覆盖真实跨主机网络、NCCL 或节点文件系统。

真实 GPU、FlashAttention、真实模型与外部推理服务继续通过 `gpu/integration` suite 人工或专用 runner
执行，不在 GitHub-hosted CPU runner 上伪造通过。

Qwen3.5/3.6 MoE padded SFT 已进入 tiny-upstream validated 状态。下面的 CPU 用例回归 Transformers MoE
class、真实 processor、router objective、full-finetune、exact resume 与 HF reload：

```bash
CUDA_VISIBLE_DEVICES='' uv run pytest -q \
  tests/test_integration_qwen_standard.py::test_qwen35_qwen36_moe_cpu_train_save_exact_resume_and_hf_reload
```

该用例覆盖两 rank Gloo 的多模态 forward/backward、full-finetune、committed save、
`full_determinism + DDP static_graph` exact resume、标准 HF reload，以及 Qwen3.6 alias 对导出目录的验证。
测试故意令 `save_steps=1`、`logging_steps=2`，从 logging window 中间的 checkpoint 恢复，并比较最终
`log_history/on_log` 与 MoE `aux/*` 日志。两 rank 的 pending loss 必须不同，恢复后的 root `train_loss`
还要等于相邻 checkpoint 全局累计 loss 的差值，防止 rank-0 baseline 或最终摘要产生假阳性。
它不覆盖真实 35B 权重的显存、吞吐或长程数值，不能单独作为生产容量声明。

Qwen 训练 release gate 需要本地 `models/Qwen3-VL-2B-Instruct`、`models/Qwen3-VL-4B-Instruct`、
`models/Qwen3.6-27B` processor 资产和两张可用 CUDA 卡，显式执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 \
  uv run pytest -q tests/test_integration_qwen_standard.py \
  -k '2b_two_rank_fp16 or two_rank_train_save_and_exact_resume_release_gate or multi_rank_lora_varlen_and_export_release_gate or sharded_backend_release_gate'
```

它覆盖真实 Qwen3VL-2B FP16 AMP + FP32-load padded LoRA 的 8-step fresh、checkpoint-4→step-8 exact
resume、GradScaler state、非零更新与双侧 export/reload，以及真实 Qwen3VL-4B LoRA greedy-varlen 的
fresh/checkpoint resume、PEFT validate、标准
`PeftModel.from_pretrained` + export processor forward 与 adapter reload，以及 tiny upstream Qwen3.5 dense
architecture 的 Qwen3.5 fixed padded、Qwen3.6 greedy-varlen fresh/save/exact-resume 和最终 HF
processor+model forward。exact-resume gate 比较模型/adapter、optimizer、scheduler、每 rank RNG、
  Trainer/stateful sampler、DDP `committed_manifest` validator 及 batch-planning extension，以及
  transaction/manifest generation
  交叉验证后可由 2-rank runtime 完整恢复的 telemetry workload/span；wall-clock efficiency 字段不参与
  bitwise 比较。所有 gate 显式启用
`full_determinism`。资源守护进程只负责让卡，不得由测试或操作人停止、重启、发送信号或修改配置。

greedy-varlen packing 的相同 release gate 可扩为四卡；验证器会从 checkpoint efficiency transaction 读取
world size，不允许用固定两 rank 的恢复器误验多 rank snapshot：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 SHAFT_RUN_QWEN_TRAIN_RELEASE_GATE=1 \
  SHAFT_QWEN_PACKING_WORLD_SIZE=4 uv run pytest -q \
  tests/test_integration_qwen_standard.py::\
test_qwen3vl_multi_rank_lora_varlen_and_export_release_gate -s
```

OPD 的发布权重门禁使用真实 Qwen3VL-2B student 与 Qwen3VL-4B frozen teacher，和 SFT gate 分开显式执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 SHAFT_RUN_QWEN_OPD_RELEASE_GATE=1 \
  uv run pytest -q tests/test_integration_qwen_standard.py::\
test_qwen3vl_opd_release_weights_single_and_two_rank_exact_resume_gate -s
```

该门禁先用两张有序图片执行单卡 BF16 LoRA generate/score/backward、非零 adapter 更新与 PEFT validate，
再用单图执行两卡 DDP、GA=2 的两步 fresh 和 checkpoint-1→step-2 resume。验收逐项比较 adapter、optimizer、
scheduler、每 rank RNG、Trainer state 与最终 export，并使用 sampled rollout 验证 generation RNG 恢复。
该门禁已于 2026-08-11 在 CUDA 0、1 上通过。CPU tiny/两进程 Gloo 证据仍不能替代它；后续重跑也必须
等待两张卡无用户训练进程，禁止抢占训练任务或操作 `gpu-holder`。单卡子进程只暴露第一张卡，外层 pytest
和双卡 DDP 子进程暴露两张卡，避免误入单进程 DataParallel。

OPD 的 vLLM server 与 sharded backend 使用两个独立 manual gate：

```bash
CUDA_VISIBLE_DEVICES=0,1 SHAFT_RUN_QWEN_OPD_VLLM_GATE=1 \
  uv run pytest -q tests/test_integration_qwen_standard.py::\
test_qwen3vl_opd_vllm_server_rollout_release_gate

CUDA_VISIBLE_DEVICES=0,1 SHAFT_RUN_OPD_SHARDED_GATE=1 \
  uv run pytest -q tests/test_smoke_distributed.py \
  -k 'opd_backend_native_exact_resume_gate'
```

vLLM gate 在 CUDA1 启动真实 Qwen3VL-2B TRL server，在 CUDA0 装配 2B LoRA student 与 4B local teacher，
覆盖 weight sync、未展开/展开多模态 prompt 对齐、rollout、score、backward 和 telemetry。sharded gate 分别
覆盖两卡 FSDP 与 DeepSpeed ZeRO-3 tiny fresh/resume/export，并验证每 rank OPD telemetry 连续历史。
两者只终止自身启动的进程组，禁止操作 `gpu-holder`。

完整四卡验收使用三种互补拓扑：

```bash
# 4-rank DDP student + 真实 HTTP teacher；top-k-tail/chunk、telemetry、fresh/resume
CUDA_VISIBLE_DEVICES=0,1,2,3 SHAFT_RUN_OPD_FOUR_GPU_GATE=1 \
  uv run pytest -q tests/test_smoke_distributed.py::\
test_torchrun_opd_four_gpu_external_teacher_objective_telemetry_exact_resume_gate -s

# 4-rank FSDP 与 4-rank ZeRO-3，分别执行 fresh/resume/export
CUDA_VISIBLE_DEVICES=0,1,2,3 SHAFT_RUN_OPD_SHARDED_GATE=1 \
  SHAFT_OPD_SHARDED_WORLD_SIZE=4 uv run pytest -q \
  tests/test_smoke_distributed.py::test_torchrun_opd_backend_native_exact_resume_gate -s

# CUDA0-2: 3-rank DDP Qwen student/local teacher；CUDA3: 独立 TRL vLLM server
CUDA_VISIBLE_DEVICES=0,1,2,3 SHAFT_RUN_QWEN_OPD_VLLM_GATE=1 \
  SHAFT_OPD_VLLM_TRAIN_WORLD_SIZE=3 SHAFT_OPD_VLLM_TRAIN_GPUS=0,1,2 \
  SHAFT_OPD_VLLM_SERVER_GPU=3 uv run pytest -q \
  tests/test_integration_qwen_standard.py::\
test_qwen3vl_opd_vllm_server_rollout_release_gate -s
```

第一条的 HTTP teacher 服务在 CPU 上运行，证明训练进程不加载 teacher；它不是“Qwen teacher GPU 独立部署”
的容量证据。三条门禁和四卡 packing gate 均已于 2026-08-11 在 CUDA0–3 通过。

MoE 是同一 release-gate 组内的独立 experimental capability gate，另外验证：DDP padded LoRA 与 DDP
varlen full、FSDP padded fused-parameter LoRA、DeepSpeed
ZeRO-3 padded full；FSDP/ZeRO-3 gate 使用 GA=2，FSDP 同时打开 model-side gradient checkpointing，避免
只证明最简单的 GA=1/no-recompute 组合。fresh 与 checkpoint-1→step-2 resume 的最终权重、
optimizer/scheduler、每 rank RNG、
Trainer state 与恢复后的 telemetry 必须语义一致。full finetune 继续比较 backend-native model/optimizer
state；FSDP+PEFT 则以完整标准 PEFT adapter 作为逻辑模型真源，只把 backend-native optimizer 等训练状态
纳入比较。router/expert/ordinary LoRA 或 full weights 必须确实更新，
fresh/resumed export 均要由标准 HF/PEFT loader 回载。这里的确定性状态包括 `total_flos`：模型参数量必须
在 distributed wrapping 前固定，不能随 ZeRO-3 checkpoint gather 后的 live shard `numel()` 漂移，也不能从
Trainer-state 等价检查中直接忽略。`ZeRO-3 + target_parameters`、MoE QLoRA 与预量化 FP8
训练属于明确负例。上述 Qwen3.5/3.6 证据仍是 tiny architecture release gate，不替代真实 35B
checkpoint canary。
gate 还会断言 efficiency contract 中 GA=2、两步对应的 epoch 几何；Trainer 在进入 inner loop 前验证
`model.is_gradient_checkpointing` 已真实激活，因此 FSDP 配置不能被静默忽略。

Qwen3VL 30B-A3B 的真实 MoE SFT 门禁单独执行，避免日常 integration 意外占用两张 80GB GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  SHAFT_RUN_QWEN3VL_MOE_30B_GATE=1 \
  uv run pytest -q tests/test_integration_qwen_standard.py::\
test_qwen3vl_30b_moe_two_rank_fsdp_lora_release_gate -s
```

该门禁使用真实 `models/Qwen3-VL-30B-A3B-Instruct`、BF16、显式 `grouped_mm`、padded 单图、FSDP
full-shard、model-side gradient checkpointing 和 fused-parameter LoRA。它执行两步 fresh，再从
checkpoint-1 在独立 output 目录恢复到 step 2，比较标准 adapter、FSDP optimizer、scheduler、RNG 与
Trainer state，
并显式使用 `warmup_ratio=0`，要求 checkpoint-1 已发生非零更新且 checkpoint-1/2 的全部 adapter tensor
逐项变化；同时核对 adapter 参数总数、finetune trainable 参数数与 optimizer 参数数完全一致。门禁从模型
config 推导层数，检查 48 个 language/router/expert 层与 27 个 vision block 的 LoRA 均有非零更新，最后由
标准 HF+PEFT 重载并验证
48 层×128 experts 的有限 forward。该门禁证明真实 30B MoE LoRA SFT 链可训练，不证明两卡全参数 SFT
容量或长程目标数据收敛。

FSDP+PEFT checkpoint 中的 `pytorch_model_fsdp.bin` 是 Transformers/Accelerate adapter-only 保存路径产生的
rank-local DTensor 派生物，不是完整逻辑 adapter，门禁不得把它当作模型恢复真源。SFT resume 会在 FSDP
wrap 前从已解析、已绑定 size/SHA-256 的标准 PEFT artifact 完整预载每个 rank，再跳过该 native 模型文件；
optimizer、scheduler、scaler、RNG 和 Trainer state 仍走 backend-native 恢复。该组合只允许
`state_dict_type=full_state_dict` 且要求 `load_best_model_at_end=false`。

## 5. Branch protection 迁移

门禁变更必须按以下顺序执行，避免 required context 消失：

1. 先推送包含 `framework-ci / required` 的提交并确认该 check 至少成功出现一次。
2. 在 GitHub ruleset/branch protection 中新增 `framework-ci / required`。
3. 删除旧 `framework-contract` 和 `framework-goals` requirements。
4. 确认新 PR 只依赖一个稳定 framework context。

## 6. 新增与删改测试

新增测试前必须回答：

1. 它保护哪个稳定功能目标？
2. 调用者能否从 public API、最终 batch、artifact、metric 或 CLI 结果感知该行为？
3. 是否已有更高层 contract 覆盖同一语义？
4. 它需要哪个 suite 和哪些可选依赖？

新增 `test_*.py` 后必须在对应 conftest manifest 中恰好登记一次。新增 framework 内核能力至少需要：

- 一条 focused contract/component 测试。
- 涉及装配链时的一条 CPU smoke 或资源匹配的 runtime/integration 测试。

以下测试应删除或改写：

- production schema/manifest/常量的逐字段镜像。
- 扫描 `configs/` 或绑定某个实验 YAML 的文件名、权重、学习率、step 数等当前训练配方；配置 parser/schema/
  normalize 必须使用测试内构造的最小 fixture 验证。训练配方是否可用由启动前 config validation/canary 负责，
  不能让本机未追踪配置隐式进入 CI。
- 只验证 FlashAttention、Transformers 等第三方 kernel/model 能否独立运行，且不经过 Shaft adapter/policy 的
  环境诊断；只有穿过 Shaft 公共边界并验证框架语义的 runtime gate 才属于本仓库测试。
- 已被同文件真实 build/resolve/dispatch 路径完全支配的 `REGISTRY.has(...)` 或 keys 枚举断言。
- 只断言私有 helper 或内部调用顺序。
- 对同一调用链多层 patch 后只检查 mock 次数。
- 临时 task、visual、性能或真实服务测试进入 `framework` suite。

## 7. Release/CD 边界

`.github/workflows/release.yml` 提供标准 Python package 的最小 CD：

- `workflow_dispatch` 只运行 lock、framework/smoke、wheel/sdist build、干净 venv install/import 校验，
  并保存 package artifacts，不发布版本。
- `v*` tag 必须与 `pyproject.toml` 的 `project.version` 完全一致，验证通过后创建或更新对应
  GitHub Release。
- workflow 只在 tag-only `publish` job 获得 `contents: write`，package 和其他 workflow 保持只读。
- 当前不发布 PyPI。配置 GitHub trusted publishing/environment 审批后才能增加 registry publish job；
  发布 token 不得进入仓库、命令行参数或测试日志。
