# Shaft 测试与 CI 规范

本文档是 Shaft 测试分层、执行入口和 GitHub 门禁的真源。测试必须证明稳定的框架行为；
suite membership、资源需求和 GitHub required check 是三类不同语义，不能混在一组 marker 中。

## 1. 核心原则

- 默认测试只保护 Shaft 框架主链的稳定功能目标。
- 优先断言 public API、配置、CLI、artifact、数据语义和跨模块装配结果。
- 不长期保留只锁私有 helper、调用顺序、mock 次数、字段排列或 production 常量镜像的测试。
- CPU 功能契约、CPU smoke、distributed runtime、GPU、真实模型、临时 task、review visual 和
  Eval Bench 分开执行。
- CI 通过明确的文件 suite 选择测试，不能先 import 整个目录再靠 `pytest -m` 排除可选依赖。
- 新测试文件必须显式分类；当前存在的测试文件未分类或重复分类时，pytest 在 collection 前直接失败。

## 2. Suite 真源

Framework suite membership 的唯一真源是 `tests/conftest.py` 中的 `_SUITE_FILES`：

| Suite | 用途 | 默认/required CI |
| --- | --- | --- |
| `framework` | config/data/model/template/pipeline/training/infer/eval/export/webui CPU 功能契约 | 是 |
| `smoke` | tiny/fake SFT、DPO/PPO/GRPO、online eval 最短主链 | 是 |
| `distributed` | DeepSpeed/FSDP/torchrun runtime contract | 否，独立 workflow |
| `integration` | 真实模型、真实推理服务或仓库 fixture 主链 | 否 |
| `gpu` | CUDA kernel、FlashAttention 等 GPU runtime | 否 |
| `task` | 临时数据构建、迁移与 review task | 否 |
| `visual` | render/overlay/dashboard 观感检查 | 否 |

Eval Bench 使用 `projects/eval_bench/tests/conftest.py` 中的独立 manifest：

| Suite | 用途 | Required CI |
| --- | --- | --- |
| `backend` | backend public contract、store、CLI、metric、worker orchestration | 否 |
| `visual` | dashboard route/layout/review | 否 |
| `performance` | 性能 guard | 否 |
| `runtime` | Docker、worker subprocess、真实 vLLM | 否 |

Suite 负责“运行哪批文件”。pytest marker 仅补充描述 `unit/component/contract/smoke/integration/manual`
等测试属性，不再决定 required gate membership。

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

Eval Bench：

```bash
uv run pytest -q projects/eval_bench/tests --eval-bench-suite backend
uv run pytest -q projects/eval_bench/tests --eval-bench-suite runtime
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

真实 GPU、FlashAttention、真实模型与外部推理服务继续通过 `gpu/integration` suite 人工或专用 runner
执行，不在 GitHub-hosted CPU runner 上伪造通过。

### 4.3 Eval Bench

`.github/workflows/eval-bench.yml` 不是 branch protection required gate，只在相关路径变化或手动触发时：

- 安装 `dev + eval-bench` extras。
- 运行 Eval Bench ruff、compileall 与 `backend` suite。
- dashboard visual、performance、Docker/vLLM runtime 保持专项或人工验收。

## 5. Branch protection 迁移

门禁变更必须按以下顺序执行，避免 required context 消失：

1. 先推送包含 `framework-ci / required` 的提交并确认该 check 至少成功出现一次。
2. 在 GitHub ruleset/branch protection 中新增 `framework-ci / required`。
3. 删除旧 `framework-contract`、`framework-goals`、`eval-bench / backend-contract` 和
   `eval-bench / frontend-contract` requirements。
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
