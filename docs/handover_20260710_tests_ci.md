# Handover: Shaft 测试与 CI/CD 功能目标重构

更新时间：2026-07-10 03:52 UTC

> 归档说明（2026-07-12）：本文保留 2026-07-10 交接时的工作树与风险快照，不再代表当前仓库状态。
> 其中测试分层与 CI 收口已由后续 `refactor(test): define explicit framework suites` 和
> `ci: establish layered framework gates` 完成；当前使用方式以 `docs/testing.md` 为准。

本文档用于把当前工作交给下一位 agent。重点是本轮测试/CI 重构，同时记录当前工作树中与之并行的其他改动、仍有效的用户决策和接手风险。

## 1. 当前仓库状态

- 仓库：`/root/workspace/shaft`
- 分支：`feat/eval-bench`
- HEAD：`f8f1630 ci: keep eval-bench compatibility check`
- 远端：`origin/feat/eval-bench` 也在 `f8f1630`
- 本轮测试/CI 重构尚未 commit，也未 push。
- 工作树很脏，包含多个并行主题。不要执行 `git add -A`，不要把所有 dirty 文件混成一个 commit。
- 用户允许使用本地代理。

最近与 CI 有关的已提交历史：

```text
f8f1630 ci: keep eval-bench compatibility check
1353a85 ci: include distributed contracts in framework gate
16acaa2 ci: install training extras for framework gate
11e2017 ci: split framework and eval-bench gates
```

历史 GitHub Actions 中，旧失败 run 会永久保留，后续提交不能“覆盖”旧 run。曾确认：

- `eval-bench` Run 7，run id `29008315863`，commit `f8f1630`：成功。
- `framework-ci` Run 3，run id `29006893214`，commit `1353a85`：成功。

## 2. 用户对测试体系的明确要求

用户不接受为了覆盖率或为了测试而测试。测试必须从真实功能目标出发，不能锁死实现细节。

核心要求：

- 参考 SWIFT 和 LLaMA-Factory 的测试与 CI 设计思想。
- 默认 CI 只保护 Shaft 训练框架主链。
- eval-bench 前端、浏览器布局、review 可视化、临时数据脚本不应阻塞训练框架合并。
- 不需要的测试和 CI 可以直接删除。
- 测试应覆盖 public API、配置语义、数据语义、训练/推理/评估/导出主链结果，而不是私有 helper、调用顺序或字段排列。

参考上游：

- SWIFT workflows: <https://github.com/modelscope/ms-swift/tree/main/.github/workflows>
- SWIFT test runner: <https://github.com/modelscope/ms-swift/blob/main/tests/run.py>
- LLaMA-Factory CPU tests: <https://github.com/hiyouga/LLaMA-Factory/blob/main/.github/workflows/tests.yml>
- LLaMA-Factory CUDA tests: <https://github.com/hiyouga/LLaMA-Factory/blob/main/.github/workflows/tests_cuda.yml>
- LLaMA-Factory pytest resource markers: <https://github.com/hiyouga/LLaMA-Factory/blob/main/tests/conftest.py>

可借鉴的原则不是复制上游代码，而是：

- 按 `data / model / train / eval / infer / export` 能力面组织测试。
- CPU 功能契约与 GPU/runtime/真实模型测试分开。
- 测试声明资源条件，由收集层统一选择或 skip。
- CI 选择功能套件，不无差别扫描整个测试目录。

## 3. 本轮已经完成的改动

### 3.1 默认 pytest 改为正向选择功能目标

`pyproject.toml` 新增 marker：

- `goal`
- `slow`
- `gpu`
- `task`
- `visual`
- `perf`

默认选择表达式改为：

```text
goal and not slow and not gpu and not smoke and not integration and not manual and not task and not visual and not perf
```

目标是结束原来的反向筛选：

```text
not smoke and not integration and not manual
```

旧表达式会让所有未分类测试自动进入 CI，这是之前测试范围失控的主要原因之一。

### 3.2 集中分类策略

新增/修改：

- `tests/conftest.py`
- `projects/eval_bench/tests/conftest.py`

当前规则：

- Shaft 根测试中，没有 `slow/gpu/task/visual/perf/smoke/integration/manual` 的测试会自动获得 `goal`。
- `test_build_grounding_structured.py`、`test_build_sft_from_structured.py` 被标记为 `task + manual`。
- `test_prediction_visualization.py` 被标记为 `visual + manual`。
- eval-bench dashboard 测试被标记为 `visual + manual`。
- eval-bench perf、Docker contract、worker runtime/vLLM 测试退出默认 goal 集合。

这套自动分类是第一阶段止血，不是最终完成态。详见第 6 节。

### 3.3 GitHub workflows 收口

`framework-ci.yml`：

- 成为唯一真实训练框架 gate。
- 所有 PR 都触发，避免 required check 因 path filter 不出现而长期 pending。
- `main` 和 `feat/**` push 触发。
- 安装 `dev + train + rlhf + distributed`。
- 编译 `src/shaft tests`。
- 只跑 framework functional goal tests。
- job context 从 `framework-contract` 改为 `framework-goals`。

`eval-bench.yml`：

- 保留 workflow 名和 `backend-contract` / `frontend-contract` job context。
- 两个 job 都是 no-op 兼容 shim。
- 目的只是兼容旧 branch protection，避免旧 required check 消失后阻塞合并。
- 不再安装依赖，也不运行 eval-bench backend/frontend 测试。

已删除：

- `.github/workflows/eval-bench-backend.yml`

### 3.4 删除或移动的测试

已删除以下明显绑定一次性 task script 的干净测试：

- `tests/test_build_point_line_structured.py`
- `tests/test_convert_grounding_structured_to_sft.py`
- `projects/eval_bench/tests/test_banana_benchmark_script.py`

已移动：

- `tests/test_eval_bench_vllm_adapter.py`
- 移到 `projects/eval_bench/tests/test_vllm_adapter.py`

注意：Git 当前显示为旧文件删除 + 新文件 untracked，因为移动操作已 unstaged，避免留下半 staged 状态。提交时要同时 stage 两边，Git 通常会识别为 rename。

没有删除以下带并行未提交修改的文件，只把它们移出默认 gate：

- `tests/test_build_grounding_structured.py`
- `tests/test_build_sft_from_structured.py`
- `tests/test_prediction_visualization.py`

### 3.5 文档

`docs/testing.md` 已重写为当前测试与 CI 真源，内容包括：

- functional goal tests 定义。
- marker 与默认命令。
- framework/eval-bench 的 CI 边界。
- 删测规则。
- 新测试必须回答的功能目标问题。

## 4. 本轮测试/CI 变更的文件范围

这些文件属于本轮测试/CI 重构，可以作为一个独立 commit 审查：

```text
D  .github/workflows/eval-bench-backend.yml
M  .github/workflows/eval-bench.yml
M  .github/workflows/framework-ci.yml
M  docs/testing.md
M  pyproject.toml
M  tests/conftest.py
A  projects/eval_bench/tests/conftest.py
D  projects/eval_bench/tests/test_banana_benchmark_script.py
D  tests/test_build_point_line_structured.py
D  tests/test_convert_grounding_structured_to_sft.py
D  tests/test_eval_bench_vllm_adapter.py
A  projects/eval_bench/tests/test_vllm_adapter.py
```

本 handover 文档也应单独 stage：

```text
A  docs/handover_20260710_tests_ci.md
```

## 5. 当前并行 dirty 变更：不要误删或混提

下列文件不是本轮测试/CI 重构的主体。它们来自用户或其他并行工作，接手者必须先阅读再决定如何提交：

```text
M  .codex/skills/shaft-project/shaft-data-manager/references/augmentation-grounding.md
M  .codex/skills/shaft-project/shaft-data-manager/references/counterintuitive-rules.md
M  .codex/skills/shaft-project/shaft-data-manager/references/derived-datasets.md
M  .codex/skills/shaft-project/shaft-data-manager/references/layout-grounding.md
M  .codex/skills/shaft-project/shaft-data-manager/references/prompt-policy.md
M  .codex/skills/shaft-project/shaft-model-quick-test/SKILL.md
M  projects/eval_bench/eval_bench/prediction_parser.py
M  projects/eval_bench/tests/test_prediction_parser.py
M  projects/eval_bench/tests/test_worker_vllm.py
M  scripts/tasks/arrow_keypoint_demo.py
M  scripts/tasks/build_grounding_structured.py
M  scripts/tasks/build_point_line_structured.py
M  scripts/tasks/build_reconstruction_from_gt_standard.py
M  scripts/tasks/build_sft_from_structured.py
M  scripts/tasks/convert_grounding_structured_to_sft.py
M  src/shaft/codec/__init__.py
M  src/shaft/metrics/builtin.py
M  src/shaft/metrics/prediction_visualization.py
M  tests/test_build_grounding_structured.py
M  tests/test_build_sft_from_structured.py
M  tests/test_eval_metrics.py
M  tests/test_prediction_visualization.py
?? .codex/skills/shaft-project/shaft-model-quick-test/references/reconstruction-review.md
?? scripts/tasks/prelabel_image_shape_attrs_bedrock.py
?? src/shaft/codec/coordinates.py
?? tests/test_qwen_coordinates.py
```

特别注意：

- `scripts/tasks/build_point_line_structured.py` 当前有未提交修改，但对应旧 task 测试已在本轮删除。不要因为测试删除就还原脚本。
- `src/shaft/codec/coordinates.py` 与 `tests/test_qwen_coordinates.py` 是一组未跟踪的新能力，应一起审查。
- `src/shaft/metrics/*`、`projects/eval_bench/eval_bench/prediction_parser.py` 与相关测试存在语义联动，不能拆开随意提交。

## 6. 重要未完成项和风险

### P0：逐文件审查当前 goal 集合

当前 `tests/conftest.py` 的策略是“未标为非 goal 的测试自动成为 goal”。这能防止 task/visual/runtime 混入默认 CI，但不能证明剩余测试真的都是功能目标测试。

下一位 agent 必须继续做第二阶段审计：

- 每个默认测试必须指出它保护哪个真实能力目标。
- 删除只断言私有 helper、内部字段排列、调用顺序、production 常量镜像的测试。
- 合并重复覆盖同一 contract 的测试。
- 不要通过补更多 marker 把问题掩盖过去。

更严格的最终方案可以二选一：

1. 把默认功能目标测试移动到明确目录，如 `tests/goals/`，CI 只跑该目录。
2. 保留现有目录，但要求每个文件显式声明 `goal`，collection 时拒绝未分类测试。

不要长期保留“所有未排除测试自动 goal”的策略，否则新实现细节测试仍可能自动进入 CI。

### P0：更新 GitHub branch protection

当前真实 required check 应为：

```text
framework-ci / framework-goals
```

旧的：

```text
eval-bench / backend-contract
eval-bench / frontend-contract
```

暂时通过 no-op shim 保留。branch protection 更新后，应删除 `.github/workflows/eval-bench.yml`，不要让假 gate 永久存在。

### P1：评估 distributed extra 成本

framework CI 当前安装 `distributed`，因为测试覆盖 DeepSpeed/FSDP 配置与 HF/TRL 参数契约。

不要直接删除该 extra；先确认这些 goal tests 是否真的需要 import/runtime 依赖。如果只需要 schema/参数转换，可将测试改成不加载 DeepSpeed runtime，从而降低 CI 安装成本和环境副作用。

### P1：进一步清理 eval-bench 测试

eval-bench 已退出 required CI，但其测试仍有大量 dashboard route、job runtime、perf 和 subprocess 场景。后续如果继续维护 eval-bench，应按 backend public contract 精简；前端和 dashboard 观感不应回到框架 gate。

## 7. 验证状态

在上一轮普通、非受限本地环境中，本轮改动曾通过：

```text
uv run python -m compileall src/shaft tests
uv run pytest -q tests
uv run python -m compileall projects/eval_bench/eval_bench projects/eval_bench/tests
uv run pytest -q projects/eval_bench/tests -m "goal and not slow and not manual and not task and not visual and not perf"
uv run ruff check tests/conftest.py projects/eval_bench/tests/conftest.py projects/eval_bench/tests/test_vllm_adapter.py
```

但之后工作树又出现了并行修改，例如 `tests/test_eval_metrics.py`。因此接手者必须基于当前工作树重新验证，不能只引用旧通过记录。

本 handover 生成时，当前 agent 运行在受限沙箱：

- `uv` 默认 `/root/.cache/uv` 不可写，需要临时设置 `UV_CACHE_DIR=/tmp/uv-cache`。
- framework 复跑在 `test_merge_peft_adapter_exports_full_layout` 处失败：DeepSpeed import 通过 `psutil` 查询受限 PID，得到 `psutil.NoSuchProcess`。这是当前 process namespace 限制，不能直接判定为代码回归。
- eval-bench goal 复跑有 16 个失败，全部在 `job_spec._port_available()` 创建 TCP socket 时得到 `PermissionError: Operation not permitted`。这也是沙箱限制。
- ruff 针对本轮 Python 文件重新运行并通过。
- 一次 framework 全量复跑因受限环境下异常拖长被手动终止；不要把它记为代码 hang，需在正常 shell/GitHub runner 复核。

建议在正常开发 shell 中运行：

```bash
uv run python -m compileall src/shaft tests
uv run pytest -q tests
uv run ruff check tests/conftest.py projects/eval_bench/tests/conftest.py projects/eval_bench/tests/test_vllm_adapter.py
uv run pytest -q projects/eval_bench/tests \
  -m "goal and not slow and not manual and not task and not visual and not perf"
```

workflow YAML 也应检查：

```bash
uv run python -c 'from pathlib import Path; import yaml; [yaml.safe_load(p.read_text()) for p in Path(".github/workflows").glob("*.yml")]'
```

## 8. 推荐接手顺序

1. 阅读 `AGENTS.md` 和 `docs/testing.md`。
2. 查看 `git status --short`，确认并行 dirty 文件是否又有变化。
3. 在正常环境重新跑 framework goal tests。
4. 按功能面逐批审计默认 goal：
   - config
   - data/collator/mixing
   - template/model
   - training/pipeline/checkpoint/export
   - infer/codec/metrics/online eval
   - webui shell
5. 每批先列出“真实功能目标 -> 保留测试”，再删除或改写实现细节测试。
6. 更新 branch protection 到 `framework-ci / framework-goals`。
7. 删除旧 `eval-bench` shim。
8. 只 stage 本轮测试/CI 文件，单独 commit 和 push。
9. push 后用 `gh run list` / `gh run view` 确认新 workflow，而不是看历史旧失败 run。

## 9. 功能目标测试应长什么样

推荐保留/补齐的主链场景：

- 配置：真实最小 YAML -> strict load/normalize -> 明确 RuntimeConfig 或明确错误。
- 数据：真实最小 multimodal sample -> template -> collator -> 最终 `input_ids/labels/attention_mask/pixel_values` 语义。
- 训练：tiny/fake model 完成 1 step -> checkpoint artifact 可恢复/导出。
- Packing：多个短样本打包后，sequence boundary、attention isolation、labels、position ids 和多模态 token 对齐正确。
- 推理评估：infer response -> shared codec -> metric -> aggregate，证明没有平行解析语义。
- 导出：标准 HF/PEFT artifact contract，不锁内部函数调用顺序。

应删除或降级的典型测试：

- 只断言某个私有函数被调用几次。
- 逐字段复制 production schema 或 manifest 常量。
- 为临时 review 页面检查像素颜色、label 位置或布局观感。
- 直接 import `scripts/tasks/*` 的一次性数据脚本测试。
- 真实 GPU、vLLM、网络服务、浏览器、性能计时，却没有显式 runtime/manual marker。

## 10. 本线程中仍有效的其他项目决策

这些不是本轮测试/CI 代码，但接手者应知道：

### 长上下文与 packing

- 当前短期目标是稳定支持 8k，不急着上 context parallel。
- 32k 是后续训练目标；当前 8k 已出现显存/效率问题，仍有优化空间。
- `max_length` 讨论口径是整个序列长度，即输入/prefix 与输出 token 合计，不只是输出 token。
- `prefix_length` 是动态的，截断和监督测试必须覆盖这一点。
- 用户认可 packing：短样本可以拼成一个 packed sequence，通过 attention mask/position boundary 隔离，teacher forcing 本身不是障碍。
- 任务长度差异很大，约 500 token 到 8k；同 batch 动态 padding 会造成真实计算浪费，不只是少量显存浪费。
- context parallel 暂缓，先把 packing、length grouping/bucketing、attention backend 和 batch 语义做正确。

### Reconstruction review/render/overlay

- render/overlay 是临时 review 能力，不应长期维护在 `scripts/tasks`。
- 长期只维护 skill/ref 中的规则；需要 review 时临时生成脚本/页面。
- render 需要高分辨率、透明背景，忠实渲染 border/fill/color、圆弧圆角、dash/dot、line corner style、P0/P1 风格。
- line 的 4 points 是三等分点，不是 Bézier 控制点。
- 圆角的 3 个点需要可视化，但标签不能遮挡图像或显示不全。
- line 点必须连线；`corner_style: round` 需要生效。
- overlay 使用 relaxed geometry，不能再次把 oval 放大。
- review JSON 只显示预测对象的 `parameters` 和 `type` 等必要部分，并允许滚动。
- 相关长期说明在：
  - `.codex/skills/shaft-project/shaft-model-quick-test/SKILL.md`
  - `.codex/skills/shaft-project/shaft-model-quick-test/references/reconstruction-review.md`

### 推理资源与数据

- 不要操作 `gpu-holder`。提高推理并发即可，holder 会自行让出资源。
- zero-shot/对比评测应使用真实 175 张测试集，不要拿合成数据替代。
- 用户已更新 Opus API key；本地代理允许使用。

## 11. 安全提交建议

不要使用 `git add -A`。建议显式 stage：

```bash
git add \
  .github/workflows/eval-bench-backend.yml \
  .github/workflows/eval-bench.yml \
  .github/workflows/framework-ci.yml \
  docs/testing.md \
  docs/handover_20260710_tests_ci.md \
  pyproject.toml \
  tests/conftest.py \
  projects/eval_bench/tests/conftest.py \
  projects/eval_bench/tests/test_banana_benchmark_script.py \
  tests/test_build_point_line_structured.py \
  tests/test_convert_grounding_structured_to_sft.py \
  tests/test_eval_bench_vllm_adapter.py \
  projects/eval_bench/tests/test_vllm_adapter.py
```

然后先检查：

```bash
git diff --cached --stat
git diff --cached
```

建议 commit message：

```text
refactor(test): gate CI on functional goals
```

是否把 handover 文档放进正式 commit 可由接手者决定；如果不希望长期保留 handover，可在交接完成后单独删除，不要影响测试/CI commit。
