# Handover: Shaft 测试与 CI/CD

更新时间：2026-07-16 UTC

本文档记录当前测试和 CI 主线。测试命令与 suite 语义以 `docs/testing.md` 为最终真源。

## 1. 当前方向

- 仓库继续以 HF-first 多模态训练/推理框架为主线。
- 当前优先保证 Qwen3VL/Qwen3.5/Qwen3.6 的 SFT、保存、恢复、导出和推理契约。
- 原 `projects/eval_bench` 独立评测工作台已从主线切除。它没有被 `src/shaft` 反向依赖，且维护了
  独立 evaluator/metric 数据流，不应与在线评估形成双真源。
- 分支名 `feat/eval-bench` 只是历史名称，不再表示当前树包含该产品。

切除独立工作台时保留以下共享能力：

- `src/shaft/codec`
- `src/shaft/metrics`
- `src/shaft/training/online_eval.py`
- `src/shaft/training/eval_dataloader.py`
- `src/shaft/training/eval_policy.py`
- `src/shaft/prompting.py`
- `src/shaft/utils/qwen_pixel_budget.py`
- `src/shaft/infer`、`serve` extra 与 `docker/inference`

这些模块均有训练、推理、GRPO 或在线评估消费者，不属于已切除的独立工作台。

## 2. 测试真源

`tests/conftest.py` 中的 `_SUITE_FILES` 是测试文件 membership 唯一真源。每个 `test_*.py` 必须恰好
属于一个 suite；marker 只描述测试属性，不决定 required gate。

```bash
uv run pytest -q tests --suite framework
uv run pytest -q tests --suite smoke
uv run pytest -q tests --suite distributed
uv run pytest -q tests --suite integration
uv run pytest -q tests --suite gpu
```

新增测试必须保护 public API、配置语义、最终 batch、artifact、metric、CLI 或主链装配结果。不要为
私有 helper、调用次数、当前实验 YAML 或一次性 task 脚本增加长期测试。

测试必须能在 clean clone 中独立运行：

- 不依赖未跟踪的 `models/`、本地实验配置、Hub cache 或外网；模型 descriptor 使用 `tmp_path` fixture。
- optional backend/package 的单测显式模拟 package metadata，并保留缺依赖 fail-closed 负例。
- `torchrun` support script 的启动器必须显式提供仓库根目录 import contract，不能依赖 editable 环境偶然
  把 `tests.support` 放入 `sys.path`。

## 3. CI 分层

### `framework-ci`

- 对所有 PR、`main` push 和手动触发运行。
- 执行 lock、ruff、compileall、workflow YAML parse 和 wheel build。
- 执行 `framework` 与 `smoke` suites。
- 稳定 required context 是 `framework-ci / required`。

### `framework-runtime`

- 非 required、按训练相关路径触发。
- 安装 distributed extra 并执行 `--suite distributed`。
- 同机 Gloo 多 launcher 只能验证 rank/topology/checkpoint 合同，不能冒充真实跨主机 NCCL 验收。

### `release`

- 手动触发只构建和校验 package artifact。
- `v*` tag 才创建 GitHub Release。
- 当前不发布 PyPI。

## 4. 当前验证边界

CPU 回归已经覆盖：

- framework 和 smoke 主链
- 同机 Gloo distributed contract
- Qwen3.5/Qwen3.6 tiny MoE train/save/exact-resume/HF reload
- lockfile、workflow YAML、wheel build

合并主线前仍需基于最终冻结 SHA：

1. 让远端 `framework-ci / required` 与 `framework-runtime / distributed-contracts` 成功运行。
2. 补跑当前 HEAD 的 CUDA Qwen release gates。
3. 如果对外承诺真实多机能力，再增加双主机 NCCL/NIC/共享存储 canary；否则在能力矩阵明确标记未验收。

## 5. 当前收口状态

- `allow_unverified_base_model` 与 `ShaftInferAdapterCapabilities` 已使用 exact-bool 校验，并覆盖字符串、
  整数和 `None` 的拒绝测试。
- `docs/development_log.md` 与 `docs/todo.md` 已恢复为仓库真源；本轮没有修改 `.gitignore`。
- 首次 `main` clean-runner 暴露的本地模型目录、optional dependency 和 `torchrun` import 隔离问题已修复；
  最终结论仍以修复 commit 对应的远端 `framework-ci` 与 `framework-runtime` 终态为准。

## 6. 协作与提交

- 不操作或终止 `gpu-holder`；需要 GPU 时只启动明确有界的 canary。
- 配置文件可能按 `.gitignore` 保持本地，不要为了测试强制加入 Git。
- 提交前使用显式路径核对 scope，不使用无差别 `git add -A`。
- 正常 `git push` 即可，不依赖 `gh`。
