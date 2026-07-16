# Shaft 开发日志

本文档记录已经暴露过的工程问题、语义偏差和后续防线。每条记录至少包含现象、根因、影响范围、
修复方式、回归测试和后续防线；结论不能只保留在聊天、临时日志或本机环境中。

历史开发日志在 squash 前的仓库历史中仍可审计；本文件从当前 HF-first 主线继续维护。

## 2026-07-16：本地环境掩盖 clean-runner 测试依赖与布尔契约缺口

### 现象

最终 squash commit 在开发机上通过 framework、smoke 和 distributed 回归，但首次推送 `main` 后：

- `framework-ci` 有 7 个失败；
- `framework-runtime` 有 4 个 distributed 失败；
- 两个公开运行时边界允许非布尔 truthy 值进入安全相关能力判断。

### 根因

测试隔离问题有三类：

1. 部分测试写死 `models/Qwen3-VL-4B-Instruct`、`models/Qwen3.6-27B`。开发机存在未跟踪模型目录，
   clean runner 不存在时，相对路径被当作 Hub repo 并触发远程 config 解析。
2. Qwen3.5/Qwen3.6 varlen 正路径测试依赖开发机已安装的 CUDA isolation kernels；GRPO vLLM 测试使用
   `dict.get(key, distribution_version(key))`，其默认参数会被 eager 求值，导致 clean runner 查询未安装的
   vLLM distribution。
3. distributed support 脚本以文件路径交给 `torchrun` 后，子进程的 `sys.path` 只保证脚本目录，未保证
   仓库根目录；开发机 editable 环境掩盖了 `tests.support` 无法导入的问题。

另外两个生产边界使用了 Python truthiness：

- `allow_unverified_base_model="false"` 会被当作真值，跳过 adapter/base provenance 验证；
- `ShaftInferAdapterCapabilities` 接受字符串形式的 capability，可能绕过 execution-control fail-closed。

### 影响范围

- 7+4 个 CI 失败属于测试环境契约缺失，不表示 Qwen、FSDP、GRPO 或 DDP 主链本身失效。
- provenance override 与 infer capability 属于真实运行时边界缺口；配置对象或第三方调用方传入非布尔值时
  可能错误放宽安全约束。
- 本次问题不涉及模型能力，也不属于 eval、codec、metric 或 data 误判。

### 修复方式

- 测试统一用 `tmp_path/config.json` 生成最小本地 HF descriptor，不依赖未跟踪模型资产或网络。
- 可选 CUDA kernel 与 vLLM distribution 在测试中通过已有 seam 显式模拟；生产运行时仍保持缺依赖即拒绝。
- distributed 测试启动器集中构造子进程环境，把仓库根目录前置到 `PYTHONPATH`，同时保留调用方原值，
  并继续隐藏 CUDA。
- adapter provenance override 和 infer adapter capability 使用 exact-bool 校验；字符串、整数和 `None`
  均在执行工作前失败。

### 回归测试

- 原 7 个 framework 失败用例与新增缺 kernel 负例：全部通过，并在
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 下复验。
- 原 4 个 distributed 失败用例：全部通过。
- `ruff check src/shaft tests`：通过。
- `python -m compileall -q src/shaft tests`：通过。
- `pytest -q tests --suite framework`：通过。
- `pytest -q tests --suite smoke`：通过。
- `pytest -q tests --suite distributed`：通过。

### 后续防线

- framework/smoke 测试不得依赖未跟踪的 `models/`、`configs/`、缓存目录或外网可达性。
- 测试可选后端时应模拟 package metadata，并保留独立的缺依赖 fail-closed 负例；真实 kernel correctness
  只在对应 GPU suite 验证。
- 任何 `torchrun` 测试 helper 必须显式建立 support-module import contract，不能依赖 editable install 的偶然路径。
- 安全豁免、能力声明、checkpoint/provenance override 等布尔边界必须使用 exact-bool 校验，禁止用
  `bool(value)` 解释外部输入。
- 最终候选必须在 clean runner 上通过 required CI；本机全绿不能替代远端门禁。
