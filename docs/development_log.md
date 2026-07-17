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

## 2026-07-17：v5.3 context reconstruction 与派生数据安全收口

### 现象

- v5.3 需要从既有 selection 恢复 shape/line/image 上下文重建样本，同时模拟有界 detector proposal
  偏差，并让 proposal 与 target geometry 共用当前 crop 的 Qwen `0..999` 坐标系。
- 真实 shape attribute weak label 中曾出现 prompt 合同外的嵌套字段；只检查 required value 会把
  `border.color2` 等 teacher extra 原样带入监督目标。
- grounding padding 在原图尺寸未按 processor factor 对齐、但又无需降采样时，metadata 使用了 floor
  后的 content size，实际像素却保持原尺寸，导致图像与 bbox 错位。现有 v5.3 派生数据中约576条
  padded row 命中该分支。
- 部分 task 测试曾导入 ignored `subTasks/` 脚本或依赖本地 ignored prompt pool；开发机可通过，clean
  checkout 无法复现。若 converter 在加载 prompt 前执行 `--clean`，缺文件还会先删除旧 SFT。

### 根因

- selection identity、source truth、weak-label truth 和 derived audit metadata 的边界此前没有统一到一个
  fail-closed builder 合同。
- padding 的 factor 对齐同时改变了声明 content geometry，却只在超预算分支执行真实 resize。
- prompt 版本通过脚本默认值静默切换，输入/split/path 校验发生在 destructive clean 之后。
- bbox minimum extent 最初由 task 脚本手工修补，未复用共享 coordinate codec。

### 影响范围

- raw 图片和人工 JSON 未被本轮代码修改；错位影响只在可重建的 grounding derived PNG/structured/SFT。
  受影响 padded rows 在继续训练前必须 clean rebuild，不能沿用旧派生结果。
- `image_context_reconstruction` 当前 selection 还有75个 source JSON 不在现行 `data/raw/json`，影响447行。
  builder 会明确失败并保留旧输出；这是本地数据版本可重建性问题，不允许静默回退 archive 真值。
- weak labels 只用于 train-only 辅助任务，不成为正式 eval truth。

### 修复方式

- 新增维护入口 `scripts/tasks/build_context_reconstruction_sft.py`：按 source 分组解码，重载 source truth，
  生成确定性 context crop/proposal、task-local media/selection/structured/SFT，并通过同盘 staging 原子发布。
- 新增共享 `shaft.data.context_attribute_contract.validate_shape_parameters`；weak sidecar consumer 执行 exact
  nested-key 与 API provenance gate，拒绝空 selection、非正 `--limit` 和缺少 provenance 的 weak rows。
- padding 仅在实际超预算时调整 content size并执行 resize；原生预算可容纳时保持原图像素和 bbox 尺寸。
- SFT converter 恢复 tracked v5.0 默认值，v5.3 使用显式 `--prompt-config TASK=PATH`；prompt 和全部
  structured split 在 `--clean` 前预检。v5.3 prompt pools继续按项目策略保持 local/ignored。
- grounding builder 在 clean 前检查 train/val overlap、缺失 GT、workers 和 raw/output path；synthetic sync
  builder拒绝输入输出路径重叠并记录真实 split provenance。
- `quantize_qwen_bbox` 增加可选 `minimum_extent_bins`，grounding/context bbox 使用共享1-bin最小尺寸；像素级
  zero-area bbox直接拒绝，量化碰撞再按 `label+bbox_2d` 稳定去重。
- tracked task tests全部使用测试内最小 prompt fixture，不再读取 ignored配置或本地子项目脚本。

### 回归测试

- `ruff check src/shaft scripts/tasks tests`：通过。
- `python -m compileall -q src/shaft scripts/tasks tests`：通过。
- `pytest -q tests --suite task`：通过。
- `pytest -q tests --suite framework`：通过。
- focused 回归覆盖非对齐 native padding、低预算 resize、clean 前置预检、split/path 重叠、zero/empty guard、
  weak exact-schema/provenance、共享 codec edge extent 和 worker failure旧输出保留。
- 真实 clean weak sidecar 19,709/19,709 行通过新增 provenance gate。

### 后续防线

- grounding v5.3 必须先重建受影响的 padded/structured/SFT 后再训练，并重新核对图片引用、bbox、row count、
  test overlap 和 stale files；raw 不需要回写。
- image context 重建前必须恢复或版本化缺失的75个 source JSON；不得因为当前 raw 缺失而自动把 archive
  当作真值。
- prompt pool可以是本地运行资产，但 tracked tests只能使用最小 fixture；任何 destructive clean 都必须在
  所有输入、schema、split 和路径预检完成后执行。
- context builder 当前会物化 selections/work items；269,904 shape rows实测增加约214MB RSS，属于后续流式化
  的 P2 优化，不阻断当前离线构建。
