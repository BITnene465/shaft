---
name: shaft-model-quick-test
description: 在仓库根目录为外部/新模型快速搭建临时评测工作区：必要时下载上游源码仓库与工具依赖，收口为共享工具模块、批量脚本、独立 Gradio app、轻量单测与最小文档入口。适用于“先验证模型能力，不接训练主链”的请求。
---

# Skill：临时模型快速评测

## 触发场景
- 用户要“临时测试一个模型的能力”，重点是先验证效果，不接入 `src/shaft` 正式训练/推理主链。
- 需求通常包含其中几项：
  - demo 脚本
  - 批量推理脚本
  - 独立 Gradio app
  - 本地 `models/` checkpoint 路径约定
  - 上游源码仓库下载 / vendor
  - requirements / setup / 缺失工具依赖检查
  - 轻量 smoke test
- 任务对象通常是仓库外部模型或第三方仓库模型，如 `Rex-Omni`。

## 目标边界
- 临时评测工具放在仓库根目录，目录名统一为 `<model-slug>-test/`。
- 不把这类临时实验能力塞进 `src/shaft`、`scripts/train.py` 或正式 CLI/UI surface。
- 不复制一套训练、数据、checkpoint 语义。
- 产物统一落在 `outputs/<model-slug>-test/`。

## 首选做法
1. 先看当前仓库是否已有同类临时工具。
2. 如果任务和图像检测/可视化/批量评测相近，优先参考：
   - `rex-omni-test/rex_omni_tools.py`
   - `rex-omni-test/batch_infer.py`
   - `rex-omni-test/app.py`
   - `tests/test_rex_omni_tools.py`
3. 如果任务是 reconstruction 结果 review、render/overlay 临时可视化，读取
   [references/reconstruction-review.md](references/reconstruction-review.md)。
4. 只在需要更具体结构时再读 [references/layout.md](references/layout.md)。

## 固定结构
- `<model-slug>-test/<model_slug>_tools.py`
  - 共享能力收口点。
  - 放模型路径解析、依赖导入保护、批量遍历、结果 JSON 落盘、可视化导出。
- `<model-slug>-test/<UpstreamRepoName>/`（可选）
  - 当第三方模型依赖官方 wrapper、parser、visualizer 或自定义 utils 时，把上游源码 vendoring 到测试目录。
  - 优先去掉 `.git` 元数据，不把临时 clone 留在 `.tmp/`。
- `<model-slug>-test/batch_infer.py`
  - 薄 CLI。
  - 只做 argparse、参数转换、调用 tools。
- `<model-slug>-test/app.py`
  - 独立 Gradio app。
  - 复用 tools，不重写推理逻辑。
- `tests/test_<model_slug>_tools.py`
  - 只做轻量测试。
  - 不真实加载大模型。
  - 用 monkeypatch/mock 替代真实推理与可视化。

## 实施步骤
1. 选择目录名 `<model-slug>-test/`，保持和模型名显式对应。
2. 先判断是否需要上游源码和工具依赖：
   - 如果官方 demo 依赖 `from xxx import Wrapper, Visualize` 这类包装层，而本地只有权重目录，优先下载上游源码仓库。
   - 读取上游 `README.md`、`requirements.txt`、`setup.py`、`pyproject.toml`，确认最小可运行依赖。
   - 需要 vendoring 时，把源码放到 `<model-slug>-test/<UpstreamRepoName>/`，不要依赖 `.tmp/` 的临时 clone。
3. 在 tools 模块中先定义：
   - `ModelConfig` dataclass
   - `resolve_model_path()`
   - `collect_*()` / `load_*()` helpers
   - `infer_*()` 共享推理函数
   - `run_*_batch()` 批处理入口
4. 外部模型依赖必须做明确导入保护。
   - 缺包时报清晰错误。
   - 不要在 import 时静默失败。
   - 推荐导入顺序：
     1. 已安装的官方包
     2. `<model-slug>-test/<UpstreamRepoName>/` 下的 vendored 源码
     3. 本地 fallback 实现
5. 本地模型路径优先从 `models/` 自动发现。
   - 找到本地目录则用本地目录。
   - 否则回退到用户传入路径或远端 repo id。
6. 如果上游依赖额外工具库：
   - 先用 `importlib.util.find_spec()` 或等价方式检查是否已安装。
   - 再决定是提示缺失、写 fallback，还是在用户明确需要时安装。
   - 不要默认假设环境里已经有上游工具包。
7. 批量脚本与 Gradio app 必须共用一套推理/导出逻辑。
   - 禁止在 `app.py` 里重新写一遍 batch 逻辑。
8. 输出至少包含：
   - `summary.json`
   - `manifest.jsonl`
   - `json/*.json`
   - 可选 `visualizations/*.jpg`
9. 补最小测试：
   - 类别/参数解析
   - 本地模型路径发现
   - vendored / fallback 导入路径
   - 图片扫描
   - batch 输出落盘
   - app smoke
10. 如新增了新的根目录临时工具，在 `docs/module_reference.md` 附录补一句边界说明即可。

## 验收
- `.venv/bin/python -m compileall <model-slug>-test`
- `.venv/bin/pytest -q tests/test_<model_slug>_tools.py`
- 不跑真实大模型推理，除非用户明确要求。
- 不修改 `src/shaft` 正式内核，除非用户明确要求集成。
- 如果引入了 vendored 上游源码，确认导入路径稳定，不依赖临时目录或未记录的 shell 状态。

## GPU Runtime 排障
- 如果 `nvidia-smi` 显示 `[Not Found]`、残留 PID，或者 `kill <pid>` 返回 `No such process`，不要直接尝试 GPU reset、重启容器或宿主机介入。
- 第一时间用 `lsof` 查真实持有 NVIDIA 设备文件的当前命名空间进程：
  - `lsof /dev/nvidia*`
  - `lsof -t /dev/nvidia* | sort -u | xargs -r ps -o pid,ppid,pgid,sid,stat,cmd -p`
- 以 `lsof` 结果作为 GPU holder 排查真源，它能暴露 vLLM EngineCore/Worker、孤儿多进程子进程、`nvtop`、临时占卡脚本等 `nvidia-smi` 可能显示不清的进程。
- 清理顺序：
  1. 对明确属于本次任务的进程先普通 `kill`
  2. 短暂等待后复查 `lsof /dev/nvidia*` 和 `nvidia-smi`
  3. 仍残留时再对同一批 PID 使用 `kill -9`
- 不要杀 PID 1、当前 shell、无关服务或用户未授权的业务进程。必要时按进程组清理自己启动的 runtime 进程。
- 只有在 `lsof` 已确认没有可清理 holder、但显存仍异常占用时，才考虑宿主机侧 reset 或运维介入。

## 注意事项
- 这是“快速验证模型能力”的工作流，不是长期产品化入口。
- 如果已有相近目录，优先在其基础上复用/改造，而不是再发明一套新结构。
- 如果模型接口和 `Rex-Omni` 差异较大，也保留“共享 tools + 薄入口 + 轻量测试”的总结构，只替换模型专属调用部分。
- 如果用户明确只想“把模型放到 `models/` 就能测”，就要把源码/工具准备收口进测试目录或 fallback 逻辑，不能把安装上游仓库变成隐性前置条件。
