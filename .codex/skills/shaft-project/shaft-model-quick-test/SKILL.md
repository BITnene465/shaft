---
name: shaft-model-quick-test
description: 在仓库根目录为外部/新模型快速搭建临时评测工作区，或执行 task-local checkpoint 推理、detection/reconstruction 评测与 review；覆盖推理前合同确认、批量脚本、可视化、轻量测试和最小文档入口，不接训练主链。
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
- 也适用于 Shaft checkpoint 的 task-local 离线推理、layout recognition detection/reconstruction 两阶段评测
  和 review；这类任务不因使用仓库内权重就跳过推理合同确认。

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
3. 如果任务包含真实模型推理、layout recognition detection/reconstruction、结果 review 或 render/overlay，
   必须先完整读取
   [references/reconstruction-review.md](references/reconstruction-review.md)，并在首次真实请求前执行其中的
   “推理前确认门禁”。该 reference 同时是临时 reconstruction renderer 的可迁移视觉合同；一次性脚本、
   历史 HTML 和生产编辑器源码都只能作为实现或交叉核验材料，不能替代该合同。
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
1. 如需向真实模型发请求，先停在配置阶段，把 reference 规定的完整推理合同列给用户确认。未确认前只能做
   只读审计、配置准备和 dry-run；不能启动 canary 或正式生成。历史 run 和推荐基线不能代替本次确认。
2. 选择目录名 `<model-slug>-test/`，保持和模型名显式对应。
3. 先判断是否需要上游源码和工具依赖：
   - 如果官方 demo 依赖 `from xxx import Wrapper, Visualize` 这类包装层，而本地只有权重目录，优先下载上游源码仓库。
   - 读取上游 `README.md`、`requirements.txt`、`setup.py`、`pyproject.toml`，确认最小可运行依赖。
   - 需要 vendoring 时，把源码放到 `<model-slug>-test/<UpstreamRepoName>/`，不要依赖 `.tmp/` 的临时 clone。
4. 在 tools 模块中先定义：
   - `ModelConfig` dataclass
   - `resolve_model_path()`
   - `collect_*()` / `load_*()` helpers
   - `infer_*()` 共享推理函数
   - `run_*_batch()` 批处理入口
5. 外部模型依赖必须做明确导入保护。
   - 缺包时报清晰错误。
   - 不要在 import 时静默失败。
   - 推荐导入顺序：
     1. 已安装的官方包
     2. `<model-slug>-test/<UpstreamRepoName>/` 下的 vendored 源码
     3. 本地 fallback 实现
6. 本地模型路径优先从 `models/` 自动发现。
   - 找到本地目录则用本地目录。
   - 否则回退到用户传入路径或远端 repo id。
7. 如果上游依赖额外工具库：
   - 先用 `importlib.util.find_spec()` 或等价方式检查是否已安装。
   - 再决定是提示缺失、写 fallback，还是在用户明确需要时安装。
   - 不要默认假设环境里已经有上游工具包。
8. 批量脚本与 Gradio app 必须共用一套推理/导出逻辑。
   - 禁止在 `app.py` 里重新写一遍 batch 逻辑。
9. 输出至少包含：
   - `summary.json`
   - `manifest.jsonl`
   - `json/*.json`
   - 可选 `visualizations/*.jpg`
10. 多 GPU replica 的批量推理必须使用共享动态请求队列：endpoint 完成一个请求后立即领取下一条，不能按
    sample ID 固定绑定 endpoint。每个 endpoint 的 in-flight 上限应与 vLLM `max_num_seqs` 和显存容量对齐；
    canary 少于 replica 数时允许部分 GPU 空闲，正式批次则必须验证动态补位和尾部利用率。
    多个 Qwen3.5/FLA replica 同时首次 warmup 时，还必须将 `TRITON_CACHE_DIR` 和
    `TORCHINDUCTOR_CACHE_DIR` 放在节点本地盘并按 replica/GPU 隔离；禁止共享默认的
    `~/.triton/cache`，否则并发 JIT 可能互相删除临时 metadata，表现为 EngineCore 启动期随机
    `FileNotFoundError`/`Device or resource busy`。直接调用 `vllm serve` 不会自动继承 Shaft 训练 CLI 的
    rank cache 派生逻辑，launcher 必须显式设置。
11. reconstruction 的 crop/manifest 准备不能在模型已经驻留后按数据集串行阻塞 GPU。启动 vLLM 前应并发
    完成全部数据集的 crop，或把下一数据集准备与当前数据集生成做成流水线；大批量 crop 必须显式设置并核验
    CPU worker 数。看到全部 GPU 为 0% 时先检查 `prepare-reconstruction`、render、merge/evaluate 等 CPU
    阶段及 artifact 增长，不能只提高 vLLM in-flight。已有 crop 必须验证可解码后复用，并使用临时文件原子发布。
12. 用户明确要求 invalid 直接跳过时，失败请求原子记录到 error artifact，后续 resume 不再重复生成；summary
    必须分别报告 complete/error，评测将缺失预测计为 parse failure/FN，并保留失败 ID，不能伪造空预测为模型成功。
13. 补最小测试：
   - 类别/参数解析
   - 本地模型路径发现
   - vendored / fallback 导入路径
   - 图片扫描
   - batch 输出落盘
   - app smoke
14. 如新增了新的根目录临时工具，在 `docs/module_reference.md` 附录补一句边界说明即可。

## 验收
- `.venv/bin/python -m compileall <model-slug>-test`
- `.venv/bin/pytest -q tests/test_<model_slug>_tools.py`
- 不跑真实大模型推理，除非用户明确要求。
- 不修改 `src/shaft` 正式内核，除非用户明确要求集成。
- 如果引入了 vendored 上游源码，确认导入路径稳定，不依赖临时目录或未记录的 shell 状态。
- reconstruction renderer 在全量重建前必须通过 reference 规定的 marker matrix、旋转箭头、card 分区和
  坐标空间 canary；只验证脚本可运行或图片可解码不算视觉验收。

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
