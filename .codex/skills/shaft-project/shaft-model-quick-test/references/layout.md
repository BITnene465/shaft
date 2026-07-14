# 临时模型评测布局参考

仅在需要更具体落盘结构或代码分层时读取本文件。

## 推荐目录

```text
<model-slug>-test/
├── <UpstreamRepoName>/        # optional vendored upstream source
├── <model_slug>_tools.py
├── batch_infer.py
└── app.py

tests/
└── test_<model_slug>_tools.py
```

## `*_tools.py` 职责

- 统一模型配置 dataclass
- 本地 `models/` 自动发现
- 外部包导入保护
- vendored 上游源码导入兜底
- 输入图片扫描 / 读取 / 预处理
- 单图推理
- 批量推理
- JSON 安全序列化
- 结果与可视化落盘

如果官方用法依赖自定义 wrapper 或 parser，而本地只有权重目录，推荐顺序是：

1. 优先尝试环境里已安装的官方包
2. 再尝试 `<model-slug>-test/<UpstreamRepoName>/`
3. 最后才写本地 fallback

不要把上游源码放在 `.tmp/` 然后依赖一次性的 `sys.path` 状态。

如果任务是“图片输入 + 检测/定位/可视化”这一类，优先直接看：

- `rex-omni-test/rex_omni_tools.py`

重点看这些函数：

- `resolve_model_path()`
- `collect_image_paths()`
- `infer_images()`
- `run_detection_batch()`
- `save_prediction_visualization()`

## `batch_infer.py` 职责

- 只做 CLI 参数解析
- 调用 tools 层
- stdout 打印 summary JSON

如果你开始在 `batch_infer.py` 里写图片遍历、模型缓存、落盘逻辑，说明分层已经错了。

## `app.py` 职责

- 独立 Gradio app
- 只做 UI 组件、输入整理、调用 tools 层
- 单图页和批量页共用同一套推理函数

如果任务是单图 + 批量都要支持，建议：

- `Single Image` tab：预览、结构化 JSON、raw output
- `Batch` tab：目录/多图输入、summary JSON、gallery 预览

参考：

- `rex-omni-test/app.py`

## 测试策略

真实模型通常不在 CI 环境里，因此测试只校验：

- 参数解析
- 路径发现
- vendored / fallback 导入路径
- 图片扫描
- 落盘结构
- app 能否创建

不要在测试里真的加载第三方大模型；用 monkeypatch 替代：

- 推理函数
- 可视化导出函数

参考：

- `tests/test_rex_omni_tools.py`

## 上游仓库与工具依赖

如果第三方模型需要官方仓库源码或工具库：

- 先读上游 `README.md`
- 再读 `requirements.txt` / `setup.py` / `pyproject.toml`
- 识别最小运行依赖，而不是无脑搬完整环境

推荐做法：

- 上游源码 vendoring 到 `<model-slug>-test/<UpstreamRepoName>/`
- 导入逻辑写在 `*_tools.py`
- 缺失依赖给出清晰错误，或为当前测试路径实现最小 fallback

## 常用验收命令

```bash
.venv/bin/python -m compileall <model-slug>-test
.venv/bin/pytest -q tests/test_<model_slug>_tools.py
```

## 与正式内核的边界

- 临时评测目录留在仓库根目录
- 不进入 `src/shaft`
- 不进入 `src/shaft`，也不新增正式 UI/CLI 入口
- 文档只在 `docs/module_reference.md` 附录补一个入口说明
