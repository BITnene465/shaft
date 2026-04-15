# 测试规范（shaft）

本文档定义当前 `src/shaft` 的测试分层与执行规范。

## 一、测试类型

- **快速回归（默认）**：`pytest -q`
  - 覆盖绝大部分单元与轻量逻辑。
  - 不加载大模型，不依赖外部推理服务。

- **integration**：`pytest -q -m integration`
  - 真实模型加载/推理链路。
  - 包含推理编排集成测试（单阶段/多阶段），固定读取 `tests/fixtures/infer_images/` 下受 git 追踪的测试图片。
  - 默认不在主执行命令中运行（由 CI 配置排除）。

- **manual**：`pytest -q -m manual`
  - 人工触发的耗时检查。
  - 用于本地模型完整加载、长耗时验证。

## 二、运行约定

1. `pytest.ini` / `pyproject.toml` 中默认通过标记排除重型测试。
2. integration/manual 用例必须支持 `skip`：
   - 模型目录不存在
   - 适配器未注册
   - 运行环境缺少可用 GPU/大模型依赖（可按需放宽）
3. 集成用例优先使用 `--maxfail=1 -q` 保护本地实验时间。

## 三、新增测试规则

- 新加推理、训练关键改动后，先补对应单元测试。
- 需要校验“真实流程”的场景，至少放一条 integration/manual 用例。
- 若改动多数据源加载、mixing、增强编排，至少覆盖：
  - `tests/test_data_sources.py`
  - `tests/test_mixing.py`
  - `tests/test_data_center.py`
- 新增算法（如 DPO/PPO）必须覆盖：
  - 配置归一化校验（参数与 source_type 匹配）
  - collator 行为
  - trainer loss 前向
  - pipeline smoke（最短可训练链路）
- 现阶段 PPO 用例按 smoke 级别维护，不作为生产能力验收；细节见 `docs/ppo_todo.md`。
- 用例命名遵循：`test_<模块>_<行为>_integration` 或 `test_<模块>_<行为>_manual`（非强制）。
- 新增/修改 marker 时同步更新 `pyproject.toml`。

## 四、与文档联动

- 与 README 的测试区块保持一致。
- 与 `docs/architecture.md` 的测试边界描述保持一致。
