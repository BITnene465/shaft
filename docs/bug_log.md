# 缺陷记录

用于记录已确认问题、根因、修复方式与验证结果。

## 2026-04-07：Stage2 极端长宽比 Crop 触发 Qwen Processor 报错

- 现象：
  - 两阶段推理在 Stage2 进入 processor 前报错：
    `ValueError: absolute aspect ratio must be smaller than 200`
- 影响：
  - `scripts/arrow/infer_two_stage.py`
  - `src/vlm_structgen/domains/arrow/infer/two_stage.py`
- 根因：
  - 某些箭头 bbox 极细长，按默认规则裁剪后仍超过 processor 长宽比限制。
- 修复：
  - 在 `TwoStageInferenceRunner._build_stage2_requests_for_image(...)` 增加长宽比规范化。
  - 当 crop 长宽比过大时扩展 `crop_box` 并黑边补齐。
  - 推理侧安全阈值设为 `180.0`（低于 processor 的 `<200` 硬限制）。
- 设计边界：
  - 修复放在 `domains/arrow/infer`，不放进 `core`。
- 验证：
  - 增加极端 bbox 用例，断言 Stage2 请求长宽比在安全阈值内。
  - 相关回归测试通过。
