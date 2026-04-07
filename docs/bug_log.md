# Bug Log

本文件专门记录已确认 bug、根因、修复方案和验证结果。

## 2026-04-07 Stage2 极端长宽比 Crop 触发 Qwen Processor 报错

- 症状：
  - 两阶段离线批量推理在 Stage2 进入 processor 前报错：
    `ValueError: absolute aspect ratio must be smaller than 200, got 684.0`
- 影响范围：
  - `scripts/arrow/infer_two_stage.py`
  - `src/vlm_structgen/domains/arrow/infer/two_stage.py`
- 根因：
  - 某些箭头 bbox 极细长，Stage2 crop 在进入 Qwen image processor 前仍可能保持极端长宽比。
  - Qwen2/Qwen3-VL 的 image processor 对输入图像存在硬限制：绝对长宽比必须小于 `200`。
- 解决方案：
  - 在 `TwoStageInferenceRunner._build_stage2_requests_for_image(...)` 中增加 Stage2 request 级别的长宽比规范化。
  - 当 crop 长宽比超过阈值时，先扩展 `crop_box`，再用黑边重建 crop，确保送入 Stage2 的图像长宽比不超过安全阈值。
  - 当前推理侧安全阈值固定为 `180.0`，给 processor 的 `< 200` 限制预留缓冲。
- 设计说明：
  - 该修复放在 `domains/arrow/infer/two_stage.py`，不进入 `core`。
  - 原因是这是 arrow two-stage 的 domain 编排问题，不是通用单阶段推理问题。
- 验证：
  - 单元测试新增极端细长 bbox case，断言 Stage2 实际收到的 crop request 长宽比 `<= 160.0`。
  - 相关回归：
    - `tests.test_two_stage_pipeline_batching`
    - `tests.test_two_stage_stage1_report`
    - `tests.test_infer_protocol_alignment`
    - `tests.test_one_stage_infer_batching`
    - `tests.test_weighted_loss_boundaries`
    - `tests.test_dataset_structured_gt_source`
    - `tests.test_checkpoint_loading`
