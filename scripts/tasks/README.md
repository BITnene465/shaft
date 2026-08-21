# 离线任务脚本

`scripts/tasks/` 保存可复现的数据准备、转换和审计工具。这些脚本服务具体项目任务，不属于 Shaft 训练
运行时或公共 CLI。

## 当前任务文档

- [banana_v5_7.md](banana_v5_7.md)：Banana v5.7 的数据、prompt、训练配置、构建顺序与完整性基线。
- [banana_v5_8.md](banana_v5_8.md)：Banana v5.8 的 source/structured/SFT、人工 formulation 与在线随机合同。
- `recover_v5_8_real_tasks.py`：从显式历史来源恢复 v5.8 `background` 与
  `image_context_reconstruction`，完成测试集排除、target 对账和全量媒体校验后原子发布。
- `prepare_gt_standard_v5_7.py --selection-profile v5.8`：审计 V9 后生成稀有 shape 全保留、
  多叉 line 全保留、普通头部限比以及多叉 points 完整属性 rarity-first 的 source-identity selection。
- `prepare_real_line_context_points.py`：排除 canonical test 后保留 active compact raw 中全部非空真实
  line points，不采样、不补造空标注。
- `build_context_reconstruction_sft.py`：一次 crop pass 同时物化 shape/line 的全部 eligible
  formulations；合成 crop 使用 `synthetic_realism_v1`，真实 points crop 保持干净像素。

## 边界

- 任务脚本必须显式声明输入、输出和清理行为，不覆盖原始数据。
- 任务字段和业务标注规则在离线派生阶段收口，不进入 planner、collator 或训练循环。
- 框架能力与公共接口以 [`docs/`](../../docs/README.md) 为准；任务文档不能作为框架支持声明。
