---
name: shaft-data-center
description: 多数据源、多 jsonl、sample-level 混合、增强策略的统一入口。
---

# Skill：数据中心

## 触发场景
- 同时训练多份数据
- 需要权重混合/采样策略
- 离线与在线增强并行

## 步骤
1. 在 `data.datasets` 配置多个数据源条目。
2. 数据源在 registry 中注册。
3. 由 `src/shaft/data/center.py` 中的 `ShaftDataCenter` 统一完成：数据源加载、offline transform、sample-level mixing、dataset-aware online transform 编排。
4. pipeline 只调用 `ShaftDataCenter`，不要在 pipeline 内重新手写 mixing 或数据来源分支。
5. 增加/更新 `tests/test_data_sources.py`、`tests/test_mixing.py`、`tests/test_data_center.py`。

## 验收
- 多源可稳定加载；
- 混合策略复现；
- 不在训练核心里写数据来源分支。
