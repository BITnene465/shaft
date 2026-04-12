# 新增 Task / Domain 指南

## 1. 现有层级

```text
core/      通用框架（不理解具体业务语义）
tasks/     任务语义与 adapter（实现 TaskAdapter 协议）
domains/   域语义（codec、排序、数据准备、域推理约定）
```

## 2. 新增 Task（同一 domain）

步骤：

1. 在 `core/registry.py` 注册 task 类型。
2. 新建 `tasks/<task>/adapter.py`，实现 `TaskAdapter`。
3. 在 registry 路由分发中接入该 task。
4. 在 evaluator 汇总中加入该任务主指标。
5. 新增训练配置（`configs/train/`）。
6. 准备该 task 对应 JSONL 数据（route 可来自配置绑定或样本字段）。

检查清单：

- [ ] registry 已注册新 task
- [ ] adapter 已实现并可被路由
- [ ] evaluator 可汇总该任务指标
- [ ] 训练配置可跑通
- [ ] 文档已更新（架构、数据格式）

## 3. 新增 Domain（跨域扩展）

步骤：

1. 在 `core/registry.py` 注册 domain 类型。
2. 建立 `domains/<domain>/` 目录，至少包含：
   - `schema.py`
   - `ordering.py`
   - `task_support.py`
   - `codecs/`
   - `data/`
   - `infer/`（若有两阶段编排）
3. 为各 task 提供对应 codec。
4. 在 task 的 adapter factory 中加入新 domain 分支。
5. 如指标口径不同，更新 evaluator。
6. 提供数据准备脚本并产出标准 JSONL。

检查清单：

- [ ] registry 已注册新 domain
- [ ] codec 与 schema 已落地
- [ ] task adapter 已支持新 domain
- [ ] evaluator 与文档同步

## 4. TaskAdapter 关键接口（摘要）

- `build_gt_struct_from_record(record)`
- `encode_target_text(gt_struct, w, h)`
- `build_training_target(gt_struct, w, h)`（返回 `target_text + loss_meta`）
- `decode(text, w, h, strict)`
- `decode_with_meta(text, w, h, strict)`
- `score_prediction(gt, pred, ...)`

## 5. 常见错误

1. 只改了 adapter，没改 registry 分发。
2. 只改了 codec，没改 evaluator 汇总。
3. 忘记固化 canonical order，导致训练目标不稳定。
4. 在 trainer 内写 task/domain 语义逻辑，破坏层级边界。
