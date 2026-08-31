# Layout recognition 两阶段评测

入口：`scripts/tasks/run_layout_recognition_eval.py`。

该脚本服务可复现的离线评测，不进入 Shaft 正式推理 CLI。输入包括测试图片目录、Shaft v5.7 prompt pool、
独立 vLLM endpoint 和显式工作目录；只有 `evaluate` 阶段读取 benchmark GT。

## 阶段与合同

正式推理前必须先向用户列出并确认完整推理合同，至少包括 thinking/reasoning、detection 与 reconstruction
像素预算、prompt、generation 参数、crop 设置、fallback、模型/checkpoint 和输出覆盖策略。当前推荐基线是
detection 0.5M–4M、reconstruction 0.5M–4M，但不得因为它们数值相同或属于推荐值就跳过分阶段确认。

1. `detect`：对完整图片预测 `shape/icon/image/line`。推荐像素预算为 0.5M–4M；thinking 是否关闭必须在
   本次推理前确认。只有 `finish_reason=stop` 且完整 JSON 才能落盘。4M 是常规评测硬上限，任务脚本不得
   静默覆写为更大的预算；确需超过 4M 时，必须在启动前单独列出预算、受影响图片数并获得显式确认。
2. `prepare-reconstruction`：只从当前 detection 的 shape/line bbox 生成上下文 crop 和 manifest。
   每条记录固定保存 `proposal_source=detection`、`gt_read=false`、detection index 和原始 bbox。
   detection 阶段已显式接受 invalid 并跳过时，必须同时传入 `--allow-missing-detection`；该阶段跳过对应图片，
   并在 `prepare_summary.json` 保存 missing count/stems，最终 detection 评测仍把它计为 FN。
3. `reconstruct`：对上下文 crop 运行 shape/line 属性恢复。推荐使用 0.5M–4M、`padding_ratio=0.65`、
   `minimum_crop_size=256`；每次启动前仍需逐项确认。只有用户明确要求原生 crop 且确认放宽预算时，才允许
   改变该合同。随后把 0..999 crop 几何映射回原图坐标。
4. `merge`：按 detection index、label 和 proposal bbox 三重一致性检查后安装参数。默认遇到缺失的
   reconstruction 立即失败；当用户明确要求“不重跑、错误结果不要”时，`reconstruct` 可显式使用
   `--allow-invalid-output` 保留错误并继续，`merge` 同时使用 `--allow-missing-reconstruction` 丢弃对应
   `parameters`。这两个开关默认均关闭，不修补或伪造模型输出，报告必须列出 dropped 数量。
   如果上游 detection 已显式接受缺失，merge 还必须传入 `--allow-missing-detection`，不为该图片伪造最终
   JSON；missing stem 原样写入 summary，并由 evaluate 的 `--allow-missing-predictions` 按全量 GT 计 FN。
5. `evaluate`：冻结预测后才读取 GT，写入本地 `internal_score.json`，同时记录 GT revision、evaluator hash、
   checkpoint index hash 和像素预算。
6. `package`：按显式 `--dataset-name` 只复制 `<dataset>/pred/*.json`；默认仍是 `real_v1`，已有方法目录可追加
   新数据集子目录；`method.json`、`score.json`、`methods.json` 由远端自动流程负责。

各推理阶段按单条 JSON 原子落盘，重复执行会复用已完成结果。用户显式启用
`detect --allow-invalid-output` 后，失败项会保留 error artifact 且 resume 不再重复请求；
`evaluate --allow-missing-predictions` 将缺失预测作为 parse failure/FN 计入全量指标并记录 ID。summary 从
manifest/图片全集和实际落盘文件重新计算，因此中断恢复不会漏报历史错误或 contract issue。canary 可通过
`detect --include-stems` 选择样本，并应使用独立工作目录。`detect`/`reconstruct` 可重复传入 `--endpoint`；
正式批次使用共享动态 endpoint-slot 队列，默认每 endpoint 最多 4 个 in-flight 请求，请求完成即归还 slot，
由等待任务动态补位，不把 sample ID 固定绑定到某张 GPU。

推理前确认门禁和完整参数清单的唯一流程文档是
`.codex/skills/shaft-project/shaft-model-quick-test/references/reconstruction-review.md`；本任务文档不重复维护。

## Reconstruction review 可迁移性

review/overlay/render 属于 task-local 产物，不由 `run_layout_recognition_eval.py` 持有一套长期 renderer。
但视觉语义不能因此只存在于一次性代码中：shape/card/line 的可迁移合同统一维护在上述
`reconstruction-review.md`，其中包含箭头 marker 表、filled-body 内缩、斜向 bbox 厚度反解、非法
style-marker 组合和 canary matrix。每次新建或替换临时 renderer 都必须先读取该 reference，并在全量生成前
完成其视觉 canary；历史 `temp/` 脚本和页面不构成规范真源。

查看完整参数：

```bash
uv run python scripts/tasks/run_layout_recognition_eval.py --help
uv run python scripts/tasks/run_layout_recognition_eval.py detect --help
```

focused 回归：

```bash
uv run pytest -q tests/test_run_layout_recognition_eval.py
```
