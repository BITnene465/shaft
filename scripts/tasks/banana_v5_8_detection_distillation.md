# Banana v5.8 detection 蒸馏数据

## 冻结合同

- teacher：Qwen3.8-27B v5.8 `checkpoint-4000`
- task：仅 detection；prompt 固定为 `grounding_layout.v5.8#detailed`
- generation：greedy，thinking off，`max_tokens=8000`
- KD：`topk_tail`，`temperature=1`，`top_k=64`
- 像素：按 sample ID 确定性分配 25% `0.5M–1M`、50% `1M–2M`、25% `2M–4M`；
  桶内 log-uniform。JSONL 记录预算与 smart-resize 目标尺寸，只引用原图。

selection：

```bash
uv run python scripts/tasks/build_detection_distill_selection.py \
  --data-root /path/to/tanjingyuan/data \
  --output /path/to/data/distillation/banana_v5_8_detection_unlabeled/selection.jsonl \
  --prompt-pool configs/prompts/pools/grounding_layout.v5.8.yaml \
  --prompt-variant detailed --seed 465 --workers 50 --content-dedupe
```

当前冻结结果：79,428 个候选，经已标注/test 内容泄漏排除和候选精确去重后保留 70,960 个；不复制图片。

## GPU map-reduce

20 条单卡 canary 通过后，用 worker-0/worker-1 的 16 张卡启动 16 个 TP1 map；每个 rank 处理
`source_index % 16 == rank` 的确定性分片。每卡通过 AsyncLLM 保持 16 个活跃请求；CPU 每准备完一个
sample 就立即提交，任一请求完成后马上补位，使单样本预处理、GPU prefill/decode 与落盘重叠。完成结果
通过有界重排缓冲按 source index 提交 writer，保留严格 resume 语义。worker-0 等待全部 16 个 manifest
后合并为一个标准 v1 artifact：

```bash
uv run python scripts/tasks/run_detection_pseudo_kd_map.py \
  --config configs/train/banana_detection_pseudo_kd_teacher_qwen38_ckpt4000_v5_8.yaml \
  --denylist configs/data/banana_detection_distill_denylist.v1.json \
  --output-root /path/to/data/distillation/banana_v5_8_detection_unlabeled/qwen38_ckpt4000_top64 \
  --gpus 8 --rank-offset 0 --source-world-size 16 \
  --allow-existing-root --merge --scoring-batch-size 16
```

worker-1 使用同一命令，但设置 `--rank-offset 8` 并省略 `--merge`。

每个 rank 的无效输出保留在 `rank-NN.bad_cases.jsonl`，但不进入该 rank artifact。merge 先验证所有
manifest 语义和 shard checksum，再用 hardlink（不可用时才复制）组装最终 `merged/`；引用会改写为新的
canonical artifact ID。

## 过滤与发布

`filter_detection_distillation_candidates.py` 消费已有/新增 artifact JSONL、公开 eval 图片目录和对应
manifest，执行严格 bbox 规则、eval 泄漏检查、三种感知 hash 近重复聚类，并以 teacher completion mean NLL
选择重复簇代表；它只写新的 accepted/exclusions/drop-existing 文件，不修改输入 artifact。

`package_detection_distillation.py` 消费冻结 selection、accepted JSONL、原始 artifact 和 exclusion audit，发布
新的自包含 bundle。图片、`gt_standard`、index、audit 和 compact Offline-KD artifact 全部先写到同级 staging，
成功后原子发布；artifact 重写必须复用框架 `OfflineKDArtifactWriter`，不得在 task 脚本维护第二套 manifest/
shard writer 或自定义 resume 状态。

新增 paper 图片使用 `prepare_new_paper_detection_distill.py` 生成新的 selection/audit，再经同一 map 与 filter
合同处理。`finalize_detection_distillation_bundle.py` 接受旧 bundle 和显式 `--output`，通过 hardlink/copy staging
生成新版本；旧 bundle 始终只读，output 已存在时立即失败。该增量入口不得原地更新已发布 bundle。
