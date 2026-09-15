# v5.10 真实标注质量清洗

入口 `scripts/tasks/clean_real_raw_annotations.py`，只处理显式 `--raw-root` 下的 compact
`json/` 与对应 `images/`。不触碰其他 raw 副本、原图、split 或合成重建产物。

```bash
uv run --no-sync python scripts/tasks/clean_real_raw_annotations.py \
  --raw-root /path/to/raw_data --report /path/to/audit.json --workers 50
uv run --no-sync python scripts/tasks/clean_real_raw_annotations.py \
  --raw-root /path/to/raw_data --report /path/to/audit.json --apply
```

先全量审计（图片完整解码），再检查报告并显式 apply。报告包含每份输入 SHA256、脚本 SHA256、
实例下标（清洗前）、问题原因；输入/脚本变化会拒绝 apply。不要修改报告中的处理等级。
Apply 保留 `json.before_quality_cleaning_v5_10` 完整快照，将严重错误的整份文件复制到
`json.quarantine_quality_cleaning_v5_10`，校验清洗幂等后发布新 `json`。备份/输出已存在时拒绝覆盖。

规则真源是 `inspect_annotation`：

- 修复：完全相同实例去重；不超过 3px 的 bbox 越界裁到 `[0,width] × [0,height]`。
- 相同 shape 框存在完整属性标注时，删除只有 type/bbox 的重复副本；若完整属性相互冲突，
  按高质量筛选策略隔离整份 JSON，保留人工复核机会，不猜测哪套属性正确。
- 整份隔离：无效 JSON（含重复键/非有限数字）、无效尺寸/类别/框/点、非空退化路径、
  缺图/多图歧义/完整解码失败、不符合 EXIF 的尺寸冲突、较大 bbox 越界。
- line 路径端点超出主 bbox 超过 `max(20px, 图像最长边的1%)` 时，按高质量筛选策略整份隔离，
  不猜测是 bbox 错还是路径错；此阈值是保守筛选策略，不是普遍标注标准。
  不用外部曲线控制点直接判定端点冲突。
- 保留并告警：同框不同内容、源路径连续重复点/重复段、较小 bbox/path 差异、需要 EXIF 转置的图。
- 空 points/缺少子属性不是错误；grounding 与 reconstruction 的准入由派生任务分别决定。
- `subbbox` 不使用、不校验、不作为删除条件；保留源字段。
- `split` 与 `splits` 两种源格式均保留，不补猜颜色/类型，不改路径顺序。

分阶段清洗可通过 `--snapshot-tag quality_review_v5_10` 指定另一组备份/隔离名称，
不得覆盖上一轮备份。单次运行当前规则即可复现合并后的清洗结果。

限制：不代表全量人工视觉语义验收，不进行图片跨 ID 去重或测试集泄漏清理。
后续构建仍须实施 real_v1/real_v2/canonical test 内容级排除、任务目标 schema 和量化检查。
告警不能被报告为“全部标注零问题”；需要 EXIF 的图片必须由派生读取器按合同处理。

测试：`uv run --no-sync pytest -q tests/test_clean_real_raw_annotations.py`。
