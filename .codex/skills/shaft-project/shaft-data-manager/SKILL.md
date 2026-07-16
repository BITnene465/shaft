---
name: shaft-data-manager
description: 管理 Shaft 数据整理任务，包括 raw 数据清洗、标注合并、去重、split、preview、grounding 数据增强、structured/SFT 派生数据重建，以及训练配置中的数据源登记准备；不用于修改 ShaftDataCenter 或 mixing 实现。
---

# Skill: Shaft Data Manager

Use this skill for data work in this repository: raw dataset cleanup, annotation merge,
preview policy, grounding data augmentation, derived dataset rebuilds, and data catalog
preparation from the data side.

## Core Workflow

1. Identify the layer being changed:
   - `raw_*`: source annotations and images.
   - `grounding` / `point_arrow` / `structured` / `sft`: derived artifacts that should be
     rebuildable.
   - `configs/data/*.yaml`: catalog and mixing configuration.
2. Always read the counterintuitive rules first, then only the task-specific references needed:
   - Counterintuitive data rules: `references/counterintuitive-rules.md`
   - Raw source handling: `references/raw-data.md`
   - Cleaning and dedupe: `references/cleaning.md`
   - Data merge/import: `references/merge.md`
   - General grounding task and augmentation: `references/augmentation-grounding.md`
   - Layout raw-layer rules for grounding: `references/layout-grounding.md`
   - Arrow raw-layer rules for grounding/point: `references/arrow-grounding.md`
   - Model-assisted prelabeling: `references/prelabeling.md`
   - Prompt policy for SFT/eval data: `references/prompt-policy.md`
   - Preview policy: `references/preview.md`
   - Derived structured/SFT rebuilds: `references/derived-datasets.md`
   - Data catalog usage from the data-prep side: `references/data-center.md`
3. Before acting, make these explicit when they are not obvious from the user request:
   input path, output path, split source, target schema, and whether rich structure should be
   preserved in `extra` / `subattr`.
4. Before writing raw annotations, create a small backup of the touched JSON directory.
5. Maintained raw annotations should use normalized `instances` schemas, not importer-native
   LabelMe `shapes` as the long-term source format.
6. Keep raw directory state in that directory's `README.md`; do not create long reports unless
   the user asks for machine-readable audit artifacts.
7. Validate with counts and invariants, then report the operational result.

## Non-Negotiables

- Do not overwrite or delete original images unless the user explicitly asks.
- Do not let derived data become the source of truth when raw annotations are available.
- Do not mix train-only augmentation into validation data.
- Do not leave temporary scripts behind. If a script should persist, place it under
  `scripts/tasks/` as a maintained entry.
- Prelabeling outputs are review artifacts, not raw truth. Keep them under `temp/task*` or another
  explicit handoff directory until a human imports them into raw annotations.
- Keep rich annotation details in raw `extra` or `subattr`; do not copy that raw metadata into
  derived structured/SFT rows unless it is directly consumed by training or needed as a minimal
  source id.
- Do not preserve LabelMe `points`, `shape_type`, or `group_id` as live instance fields in raw
  maintenance schemas. Normalize geometry into `bbox` / `linestrip`, and keep source-only
  details in `extra`.
- When a rule feels obvious but was previously a source of error, record it in
  `references/counterintuitive-rules.md` rather than burying it in chat history.
- If the task changes `ShaftDataCenter`, registry behavior, or mixing implementation, use the
  separate `shaft-data-center` development skill.

## Validation Checklist

- JSON/image coverage and missing pairs are counted.
- Label distribution is checked before and after destructive cleanup.
- Degenerate boxes and same-label same-bbox duplicates are checked when touching annotations.
- Raw schemas expose `instances`; layout instances use `label + bbox + extra`, while arrow
  instances use `label + bbox + linestrip + subattr + extra`.
- No maintained raw instance has live `points`, `shape_type`, `group_id`, or `flags`.
- Split boundaries are explicit before augmentation.
- Derived JSONL rows and image files are aligned.
- Validation data is not silently augmented with train-only views.
- Raw README reflects the current schema and notable caveats.
