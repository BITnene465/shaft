---
name: shaft-data-manager
description: 管理 Shaft 数据整理任务，包括 raw 数据清洗、标注合并、去重、split、preview、grounding 数据增强、structured/SFT 派生数据重建，以及训练配置中的数据源登记准备；不用于修改 ShaftDataCenter 或 mixing 实现。
---

# Shaft Data Manager

Use this skill for raw cleanup, annotation import, split management, grounding augmentation,
prompt-aware structured/SFT rebuilds, and data catalog preparation in this repository.

## Core Workflow

1. Identify the layer:
   - `data/raw`: active human source truth.
   - `regulated_layout_dataset_v9_20260802/gt_standard`: active synthetic reconstruction truth.
   - task-local `selection` / `structured` / `sft` / `images`: rebuildable derived artifacts.
   - `configs/data` and `configs/prompts/pools`: runtime dataset/prompt contracts.
2. Read `references/counterintuitive-rules.md`, then only the references needed:
   - raw source: `raw-data.md`
   - cleaning/dedupe/import: `cleaning.md`, `merge.md`
   - grounding and line paths: `augmentation-grounding.md`, `layout-grounding.md`,
     `arrow-grounding.md`
   - prelabeling and preview: `prelabeling.md`, `preview.md`
   - prompt and derived outputs: `prompt-policy.md`, `derived-datasets.md`
   - catalog preparation: `data-center.md`
3. State input, output, split source, target schema, and source of truth before writing.
4. Back up only the touched raw JSON directory before raw batch edits. Derived rebuilds should
   publish through staging/new output or an explicit supported `--clean` workflow.
5. Preserve the active raw contract. `data/raw/json` currently uses compact
   `size + layout[].type/bbox/parameters`; normalized imports may use `instances`. Adapt at the
   derived boundary instead of rewriting one maintained contract into the other.
6. Keep raw state in its directory README and derived build state in task-local README/summary.
7. Validate counts, split boundaries, structured/SFT alignment, media coverage, target schema,
   prompt version, catalog mapping, and model/checkpoint prerequisites.

## Non-Negotiables

- Never overwrite or delete original images without explicit authorization.
- Never promote derived rows or model prelabels to source truth when raw/`gt_standard` exists.
- Never augment validation/test with train-only views.
- Keep temporary handoff artifacts outside raw; persistent rebuild logic belongs in
  `scripts/tasks/`.
- Keep selection manifests identity-only when authoritative attributes/geometry can be reread from
  source truth. Do not create a parallel target truth in selection or SFT.
- Preserve compact raw `parameters`, including ordered line `points`. For normalized imports,
  normalize importer-native `shapes`, `shape_type`, `group_id`, and flags before maintenance.
- `grounding_layout` consumes only `shape/icon/image/line` bbox; it excludes `full_text` and does
  not consume reconstruction parameters.
- Same-label/same-bbox line instances are not duplicates when their ordered paths differ.
- If work changes `ShaftDataCenter`, registry behavior, or mixing implementation, leave this data
  skill and follow repository architecture, extension, and feature-review requirements.

## Validation Checklist

- JSON/image coverage, parse errors, decoded dimensions, and missing pairs are counted.
- Train/val/test membership is explicit and pairwise disjoint before augmentation.
- Labels, degenerate boxes, and semantic duplicates are checked without bbox-only line dedupe.
- Active compact rows keep `size + layout`; normalized imports expose their documented
  `instances` contract. Neither is silently converted in place.
- Every derived JSONL media path exists, is task-local where required, and has a unique sample id.
- Structured and SFT rows align one-to-one; SFT target and `prompt_args` can be recomputed from
  structured/source truth.
- Runtime catalog names and prompt pools match the published training config exactly.
- Validation data is clean and unaugmented; intentional empty validation is documented as
  train-only rather than mistaken for a missing artifact.
