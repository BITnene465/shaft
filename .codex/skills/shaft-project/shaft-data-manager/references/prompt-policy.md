# Prompt Policy

Prompt configs are the source of truth for SFT conversion and eval prompt seeding. Keep them
small, task-scoped, and model-output-oriented.

## Current Prompt Files

Active v5.0 training/eval prompt pools:

- `configs/prompts/pools/grounding_layout.v5.0.yaml`
- `configs/prompts/pools/point_line.v5.0.yaml`

Business reconstruction prompt pools:

- `configs/prompts/pools/shape_reconstruction.v5.0.yaml`
- `configs/prompts/pools/line_reconstruction.v5.0.yaml`
- `configs/prompts/pools/image_reconstruction.v5.0.yaml`

Historical `arrow` annotations are normalized to the model-facing `line` label inside
`grounding_layout`. Do not introduce new detection prompts with an `arrow` label. v5.0 has no
standalone `grounding_shape`, `grounding_icon_image`, or `grounding_line` prompt pools; all
detection labels live in the unified `grounding_layout` task.

## Method

- Keep one prompt config per train/eval subtask.
- Equivalent train-time prompt variants may be configured as runtime prompt pools. Keep one
  versioned pool YAML per subtask.
- Every prompt pool must include a `main` variant. `main` is the canonical direct-task prompt used
  by SFT conversion and eval seeding when a single deterministic prompt is needed.
- Organize non-main prompt variants as prompt families rather than near-duplicate synonyms.
  Preferred family names include `visual_elements` or `visible_object`, `schema_first`,
  `extraction_request` or `attribute_extraction`, and `minimal_contract`.
- Prompt variants may change wording and prompt framing only. They must not change target labels,
  output schema, field names, empty-output behavior, or task semantics.
- When runtime prompt pools are enabled, generated SFT rows should keep both `system_prompt` and
  `user_prompt` empty; the pool is the train prompt source and every enabled train dataset must
  have an explicit pool.
- Metadata should identify the stable task family (`grounding` or `point`), the subtask, and the
  target labels.
- The system prompt should only define response discipline: valid compact JSON, no markdown, no
  explanations.
- User prompts should be clear, task-scoped, and easy to generalize. Prefer light task
  descriptions plus the exact JSON schema. Do not add coordinate-space, ordering, or tight-box
  rules unless the training targets and eval contract explicitly require them.
- Grounding prompts must return a Qwen-style JSON array of objects:
  `[{"bbox_2d":[x1,y1,x2,y2],"label":"shape|icon|image|line"}]`. Empty images return `[]`.
- Do not use the old grouped numeric-only grounding schema
  `{"shape":[...],"icon":[...],"image":[...],"line":[...]}` for new v5.0+ data. It is compact,
  but it is too far from the model's common detection output format and has shown numeric tail
  repetition in dense outputs.
- Eval parsing, metric input, reports, and business handoff still normalize predictions back to
  the canonical instance-list document shape with `instances[].label` and `instances[].bbox`.
- `grounding_layout` uses labels `shape`, `icon`, `image`, and `line`. Raw `arrow` and raw
  `line` instances both become model-facing `label: "line"` in this unified detection task.
- `point_line` prompts must return a JSON object with
  `{"label":"line","points_2d":[[x1,y1],[x2,y2],...]}`.
- Shape reconstruction prompts should follow the current PDF shape DSL directly: lowercase
  `shape_type`, nested `border`, nested `fill`, `effect`, clockwise semantic `corners`, and
  callout-specific `body_type` / `body_corners` or `body_bbox` / `tail.points`. The shape
  reconstruction model output should not include `bbox`; bbox is provided by crop metadata.
- Line reconstruction prompts should follow the current PDF line DSL and the project extension
  for endpoint markers: `tee` for T-bar inhibition endpoints and `circle` for circular endpoint
  markers. `points` follows `gt_standard`: it is a list of one or more center-path segments,
  for example `[[[x1,y1],[x2,y2]]]` for a single line and multiple inner lists for forked or
  multi-branch lines. The line reconstruction model output should not include `bbox`; bbox is
  provided by crop metadata.
- Image reconstruction prompts currently use only the PDF `image_type` field and return
  `{"type":"image","parameters":{"image_type":"photo|screenshot|chart|table|diagram|document|other"}}`.
  Do not add `clip_shape`, `border`, `effect`, or corner fields until those fields are explicitly
  needed by training or downstream reconstruction.
- Do not ask business reconstruction tasks to emit weak-label audit fields. Fields such as
  `evidence`, `confidence`, and `abstain_reason` are weak-label process metadata, not SFT targets.
- Avoid examples beyond the minimal schema shape; examples can accidentally become style anchors.
- Update SFT conversion, eval prompt seeding, and prelabeling references together when a prompt
  file is renamed or removed.

## Git Tracking

Generated structured/SFT data stays ignored. For data catalog examples, keep only
`configs/data/example.yaml` tracked; local project-specific catalogs should remain untracked.
Prompt pool YAMLs under `configs/prompts/` are also ignored by default in this repository. When a
new prompt pool becomes part of a reproducible training run, call out whether it should stay local
or be force-added intentionally.
