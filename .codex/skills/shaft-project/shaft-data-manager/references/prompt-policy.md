# Prompt Policy

Prompt configs are the source of truth for SFT conversion and eval prompt seeding. Keep them
small, task-scoped, and model-output-oriented.

## Current Prompt Files

Historical v5.0 training/eval prompt pools:

- `configs/prompts/pools/grounding_layout.v5.0.yaml`
- `configs/prompts/pools/point_line.v5.0.yaml`

Active business reconstruction prompt pools:

- `configs/prompts/pools/shape_region_reconstruction.v5.2.yaml`
- `configs/prompts/pools/line_region_reconstruction.v5.2.yaml`
- `configs/prompts/pools/image_region_reconstruction.v5.2.yaml`
- `configs/prompts/pools/background.v5.0.yaml`

Active v5.3 training prompt pools:

- `configs/prompts/pools/grounding_layout.v5.3.yaml`
- `configs/prompts/pools/shape_context_reconstruction.v5.3.yaml`
- `configs/prompts/pools/shape_context_attributes.v5.3.yaml`
- `configs/prompts/pools/line_context_reconstruction.v5.3.yaml`
- `configs/prompts/pools/line_context_points.v5.3.yaml`
- `configs/prompts/pools/image_context_reconstruction.v5.3.yaml`

The three context reconstruction pools use dynamic `proposal_bbox_2d` in crop-local Qwen `0..999`
coordinates. The proposal is an approximate first-stage condition, not GT and not a bbox-local
output frame. Their formal contextual-crop datasets were generated and validated on 2026-07-16.

`shape_context_attributes` is a separate real-domain API weak-label task. It uses the same
contextual crop and approximate `proposal_bbox_2d` input contract, but target parameters contain
only shape type, border, fill, effect, and optional callout body type. It must never emit corners,
body/tail geometry, bbox, or points. Its omission of geometry is task semantics, not a partial or
malformed `shape_context_reconstruction` target.

`line_context_points` is a separate geometry subset task combining archived-real single paths
with a capped synthetic multi-segment supplement. It uses the same v5.3 contextual crop and
approximate `proposal_bbox_2d` contract, but its exact target is only
`{"type":"line","parameters":{"is_single":...,"points":[...]}}`. It must not ask for or emit
style, arrow endpoint, dash, color, border, confidence, or bbox fields outside the subset contract.
Preserve source segment order and point order.
For directional arrows, the archived contract is explicitly tail-to-arrowhead; every prompt
variant must state that order because this subset omits separate begin/end arrow fields.

The v5.3 grounding pool keeps the existing labels/schema but makes the line instance contract
explicit: one connected multi-segment, forked, branched, or multi-way connector is one `line`
instance with one bbox covering the complete connected structure. Only disconnected line objects
use separate bboxes.

These v5.3 prompt pools are local run assets and remain ignored under the repository's current
config policy. Maintained builders may use their conventional paths as local defaults, but tracked
tests must provide minimal temporary prompt fixtures and must not assume those files exist in a
clean checkout.

The v5.0 `shape_reconstruction`, `line_reconstruction`, and `image_reconstruction` pools are
historical crop-task contracts. Active v5.2 reconstruction training uses the region task names.

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
- `point_line` prompts must return the simplified line reconstruction object
  `{"type":"line","parameters":{"is_single":true|false,"points":[[[x1,y1],[x2,y2],...]]}}`.
  It intentionally omits style, arrowhead, color, and other reconstruction attributes.
- Shape reconstruction prompts should follow the current PDF shape DSL directly: lowercase
  `shape_type`, nested `border`, nested `fill`, `effect`, clockwise semantic `corners`, and
  callout-specific `body_type` / `body_corners` or `body_bbox` / `tail.points`. The shape
  reconstruction model output should not include `bbox`; bbox is provided by crop metadata.
  An icon/image asset with no independent editable outer shape returns only
  `{"shape_type":"other"}`; its rectangular bitmap boundary is not a rectangle container. A
  distinct enclosing tile, badge, or panel is still classified by that outer geometric shape.
- Region reconstruction consumes the full image and a dynamic `bbox_2d` prompt argument. The
  prompt bbox and every model-facing shape/line geometry field use the same Qwen integer `0..999`
  coordinate space normalized against the full input image. Do not normalize target geometry
  against the selected bbox.
- Context reconstruction consumes a bounded contextual crop and a dynamic
  `proposal_bbox_2d` prompt argument. The proposal may be imprecise. It and all shape/line target
  geometry use the same Qwen integer `0..999` space normalized against the current crop; output
  geometry may extend outside the proposal. Prompts must tell the model to follow visual evidence,
  select the intended target, and not copy the proposal rectangle as geometry.
- Line reconstruction prompts should follow the current PDF line DSL and the project extension
  for endpoint markers: `tee` for T-bar inhibition endpoints and `circle` for circular endpoint
  markers. `points` follows `gt_standard`: it is a list of one or more center-path segments,
  for example `[[[x1,y1],[x2,y2]]]` for a single line and multiple inner lists for forked or
  multi-branch lines. The line reconstruction model output should not include `bbox`; bbox is
  provided by crop metadata.
- For the `line_context_points` field-subset task, prompts must support both one-path and connected
  multi-segment targets without implying that multiple independent objects belong in one answer.
  A single directional arrow keeps tail-to-arrowhead order; a connected branched target keeps each
  visible branch in a separate path segment. The target remains exactly `is_single + points`.
- Image reconstruction prompts currently use only the PDF `image_type` field and return
  `{"type":"image","parameters":{"image_type":"photo|screenshot|chart|table|diagram|document|other"}}`.
  Do not add `clip_shape`, `border`, `effect`, or corner fields until those fields are explicitly
  needed by training or downstream reconstruction.
- Background prompts consume a clean full bitmap and return only
  `{"background":true|false}`. The model-facing boolean means that a non-editable raster/photo/
  texture/complex visual backing remains after editable foreground extraction. Prelabeling fields
  such as `background_level`, reason, source model, and review status are audit metadata and must
  not enter the SFT target.
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
