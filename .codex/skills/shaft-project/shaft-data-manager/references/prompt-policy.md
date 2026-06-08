# Prompt Policy

Prompt configs are the source of truth for SFT conversion and eval prompt seeding. Keep them
small, task-scoped, and model-output-oriented.

## Current Prompt Files

- `configs/prompts/pools/grounding_arrow.v2.4.yaml`
- `configs/prompts/pools/grounding_layout.v2.4.yaml`
- `configs/prompts/pools/grounding_shape.v2.4.yaml`
- `configs/prompts/pools/grounding_icon_image.v2.4.yaml`
- `configs/prompts/pools/point_arrow.v2.4.yaml`
- `configs/prompts/pools/point_arrow_style.v2.4.yaml`
- `configs/prompts/pools/drawio_shape.v4.0.yaml`

Do not keep legacy duplicate prompt files such as `keypoint_arrow.yaml`; the maintained point
task name is `point_arrow`.

## Method

- Keep one prompt config per train/eval subtask.
- Equivalent train-time prompt variants may be configured as runtime prompt pools. Keep one
  versioned pool YAML per subtask, and keep the canonical prompt as the `main` pool member when
  desired.
- Prompt variants may change wording only. They must not change target labels, output schema,
  ordering policy, or task semantics.
- When runtime prompt pools are enabled, generated SFT rows may keep `user_prompt` empty; the pool
  is the train prompt source and every enabled train dataset must have an explicit pool.
- Metadata should identify the stable task family (`grounding` or `point`), the subtask, and the
  target labels.
- The system prompt should only define response discipline: valid compact JSON, no markdown, no
  explanations.
- The user prompt should define the visible target labels, tight-box policy, ordering policy,
  the exact JSON schema, the normalized coordinate space, and the empty-output behavior.
- Grounding prompts must return a JSON array of `{label, bbox_2d}` objects.
- Grounding objects should be ordered row-major: top to bottom, then left to right within the
  same visual row.
- Point prompts must return a JSON object with `keypoints_2d`.
- `point_arrow` should predict the full ordered arrow `linestrip`, including bend points, ordered
  from arrow tail to arrow head. Do not collapse this task to two endpoints unless the structured
  and SFT targets are changed together.
- `drawio_shape` prompts must focus on the outermost primary visible shape in the crop for
  draw.io reconstruction. They should ignore internal text/icons/arrows/nested boxes and output
  only business fields such as shape type, orientation, stroke/fill visibility, stroke/fill style,
  and hex colors.
- Do not ask business-task models to emit weak-label audit fields. For `drawio_shape`, fields such
  as `evidence`, `confidence`, and `abstain_reason` are weak-label process metadata, not SFT
  targets.
- Avoid examples beyond the minimal schema shape; examples can accidentally become style anchors.
- Update SFT conversion, eval prompt seeding, and prelabeling references together when a prompt
  file is renamed or removed.

## Git Tracking

Generated structured/SFT data stays ignored. For data catalog examples, keep only
`configs/data/example.yaml` tracked; local project-specific catalogs should remain untracked.
Prompt pool YAMLs under `configs/prompts/` are also ignored by default in this repository. When a
new prompt pool becomes part of a reproducible training run, call out whether it should stay local
or be force-added intentionally.
