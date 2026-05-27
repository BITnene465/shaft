# Prompt Policy

Prompt configs are the source of truth for SFT conversion and eval prompt seeding. Keep them
small, task-scoped, and model-output-oriented.

## Current Prompt Files

- `configs/prompts/grounding_arrow.yaml`
- `configs/prompts/grounding_layout.yaml`
- `configs/prompts/grounding_shape.yaml`
- `configs/prompts/grounding_icon_image.yaml`
- `configs/prompts/point_arrow.yaml`

Do not keep legacy duplicate prompt files such as `keypoint_arrow.yaml`; the maintained point
task name is `point_arrow`.

## Method

- Keep one prompt config per train/eval subtask.
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
- Avoid examples beyond the minimal schema shape; examples can accidentally become style anchors.
- Update SFT conversion, eval prompt seeding, and prelabeling references together when a prompt
  file is renamed or removed.

## Git Tracking

Generated structured/SFT data stays ignored. For data catalog examples, keep only
`configs/data/example.yaml` tracked; local project-specific catalogs should remain untracked.
