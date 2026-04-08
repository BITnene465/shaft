# Prompt Profiles

Prompt templates are now managed under `configs/prompts/` and referenced from train/infer configs via `prompt.profile`.

## Naming Convention

Use dot-separated names:

- `<domain>.<task>.<stage_or_mode>.v<version>.yaml`

Examples:

- `arrow.grounding.stage1.v1.yaml`
- `arrow.grounding.stage1.v2.yaml`
- `arrow.keypoint_sequence.stage2_template.v1.yaml`
- `arrow.joint_structure.one_stage.v1.yaml`

## File Schema

Each prompt profile file should contain:

```yaml
metadata:
  id: arrow.grounding.stage1.v2
  name: arrow grounding stage1 v2
  created_at: "2026-04-08"
  scope:
    - train:grounding/arrow
    - infer:stage1

prompt:
  system_prompt: ""
  user_prompt: |
    ...
```

`prompt` supports these keys only:

- `system_prompt`
- `system_prompt_template`
- `user_prompt`
- `user_prompt_template`

## Usage in Configs

Train config example:

```yaml
prompt:
  profile: arrow.grounding.stage1.v2
```

Infer config example:

```yaml
prompt:
  profile: arrow.keypoint_sequence.stage2_template.v1
```

Optional local overrides are still supported by adding explicit prompt fields in the same config.
