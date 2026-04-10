# Prompt Profile 管理

Prompt 模板统一放在 `configs/prompts/`，并在 train/infer 配置中通过 profile 引用。

## 1. 命名约定

建议格式：

```text
<domain>.<task>.<stage_or_mode>.v<version>.yaml
```

示例：

- `arrow.grounding.stage1.v1.yaml`
- `arrow.grounding.stage1.v2.yaml`
- `arrow.keypoint_sequence.stage2_fixed.v2.yaml`
- `arrow.joint_structure.one_stage.v1.yaml`

## 2. 文件结构

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

`prompt` 仅支持：

- `system_prompt`
- `system_prompt_template`
- `user_prompt`
- `user_prompt_template`

## 3. 在配置中的使用

单任务：

```yaml
prompt:
  profile: arrow.grounding.stage1.v2
```

多任务（按 route）：

```yaml
prompt:
  route_prompts:
    grounding/arrow:
      profile: arrow.grounding.stage1.v2
    keypoint_sequence/arrow:
      profile: arrow.keypoint_sequence.stage2_fixed.v2
```

## 4. 约束

- prompt 只负责任务接口，不是监督真源。
- 若输出协议变化，必须同时更新 codec 与评估逻辑。
