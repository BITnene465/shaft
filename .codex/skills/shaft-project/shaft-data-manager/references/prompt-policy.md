# Prompt Policy

Prompt pools are versioned runtime contracts. They define task wording and output schema; they do
not make old derived targets compatible. Rebuild or validate data before switching pools.

## Banana v5.7 Pools

Every formal v5.7 training config maps exactly these datasets:

| dataset | pool |
| --- | --- |
| `grounding_layout` | `configs/prompts/pools/grounding_layout.v5.7.yaml` |
| `shape_context_reconstruction` | `configs/prompts/pools/shape_context_reconstruction.v5.7.yaml` |
| `line_context_reconstruction` | `configs/prompts/pools/line_context_reconstruction.v5.7.yaml` |
| `line_context_points` | `configs/prompts/pools/line_context_points.v5.7.yaml` |
| `image_context_reconstruction` | `configs/prompts/pools/image_context_reconstruction.v5.3.yaml` |

Image type intentionally retains v5.3 because its reviewed 13-class contract is unchanged. This
is the only prompt-version exception in the v5.7 mix. `shape_context_attributes`, background,
region reconstruction, `grounding_arrow`, and `point_arrow` are not formal v5.7 tasks.

These production pools, the v5.7 data catalog, and all six v5.7 training YAMLs are tracked repo
assets. Generated data remains ignored.

## Pool Rules

- One task owns one versioned pool with stable `metadata.id/version` and a canonical `main`
  variant.
- Variants may change wording only. Labels, exact output keys, coordinate frame, empty behavior,
  and task semantics must stay identical.
- Runtime prompt sampling is train-only. Generated SFT rows keep `system_prompt` and
  `user_prompt` empty; every enabled dataset has an explicit pool mapping.
- Parameterized variants declare pool-level argument schema. Structured `prompt_args` must pass
  exact validation before SFT publication.
- System prompts enforce compact valid JSON with no Markdown or prose. User prompts state the
  exact task and minimal schema without adding unimplemented fields.
- Rename/remove a pool only with synchronized builder, catalog/training YAML, tests, and docs.

## Task Contracts

### Grounding

Return a Qwen-style array:

```json
[{"bbox_2d":[x1,y1,x2,y2],"label":"shape|icon|image|line"}]
```

Coordinates are integer `0..999` over the full input image. Empty target is `[]`. The prompt must
not request raw layer order; SFT uses the validated row-major canonical order
`row_bucket(y1,20) -> x1 -> y1 -> -area -> x2 -> y2 -> label`. One connected branched line is
one instance with a bbox covering its complete visible structure. Text-only regions and page
background are excluded.

### Context reconstruction

Shape, line, line-points, and image tasks consume a contextual crop and dynamic
`proposal_bbox_2d`. The proposal is approximate, not ground truth and not a second output frame.
Proposal and all target geometry use the same crop-local Qwen `0..999` coordinates; output may
extend beyond the proposal.

Shape follows the v9 `gt_standard` DSL, including `card`, ordered fill regions/splits, nested
`border`/`fill`, `effect`, clockwise corners, and callout geometry. It never emits a target bbox.

Line follows the v9 DSL with nested `fill` and `border`; legacy flat color/border fields are not
allowed. `points` preserves path and point order. A curved segment has exactly four sampled
points under the source contract. It never emits a target bbox.

Line-points is an exact field subset:

```json
{"type":"line","parameters":{"is_single":true,"points":[[[x1,y1],[x2,y2]]]}}
```

Only `is_single + points` are allowed. Do not request style, fill, border, color, endpoint markers,
confidence, or bbox. Multiple inner arrays represent connected branches, not independent objects.

Image type returns only one of the reviewed 13 values:

```text
photo, screenshot, chart, table, diagram, document, map, medical, microscopy,
rendering, illustration, infographic, other
```

Do not map v9 synthetic `image_type=N/A` to `other`; the v5.7 image task comes from reviewed real
data only.

## Validation

- Compile every pool with `shaft.prompting`.
- Verify all variants share the same argument and output contract.
- Verify every training YAML maps the same five dataset names to the expected pool versions.
- Recompute representative SFT targets and rendered prompts from structured/source truth.
- Reject schema extras instead of moving or inventing fields.
