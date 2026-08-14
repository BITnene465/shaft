# Line Geometry and Legacy Arrow Imports

Use this reference when line/arrow annotations are imported or when v5.7 `line_context_points` is
rebuilt. The maintained model-facing label is `line`; there is no current `point_arrow` dataset or
`arrow` output label.

## Active Compact Source

The current human truth is `data/raw/json`:

- line item: `layout[].type == "line"`;
- bbox: `layout[].bbox` in source-image boundary coordinates;
- ordered path: optional non-empty `layout[].parameters.points`;
- decoded image size must equal top-level `size`.

Grounding consumes only the line bbox. `line_context_points` consumes the ordered path but emits a
strict target subset containing only `is_single + points`. Do not infer missing paths, style,
color, or endpoint attributes.

All non-empty train paths are selected by `scripts/tasks/prepare_real_line_context_points.py`,
after excluding `data/raw/splits/vlm.test.json`. Preserve each instance identity and order. Adjacent
duplicate source points and adjacent duplicates introduced by `0..999` quantization may be removed
only in the derived target; the source instance stays intact.

Same-label/same-bbox lines can encode different routes, crossings, or directions. Compare ordered
points before any dedupe and never delete them by bbox alone.

## Normalized Imports

An importer may normalize legacy arrow/line annotations to an `instances` record before merging:

- `label: line`;
- positive two-corner `bbox`;
- ordered `linestrip` when available;
- optional exact-contract semantic attributes in `subattr`;
- importer/provenance details in `extra`.

Do not keep importer-native LabelMe `points`, `shape_type`, `group_id`, or `flags` as parallel live
fields. This normalized import contract does not authorize rewriting the active compact human
batch; merge through an explicit reviewed conversion.

## v5.7 Derived Task

`line_context_points` contains:

- 122,218 real compact-raw paths, all usable paths retained;
- 15,000 audited v9 synthetic `is_single=false` multi-branch paths.

Real contextual crops remain clean. Synthetic crops use `synthetic_realism_v1`. Both sources get a
fresh proposal/context crop and crop-local Qwen coordinate transform. Synthetic single paths do
not enter the supplement.

Validate source membership, ordered-point preservation, full bbox/path coverage by the crop,
positive quantized paths, exact target keys, task-local media existence, unique sample ids, and
one-to-one structured/SFT alignment.
