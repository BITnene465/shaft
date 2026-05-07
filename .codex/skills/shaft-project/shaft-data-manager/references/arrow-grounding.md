# Arrow Grounding

Use this for arrow layers inside unified `raw_data` and for derived `grounding_arrow`.

## Raw Schema

Current unified raw arrow annotations live in `data/raw_data/part1/json` entries whose
`annotation.layers` include `arrow`. Arrow instances use:

- each instance has `label: "arrow"`
- `bbox` stores the two-corner detection box `[x1, y1, x2, y2]`
- `linestrip` stores the route/keypoint path in source order
- `subattr` stores normalized arrow attributes
- `extra` stores source-specific details

Do not maintain arrows as separate LabelMe bbox/path shapes connected by `group_id`. The bbox
and optional path belong in the same arrow instance. Bbox-only annotations are valid and should
use an empty `linestrip`.

Do not keep live `points`, `shape_type`, or `group_id` fields on arrow instances. Source group
ids can be kept under `extra.source_group_id` only as provenance.

## Legacy Arrow Labels

Old `c0-c7` bbox labels encode arrow attributes and must not be treated as disposable labels:

| label | canonical | geometry | line_style | arrow_type |
| --- | --- | --- | --- | --- |
| c0 | Str-Sol-Single | straight | solid | single |
| c1 | Str-Das-Single | straight | dashed | single |
| c2 | Cur-Sol-Single | curved | solid | single |
| c3 | Cur-Das-Single | curved | dashed | single |
| c4 | Str-Sol-Double | straight | solid | double |
| c5 | Str-Das-Double | straight | dashed | double |
| c6 | Cur-Sol-Double | curved | solid | double |
| c7 | Cur-Das-Double | curved | dashed | double |

Old `p*` labels are point annotations. Ignore the point label names, but preserve point coordinates
in original JSON order as `linestrip`.

Do not rebuild old arrow raw annotations from downstream `single_arrow` / `double_arrow` fields
when original `c0-c7` labels are available; that loses geometry and line-style semantics.

## Connector Imports

New connector annotations are grouped LabelMe shapes. Convert a group into one arrow instance:

- bbox from `rectangle` points when present
- `linestrip` from `linestrip` or `point` shapes in original JSON order
- no coordinate sorting for point-derived linestrips
- `arrow_type=unknown` unless the source explicitly provides single/double directionality

Empty arrow layers can be kept as negative samples only when `annotation.layers` includes
`arrow`, `annotation.status.arrow` is completed, and there are no arrow instances. If the
`arrow` layer is missing, the image is unannotated for arrow and must not be used as a negative.

## Derived Grounding Policy

- Main target remains `label + bbox`.
- Preserve `linestrip`, c0-c7 attributes, source group id, and original schema in `extra`.
- Validation should stay full-image unless explicitly requested otherwise.
- When deriving model-facing grounding rows, expose only `label + bbox` in `instances`; keep
  route and style metadata in `extra`.
