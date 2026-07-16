# Raw Data Handling

Raw directories are the source of truth. Derived datasets should be rebuilt from raw data, not
patched directly unless the user explicitly asks for a temporary experiment.

The day-to-day raw data entrypoint in this repo is `data/raw`. Legacy `raw_layout`, `raw_arrow`,
and `raw_data` directories are historical; do not start new maintenance work from the legacy split
directories.

Do not use broad destructive commands such as `rm -rf data/raw`, `rm -rf data/raw/*`, or unchecked
bulk deletes. For large replacements or reorganizations, operate directly through `data/raw`,
state the scope, and verify file counts, active image/json coverage, and split summaries afterward.

## Directory Rules

- Current maintained `raw` is flat:
  - `images/` contains source images.
  - `json/` contains maintained GT annotation JSON for the labeled subset.
  - Many images may not have JSON and are image-only inventory/test items.
- JSON `image_path` is always relative to the `raw` root, for example `images/prod_000995.png`.
- Current maintained split manifests live under `data/raw/splits/`. `vlm.test.json` is the
  canonical VLM test/hand-off split. It is an image-level JSON manifest with `items[].image_path`
  and optional `items[].id`; it is not a train split and must not be silently mixed into training.
  Tools that need GT may resolve each item to `json/<id>.json` or `json/<image_stem>.json`, but
  missing JSON means the item is image-only and cannot contribute to metric computation.
- Keep annotations and images in predictable sibling directories. If an imported source uses
  another name such as `figure`, normalize it before treating the raw directory as maintained.
- Count labeled samples by JSON coverage. Extra images without JSON are unlabeled unless the
  user says otherwise.
- Maintain a short `README.md` inside each raw directory. It should describe current schema,
  known caveats, and the last important cleanup in natural language.
- Avoid long JSON reports for routine work. Use a report only when the user asks for an audit
  artifact or when a script needs a machine-readable handoff.

## Maintained Raw Schema

Raw JSON files should be maintained in a normalized schema, not in importer-native LabelMe
`shapes` form:

- top-level fields: `schema`, `image_path`, `image_width`, `image_height`, `annotation`,
  `instances`, `extra`
- `annotation.layers` records covered raw annotation layers in stable order: `layout`, then `arrow`
- `annotation.status` records per-layer workflow status: `preannotated`, `annotated`,
  `needs_revision`, or `completed`
- layout instance: `label`, two-corner `bbox: [x1, y1, x2, y2]`, `extra`
- arrow instance: `label`, two-corner `bbox`, `linestrip`, `subattr`, `extra`

Image-only inventory/test samples are valid in unified raw data. They are future annotation
inventory or hand-off inputs and must not be treated as negative samples for any task.

Do not keep `points`, `shape_type`, `group_id`, or `flags` as live instance fields. Preserve
source-only details inside `extra` only when they are needed for traceability. This prevents
future rebuild scripts from accidentally depending on import artifacts instead of the maintained
raw contract.

## Write Policy

- Back up the JSON directory before batch edits.
- Do not regenerate previews unless requested.
- Do not delete raw images during annotation cleanup.
- Prefer in-place raw JSON cleanup only after the rule is clear and reversible through backup.

## Image Size Policy

If raw images exceed the stable PIL/training loading range, normalize them before derived rebuilds
instead of disabling PIL safety globally. Use `max(width, height) <= 4096` with aspect ratio
preserved unless the user specifies a different cap. When resizing a JSON-covered image, scale
all image-size fields and annotation coordinates by the exact same factors:

- layout: `image_width`, `image_height`, `bbox`, and trace `extra.source_points`
- arrow: `image_width`, `image_height`, `bbox`, and `linestrip`

Regenerate previews after image resize, because old previews no longer reflect raw coordinates.

## Parallelism

Large mechanical data work can use multiprocessing, but default to conservative worker counts.
Use 8 workers unless the user asks for more and the job is known not to stress memory or file I/O.
