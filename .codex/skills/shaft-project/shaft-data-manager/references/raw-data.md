# Raw Data Handling

Raw directories are the source of truth. Derived datasets should be rebuilt from raw data, not
patched directly unless the user explicitly asks for a temporary experiment.

The day-to-day raw data entrypoint in this repo is `data/raw_data`. It should normally be a symlink
to the shared managed root `/mnt/data/shared_data/vlm_data/raw_data`, so repo-relative operations on
`data/raw_data` directly affect the shared dataset. Legacy `raw_layout` and `raw_arrow` directories
were merged into this unified source; do not start new maintenance work from the legacy split
directories.

Do not use broad destructive commands such as `rm -rf data/raw_data`, `rm -rf data/raw_data/*`, or
unchecked bulk deletes. Because `data/raw_data` points at the shared managed dataset, do not plan
routine `rsync` synchronization. For large replacements or reorganizations, operate directly through
`data/raw_data`, state that it updates the shared dataset, and verify file counts, active image/json
coverage, and split summaries afterward.

## Directory Rules

- Current maintained `raw_data` is partitioned by annotation stage:
  - `part1/images` and `part1/json` contain samples with non-empty maintained instances.
  - `part2/images` and `part2/json` contain unannotated inventory samples with
    `annotation.layers=[]`, `annotation.status={}`, and `instances=[]`.
- JSON `image_path` is always relative to the `raw_data` root, for example
  `part1/images/00001.png` or `part2/images/pic_1001.png`.
- Split files stay task-named by model task, not raw layer. Current maintained task splits are
  `grounding_train.txt`, `grounding_val.txt`, and `point_arrow_val.txt`. Each non-empty line is a
  JSON path relative to `raw_data`, for example `part1/json/00001.json`, not a bare sample stem.
- Keep annotations and images in predictable sibling directories inside each part. If an imported
  source uses another name such as `figure`, normalize it before treating the raw directory as
  maintained.
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

Empty inventory samples are valid in unified raw data and should live in `part2`:
`annotation.layers=[]`, `annotation.status={}`, and `instances=[]`. They are future annotation
inventory and must not be treated as negative samples for any task.

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
