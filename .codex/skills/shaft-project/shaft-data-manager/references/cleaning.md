# Data Cleaning

Use cleaning to make the raw annotation truth consistent before building derived datasets.

## Standard Checks

- Parse all JSON files and count parse errors.
- Check JSON/image alignment.
- Count labels before and after cleanup.
- Check zero-area boxes: remove instances whose normalized bbox has zero width or zero height.
- Check per-file duplicates by `(label, bbox)` when the task is bbox grounding.
- For structured arrows or grouped annotations, do not dedupe solely by `label + bbox` if
  `linestrip`, `subattr`, or source group id differ; use the domain reference.
- Preserve source-specific attributes in `extra` or `subattr` instead of encoding them in the
  model-facing label.

## Duplicate Policy

- Same file, same label, same bbox: keep the first instance unless the domain reference says the
  repeated instance carries different structure.
- Same box with different route, direction, style, group id, or task layer is not automatically
  a duplicate.
- Same basename across incoming datasets: inspect source paths and image identity before merging.
- Same image with different task labels can coexist when the labels represent different domains
  or annotation layers.

## Output

After cleanup, keep the user-facing summary short:

- files touched
- instances removed
- duplicate count after cleanup
- remaining caveats

Put durable notes in the raw directory README.
