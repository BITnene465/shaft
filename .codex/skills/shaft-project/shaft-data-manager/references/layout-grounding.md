# Layout Grounding

Use this for layout raw layers inside unified `raw_data` when deriving the unified `grounding`
task.

## Raw Input

- Typical maintained layout input is `data/raw_data/part1/json` plus
  `data/raw_data/part1/images`.
- Layout-labeled samples are JSON entries whose `annotation.layers` include `layout`.
- Current maintained layout labels are `icon`, `image`, and `shape`.
- Current maintained raw schema is `shaft.raw_data.v1`.
- Each instance is `label + bbox + extra`; `bbox` is always two-corner
  `[x1, y1, x2, y2]`.
- Do not maintain layout instances with live `points`, `shape_type`, or `group_id`. If source
  polygon/rectangle points are useful for traceability, keep them under `extra.source_points`.

## Cleaning

- Remove zero-area instances.
- Deduplicate same-label same-bbox instances within each JSON file.
- Keep image inventory samples as raw JSON with `annotation.layers=[]`; do not count them as
  layout negatives unless a completed `layout` layer explicitly contains no layout instances.

## Derived Grounding Policy

- Split before augmentation. Prefer `data/raw_data/splits/grounding_train.txt` and
  `data/raw_data/splits/grounding_val.txt` for grounding derivation. Split entries are
  raw-data-relative JSON paths such as `part1/json/gemini_0001.json`, not bare stems.
- Validation uses full-image only.
- Train keeps full images and may add random image-relative density crops plus controlled hard
  negatives.
- Do not use fixed crop-size grids as the default. Crop size should depend on source image size
  and sampled local density.
- Light JPEG/resize blur can be applied to train full/crop rows without creating extra rows.

## Structured Row

Keep `instances` as `label + bbox`. Put source points, original image size, crop box, and source
instance indices in `extra`.

Do not regenerate SFT or previews as a side effect of layout structured rebuilds unless the user
asks for that artifact.
