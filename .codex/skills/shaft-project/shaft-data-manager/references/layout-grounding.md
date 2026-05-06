# Layout Grounding

Use this for layout layers inside unified `raw_data` and for derived `grounding_layout`.

## Raw Input

- Typical raw input is `data/raw_data/json` plus `data/raw_data/images`.
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

- Split before augmentation. Prefer `data/raw_data/splits/layout_train.txt` and
  `data/raw_data/splits/layout_val.txt` for layout derivation.
- Validation uses full-image only.
- Train keeps full images and may add large density/sliding crops plus controlled hard negatives.
- Use large crop sizes by default: 896, 1024, 1152, 1280.
- Do not create local crops smaller than 896 unless the user changes strategy.
- For positive crops, generate at most one light augmentation.

## Structured Row

Keep `instances` as `label + bbox`. Put source points, original image size, crop box, and source
instance indices in `extra`.

Do not regenerate SFT or previews as a side effect of layout structured rebuilds unless the user
asks for that artifact.
