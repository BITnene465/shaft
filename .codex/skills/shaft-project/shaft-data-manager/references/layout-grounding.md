# Layout Grounding

Use this for layout raw layers inside unified `data/raw` when deriving the unified `grounding`
task.

## Raw Input

- Typical maintained layout input is `data/raw/json` plus `data/raw/images`.
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

- Split before augmentation. Current VLM test/hand-off split is
  `data/raw/splits/vlm.test.json`; do not include those test items in train-derived grounding
  data. For GT-based structured/eval data, resolve image-level split items to raw-relative JSON
  paths such as `json/gemini_0001.json` only when the JSON exists.
- Validation uses full-image only.
- Train keeps one clean full-image row for every covered source image.
- Default train augmentation for `grounding_layout` is:
  - `density_crop`: about `0.3x`, including only a small minority of negative samples.
  - `blur_full + blur_crop`: `1.0x`, using light-to-moderate Gaussian blur, resize blur, or
    JPEG compression.
  - `random_padded_full`: `0.2x`, applied only to clean full-image rows.
- Do not use fixed crop-size grids as the default. Crop size should depend on source image size
  and sampled local density.
- Do not apply crop, blur, padding, or hard negatives to validation/test rows unless explicitly
  requested.

## Structured Row

Keep `instances` as `label + bbox`. Put source points, original image size, crop box, and source
instance indices in `extra`.

Do not regenerate SFT or previews as a side effect of layout structured rebuilds unless the user
asks for that artifact.
