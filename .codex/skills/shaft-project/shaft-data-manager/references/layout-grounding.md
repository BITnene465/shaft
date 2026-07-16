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
- Validation and test use native clean full-image rows only.
- Train keeps one native clean full-image row for every covered source image.
- The historical full/crop/blur/padded snapshot is retained as
  `data/archive2/grounding_layout_v5.1_bak0714`.
- The maintained `data/grounding_layout` rebuild contains 49,666 rows from 9,118 sources and uses:
  `native clean = 1.0x`, `continuous clean resize = 2.9x`,
  `random padded clean = 0.1x`,
  `degraded resize = 1.2x`, `density crop ~= 0.25x`, and
  `hard-negative crop ~= 0.03x`. The actual clean-resize count is 26,140 rather than the nominal
  26,442 because 165 small or narrow sources cannot fill all slots under the scale and
  deduplication constraints.
- Sample resize targets continuously in log-pixel space while preserving aspect ratio. Align
  output dimensions to the Qwen processor factor, cap offline linear upscale at `2x`, and avoid
  near-native duplicates. The `0.2-0.5M`, `0.5-1M`, `1-2M`, and `2-4M` ranges are balancing and
  reporting bands, not four fixed output sizes.
- Build degraded rows from selected clean resize dimensions. Apply resize first and exactly one
  bounded Gaussian blur or Gaussian noise operation second; do not materialize every
  scale/degradation combination for every source.
- Build the small padded family from native or selected clean resize views. Expand width and
  height continuously, place the source at a uniformly random offset instead of centering it,
  transform coordinates exactly, and keep the final aligned canvas inside the pixel budget.
- Do not use fixed crop-size grids as the default. Crop size should depend on source image size
  and sampled local density.
- Do not apply resize, crop, blur, noise, padding, or hard negatives to validation/test rows
  unless explicitly requested.

## Structured Row

Keep `instances` as `label + bbox`. Put source points, original image size, crop box, and source
instance indices in `extra`.

Do not regenerate SFT or previews as a side effect of layout structured rebuilds unless the user
asks for that artifact.
