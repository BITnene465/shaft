# Layout Grounding

Use this for layout raw layers inside unified `data/raw` when deriving the unified `grounding`
task.

## Raw Input

- Typical maintained layout input is `data/raw/json` plus `data/raw/images`.
- The active 2026-08-04 human batch uses compact `size + layout[]` JSON. Each layout item is
  `type + bbox + parameters?`; `bbox` is always two-corner `[x1, y1, x2, y2]`.
- The four model-facing targets are `shape`, `icon`, `image`, and `line`. `full_text` remains valid
  raw annotation but is excluded from this task.
- Compact human parameters are source truth. Grounding reads only type and bbox: line `points`
  are ignored, branches are not split, and rich shape/image fields are neither copied into the
  structured target nor fabricated when absent.
- The builder also accepts the legacy normalized `instances[].label` contract. Do not rewrite
  compact raw JSON solely to use that compatibility path.

## Cleaning

- Reject or review zero-area instances before a derived rebuild.
- Deduplicate only semantic duplicates. Same-label/same-bbox lines with different ordered paths
  are distinct routes and must remain.
- Image-only inventory is not a layout negative. A compact labeled row with no four-class target
  may become one native empty grounding row; a normalized row needs an explicitly completed empty
  layout layer.

## Derived Grounding Policy

- Split before augmentation. Current VLM test/hand-off split is
  `data/raw/splits/vlm.test.json`; do not include those test items in train-derived grounding
  data. For GT-based structured/eval data, resolve image-level split items to raw-relative JSON
  paths such as `json/gemini_0001.json` only when the JSON exists.
- Validation and test use native clean full-image rows only.
- Train keeps one native clean full-image row for every covered source image.
- The 2026-08-04 structured rebuild contains 58,440 rows from 20,060 sources. It uses
  `native clean = 20,060`, `continuous clean resize = 17,882`,
  `random padded clean = 1,995`, `degraded resize = 14,965`,
  `density crop = 2,993`, and `hard-negative crop = 545`.
- Augmentation quotas use the 19,953 positive sources as their base. The 107 sources with no
  four-class target keep exactly one native empty row and are not augmented. Hard-negative
  generation may fall below its nominal `0.03x` quota when no bbox-disjoint candidate is feasible.
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

The 2026-08-04 structured rebuild was subsequently converted one-to-one into 58,440 train SFT
rows with the v5.7 grounding prompt metadata and the existing row-major canonical target order.
Validation remains empty. Runtime prompt sampling owns the actual prompt text, so SFT row-level
`system_prompt` and `user_prompt` stay empty.
