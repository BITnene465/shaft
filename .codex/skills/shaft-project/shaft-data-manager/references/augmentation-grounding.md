# Grounding Augmentation

Grounding targets should stay simple: `label + bbox` in the model-facing `instances`. Rich
structure belongs in `extra`.

## Split First

- Decide train/val/test before augmentation.
- Validation defaults to full-image views only.
- Do not apply crop, hard negative, or blur augmentation to validation unless the user
  explicitly asks.

## Crop Integrity

For every crop candidate:

1. Keep GT boxes fully inside the crop and translate them into crop coordinates.
2. Ignore GT boxes with no intersection.
3. Reject the entire crop if any GT is partially intersected.

Never train on clipped partial GT boxes.

## Train Views

- `full_image`
- `full_image_blur` as a bounded replacement subset of full-image train rows
- `density_crop`
- controlled `hard_negative_crop`
- local task-owned image copies for every row

Validation remains full-image only. Do not apply crop, hard negative, JPEG, blur, or resize
degradation to validation.

## Maintained Grounding Subtasks

Generate structured data independently for these detection subtasks:

| dataset | target labels | required raw coverage |
| --- | --- | --- |
| `grounding_arrow` | `arrow` | `arrow` |
| `grounding_layout` | `icon`, `image`, `shape` | `layout` |
| `grounding_shape` | `shape` | `layout` |
| `grounding_icon_image` | `icon`, `image` | `layout` |

Use `data/raw_data/splits/grounding_train.txt` and `grounding_val.txt` as the split source.
Filter by required raw coverage before deriving each subtask. For example, `part1` arrow-only
samples are valid for `grounding_arrow`, but must not become layout, shape, icon/image, or
other grounding negatives.

## Maintained Generator

Use `scripts/tasks/build_grounding_structured.py` for rebuilds instead of temporary scripts. Keep
the command line in run notes or the generated README, but keep this skill focused on method:

- derive each grounding subtask independently from the same image-level split source;
- filter by required raw coverage before producing rows;
- write task-local row images instead of referencing raw images;
- keep train-only augmentation out of validation;
- clean stale generated images so each image file is referenced by a structured row;
- seed randomness by stable source identity so rebuilds are approximately reproducible when raw
  data, split files, PIL behavior, and script code are unchanged.

## Density Crop Selection

For each grounding subtask, train augmentation should prefer high-density local regions instead
of mechanically cropping every image:

- Keep the full-image row for every train sample.
- Do not reference raw images directly from structured rows. Copy or render every row image under
  the task dataset directory, for example `data/grounding_arrow/images/train/`.
- For train, each covered source image contributes one full-image row total: either clean
  `full_image` or degraded `full_image_blur`. Clean full and blur full counts should add up to
  approximately the original source full-image count. Blur is a bounded replacement subset, not a
  duplicate row for every clean full-image row.
- Keep positive crop volume controlled by task so local views do not dominate full-image rows.
- Candidate crops should be random but density-biased, not fixed-size tiles. Sample crop width
  and height from image-relative ranges so the view scale follows the source image size. Use a
  wider crop-ratio range for large images and reject crops that are effectively full-image
  duplicates.
- Use image-relative crop size ranges keyed by source resolution rather than fixed-size tiles.
- Candidate centers may be lightly randomized around target-instance centers or dense regions.
- Bias crop centers around target instances or target clusters, then score by contained target
  count.
- Score candidates primarily by the number of fully contained task GT instances.
- Reject a crop if any task GT partially intersects the crop boundary. Do not train on clipped
  partial GT.
- Reject crops that are too close to the full image; a local crop should materially reduce the
  visible canvas, not duplicate the full-image row.
- Require enough fully contained target GT before accepting a crop; tune this threshold by task
  density instead of using one global value.
- Deduplicate selected crops by high overlap and identical contained instance sets.
- Validation remains full-image only.

## Full-Image Blur Rows

Grounding may use light pixel degradation as a train full-image replacement subset. Do not double
the full-image portion by adding a blur copy for every clean full-image row.

- Apply only to train full-image blur rows.
- Do not apply to density crops or hard negatives by default.
- Do not apply to validation.
- Use exactly one degradation per full-image blur row.
- `jpeg_blur`: lightly round-trip through JPEG.
- `resize_blur`: for high-resolution views only, downscale and resize back to the original view
  size.
- JPEG blur plus resize blur row counts should be at most half of the full-image source count by
  default. Within blur rows, sample resize blur only when the source view is high resolution;
  otherwise use JPEG blur.
- Keep resize blur as a minority/high-resolution-specific degradation; non-high-resolution views
  should fall back to JPEG-style blur.

Record the selected degradation in structured row `extra.pixel_augmentation`. Keep coordinates
unchanged because the output view dimensions stay unchanged.

## Density Crop

- Build candidates around instance centers or dense regions.
- Keep only crops that fully contain all retained instances.
- Limit minimum instances, maximum instances, and maximum crops per scale.
- Deduplicate by instance set and crop overlap.

## Hard Negatives

- Use only clean empty windows with no full GT and no partial overlap.
- Keep empty ratio controlled and small relative to positive/full rows. The maintained default
  hard-negative sampling target is `0.008` of `full + positive crop` rows.
- Do not augment hard negatives with blur by default.
- Hard negative candidates should use the same image-relative crop philosophy as positive crops,
  then be sampled down after candidates are generated.

## Deduplication

Deduplicate near-identical crop views by instance set and crop overlap. The goal is to avoid
letting repetitive views dominate training.

## Minimal Rebuild Summary

For derived datasets, prefer a short README over long reports. Include split row counts, view
type counts, empty-sample ratio, augmentation settings, and validation invariants.

## Rebuild Validation Invariants

After a grounding rebuild, check:

- Every structured row image path exists.
- No structured row image path points into `data/raw_data`.
- Number of files in `images/train` equals `structured/train.jsonl` rows.
- Number of files in `images/val` equals `structured/val.jsonl` rows.
- Train `full_image + full_image_blur` row count equals the covered train source count.
- Train `full_image_blur` row count is no more than half of covered train sources unless the user
  explicitly asks for a stress dataset.
- Val contains only clean `full_image` rows with `pixel_augmentation.name == "none"`.
- All bboxes are positive-area and inside the row image dimensions.
