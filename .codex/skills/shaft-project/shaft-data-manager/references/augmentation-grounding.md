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
- `random_padded_full` as a small subset of clean full-image train rows
- `density_crop`
- `blur_full` and `blur_crop` as bounded robustness rows
- controlled `hard_negative_crop`
- local task-owned image copies for every row

Validation remains full-image only. Do not apply crop, hard negative, JPEG, blur, or resize
degradation to validation.

## Maintained Grounding Task

For v5.0 and later, new detection training should use the unified `grounding_layout` task.
It emits model-facing labels `shape`, `icon`, `image`, and `line`; raw `arrow` and raw `line`
instances are both normalized to `line`.

Use the current raw split manifest, especially `data/raw/splits/vlm.test.json`, as the train
exclusion source. Do not derive train rows from VLM test items.

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

For `grounding_layout`, train augmentation should prefer high-density local regions instead of
mechanically cropping every image:

- Keep the full-image row for every train sample.
- Do not reference raw images directly from structured rows. Copy or render every row image under
  the task dataset directory, for example `data/grounding_layout/images/train/`.
- For train, each covered source image contributes one clean `full_image` row.
- Add `density_crop` rows at about `0.3x` of the clean full-image source count by default.
  Density crops are useful for dense regions and small objects, but they are not a perfect match
  for the full-image business standard and should remain limited.
- Include a small controlled number of negative crop samples inside the density-crop budget.
  Negative samples should remain a minority and should not dominate positive local views.
- Generate hard negatives only from clean raw sources whose GT is complete for the task. A hard
  negative is a crop with no full GT and no partial GT overlap; do not use partially annotated
  raw JSON as a source for negative sampling.
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

## Blur Rows

Grounding uses bounded light-to-moderate pixel degradation as additional train robustness rows.
Do not replace clean full-image rows or clean crop rows with blur rows.

- Apply only to train rows.
- Do not apply to validation.
- Use exactly one degradation per blur row.
- `gaussian_blur`: light-to-moderate Gaussian blur.
- `resize_blur`: downscale and resize back to the original view size.
- `jpeg_compression`: light-to-moderate JPEG round-trip compression.
- The combined `blur_full + blur_crop` row count should default to `1.0x` of the clean full-image
  source count.
- Sample blur rows from both full-image and density-crop views. Keep the source view dimensions
  unchanged after degradation so coordinates remain unchanged.
- Keep degradation strength light to moderate. Do not use severe corruption as the default
  grounding robustness policy.

Record the selected degradation in structured row `extra.pixel_augmentation`. Keep coordinates
unchanged because the output view dimensions stay unchanged.

## Random Padded Full

Random full-image padding is a small zoom-out augmentation, not the detection backbone.

- Default row count is `0.2x` of the clean full-image source count.
- Apply only to clean full-image train rows.
- Do not apply to density crops, blur rows, hard negatives, validation, or test rows.
- Sample padding ratios from `0.1` to `0.2` per side unless the task explicitly overrides this.
- Transform `bbox` coordinates by exact padding offsets and clamp to the new padded canvas.
- Record the padding settings in structured row `extra.spatial_augmentation`.

## Density Crop

- Build candidates around instance centers or dense regions.
- Keep only crops that fully contain all retained instances.
- Limit minimum instances, maximum instances, and maximum crops per scale.
- Deduplicate by instance set and crop overlap.

## Hard Negatives

- Use only clean empty windows with no full GT and no partial overlap.
- Keep empty ratio controlled and small relative to positive/full rows. For `grounding_layout`,
  hard negatives are included as a minority part of the `density_crop` budget rather than a large
  independent augmentation family.
- Do not augment hard negatives with blur by default.
- Hard negative candidates should use the same image-relative crop philosophy as positive crops,
  then be sampled down after candidates are generated.
- Hard negative correctness depends on raw annotation completeness. If an annotation source is
  partial, remove it from the training raw/split rather than adding visual heuristics to the
  negative sampler.

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
- Train `full_image` row count equals the covered train source count.
- Train `density_crop` row count is about `0.3x` of covered train sources by default, with hard
  negatives only as a small minority inside that budget.
- Train `blur_full + blur_crop` row count is approximately `1.0x` of covered train sources by
  default.
- Train `random_padded_full` row count is approximately `0.2x` of covered train sources by
  default.
- Val contains only clean `full_image` rows with `pixel_augmentation.name == "none"`.
- All bboxes are positive-area and inside the row image dimensions.
