# Grounding Augmentation

Grounding targets should stay simple: `label + bbox` in the model-facing `instances`. Rich
structure belongs in `extra`.

## Split First

- Decide train/val/test before augmentation.
- Validation defaults to full-image views only.
- Do not apply resize, crop, hard negative, blur, noise, or padding augmentation to validation
  unless the user explicitly asks.

## Crop Integrity

For every crop candidate:

1. Keep GT boxes fully inside the crop and translate them into crop coordinates.
2. Ignore GT boxes with no intersection.
3. Reject the entire crop if any GT is partially intersected.

Never train on clipped partial GT boxes.

## Train Views

- `full_image`
- `continuous_resize_full` as the primary multi-resolution view
- `random_padded_full` as a small, asymmetric spatial-context view
- `degraded_resize_full` for bounded blur/noise robustness across selected resolutions
- `density_crop`
- controlled `hard_negative_crop`
- local task-owned image copies for every row

Validation remains native full-image only. Do not apply continuous resize, crop, hard negative,
JPEG, blur, noise, padding, or resize degradation to validation.

## Maintained Grounding Task

For v5.0 and later, new detection training should use the unified `grounding_layout` task.
It emits model-facing labels `shape`, `icon`, `image`, and `line`; raw `arrow` and raw `line`
instances are both normalized to `line`.

Use the current raw split manifest, especially `data/raw/splits/vlm.test.json`, as the train
exclusion source. Do not derive train rows from VLM test items.

## Current Layout Multi-Resolution Profile

The maintained `data/grounding_layout` dataset was rebuilt on 2026-07-14 with direct,
aspect-preserving resize as its primary multi-resolution augmentation. The earlier
`full + density crop + blur + padded full` snapshot is retained as
`data/archive2/grounding_layout_v5.1_bak0714`; do not mix its rows into the current dataset.

For the current 9,118-source layout split, target approximately 50,000 train rows:

- native clean full images: `1.0x` / 9,118 rows;
- continuous clean resize views: target `2.9x` / 26,442 rows;
- random padded clean views: `0.1x` / 912 rows;
- degraded resize views: `1.2x` / 10,942 rows;
- density crops: about `0.25x` / 2,280 rows;
- hard-negative crops: about `0.03x` / 274 rows.

The nominal plan gives 49,968 rows before feasibility adjustments. The actual rebuild contains
49,666 rows: 9,118 native, 26,140 continuous resize, 912 padded, 10,942 degraded resize,
2,280 density crop, and 274 hard-negative rows. A total of 165 very small or narrow sources could
not fill every requested resize slot without violating the `2x` linear-upscale cap, 10% native
deduplication, or `1.35x` same-source pixel separation. Treat the ratios as the reusable policy
and report actual feasible counts after every rebuild.
Do not materialize every augmentation combination for every image. Keep one native clean row for
every covered source, select about three additional clean scale/spatial rows per source on
average, give every normal source one degraded resize row, and give a deterministic stratified
20% subset a second degraded row. A source may contribute fewer rows when its feasible scale
range is too narrow. Padding replaces about `0.1x` of the clean-resize quota rather than expanding
the total dataset.

### Continuous Resize Sampling

For a source with `source_pixels = width * height`, use this requested target-pixel interval:

```text
lower = 200704
upper = min(configured_max_pixels, source_pixels * 4)
```

The `source_pixels * 4` cap means linear upscale is at most `2x`. If `upper < lower`, do not create
an offline upscale that violates this cap; keep the native row and let the ordinary processor
contract handle that exceptional tiny source.

Generate candidate target pixels continuously in log space. The reporting/quota bands
`0.2-0.5M`, `0.5-1M`, `1-2M`, and `2-4M` are not fixed resize levels. Fill approximately 25% of
selected clean resize rows from each feasible band while also stratifying by source-resolution
quartile and object-count quartile. A source should normally receive low-, middle-, and high-range
candidates from its own feasible interval. Two selected resize views from the same source must
differ by at least about `1.35x` in pixel count, and a resize within 10% of the effective native
processor size should be deduplicated.

Preserve aspect ratio and align final width/height to the Qwen processor factor
`patch_size * merge_size` (currently `16 * 2 = 32`). Recheck actual pixel count after alignment.
The training processor must still receive call-time `min_pixels` / `max_pixels`; the checkpoint's
persisted processor defaults are not the training budget.

### Resize Kernels

Clean resize uses high-quality antialiased resampling rather than treating every resize as
bilinear interpolation:

- clean downscale: 60% bicubic with antialiasing, 25% Lanczos, 15% area resampling;
- clean upscale: 75% bicubic, 25% Lanczos;
- when linear downscale is below `0.5x`, progressive area/Lanczos reduction may be used before the
  final aligned resize;
- area resampling is downscale-only;
- neural super-resolution is forbidden for grounding GT because it can redraw text, icons, lines,
  and boundaries.

Record the selected kernel and actual source/output sizes in structured `extra`.

### Degraded Resize Views

Build degraded rows from already selected clean resize dimensions so each degraded image has a
clean counterpart with identical geometry. Select approximately 50% Gaussian blur and 50%
Gaussian noise. Severity quotas are approximately L1 40%, L2 35%, and L3 25%:

```text
Gaussian blur:
  L1 radius = max(0.4, output_short_edge * 0.0004)
  L2 radius = max(0.8, output_short_edge * 0.0008)
  L3 radius = max(1.2, output_short_edge * 0.0015)

Gaussian noise on 0..255 pixels:
  L1 sigma = 2
  L2 sigma = 5
  L3 sigma = 10
```

Apply resize first and exactly one degradation second. Do not combine blur and noise in one row.
Do not use L3 in the lowest target-pixel band. A source's second degraded row must differ in
family, severity, or selected resolution from its first row. Clip noisy pixels back to the valid
range and keep image dimensions unchanged.

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
- Add positive `density_crop` rows at about `0.25x` of the clean full-image source count in the
  current profile.
  Density crops are useful for dense regions and small objects, but they are not a perfect match
  for the full-image business standard and should remain limited.
- Keep hard-negative crops as a separately reported, bounded `0.03x` family. Negative samples
  should remain a minority and should not dominate positive local views. The current snapshot's
  2,280 density crops and 274 hard negatives are audit counts, not one combined quota.
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

## Legacy Blur Rows

The historical `data/archive2/grounding_layout_v5.1_bak0714` snapshot has `blur_full` and
`blur_crop` rows totaling about `1.0x` of clean sources. Do not carry that row policy into a
current multi-resolution rebuild. Use the bounded `degraded_resize_full` profile above while
preserving the rule that degradation never replaces the native clean backbone.

- Apply only to train rows.
- Do not apply to validation.
- Use exactly one degradation per blur row.
- `gaussian_blur`: light-to-moderate Gaussian blur.
- `resize_blur`: downscale and resize back to the original view size.
- `jpeg_compression`: light-to-moderate JPEG round-trip compression.
- Historical `blur_full + blur_crop` counts are audit information, not the next-profile quota.
- Sample blur rows from both full-image and density-crop views. Keep the source view dimensions
  unchanged after degradation so coordinates remain unchanged.
- Keep degradation strength light to moderate. Do not use severe corruption as the default
  grounding robustness policy.

Record the selected degradation in structured row `extra.pixel_augmentation`. Keep coordinates
unchanged because the output view dimensions stay unchanged.

## Random Padded Full

Random full-image padding is geometry zoom-out, not YOLO-style input-resolution multi-scale. It is
kept as a small complementary family in the current direct-resize profile.

- The historical row count was `0.2x` of the clean full-image source count.
- The current-profile row count is about `0.1x`; these rows replace the same amount of the
  continuous-clean-resize quota so the total remains approximately 50k.
- Sample base views from native or selected continuous clean resize views, stratified by source
  resolution and object-count quartiles.
- Apply only to clean train views.
- Do not apply to density crops, blur rows, hard negatives, validation, or test rows.
- Sample total horizontal and vertical canvas expansion independently, then sample the image's
  `x` and `y` offset uniformly within the available canvas. Do not force centered or symmetric
  padding: left/right/top/bottom padding may differ, so the original image can appear anywhere on
  the expanded canvas.
- Keep expansion bounded and continuous rather than using fixed levels. The default total
  expansion range is `0.05-0.25` of the base width and independently `0.05-0.25` of the base
  height.
- Choose a base resolution that leaves room for padding, align the final canvas to the processor
  factor, and keep the final canvas within the configured pixel budget.
- Transform `bbox` / `linestrip` coordinates by the exact base resize and padding offsets before
  encoding them into Qwen `0..999` coordinates.
- Record the padding settings in structured row `extra.spatial_augmentation`.

## Density Crop

- Build candidates around instance centers or dense regions.
- Keep only crops that fully contain all retained instances.
- Limit minimum instances, maximum instances, and maximum crops per scale.
- Deduplicate by instance set and crop overlap.

## Hard Negatives

- Use only clean empty windows with no full GT and no partial overlap.
- Keep empty ratio controlled and small relative to positive/full rows. For `grounding_layout`,
  hard negatives are tracked separately at about `0.03x`, while remaining much smaller than the
  positive `density_crop` family.
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
type counts, source/output pixel-band distributions, resize-kernel counts, degradation family and
severity counts, empty-sample ratio, augmentation settings, and validation invariants. State
whether the artifact is the historical snapshot or the current multi-resolution rebuild.

## Rebuild Validation Invariants

After a grounding rebuild, check:

- Every structured row image path exists.
- No structured row image path points into `data/raw_data`.
- Number of files in `images/train` equals `structured/train.jsonl` rows.
- Number of files in `images/val` equals `structured/val.jsonl` rows.
- Train `full_image` row count equals the covered train source count.
- Selected continuous clean resize rows are approximately `2.9x` and random padded clean rows are
  approximately `0.1x` of covered train sources; together they retain the `3.0x` clean
  scale/spatial-view budget.
- Selected degraded resize rows are approximately `1.2x` of covered train sources and satisfy the
  blur/noise and severity matrices.
- Target-pixel quota bands are balanced without collapsing to four fixed resolutions.
- Every offline upscale has linear scale at most `2x`.
- Final resized dimensions are processor-factor aligned and remain within the configured pixel
  budget.
- Density crops remain about `0.25x` and hard negatives about `0.03x` for the current profile.
- Val contains only clean `full_image` rows with `pixel_augmentation.name == "none"`.
- All bboxes are positive-area and inside the row image dimensions.
