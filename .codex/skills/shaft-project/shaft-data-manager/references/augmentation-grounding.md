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
- `density_crop`
- `sliding_window_crop`
- controlled `hard_negative_crop`
- light positive view augmentation only when it targets a clear task failure mode

Do not use JPEG compression as a default grounding augmentation. It mostly simulates image
degradation and has weak expected value for precise bbox localization compared with view
construction, scale coverage, clean hard negatives, and dense-region crops.

## Density Crop Selection

For `grounding_arrow` and `grounding_layout`, train augmentation should prefer high-density local
regions instead of mechanically cropping every image:

- Keep the full-image row for every train sample.
- Generate at most 2 positive crop rows per source image.
- Candidate crops may come from sliding windows, instance-center windows, or other deterministic
  window proposals.
- Score candidates primarily by the number of fully contained task GT instances.
- Reject a crop if any task GT partially intersects the crop boundary. Do not train on clipped
  partial GT.
- Reject crops that are too close to the full image; a local crop should materially reduce the
  visible canvas, not duplicate the full-image row.
- Require enough contained GT before accepting a crop. The threshold can be task-specific, but the
  default should favor dense regions rather than sparse positives.
- Deduplicate selected crops by high overlap and identical contained instance sets.
- Validation remains full-image only.

## Context Padding Jitter

Use `context_padding_jitter` as the preferred lightweight positive augmentation when a new
variant is needed. This is a zoom-out view: shrink the existing positive view, paste it onto a
same-size clean canvas with random offset, and transform all `bbox` and arrow `linestrip`
coordinates with the same scale and offset.

Recommended conservative defaults:

- Train only; never apply to validation.
- Positive views only; never apply to hard negatives.
- Use shrink-only scale, e.g. `0.75-0.95`; do not enlarge and crop.
- Keep the output canvas size unchanged.
- Use a clean background, normally white or an edge-sampled near-background color.
- Reject the augmented sample if any transformed bbox or linestrip point becomes invalid.
- Record the transform in `extra.augmentation`, for example:
  `{"name": "context_padding_jitter", "scale": 0.86, "offset": [43, 71]}`.

This complements density and sliding-window crops: those mostly create zoom-in views, while
context padding jitter creates controlled zoom-out views without degrading image quality.

## Sliding Window

- Use multiple tile scales.
- Keep only instances fully inside the tile.
- Require a minimum instance count so sparse positives do not dominate.
- Tiles with no full instance and no partial overlap may enter the hard-negative pool.

## Density Crop

- Build candidates around instance centers or dense regions.
- Keep only crops that fully contain all retained instances.
- Limit minimum instances, maximum instances, and maximum crops per scale.
- Deduplicate by instance set and crop overlap.

## Hard Negatives

- Use only clean empty windows with no full GT and no partial overlap.
- Keep empty ratio controlled; for layout grounding, 3%-5% is a good default.
- Do not augment hard negatives with blur by default.

## Deduplication

Deduplicate near-identical crop views by instance set and crop overlap. The goal is to avoid
letting repetitive views dominate training.

## Minimal Rebuild Summary

For derived datasets, prefer a short README over long reports. Include split row counts, view
type counts, empty-sample ratio, augmentation settings, and validation invariants.
