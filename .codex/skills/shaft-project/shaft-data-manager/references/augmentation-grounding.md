# Grounding Augmentation

Use this reference for the maintained real `grounding_layout` dataset. Model-facing targets are
only `label + bbox`; source-only details remain in raw data or minimal derived audit metadata.

## Source and Split Contract

- Source truth is compact `data/raw/json` plus `data/raw/images`.
- Use `data/raw/splits/grounding_layout.train.txt` and `.val.txt` explicitly.
- Exclude `data/raw/splits/vlm.test.json` from train selection.
- Grounding labels are `shape`, `icon`, `image`, and `line`. `full_text` is not a target.
- Ordered line points and reconstruction parameters do not affect grounding targets.
- A labeled source with no four-class target keeps one native empty row and receives no
  augmentation. Image-only inventory is not a negative sample.

Split before augmentation. Validation/test contains only native clean full-image rows; never add
resize, padding, degradation, density crop, or hard-negative views there.

## Maintained Profile

Rebuild with `scripts/tasks/build_grounding_structured.py` and the `layout_multiscale_v1` profile.
The verified 2026-08-14 artifact has 20,060 train sources and 58,440 rows:

| view | rows | target ratio over 19,953 positive sources |
| --- | ---: | ---: |
| native clean full | 20,060 | one per source |
| continuous clean resize | 17,882 | about `0.9x` |
| random padded clean | 1,995 | about `0.1x` |
| degraded resize | 14,965 | about `0.75x` |
| density crop | 2,993 | about `0.15x` |
| hard-negative crop | 545 | bounded by `0.03x` feasibility |

The 107 sources containing only non-target `full_text` account for the native empty rows.
Additional quotas use positive sources only. Hard-negative count may stay below its nominal quota
when no safe crop exists.

Every row uses a task-local image under `data/grounding_layout`; structured/SFT rows must never
point into `data/raw`. SFT conversion is one-to-one and does not change augmentation distribution.

## Continuous Resize

For `source_pixels = width * height`, request target pixels in:

```text
lower = 200704
upper = min(2000000, source_pixels * 4)
```

The `source_pixels * 4` cap limits linear upscale to `2x`. Sample candidates continuously in log
space, preserve aspect ratio, and align final dimensions to the Qwen processor factor `32`.
Selected views from the same source should differ by at least about `1.35x` in pixel count; remove
a resize within 10% of the effective native processor size.

Clean downscale uses antialiased bicubic/Lanczos/area resampling; area is downscale-only. Clean
upscale uses bicubic/Lanczos. Do not use neural super-resolution because it can redraw target
boundaries. Record requested/actual sizes and kernel in structured `extra`.

## Degraded Resize

Build a degraded row at an already selected clean-resize geometry. Apply resize first and exactly
one of Gaussian blur or Gaussian noise. Keep output dimensions and coordinates unchanged.

- family balance: about 50% blur and 50% noise;
- severity balance: L1/L2/L3 about 40/35/25%;
- no L3 in the lowest pixel band;
- a second degraded view from one source must differ by family, severity, or resolution.

This single-degradation rule is grounding-specific. Synthetic reconstruction uses a different
`synthetic_realism_v1` contract.

## Random Padded Full

- Sample from native or clean-resize train views only.
- Expand horizontal and vertical canvas independently by `0.05..0.25`.
- Sample the source offset independently; padding is intentionally asymmetric.
- Align the final canvas to factor 32 and keep it within the configured pixel budget.
- Transform bbox coordinates with the exact resize and padding offsets.
- Record the operation in `extra.spatial_augmentation`.

Padding replaces part of the clean-resize budget; it does not increase the overall clean-view
quota.

## Crop Integrity

For density and hard-negative candidates:

1. Retain only boxes fully inside the crop and translate them exactly.
2. Ignore boxes with no intersection.
3. Reject the whole crop if any target box is partially intersected.

Density crops are image-relative, biased toward instance centers/dense regions, and scored by the
number of fully contained targets. Reject nearly full-image duplicates and deduplicate by retained
instance set plus crop overlap.

Hard negatives require complete raw annotation and zero full or partial target overlap. Keep them
rare, separate in reporting, clean by default, and never manufacture them from partially annotated
or image-only sources.

## Bbox Boundary Rule

Source bboxes use image-boundary coordinates: `x2 == width` and `y2 == height` are valid. Clip in
`[0,width] x [0,height]`, verify positive area, then quantize to Qwen `0..999`. Do not clip to
`width-1/height-1` before quantization; that can collapse a right/bottom 1px line.

## Rebuild Validation

- Row counts equal task-local media file counts for each split.
- Every source has exactly one native row; only positive train sources have augmented views.
- Clean resize, padding, degraded, density, and hard-negative counts match the current profile.
- Offline scale is at most `2x`; generated dimensions are factor-32 aligned and within 2M pixels.
- Every bbox has positive area and fits its row image under the boundary-coordinate rule.
- Validation contains only native clean rows.
- Structured and SFT rows align one-to-one and targets recompute exactly.
- Build summary records seed, profile, view counts, pixel bands, kernels, degradation, empty rows,
  split boundaries, and media invariants.
