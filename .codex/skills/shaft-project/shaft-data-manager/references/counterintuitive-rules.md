# Counterintuitive Data Rules

These are the rules that are easy to get wrong because the wrong action looks reasonable at
first glance. Read this reference before doing raw data cleanup, merge, or rebuild work.

## Raw Data Is Usually the Truth

Do not use a derived dataset to rewrite raw semantics when the original raw annotation still
exists. Derived structured/SFT data may have already dropped attributes, normalized labels, or
collapsed task details.

Example: old arrow structured rows may say only `single_arrow` or `double_arrow`, but the raw
`c0-c7` bbox label also encodes straight/curved and solid/dashed. Use the raw `c0-c7` label as
the semantic source.

## A Label May Encode Attributes, Not Just Class

Do not blindly replace source labels with the final model-facing class. First ask whether the
source label encodes attributes that must survive in `subattr` or `extra`.

Example: legacy arrow labels all become model-facing `label: "arrow"`, but `c0-c7` still map to
`arrow_type`, `geometry`, and `line_style`.

## Missing Semantics Should Stay Unknown

Do not infer a missing attribute just to make the dataset look complete. If the source does not
distinguish single-headed and double-headed arrows, keep `arrow_type=unknown` unless the user
approves a heuristic.

Example: connector groups provide geometry shapes, not single/double arrow labels.

## Point Order Is Annotation Order

Do not sort points by coordinate unless the source schema explicitly says order is irrelevant.
For route-like annotations, the JSON order is often the only path information.

Example: connector `rectangle + point` groups become `linestrip` by preserving point shape order
from the original JSON.

## LabelMe Fields Are Import Artifacts

Do not keep LabelMe `shape_type`, `points`, or `group_id` as live fields in maintained raw
annotations. They are useful when importing, but raw maintenance should normalize them into a
stable schema: `bbox` for boxes, `linestrip` for arrow paths, and `extra` for source provenance.

Example: arrow bbox and route information should live in one instance. `group_id` may be kept in
`extra.source_group_id` for traceability, but downstream code must not rely on it to join shapes.

## Same Image Name Does Not Mean Same Annotation Layer

Do not collapse files only because basenames match. The same image can have separate icon/image,
shape, arrow, connector, or negative-sample annotations.

Before merging, decide whether the incoming JSON is a replacement, another layer, or a negative
sample.

## Unlabeled Images Are Not Labeled Samples

When counting raw dataset coverage, JSON coverage defines labeled samples unless the user says
to include unlabeled images. Extra images in an image directory are not automatically missing
annotations; they may be pre-annotation inventory.

## Zero-Area Boxes Are Noise

Zero-width or zero-height boxes are usually annotation errors for bbox grounding. Delete those
instances from raw annotations after backing up, then document the cleanup in the raw README.

## Duplicate Means Same Semantics, Not Just Same Box

Same label and same bbox is a useful duplicate check, but grouped tasks can carry route, style,
direction, or source-group differences. Do not dedupe arrows or connectors without checking
`linestrip`, `subattr`, and source group identity.

## Validation Should Stay Boring

Do not apply train-only augmentation to validation data. Validation should normally be full-image
and deterministic, even if train gets crops, hard negatives, or blur.

Exception: `point_arrow` validation is crop-level by task definition. It should be deterministic
arrow crops, not raw full images.

## VLM Samples And Requests Are Image-First

For Qwen3VL training, eval, and runtime inference, multimodal user content must be image-first:
image content comes before the text instruction in the user message.

Use this order for all model-facing paths:

- HF/local chat messages: `{"type": "image"}` before `{"type": "text", "text": ...}`.
- OpenAI/vLLM-compatible messages: `{"type": "image_url", ...}` before
  `{"type": "text", "text": ...}`.
- Derived SFT rows and prompt/template helpers must preserve the same semantic order.

Do not switch to text-first when writing temporary eval scripts, production wrappers, or data
conversion utilities. Message order is part of the runtime contract; results from image-first and
text-first runs should not be treated as directly comparable unless that difference is explicitly
being tested. New eval/review summaries should record `message_order: image_first` when they create
model requests.

## Clean Full Images Are The Grounding Backbone

For grounding train data, every covered source image should keep one clean `full_image` row. This
row is the detection backbone and must not be replaced by crop, blur, or padded variants.

The historical snapshot at `data/archive2/grounding_layout_v5.1_bak0714` contains clean full images,
density/hard-negative crops, `1.0x` legacy blur rows, and `0.2x` random-padded rows. Do not mix
those rows with the maintained multi-resolution dataset.

The maintained `data/grounding_layout` rebuild policy is:

- native clean full images: `1.0x`
- continuous clean resize views: about `2.9x`
- random padded clean views: about `0.1x`
- degraded resize views: about `1.2x`, with exactly one bounded Gaussian blur or noise operation
- density crops: about `0.25x`
- hard-negative crops: about `0.03x`

These are bounded sampled views, not a Cartesian product of source images, scales, kernels, and
degradation levels.

Validation and VLM test data should remain clean full-image only.

Synthetic layout detection is a separate replay source named `grounding_layout_sync`. Its source
archive already has enough rendered diversity, so keep exactly one clean full-image view from the
source train split and control exposure through the training catalog weight. Do not run the real
layout multiscale augmentation profile on it, do not merge its files into `grounding_layout`, and
do not use the synthetic validation split as formal model evaluation.

## Canonical Order Needs GT Validation

For detection-style SFT targets, canonical order is part of the training signal, not harmless
formatting. Do not pick or change `grounding_layout` target order only because the rule sounds
natural or is simple to implement.

Before regenerating SFT data with a new order, analyze the current GT distribution and record the
result. At minimum check parent-before-child behavior for large containing boxes, y-direction
backtracking, stability under small bbox jitter, dense-sample truncation risk, and existing model
response sortedness as a secondary diagnostic.

The 2026-07-09 review found that the old `row_bucket(y_center)` order was a poor fit for business
diagram images: large containers were often placed after their children. Keep the living analysis
under `notes/canonical_order/`.

## JPEG Is Optional, Not An Implicit Default

JPEG compression may be used by an explicitly selected legacy or experimental profile, but the
next direct-resize profile uses Gaussian blur and Gaussian noise only. Do not silently add JPEG,
resize-back blur, or combined corruptions to that profile. Any optional degradation must remain
bounded and must not replace the native clean row.

## Zoom Out Is Different From Pixel Budget

Processor pixel budget controls the final visual token budget, but it does not replace geometry
augmentation. The current profile obtains scale diversity through direct, aspect-preserving resize;
padding remains a distinct zoom-out transform at a bounded `0.1x` quota. Sample padding
asymmetrically so the image can appear anywhere on the expanded canvas, and transform `bbox` /
`linestrip` coordinates exactly. Padding replaces `0.1x` of the continuous-clean-resize budget;
it does not increase the approximately 50k total.

The training call must still pass runtime `min_pixels` / `max_pixels`. Persisted processor
defaults inside a checkpoint are not proof that the intended training pixel budget is active.

## Multi-Resolution Does Not Mean Fixed Five Sizes

Sample target pixel counts continuously in log space from each source's feasible interval. Cap
offline enlargement at `2x` linear scale, align output dimensions to the processor factor, and
select only a bounded number of sufficiently separated views per source. Pixel ranges such as
`0.2-0.5M`, `0.5-1M`, `1-2M`, and `2-4M` are quotas for balancing and reporting, not fixed resize
levels.

Use antialiased bicubic, Lanczos, and downscale-only area resampling according to resize direction;
do not reduce every path to bilinear interpolation. Neural super-resolution is unsuitable for
grounding augmentation because it can redraw supervised text, icons, lines, and boundaries.

## Rebuild Derived Data Cleanly

Derived image directories can contain stale files from earlier runs. Do not assume files on disk
are referenced. When rebuilding derived datasets, either clean the derived output directory first
or write to a fresh directory, then verify every JSONL image reference exists and every generated
image is referenced.

## Qwen 0..999 Coordinates Need One Codec

Model-facing geometry coordinates use the project-standard Qwen-style integer `0..999` space.
Do not hand-roll conversions in task scripts, eval parsers, or visualization code.

Use the shared `shaft.codec.coordinates` helpers. Encoding must use nearest-integer rounding,
not `int()` truncation. Decoding must use the same `0..999` scale. Mixing
`pixel / (size - 1) * 999` during SFT generation with `bin / 1000 * size` during eval creates a
small but systematic right/bottom shrink, which shows up as predicted boxes being shifted left/up
or too tight.

## Derived Data Is Not A Metadata Backup

Do not copy raw `extra`, `subattr`, importer fields, or audit details into structured/SFT rows
just because the information might be useful later. Raw data is the source of truth for rich
metadata. Derived data should be a small rebuildable training artifact: image reference, minimal
source id, and model-facing target fields.

For weak-label datasets, this is still true even if there is no human raw truth yet. Keep
weak-label audit information such as `evidence`, `confidence`, `abstain_reason`, source model,
and source job id out of `target_text`; store only minimal traceability in `extra`.

## Shape Attributes Follow The Editable Outer Container

For shape subattribute prelabeling, classify the editable outer container or base shape, not the
most salient semantic content inside the crop.

Example: a gear icon inside a visible square/rectangular tile should be `shape_type=rectangle`,
not `shape_type=other`. A number inside a circular badge should be `shape_type=oval`. Only use
`other` when there is no visible geometric container or the instance is genuinely a complex
freeform symbol/decoration.

The rectangular pixel boundary of an icon or image crop is not an editable rectangle container.
When the crop is the icon/image asset itself and no independent outer geometric container is
visible, the shape reconstruction fallback is `shape_type=other`. If a distinct tile, badge, or
panel encloses that asset, classify the enclosing editable shape instead.

## Reconstruction Negatives Must Match The Invocation Contract

Do not add raw `icon` or raw `image` crops to `shape_reconstruction` merely to teach
`shape_type=other`. In the maintained pipeline, upstream detection chooses the task label first;
`shape_reconstruction` is invoked on a crop already classified as `shape`. Cross-label visual
negatives change the task into joint rejection/classification and encourage the shortcut
"complex, low-resolution, or asset-like content => other".

Within `shape_reconstruction`, reserve `other` for source instances whose label is genuinely
`shape` but whose editable outer geometry cannot be represented by the current DSL. A visible
rectangle/oval/callout container remains that shape even when it contains text, icons, or images.
If robustness to upstream detection mistakes is needed, evaluate it as a separate routed-system
experiment; do not silently change the reconstruction target distribution.

The v5.0-re `balanced_v2` snapshot contains 10,000 `icon_as_other` and 10,000
`image_as_other` rows. The 2026-07-12 checkpoint-8000 audit found this design counterproductive and
it must not be repeated in the next shape reconstruction rebuild.

## Shape Fill Means Independent Fill, Not Pixel Color

For shape subattribute prelabeling, `fill` describes whether the editable shape has its own fill
layer. If the shape interior is transparent or visually the same as the immediate background or
parent container, use `fill.type=none` even when the pixels inside the bbox are white, gray, or
lightly tinted.

Example: a white rectangle on a white page, or a same-gray label area on a gray parent panel,
should not be marked as `solid #FFFFFF` merely because the crop pixels are white. Treat it as
transparent/no fill unless there is an independent visible fill different from the background.

## Weak Labels Are Training Hints, Not Evaluation Truth

Do not silently turn model-generated weak labels into validation or benchmark data. They can be
useful for beta training, but formal eval should remain on human-maintained validation/benchmark
sources unless the user explicitly requests a weak-label eval experiment.

## Small Datasets Can Cap Interleave Sampling

With `mix_strategy=interleave_under`, the smallest dataset relative to its configured weight can
limit the whole mixed epoch. Do not judge a new task's effect from YAML weights alone; compute the
actual mixed quotas from current row counts before deciding whether the ratio is reasonable.

## Preview Is Inspection, Not Data

Do not regenerate previews as a side effect of raw cleaning or derived rebuilds. Generate previews
only when requested or when needed for a small suspicious subset. Prefer drawing boxes directly
on original images; avoid zoom panels unless they answer a specific inspection question.

## Keep Routine State in README, Not Long Reports

For raw directories, a short README is the maintenance surface. Long JSON reports are only for
explicit audit needs or machine handoff. Do not create a second, stale source of truth.

## Use Conservative Parallelism

Data jobs can be parallel, but too many workers can crash the machine or thrash I/O. Use 8 workers
by default for large preview/cleanup tasks unless the user explicitly chooses a higher number.
