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

## JPEG Is Not a Default Grounding Augmentation

Do not add JPEG compression just because it is a common vision augmentation. For these diagram
grounding tasks, localization errors are more likely to come from scale, dense regions, crop
coverage, partial-object handling, and negative sampling than from JPEG artifacts. Prefer
task-shaped views over image-degradation variants.

## Zoom Out Is Different From Pixel Budget

Processor pixel budget controls the final visual token budget, but it does not replace geometry
augmentation. If a task needs more scale robustness, prefer shrink-and-pad context views over
quality degradation. Density and sliding-window crops mostly create zoom-in views; context
padding jitter adds controlled zoom-out views by shrinking a positive view on a same-size canvas
and transforming `bbox` / `linestrip` coordinates exactly.

## Rebuild Derived Data Cleanly

Derived image directories can contain stale files from earlier runs. Do not assume files on disk
are referenced. When rebuilding derived datasets, either clean the derived output directory first
or write to a fresh directory, then verify every JSONL image reference exists and every generated
image is referenced.

## Derived Data Is Not A Metadata Backup

Do not copy raw `extra`, `subattr`, importer fields, or audit details into structured/SFT rows
just because the information might be useful later. Raw data is the source of truth for rich
metadata. Derived data should be a small rebuildable training artifact: image reference, minimal
source id, and model-facing target fields.

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
