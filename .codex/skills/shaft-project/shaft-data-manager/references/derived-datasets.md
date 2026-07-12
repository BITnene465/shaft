# Derived Datasets

Derived datasets include `grounding`, `point_arrow`, `structured`, and `sft` artifacts. They
should be rebuildable from raw data plus config.

The maintained v5.0 model-task taxonomy is:

- `grounding_layout`: full-image detection over model-facing labels `shape`, `icon`, `image`,
  and `line`. Raw `arrow` and raw `line` instances are both normalized to `label: "line"` in this
  unified detection target.
- `point_line`: simplified line reconstruction over line crops. This task must consume only
  line/arrow instances with a valid raw `linestrip`, including multi-segment `linestrip` values.

Raw annotation layers such as `layout` and `arrow` are schema concepts, not separate train/eval
task families.

## Rules

- Do not patch derived JSONL as a substitute for fixing raw annotations.
- If raw data changes, plan whether corresponding structured/SFT artifacts need regeneration.
- Rebuild into a clean output directory or explicitly remove stale derived images before writing.
- Keep train and eval outputs separate.
- Keep validation augmentation-free unless requested.
- Use explicit split manifests as the source of truth. For the current `data/raw` layout,
  `data/raw/splits/vlm.test.json` is the canonical VLM test/hand-off manifest. It is image-level,
  may include items without GT JSON, and must not be used as a train source. Training derived data
  should come from GT JSON not present in this test manifest unless the user explicitly defines a
  different split.
- Current v5.0 grounding structured data is the unified `grounding_layout` task. Historical
  `grounding_arrow`, standalone `grounding_line`, `grounding_shape`, and `grounding_icon_image`
  outputs should not be used for new v5.0 training.
- Current v5.0 SFT conversion emits `grounding_layout` targets as Qwen-style object arrays with
  `bbox_2d` and `label`, not grouped label arrays. `point_line` targets use the simplified
  line reconstruction object:
  `{"type":"line","parameters":{"is_single":true|false,"points":[[[x1,y1],[x2,y2],...]]}}`.
- Current v5.0 crop reconstruction targets such as `shape_reconstruction` and
  `line_reconstruction` must also use Qwen-style integer `0..999` coordinates inside the crop for
  all model-facing geometry fields, including `corners`, `body_corners`, `body_bbox`,
  `tail.points`, and line `points`. Do not leave target geometry as crop-local pixels; variable
  crop sizes make that harder for the model to learn and inconsistent with Qwen grounding
  pretraining.
- Rebuild synthetic shape/line reconstruction data with
  `scripts/tasks/build_reconstruction_from_gt_standard.py`. Sampling is deterministic. The current
  on-disk v5.0-re `balanced_v2` snapshot keeps all ten non-head shape types, stratifies the
  remaining head budget, and also contains 10,000 icon plus 10,000 image crops labeled as
  `shape_type=other`. Post-training review showed that these cross-label negatives violate the
  routed task contract and encourage excessive `other`; treat that profile as historical and do
  not repeat its visual-object negatives. The next shape rebuild must sample only source
  `label=shape`, with `other` reserved for genuine shape instances outside the current DSL, and
  must record the resulting shape-only type distribution. `balanced_v1` is the earlier profile
  without visual-object negatives, but its head quotas still require explicit review before reuse.
  Line sampling keeps
  every curved shape-style instance and stratifies the remaining budget across curved path,
  straight shape-style, multi-segment path, and common straight path rows. Optional multi-scale
  generation uses 70% tight, 25% medium, and 5% context padding, with bounded low-resolution
  downsampling recorded in `extra.structured_extra.augmentation`. This changes training views,
  not the compact target DSL.
- Build the reviewed real-image `background` task with
  `scripts/tasks/build_background_sft.py`. It uses one clean full-image row per reviewed annotation,
  excludes existing `*.test.json` manifest IDs from train, and materializes task-local images with
  hardlinks when possible. The derived target is only `{"background":true|false}`; the reviewed
  source JSONL remains the truth for levels, reasons, and audit provenance.
- Build real `image_reconstruction` crops with
  `scripts/tasks/build_image_reconstruction_sft.py`. The maintained profile keeps only
  `parameters.image_type`, excludes existing test-manifest images, and constrains each of the 13
  classes to a configurable count band. Head classes are deterministically capped; classes below
  the floor receive deterministic additional padding views of the same reviewed instance. The
  default view distribution is 70% tight, 25% medium, and 5% context.
- Use `shaft.codec.coordinates` for every pixel <-> Qwen `0..999` conversion in derived data,
  eval parsing, and visualization. Do not reintroduce local `/1000 * width` or `int()` truncation
  conversions.
- Current `grounding_layout` SFT canonical order is visual row-major across all labels:
  `row_bucket(y1, 20) -> x1 -> y1 -> -area -> x2 -> y2 -> label`. The `-area` tie-break is weak:
  it is applied only after row, left edge, and top edge so likely container boxes precede smaller
  inner boxes without letting large areas dominate global reading order. Do not group by label and
  do not force `line` to the end; doing so makes truncation label-biased and reintroduces
  grouped-output behavior. The 2026-07-09 GT analysis that motivated this order is tracked under
  `notes/canonical_order/`.
- `point_line` SFT canonical order preserves the source `linestrip` segment order and each
  segment's point order. Directionless line metric policy can be revisited later, but the
  training target should not reorder points unless raw/source semantics are changed together.
- Grounding structured rows should reference task-local images, not raw-data image paths. The
  maintained default `grounding_layout` train augmentation policy is:
  `clean full_image = 1.0x`, `density_crop ~= 0.3x` with a small minority of negative samples,
  `blur_full + blur_crop = 1.0x` using light-to-moderate Gaussian blur / resize blur /
  JPEG compression, and `random_padded_full = 0.2x` applied only to clean full-image rows.
  Validation and VLM test rows remain clean full-image only.
- Rebuild grounding structured data with `scripts/tasks/build_grounding_structured.py`. This
  script writes `data/<grounding_task>/structured/{train,val}.jsonl`, task-local images under
  `data/<grounding_task>/images/{train,val}`, a per-task README, and removes unreferenced
  generated images after hard-negative sampling. Pass `--train-split` and `--val-split`
  explicitly; the script intentionally has no stale default split files.
- If SFT conversion is requested, preserve the same split and source ids from structured data.
- Do not duplicate raw `extra` / `subattr` into structured or SFT rows. Raw data is the metadata
  truth; derived rows should carry only the model-facing target plus minimal traceability fields
  such as source id / source image when needed.
- For weak-supervised business tasks such as `drawio_shape`, keep model-facing training targets
  limited to business fields. Process fields such as `evidence`, `confidence`, and
  `abstain_reason` may be kept only in `extra` for audit/filtering and must not appear in
  `target_text`.
- Weak-label derived datasets should default to train-only. Do not invent a validation split from
  weak labels unless the user explicitly wants weak validation; prefer keeping formal eval on
  human-maintained benchmark/validation data.
- When a weak-label source is heavily imbalanced, filter low-quality rows first, then cap dominant
  classes and keep rare classes. Record source counts, clean counts, caps, selected counts, and
  final label distribution in the dataset README / summary.
- Before long multimodal SFT runs, profile train-only sequence tails with the training tokenizer.
  Filter only pathological outliers that harm DDP balance; keep validation unchanged unless
  explicitly requested. For Banana v2.1, use the looser grounding guard
  `target_tokens > 4000` or `instances > 160`; tighter bounds are only temporary debugging
  tools and should be reverted after diagnosis.
- For point/crop/reconstruction tasks, filter degenerate crops such as
  `extra.image_width < 4` or `extra.image_height < 4`.
- For Qwen VL training, do not emit extreme-aspect crop images. The HF Qwen image processor
  rejects crops whose absolute aspect ratio is `>= 200`, and one crashing DataLoader worker can
  surface as a later NCCL timeout on the other ranks. Prefer expanding the shorter crop side to
  add context and keep real thin line samples, using a conservative cap such as `60`, then verify
  max aspect ratio before launching a long SFT run.
- For `point_line`, use `scripts/tasks/build_point_line_structured.py` to build
  `data/point_line/structured/{train,val}.jsonl` and crop images under
  `data/point_line/images/{train,val}`. The maintained method is one padded crop per valid raw
  `arrow` or raw `line` instance with a `linestrip`; output label is always model-facing `line`.
  Train padding is deterministic-random within the configured range, while validation padding is
  fixed. Do not apply pixel augmentation or jitter row doubling unless explicitly requested.
- When filtering SFT rows, create a timestamped `.bak_*`, report counts, and verify no matching
  rows remain.

## Validation

- JSONL rows and referenced image files align.
- Every generated image is referenced by a JSONL row; there are no stale unreferenced derived
  images.
- Source ids remain traceable back to raw files.
- `instances` / target fields only contain model-facing fields.
- SFT train rows pass task-specific safety bounds, including long sequence guardrails and
  minimum crop dimensions for point/crop tasks.
- Rich details remain in raw `extra` / `subattr`, not duplicated into derived JSONL.
- README or a short summary records row counts, split policy, and generation settings.
- Weak-label datasets explicitly document that they are derived training artifacts, not raw truth.
