# Derived Datasets

Derived datasets include `grounding`, `point_arrow`, `structured`, and `sft` artifacts. They
should be rebuildable from raw data plus config.

The maintained model-task taxonomy is:

- `grounding`: full-image detection over model-facing labels `arrow`, `icon`, `image`, and
  `shape`. This task may consume bbox-only arrow annotations.
- `point_arrow`: arrow crop/point prediction. This task must consume only arrow instances with a
  valid raw `linestrip`; currently that means the `part1` subset.
- `drawio_shape`: crop-level outer-shape classification/style fields for draw.io reconstruction.
  Current `drawio_shape.v4.0` is weak-supervised train-only data generated from VLM labels, not raw
  truth and not an eval source.

Raw annotation layers such as `layout` and `arrow` are schema concepts, not separate train/eval
task families.

## Rules

- Do not patch derived JSONL as a substitute for fixing raw annotations.
- If raw data changes, plan whether corresponding structured/SFT artifacts need regeneration.
- Rebuild into a clean output directory or explicitly remove stale derived images before writing.
- Keep train and eval outputs separate.
- Keep validation augmentation-free unless requested.
- Use `data/raw_data/splits/grounding_train.txt` and `grounding_val.txt` as the image-level
  source of truth for grounding. Derive point-arrow train rows by filtering `grounding_train.txt`
  to `part1` arrow instances with `linestrip`; use `point_arrow_val.txt` for point validation.
- Current grounding structured subtasks are `grounding_arrow`, `grounding_layout`,
  `grounding_shape`, and `grounding_icon_image`. Generate them
  independently from the same grounding split source using each subtask's coverage and label
  filters.
- Grounding structured rows should reference task-local images, not raw-data image paths. Each
  covered train source contributes one clean `full_image` row. JPEG blur plus resize blur are a
  bounded additional `full_image_blur` subset, defaulting to at most half of covered train
  sources. Density crops are also bounded and should not exceed half of covered train sources
  unless explicitly requested.
- Rebuild grounding structured data with `scripts/tasks/build_grounding_structured.py`. This
  script writes `data/<grounding_task>/structured/{train,val}.jsonl`, task-local images under
  `data/<grounding_task>/images/{train,val}`, a per-task README, and removes unreferenced
  generated images after hard-negative sampling.
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
- For point/crop tasks, filter degenerate crops such as `extra.image_width < 4` or
  `extra.image_height < 4`.
- For `point_arrow`, use `scripts/tasks/build_point_arrow_structured.py` to build
  `data/point_arrow/structured/{train,val}.jsonl` and crop images under
  `data/point_arrow/images/{train,val}`. The maintained method is one padded crop per valid arrow
  linestrip, randomized train padding, stable validation padding, and no jitter row doubling
  unless explicitly requested.
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
