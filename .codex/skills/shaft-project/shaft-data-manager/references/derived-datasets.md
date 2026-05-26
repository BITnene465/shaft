# Derived Datasets

Derived datasets include `grounding`, `point_arrow`, `structured`, and `sft` artifacts. They
should be rebuildable from raw data plus config.

The maintained model-task taxonomy is:

- `grounding`: full-image detection over model-facing labels `arrow`, `icon`, `image`, and
  `shape`. This task may consume bbox-only arrow annotations.
- `point_arrow`: arrow crop/point prediction. This task must consume only arrow instances with a
  valid raw `linestrip`; currently that means the `part1` subset.

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
  `grounding_shape`, `grounding_icon_image`, and `grounding_shape_arrow`. Generate them
  independently from the same grounding split source using each subtask's coverage and label
  filters.
- Grounding structured rows should reference task-local images, not raw-data image paths. Clean
  full-image rows copy/render the raw image into `images/<split>/`; train additionally keeps one
  full-image blur row per covered source, with JPEG blur plus resize blur counts matching the
  clean full-image count.
- Rebuild grounding structured data with `scripts/tasks/build_grounding_structured.py`. This
  script writes `data/<grounding_task>/structured/{train,val}.jsonl`, task-local images under
  `data/<grounding_task>/images/{train,val}`, a per-task README, and removes unreferenced
  generated images after hard-negative sampling.
- If SFT conversion is requested, preserve the same split and source ids from structured data.
- Do not duplicate raw `extra` / `subattr` into structured or SFT rows. Raw data is the metadata
  truth; derived rows should carry only the model-facing target plus minimal traceability fields
  such as source id / source image when needed.
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
