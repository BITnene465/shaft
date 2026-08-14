# Derived Datasets

Derived `selection`, task-local media, `structured`, and `sft` artifacts must be rebuildable from
raw/`gt_standard`, split manifests, prompt pools, and maintained scripts. Do not patch JSONL to
hide source or builder errors.

## Formal v5.7 Bundle

The catalog `configs/data/banana_v5_7.yaml` contains exactly five train-only sources:

| dataset | rows | source truth |
| --- | ---: | --- |
| `grounding_layout` | 58,440 | compact human `data/raw` |
| `shape_context_reconstruction` | 300,000 | v9 `gt_standard` |
| `line_context_reconstruction` | 300,000 | v9 `gt_standard` |
| `line_context_points` | 137,218 | 122,218 real paths + 15,000 v9 multi-branch |
| `image_context_reconstruction` | 21,184 | reviewed real image types |

All validation JSONL files are intentionally empty; catalog `use_for_eval=false` and training
`eval.enabled=false`. `grounding_layout_sync`, background, shape weak attributes, region tasks,
and old arrow/point tasks are not part of the published v5.7 mix.

## Rebuild Order

1. Audit v9 and create identity-only selections:

   ```bash
   uv run python scripts/tasks/prepare_gt_standard_v5_7.py --workers 8
   ```

2. Rebuild real grounding, then convert structured rows to SFT:

   ```bash
   uv run python scripts/tasks/build_grounding_structured.py \
     --raw-root data/raw --output-root data \
     --train-split data/raw/splits/grounding_layout.train.txt \
     --val-split data/raw/splits/grounding_layout.val.txt \
     --task grounding_layout --workers 8 --clean

   uv run python scripts/tasks/build_sft_from_structured.py \
     --data-root data --task grounding_layout --workers 8 --clean
   ```

3. Rebuild synthetic shape/line and reviewed real image context tasks with
   `build_context_reconstruction_sft.py`.
4. Select all usable real line paths with `prepare_real_line_context_points.py`, then build the
   line-points task with the explicit real and v9 multi-branch selections.

The exact commands, prompt mappings, and current audit baseline live in `docs/data_v5_7.md`.

## Grounding Contract

- Builder accepts compact `size + layout[].type` or a documented normalized `instances[].label`
  source at the derived boundary; never rewrite compact raw merely for compatibility.
- Consume only four-class bbox targets and preserve one native row for every labeled source.
- Materialize every row image under the task directory. Apply train augmentation only according
  to `layout_multiscale_v1`; validation/test stays native clean.
- Bbox uses image-boundary coordinates. Clip in `[0,width] x [0,height]`, then use
  `shaft.codec.coordinates` for Qwen `0..999` conversion.
- Grounding target order is row-major across labels, never grouped by label.

## Context Reconstruction Contract

- V9 `gt_standard` remains the only synthetic attribute/geometry truth. Selection snapshots store
  identity, not copied target fields.
- Every shape/line crop uses `synthetic_realism_v1`: deterministic 1–3 size-preserving operations;
  an extremely small target gets one mild operation. Store the plan in
  `extra.pixel_augmentation`; never repeat it online.
- Proposal bbox and all target geometry share the contextual crop coordinate frame.
- Real image crops and real line-point crops remain clean.
- Line-points target contains only `is_single + points`; keep every usable real path and cap the
  supplement at 15,000 synthetic multi-branch rows. Do not add synthetic single paths.
- Keep aspect ratio below the safe cap (`60`) by expanding context instead of deleting valid thin
  lines. Never allow the Qwen processor hard limit (`>=200`) to fail later in a worker.

## Publication and Validation

- Preflight every input and prompt before deleting old derived output.
- Publish via same-filesystem staging or an explicit builder `--clean` workflow; remove stale
  unreferenced media only inside the exact derived target.
- Require unique sample ids/media paths and one media file per structured/SFT row.
- Require structured/SFT one-to-one alignment and exact target recomputation.
- Verify source ids, split exclusions, task schema, prompt version, catalog names/weights,
  train-only/eval flags, media snapshot id, and training config prerequisites.
- Keep a short task README/build summary with source versions, command/seed, counts, selection
  policy, augmentation distribution, exclusions, and invariants.
