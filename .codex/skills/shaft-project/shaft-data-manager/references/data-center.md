# Data Catalog Preparation

Use this only to register already prepared data in YAML. It does not authorize changes to
`ShaftDataCenter`, source registries, samplers, batching, or mixing implementations; those changes
must follow repository architecture, extension, testing, and feature-review rules.

## Catalog Rules

- Register production sources in a tracked `configs/data/*.yaml` catalog.
- Use canonical task names and explicit `source_type`, train/val paths, weight, enabled state,
  `use_for_eval`, help text, and tags.
- Resolve paths relative to the catalog file and load the consuming training YAML through
  `shaft.config.load_config` before publication.
- Keep train-only and eval semantics explicit. Empty validation is valid only when documented and
  paired with `use_for_eval: false` / disabled eval.
- Weighted mixing affects draw probability, not dataset completeness. Review current row counts
  and intended task importance before changing weights.
- Every enabled runtime prompt-sampled dataset must have one exact dataset-name-to-pool mapping.
- Change `media_snapshot_id` when the referenced media/JSONL snapshot changes materially.
- Do not register an optional or unmaterialized dataset merely because a builder exists.

## Banana v5.7

`configs/data/banana_v5_7.yaml` has exactly five active datasets and is consumed by all six v5.7
training YAMLs. Its names, weights, prompt pools, and row-count baseline are documented in
`scripts/tasks/banana_v5_7.md`. `grounding_layout_sync` is not materialized or registered.

## Validation

- Load every consuming config through the strict loader.
- Verify every enabled JSONL and prompt pool exists and every prompt pool compiles.
- Verify catalog names, prompt mappings, weights, train-only/eval flags, and media snapshot are
  identical across the intended config family.
- Separately verify model/checkpoint paths; schema-valid continuation configs can still be
  non-runnable when their initialization checkpoint is absent.
- Update `tests/test_config_loader.py` or a focused catalog test with the published bundle.
