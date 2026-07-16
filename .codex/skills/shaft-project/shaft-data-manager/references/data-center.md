# Data Catalog and Mixing

Use this when wiring prepared datasets into Shaft training configs from the data-management side.
For framework implementation changes to `ShaftDataCenter`, registry behavior, or sampler/mixing
code, use the separate `shaft-data-center` development skill.

## Responsibilities

`ShaftDataCenter` is the central path for:

- data source loading
- offline transforms
- sample-level mixing
- dataset-aware online transforms

Pipelines should call the data center rather than branching on data source names.

## Config Guidance

- Register datasets in `configs/data/*.yaml`.
- Use explicit source entries in `data.datasets`.
- Prefer `concat` for full sample coverage unless the user asks for balanced sampling.
- Keep train-only and eval datasets semantically separate.
- For `interleave_under`, calculate effective quotas with current row counts before training.
  Dominant large datasets may be undersampled, while a small train-only task with too much weight
  can cap the epoch and distort the intended mix.
- Do not set `use_for_eval: true` for weak-label-only datasets unless the user explicitly wants a
  weak validation experiment.

## Tests

When changing catalog or mixing behavior, update focused tests such as:

- `tests/test_data_sources.py`
- `tests/test_mixing.py`
- `tests/test_data_center.py`
- config loader tests that consume the new fields
