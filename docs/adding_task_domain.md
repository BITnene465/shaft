# Adding a New Task or Domain

## Architecture Recap

```
core/          → Generic framework (does NOT understand task/domain semantics)
  └── registry.py  → get_adapter(task_type, domain_type) routes to task adapter

tasks/         → Task-specific adapters (implements TaskAdapter protocol)
  └── <task>/adapter.py  → build_*_adapter() factory

domains/       → Domain-specific logic (codecs, schema, ordering, data prep)
  └── <domain>/codecs/  → Serialization/deserialization
```

## Adding a New Task Type

### Steps

1. **Register** -- Add to `SUPPORTED_TASK_TYPES` in `core/registry.py`
2. **Create adapter** -- `tasks/<task>/adapter.py` with:
   - A class implementing `TaskAdapter` protocol (inherit `BaseArrowAdapter` for arrow domain)
   - A `build_*_adapter()` factory function
3. **Wire routing** -- Add dispatch branch in `get_adapter()` in `core/registry.py`
4. **Update evaluator** -- Add metric summarization in `core/eval/evaluator.py`
5. **Create config** -- `configs/train/train_<task>_lora.yaml`
6. **Prepare data** -- Script that outputs JSONL with the new `task_type`

### Checklist

- [ ] Add to `SUPPORTED_TASK_TYPES` in `registry.py`
- [ ] Create `tasks/<task>/adapter.py` with `build_*_adapter()` factory
- [ ] Add dispatch branch in `get_adapter()` in `registry.py`
- [ ] Update `summarize()` in `eval/evaluator.py`
- [ ] Create training config in `configs/train/`
- [ ] Create data preparation script
- [ ] Update `docs/standard_data_format.md`
- [ ] Update `docs/architecture.md`

---

## Adding a New Domain Type

### Steps

1. **Register** -- Add to `SUPPORTED_DOMAIN_TYPES` in `core/registry.py`
2. **Create domain directory** -- `domains/<domain>/` with:
   - `schema.py` -- Data model
   - `ordering.py` -- Canonical sort
   - `task_support.py` -- Base adapter, matching logic
   - `codecs/` -- One codec per task type
   - `data/` -- Data preparation
   - `infer/` -- Two-stage inference (if applicable)
3. **Implement codecs** -- Each codec implements `encode`, `encode_with_loss_meta`, `decode`, `decode_with_meta`, `validate_struct`
4. **Update task adapters** -- Add domain branch in each task's `build_*_adapter()` factory
5. **Update evaluator** (if metrics differ)
6. **Prepare data** -- Script that outputs JSONL with the new `domain_type`

### Checklist

- [ ] Add to `SUPPORTED_DOMAIN_TYPES` in `registry.py`
- [ ] Create `domains/<domain>/` with schema, ordering, codecs, data prep
- [ ] Update each task adapter's `build_*_adapter()` to handle new domain
- [ ] Update evaluator if metrics differ
- [ ] Create data preparation script
- [ ] Update `docs/standard_data_format.md`
- [ ] Update `docs/architecture.md`

---

## TaskAdapter Protocol Reference

| Method | Returns | Description |
|---|---|---|
| `task_type` | `str` | Task identifier |
| `domain_type` | `str` | Domain identifier |
| `num_bins` | `int` | Quantization bins (delegates to codec) |
| `task_bucket_key` | `str` | Bucket key for batch sorting |
| `build_gt_struct_from_record(record)` | `dict` | Extract GT from JSONL record |
| `encode_target_text(gt_struct, w, h)` | `str` | Serialize GT to target text |
| `build_training_target(gt_struct, w, h)` | `dict` | Returns `{target_text, loss_meta}` |
| `decode(text, w, h, strict)` | `dict` | Parse model output |
| `decode_with_meta(text, w, h, strict)` | `(dict, dict)` | Parse + metadata |
| `empty_prediction()` | `dict` | Default empty output |
| `score_prediction(gt, pred, ...)` | `dict[str, float]` | Compute metrics |
| `compute_loss(outputs, batch, tokenizer)` | `object` | Task-specific loss |

---

## Common Pitfalls

1. **Forgetting to update `SUPPORTED_TASK_TYPES` or `SUPPORTED_DOMAIN_TYPES`** -- Registry rejects the new type
2. **Not adding the dispatch branch in `get_adapter()`** -- Routing fails
3. **Missing `task_bucket_key`** -- Batch sorting fails
4. **Not updating evaluator `summarize()`** -- Metrics not logged
5. **Inconsistent JSON field names** -- Codec output must match evaluator expectations
6. **Forgetting to canonicalize ordering** -- Instance order must be deterministic
