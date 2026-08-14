# Model-Assisted Prelabeling

Prelabels are temporary review artifacts, never raw truth or SFT targets by themselves.

## Scope and Handoff

- Read candidate images from the maintained `data/raw/images` inventory and an explicit manifest.
- Exclude current train/test items according to the task; do not infer unlabeled images are
  negatives.
- Write model JSON, raw responses, previews, run metadata, and failures under an explicit
  `temp/<task>/` handoff directory.
- Never write directly into `data/raw/json` or `data/raw/weak_labels`. Promotion requires human
  review, an explicit import decision, schema validation, and a raw backup.

## Model Identity

Resolve the intended checkpoint path and artifact identity. If using an HTTP/vLLM endpoint, query
its identity/models endpoint and verify the served model and root path. Do not silently reuse an
old process on the expected port. Record model path/name, artifact fingerprint when available,
endpoint, prompt pool/version, seed, worker count, and run time in `_run_meta.json` or
`summary.json`.

## Current Task Contracts

For grounding prelabels, use the current `grounding_layout.v5.7` schema and the four model labels
`shape|icon|image|line`. Output Qwen bbox arrays are dequantized to image-boundary pixel
coordinates before handoff. Do not create `arrow`, `grounding_arrow`, or `point_arrow` task
families.

For line geometry, keep predicted ordered points on the same `line` instance and preserve the raw
model response. Failure to predict a path must not delete a valid bbox or fabricate points.

For shape/line/image reconstruction, a model may help propose fields for review, but must not
invent missing geometry and must pass the exact current prompt schema. `shape_context_attributes`
is not part of v5.7 training; do not publish historical API weak labels into the v5.7 catalog.

## Suggested Handoff Shape

```json
{
  "image_path": "data/raw/images/example.png",
  "image_size": [1920, 1080],
  "model_path": "outputs/.../checkpoint",
  "prompt_pool": "grounding_layout.v5.7",
  "instances": [
    {"label": "line", "bbox": [10, 20, 300, 220], "points": [[[20, 30], [280, 200]]]}
  ],
  "raw_output": "...",
  "errors": []
}
```

This is a review interchange shape, not permission to overwrite either maintained raw contract.
An accepted importer must map it explicitly to compact `layout[].type/bbox/parameters` or a
documented normalized `instances` batch.

## Coordinate and Schema Gate

- Use `shaft.codec.coordinates` for pixel/Qwen conversion.
- Clamp to image boundary, sort bbox corners, require positive area, and preserve ordered paths.
- For crop inference, save the exact crop transform and map every point/bbox back to source
  coordinates before review.
- Enforce exact nested keys and enum values. Reject extras/missing values instead of moving,
  defaulting, or synthesizing them.
- Keep evidence, confidence, abstention reason, model provenance, and raw responses as review
  metadata only; never copy them into business SFT targets.

## Review and Validation

- Generate one full-image overlay per item; draw boxes and ordered paths when present.
- Report input, JSON, preview, failure, per-label, and path-present/path-missing counts.
- Keep accepted, rejected, and schema-valid-but-unreviewed rows distinguishable.
- Large runs default to conservative parallelism (8 workers) unless resource behavior is known.
