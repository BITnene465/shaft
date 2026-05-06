# Derived Datasets

Derived datasets include `grounding_*`, `keypoint_*`, `structured`, and `sft` artifacts. They
should be rebuildable from raw data plus config.

## Rules

- Do not patch derived JSONL as a substitute for fixing raw annotations.
- If raw data changes, plan whether corresponding structured/SFT artifacts need regeneration.
- Rebuild into a clean output directory or explicitly remove stale derived images before writing.
- Keep train and eval outputs separate.
- Keep validation augmentation-free unless requested.
- If SFT conversion is requested, preserve the same split and source ids from structured data.
- Do not duplicate raw `extra` / `subattr` into structured or SFT rows. Raw data is the metadata
  truth; derived rows should carry only the model-facing target plus minimal traceability fields
  such as source id / source image when needed.

## Validation

- JSONL rows and referenced image files align.
- Every generated image is referenced by a JSONL row; there are no stale unreferenced derived
  images.
- Source ids remain traceable back to raw files.
- `instances` / target fields only contain model-facing fields.
- Rich details remain in raw `extra` / `subattr`, not duplicated into derived JSONL.
- README or a short summary records row counts, split policy, and generation settings.
