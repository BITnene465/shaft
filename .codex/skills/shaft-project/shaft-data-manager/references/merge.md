# Data Merge and Import

Use this when importing new JSON annotations, compressed archives, unpacked archives, or separate
raw datasets.

## Merge Procedure

1. Inventory incoming files:
   - number of JSON files
   - parse errors
   - label counts
   - normalized image names
   - images with and without target labels
   - archive/source paths when multiple folders contain the same basename
2. Compare against existing raw data by normalized basename and, when needed, image path or hash.
3. Decide whether incoming samples are positive, negative, or another annotation layer.
4. Copy or link images only when the target raw image is missing and there is no conflict.
5. Write raw JSON in the current raw schema.
6. Update the raw README with what was merged and any missing semantics.

## Archives

- Extract every archive the user identifies as part of the source set.
- Preserve enough source path metadata in `extra` to explain where each imported annotation came
  from.
- Do not treat repeated basenames as duplicates until the annotation layer and image identity are
  checked.

## Negative Samples

Negative samples are valid only when the task definition says absence of target instances is
meaningful. Keep them as empty `shapes` and document why they exist.

## Do Not

- Do not silently overwrite existing annotations with a different schema.
- Do not collapse distinct annotation layers just because they share the same image name.
- Do not infer missing semantic attributes unless the user approves the heuristic.
