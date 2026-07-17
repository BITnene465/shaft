# Model-Assisted Prelabeling

Use this reference when a trained checkpoint is used to produce temporary annotations for
human review. Prelabeling is a handoff workflow, not a raw-data update by itself.

## Scope

- Input should normally come from the maintained raw image inventory. Use
  `data/raw_data/part2/images` for unannotated inventory prelabeling, and
  `data/raw_data/part1/images` only when intentionally rechecking already annotated samples.
- Output should go to an explicit temporary task directory such as `temp/task4/`.
- Keep generated JSON under `pre_label/` and inspection images under `previews/`.
- Do not write model outputs directly into `data/raw_data/part*/json` unless the user explicitly
  asks to import them after review.

## Model And Endpoint Checks

- Resolve "new best" by checking the checkpoint path and modification time. Do not assume an
  existing vLLM endpoint is serving the requested checkpoint.
- Before running, query `/v1/models` and verify both the served model name and `root` path match
  the intended checkpoint.
- If an old endpoint is already bound to the usual port, start the new checkpoint on another port
  rather than silently reusing the old service.
- Record `model_path`, `model_name`, endpoint, prompts, and worker count in `_run_meta.json` or
  `summary.json`.

## Required Task Coverage

For diagram prelabeling, run the task heads the user expects. The maintained task families are
unified `grounding` detection and optional `point_arrow`.

Recommended stages:

1. Grounding/layout labels on the full image.
   - Use the prompt in `configs/prompts/pools/grounding_layout.v2.4.yaml`.
   - Keep labels `icon`, `image`, and `shape`.
   - If the model emits `shape_combination`, normalize the label to `icon` and preserve the
     original label in `extra`.
2. Grounding/arrow labels on the full image.
   - Use the prompt in `configs/prompts/pools/grounding_arrow.v2.4.yaml`.
   - Normalize model-facing arrow detections to `label: "arrow"`.
3. Arrow point prediction on padded arrow crops when point annotations are needed.
   - Use the prompt in `configs/prompts/pools/point_arrow.v2.4.yaml`.
   - Map crop-local keypoints back to original image coordinates.
   - Store them on the same arrow instance as `linestrip`; if keypoint prediction fails, keep the
     arrow bbox and record the stage error.

For real-domain reconstruction attributes without trusted geometry, keep prelabeling as an
explicit local/API handoff workflow instead of asking the teacher model to invent control points.
The repository-owned builder consumes only a reviewed, versioned weak-label sidecar:

- `shape_context_attributes` keeps shape type, border, fill, effect, and optional callout body
  type, but omits corners, body geometry, and tail geometry.
- `line_context_attributes` keeps line/style/topology/color fields but omits points. One continuous
  polyline remains `is_single=true`; only forked/multi-branch targets are false. Branched targets
  omit global begin/end markers because no unique endpoint pair exists.
- Both tasks use one clean image-first crop per call plus crop-local `proposal_bbox_2d`. They are
  independent auxiliary tasks and must not be mixed into full reconstruction targets that require
  geometry.
- Current raw bbox annotations are selection truth. New API outputs remain under `temp/` until
  review; only a versioned sidecar explicitly accepted for auxiliary training may be promoted to
  `data/raw/weak_labels/`.

## Output Schema

Prelabel JSON can be richer than derived training rows, but should stay easy to import into raw:

```json
{
  "image_path": "data/raw_data/part2/images/pic_1001.png",
  "image_relative_path": "gemini_0001.png",
  "image_size": [width, height],
  "model_path": "outputs/.../best",
  "model_name": "task4_best",
  "num_bins": 1000,
  "instances": [
    {
      "label": "icon",
      "bbox": [x1, y1, x2, y2],
      "layer": "layout",
      "extra": {"prelabel_model": "task4_best"}
    },
    {
      "label": "arrow",
      "bbox": [x1, y1, x2, y2],
      "layer": "arrow",
      "linestrip": [[x0, y0], [x1, y1]],
      "subattr": {},
      "extra": {"prelabel_model": "task4_best"}
    }
  ],
  "raw_outputs": {
    "layout": "...",
    "arrow": "..."
  },
  "errors": []
}
```

Raw model text should be kept for audit/debugging, but it is not part of the long-term raw schema.

## Coordinate Rules

- Grounding outputs are usually 0-999 binned coordinates; dequantize them to pixel coordinates
  before writing task JSON.
- Clamp boxes and points to the image boundary.
- Sort bbox corners into `[x1, y1, x2, y2]` and enforce positive width/height.
- For keypoint crops, use the exact crop box used for inference to remap points back to global
  coordinates. Do not store crop-local points as raw `linestrip`.

## Preview

- Generate one overlay preview per image in `previews/`.
- Draw layout boxes and arrow boxes on the original image. Draw arrow `linestrip` when available.
- Avoid zoom panels by default; previews are for quick human review of the full image.

## Validation

After the run, report:

- input image count
- generated JSON count
- generated preview count
- failed image count
- total instances
- per-layer and per-label counts
- arrow instances with and without `linestrip`
- stage error count

Keep `summary.json` in the task directory for handoff. For large batches, use conservative
parallelism, normally 8 workers unless the user asks otherwise.

## Context Attribute API Gate

- Resolve available Bedrock inference profiles before starting. Listing a foundation model does
  not prove its on-demand or regional profile is callable.
- The 2026-07-16 maintained pilot uses `au.anthropic.claude-sonnet-5`; this model rejects the old
  `temperature` request field, so the maintained caller omits it.
- Do not move misplaced fields, synthesize missing colors, or fill omitted style fields in the
  parser. Invalid responses stay rejected and may be retried with `--retry-from`.
- Handoff output separates `manifest.jsonl`, `schema_valid.jsonl`, and `rejected.jsonl`, and
  provides a full crop/JSON/color-swatch review page. Schema-valid does not mean human-reviewed.
- Schema-valid must enforce exact nested key sets, not only required values. API extras outside the
  prompt contract remain rejected and must never be copied into SFT `target_text`.
- Large shape-only runs may use a local bulk caller with Opus 4.8. Send independent image blocks
  before one batch prompt; do not downscale them into a sheet. The validated batch size is 8.
  API-label crops use four independently sampled padding ratios with the same v5.3
  tight/medium/large/extreme `20/50/25/5%` policy. The resulting `weak_labels.jsonl` may be promoted
  to `data/raw/weak_labels/` only when the user explicitly accepts API noise for an auxiliary task.
  Local callers under ignored work directories are operational tools, not repository APIs and
  must not be imported by tracked tests.
