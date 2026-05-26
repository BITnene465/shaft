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
   - Use the prompt in `configs/prompts/grounding_layout.yaml`.
   - Keep labels `icon`, `image`, and `shape`.
   - If the model emits `shape_combination`, normalize the label to `icon` and preserve the
     original label in `extra`.
2. Grounding/arrow labels on the full image.
   - Use the prompt in `configs/prompts/grounding_arrow.yaml`.
   - Normalize model-facing arrow detections to `label: "arrow"`.
3. Arrow point prediction on padded arrow crops when point annotations are needed.
   - Use the prompt in `configs/prompts/point_arrow.yaml`.
   - Map crop-local keypoints back to original image coordinates.
   - Store them on the same arrow instance as `linestrip`; if keypoint prediction fails, keep the
     arrow bbox and record the stage error.

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
