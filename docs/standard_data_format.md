# Standard Data Format

All training and validation data must conform to the standard JSONL format described below. The dataset never guesses `task_type` or `domain_type` -- every record must explicitly declare them.

## Common Fields

Every JSONL record, regardless of task type, must contain:

| Field | Type | Required | Description |
|---|---|---|---|
| `sample_id` | `str` | Yes | Unique identifier for the sample |
| `image_path` | `str` | Yes | Relative or absolute path to the image file |
| `image_width` | `int` | Yes | Image width in pixels |
| `image_height` | `int` | Yes | Image height in pixels |
| `task_type` | `str` | Yes | One of: `joint_structure`, `grounding`, `keypoint_sequence` |
| `domain_type` | `str` | Yes | Currently: `arrow` |

## Label Constraints

- `label` must be one of:
  - `single_arrow`
  - `double_arrow`
- No other label values are permitted

## Coordinate Constraints

- All quantized coordinates are integers in `[0, 999]`
- `bbox_2d` is `[x1, y1, x2, y2]` where `x1 < x2` and `y1 < y2`
- `keypoints_2d` is a list of `[x, y]` pairs

## Keypoint Order

### single_arrow

Points are ordered from **tail to head**:

```
keypoints_2d = [[tail_x, tail_y], ..., [head_x, head_y]]
```

Minimum 2 points.

### double_arrow

The two head tips are at `keypoints_2d[0]` and `keypoints_2d[-1]`. The **upper-left head comes first** (smaller x, then smaller y as tie-breaker):

```
keypoints_2d = [[head1_x, head1_y], ..., [head2_x, head2_y]]
```

where `(head1_x, head1_y) <= (head2_x, head2_y)` lexicographically.

## Canonical Instance Ordering

Within a single record, instances must be sorted by the canonical sort key:

```
(y1, x1, y2, x2, tail_y, tail_x, head_y, head_x, n_points)
```

This ordering must be **frozen during data preparation**. The dataset reads instances in the order they appear -- it does not re-sort.

For Stage1 grounding, the sort key is:

```
(y1, x1, y2, x2, label)
```

---

## One-Stage: joint_structure

### Path

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

### Record Format

```json
{
  "sample_id": "img_001",
  "image_path": "data/processed/images/train/img_001.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "task_type": "joint_structure",
  "domain_type": "arrow",
  "instances": [
    {
      "label": "single_arrow",
      "bbox_2d": [100, 200, 300, 400],
      "keypoints_2d": [[120, 250], [200, 300], [280, 350]]
    },
    {
      "label": "double_arrow",
      "bbox_2d": [500, 100, 700, 300],
      "keypoints_2d": [[520, 120], [600, 200], [680, 280]]
    }
  ]
}
```

### Output Format (model prediction)

```json
[
  {
    "label": "single_arrow",
    "bbox_2d": [100, 200, 300, 400],
    "keypoints_2d": [[120, 250], [200, 300], [280, 350]]
  }
]
```

### Codec

`ArrowCodec` (`domains/arrow/codecs/structure.py`)

---

## Stage1: grounding

### Path

- `data/two_stage/stage1/train.jsonl`
- `data/two_stage/stage1/val.jsonl`

### Record Format

```json
{
  "sample_id": "img_001_full",
  "image_path": "data/two_stage/stage1/images/train/img_001_full.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "task_type": "grounding",
  "domain_type": "arrow",
  "instances": [
    {
      "label": "single_arrow",
      "bbox_2d": [100, 200, 300, 400]
    },
    {
      "label": "double_arrow",
      "bbox_2d": [500, 100, 700, 300]
    }
  ]
}
```

### Output Format (model prediction)

```json
[
  {
    "label": "single_arrow",
    "bbox_2d": [100, 200, 300, 400]
  }
]
```

### Codec

`GroundingCodec` (`domains/arrow/codecs/grounding.py`)

### Tile Records

Stage1 also contains tile samples (cropped regions from multi-scale sliding windows). Tile records include the same fields, but `image_path` points to the cropped image. Tile deduplication is performed during data preparation using:

- Instance set matching
- Crop IoU threshold: `stage1_dedup_iou_threshold`

Only tiles where **all bounding boxes are fully contained** are kept. Tiles with partial bbox overlap are discarded.

---

## Stage2: keypoint_sequence

### Path

- `data/two_stage/stage2/train.jsonl`
- `data/two_stage/stage2/val.jsonl`

### Record Format

```json
{
  "sample_id": "img_001_inst_0",
  "image_path": "data/two_stage/stage2/images/train/img_001_inst_0.jpg",
  "image_width": 512,
  "image_height": 512,
  "task_type": "keypoint_sequence",
  "domain_type": "arrow",
  "condition": {
    "label": "single_arrow",
    "bbox_2d": [50, 80, 400, 420]
  },
  "gt_struct": {
    "label": "single_arrow",
    "keypoints_2d": [[80, 120], [200, 250], [380, 400]]
  }
}
```

### Output Format (model prediction)

```json
{
  "keypoints_2d": [[80, 120], [200, 250], [380, 400]]
}
```

### Codec

`KeypointSequenceCodec` (`domains/arrow/codecs/keypoint_sequence.py`)

### Key Points

- Each record corresponds to a **single target arrow**
- The image is a **padded crop** around the target instance (default `padding_ratio = 0.3`)
- Coordinates in `condition.bbox_2d` and `gt_struct.keypoints_2d` are **crop-local**, quantized to `[0, 999]`
- The prompt explicitly injects the crop-local `label` and `bbox_2d` as condition
- Even if the crop contains other arrows, the model should only output the target arrow's keypoints
- Stage2 JSONL does **not** treat `target_text` as data truth; training targets are generated from `gt_struct` via `KeypointSequenceCodec` at dataset load time

---

## Data Preparation Commands

### One-Stage

```bash
python scripts/arrow/prepare_data.py \
  --raw-json-dir data/raw/json \
  --image-dir data/raw/figure \
  --output-dir data/processed
```

### Stage1

```bash
python scripts/arrow/prepare_stage1_data.py \
  --input-jsonl data/processed/train.jsonl \
  --image-dir data/processed/images/train \
  --output-dir data/two_stage/stage1
```

### Stage2

```bash
python scripts/arrow/prepare_stage2_data.py \
  --input-jsonl data/processed/train.jsonl \
  --image-dir data/processed/images/train \
  --output-dir data/two_stage/stage2
```

---

## LabelMe Mapping

Raw LabelMe annotations are mapped as follows:

| LabelMe Rectangle Label | Arrow Label |
|---|---|
| `c0`, `c1`, `c2`, `c3` | `single_arrow` |
| `c4`, `c5`, `c6`, `c7` | `double_arrow` |

Shapes are grouped by `group_id`. Rectangles provide `bbox`, point shapes provide `keypoints`.
