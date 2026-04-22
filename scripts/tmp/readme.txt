uv run python scripts/tmp/eval_arrow_subattr.py \
  --config configs/train/train_sft_4b_arrow.yaml \
  --checkpoint outputs/qwen3vl-sft/4b/arrow-full-mix/best \
  --batch-size 8


uv run python scripts/tmp/eval_grounding.py \
  --config configs/train/train_sft_4b_arrow.yaml \
  --checkpoint outputs/qwen3vl-sft/4b/arrow-full-mix/best \
  --batch-size 8


uv run python scripts/tmp/eval_layout.py \
  --config configs/train/train_sft_4b_grounding.yaml \
  --checkpoint outputs/qwen3vl-sft/4b/grounding-full-mix/best \
  --batch-size 8


uv run python scripts/tmp/eval_grounding_suite.py \
  --config configs/train/train_sft_4b_grounding.yaml \
  --checkpoint outputs/qwen3vl-sft/4b/grounding-full-mix/best \
  --batch-size 8
