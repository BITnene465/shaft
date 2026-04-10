# App 使用说明

## 1. 单阶段 Demo

```bash
python app/demo.py \
  --config configs/infer/infer_stage1_grounding.yaml \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/qwen3vl-ft/4b/your-run/checkpoints/best
```

## 2. 两阶段 Demo

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-dense-model models/Qwen3-VL-4B-Instruct \
  --stage1-lora-adapter outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best \
  --stage2-dense-model models/Qwen3-VL-4B-Instruct \
  --stage2-lora-adapter outputs/qwen3vl-s2-lora/4b/your-stage2-run/checkpoints/best
```

如果不传 Stage2 adapter，Stage2 将直接使用 dense model：

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-dense-model models/Qwen3-VL-4B-Instruct \
  --stage1-lora-adapter outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best
```

页面固定展示三张图：

- 输入图
- Stage1 可视化
- Stage2（最终）可视化
