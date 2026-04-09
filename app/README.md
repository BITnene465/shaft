# App Usage

## One-Stage Demo

```bash
python app/demo.py \
  --config configs/infer/infer_stage1_grounding.yaml \
  --dense-model models/Qwen3-VL-4B-Instruct \
  --lora-adapter outputs/qwen3vl-ft/4b/your-run/checkpoints/best
```

## Two-Stage Demo

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-dense-model models/Qwen3-VL-4B-Instruct \
  --stage1-lora-adapter outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best \
  --stage2-dense-model models/Qwen3-VL-4B-Instruct
```

不传 Stage2 adapter 时，Stage2 会直接使用 dense model 推理：

```bash
python app/demo_two_stage.py \
  --config configs/infer/infer_two_stage.yaml \
  --stage1-dense-model models/Qwen3-VL-4B-Instruct \
  --stage1-lora-adapter outputs/qwen3vl-s1-lora/4b/your-stage1-run/checkpoints/best
```

页面会固定显示三张图：

- 输入图
- Stage1 可视化
- Stage2 / 最终可视化

当前 `demo_two_stage` 同时支持：

- Stage1 grounding + Stage2 dense fallback
- 完整 two-stage 推理可视化
