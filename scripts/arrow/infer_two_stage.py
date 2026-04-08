#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from vlm_structgen.domains.arrow import draw_prediction, format_prediction_summary

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage arrow inference on one image or a directory.")
    parser.add_argument("--config", default="configs/infer/infer_two_stage.yaml", help="Two-stage inference config path.")
    parser.add_argument("--stage1-dense-model", default=None, help="Optional Stage1 dense model path/name override.")
    parser.add_argument(
        "--stage1-lora-adapter",
        default=None,
        help="Optional Stage1 LoRA adapter directory. Omit to load the dense model only.",
    )
    parser.add_argument("--stage2-dense-model", default=None, help="Optional Stage2 dense model path/name override.")
    parser.add_argument(
        "--stage2-lora-adapter",
        default=None,
        help="Optional Stage2 LoRA adapter directory. Omit to load the dense model only.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cuda:0 or cpu.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", default=None)
    input_group.add_argument("--image-dir", default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --image-dir for images.")
    parser.add_argument("--stage1-max-new-tokens", type=int, default=None)
    parser.add_argument("--stage1-batch-size", type=int, default=None)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=None)
    parser.add_argument("--stage2-batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _iter_image_paths(image_dir: Path, *, recursive: bool) -> list[Path]:
    globber = image_dir.rglob if recursive else image_dir.glob
    return sorted(
        path
        for path in globber("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _relative_key(image_path: Path, image_dir: Path | None) -> Path:
    if image_dir is None:
        return Path(image_path.name)
    return image_path.relative_to(image_dir)


def _save_outputs(
    output_dir: Path,
    *,
    image: Image.Image,
    image_path: Path,
    relative_key: Path,
    report: dict[str, object],
) -> tuple[Path, Path, Path]:
    report_dir = output_dir / "reports"
    stage1_dir = output_dir / "stage1_overlay"
    final_dir = output_dir / "final_overlay"
    report_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / relative_key.with_suffix(".two_stage.json")
    stage1_overlay_path = stage1_dir / relative_key.with_suffix(".png")
    final_overlay_path = final_dir / relative_key.with_suffix(".png")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    final_overlay_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_prediction = report["stage1_report"]["strict"]["prediction"] or report["stage1_report"]["lenient"]["prediction"]
    stage1_overlay = draw_prediction(image, stage1_prediction) if stage1_prediction is not None else image.convert("RGB")
    final_overlay = draw_prediction(image, report["final_prediction"])

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage1_overlay.save(stage1_overlay_path)
    final_overlay.save(final_overlay_path)
    return report_path, stage1_overlay_path, final_overlay_path


def _run_one(
    *,
    runner,
    image_path: Path,
    stage1_max_new_tokens: int | None,
    stage1_batch_size: int | None,
    stage2_max_new_tokens: int | None,
    stage2_batch_size: int | None,
) -> tuple[Image.Image, dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    try:
        report = runner.predict(
            image,
            stage1_max_new_tokens=stage1_max_new_tokens,
            stage1_batch_size=stage1_batch_size,
            stage2_max_new_tokens=stage2_max_new_tokens,
            stage2_batch_size=stage2_batch_size,
        )
        return image, report
    except Exception:
        image.close()
        raise


def _resolve_stage1_batch_size(args: argparse.Namespace, runner) -> int:
    if args.stage1_batch_size is not None:
        return max(int(args.stage1_batch_size), 1)
    return max(int(getattr(getattr(runner.stage1_runner, "settings", None), "batch_size", 1)), 1)


def main() -> None:
    args = parse_args()
    from vlm_structgen.core.infer.config import load_two_stage_inference_config
    from vlm_structgen.domains.arrow import load_two_stage_inference_runner

    infer_config = load_two_stage_inference_config(args.config)
    runner = load_two_stage_inference_runner(
        config_path=args.config,
        stage1_dense_model_name_or_path=args.stage1_dense_model,
        stage1_lora_adapter_path=args.stage1_lora_adapter,
        stage2_dense_model_name_or_path=args.stage2_dense_model,
        stage2_lora_adapter_path=args.stage2_lora_adapter,
        device_name=args.device,
    )
    output_dir = args.output_dir or infer_config.output_dir
    if args.image is not None:
        image_path = Path(args.image)
        image, report = _run_one(
            runner=runner,
            image_path=image_path,
            stage1_max_new_tokens=args.stage1_max_new_tokens,
            stage1_batch_size=args.stage1_batch_size,
            stage2_max_new_tokens=args.stage2_max_new_tokens,
            stage2_batch_size=args.stage2_batch_size,
        )
        try:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            final_prediction = report["final_prediction"]
            print("\n[summary]")
            print(format_prediction_summary(final_prediction))
            if output_dir is not None:
                report_path, stage1_overlay_path, final_overlay_path = _save_outputs(
                    Path(output_dir),
                    image=image,
                    image_path=image_path,
                    relative_key=Path(image_path.name),
                    report=report,
                )
                print(f"Saved report to: {report_path}")
                print(f"Saved stage1 overlay to: {stage1_overlay_path}")
                print(f"Saved final overlay to: {final_overlay_path}")
        finally:
            image.close()
        return

    image_dir = Path(args.image_dir)
    image_paths = _iter_image_paths(image_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")
    if output_dir is None:
        raise ValueError("Batch directory inference requires --output-dir or infer config output_dir.")

    resolved_output_dir = Path(output_dir)
    stage1_batch_size = _resolve_stage1_batch_size(args, runner)
    manifest: list[dict[str, object]] = []
    print(f"Found {len(image_paths)} images under {image_dir} | stage1_batch_size={stage1_batch_size}")
    for batch_start in range(0, len(image_paths), stage1_batch_size):
        batch_image_paths = image_paths[batch_start : batch_start + stage1_batch_size]
        batch_images = [Image.open(image_path).convert("RGB") for image_path in batch_image_paths]
        try:
            batch_reports = runner.predict_batch(
                batch_images,
                stage1_max_new_tokens=args.stage1_max_new_tokens,
                stage1_batch_size=stage1_batch_size,
                stage2_max_new_tokens=args.stage2_max_new_tokens,
                stage2_batch_size=args.stage2_batch_size,
            )
            for batch_index, (image_path, image, report) in enumerate(
                zip(batch_image_paths, batch_images, batch_reports, strict=False),
                start=1,
            ):
                index = batch_start + batch_index
                relative_key = _relative_key(image_path, image_dir)
                report_path, stage1_overlay_path, final_overlay_path = _save_outputs(
                    resolved_output_dir,
                    image=image,
                    image_path=image_path,
                    relative_key=relative_key,
                    report=report,
                )
                final_prediction = report["final_prediction"]
                print(
                    f"[{index}/{len(image_paths)}] {relative_key} | "
                    f"{format_prediction_summary(final_prediction).replace(chr(10), ' | ')}"
                )
                manifest.append(
                    {
                        "image_path": str(image_path),
                        "relative_key": str(relative_key),
                        "report_path": str(report_path),
                        "stage1_overlay_path": str(stage1_overlay_path),
                        "final_overlay_path": str(final_overlay_path),
                        "num_instances": len(final_prediction.get("instances", [])),
                        "stage2_refined": len(report.get("stage2_results", [])),
                    }
                )
        finally:
            for image in batch_images:
                image.close()
    manifest_path = resolved_output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved batch manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
