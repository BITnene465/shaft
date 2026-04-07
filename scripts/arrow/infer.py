from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_structgen.domains.arrow import draw_prediction, format_prediction_summary

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-stage inference on one image or a directory.")
    parser.add_argument("--config", default="configs/infer/infer_one_stage.yaml", help="Inference config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Falls back to CHECKPOINT_PATH in .env.")
    parser.add_argument("--env-file", default=None, help="Optional path to a .env file when checkpoint falls back to CHECKPOINT_PATH.")
    parser.add_argument("--model", default=None, help="Optional model path/name override.")
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cuda:0 or cpu.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", default=None)
    input_group.add_argument("--image-dir", default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --image-dir for images.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override inference max_new_tokens for this run.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for directory inference.")
    parser.add_argument("--output-dir", default=None, help="Optional directory to save parsed prediction files.")
    parser.add_argument(
        "--save-preview",
        dest="save_preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save per-image visualization previews when output_dir is provided.",
    )
    return parser.parse_args()


def _save_outputs(
    output_dir: Path,
    image_path: Path,
    image: Image.Image,
    raw_text: str,
    parse_report: dict[str, object],
    *,
    save_preview: bool,
) -> tuple[Path, Path, Path | None]:
    report_dir = output_dir / "reports"
    raw_dir = output_dir / "raw"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    if save_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    prediction_path = report_dir / f"{stem}.one_stage.json"
    raw_text_path = raw_dir / f"{stem}.raw.txt"
    preview_path = preview_dir / f"{stem}.png"
    prediction_path.write_text(json.dumps(parse_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_text_path.write_text(raw_text + "\n", encoding="utf-8")

    strict_prediction = parse_report.get("strict", {}).get("prediction")
    lenient_prediction = parse_report.get("lenient", {}).get("prediction")
    prediction_for_draw = strict_prediction or lenient_prediction
    if save_preview and prediction_for_draw is not None:
        draw_prediction(image, prediction_for_draw).save(preview_path)
        return prediction_path, raw_text_path, preview_path
    return prediction_path, raw_text_path, None


def _iter_image_paths(image_dir: Path, *, recursive: bool) -> list[Path]:
    globber = image_dir.rglob if recursive else image_dir.glob
    return sorted(
        path
        for path in globber("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _run_one(*, runner, image_path: Path, max_new_tokens: int | None) -> tuple[str, dict[str, object], Image.Image]:
    image = Image.open(image_path).convert("RGB")
    raw_text, parse_report = runner.predict(image, max_new_tokens=max_new_tokens)
    return raw_text, parse_report, image


def _resolve_batch_size(args: argparse.Namespace, runner: Any) -> int:
    if args.batch_size is not None:
        return max(int(args.batch_size), 1)
    return max(int(getattr(getattr(runner, "settings", None), "batch_size", 1)), 1)


def _run_batch(
    *,
    runner,
    image_paths: list[Path],
    max_new_tokens: int | None,
) -> list[tuple[Path, str, dict[str, object], Image.Image]]:
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    predictions = runner.predict_batch(images, max_new_tokens=max_new_tokens)
    return [
        (image_path, raw_text, parse_report, image)
        for image_path, image, (raw_text, parse_report) in zip(image_paths, images, predictions, strict=False)
    ]


def _print_single_result(raw_text: str, parse_report: dict[str, object]) -> None:
    print(json.dumps(parse_report, ensure_ascii=False, indent=2))
    print("\n[raw-output]")
    print(raw_text)
    generation = parse_report["generation"]
    print(
        "\n".join(
            [
                "[generation]",
                f"requested_max_new_tokens={generation['requested_max_new_tokens']}",
                f"generated_tokens={generation['generated_tokens']}",
                f"returned_tokens={generation['returned_tokens']}",
                f"stop_reason={generation['stop_reason']}",
                f"closed_json_payload={generation['closed_json_payload']}",
                f"hit_max_new_tokens={generation['hit_max_new_tokens']}",
            ]
        )
    )

    lenient_prediction = parse_report["lenient"]["prediction"]
    if lenient_prediction is not None:
        print(format_prediction_summary(lenient_prediction))
    else:
        print("Detected arrows: parse failed")

    print(
        "\n".join(
            [
                f"Lenient parse ok: {parse_report['lenient']['ok']}",
                f"Lenient recovered_prefix: {parse_report['lenient']['recovered_prefix']}",
                f"Strict parse ok: {parse_report['strict']['ok']}",
            ]
        )
    )


def _run_directory_inference(
    *,
    runner,
    image_paths: list[Path],
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int | None,
    save_preview: bool,
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for batch_start in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[batch_start : batch_start + batch_size]
        batch_results = _run_batch(
            runner=runner,
            image_paths=batch_image_paths,
            max_new_tokens=max_new_tokens,
        )
        for batch_index, (image_path, raw_text, parse_report, image) in enumerate(batch_results, start=1):
            index = batch_start + batch_index
            prediction_path, raw_text_path, preview_path = _save_outputs(
                output_dir,
                image_path,
                image,
                raw_text,
                parse_report,
                save_preview=save_preview,
            )
            num_instances = len((parse_report.get("strict", {}).get("prediction") or parse_report.get("lenient", {}).get("prediction") or {"instances": []}).get("instances", []))
            print(f"[{index}/{len(image_paths)}] {image_path.name} | instances={num_instances}")
            manifest.append(
                {
                    "image_path": str(image_path),
                    "report_path": str(prediction_path),
                    "raw_text_path": str(raw_text_path),
                    "preview_path": str(preview_path) if preview_path is not None else None,
                    "num_instances": int(num_instances),
                    "lenient_ok": bool(parse_report.get("lenient", {}).get("ok", False)),
                    "strict_ok": bool(parse_report.get("strict", {}).get("ok", False)),
                }
            )
    return manifest


def main() -> None:
    args = parse_args()
    from vlm_structgen.core.infer import load_inference_runner

    runner = load_inference_runner(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        env_file=args.env_file,
        model_name_or_path=args.model,
        device_name=args.device,
    )
    output_dir = args.output_dir or runner.settings.output_dir
    if args.image is not None:
        image_path = Path(args.image)
        raw_text, parse_report, image = _run_one(
            runner=runner,
            image_path=image_path,
            max_new_tokens=args.max_new_tokens,
        )
        _print_single_result(raw_text, parse_report)
        if output_dir is not None:
            prediction_path, raw_text_path, preview_path = _save_outputs(
                Path(output_dir),
                image_path,
                image,
                raw_text,
                parse_report,
                save_preview=args.save_preview,
            )
            print(f"Saved parsed prediction to: {prediction_path}")
            print(f"Saved raw output to: {raw_text_path}")
            if preview_path is not None:
                print(f"Saved preview to: {preview_path}")
        return

    image_dir = Path(args.image_dir)
    image_paths = _iter_image_paths(image_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")
    if output_dir is None:
        raise ValueError("Batch directory inference requires --output-dir or infer config output_dir.")
    resolved_output_dir = Path(output_dir)
    batch_size = _resolve_batch_size(args, runner)
    print(f"Found {len(image_paths)} images under {image_dir} | batch_size={batch_size}")
    manifest = _run_directory_inference(
        runner=runner,
        image_paths=image_paths,
        output_dir=resolved_output_dir,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        save_preview=args.save_preview,
    )
    manifest_path = resolved_output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved batch manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
