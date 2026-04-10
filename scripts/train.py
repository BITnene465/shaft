#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from vlm_structgen.core import apply_run_id, config_to_dict, load_config
from vlm_structgen.core.data import SFTCollator, SFTDataset, build_mixed_train_loader
from vlm_structgen.core.eval import Evaluator
from vlm_structgen.core.modeling import build_model_tokenizer_processor
from vlm_structgen.core.train import Trainer
from vlm_structgen.core.train.optim import build_optimizer, build_scheduler
from vlm_structgen.core.utils.distributed import barrier, cleanup_distributed, init_distributed, seed_everything
from vlm_structgen.core.utils.logging import ExperimentLogger, format_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on routed multimodal structured-generation tasks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--mix-strategy",
        choices=["concat", "interleave_under", "interleave_over"],
        default=None,
    )
    return parser.parse_args()


def _build_dataloader(dataset, collator, batch_size, num_workers, pin_memory, persistent_workers, distributed, shuffle):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collator,
    )


class _SortedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        indices: list[int],
        batch_size: int,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.indices = list(indices)[int(rank) :: max(int(world_size), 1)]

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.indices), self.batch_size):
            yield self.indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)


def _build_val_dataloader(
    dataset,
    collator,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    distributed,
    world_size,
    rank,
    tokenizer,
    bucket_by_target_length: bool,
):
    if not bucket_by_target_length:
        return _build_dataloader(
            dataset,
            collator,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            distributed,
            shuffle=False,
        )
    target_lengths = dataset.get_target_token_lengths(tokenizer)
    sorted_indices = [index for index, _length in sorted(enumerate(target_lengths), key=lambda item: item[1])]
    batch_sampler = _SortedBatchSampler(
        sorted_indices,
        batch_size,
        world_size=world_size if distributed else 1,
        rank=rank if distributed else 0,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collator,
    )


def _resolve_jsonl_paths(
    configured_paths: str | list[str],
    *,
    field_name: str,
) -> list[str]:
    if isinstance(configured_paths, list):
        raw_paths = [str(path) for path in configured_paths]
    else:
        raw_paths = str(configured_paths).replace(";", ",").split(",")
    resolved = [str(Path(path.strip())) for path in raw_paths if path.strip()]
    if not resolved:
        raise ValueError(f"No paths were resolved from {field_name}.")
    return resolved


def _resolve_dataset_routes(
    *,
    paths: list[str],
    config_route: str | None,
    config_route_map: dict[str, str] | None,
    fallback_route: str | None,
    field_name: str,
) -> list[str]:
    route_map = {
        str(Path(path_key)): str(route_value).strip()
        for path_key, route_value in dict(config_route_map or {}).items()
        if str(route_value).strip()
    }
    if route_map:
        normalized_paths = [str(Path(path)) for path in paths]
        missing_paths = [path for path in normalized_paths if path not in route_map]
        if missing_paths:
            raise ValueError(
                f"{field_name} is missing routes for paths: {missing_paths}. "
                "Provide a path->route entry for each dataset path."
            )
        unknown_paths = sorted(set(route_map.keys()) - set(normalized_paths))
        if unknown_paths:
            raise ValueError(
                f"{field_name} contains unknown path keys: {unknown_paths}. "
                f"Expected one of: {normalized_paths}."
            )
        return [route_map[path] for path in normalized_paths]
    if config_route is not None and str(config_route).strip():
        return [str(config_route).strip()] * len(paths)
    if fallback_route is not None and str(fallback_route).strip():
        return [str(fallback_route).strip()] * len(paths)
    raise ValueError(
        f"Missing dataset route config for {field_name}. "
        "Please set data.train_route_map/data.val_route_map, "
        "or data.train_route/data.val_route, or task.route."
    )


def main() -> None:
    args = parse_args()
    print("[startup] loading config...", flush=True)
    config = load_config(args.config)
    if args.run_id:
        config = apply_run_id(config, args.run_id, stage_name=args.stage_name)
        print(
            "[startup] applied "
            f"run_id={args.run_id!r} stage_name={args.stage_name!r} "
            f"output_dir={config.experiment.output_dir}",
            flush=True,
        )
    if args.seed is not None:
        config.experiment.seed = int(args.seed)
    if args.epochs is not None:
        config.train.epochs = int(args.epochs)
    if args.lr is not None:
        config.train.learning_rate = float(args.lr)
    if args.init_from is not None:
        config.checkpoint.init_from = str(args.init_from)
    if args.resume_from is not None:
        config.checkpoint.resume_from = str(args.resume_from)
    if args.mix_strategy is not None:
        config.task.mix_strategy = str(args.mix_strategy)
    train_paths = _resolve_jsonl_paths(
        config.data.train_path,
        field_name="config.data.train_path",
    )
    val_paths = _resolve_jsonl_paths(
        config.data.val_path,
        field_name="config.data.val_path",
    )
    train_routes = _resolve_dataset_routes(
        paths=train_paths,
        config_route=config.data.train_route,
        config_route_map=config.data.train_route_map,
        fallback_route=config.task.route,
        field_name="data.train_route_map",
    )
    val_routes = _resolve_dataset_routes(
        paths=val_paths,
        config_route=config.data.val_route,
        config_route_map=config.data.val_route_map,
        fallback_route=config.task.route,
        field_name="data.val_route_map",
    )
    dist_ctx = init_distributed()
    seed_everything(config.experiment.seed, rank=dist_ctx.rank)

    print("[startup] building model, tokenizer, and processor...", flush=True)
    build_artifacts = build_model_tokenizer_processor(config)
    print("[startup] building codec and collator...", flush=True)
    train_collator = SFTCollator(
        processor=build_artifacts.processor,
        tokenizer=build_artifacts.tokenizer,
        num_bins=config.tokenizer.num_bins,
        task_route_options=config.task.route_options,
        add_eos_token=config.tokenizer.add_eos_token,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_targets_in_inputs=True,
    )
    val_collator = SFTCollator(
        processor=build_artifacts.processor,
        tokenizer=build_artifacts.tokenizer,
        num_bins=config.tokenizer.num_bins,
        task_route_options=config.task.route_options,
        add_eos_token=config.tokenizer.add_eos_token,
        min_pixels=config.model.min_pixels,
        max_pixels=config.model.max_pixels,
        include_targets_in_inputs=False,
        padding_side="left",  # decoder-only models typically benefit from left-padding during evaluation for better efficiency
    )
    print("[startup] loading datasets...", flush=True)
    train_dataset = SFTDataset(
        jsonl_path=train_paths,
        path_routes=train_routes,
        num_bins=config.tokenizer.num_bins,
        system_prompt=config.prompt.system_prompt,
        user_prompt=config.prompt.user_prompt,
        system_prompt_template=config.prompt.system_prompt_template,
        user_prompt_template=config.prompt.user_prompt_template,
        route_prompts=config.prompt.route_prompts,
    )
    val_dataset = SFTDataset(
        jsonl_path=val_paths,
        path_routes=val_routes,
        num_bins=config.tokenizer.num_bins,
        system_prompt=config.prompt.system_prompt,
        user_prompt=config.prompt.user_prompt,
        system_prompt_template=config.prompt.system_prompt_template,
        user_prompt_template=config.prompt.user_prompt_template,
        route_prompts=config.prompt.route_prompts,
    )
    print("[startup] building dataloaders...", flush=True)
    train_loader = build_mixed_train_loader(
        train_dataset,
        train_collator,
        config.train.per_device_batch_size,
        config.data.num_workers,
        config.data.pin_memory,
        config.data.persistent_workers,
        dist_ctx.distributed,
        dist_ctx.world_size,
        dist_ctx.rank,
        shuffle=True,
        route_options=config.task.route_options,
        mix_strategy=config.task.mix_strategy,
        seed=config.experiment.seed,
    )
    val_loader = _build_val_dataloader(
        val_dataset,
        val_collator,
        config.eval.per_device_batch_size,
        config.data.num_workers,
        config.data.pin_memory,
        config.data.persistent_workers,
        dist_ctx.distributed,
        dist_ctx.world_size,
        dist_ctx.rank,
        build_artifacts.tokenizer,
        config.eval.bucket_by_target_length,
    )

    total_steps_per_epoch = math.ceil(
        len(train_loader) / max(config.train.grad_accum_steps, 1)
    )
    total_steps = max(total_steps_per_epoch * config.train.epochs, 1)
    print("[startup] building optimizer and scheduler...", flush=True)
    optimizer = build_optimizer(build_artifacts.model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)
    print("[startup] initializing logger...", flush=True)
    logger = ExperimentLogger(
        output_dir=config.experiment.output_dir,
        use_wandb=config.logging.use_wandb,
        project=config.logging.project,
        run_name=config.logging.run_name or config.experiment.name,
        config=config_to_dict(config),
    )
    trainable_params = build_artifacts.trainable_summary["trainable_params"]
    total_params = build_artifacts.trainable_summary["total_params"]
    trainable_ratio = 100.0 * trainable_params / max(total_params, 1)
    logger.info(
        "Loaded model; "
        f"trainable={format_count(trainable_params)} / {format_count(total_params)} "
        f"({trainable_ratio:.2f}%)"
    )
    evaluator = Evaluator(
        num_bins=config.tokenizer.num_bins,
        tokenizer=build_artifacts.tokenizer,
        task_route_options=config.task.route_options,
        max_new_tokens=config.eval.max_new_tokens,
        num_beams=config.eval.num_beams,
        do_sample=config.eval.do_sample,
        temperature=config.eval.temperature,
        top_p=config.eval.top_p,
        top_k=config.eval.top_k,
        use_cache=config.eval.use_cache,
        bbox_iou_threshold=config.eval.bbox_iou_threshold,
        strict_point_distance_px=config.eval.strict_point_distance_px,
    )
    print("[startup] building trainer...", flush=True)
    trainer = Trainer(
        model=build_artifacts.model,
        tokenizer=build_artifacts.tokenizer,
        processor=build_artifacts.processor,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=dist_ctx.device,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
        evaluator=evaluator,
        logger=logger,
    )
    init_path = config.checkpoint.init_from
    resume_path = config.checkpoint.resume_from
    if init_path and resume_path:
        raise ValueError("`init-from` and `resume-from` are mutually exclusive. Choose only one.")
    if init_path:
        print(f"[startup] initializing model weights from checkpoint: {init_path}", flush=True)
        init_meta = trainer.initialize_model_from_checkpoint(init_path, strict=True)
        init_mode = init_meta.get("config", {}).get("finetune", {}).get("mode")
        if init_mode:
            print(f"[startup] loaded initialization weights from finetune.mode={init_mode}", flush=True)
    if resume_path:
        print(f"[startup] resuming from checkpoint: {resume_path}", flush=True)
        trainer.load_checkpoint(resume_path, strict=True, resume_training_state=True)
    print("[startup] start training.", flush=True)
    trainer.fit()
    barrier()
    logger.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
