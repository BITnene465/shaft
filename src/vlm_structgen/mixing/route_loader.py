from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


def _route_key_from_record(record: Mapping[str, Any]) -> str:
    task_type = str(record.get("task_type", "")).strip()
    domain_type = str(record.get("domain_type", "")).strip()
    if not task_type or not domain_type:
        raise ValueError(
            "Each training record must include task_type and domain_type for route-aware mixing."
        )
    return f"{task_type}/{domain_type}"


def collect_route_groups(records: Iterable[Mapping[str, Any]]) -> dict[str, list[int]]:
    route_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        route_groups[_route_key_from_record(record)].append(index)
    return dict(route_groups)


def extract_route_weights(
    route_options: Mapping[str, Mapping[str, Any]] | None,
    route_keys: Iterable[str],
    *,
    default_weight: float = 1.0,
) -> dict[str, float]:
    route_weights: dict[str, float] = {}
    options = dict(route_options or {})
    for route_key in route_keys:
        route_cfg = dict(options.get(route_key, {}))
        raw_weight = route_cfg.get("mix_weight", route_cfg.get("sampling_weight", default_weight))
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = float(default_weight)
        route_weights[route_key] = max(weight, 0.0)
    return route_weights


@dataclass
class RouteEpochController:
    route_loaders: dict[str, Any]
    route_weights: dict[str, float]
    weight_resolution: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for loader in self.route_loaders.values():
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)

    def build_route_schedule(self, active_routes: Iterable[str], cycle_index: int) -> list[str]:
        schedule: list[str] = []
        for route_key in active_routes:
            weight = float(self.route_weights.get(route_key, 1.0))
            repeat = int(round(weight * float(self.weight_resolution)))
            if repeat > 0:
                schedule.extend([route_key] * repeat)
        if not schedule:
            schedule = list(active_routes)
        rng = random.Random(self.seed + self.epoch * 1009 + cycle_index * 9173)
        rng.shuffle(schedule)
        return schedule


class RouteAwareTrainLoader:
    def __init__(
        self,
        route_loaders: dict[str, Any],
        *,
        controller: RouteEpochController,
    ) -> None:
        self.route_loaders = dict(route_loaders)
        self.sampler = controller

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.route_loaders.values())

    def __iter__(self) -> Iterator[Any]:
        active_iterators = {route_key: iter(loader) for route_key, loader in self.route_loaders.items()}
        cycle_index = 0
        while active_iterators:
            active_routes = list(active_iterators.keys())
            route_schedule = self.sampler.build_route_schedule(active_routes, cycle_index)
            cycle_index += 1
            progressed = False
            for route_key in route_schedule:
                iterator = active_iterators.get(route_key)
                if iterator is None:
                    continue
                try:
                    batch = next(iterator)
                except StopIteration:
                    active_iterators.pop(route_key, None)
                    continue
                progressed = True
                yield batch
            if not progressed:
                break


def build_route_aware_train_loader(
    dataset,
    collator,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    distributed: bool,
    world_size: int,
    rank: int,
    *,
    shuffle: bool = True,
    route_options: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int = 0,
    weight_resolution: int = 10,
) -> RouteAwareTrainLoader:
    from torch.utils.data import DataLoader, DistributedSampler, Subset

    route_groups = collect_route_groups(dataset.records)
    if not route_groups:
        raise ValueError("Training dataset is empty; cannot build route-aware loader.")

    route_weights = extract_route_weights(route_options, route_groups.keys())
    route_loaders: dict[str, DataLoader] = {}

    for route_key, indices in sorted(route_groups.items(), key=lambda item: item[0]):
        route_subset = Subset(dataset, indices)
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                route_subset,
                num_replicas=max(int(world_size), 1),
                rank=max(int(rank), 0),
                shuffle=shuffle,
                seed=seed,
            )
        route_loaders[route_key] = DataLoader(
            route_subset,
            batch_size=max(int(batch_size), 1),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collator,
        )

    controller = RouteEpochController(
        route_loaders=route_loaders,
        route_weights=route_weights,
        weight_resolution=weight_resolution,
        seed=seed + (world_size * 13 if distributed else 0),
    )
    return RouteAwareTrainLoader(route_loaders, controller=controller)
