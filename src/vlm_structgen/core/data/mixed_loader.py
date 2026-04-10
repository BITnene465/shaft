from __future__ import annotations

import math
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
            "Each training record must include task_type and domain_type for mixed sampling."
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
        raw_weight = route_cfg.get("mix_weight", default_weight)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = float(default_weight)
        route_weights[route_key] = max(weight, 0.0)
    return route_weights


@dataclass
class RouteInterleaveSampler:
    route_groups: dict[str, list[int]]
    route_weights: dict[str, float]
    strategy: str = "interleave_under"
    shuffle: bool = True
    seed: int = 0
    world_size: int = 1
    rank: int = 0
    weight_resolution: int = 100

    def __post_init__(self) -> None:
        self._validate_strategy(self.strategy)
        self.world_size = max(int(self.world_size), 1)
        self.rank = max(int(self.rank), 0)
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank must be in [0, world_size), got rank={self.rank}, world_size={self.world_size}."
            )
        self.epoch = 0
        self._global_length = self._estimate_global_length()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self._global_length / self.world_size)

    def __iter__(self) -> Iterator[int]:
        global_indices = self._build_global_indices()
        yield from global_indices[self.rank :: self.world_size]

    @staticmethod
    def _validate_strategy(strategy: str) -> None:
        supported = {"concat", "interleave_under", "interleave_over"}
        if strategy not in supported:
            raise ValueError(
                f"Unsupported mix strategy: {strategy!r}. Supported strategies: {sorted(supported)}."
            )

    def _active_routes(self) -> list[str]:
        routes: list[str] = []
        for route_key, indices in sorted(self.route_groups.items(), key=lambda item: item[0]):
            if len(indices) <= 0:
                continue
            if float(self.route_weights.get(route_key, 0.0)) <= 0.0:
                continue
            routes.append(route_key)
        if not routes:
            raise ValueError("No active routes available for mixed training.")
        return routes

    def _normalized_route_weights(self, active_routes: list[str]) -> dict[str, float]:
        total = sum(max(float(self.route_weights.get(route_key, 0.0)), 0.0) for route_key in active_routes)
        if total <= 0.0:
            uniform = 1.0 / float(len(active_routes))
            return {route_key: uniform for route_key in active_routes}
        return {
            route_key: max(float(self.route_weights.get(route_key, 0.0)), 0.0) / total
            for route_key in active_routes
        }

    def _estimate_global_length(self) -> int:
        active_routes = self._active_routes()
        route_sizes = {route_key: len(self.route_groups[route_key]) for route_key in active_routes}
        if self.strategy == "concat":
            return int(sum(route_sizes.values()))

        weights = self._normalized_route_weights(active_routes)
        if self.strategy == "interleave_under":
            base = min(route_sizes[route_key] / max(weights[route_key], 1e-12) for route_key in active_routes)
            quotas = {
                route_key: min(route_sizes[route_key], int(math.floor(base * weights[route_key])))
                for route_key in active_routes
            }
        else:
            base = max(route_sizes[route_key] / max(weights[route_key], 1e-12) for route_key in active_routes)
            quotas = {
                route_key: max(route_sizes[route_key], int(math.ceil(base * weights[route_key])))
                for route_key in active_routes
            }
        if sum(quotas.values()) <= 0:
            quotas = dict(route_sizes)
        return int(sum(quotas.values()))

    def _build_route_quotas(self, active_routes: list[str]) -> dict[str, int]:
        route_sizes = {route_key: len(self.route_groups[route_key]) for route_key in active_routes}
        if self.strategy == "concat":
            return route_sizes

        weights = self._normalized_route_weights(active_routes)
        if self.strategy == "interleave_under":
            base = min(route_sizes[route_key] / max(weights[route_key], 1e-12) for route_key in active_routes)
            quotas = {
                route_key: min(route_sizes[route_key], int(math.floor(base * weights[route_key])))
                for route_key in active_routes
            }
        else:
            base = max(route_sizes[route_key] / max(weights[route_key], 1e-12) for route_key in active_routes)
            quotas = {
                route_key: max(route_sizes[route_key], int(math.ceil(base * weights[route_key])))
                for route_key in active_routes
            }
        if sum(quotas.values()) <= 0:
            return route_sizes
        return quotas

    def _build_weighted_route_cycle(
        self,
        *,
        active_routes: list[str],
        normalized_weights: dict[str, float],
    ) -> list[str]:
        cycle: list[str] = []
        for route_key in active_routes:
            repeat = int(round(normalized_weights[route_key] * float(self.weight_resolution)))
            cycle.extend([route_key] * max(repeat, 1))
        if self.shuffle:
            random.Random(self.seed + self.epoch * 1009).shuffle(cycle)
        return cycle

    def _build_route_sequence(self, *, route_quotas: dict[str, int]) -> list[str]:
        active_routes = [route_key for route_key, quota in route_quotas.items() if quota > 0]
        if not active_routes:
            return []
        if self.strategy == "concat":
            route_sequence: list[str] = []
            for route_key in active_routes:
                route_sequence.extend([route_key] * route_quotas[route_key])
            if self.shuffle:
                random.Random(self.seed + self.epoch * 1009 + 1).shuffle(route_sequence)
            return route_sequence

        normalized_weights = self._normalized_route_weights(active_routes)
        cycle = self._build_weighted_route_cycle(
            active_routes=active_routes,
            normalized_weights=normalized_weights,
        )
        remaining = dict(route_quotas)
        route_sequence: list[str] = []
        while sum(remaining.values()) > 0:
            progressed = False
            for route_key in cycle:
                if remaining.get(route_key, 0) <= 0:
                    continue
                route_sequence.append(route_key)
                remaining[route_key] -= 1
                progressed = True
                if sum(remaining.values()) <= 0:
                    break
            if not progressed:
                break
        return route_sequence

    def _build_global_indices(self) -> list[int]:
        active_routes = self._active_routes()
        route_quotas = self._build_route_quotas(active_routes)
        route_sequence = self._build_route_sequence(route_quotas=route_quotas)
        rng = random.Random(self.seed + self.epoch * 1009 + 17)
        route_indices = {route_key: list(self.route_groups[route_key]) for route_key in active_routes}
        if self.shuffle:
            for indices in route_indices.values():
                rng.shuffle(indices)
        cursors = {route_key: 0 for route_key in active_routes}

        mixed_indices: list[int] = []
        for route_key in route_sequence:
            candidates = route_indices[route_key]
            if not candidates:
                continue
            cursor = cursors[route_key]
            if cursor > 0 and cursor % len(candidates) == 0 and self.shuffle:
                rng.shuffle(candidates)
            mixed_indices.append(candidates[cursor % len(candidates)])
            cursors[route_key] = cursor + 1
        return mixed_indices


def build_mixed_train_loader(
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
    mix_strategy: str = "interleave_under",
    seed: int = 0,
    weight_resolution: int = 100,
):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError:  # pragma: no cover
        import torch

        DataLoader = torch.utils.data.DataLoader

    route_groups = collect_route_groups(dataset.records)
    if not route_groups:
        raise ValueError("Training dataset is empty; cannot build mixed train loader.")

    route_weights = extract_route_weights(route_options, route_groups.keys())
    sampler = RouteInterleaveSampler(
        route_groups=route_groups,
        route_weights=route_weights,
        strategy=str(mix_strategy),
        shuffle=bool(shuffle),
        seed=int(seed),
        world_size=int(world_size) if distributed else 1,
        rank=int(rank) if distributed else 0,
        weight_resolution=weight_resolution,
    )
    return DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collator,
    )
