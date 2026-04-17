from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler

from shaft.utils.distributed import get_rank, get_world_size

from .mixing import MixedDatasetBuilder


class ShaftMixedIndexSampler(Sampler[tuple[str, int]]):
    def __init__(
        self,
        records_by_dataset: dict[str, list[Any]],
        dataset_weights: dict[str, float],
        *,
        strategy: str = "interleave_under",
        refresh_mode: str = "static",
        shuffle: bool = True,
        seed: int = 42,
        rank: int | None = None,
        world_size: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.records_by_dataset = {str(name): list(records) for name, records in records_by_dataset.items()}
        self.dataset_weights = {str(name): float(weight) for name, weight in dataset_weights.items()}
        self.strategy = str(strategy).strip().lower()
        self.refresh_mode = str(refresh_mode).strip().lower()
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(get_rank() if rank is None else rank)
        self.world_size = int(get_world_size() if world_size is None else world_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.refresh_count = 0
        self.global_indices: list[tuple[str, int]] = []
        self.current_indices: list[tuple[str, int]] = []
        self._rebuild(epoch=0)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(list(self.current_indices))

    def __len__(self) -> int:
        return len(self.current_indices)

    def set_epoch(self, epoch: int) -> None:
        epoch_value = int(epoch)
        if self.refresh_mode == "epoch_refresh":
            if epoch_value != self.epoch:
                self._rebuild(epoch=epoch_value)
            return
        self.epoch = epoch_value

    def _rebuild(self, *, epoch: int) -> None:
        builder = MixedDatasetBuilder(seed=self.seed + int(epoch))
        global_indices = builder.build_indices(
            self.records_by_dataset,
            self.dataset_weights,
            strategy=self.strategy,
            shuffle=self.shuffle,
        )
        self.global_indices = list(global_indices)
        self.current_indices = self._shard_indices(self.global_indices)
        self.epoch = int(epoch)
        self.refresh_count += 1

    def _shard_indices(self, global_indices: list[tuple[str, int]]) -> list[tuple[str, int]]:
        if self.world_size <= 1:
            return list(global_indices)
        if not global_indices:
            return []
        if self.drop_last:
            total_size = len(global_indices) - (len(global_indices) % self.world_size)
            sharded = global_indices[:total_size]
        else:
            total_size = int(math.ceil(len(global_indices) / self.world_size) * self.world_size)
            padding = total_size - len(global_indices)
            sharded = list(global_indices)
            if padding > 0:
                sharded.extend(global_indices[:padding])
        return list(sharded[self.rank:total_size:self.world_size])
