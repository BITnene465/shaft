from __future__ import annotations

from collections.abc import Iterator
import math

from torch.utils.data import Sampler

from shaft.utils.distributed import get_rank, get_world_size

from .mixing import ShaftSamplePlan, ShaftSampleRef, _affine_permute, _splitmix64


class ShaftSampleSampler(Sampler[ShaftSampleRef]):
    """Lazily emit immutable refs; no Python tuple plan is materialized or copied."""

    def __init__(
        self,
        plan: ShaftSamplePlan,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        drop_last: bool = False,
    ) -> None:
        self.plan = plan
        self.rank = int(get_rank() if rank is None else rank)
        self.world_size = int(get_world_size() if world_size is None else world_size)
        self.drop_last = bool(drop_last)
        self.plan_cycle = 0
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid sampler rank/world_size: {self.rank}/{self.world_size}.")

    def __iter__(self) -> Iterator[ShaftSampleRef]:
        total_size = len(self.plan)
        if self.drop_last:
            total_size -= total_size % self.world_size
        for position in range(self.rank, total_size, self.world_size):
            yield self.plan.ref_at(position, plan_cycle=self.plan_cycle)

    def __len__(self) -> int:
        total_size = len(self.plan)
        if self.drop_last:
            return total_size // self.world_size
        if total_size <= self.rank:
            return 0
        return int(math.ceil((total_size - self.rank) / self.world_size))

    def set_epoch(self, epoch: int) -> None:
        self.plan_cycle = int(epoch)


class ShaftGroupedSampleSampler(Sampler[ShaftSampleRef]):
    """GRPO-compatible grouped repeats with deterministic, resumable plan cycles."""

    def __init__(
        self,
        plan: ShaftSamplePlan,
        *,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.plan = plan
        self.mini_repeat_count = int(mini_repeat_count)
        self.batch_size = int(batch_size)
        self.repeat_count = int(repeat_count)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.plan_cycle = 0
        if min(self.mini_repeat_count, self.batch_size, self.repeat_count) <= 0:
            raise ValueError("Grouped sampler repeat counts and batch_size must be > 0.")

    def __iter__(self) -> Iterator[ShaftSampleRef]:
        usable_size = (len(self.plan) // self.batch_size) * self.batch_size
        permutation_seed = _splitmix64(self.seed ^ self.plan_cycle)
        for chunk_start in range(0, usable_size, self.batch_size):
            chunk: list[ShaftSampleRef] = []
            for offset in range(self.batch_size):
                position = chunk_start + offset
                if self.shuffle:
                    position = _affine_permute(
                        position,
                        len(self.plan),
                        seed=permutation_seed,
                    )
                chunk.append(self.plan.ref_at(position, plan_cycle=self.plan_cycle))
            for _ in range(self.repeat_count):
                for sample_ref in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield sample_ref

    def __len__(self) -> int:
        usable_size = (len(self.plan) // self.batch_size) * self.batch_size
        return usable_size * self.mini_repeat_count * self.repeat_count

    def set_epoch(self, epoch: int) -> None:
        self.plan_cycle = int(epoch)
