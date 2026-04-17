from __future__ import annotations

from typing import Any


class ShaftTrainSamplerMixin:
    def __init__(self, *args: Any, train_sampler: Any = None, **kwargs: Any) -> None:
        self.train_sampler = train_sampler
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self.train_sampler is not None:
            return self.train_sampler
        return super()._get_train_sampler(train_dataset)
