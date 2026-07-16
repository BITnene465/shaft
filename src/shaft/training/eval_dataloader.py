from __future__ import annotations

from typing import Any


class ShaftEvalDataLoaderMixin:
    """Route loss and generation eval through collators distinct from training."""

    def _configure_eval_data_collator(self, eval_data_collator: Any | None) -> None:
        self.eval_data_collator = eval_data_collator
        self._shaft_train_data_collator = self.data_collator

    def get_eval_dataloader(self, eval_dataset: Any = None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        resolved_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        cache_key = (
            eval_dataset
            if isinstance(eval_dataset, str)
            else f"eval:{id(resolved_dataset)}"
            if eval_dataset is not None
            else "eval"
        )
        data_collator = self.data_collator
        if (
            self.eval_data_collator is not None
            and data_collator is self._shaft_train_data_collator
        ):
            data_collator = self.eval_data_collator
        return self._build_shaft_eval_loader(
            dataset=resolved_dataset,
            data_collator=data_collator,
            cache_key=str(cache_key),
            description="Evaluation",
        )

    def get_online_eval_dataloader(
        self,
        eval_dataset: Any,
        *,
        data_collator: Any,
        dataset_key: str,
    ):
        resolved_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
        )
        normalized_key = str(dataset_key).strip()
        if not normalized_key:
            raise ValueError("Online eval dataset_key must not be empty.")
        return self._build_shaft_eval_loader(
            dataset=resolved_dataset,
            data_collator=data_collator,
            cache_key=f"shaft-online:{normalized_key}",
            description="Online Evaluation",
        )

    def _build_shaft_eval_loader(
        self,
        *,
        dataset: Any,
        data_collator: Any,
        cache_key: str,
        description: str,
    ):
        if dataset is None:
            raise ValueError(f"Trainer: {description.lower()} requires an eval_dataset.")
        cached_loaders = getattr(self, "_eval_dataloaders", {})
        if self.args.dataloader_persistent_workers and cache_key in cached_loaders:
            return cached_loaders[cache_key]
        previous_collator = self.data_collator
        self.data_collator = data_collator
        try:
            return self._get_dataloader(
                dataset=dataset,
                description=description,
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler,
                dataloader_key=cache_key,
            )
        finally:
            self.data_collator = previous_collator
