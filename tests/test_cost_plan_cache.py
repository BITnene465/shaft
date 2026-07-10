from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import json

import pytest

from shaft.data import (
    ShaftCostAwareSampler,
    ShaftFixedBatchPlanningSpec,
    ShaftSampleCost,
    ShaftSamplePlan,
    cost_plan_reference_path,
    load_cost_plan_reference,
    materialize_cost_plan,
    write_cost_plan_reference,
)


pytestmark = pytest.mark.component


class _DrawCostProvider:
    def __init__(
        self,
        *,
        fingerprint: str = "draw-cost-v1",
        fail_on_call: bool = False,
        token_offset: int = 0,
    ):
        self.fingerprint = fingerprint
        self.fail_on_call = bool(fail_on_call)
        self.token_offset = int(token_offset)
        self.calls: list[int] = []

    def __call__(self, sample_ref):
        if self.fail_on_call:
            raise AssertionError("cache hit must not recompute sample costs")
        draw_id = int(sample_ref.context.draw_id)
        self.calls.append(draw_id)
        return ShaftSampleCost(
            llm_tokens=draw_id + 1 + self.token_offset,
            supervised_tokens=draw_id,
            vision_patches=(draw_id + 1) * 16,
            loss_weight_sum=None if draw_id == 0 else draw_id + 0.5,
            exact=(draw_id % 2 == 0),
        )


class _FileCountingCostProvider(_DrawCostProvider):
    def __init__(self, calls_path: str):
        super().__init__(fingerprint="concurrent-cost-v1")
        self.calls_path = calls_path

    def __call__(self, sample_ref):
        cost = super().__call__(sample_ref)
        descriptor = os.open(
            self.calls_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, f"{sample_ref.context.draw_id}\n".encode("utf-8"))
        finally:
            os.close(descriptor)
        return cost


def _build_repeating_plan(*, seed: int = 17) -> ShaftSamplePlan:
    return ShaftSamplePlan(
        {"dataset": 2},
        {"dataset": 1.0},
        strategy="concat",
        num_samples=4,
        shuffle=False,
        seed=seed,
    )


def _materialize_in_subprocess(cache_dir: str, calls_path: str) -> None:
    materialized = materialize_cost_plan(
        _build_repeating_plan(),
        cost_provider=_FileCountingCostProvider(calls_path),
        cache_dir=cache_dir,
    )
    materialized.provider.close()


def test_mmap_cost_plan_roundtrip_preserves_draw_specific_costs(tmp_path: Path) -> None:
    plan = _build_repeating_plan()
    runtime_provider = _DrawCostProvider()

    materialized = materialize_cost_plan(
        plan,
        cost_provider=runtime_provider,
        cache_dir=tmp_path / "cache",
    )
    try:
        assert materialized.cache_hit is False
        assert runtime_provider.calls == [0, 1, 2, 3]
        assert materialized.data_bytes > 0
        assert materialized.provider.semantic_fingerprint == runtime_provider.fingerprint
        assert materialized.provider.fingerprint == materialized.provider.manifest.fingerprint
        assert [materialized.provider(plan.ref_at(index)) for index in range(4)] == [
            ShaftSampleCost(
                llm_tokens=index + 1,
                supervised_tokens=index,
                vision_patches=(index + 1) * 16,
                loss_weight_sum=None if index == 0 else index + 0.5,
                exact=(index % 2 == 0),
            )
            for index in range(4)
        ]

        run_dir = tmp_path / "run"
        reference = write_cost_plan_reference(run_dir, materialized)
        assert reference == cost_plan_reference_path(run_dir)
    finally:
        materialized.provider.close()

    loaded = load_cost_plan_reference(run_dir, plan=plan, verify_checksum=True)
    try:
        assert loaded(plan.ref_at(3)).llm_tokens == 4
        assert loaded(plan.ref_at(0)).loss_weight_sum is None
    finally:
        loaded.close()


def test_cost_plan_cache_hit_does_not_recompute_costs(tmp_path: Path) -> None:
    plan = _build_repeating_plan()
    first = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(),
        cache_dir=tmp_path,
    )
    first_manifest_path = first.manifest_path
    first.provider.close()

    second = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(fail_on_call=True),
        cache_dir=tmp_path,
    )
    try:
        assert second.cache_hit is True
        assert second.manifest_path == first_manifest_path
        assert second.provider(plan.ref_at(2)).llm_tokens == 3
    finally:
        second.provider.close()


def test_cost_plan_cache_key_invalidates_on_cost_fingerprint_change(
    tmp_path: Path,
) -> None:
    plan = _build_repeating_plan()
    first = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(fingerprint="cost-v1"),
        cache_dir=tmp_path,
    )
    first.provider.close()
    changed_provider = _DrawCostProvider(fingerprint="cost-v2")

    second = materialize_cost_plan(
        plan,
        cost_provider=changed_provider,
        cache_dir=tmp_path,
    )
    try:
        assert second.cache_hit is False
        assert changed_provider.calls == [0, 1, 2, 3]
        assert second.manifest_path != first.manifest_path
    finally:
        second.provider.close()


def test_batch_signature_binds_cost_plan_content_not_only_semantic_fingerprint(
    tmp_path: Path,
) -> None:
    plan = _build_repeating_plan()
    first = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(fingerprint="same-semantic-cost"),
        cache_dir=tmp_path / "first",
    )
    second = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(
            fingerprint="same-semantic-cost",
            token_offset=10,
        ),
        cache_dir=tmp_path / "second",
    )
    spec = ShaftFixedBatchPlanningSpec.from_plan(
        plan,
        per_device_batch_size=1,
        data_world_size=1,
        gradient_accumulation_steps=1,
        planning_window=4,
    )
    try:
        first_signature = ShaftCostAwareSampler(
            plan,
            cost_provider=first.provider,
            spec=spec,
        ).signature
        second_signature = ShaftCostAwareSampler(
            plan,
            cost_provider=second.provider,
            spec=spec,
        ).signature

        assert first.provider.semantic_fingerprint == second.provider.semantic_fingerprint
        assert first.provider.fingerprint != second.provider.fingerprint
        assert first_signature.cost_fingerprint != second_signature.cost_fingerprint
        assert first_signature.fingerprint != second_signature.fingerprint
    finally:
        first.provider.close()
        second.provider.close()


def test_cost_plan_rebuilds_a_truncated_cache_atomically(tmp_path: Path) -> None:
    plan = _build_repeating_plan()
    first = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(),
        cache_dir=tmp_path,
    )
    data_path = first.provider.data_path
    first.provider.close()
    data_path.write_bytes(b"truncated")
    rebuilding_provider = _DrawCostProvider()

    rebuilt = materialize_cost_plan(
        plan,
        cost_provider=rebuilding_provider,
        cache_dir=tmp_path,
    )
    try:
        assert rebuilt.cache_hit is False
        assert rebuilding_provider.calls == [0, 1, 2, 3]
        assert rebuilt.provider(plan.ref_at(3)).vision_patches == 64
    finally:
        rebuilt.provider.close()


@pytest.mark.parametrize("malformed_value", ["broken", None, []])
def test_cost_plan_rebuilds_a_manifest_with_malformed_scalar(
    tmp_path: Path,
    malformed_value,
) -> None:
    plan = _build_repeating_plan()
    first = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(),
        cache_dir=tmp_path,
    )
    manifest_path = first.manifest_path
    first.provider.close()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sample_count"] = malformed_value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    rebuilding_provider = _DrawCostProvider()

    rebuilt = materialize_cost_plan(
        plan,
        cost_provider=rebuilding_provider,
        cache_dir=tmp_path,
    )
    try:
        assert rebuilt.cache_hit is False
        assert rebuilding_provider.calls == [0, 1, 2, 3]
    finally:
        rebuilt.provider.close()


def test_cost_plan_rebuilds_equal_size_bit_flip(tmp_path: Path) -> None:
    plan = _build_repeating_plan()
    first = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(),
        cache_dir=tmp_path,
    )
    data_path = first.provider.data_path
    first.provider.close()
    corrupted = bytearray(data_path.read_bytes())
    corrupted[-1] ^= 0x01
    data_path.write_bytes(corrupted)
    rebuilding_provider = _DrawCostProvider()

    rebuilt = materialize_cost_plan(
        plan,
        cost_provider=rebuilding_provider,
        cache_dir=tmp_path,
    )
    try:
        assert rebuilt.cache_hit is False
        assert rebuilding_provider.calls == [0, 1, 2, 3]
    finally:
        rebuilt.provider.close()


def test_cost_plan_reference_rejects_plan_or_cycle_mismatch(tmp_path: Path) -> None:
    plan = _build_repeating_plan(seed=17)
    materialized = materialize_cost_plan(
        plan,
        cost_provider=_DrawCostProvider(),
        cache_dir=tmp_path / "cache",
    )
    write_cost_plan_reference(tmp_path / "run", materialized)
    materialized.provider.close()

    with pytest.raises(ValueError, match="SamplePlan fingerprint"):
        load_cost_plan_reference(
            tmp_path / "run",
            plan=_build_repeating_plan(seed=19),
        )

    loaded = load_cost_plan_reference(tmp_path / "run", plan=plan)
    try:
        with pytest.raises(ValueError, match="plan_cycle=0"):
            loaded(plan.ref_at(0, plan_cycle=1))
    finally:
        loaded.close()


def test_concurrent_cost_plan_materialization_builds_once(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    cache_dir = str(tmp_path / "cache")
    calls_path = str(tmp_path / "calls.txt")
    processes = [
        context.Process(
            target=_materialize_in_subprocess,
            args=(cache_dir, calls_path),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    calls = (tmp_path / "calls.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(calls) == ["0", "1", "2", "3"]
