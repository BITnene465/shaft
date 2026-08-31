from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from safetensors import safe_open
import torch

from shaft.opd.input_abi import (
    ShaftOPDInputABI,
    validate_opd_input_abi_compatibility,
)
from shaft.training.distribution_loss import TeacherDistribution


ARTIFACT_VERSION = "shaft-offline-kd-artifact-v1"
INPUT_CONTRACT_VERSION = "shaft-offline-kd-input-contract-v1"
_HEX = frozenset("0123456789abcdef")


def _digest(value: Any, *, role: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in _HEX for character in normalized):
        raise ValueError(f"{role} must be a SHA-256 digest.")
    return normalized


def _exact_mapping(value: Any, *, role: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{role} must be a mapping.")
    payload = dict(value)
    if set(payload) != keys:
        raise ValueError(
            f"{role} keys differ: missing={sorted(keys - set(payload))} "
            f"unknown={sorted(set(payload) - keys)}."
        )
    return payload


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without creating a second in-memory copy of the shard."""

    if int(chunk_size) <= 0:
        raise ValueError("SHA-256 chunk_size must be > 0.")
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            hasher.update(chunk)
    return hasher.hexdigest()


def offline_kd_artifact_identity(
    *,
    teacher: Mapping[str, Any],
    input_abi: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    distribution: Mapping[str, Any],
    build: Mapping[str, Any],
) -> str:
    """Canonical semantic identity shared by the v1 producer and reader."""

    return canonical_sha256(
        {
            "version": ARTIFACT_VERSION,
            "teacher": dict(teacher),
            "input_abi": dict(input_abi),
            "input_contract": dict(input_contract),
            "distribution": dict(distribution),
            "build": dict(build),
        }
    )


def merge_offline_kd_artifacts(
    artifact_dirs: list[str | Path],
    *,
    output_dir: str | Path,
) -> Path:
    """Atomically merge compatible map shards into one reader-compatible artifact."""

    inputs = [Path(path).resolve() for path in artifact_dirs]
    if not inputs:
        raise ValueError("Offline KD merge requires at least one input artifact.")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Offline KD merge input artifact directories must be unique.")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Offline KD merge output already exists: {output}")
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in inputs]
    first = manifests[0]
    expected_input_abi = ShaftOPDInputABI.from_mapping(first["input_abi"])
    expected_input_contract = ShaftOfflineKDInputContract.from_mapping(
        first["input_contract"]
    )
    # Reuse the reader's fail-closed manifest validation rather than maintaining a
    # second, weaker merge-only parser. Shard bytes are verified below before publication.
    for input_dir in inputs:
        OfflineKDArtifactStore(
            input_dir / "manifest.json",
            student_input_abi=expected_input_abi,
            student_input_contract=expected_input_contract,
            max_cached_shards=1,
        )
    shared_fields = ("version", "teacher", "input_abi", "input_contract", "distribution")
    for index, manifest in enumerate(manifests):
        if any(manifest.get(field) != first.get(field) for field in shared_fields):
            raise ValueError(f"Offline KD input artifact {index} is not merge-compatible.")
        if manifest.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported Offline KD merge input {manifest.get('version')!r}.")
        if manifest.get("build", {}).get("denylist_fingerprint") != first["build"][
            "denylist_fingerprint"
        ]:
            raise ValueError("Offline KD merge inputs use different denylists.")
    source_fingerprint = canonical_sha256(
        {
            "kind": "shaft-offline-kd-map-merge-v1",
            "input_artifact_ids": [manifest["artifact_id"] for manifest in manifests],
        }
    )
    build = {
        "source_fingerprint": source_fingerprint,
        "denylist_fingerprint": first["build"]["denylist_fingerprint"],
    }
    artifact_id = offline_kd_artifact_identity(
        teacher=first["teacher"],
        input_abi=first["input_abi"],
        input_contract=first["input_contract"],
        distribution=first["distribution"],
        build=build,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.merge-", dir=output.parent))
    merged_shards: dict[str, str] = {}
    try:
        with (temporary / "train.jsonl").open("w", encoding="utf-8") as target_jsonl:
            for input_index, (input_dir, manifest) in enumerate(
                zip(inputs, manifests, strict=True)
            ):
                shard_names = list(manifest.get("shards", {}))
                shard_map: dict[str, str] = {}
                for shard_index, old_name in enumerate(shard_names, start=1):
                    new_name = f"teacher-{input_index + 1:03d}-{shard_index:05d}.safetensors"
                    source_path = input_dir / old_name
                    expected_digest = manifest["shards"][old_name]
                    if file_sha256(source_path) != expected_digest:
                        raise ValueError(f"Offline KD merge shard checksum mismatch: {source_path}")
                    target_path = temporary / new_name
                    try:
                        os.link(source_path, target_path)
                    except OSError:
                        shutil.copy2(source_path, target_path)
                    merged_shards[new_name] = expected_digest
                    shard_map[old_name] = new_name
                with (input_dir / "train.jsonl").open(encoding="utf-8") as source_jsonl:
                    for line_no, line in enumerate(source_jsonl, start=1):
                        row = json.loads(line)
                        reference = OfflineKDArtifactReference.from_mapping(
                            row.get("distillation_ref")
                        )
                        if reference.artifact_id != manifest["artifact_id"]:
                            raise ValueError(
                                f"Offline KD merge reference identity differs: "
                                f"{input_dir}/train.jsonl:{line_no}"
                            )
                        if reference.shard not in shard_map:
                            raise ValueError("Offline KD merge reference names an unknown shard.")
                        row["distillation_ref"] = {
                            "artifact_id": artifact_id,
                            "shard": shard_map[reference.shard],
                            "row": reference.row,
                        }
                        target_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            target_jsonl.flush()
            os.fsync(target_jsonl.fileno())
        merged_manifest = {
            "version": ARTIFACT_VERSION,
            "artifact_id": artifact_id,
            "teacher": first["teacher"],
            "input_abi": first["input_abi"],
            "input_contract": first["input_contract"],
            "distribution": first["distribution"],
            "build": build,
            "shards": merged_shards,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(merged_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@lru_cache(maxsize=65536)
def _cached_file_sha256(
    resolved_path: str,
    stat_identity: tuple[int, int, int, int, int],
) -> bytes:
    _ = stat_identity
    return hashlib.sha256(Path(resolved_path).read_bytes()).digest()


def clear_media_fingerprint_cache() -> None:
    _cached_file_sha256.cache_clear()


def media_content_fingerprint(image_paths: tuple[str, ...]) -> bytes:
    if not image_paths:
        raise ValueError("Offline KD media fingerprint requires at least one image path.")
    hasher = hashlib.sha256()
    for raw_path in image_paths:
        path = Path(str(raw_path)).resolve()
        stat = path.stat()
        stat_identity = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        digest = _cached_file_sha256(str(path), stat_identity)
        hasher.update(len(digest).to_bytes(4, "big"))
        hasher.update(digest)
    return hasher.digest()


@dataclass(frozen=True, slots=True)
class OfflineKDArtifactReference:
    artifact_id: str
    shard: str
    row: int

    @classmethod
    def from_mapping(cls, value: Any) -> "OfflineKDArtifactReference":
        payload = _exact_mapping(
            value,
            role="distillation_ref",
            keys={"artifact_id", "shard", "row"},
        )
        shard = str(payload["shard"]).strip()
        row = payload["row"]
        if not shard or Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError("distillation_ref.shard must be a safe relative path.")
        if type(row) is not int or row < 0:
            raise ValueError("distillation_ref.row must be a non-negative integer.")
        return cls(
            artifact_id=_digest(payload["artifact_id"], role="distillation_ref.artifact_id"),
            shard=shard,
            row=row,
        )


@dataclass(frozen=True, slots=True)
class ShaftOfflineKDInputContract:
    """Canonical preprocessing controls shared by artifact producer and student."""

    max_length: int | None
    add_eos_token: bool
    min_pixels: int | None
    max_pixels: int | None
    media_snapshot_id: str | None

    def __post_init__(self) -> None:
        for field_name in ("max_length", "min_pixels", "max_pixels"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"Offline KD input contract {field_name} must be > 0 or null.")
        if type(self.add_eos_token) is not bool:
            raise TypeError("Offline KD input contract add_eos_token must be a boolean.")
        snapshot = (
            None
            if self.media_snapshot_id is None
            else str(self.media_snapshot_id).strip() or None
        )
        object.__setattr__(self, "media_snapshot_id", snapshot)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": INPUT_CONTRACT_VERSION,
            "max_length": self.max_length,
            "add_eos_token": self.add_eos_token,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "media_snapshot_id": self.media_snapshot_id,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "ShaftOfflineKDInputContract":
        payload = _exact_mapping(
            value,
            role="offline KD input contract",
            keys={
                "version",
                "max_length",
                "add_eos_token",
                "min_pixels",
                "max_pixels",
                "media_snapshot_id",
            },
        )
        if payload["version"] != INPUT_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported Offline KD input contract version {payload['version']!r}."
            )
        return cls(
            max_length=payload["max_length"],
            add_eos_token=payload["add_eos_token"],
            min_pixels=payload["min_pixels"],
            max_pixels=payload["max_pixels"],
            media_snapshot_id=payload["media_snapshot_id"],
        )


def build_offline_kd_input_contract(config: Any) -> ShaftOfflineKDInputContract:
    data = config.data
    return ShaftOfflineKDInputContract(
        max_length=data.max_length,
        add_eos_token=bool(data.add_eos_token),
        min_pixels=data.min_pixels,
        max_pixels=data.max_pixels,
        media_snapshot_id=data.media_snapshot_id,
    )


@dataclass(frozen=True, slots=True)
class _OfflineKDShardIndex:
    path: Path
    row_offsets: torch.Tensor
    input_row_offsets: torch.Tensor
    media_sha256: torch.Tensor


class OfflineKDArtifactStore:
    """Fail-closed reader for versioned, row-indexed safetensors distributions."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        student_input_abi: ShaftOPDInputABI,
        student_input_contract: ShaftOfflineKDInputContract,
        max_cached_shards: int = 2,
    ):
        if type(max_cached_shards) is not int or max_cached_shards <= 0:
            raise ValueError("Offline KD max_cached_shards must be a positive integer.")
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.manifest_fingerprint = hashlib.sha256(
            json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest = _exact_mapping(
            raw,
            role="offline KD manifest",
            keys={
                "version",
                "artifact_id",
                "teacher",
                "input_abi",
                "input_contract",
                "distribution",
                "build",
                "shards",
            },
        )
        if manifest["version"] != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported offline KD artifact version {manifest['version']!r}.")
        self.artifact_id = _digest(manifest["artifact_id"], role="manifest.artifact_id")
        teacher = _exact_mapping(
            manifest["teacher"], role="manifest.teacher", keys={"model", "checkpoint_fingerprint"}
        )
        if not str(teacher["model"]).strip():
            raise ValueError("manifest.teacher.model must not be empty.")
        _digest(teacher["checkpoint_fingerprint"], role="manifest.teacher.checkpoint_fingerprint")
        producer_abi = ShaftOPDInputABI.from_mapping(manifest["input_abi"])
        self.input_abi_fingerprint = validate_opd_input_abi_compatibility(
            student=student_input_abi,
            teacher=producer_abi,
        )
        producer_input_contract = ShaftOfflineKDInputContract.from_mapping(
            manifest["input_contract"]
        )
        if producer_input_contract != student_input_contract:
            raise ValueError(
                "Offline KD producer/student input contract differs: "
                f"producer={producer_input_contract.to_dict()!r} "
                f"student={student_input_contract.to_dict()!r}."
            )
        self.input_contract_fingerprint = producer_input_contract.fingerprint
        distribution = _exact_mapping(
            manifest["distribution"],
            role="manifest.distribution",
            keys={"mode", "temperature", "top_k", "vocab_size"},
        )
        self.mode = str(distribution["mode"]).strip().lower()
        self.temperature = (
            None
            if distribution["temperature"] is None
            else float(distribution["temperature"])
        )
        self.top_k = (
            None if distribution["top_k"] is None else int(distribution["top_k"])
        )
        self.vocab_size = int(distribution["vocab_size"])
        if self.mode not in {"dense_logits", "topk_tail"}:
            raise ValueError(f"Unsupported manifest distribution mode {self.mode!r}.")
        if self.mode == "dense_logits" and self.temperature is not None:
            raise ValueError("Dense manifest distribution must not bind a temperature.")
        if self.mode == "topk_tail" and (
            self.temperature is None
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("Top-k manifest distribution temperature must be > 0.")
        if self.vocab_size <= 0:
            raise ValueError("manifest.distribution.vocab_size must be > 0.")
        if self.vocab_size != student_input_abi.logits_vocab_size:
            raise ValueError("Offline KD artifact vocabulary differs from the student logits ABI.")
        if (self.mode == "topk_tail") != (self.top_k is not None):
            raise ValueError("manifest distribution top_k does not match its mode.")
        if self.top_k is not None and not 0 < self.top_k <= self.vocab_size:
            raise ValueError("manifest distribution top_k must be in [1, vocab_size].")
        build = _exact_mapping(
            manifest["build"],
            role="manifest.build",
            keys={"source_fingerprint", "denylist_fingerprint"},
        )
        self.source_fingerprint = _digest(
            build["source_fingerprint"], role="manifest.build.source_fingerprint"
        )
        self.denylist_fingerprint = _digest(
            build["denylist_fingerprint"], role="manifest.build.denylist_fingerprint"
        )
        expected_artifact_id = offline_kd_artifact_identity(
            teacher=teacher,
            input_abi=manifest["input_abi"],
            input_contract=manifest["input_contract"],
            distribution=distribution,
            build=build,
        )
        if self.artifact_id != expected_artifact_id:
            raise ValueError(
                "Offline KD manifest artifact_id does not match its canonical semantic identity."
            )
        if not isinstance(manifest["shards"], Mapping) or not manifest["shards"]:
            raise ValueError("manifest.shards must be a non-empty mapping.")
        self._shards = {
            str(name): _digest(digest, role=f"manifest.shards.{name}")
            for name, digest in manifest["shards"].items()
        }
        for name in self._shards:
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("manifest shard names must be safe relative paths.")
        self._max_cached_shards = max_cached_shards
        self._index_cache: OrderedDict[str, _OfflineKDShardIndex] = OrderedDict()
        self._verified_shards: set[str] = set()

    def validate_objective(self, config: Any) -> None:
        mode = str(config.mode).strip().lower()
        if mode != self.mode:
            raise ValueError(
                "Offline KD artifact distribution mode differs from runtime objective: "
                f"artifact={self.mode!r} runtime={mode!r}."
            )
        if self.mode == "topk_tail":
            actual = (
                float(config.temperature),
                None if config.top_k is None else int(config.top_k),
            )
            expected = (self.temperature, self.top_k)
            if actual != expected:
                raise ValueError(
                    "Offline KD artifact top-k projection differs from runtime objective: "
                    f"artifact={expected!r} runtime={actual!r}."
                )

    @staticmethod
    def _require_slice(
        handle: Any,
        name: str,
        *,
        dtype: str,
        shape: tuple[int, ...],
    ) -> Any:
        tensor_slice = handle.get_slice(name)
        actual_shape = tuple(int(value) for value in tensor_slice.get_shape())
        actual_dtype = str(tensor_slice.get_dtype())
        if actual_dtype != dtype or actual_shape != shape:
            raise ValueError(
                f"Offline KD tensor {name!r} has dtype/shape "
                f"{actual_dtype}/{actual_shape}, expected {dtype}/{shape}."
            )
        return tensor_slice

    def _load_shard_index(self, name: str) -> _OfflineKDShardIndex:
        if name not in self._shards:
            raise ValueError(f"Offline KD reference names undeclared shard {name!r}.")
        cached = self._index_cache.get(name)
        if cached is not None:
            self._index_cache.move_to_end(name)
            return cached
        path = (self.manifest_path.parent / name).resolve()
        if not path.is_relative_to(self.manifest_path.parent):
            raise ValueError("Offline KD shard resolves outside the artifact directory.")
        if name not in self._verified_shards:
            if file_sha256(path) != self._shards[name]:
                raise ValueError(f"Offline KD shard checksum mismatch for {name!r}.")
            self._verified_shards.add(name)
        required = {
            "row_offsets",
            "completion_token_ids",
            "input_row_offsets",
            "input_token_ids",
            "media_sha256",
        }
        required |= {"dense_logits"} if self.mode == "dense_logits" else {
            "topk_token_ids", "topk_log_probs"
        }
        if self.mode == "topk_tail" and self.top_k != self.vocab_size:
            required.add("tail_log_probs")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != required:
                raise ValueError(
                    f"Offline KD shard tensor schema differs for {name!r}: "
                    f"missing={sorted(required - keys)} unknown={sorted(keys - required)}."
                )
            row_offsets_slice = handle.get_slice("row_offsets")
            row_shape = tuple(int(value) for value in row_offsets_slice.get_shape())
            if str(row_offsets_slice.get_dtype()) != "I64" or len(row_shape) != 1 or row_shape[0] < 2:
                raise ValueError("Offline KD row_offsets must be a 1-D int64 tensor.")
            offsets = row_offsets_slice[:]
            if int(offsets[0]) != 0 or bool((offsets[1:] < offsets[:-1]).any().item()):
                raise ValueError("Offline KD row_offsets must start at zero and be monotonic.")
            count = int(offsets[-1])
            rows = int(offsets.numel()) - 1
            input_offsets = self._require_slice(
                handle,
                "input_row_offsets",
                dtype="I64",
                shape=(rows + 1,),
            )[:]
            if (
                int(input_offsets[0]) != 0
                or bool((input_offsets[1:] < input_offsets[:-1]).any().item())
            ):
                raise ValueError(
                    "Offline KD input_row_offsets must be monotonic int64 and align with rows."
                )
            self._require_slice(
                handle,
                "completion_token_ids",
                dtype="I64",
                shape=(count,),
            )
            self._require_slice(
                handle,
                "input_token_ids",
                dtype="I64",
                shape=(int(input_offsets[-1]),),
            )
            media_sha256 = self._require_slice(
                handle,
                "media_sha256",
                dtype="U8",
                shape=(rows, 32),
            )[:]
            if self.mode == "dense_logits":
                dense = handle.get_slice("dense_logits")
                if str(dense.get_dtype()) not in {"F16", "BF16", "F32"} or tuple(
                    int(value) for value in dense.get_shape()
                ) != (count, self.vocab_size):
                    raise ValueError(
                        "Offline KD dense_logits must be floating point with shape "
                        "[positions, vocabulary]."
                    )
            else:
                assert self.top_k is not None
                self._require_slice(
                    handle,
                    "topk_token_ids",
                    dtype="I64",
                    shape=(count, self.top_k),
                )
                topk_log_probs = handle.get_slice("topk_log_probs")
                if str(topk_log_probs.get_dtype()) != "F32" or tuple(
                    int(value) for value in topk_log_probs.get_shape()
                ) != (count, self.top_k):
                    raise ValueError(
                        "Offline KD topk_log_probs must be float32 with shape "
                        "[positions, top_k]."
                    )
                if self.top_k != self.vocab_size:
                    tail = handle.get_slice("tail_log_probs")
                    if str(tail.get_dtype()) != "F32" or tuple(
                        int(value) for value in tail.get_shape()
                    ) != (count,):
                        raise ValueError(
                            "Offline KD tail_log_probs must be float32 with shape "
                            "[positions]."
                        )
        index = _OfflineKDShardIndex(
            path=path,
            row_offsets=offsets,
            input_row_offsets=input_offsets,
            media_sha256=media_sha256,
        )
        self._index_cache[name] = index
        self._index_cache.move_to_end(name)
        while len(self._index_cache) > self._max_cached_shards:
            self._index_cache.popitem(last=False)
        return index

    def get(
        self,
        reference: OfflineKDArtifactReference,
        *,
        completion_token_ids: torch.Tensor,
        input_token_ids: torch.Tensor,
        media_sha256: bytes,
    ) -> TeacherDistribution:
        if reference.artifact_id != self.artifact_id:
            raise ValueError("Offline KD record artifact_id differs from the manifest.")
        index = self._load_shard_index(reference.shard)
        offsets = index.row_offsets
        if reference.row + 1 >= int(offsets.numel()):
            raise IndexError(f"Offline KD row {reference.row} is outside shard bounds.")
        start, end = int(offsets[reference.row]), int(offsets[reference.row + 1])
        input_offsets = index.input_row_offsets
        input_start = int(input_offsets[reference.row])
        input_end = int(input_offsets[reference.row + 1])
        with safe_open(index.path, framework="pt", device="cpu") as handle:
            stored_ids = handle.get_slice("completion_token_ids")[start:end]
            stored_input_ids = handle.get_slice("input_token_ids")[input_start:input_end]
            if self.mode == "dense_logits":
                dense_logits = handle.get_slice("dense_logits")[start:end]
                topk_token_ids = None
                topk_log_probs = None
                tail_log_probs = None
            else:
                dense_logits = None
                topk_token_ids = handle.get_slice("topk_token_ids")[start:end]
                topk_log_probs = handle.get_slice("topk_log_probs")[start:end].float()
                tail_log_probs = (
                    None
                    if self.top_k == self.vocab_size
                    else handle.get_slice("tail_log_probs")[start:end].float()
                )
        observed_ids = completion_token_ids.detach().to(device="cpu", dtype=torch.long)
        if not torch.equal(stored_ids, observed_ids):
            raise ValueError(
                "Offline KD completion token alignment changed; tokenizer, template, target, "
                "truncation, or EOS policy no longer matches the artifact."
            )
        observed_input_ids = input_token_ids.detach().to(device="cpu", dtype=torch.long)
        if not torch.equal(stored_input_ids, observed_input_ids):
            raise ValueError(
                "Offline KD full input token alignment changed; prompt, template, media token "
                "expansion, truncation, or target no longer matches the artifact."
            )
        if len(media_sha256) != 32:
            raise ValueError("Offline KD observed media fingerprint must be SHA-256 bytes.")
        observed_media = torch.tensor(list(media_sha256), dtype=torch.uint8)
        if not torch.equal(index.media_sha256[reference.row], observed_media):
            raise ValueError(
                "Offline KD media content changed; teacher distribution inputs no longer match."
            )
        if self.mode == "dense_logits":
            assert dense_logits is not None
            return TeacherDistribution.from_dense_logits(dense_logits)
        assert topk_token_ids is not None and topk_log_probs is not None
        return TeacherDistribution(
            kind="topk_tail",
            vocab_size=self.vocab_size,
            topk_token_ids=topk_token_ids,
            topk_log_probs=topk_log_probs,
            tail_log_probs=tail_log_probs,
            temperature=float(self.temperature or 0.0),
        )
