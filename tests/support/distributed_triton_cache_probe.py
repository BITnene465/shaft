from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys


def _compile_gpu_probe(local_rank: int) -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_one_kernel(source, output, size: tl.constexpr):
        offsets = tl.arange(0, size)
        tl.store(output + offsets, tl.load(source + offsets) + 1.0)

    torch.cuda.set_device(local_rank)
    source = torch.arange(256, device=f"cuda:{local_rank}", dtype=torch.float32)
    output = torch.empty_like(source)
    add_one_kernel[(1,)](source, output, size=256)
    torch.cuda.synchronize(local_rank)
    if not torch.equal(output, source + 1.0):
        raise RuntimeError("Triton compile probe produced an incorrect result.")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: distributed_triton_cache_probe.py REPO_ROOT OUTPUT_DIR"
        )
    repo_root = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    cache_root = output_dir / "triton-cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.pop("TRITON_CACHE_DIR", None)
    os.environ["SHAFT_TRITON_CACHE_ROOT"] = str(cache_root)
    runpy.run_path(
        str(repo_root / "scripts" / "train.py"),
        run_name="shaft_train_environment_probe",
    )

    import torch.distributed as dist
    from triton import knobs

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    resolved = Path(os.environ["TRITON_CACHE_DIR"])
    if resolved != Path(knobs.cache.dir):
        raise RuntimeError(
            f"Triton imported a different cache directory: {knobs.cache.dir} != {resolved}"
        )
    if resolved.parent.parent != cache_root or resolved.name != f"rank-{local_rank}":
        raise RuntimeError(f"Unexpected rank-local Triton cache path: {resolved}")
    if os.environ.get("SHAFT_TRITON_COMPILE_PROBE") == "1":
        _compile_gpu_probe(local_rank)
        if not any(resolved.rglob("*.json")):
            raise RuntimeError(f"Triton did not publish compiler metadata under {resolved}")

    dist.init_process_group("gloo")
    try:
        (output_dir / f"rank-{rank}.json").write_text(
            json.dumps({"local_rank": local_rank, "cache_dir": str(resolved)}),
            encoding="utf-8",
        )
        dist.barrier()
        if rank == 0:
            payloads = [
                json.loads((output_dir / f"rank-{index}.json").read_text(encoding="utf-8"))
                for index in range(int(os.environ["WORLD_SIZE"]))
            ]
            cache_dirs = {payload["cache_dir"] for payload in payloads}
            if len(cache_dirs) != len(payloads):
                raise RuntimeError(f"Triton cache directories are not rank-isolated: {payloads}")
            print("distributed Triton cache isolation probe passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
