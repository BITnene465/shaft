from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from vlm_structgen.core.utils.checkpoint import load_training_checkpoint


class _DummyScheduler:
    def load_state_dict(self, state):
        self.state = state


class CheckpointLoadingTests(unittest.TestCase):
    def test_resume_loads_rng_with_weights_only_disabled(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = _DummyScheduler()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            (checkpoint_dir / "model").mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint_dir / "model" / "state_dict.pt")
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save({"step": 1}, checkpoint_dir / "scheduler.pt")
            torch.save({"python": (), "numpy": (), "torch": torch.random.get_rng_state()}, checkpoint_dir / "rng_state.pt")
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": 12}),
                encoding="utf-8",
            )

            load_calls: list[tuple[str, bool]] = []

            real_torch_load = torch.load

            def recording_load(path, *args, **kwargs):
                load_calls.append((Path(path).name, bool(kwargs.get("weights_only"))))
                return real_torch_load(path, *args, **kwargs)

            with patch("vlm_structgen.core.utils.checkpoint.torch.load", side_effect=recording_load):
                with patch("vlm_structgen.core.utils.checkpoint.set_rng_state") as set_rng_state:
                    trainer_state = load_training_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        strict=True,
                        resume_training_state=True,
                    )

            self.assertEqual(trainer_state["global_step"], 12)
            self.assertEqual(
                load_calls,
                [
                    ("state_dict.pt", True),
                    ("optimizer.pt", False),
                    ("scheduler.pt", False),
                    ("rng_state.pt", False),
                ],
            )
            set_rng_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()
