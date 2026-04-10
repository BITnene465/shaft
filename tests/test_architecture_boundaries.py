from __future__ import annotations

import re
import unittest
from pathlib import Path


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_core_has_no_infer_package(self) -> None:
        self.assertFalse(
            Path("src/vlm_structgen/core/infer").exists(),
            "core/infer 已迁移到 runtime，不应继续存在。",
        )

    def test_core_has_no_direct_import_to_tasks_domains_runtime(self) -> None:
        forbidden_import = re.compile(
            r"^\s*(from|import)\s+vlm_structgen\.(tasks|domains|runtime)\b",
            re.MULTILINE,
        )
        core_root = Path("src/vlm_structgen/core")
        for path in core_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                forbidden_import.search(text),
                f"core 不应直接依赖 tasks/domains/runtime 具体实现: {path}",
            )


if __name__ == "__main__":
    unittest.main()

