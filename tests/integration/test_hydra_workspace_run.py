from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_ancestor_with(start: Path, filename: str) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / filename).exists():
            return candidate
    raise AssertionError(f"Could not find {filename} above {start}")


def test_run_module_accepts_smoke_experiment() -> None:
    test_dir = Path(__file__).resolve().parent
    project_root = _find_ancestor_with(test_dir, "pyproject.toml")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dual_arm_gpu_mpc.run",
            "experiment=kmeans_adpan_smoke",
            "runtime.device=cpu",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
