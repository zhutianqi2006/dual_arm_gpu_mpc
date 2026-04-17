from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def _find_ancestor_with(start: Path, filename: str) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / filename).exists():
            return candidate
    raise AssertionError(f"Could not find {filename} above {start}")


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_ancestor_with(TEST_DIR, "pyproject.toml")


def test_importing_kmeans_submodule_does_not_eagerly_import_adpan():
    check = dedent(
        """
        import sys
        from pathlib import Path

        project_root = Path.cwd().resolve()
        sys.path[:] = [str(project_root), str(project_root / "src")] + [
            entry
            for entry in sys.path
            if entry and Path(entry).resolve() not in {project_root, project_root / "src"}
        ]

        import dual_arm_gpu_mpc.controllers.high_level.mppi.kmeans_adpan  # noqa: F401

        raise SystemExit(
            0 if "dual_arm_gpu_mpc.controllers.high_level.mppi.adpan" not in sys.modules else 1
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", check],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "Importing the kmeans submodule should not eagerly import adpan.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
