from pathlib import Path
import sys
import subprocess


def _find_ancestor_with(start: Path, filename: str) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / filename).exists():
            return candidate
    raise AssertionError(f"Could not find {filename} above {start}")


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_ancestor_with(TEST_DIR, "pyproject.toml")
WORKSPACE_ROOT = _find_ancestor_with(PROJECT_ROOT, "workspace.toml")


def test_workspace_manifest_exists():
    workspace_manifest = WORKSPACE_ROOT / "workspace.toml"
    assert workspace_manifest.exists()


def test_workspace_resolver_is_cwd_independent():
    resolver = WORKSPACE_ROOT / "scripts" / "resolve_workspace.py"
    result = subprocess.run(
        [sys.executable, str(resolver)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()) == WORKSPACE_ROOT
