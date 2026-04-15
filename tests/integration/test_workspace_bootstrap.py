from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_workspace_manifest_exists():
    workspace_manifest = REPO_ROOT / "workspace.toml"
    assert workspace_manifest.exists()


def test_workspace_resolver_is_cwd_independent():
    resolver = REPO_ROOT / "scripts" / "resolve_workspace.py"
    result = subprocess.run(
        ["/usr/bin/python3.10", str(resolver)],
        cwd=str(REPO_ROOT / "dual_arm_gpu_mpc"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()) == REPO_ROOT
