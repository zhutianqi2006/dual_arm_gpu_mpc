from __future__ import annotations

from pathlib import Path
import glob
import site
import sys

_LEGACY_CUROBO_PATCHED = False
_LEGACY_PYBULLET_PATCHED = False


def resolve_project_root(example_file: str | Path) -> Path:
    current = Path(example_file).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Unable to resolve dual_arm_gpu_mpc project root from {example_file!r}.")


def bootstrap_example_paths(example_file: str | Path) -> Path:
    project_root = resolve_project_root(example_file)
    _add_project_venv_site_packages(project_root)
    for candidate in reversed(_python_roots_for_examples(project_root)):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    patch_legacy_curobo_load_from_config()
    patch_legacy_pybullet_load_urdf()
    return project_root


def _python_roots_for_examples(project_root: Path) -> list[Path]:
    ordered: list[Path] = []
    for candidate in (
        _resolve_workspace_dq_torch_root(project_root),
        project_root / "src",
        project_root,
    ):
        if candidate is None:
            continue
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _resolve_workspace_dq_torch_root(project_root: Path) -> Path | None:
    kernel_root = project_root.parent / "cuda_dq_kernel"
    direct_src_root = kernel_root / "src" / "python"
    if (direct_src_root / "dq_torch" / "__init__.py").exists():
        return direct_src_root

    build_matches = sorted(
        glob.glob(str(kernel_root / "build" / "lib.*" / "dq_torch" / "__init__.py"))
    )
    if build_matches:
        return Path(build_matches[-1]).parent.parent

    return None


def _add_project_venv_site_packages(project_root: Path) -> None:
    for candidate in _project_venv_site_packages(project_root):
        site.addsitedir(str(candidate))


def _project_venv_site_packages(project_root: Path) -> list[Path]:
    venv_root = project_root / ".venv"
    versioned = venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    candidates: list[Path] = []
    if versioned.exists():
        candidates.append(versioned)

    for match in sorted(glob.glob(str(venv_root / "lib" / "python*" / "site-packages"))):
        candidate = Path(match)
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)

    return candidates


def resolve_legacy_example_path(path_like: str | Path) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)

    project_root = resolve_project_root(__file__)
    candidates = [
        project_root / path,
        project_root / "model" / path,
        project_root / "assets" / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(path_like)


def patch_legacy_curobo_load_from_config() -> None:
    global _LEGACY_CUROBO_PATCHED
    if _LEGACY_CUROBO_PATCHED:
        return

    from curobo.wrap.model.robot_world import RobotWorldConfig

    from dual_arm_gpu_mpc.robotics.collision import curobo_loader

    original_load_from_config = RobotWorldConfig.load_from_config

    def _patched_load_from_config(robot_config="franka.yml", world_model=None, *args, **kwargs):
        if isinstance(robot_config, (str, Path)) and isinstance(world_model, (str, Path)):
            robot_path, world_path = curobo_loader.resolve_curobo_config_paths(str(robot_config), str(world_model))
            robot_config = curobo_loader._load_robot_config(robot_path)
            world_model = curobo_loader._load_yaml_dict(world_path)
        return original_load_from_config(robot_config, world_model, *args, **kwargs)

    RobotWorldConfig.load_from_config = _patched_load_from_config
    _LEGACY_CUROBO_PATCHED = True


def patch_legacy_pybullet_load_urdf() -> None:
    global _LEGACY_PYBULLET_PATCHED
    if _LEGACY_PYBULLET_PATCHED:
        return

    try:
        import pybullet as pyb
    except ImportError:
        return

    original_load_urdf = pyb.loadURDF

    def _patched_load_urdf(fileName, *args, **kwargs):
        if isinstance(fileName, (str, Path)):
            fileName = resolve_legacy_example_path(fileName)
        return original_load_urdf(fileName, *args, **kwargs)

    pyb.loadURDF = _patched_load_urdf
    _LEGACY_PYBULLET_PATCHED = True
