from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Unable to resolve dual_arm_gpu_mpc project root.")


@lru_cache(maxsize=1)
def workspace_root() -> Path:
    for parent in project_root().parents:
        if (parent / "workspace.toml").exists():
            return parent
    return project_root().parent


def configs_dir() -> Path:
    return project_root() / "configs"


def assets_dir() -> Path:
    return project_root() / "assets"


def legacy_model_dir() -> Path:
    return project_root() / "model"


def examples_dir() -> Path:
    return project_root() / "examples"


def _resolve_existing(path_like: str | Path, candidates: list[Path]) -> Path:
    path = Path(path_like)
    if path.is_absolute() and path.exists():
        return path

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve path for {path_like!r} from candidates: {candidates}")


def resolve_config_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return _resolve_existing(
        path_like,
        [
            project_root() / path,
            configs_dir() / path,
            examples_dir() / path,
        ],
    )


def resolve_world_config(path_like: str | Path) -> Path:
    path = Path(path_like)
    return _resolve_existing(
        path_like,
        [
            configs_dir() / "world" / path.name,
            legacy_model_dir() / "curobo_env" / path.name,
            project_root() / path,
        ],
    )


def resolve_robot_config(path_like: str | Path) -> Path:
    path = Path(path_like)
    return _resolve_existing(
        path_like,
        [
            configs_dir() / "robot" / path.name,
            legacy_model_dir() / "curobo_robot_collision" / path.name,
            project_root() / path,
        ],
    )


def resolve_asset_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return _resolve_existing(
        path_like,
        [
            assets_dir() / path,
            project_root() / path,
            legacy_model_dir() / path,
        ],
    )

