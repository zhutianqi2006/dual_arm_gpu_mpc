from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from dual_arm_gpu_mpc.common.paths import configs_dir


def parse_cli_overrides(argv: list[str]) -> tuple[str, list[str]]:
    experiment_name = "kmeans_adpan_smoke"
    overrides: list[str] = []

    for arg in argv:
        if arg.startswith("experiment="):
            experiment_name = arg.split("=", 1)[1]
            continue
        overrides.append(arg)

    return experiment_name, overrides


@contextmanager
def initialize_config_dir(config_dir: str) -> Iterator[Path]:
    yield Path(config_dir)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(data).__name__}")
    return data


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Unsupported override format: {override!r}")

    key, raw_value = override.split("=", 1)
    target = config
    parts = key.split(".")

    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child

    value: Any = raw_value
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        value = lowered == "true"
    else:
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value

    target[parts[-1]] = value


def compose(config_name: str, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = configs_dir() / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    config = _load_yaml_config(config_path)
    for override in overrides or []:
        _apply_override(config, override)
    return config


def load_experiment_config(name: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(configs_dir())):
        return compose(config_name=f"experiment/{name}", overrides=overrides)
