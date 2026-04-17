from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BaseWorkspace:
    config: dict[str, Any]
    run_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseWorkspace":
        runtime = config.get("runtime", {})
        run_dir = runtime.get("run_dir")
        resolved_run_dir = Path(run_dir).expanduser() if run_dir else None
        metadata = {
            "experiment_name": config.get("experiment_name"),
            "device": runtime.get("device"),
        }
        return cls(config=config, run_dir=resolved_run_dir, metadata=metadata)

    def run(self) -> None:
        return None
