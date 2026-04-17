from __future__ import annotations

import sys
from typing import Sequence

from dual_arm_gpu_mpc.config.hydra import load_experiment_config, parse_cli_overrides
from dual_arm_gpu_mpc.workspace import BaseWorkspace


def main(argv: Sequence[str] | None = None) -> int:
    experiment_name, overrides = parse_cli_overrides(list(argv or sys.argv[1:]))
    config = load_experiment_config(experiment_name, overrides)
    workspace = BaseWorkspace.from_config(config)
    workspace.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
