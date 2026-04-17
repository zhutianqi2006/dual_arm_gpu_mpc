from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["DualArmMPPICore", "MPPIAdpAnModule", "MPPIKmeansAdpAnModule"]


def __getattr__(name: str) -> Any:
    if name == "DualArmMPPICore":
        from .base import DualArmMPPICore

        return DualArmMPPICore
    if name == "MPPIAdpAnModule":
        return import_module(".adpan", __name__).MPPIAdpAnModule
    if name == "MPPIKmeansAdpAnModule":
        return import_module(".kmeans_adpan", __name__).MPPIKmeansAdpAnModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
