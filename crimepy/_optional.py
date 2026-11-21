"""Import optional dependencies the first time they are used."""

from __future__ import annotations

import importlib
from types import ModuleType


class optional_dependency:
    """Attribute access imports *module_name* and caches the module.

    A missing install raises ``ImportError`` naming the extra that provides it.
    """

    def __init__(self, module_name: str, extra: str, feature: str) -> None:
        self._module_name = module_name
        self._extra = extra
        self._feature = feature
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._module_name)
            except ImportError as exc:
                raise ImportError(
                    f"{self._feature} requires the optional package "
                    f"{self._module_name!r}. Install it with: "
                    f"pip install 'crimepy-cdc[{self._extra}]'"
                ) from exc
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)
