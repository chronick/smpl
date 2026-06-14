"""Built-in subcommand discovery.

Built-ins live as modules under ``smpl_cli.subcommands``; subcommand name ``as-wav`` maps
to module ``as_wav`` (``-`` ↔ ``_``). Discovery is filename-based (no import) so listing is
cheap and a single dispatch imports ONLY the target module — preserving lazy-import cold
start (heavy deps stay inside each module's ``run``).
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from types import ModuleType
from typing import Optional

from . import subcommands

_PKG = subcommands.__name__


def _name_to_module(name: str) -> str:
    return name.replace("-", "_")


def _module_to_name(module: str) -> str:
    return module.replace("_", "-")


def list_builtin_names() -> list[str]:
    """All built-in subcommand names, by scanning the package (no imports)."""
    names = []
    for info in pkgutil.iter_modules(subcommands.__path__):
        if info.name.startswith("_"):
            continue
        names.append(_module_to_name(info.name))
    return sorted(names)


def is_builtin(name: str) -> bool:
    mod = _name_to_module(name)
    if mod.startswith("_"):
        return False
    return importlib.util.find_spec(f"{_PKG}.{mod}") is not None


def load(name: str) -> Optional[ModuleType]:
    """Import and return the subcommand module, or None if it isn't a built-in."""
    if not is_builtin(name):
        return None
    return importlib.import_module(f"{_PKG}.{_name_to_module(name)}")
