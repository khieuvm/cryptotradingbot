"""Strategy registry and discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.base import BaseStrategy
    from engine.config import AppConfig

_STRATEGY_CLASSES: dict[str, type] = {}


def register_strategy(cls: type) -> type:
    """Decorator to register a strategy class in the global registry."""
    _STRATEGY_CLASSES[cls.name] = cls
    return cls


def get_strategy_class(name: str) -> type:
    if name not in _STRATEGY_CLASSES:
        _discover_strategies()
    if name not in _STRATEGY_CLASSES:
        raise KeyError(f"Strategy '{name}' not found in registry")
    return _STRATEGY_CLASSES[name]


def get_active_strategy_classes(config: "AppConfig") -> list[tuple[str, type]]:
    """Return (name, cls) pairs for all active strategies per config."""
    active = config.get_active_strategies()
    return [(name, get_strategy_class(name)) for name in active]


def _discover_strategies() -> None:
    """Import all strategy modules to trigger registration."""
    import importlib
    import pkgutil
    import strategies as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name == "base" or info.name.startswith("_"):
            continue
        importlib.import_module(f"strategies.{info.name}")
