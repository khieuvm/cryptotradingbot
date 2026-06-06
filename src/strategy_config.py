"""Single source of truth: loads config/strategy_config.yaml once, provides typed access."""

import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "strategy_config.yaml"
_cache: dict | None = None


def get_config() -> dict:
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def reload_config() -> dict:
    global _cache
    _cache = None
    return get_config()


def get_combo_config(combo_name: str) -> dict:
    """Get per-combo config section."""
    return get_config().get("combos", {}).get(combo_name, {})


def get_active_combos() -> list[str]:
    """Get list of active combo names."""
    return get_config().get("active_combos", [])


def get_costs() -> dict:
    """Get cost model parameters."""
    return get_config().get("costs", {})


def get_regime_config() -> dict:
    """Get regime detection parameters."""
    return get_config().get("regime", {})


def get_signal_tracker_config() -> dict:
    """Get signal tracker / auto-disable parameters."""
    return get_config().get("signal_tracker", {})


def get_validation_config() -> dict:
    """Get validation requirements."""
    return get_config().get("validation", {})


def get_funding_config() -> dict:
    """Get funding rate parameters."""
    return get_config().get("funding", {})


def get_pairs() -> list[str]:
    """Get configured trading pairs."""
    return get_config().get("pairs", [])
