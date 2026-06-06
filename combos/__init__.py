from combos.base import BaseCryptoCombo
from combos.regime_adaptive import RegimeAdaptiveCombo
from combos.meanrev_confluence import MeanRevConfluenceCombo
from combos.trend_composite import TrendCompositeCombo

COMBO_REGISTRY: dict[str, type[BaseCryptoCombo]] = {
    "regime_adaptive": RegimeAdaptiveCombo,
    "meanrev_confluence": MeanRevConfluenceCombo,
    "trend_composite": TrendCompositeCombo,
}

_instances: dict[str, BaseCryptoCombo] = {}


def get_combo(name: str) -> BaseCryptoCombo:
    """Get or create a combo instance by name."""
    if name not in _instances:
        cls = COMBO_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown combo: {name}. Available: {list(COMBO_REGISTRY.keys())}")
        _instances[name] = cls()
    return _instances[name]


def get_active_combos(pair: str | None = None) -> list[BaseCryptoCombo]:
    """Get all active combos, optionally filtered by pair support."""
    from src.strategy_config import get_active_combos as _get_active_names
    combos = []
    for name in _get_active_names():
        combo = get_combo(name)
        if pair is None or pair in combo.pairs:
            combos.append(combo)
    return combos
