from abc import ABC, abstractmethod
import pandas as pd

from src.strategy_config import get_combo_config


class BaseCryptoCombo(ABC):
    """Abstract base for all crypto trading combos.

    Each combo implements signal detection logic and loads its parameters
    from config/strategy_config.yaml via the combo registry.
    """

    name: str
    timeframe: str = "15m"

    def __init__(self):
        self._cfg = get_combo_config(self.name)

    @abstractmethod
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Add combo-specific indicators to the dataframe."""
        ...

    @abstractmethod
    def detect_long(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        """Return boolean Series marking long entry signals."""
        ...

    @abstractmethod
    def detect_short(self, dataframe: pd.DataFrame, metadata: dict) -> pd.Series:
        """Return boolean Series marking short entry signals."""
        ...

    @property
    def grade(self) -> str:
        return self._cfg.get("grade", "C")

    @property
    def pairs(self) -> list[str]:
        return self._cfg.get("pairs", [])

    @property
    def startup_candle_count(self) -> int:
        return self._cfg.get("startup_candle_count", 110)

    @property
    def entry_cfg(self) -> dict:
        return self._cfg.get("entry", {})

    @property
    def exit_cfg(self) -> dict:
        return self._cfg.get("exit", {})

    @property
    def stake_cfg(self) -> dict:
        return self._cfg.get("stake", {})

    @property
    def sl_atr_mult(self) -> float:
        return self.exit_cfg.get("sl_atr_mult", 2.0)

    @property
    def tp_atr_mult(self) -> float:
        return self.exit_cfg.get("tp_atr_mult", 3.0)
