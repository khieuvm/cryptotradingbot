"""Configuration system: base YAML + environment overlay with deep merge."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _ROOT / "config"


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep merge overlay into base. Overlay values win on conflict."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class StrategyConfig:
    """Typed configuration for a single strategy."""

    name: str
    grade: str
    pairs: list[str]
    timeframe: str
    startup_candle_count: int
    entry: dict[str, Any]
    exit: dict[str, Any]
    stake: dict[str, float] = field(default_factory=dict)
    protections: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class AppConfig:
    """Application config with base + env overlay pattern.

    Usage:
        config = AppConfig(env="dryrun")
        strat_cfg = config.get_strategy_config("regime_adaptive")
        ft_config = config.get_freqtrade_config()
    """

    def __init__(self, env: str = "dryrun"):
        self.env = env
        self._base = self._load_yaml(_CONFIG_DIR / "base.yaml")
        overlay_path = _CONFIG_DIR / "env" / f"{env}.yaml"
        self._overlay = self._load_yaml(overlay_path) if overlay_path.exists() else {}
        self._merged = _deep_merge(self._base, self._overlay)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def data(self) -> dict:
        return self._merged

    # ── Strategy Access ──────────────────────────────────────────────────────

    def get_strategy_config(self, name: str) -> StrategyConfig:
        raw = self._merged.get("strategies", {}).get(name)
        if raw is None:
            raise KeyError(f"Strategy '{name}' not found in config")
        return StrategyConfig(
            name=name,
            grade=raw.get("grade", "F"),
            pairs=raw.get("pairs", self.get_pairs()),
            timeframe=raw.get("timeframe", self.get_timeframe()),
            startup_candle_count=raw.get("startup_candle_count", 100),
            entry=raw.get("entry", {}),
            exit=raw.get("exit", {}),
            stake=raw.get("stake", {}),
            protections=raw.get("protections", {}),
            enabled=raw.get("enabled", True),
        )

    def get_active_strategies(self) -> list[str]:
        return self._merged.get("active_strategies", [])

    # ── Market ────────────────────────────────────────────────────────────────

    def get_pairs(self) -> list[str]:
        return self._merged.get("market", {}).get("pairs", [])

    def get_timeframe(self) -> str:
        return self._merged.get("market", {}).get("timeframe", "15m")

    # ── Cost Model ────────────────────────────────────────────────────────────

    def get_costs(self) -> dict:
        return self._merged.get("costs", {})

    # ── Risk ──────────────────────────────────────────────────────────────────

    def get_risk_config(self) -> dict:
        return self._merged.get("risk", {})

    def get_circuit_breaker_config(self) -> dict:
        return self._merged.get("risk", {}).get("circuit_breaker", {})

    def get_signal_tracker_config(self) -> dict:
        return self._merged.get("signal_tracker", {})

    # ── Regime ────────────────────────────────────────────────────────────────

    def get_regime_config(self) -> dict:
        return self._merged.get("regime", {})

    # ── Leverage ──────────────────────────────────────────────────────────────

    def get_leverage_config(self) -> dict:
        return self._merged.get("leverage", {})

    # ── Funding ───────────────────────────────────────────────────────────────

    def get_funding_config(self) -> dict:
        return self._merged.get("funding", {})

    # ── Sessions ──────────────────────────────────────────────────────────────

    def get_sessions(self) -> dict:
        return self._merged.get("sessions", {})

    # ── Validation ────────────────────────────────────────────────────────────

    def get_validation_config(self) -> dict:
        return self._merged.get("validation", {})

    # ── Freqtrade JSON Generation ─────────────────────────────────────────────

    def get_freqtrade_config(self) -> dict:
        """Generate freqtrade-compatible JSON config dict from merged YAML."""
        m = self._merged
        risk = m.get("risk", {})
        exchange = m.get("exchange", {})
        market = m.get("market", {})
        leverage = m.get("leverage", {})

        ft_config = {
            "strategy": "CryptoEngine",
            "strategy_path": str(_ROOT / "adapters"),
            "max_open_trades": risk.get("max_open_trades", 6),
            "stake_currency": "USDT",
            "stake_amount": m.get("stake_amount", 50),
            "tradable_balance_ratio": 0.95,
            "dry_run": m.get("dry_run", True),
            "dry_run_wallet": m.get("dry_run_wallet", 500),
            "cancel_open_orders_on_exit": False,
            "trading_mode": exchange.get("trading_mode", "futures"),
            "margin_mode": exchange.get("margin_mode", "isolated"),
            "exchange": {
                "name": exchange.get("name", "okx"),
                "key": m.get("api_key", ""),
                "secret": m.get("api_secret", ""),
                "password": m.get("api_password", ""),
                "ccxt_config": {"enableRateLimit": True},
                "ccxt_sync_config": {"enableRateLimit": True},
                "pair_whitelist": market.get("pairs", []),
            },
            "entry_pricing": {"price_side": "other", "use_order_book": True, "order_book_top": 1},
            "exit_pricing": {"price_side": "other", "use_order_book": True, "order_book_top": 1},
            "pairlists": [{"method": "StaticPairList"}],
        }

        # Telegram
        telegram = m.get("telegram", {})
        if telegram.get("enabled"):
            ft_config["telegram"] = {
                "enabled": True,
                "token": telegram.get("token", ""),
                "chat_id": telegram.get("chat_id", ""),
            }

        # API server
        api = m.get("api_server", {})
        if api.get("enabled"):
            ft_config["api_server"] = {
                "enabled": True,
                "listen_ip_address": "0.0.0.0",
                "listen_port": api.get("port", 8080),
                "username": api.get("username", "admin"),
                "password": api.get("password", "admin"),
            }

        # Data directory
        ft_config["datadir"] = str(_ROOT / "data")

        # Engine env pass-through
        ft_config["engine_env"] = self.env

        return ft_config

    def write_freqtrade_config(self, output_path: Path | None = None) -> Path:
        """Write generated freqtrade config to JSON file."""
        if output_path is None:
            output_path = _ROOT / f"config_generated_{self.env}.json"
        ft_config = self.get_freqtrade_config()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ft_config, f, indent=2, ensure_ascii=False)
        logger.info(f"Freqtrade config written to {output_path}")
        return output_path

    def reload(self) -> None:
        """Reload config from disk (useful for live parameter updates)."""
        self.__init__(env=self.env)
