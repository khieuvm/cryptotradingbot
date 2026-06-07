"""Launch ML Scalping SOL 3m bot as a separate freqtrade instance.

Usage:
  python scripts/run_ml_3m_sol.py              # dry-run (default)
  python scripts/run_ml_3m_sol.py --live       # live trading
  python scripts/run_ml_3m_sol.py --backtest   # backtest mode
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dryrun"

    config_path = ROOT / "config" / "config_ml_3m_sol.json"

    if mode == "--backtest":
        cmd = [
            sys.executable, str(ROOT / "ft_run.py"),
            "backtesting",
            "--strategy", "CryptoEngine",
            "--config", str(config_path),
            "--timerange", "20260501-",
            "--timeframe", "3m",
        ]
    elif mode == "--live":
        cmd = [
            sys.executable, str(ROOT / "ft_run.py"),
            "trade",
            "--strategy", "CryptoEngine",
            "--config", str(config_path),
        ]
    else:
        cmd = [
            sys.executable, str(ROOT / "ft_run.py"),
            "trade",
            "--strategy", "CryptoEngine",
            "--config", str(config_path),
        ]

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
