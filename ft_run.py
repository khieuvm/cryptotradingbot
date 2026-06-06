"""
Freqtrade runner.
Usage: python ft_run.py download-data --config config.json --pairs "BTC/USDT:USDT" --timeframes 5m 15m --timerange 20260101-
       python ft_run.py backtesting --config config.json --strategy ComboG_OKX --timerange 20260101-20260521
       python ft_run.py trade --config config.json
"""
import sys
import os
import warnings

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Run freqtrade ---
if __name__ == '__main__':
    from freqtrade.main import main
    sys.argv = ["freqtrade"] + sys.argv[1:]
    main()
