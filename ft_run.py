"""
SSL-patched Freqtrade runner ? bypass corporate proxy SSL inspection.
Usage: python ft_run.py download-data --config config.json --pairs "BTC/USDT:USDT" --timeframes 5m 15m --timerange 20260101-
       python ft_run.py backtesting --config config.json --strategy ComboG_OKX --timerange 20260101-20260521
       python ft_run.py trade --config config.json
"""
import ssl
import sys
import os
import warnings

PROXY = "http://khieuvm:Lamsaoday2601@fsoft-proxy:8080"

# --- Env vars (urllib, requests, aiohttp env-based proxy) ---
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[k] = PROXY
os.environ["PYTHONHTTPSVERIFY"]  = "0"
os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# --- ssl module: unverified context as default ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- aiohttp: disable SSL verify on connector ---
import aiohttp

_orig_connector_init = aiohttp.TCPConnector.__init__
def _patched_connector_init(self, *args, **kwargs):
    kwargs["ssl"] = False
    _orig_connector_init(self, *args, **kwargs)
aiohttp.TCPConnector.__init__ = _patched_connector_init

# --- ccxt async: set httpsProxy (ccxt v4.x standard) ---
try:
    import ccxt.async_support as _ca
    _ca.Exchange.httpsProxy = PROXY
except Exception:
    pass
try:
    import ccxt.pro as _cp
    _cp.Exchange.httpsProxy = PROXY
except Exception:
    pass

# --- Suppress warnings ---
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

# --- Run freqtrade ---
if __name__ == '__main__':
    from freqtrade.main import main
    sys.argv = ["freqtrade"] + sys.argv[1:]
    main()
