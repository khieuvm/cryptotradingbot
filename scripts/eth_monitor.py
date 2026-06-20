"""Unified ETH Market Monitor — S/R + Market Context + News Sentiment.

Runs continuously. Checks:
- Price vs S/R levels: every 10s (OKX ticker)
- Market context (funding, OI, taker, L/S, liquidations): every 5min
- News sentiment (CryptoPanic/RSS + FinBERT): every 5min
- Fear & Greed index: every 24h

Only sends Telegram alerts when something notable/extreme happens.

Run:
  python scripts/eth_monitor.py              # loop
  python scripts/eth_monitor.py --once       # one cycle then exit
  python scripts/eth_monitor.py --verbose    # print all details
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INST_ID = "ETH-USDT-SWAP"

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _ROOT / "config"


def _load_tg_config() -> tuple[str, str]:
    """Load Telegram token/chat from config YAML (base + dryrun overlay)."""
    base = _CONFIG_DIR / "base.yaml"
    overlay = _CONFIG_DIR / "env" / "dryrun.yaml"
    cfg: dict = {}
    for p in [base, overlay]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            tg = data.get("telegram", {})
            if tg:
                cfg.update(tg)
    return cfg.get("token", ""), str(cfg.get("chat_id", ""))


TG_TOKEN, TG_CHAT = _load_tg_config()

TIMEFRAMES = ["5m", "15m", "1H"]
SCAN_INTERVAL = 10
CONTEXT_INTERVAL = 300
NEWS_INTERVAL = 300
FNG_INTERVAL = 86400

TF_CONFIG = {
    "5m":  {"swing_window": 5, "swing_lookback": 150, "cluster_pct": 0.003, "n_levels": 6},
    "15m": {"swing_window": 5, "swing_lookback": 150, "cluster_pct": 0.003, "n_levels": 6},
    "1H":  {"swing_window": 5, "swing_lookback": 120, "cluster_pct": 0.005, "n_levels": 6},
}

TOUCH_PCT = 0.0015
BREAK_PCT = 0.0005
DEBOUNCE_H = 4

STATE_FILE = Path(__file__).parent / "monitor_state.json"
HEADERS = {"User-Agent": "freqtrade/eth_monitor"}

ETH_KEYWORDS = ["ethereum", "eth", "eip-", "vitalik", "staking", "layer 2",
                "l2", "dencun", "pectra", "etf", "defi", "ether"]

RSS_FEEDS = [
    "https://cointelegraph.com/rss/tag/ethereum",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://blockworks.co/feed",
]

# FinBERT lazy-loaded
_finbert = None


# ══════════════════════════════════════════════════════════════════════════════
# OKX API
# ══════════════════════════════════════════════════════════════════════════════
def _okx_get(path: str, params: dict | None = None) -> dict:
    url = f"https://www.okx.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
def send_tg(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"[TG] {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PRICE & CANDLES
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ticker() -> float:
    raw = _okx_get("/api/v5/market/ticker", {"instId": INST_ID})
    return float(raw["data"][0]["last"])


def fetch_candles(timeframe: str, limit: int = 200) -> pd.DataFrame:
    raw = _okx_get("/api/v5/market/candles",
                   {"instId": INST_ID, "bar": timeframe, "limit": str(limit)})
    rows = raw.get("data", [])
    if not rows:
        raise RuntimeError(f"No candle data for {timeframe}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                      "vol", "volCcy", "volOI", "confirm"])
    df["date"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["vol"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def fetch_daily_candle() -> dict | None:
    try:
        raw = _okx_get("/api/v5/market/candles",
                       {"instId": INST_ID, "bar": "1D", "limit": "3"})
        rows = raw.get("data", [])
        if len(rows) < 2:
            return None
        y = rows[1]
        return {"high": float(y[2]), "low": float(y[3]), "close": float(y[4])}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MARKET CONTEXT
# ══════════════════════════════════════════════════════════════════════════════
def fetch_market_context() -> dict:
    ctx: dict = {}
    try:
        d = _okx_get("/api/v5/public/funding-rate", {"instId": INST_ID})["data"][0]
        ctx["funding"] = float(d["fundingRate"])
    except Exception:
        ctx["funding"] = None

    try:
        d = _okx_get("/api/v5/public/open-interest",
                     {"instType": "SWAP", "instId": INST_ID})["data"][0]
        ctx["oi_usd"] = float(d.get("oiUsd", 0))
    except Exception:
        ctx["oi_usd"] = None

    try:
        rows = _okx_get("/api/v5/rubik/stat/contracts/open-interest-history",
                        {"instId": INST_ID, "period": "15m", "limit": "5"})["data"]
        if len(rows) >= 2:
            ctx["oi_delta"] = (float(rows[0][3]) - float(rows[1][3])) / float(rows[1][3]) * 100
        else:
            ctx["oi_delta"] = None
    except Exception:
        ctx["oi_delta"] = None

    try:
        rows = _okx_get("/api/v5/rubik/stat/taker-volume-contract",
                        {"instId": INST_ID, "period": "15m", "limit": "1"})["data"]
        if rows:
            buy, sell = float(rows[0][1]), float(rows[0][2])
            ctx["taker_buy_pct"] = buy / (buy + sell) * 100 if (buy + sell) else 50
        else:
            ctx["taker_buy_pct"] = None
    except Exception:
        ctx["taker_buy_pct"] = None

    try:
        rows = _okx_get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
                        {"instId": INST_ID, "period": "15m", "limit": "1"})["data"]
        ctx["ls_ratio"] = float(rows[0][1]) if rows else None
    except Exception:
        ctx["ls_ratio"] = None

    try:
        rows = _okx_get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
                        {"instId": INST_ID, "period": "15m", "limit": "1"})["data"]
        ctx["ls_top"] = float(rows[0][1]) if rows else None
    except Exception:
        ctx["ls_top"] = None

    try:
        raw = _okx_get("/api/v5/public/liquidation-orders",
                       {"instType": "SWAP", "uly": "ETH-USDT", "state": "filled", "limit": "100"})
        details = []
        for item in raw.get("data", []):
            details.extend(item.get("details", []))
        now_ms = time.time() * 1000
        recent = [d for d in details if now_ms - float(d["ts"]) < 3600_000]
        ctx["liq_long"] = sum(float(d["sz"]) for d in recent if d["posSide"] == "long")
        ctx["liq_short"] = sum(float(d["sz"]) for d in recent if d["posSide"] == "short")
    except Exception:
        ctx["liq_long"] = None
        ctx["liq_short"] = None

    return ctx


def context_alerts(ctx: dict) -> list[str]:
    alerts = []
    fr = ctx.get("funding")
    if fr is not None and abs(fr) > 0.0005:
        side = "LONG dong" if fr > 0 else "SHORT dong"
        alerts.append(f"Funding {fr*100:+.4f}% ({side})")

    bp = ctx.get("taker_buy_pct")
    if bp is not None and (bp > 60 or bp < 40):
        bias = "BUY ap dao" if bp > 60 else "SELL ap dao"
        alerts.append(f"Taker {bp:.0f}% buy ({bias})")

    oi_d = ctx.get("oi_delta")
    if oi_d is not None and abs(oi_d) > 3:
        alerts.append(f"OI {oi_d:+.1f}% (15m)")

    ls = ctx.get("ls_ratio")
    lst = ctx.get("ls_top")
    if ls is not None and lst is not None and abs(ls - lst) > 0.5:
        alerts.append(f"L/S crowd {ls:.2f} vs smart {lst:.2f}")

    ll = ctx.get("liq_long")
    sl = ctx.get("liq_short")
    if ll is not None and sl is not None and (ll > 500 or sl > 500):
        alerts.append(f"Liq 1h: L{ll:.0f}/S{sl:.0f} ETH")

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
# S/R COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_swing_sr(df: pd.DataFrame, tf: str) -> list[dict]:
    cfg = TF_CONFIG[tf]
    lookback = cfg["swing_lookback"]
    w = cfg["swing_window"]
    cluster_pct = cfg["cluster_pct"]
    n_levels = cfg["n_levels"]

    df = df.tail(lookback).reset_index(drop=True)
    n = len(df)

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(w, n - w):
        window_h = df["high"].iloc[i - w: i + w + 1]
        if df["high"].iloc[i] == window_h.max():
            swing_highs.append(float(df["high"].iloc[i]))
        window_l = df["low"].iloc[i - w: i + w + 1]
        if df["low"].iloc[i] == window_l.min():
            swing_lows.append(float(df["low"].iloc[i]))

    if not swing_highs and not swing_lows:
        return []

    current_price = float(df["close"].iloc[-1])

    def cluster_levels(prices: list[float], origin: str) -> list[dict]:
        prices.sort()
        clusters: list[list[float]] = []
        for p in prices:
            placed = False
            for cluster in clusters:
                ref = np.median(cluster)
                if abs(p - ref) / ref < cluster_pct:
                    cluster.append(p)
                    placed = True
                    break
            if not placed:
                clusters.append([p])
        result = []
        for cluster in clusters:
            level = float(np.median(cluster))
            strength = len(cluster)
            typ = "resistance" if origin == "high" else "support"
            result.append({"price": level, "type": typ, "strength": strength, "method": "swing"})
        return result

    r_levels = cluster_levels(swing_highs, "high")
    s_levels = cluster_levels(swing_lows, "low")

    half = n_levels // 2
    above = sorted([l for l in r_levels if l["price"] > current_price],
                   key=lambda x: x["price"])[:half]
    below = sorted([l for l in s_levels if l["price"] <= current_price],
                   key=lambda x: x["price"], reverse=True)[:half]
    return above + below


def compute_daily_pivots(current_price: float) -> list[dict]:
    daily = fetch_daily_candle()
    if daily is None:
        return []
    h, l, c = daily["high"], daily["low"], daily["close"]
    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    r2 = pp + (h - l)
    s1 = 2 * pp - h
    s2 = pp - (h - l)
    return [
        {"price": pp, "type": "resistance" if pp > current_price else "support",
         "name": "PP", "method": "pivot"},
        {"price": r1, "type": "resistance", "name": "R1", "method": "pivot"},
        {"price": r2, "type": "resistance", "name": "R2", "method": "pivot"},
        {"price": s1, "type": "support", "name": "S1", "method": "pivot"},
        {"price": s2, "type": "support", "name": "S2", "method": "pivot"},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# S/R EVENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_events(levels: list[dict], price: float, prev_price: float,
                  pair_state: dict) -> list[dict]:
    events = []
    now = datetime.now(timezone.utc)

    for level in levels:
        lp = level["price"]
        sk = f"{lp:.2f}"

        last = pair_state.get(sk)
        if last:
            last_ts = datetime.fromisoformat(last["ts"])
            if (now - last_ts).total_seconds() < DEBOUNCE_H * 3600:
                continue

        if prev_price < lp and price > lp * (1 + BREAK_PCT):
            events.append({**level, "event": "break_up", "close": price})
            pair_state[sk] = {"event": "break_up", "ts": now.isoformat()}
            continue
        if prev_price > lp and price < lp * (1 - BREAK_PCT):
            events.append({**level, "event": "break_down", "close": price})
            pair_state[sk] = {"event": "break_down", "ts": now.isoformat()}
            continue
        if abs(price - lp) / lp < TOUCH_PCT:
            events.append({**level, "event": "touch", "close": price})
            pair_state[sk] = {"event": "touch", "ts": now.isoformat()}

    return events


class LevelCache:
    def __init__(self):
        self._levels: dict[str, list[dict]] = {}
        self._last_candle_ts: dict[str, float] = {}
        self._pivot_levels: list[dict] = []
        self._pivot_date: str = ""

    def get_levels(self, tf: str, price: float) -> list[dict]:
        now = time.time()
        tf_seconds = {"5m": 300, "15m": 900, "1H": 3600}[tf]
        last_ts = self._last_candle_ts.get(tf, 0)

        if now - last_ts > tf_seconds:
            try:
                df = fetch_candles(tf)
                self._levels[tf] = compute_swing_sr(df, tf)
                self._last_candle_ts[tf] = now
            except Exception:
                pass

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._pivot_date:
            self._pivot_levels = compute_daily_pivots(price)
            self._pivot_date = today

        return self._levels.get(tf, []) + self._pivot_levels


# ══════════════════════════════════════════════════════════════════════════════
# NEWS SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
def load_finbert():
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        print("  [FinBERT] Loading model...")
        _finbert = pipeline("text-classification", model="ProsusAI/finbert",
                           top_k=None, truncation=True, max_length=512)
    return _finbert


def classify_headline(title: str) -> dict:
    pipe = load_finbert()
    results = pipe(title)[0]
    best = max(results, key=lambda x: x["score"])
    return {"sentiment": best["label"].upper(), "confidence": best["score"]}


def fetch_rss(limit: int = 20) -> list[dict]:
    all_items = []
    for feed_url in RSS_FEEDS:
        try:
            req = urllib.request.Request(feed_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            for item in items[:30]:
                title = (item.findtext("title") or
                         item.findtext("atom:title", namespaces=ns) or "")
                source = feed_url.split("/")[2].replace("www.", "")
                if any(kw in title.lower() for kw in ETH_KEYWORDS):
                    all_items.append({
                        "title": title.strip(),
                        "source": source,
                        "crowd_pos": 0, "crowd_neg": 0,
                    })
        except Exception:
            pass
        time.sleep(0.2)
    return all_items[:limit]


def fetch_news(limit: int = 20) -> list[dict]:
    return fetch_rss(limit)


def fetch_fear_greed() -> dict | None:
    try:
        url = "https://api.alternative.me/fng/?limit=2&format=json"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        items = data.get("data", [])
        if not items:
            return None
        today = items[0]
        yesterday = items[1] if len(items) > 1 else items[0]
        return {
            "value": int(today["value"]),
            "label": today["value_classification"],
            "yesterday": int(yesterday["value"]),
        }
    except Exception as e:
        print(f"  [FnG] {e}")
        return None


def dedup_news(headlines: list[dict], seen: list[str]) -> list[dict]:
    new = []
    for h in headlines:
        hsh = hashlib.md5(h["title"].encode()).hexdigest()[:12]
        if hsh not in seen:
            seen.append(hsh)
            new.append(h)
    # Keep seen list bounded
    while len(seen) > 200:
        seen.pop(0)
    return new


def classify_batch(headlines: list[dict]) -> list[dict]:
    results = []
    for h in headlines:
        # Use crowd votes if strong signal
        if h["crowd_neg"] > 0 and h["crowd_neg"] > h["crowd_pos"] * 2:
            results.append({**h, "sentiment": "NEGATIVE", "confidence": 0.90})
        elif h["crowd_pos"] > 0 and h["crowd_pos"] > h["crowd_neg"] * 2:
            results.append({**h, "sentiment": "POSITIVE", "confidence": 0.90})
        else:
            cls = classify_headline(h["title"])
            results.append({**h, **cls})
    return results


def news_alerts(results: list[dict]) -> list[dict]:
    """Return only notable (negative) news worth alerting."""
    notable = [r for r in results
               if r["sentiment"] == "NEGATIVE" and r["confidence"] > 0.75]
    return notable


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════
def build_sr_alert(events: list[dict], tf: str, ctx_alerts: list[str]) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M:%S")
    lines = []
    for ev in events:
        price = ev["price"]
        close = ev["close"]
        pct = (close - price) / price * 100

        sr = "\U0001f534 KC" if ev["type"] == "resistance" else "\U0001f7e2 HT"
        if ev["event"] == "touch":
            action = f"Cham {sr}"
        elif ev["event"] == "break_up":
            action = f"\u26a1 Pha len {sr}"
        else:
            action = f"\u26a1 Pha xuong {sr}"

        if ev["method"] == "swing":
            detail = f"${price:,.1f} | Swing {ev.get('strength','')}x | {pct:+.2f}%"
        else:
            detail = f"${price:,.1f} | {ev.get('name','')} | {pct:+.2f}%"

        msg = f"*ETH* [{tf}] \u2014 {action}\n\U0001f4cd {detail}"
        if ctx_alerts:
            msg += f"\n\u26a0\ufe0f {' | '.join(ctx_alerts)}"
        msg += f"\n\U0001f4e1 {now_str}"
        lines.append(msg)
    return "\n\n".join(lines)


def build_news_alert(notable: list[dict]) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M")
    n = len(notable)
    header = f"\U0001f6a8 *ETH News* \u2014 {n} tin tieu cuc\n"
    lines = []
    for r in notable[:8]:
        conf = int(r["confidence"] * 100)
        title = r["title"][:75]
        src = r["source"][:15]
        crowd = ""
        if r["crowd_neg"] > 0 or r["crowd_pos"] > 0:
            crowd = f" | \U0001f44e{r['crowd_neg']} \U0001f44d{r['crowd_pos']}"
        lines.append(f"\u274c `{conf}%` {title}\n    _{src}_{crowd}")
    body = "\n".join(lines)
    return f"{header}{body}\n\u26a0\ufe0f Canh than LONG\n\U0001f4e1 {now_str}"


def build_context_alert(alerts: list[str], price: float) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M")
    body = "\n  ".join(alerts)
    return f"\u26a0\ufe0f *ETH* ${price:,.2f}\n  {body}\n\U0001f4e1 {now_str}"


def build_fng_alert(fng: dict) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M")
    val = fng["value"]
    lbl = fng["label"]
    yest = fng["yesterday"]
    change = val - yest
    emoji = "\U0001f7e2" if val > 60 else "\U0001f534" if val < 40 else "\U0001f7e1"
    return (f"{emoji} *Fear & Greed: {val}* ({lbl})\n"
            f"  Yesterday: {yest} | Change: {change:+d}\n"
            f"\U0001f4e1 {now_str}")


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"sr": {}, "seen_news": [], "last_fng": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run(verbose: bool = False, once: bool = False) -> None:
    state = load_state()
    cache = LevelCache()
    prev_price = fetch_ticker()
    ctx: dict = {}
    last_ctx = 0
    last_news = 0
    last_fng = state.get("last_fng", 0)

    if verbose:
        print(f"[START] ETH ${prev_price:.2f}")

    # Startup: Fear & Greed
    now = time.time()
    if now - last_fng > FNG_INTERVAL:
        fng = fetch_fear_greed()
        if fng:
            if verbose:
                print(f"  [FnG] {fng['value']} ({fng['label']})")
            change = abs(fng["value"] - fng["yesterday"])
            if change >= 10:
                send_tg(build_fng_alert(fng))
            state["last_fng"] = now
            last_fng = now

    while True:
        try:
            price = fetch_ticker()
        except Exception as e:
            if verbose:
                print(f"  [WARN] ticker: {e}")
            time.sleep(SCAN_INTERVAL)
            continue

        now = time.time()

        # ── S/R (every 10s) ──────────────────────────────────────────────
        c_alerts = context_alerts(ctx)
        for tf in TIMEFRAMES:
            levels = cache.get_levels(tf, price)
            sk = f"ETH_{tf}"
            pair_state = state.setdefault("sr", {}).setdefault(sk, {})
            events = detect_events(levels, price, prev_price, pair_state)
            if events:
                if verbose:
                    for ev in events:
                        label = "KC" if ev["type"] == "resistance" else "HT"
                        print(f"  [{tf}] {ev['event']} {label} @ ${ev['price']:.1f}")
                send_tg(build_sr_alert(events, tf, c_alerts))

        # ── Market Context (every 5 min) ─────────────────────────────────
        if now - last_ctx > CONTEXT_INTERVAL:
            try:
                ctx = fetch_market_context()
                last_ctx = now
                if verbose:
                    fr = ctx.get("funding")
                    bp = ctx.get("taker_buy_pct")
                    oi_d = ctx.get("oi_delta")
                    parts = []
                    if fr is not None: parts.append(f"FR:{fr*100:+.4f}%")
                    if bp is not None: parts.append(f"Taker:{bp:.0f}%")
                    if oi_d is not None: parts.append(f"OI:{oi_d:+.1f}%")
                    print(f"  [CTX] {' | '.join(parts)}")
                alerts = context_alerts(ctx)
                if alerts:
                    send_tg(build_context_alert(alerts, price))
            except Exception as e:
                if verbose:
                    print(f"  [CTX ERROR] {e}")

        # ── News (every 5 min) ───────────────────────────────────────────
        if now - last_news > NEWS_INTERVAL:
            try:
                news = fetch_news(20)
                seen = state.setdefault("seen_news", [])
                new_headlines = dedup_news(news, seen)
                if verbose:
                    print(f"  [NEWS] {len(news)} fetched, {len(new_headlines)} new")
                if new_headlines:
                    results = classify_batch(new_headlines)
                    notable = news_alerts(results)
                    if notable:
                        if verbose:
                            for n in notable:
                                print(f"    NEGATIVE: {n['title'][:60]}")
                        send_tg(build_news_alert(notable))
                    elif verbose:
                        neg = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
                        pos = sum(1 for r in results if r["sentiment"] == "POSITIVE")
                        print(f"    Classified: +{pos} -{neg} (none notable)")
                last_news = now
            except Exception as e:
                if verbose:
                    print(f"  [NEWS ERROR] {e}")
                last_news = now

        # ── Fear & Greed (every 24h) ─────────────────────────────────────
        if now - last_fng > FNG_INTERVAL:
            fng = fetch_fear_greed()
            if fng:
                change = abs(fng["value"] - fng["yesterday"])
                if change >= 10:
                    send_tg(build_fng_alert(fng))
                state["last_fng"] = now
                last_fng = now

        # ── Verbose price display ────────────────────────────────────────
        if verbose:
            t = datetime.now(timezone.utc).strftime("%H:%M:%S")
            all_lvls = []
            for tf in TIMEFRAMES:
                all_lvls.extend(cache.get_levels(tf, price))
            nearest_above = min((l for l in all_lvls if l["price"] > price),
                               key=lambda x: x["price"], default=None)
            nearest_below = max((l for l in all_lvls if l["price"] <= price),
                               key=lambda x: x["price"], default=None)
            print(f"  [{t}] ${price:.2f}", end="")
            if nearest_above:
                d = (nearest_above["price"] - price) / price * 100
                print(f"  KC ${nearest_above['price']:.1f} ({d:+.2f}%)", end="")
            if nearest_below:
                d = (price - nearest_below["price"]) / nearest_below["price"] * 100
                print(f"  HT ${nearest_below['price']:.1f} ({d:+.2f}%)", end="")
            print()

        prev_price = price
        save_state(state)

        if once:
            break
        time.sleep(SCAN_INTERVAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ETH Market Monitor")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--verbose", action="store_true", help="Print all details")
    args = parser.parse_args()

    send_tg(
        "\U0001f680 *ETH Monitor v1*\n"
        "S/R + Context + News\n"
        "Ticker 10s | News 5min\n"
        "Chi bao khi bat thuong"
    )

    run(verbose=args.verbose, once=args.once)


if __name__ == "__main__":
    main()
