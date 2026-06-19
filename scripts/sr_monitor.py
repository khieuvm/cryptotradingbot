"""Dynamic Support & Resistance monitor with market context.

S/R: Swing high/low + Daily Pivot Points (cached, recomputed on candle close)
Price: OKX ticker (real-time, every 10s)
Context: Funding, OI, Taker volume, L/S ratio (fetched every 5m, shown only when extreme)
Timeframes: 5m, 15m, 1h

Run:
  python scripts/sr_monitor.py            # loop every 10s
  python scripts/sr_monitor.py --once     # one cycle then exit
  python scripts/sr_monitor.py --verbose  # print all details
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
INST_ID = "ETH-USDT-SWAP"
TG_TOKEN = "8644600176:AAEoExWngxwZSI27AGGoGLeOE-lkeidlCHk"
TG_CHAT = "6200159681"

TIMEFRAMES = ["5m", "15m", "1H"]
SCAN_INTERVAL = 10
CONTEXT_INTERVAL = 300  # fetch market context every 5min

TF_CONFIG = {
    "5m":  {"swing_window": 5, "swing_lookback": 150, "cluster_pct": 0.003, "n_levels": 6},
    "15m": {"swing_window": 5, "swing_lookback": 150, "cluster_pct": 0.003, "n_levels": 6},
    "1H":  {"swing_window": 5, "swing_lookback": 120, "cluster_pct": 0.005, "n_levels": 6},
}

TOUCH_PCT = 0.0015
BREAK_PCT = 0.0005
DEBOUNCE_H = 4

STATE_FILE = Path(__file__).parent / "sr_state.json"
HEADERS = {"User-Agent": "freqtrade/sr_monitor"}

# ── OKX API ──────────────────────────────────────────────────────────────────
def _okx_get(path: str, params: dict | None = None) -> dict:
    url = f"https://www.okx.com{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown",
    }).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"[TG ERROR] {e}")


# ── Real-time ticker ────────────────────────────────────────────────────────
def fetch_ticker() -> float:
    raw = _okx_get("/api/v5/market/ticker", {"instId": INST_ID})
    return float(raw["data"][0]["last"])


# ── Candle data (for S/R computation) ────────────────────────────────────────
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


# ── Market Context (fetched periodically) ────────────────────────────────────
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

    return ctx


def context_alerts(ctx: dict) -> list[str]:
    """Only return warnings for extreme/unusual conditions."""
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
    if ll is not None and (ll > 500 or sl > 500):
        alerts.append(f"Liq 1h: L{ll:.0f}/S{sl:.0f} ETH")

    return alerts


def format_context_verbose(ctx: dict) -> str:
    lines = []
    fr = ctx.get("funding")
    if fr is not None:
        lines.append(f"  FR:{fr*100:+.4f}%")
    oi = ctx.get("oi_usd")
    oi_d = ctx.get("oi_delta")
    if oi is not None:
        d_str = f"({oi_d:+.1f}%)" if oi_d is not None else ""
        lines.append(f"  OI:${oi/1e6:.0f}M {d_str}")
    bp = ctx.get("taker_buy_pct")
    if bp is not None:
        lines.append(f"  Taker:Buy {bp:.0f}%")
    ls = ctx.get("ls_ratio")
    lst = ctx.get("ls_top")
    if ls is not None:
        t = f"/Top {lst:.2f}" if lst else ""
        lines.append(f"  L/S:{ls:.2f}{t}")
    ll = ctx.get("liq_long")
    sl = ctx.get("liq_short", 0)
    if ll is not None and (ll > 0 or sl > 0):
        lines.append(f"  Liq1h:L{ll:.0f}/S{sl:.0f}")
    return " | ".join(lines)


# ── S/R Computation ───────────────────────────────────────────────────────────
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


# ── Event Detection (ticker-based) ──────────────────────────────────────────
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


# ── State ────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# ── Telegram message ────────────────────────────────────────────────────────
def build_telegram_message(events: list[dict], tf: str, ctx_alerts: list[str]) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M:%S")
    lines = []
    for ev in events:
        price = ev["price"]
        close = ev["close"]
        pct = (close - price) / price * 100

        sr = "\U0001f534 KC" if ev["type"] == "resistance" else "\U0001f7e2 HT"
        if ev["event"] == "touch":
            action = f"Ch\u1ea1m {sr}"
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


# ── Level cache ──────────────────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────
def run_loop(verbose: bool = False, once: bool = False) -> None:
    state = load_state()
    cache = LevelCache()
    prev_price = fetch_ticker()
    ctx: dict = {}
    last_ctx_fetch = 0

    if verbose:
        print(f"[START] ETH ticker: ${prev_price:.2f}")

    while True:
        try:
            price = fetch_ticker()
        except Exception as e:
            if verbose:
                print(f"[WARN] ticker: {e}")
            time.sleep(SCAN_INTERVAL)
            continue

        now = time.time()
        if now - last_ctx_fetch > CONTEXT_INTERVAL:
            try:
                ctx = fetch_market_context()
                last_ctx_fetch = now
                if verbose:
                    print(f"\n[CTX] {format_context_verbose(ctx)}")
            except Exception:
                pass

        c_alerts = context_alerts(ctx)
        all_tg: list[str] = []

        for tf in TIMEFRAMES:
            levels = cache.get_levels(tf, price)
            sk = f"ETH_{tf}"
            pair_state = state.setdefault(sk, {})

            events = detect_events(levels, price, prev_price, pair_state)

            if events:
                if verbose:
                    for ev in events:
                        label = "KC" if ev["type"] == "resistance" else "HT"
                        print(f"  [{tf}] {ev['event']} {label} @ ${ev['price']:.1f} "
                              f"(price=${price:.2f})")
                tg = build_telegram_message(events, tf, c_alerts)
                all_tg.append(tg)

        if all_tg:
            save_state(state)
            for msg in all_tg:
                send_telegram(msg)

        if verbose:
            t = datetime.now(timezone.utc).strftime("%H:%M:%S")
            all_lvls = []
            for tf in TIMEFRAMES:
                all_lvls.extend(cache.get_levels(tf, price))
            nearest_above = None
            nearest_below = None
            for lv in all_lvls:
                lp = lv["price"]
                if lp > price:
                    if nearest_above is None or lp < nearest_above["price"]:
                        nearest_above = lv
                else:
                    if nearest_below is None or lp > nearest_below["price"]:
                        nearest_below = lv
            print(f"  [{t}] ${price:.2f}")
            if nearest_above:
                d = (nearest_above["price"] - price) / price * 100
                tag = nearest_above.get("name", f"swing {nearest_above.get('strength','')}x")
                print(f"    KC  ${nearest_above['price']:.1f}  {d:+.2f}%  [{tag}]")
            if nearest_below:
                d = (price - nearest_below["price"]) / nearest_below["price"] * 100
                tag = nearest_below.get("name", f"swing {nearest_below.get('strength','')}x")
                print(f"    HT  ${nearest_below['price']:.1f}  {d:+.2f}%  [{tag}]")

        prev_price = price

        if once:
            if verbose:
                print(f"\n[LEVELS] price=${price:.2f}")
                for tf in TIMEFRAMES:
                    levels = cache.get_levels(tf, price)
                    print(f"\n  ETH [{tf}]")
                    for lv in sorted(levels, key=lambda x: x["price"]):
                        tag = lv.get("name", f"swing {lv.get('strength', '')}x")
                        label = "KC" if lv["type"] == "resistance" else "HT"
                        dist = (price - lv["price"]) / lv["price"] * 100
                        print(f"    {label}  {lv['price']:>10.2f}  [{tag}]  {dist:+.2f}%")
            break

        time.sleep(SCAN_INTERVAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    send_telegram(
        "\U0001f680 *SR Monitor v3*\n"
        "ETH/USDT | 5m + 15m + 1h\n"
        "Ticker realtime (10s)\n"
        "Context: chi hien khi bat thuong"
    )

    run_loop(verbose=args.verbose, once=args.once)


if __name__ == "__main__":
    main()
