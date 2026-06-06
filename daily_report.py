"""
Daily Report Script - Tổng hợp kết quả trading từ tất cả bots
Gửi report qua Telegram.

Usage:
    python daily_report.py          # Report hôm nay
    python daily_report.py yesterday # Report hôm qua
    python daily_report.py 2026-05-21  # Report ngày cụ thể
"""

import sqlite3
import os
import sys
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import urllib.request
import urllib.parse

# === CONFIG ===
BASE_DIR = Path(__file__).parent
TELEGRAM_TOKEN = "8644600176:AAEoExWngxwZSI27AGGoGLeOE-lkeidlCHk"
TELEGRAM_CHAT_ID = "6200159681"
PROXY = "http://khieuvm:Lamsaoday2601@fsoft-proxy:8080"

BOTS = {
    "btc2": {"db": "tradesv3.btc2.dryrun.sqlite", "strategy": "ComboH_OKX", "pair": "BTC"},
    "eth": {"db": "tradesv3.eth.dryrun.sqlite", "strategy": "ComboH_ETH", "pair": "ETH"},
    "sol": {"db": "tradesv3.sol.dryrun.sqlite", "strategy": "ComboH_SOL", "pair": "SOL"},
    "master": {"db": "tradesv3.master.dryrun.sqlite", "strategy": "ComboMaster_OKX", "pair": "ETH+SOL"},
}

LOG_DIR = BASE_DIR / "logs"


def get_report_date(arg=None):
    """Parse date argument."""
    if arg is None or arg == "today":
        return datetime.now().date()
    elif arg == "yesterday":
        return (datetime.now() - timedelta(days=1)).date()
    else:
        return datetime.strptime(arg, "%Y-%m-%d").date()


def query_trades(db_path, report_date):
    """Query closed trades for a specific date from SQLite DB."""
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    start = f"{report_date} 00:00:00"
    end = f"{report_date} 23:59:59"

    trades = conn.execute("""
        SELECT pair, is_short, open_date, close_date, 
               stake_amount, close_profit, close_profit_abs,
               exit_reason, enter_tag, leverage, funding_fees
        FROM trades
        WHERE close_date BETWEEN ? AND ?
          AND is_open = 0
        ORDER BY close_date
    """, (start, end)).fetchall()

    conn.close()
    return [dict(t) for t in trades]


def query_open_trades(db_path):
    """Query currently open trades."""
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    trades = conn.execute("""
        SELECT pair, is_short, open_date, stake_amount, leverage, enter_tag
        FROM trades
        WHERE is_open = 1
    """).fetchall()

    conn.close()
    return [dict(t) for t in trades]


def check_log_errors(bot_name, report_date):
    """Check log file for errors on the report date."""
    log_file = LOG_DIR / f"{bot_name}.log"
    if not log_file.exists():
        return []

    date_str = report_date.strftime("%Y-%m-%d")
    errors = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if date_str in line and ("ERROR" in line or "CRITICAL" in line):
                # Skip known non-critical errors
                if "telegram" in line.lower() or "websocket" in line.lower():
                    continue
                errors.append(line.strip()[:120])

    return errors[-5:]  # Last 5 errors max


def check_bot_status():
    """Check if bots are still running via API."""
    statuses = {}
    # Use no-proxy handler for localhost
    no_proxy = urllib.request.ProxyHandler({})
    local_opener = urllib.request.build_opener(no_proxy)

    for bot_name, info in BOTS.items():
        port = {"btc2": 8081, "eth": 8082, "sol": 8083, "master": 8084}[bot_name]
        try:
            import base64
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/ping")
            creds = base64.b64encode(b"freqtrader:Combo@OKX2026").decode()
            req.add_header("Authorization", f"Basic {creds}")
            resp = local_opener.open(req, timeout=5)
            statuses[bot_name] = "RUNNING" if resp.status == 200 else "ERROR"
        except Exception:
            statuses[bot_name] = "DOWN"
    return statuses


def build_report(report_date):
    """Build the daily report message."""
    date_str = report_date.strftime("%Y-%m-%d")
    lines = [f"📊 *Daily Report - {date_str}*\n"]

    total_profit = 0
    total_trades = 0
    total_wins = 0
    all_errors = []

    for bot_name, info in BOTS.items():
        db_path = BASE_DIR / info["db"]
        trades = query_trades(db_path, report_date)
        open_trades = query_open_trades(db_path)
        errors = check_log_errors(bot_name, report_date)

        num_trades = len(trades)
        wins = sum(1 for t in trades if t["close_profit_abs"] and t["close_profit_abs"] > 0)
        losses = num_trades - wins
        profit = sum(t["close_profit_abs"] or 0 for t in trades)
        profit_pct = sum(t["close_profit"] or 0 for t in trades) * 100
        wr = (wins / num_trades * 100) if num_trades > 0 else 0

        total_profit += profit
        total_trades += num_trades
        total_wins += wins

        # Bot section
        lines.append(f"{'─' * 20}")
        lines.append(f"*{bot_name.upper()}* ({info['strategy']} | {info['pair']})")

        if num_trades > 0:
            lines.append(f"  Trades: {num_trades} | W/L: {wins}/{losses} | WR: {wr:.0f}%")
            lines.append(f"  Profit: {profit:+.2f} USDT ({profit_pct:+.1f}%)")

            # Exit reasons breakdown
            reasons = {}
            for t in trades:
                r = t["exit_reason"] or "unknown"
                reasons[r] = reasons.get(r, 0) + 1
            reason_str = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
            lines.append(f"  Exits: {reason_str}")
        else:
            lines.append(f"  No closed trades")

        if open_trades:
            directions = []
            for t in open_trades:
                d = "S" if t["is_short"] else "L"
                directions.append(f"{t['pair'].split('/')[0]}({d})")
            lines.append(f"  Open: {', '.join(directions)}")

        if errors:
            all_errors.extend([(bot_name, e) for e in errors])

    # Summary
    total_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    lines.append(f"\n{'═' * 20}")
    lines.append(f"*TOTAL: {total_trades} trades | WR {total_wr:.0f}% | {total_profit:+.2f} USDT*")

    # Bot status
    statuses = check_bot_status()
    status_line = " | ".join(f"{k}:{'✅' if v == 'RUNNING' else '❌'}" for k, v in statuses.items())
    lines.append(f"\nStatus: {status_line}")

    # Errors
    if all_errors:
        lines.append(f"\n⚠️ *Errors ({len(all_errors)}):*")
        for bot, err in all_errors[:3]:
            lines.append(f"  [{bot}] {err[:80]}")

    return "\n".join(lines)


def send_telegram(message):
    """Send message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }).encode()

    # Setup proxy
    proxy_handler = urllib.request.ProxyHandler({
        "http": PROXY,
        "https": PROXY,
    })
    opener = urllib.request.build_opener(proxy_handler)

    try:
        req = urllib.request.Request(url, data=data)
        resp = opener.open(req, timeout=15)
        result = json.loads(resp.read())
        if result.get("ok"):
            print("✅ Report sent to Telegram")
        else:
            print(f"❌ Telegram error: {result}")
    except Exception as e:
        print(f"❌ Failed to send: {e}")
        # Print report to console as fallback
        print("\n" + message)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "today"

    # Handle --no-send flag
    no_send = "--no-send" in sys.argv

    report_date = get_report_date(arg.replace("--no-send", "").strip() or "today")
    print(f"Generating report for: {report_date}")

    report = build_report(report_date)

    if no_send:
        print(report)
    else:
        send_telegram(report)
        print(report)


if __name__ == "__main__":
    main()
