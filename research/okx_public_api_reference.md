# OKX Public REST API — Trading-Relevant Endpoints

Researched 2026-06-19. All endpoints verified live against ETH-USDT-SWAP.
Base URL: `https://www.okx.com`
Authentication: None required for all endpoints below.

---

## Already In Use

```
GET /api/v5/market/candles
  instId=ETH-USDT-SWAP  bar=15m  limit=300
  Returns: [ts, open, high, low, close, vol_contracts, vol_ccy, vol_ccy_quote, confirm]
```

---

## 1. Funding Rate

### Current Rate
```
GET /api/v5/public/funding-rate
  Required: instId=ETH-USDT-SWAP
```
Response fields:
- `fundingRate` — current in-period rate (e.g., -0.0000166)
- `fundingTime` — next settlement timestamp (ms)
- `nextFundingRate` — predicted next rate (often empty until ~1h before)
- `nextFundingTime` — timestamp after next settlement
- `prevFundingTime` — last settlement timestamp
- `settFundingRate` — actually realized rate at last settlement
- `premium` — current mark-spot spread (e.g., -0.000497)
- `maxFundingRate` / `minFundingRate` — clamp bounds (0.0075 / -0.0075)
- `interestRate` — base interest component (0.0001)
- `impactValue` — notional used for impact mid price calculation (20000 USDT)
- `settState` — "settled" | "processing"

Trading use:
- Rate > 0.05% (5x normal) = longs paying heavily = contrarian short signal
- Rate < -0.03% = shorts paying = contrarian long signal
- `premium` direction matches funding direction; can lead rate by minutes
- 30 min pre-settlement (before 00:00/08:00/16:00 UTC): position squeezing occurs

### Historical Rates
```
GET /api/v5/public/funding-rate-history
  Required: instId=ETH-USDT-SWAP
  Optional: before, after (ms timestamps), limit (max 100, default 100)
```
Response: array of objects with `fundingTime`, `fundingRate`, `realizedRate`, `instId`, `instType`, `method`, `formulaType`

Trading use:
- Multi-period cumulative funding = carry cost for any held position
- 3-day rolling sum of realized rates as regime filter
- Funding rate trend direction predicts crowded-trade unwind risk

---

## 2. Open Interest

### Current Snapshot
```
GET /api/v5/public/open-interest
  Required: instType=SWAP
  Optional: instId=ETH-USDT-SWAP  (omit for all SWAPs)
```
Response fields:
- `oi` — contracts (e.g., 7361773.33)
- `oiCcy` — in base currency ETH (e.g., 736177.33)
- `oiUsd` — in USD (e.g., 1,243,131,129)
- `ts`

Trading use:
- OI rising + price rising = healthy trend continuation
- OI falling + price rising = weak breakout (short covering, not new longs)
- OI spike = potential squeeze setup

### Historical OI Time Series
```
GET /api/v5/rubik/stat/contracts/open-interest-history
  Required: instId=ETH-USDT-SWAP
  Optional: period (15m | 1H | 4H | 1D), limit (default/max varies)
```
Response: arrays of `[ts, oi_contracts, oi_ccy, oi_usd]`

Trading use:
- OI divergence from price on 15m bars is a real-time signal
- OI compression before breakout = fuel accumulation
- Can compute OI rate-of-change as a standalone indicator

---

## 3. Long/Short Ratio

### All Accounts — Contract Ratio
```
GET /api/v5/rubik/stat/contracts/long-short-account-ratio-contract
  Required: instId=ETH-USDT-SWAP
  Optional: period (15m | 1H | 4H | 1D), limit
```
Response: `[ts, ratio]` pairs — ratio ~2.4 means 2.4 long accounts per 1 short account

Trading use:
- Extreme readings (>4 or <0.5) = crowded positioning = contrarian signal
- Trend confirmation: ratio rising with price = trend has account-level support
- Best combined with OI: OI up + ratio up = strong long conviction

### Top Traders — Contract Ratio (by account count)
```
GET /api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader
  Required: instId=ETH-USDT-SWAP
  Optional: period (15m | 1H | 4H | 1D), limit
```
Response: `[ts, ratio]` pairs

Trading use:
- Top trader positioning leads retail by ~1-2 bars
- Top traders shorting while retail is long = distribution signal
- Divergence between top-trader and all-account ratios = smart money vs crowd

---

## 4. Taker Buy/Sell Volume

### Contract Taker Volume
```
GET /api/v5/rubik/stat/taker-volume-contract
  Required: instId=ETH-USDT-SWAP
  Optional: period (15m | 1H | 4H | 1D), limit
```
Response: `[ts, buy_vol_contracts, sell_vol_contracts]` arrays

Example: `["1781871300000", "83208.91", "86707.08"]` — sell volume slightly higher

Trading use:
- buy_vol / (buy_vol + sell_vol) = taker buy ratio
- Ratio > 0.6 sustained = aggressive buyers dominating = bullish pressure
- Ratio < 0.4 = sellers in control = bearish
- Sharp divergence from price = exhaustion signal (volume spike rev logic)
- This is the futures-specific version of spot CVD (cumulative volume delta)

---

## 5. Order Book

### Snapshot (up to 400 levels)
```
GET /api/v5/market/books
  Required: instId=ETH-USDT-SWAP
  Optional: sz (levels per side, max 400, default 1)
```
Response: `asks` and `bids` arrays of `[price, size, _, order_count]`, plus `ts` and `seqId`

Note: The `_` (3rd field) is always "0" in REST snapshots; it represents liquidation orders in WebSocket.

### Full Depth Book (up to 5000 levels)
```
GET /api/v5/market/books-full
  Required: instId=ETH-USDT-SWAP
  Optional: sz (max 5000)
```

Trading use:
- Bid/ask imbalance ratio at top 5-10 levels = short-term direction bias
- Large wall detection: single price level with order_count=1 and huge size = spoofed wall vs genuine
- Spread width as liquidity/volatility proxy
- REST polling is adequate for 15m context; use WebSocket for tick-level signals

---

## 6. Liquidation Orders

### Recent Liquidations
```
GET /api/v5/public/liquidation-orders
  Required: instType=SWAP
  For SWAPs also required: uly=ETH-USDT  (NOT instId)
  Optional: state=filled, limit (max 100)
```
Response: object with `instFamily`, `instId`, `uly`, `totalLoss`, and `details` array.
Each detail: `bkPx` (bankruptcy price), `posSide` (long/short), `side` (sell/buy), `sz`, `time`, `ts`

Note: The `uly` parameter uses format `ETH-USDT` not `ETH-USDT-SWAP`.

Trading use:
- Cluster of long liquidations just below current price = nearby support now cleared
- Cluster of short liquidations above = resistance cleared, path opened
- `bkPx` values across recent liquidations = estimate of liquidation cascade zones
- High `totalLoss` periods = volatility regime shift signal
- Liquidations above price + price dropping = cascade risk elevated

---

## 7. Mark Price & Premium

### Current Mark Price
```
GET /api/v5/public/mark-price
  Required: instType=SWAP
  Optional: instId=ETH-USDT-SWAP
```
Response: `instId`, `markPx`, `ts`

### Premium History (mark vs spot spread, per-minute)
```
GET /api/v5/public/premium-history
  Required: instId=ETH-USDT-SWAP
  Optional: before, after (ms), limit
```
Response: `instId`, `premium`, `ts` — sampled every ~1 minute

Example premium: -0.000426 = mark is 0.043% below spot index

Trading use:
- Premium = (mark_price / index_price) - 1
- Sustained negative premium + negative funding = shorts overcrowded
- Premium spike (> 0.1%) before a red candle = mark dragged up by longs = short signal
- Premium is the real-time precursor to the next funding rate

### Mark Price Candles
```
GET /api/v5/market/mark-price-candles
  Required: instId=ETH-USDT-SWAP
  Optional: bar (15m | 1H etc.), before, after, limit
```
Response: `[ts, open, high, low, close, confirm_flag]`

Trading use:
- Mark price candles are used for liquidation calculations, not last-price
- Comparing mark candles vs trade candles reveals manipulation (wick hunting)
- If mark price wicks down but last price does not = SL hunt on longs

---

## 8. Index Price

### Spot Index Ticker
```
GET /api/v5/market/index-tickers
  Required: instId=ETH-USDT  (NOT ETH-USDT-SWAP)
```
Response: `instId`, `idxPx`, `high24h`, `low24h`, `open24h`, `sodUtc0`, `sodUtc8`, `ts`

Trading use:
- Index = composite spot price from multiple exchanges (no single-exchange manipulation)
- `premium = (markPx / idxPx) - 1` computed in real time
- `sodUtc0` / `sodUtc8` = session open references for daily range context

---

## 9. Ticker & Trades

### Real-Time Ticker
```
GET /api/v5/market/ticker
  Required: instId=ETH-USDT-SWAP
```
Response: `last`, `lastSz`, `askPx`, `askSz`, `bidPx`, `bidSz`, `open24h`, `high24h`, `low24h`, `volCcy24h`, `vol24h`, `ts`, `sodUtc0`, `sodUtc8`

Trading use:
- `vol24h` as rolling liquidity check before entry
- `high24h` / `low24h` as daily S/R reference
- Spread (`askPx - bidPx`) as execution cost sanity check

### Recent Trades
```
GET /api/v5/market/trades
  Required: instId=ETH-USDT-SWAP
  Optional: limit (max 500)
```
Response: `instId`, `side`, `sz`, `px`, `tradeId`, `source`, `ts`

### Historical Trades (paginated)
```
GET /api/v5/market/history-trades
  Required: instId=ETH-USDT-SWAP
  Optional: type (1=tradeId pagination, 2=timestamp pagination), after, before, limit (max 100)
```

Trading use:
- Large single trades (sz >> median) from `source=0` = institutional flow
- Side imbalance in recent 100 trades = micro-timeframe taker delta
- Can reconstruct 1-second CVD from raw trade stream

---

## 10. Instruments (Reference)
```
GET /api/v5/public/instruments
  Required: instType=SWAP
  Optional: instId=ETH-USDT-SWAP
```
Returns 50+ fields including `ctVal` (contract value = 0.01 ETH), `tickSz`, `lotSz`, `lever` (max leverage), `minSz`, `settleCcy`

Trading use:
- `ctVal` needed to convert contract counts to ETH/USD notional
- `lever` confirms max available leverage for position sizing
- `minSz` / `lotSz` for order quantity rounding

---

## Priority Integration Ranking

| Priority | Endpoint | Signal Type | Effort |
|----------|----------|-------------|--------|
| HIGH | `/public/funding-rate` | Contrarian (extreme rates), pre-settlement squeeze | Low — single call per 15m bar |
| HIGH | `/rubik/stat/taker-volume-contract` | CVD proxy, exhaustion confirmation | Low — matches 15m bar period |
| HIGH | `/rubik/stat/contracts/open-interest-history` | OI divergence from price, trend health | Low — matches 15m period |
| MEDIUM | `/public/liquidation-orders` | Cascade zone identification, S/R clearing | Medium — needs clustering logic |
| MEDIUM | `/rubik/stat/contracts/long-short-account-ratio-contract` | Crowding/contrarian signal | Low |
| MEDIUM | `/rubik/stat/contracts/long-short-account-ratio-contract-top-trader` | Smart money vs retail divergence | Low |
| MEDIUM | `/public/premium-history` | Leading funding rate indicator | Medium — per-minute sampling |
| LOW | `/market/books` | Bid/ask imbalance for entry timing | High — needs real-time polling |
| LOW | `/market/trades` | Micro CVD, large trade detection | High — needs aggregation |
| LOW | `/market/mark-price-candles` | SL hunt detection vs last price | Low |

---

## Parameter Notes

Valid `period` values for rubik/stat endpoints: `5m`, `15m`, `30m`, `1H`, `2H`, `4H`, `6H`, `12H`, `1D`, `1W`

Valid `bar` values for candle endpoints: `1m`, `3m`, `5m`, `15m`, `30m`, `1H`, `2H`, `4H`, `6H`, `12H`, `1D`, `1W`

Rate limits: OKX applies per-endpoint limits. Public market data endpoints typically allow 20 requests/2s. The `rubik/stat` endpoints have lower limits (~5-10 req/2s).

For real-time streaming, OKX provides WebSocket at `wss://ws.okx.com:8443/ws/v5/public` covering all the above data. The REST endpoints above are sufficient for 15m bar-level signal computation.
