# Confirmed dead-end strategies for crypto futures (OKX)
# Do NOT re-test these unless market structure fundamentally changes.

## Simple MA Crossovers (any period)
- **Tested:** Multiple EMA/SMA combinations (5/20, 9/21, 20/50, 50/200)
- **Result:** WR ~48-52%, PF < 1.2 after costs on 15m
- **Why:** Crypto volatility causes constant whipsaws; by the time MAs confirm, the move is mostly done
- **Don't retry:** The lag inherent in MAs cannot be solved

## Single-Indicator Direction Prediction
- **Tested:** RSI alone, MACD alone, Stochastic alone, ADX alone
- **Result:** WR < 53% on any single indicator
- **Why:** No single lagging indicator can predict direction in a random walk with fat tails
- **Don't retry:** Combining indicators helps slightly (current combos), but a single one never will

## Grid Bots in Trending Markets
- **Why:** Unlimited downside when price trends away from grid
- **Don't retry:** Would need perfect regime detection (which we don't have)

## DCA / Martingale Without Stop
- **Why:** Crypto can drop 50%+ in days (Luna, FTX). Infinite risk.
- **Don't retry:** Mathematically guaranteed to blow up eventually

## Pure Mean-Reversion Without Trend Filter
- **Why:** Crypto trends are stronger and longer than equity markets; mean-reversion alone gets crushed
- **Lesson:** Our meanrev_confluence uses trend filter (EMA 20/50) for exactly this reason
