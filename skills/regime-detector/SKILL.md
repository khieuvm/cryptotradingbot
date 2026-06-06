# Regime Detector Skill

Classify current market regime to enable/disable appropriate combos and adjust parameters.

## Regime Classifications

| Regime | Conditions | Trading Approach |
|--------|-----------|-----------------|
| TRENDING | ADX > 25, DI separation > 10, EMA aligned | Trend-following combos active |
| RANGING | ADX < 20, BB width < 50th pctile | Mean-reversion combos active |
| VOLATILE | ATR > 2x SMA50 of ATR | Reduce position size, widen stops |
| SQUEEZE | BB inside KC for 3+ bars | Prepare for breakout entries |
| FUNDING_EXTREME | abs(funding) > 0.1%/8h | Contrarian bias |

## Detection Logic

```python
# Computed on 15m candles
regime = "UNKNOWN"

if atr_ratio > 2.0:
    regime = "VOLATILE"
elif adx > 25 and abs(plus_di - minus_di) > 10:
    regime = "TRENDING"
elif adx < 20 and bb_width < bb_width_sma:
    regime = "SQUEEZE" if bb_inside_kc else "RANGING"

# Overlay: funding regime
if abs(funding_rate) > 0.001:
    regime += "+FUNDING_EXTREME"
```

## Regime → Combo Mapping

| Regime | regime_adaptive | meanrev_confluence | trend_composite |
|--------|----------------|-------------------|-----------------|
| TRENDING | ✓ (trend mode) | ✗ (disabled) | ✓ |
| RANGING | ✓ (range mode) | ✓ | ✗ (disabled) |
| VOLATILE | ✓ (reduced size) | ✗ | ✗ |
| SQUEEZE | ✓ | ✗ | Prepare only |

## Volume Sessions (24/7 Adaptation)

Instead of fixed market sessions, crypto has volume patterns:

| Session | UTC Hours | Character |
|---------|-----------|-----------|
| Asia | 00:00-08:00 | Low volume, range-bound, mean-reversion works |
| Europe | 08:00-13:00 | Building momentum, breakouts begin |
| US | 13:00-21:00 | Highest volume, strongest trends |
| Dead | 21:00-00:00 | Thin liquidity, unpredictable |

## Usage

The regime detector runs inside `CryptoMaster_OKX.confirm_trade_entry()` as an additional filter. If the current regime doesn't match the combo's intended regime, the entry is blocked.
