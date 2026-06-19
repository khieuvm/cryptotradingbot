---
description: "Activate or deactivate a strategy for live/dry-run trading"
---

# /deploy [strategy] [action]

Safely activate or deactivate a strategy by updating `config/base.yaml`.

## Usage
```
/deploy regime_adaptive activate
/deploy volume_spike_rev deactivate
/deploy cb_adx_breakout activate --pairs ETH SOL
```

## Actions

| Action | What it does |
|--------|-------------|
| `activate` | Set `strategies.<name>.active: true` in base.yaml. Requires Grade A or B. |
| `deactivate` | Set `strategies.<name>.active: false` in base.yaml. Safe to run at any time. |

## What happens on `activate`
1. Read current grade from `config/base.yaml` — must be A or B (rejects C/F)
2. Check that `/validate` was run (looks for grade entry in strategy config)
3. Print deployment checklist (see below)
4. **Ask for confirmation** before writing any config change
5. Update `strategies.<name>.active: true` in `config/base.yaml`

## What happens on `deactivate`
1. Set `strategies.<name>.active: false` in `config/base.yaml`
2. Note: circuit breaker may have already auto-disabled the strategy — check `/status`

## Deployment checklist (printed before confirmation)

```
Pre-deploy checklist for <strategy>:
  [ ] Grade: <grade> (A/B required)
  [ ] Validated: walk-forward OOS PF > threshold
  [ ] Pairs: <pairs> — data downloaded and fresh?
  [ ] Leverage: <leverage>x — within 5x limit?
  [ ] Stoploss: <sl> — not wider than -20%?
  [ ] Circuit breaker: no recent auto-disable for this strategy?
  [ ] QA: /validate and QA agent signed off?
```

## Grade requirements

| Grade | Eligible for |
|-------|-------------|
| A | Live trading + dry-run |
| B | Live trading + dry-run |
| C | Dry-run only (research) |
| F | Not deployable — reject |

## After deployment

Restart the bot to pick up config changes:
```bash
python ft_run.py trade --strategy CryptoEngine --env dryrun
```
