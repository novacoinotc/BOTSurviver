# MEGA BACKTEST PHASE 2 — Advanced Optimization Report

**Total runtime:** 957.6s

## 1. Combo Strategies (18 combos)

| # | Combo | ProfRate | Pairs | BestPnL | AvgPF | WR |
|---|-------|---------|-------|---------|-------|-----|
| 1 | vs_ema_aligned | 68.2% | 19/19 | $+69,085 | 1.13 | 33.6% |
| 2 | vs_macd_confirms | 68.2% | 19/19 | $+69,085 | 1.13 | 33.6% |
| 3 | vs_adx_trending | 68.0% | 19/19 | $+55,309 | 1.14 | 33.7% |
| 4 | mtf_adx_trending | 62.3% | 19/19 | $+96,520 | 1.08 | 33.5% |
| 5 | regime_rsi_neutral | 49.8% | 19/19 | $+71,811 | 1.01 | 32.8% |
| 6 | macd_adx_trending | 48.2% | 19/19 | $+22,599 | 1.01 | 33.8% |
| 7 | mtf_supertrend | 46.3% | 19/19 | $+16,401 | 0.99 | 33.4% |
| 8 | regime_vol15 | 43.9% | 19/19 | $+26,907 | 0.97 | 33.1% |
| 9 | vs_rsi_neutral | 39.1% | 19/19 | $+14,472 | 0.96 | 32.9% |
| 10 | macd_rsi_neutral | 40.2% | 19/19 | $+23,172 | 0.95 | 33.0% |
| 11 | regime_supertrend | 29.2% | 19/19 | $+11,957 | 0.89 | 31.0% |
| 12 | tf_ichimoku | 50.2% | 18/19 | $+9,888 | 1.05 | 34.8% |
| 13 | vs_supertrend | 40.7% | 18/19 | $+12,699 | 0.97 | 32.1% |
| 14 | mtf_rsi_neutral | 38.5% | 18/19 | $+25,557 | 0.95 | 33.1% |
| 15 | macd_supertrend | 24.9% | 18/19 | $+12,980 | 0.86 | 30.5% |
| 16 | vs_all_filters | 29.6% | 16/19 | $+7,678 | 0.87 | 29.1% |
| 17 | tf_supertrend | 24.3% | 14/19 | $+6,143 | 0.76 | 28.4% |
| 18 | tf_vol20 | 30.5% | 13/18 | $+7,318 | 0.91 | 28.4% |

**Best combo:** vs_ema_aligned (BestPnL=$+69,085) vs **Phase 1 winner:** volume_spike (BestPnL=$+27,995)

## 2. Walk-Forward Validation

**Verdicts:** ROBUST=15 | MARGINAL=0 | OVERFIT=0

| # | Strategy | Verdict | AvgDeg | TrainPF | TestPF | Windows |
|---|----------|---------|--------|---------|--------|---------|
| 1 | vs_ema_aligned | ROBUST | -15.7% | 1.228 | 1.363 | 209 |
| 2 | vs_ema_aligned | ROBUST | -15.7% | 1.228 | 1.363 | 209 |
| 3 | vs_macd_confirms | ROBUST | -15.7% | 1.228 | 1.363 | 209 |
| 4 | vs_ema_aligned | ROBUST | -9.4% | 1.266 | 1.304 | 209 |
| 5 | vs_ema_aligned | ROBUST | -9.4% | 1.266 | 1.304 | 209 |
| 6 | vs_macd_confirms | ROBUST | -9.4% | 1.266 | 1.304 | 209 |
| 7 | vs_macd_confirms | ROBUST | -9.4% | 1.266 | 1.304 | 209 |
| 8 | vs_ema_aligned | ROBUST | -9.3% | 1.283 | 1.320 | 209 |
| 9 | vs_macd_confirms | ROBUST | -9.3% | 1.283 | 1.320 | 209 |
| 10 | vs_ema_aligned | ROBUST | -4.9% | 1.298 | 1.277 | 209 |
| 11 | vs_ema_aligned | ROBUST | -4.9% | 1.298 | 1.277 | 209 |
| 12 | vs_macd_confirms | ROBUST | -4.9% | 1.298 | 1.277 | 209 |
| 13 | vs_macd_confirms | ROBUST | -4.9% | 1.298 | 1.277 | 209 |
| 14 | mtf_adx_trending | ROBUST | -4.2% | 1.187 | 1.208 | 209 |
| 15 | mtf_adx_trending | ROBUST | -4.2% | 1.187 | 1.208 | 209 |

## 3. Session Filter Analysis

| Session | TotalPnL | Trades | Configs | AvgPF |
|---------|----------|--------|---------|-------|
| asian_only | $+252,510 | 11388 | 95 | 1.147 |
| london_only | $+322,723 | 12419 | 95 | 1.176 |
| ny_only | $+359,260 | 12045 | 95 | 1.157 |
| no_asian | $+547,780 | 16947 | 95 | 1.167 |
| london_ny | $+547,780 | 16947 | 95 | 1.167 |
| all_sessions | $+528,434 | 19172 | 95 | 1.143 |

## 4. Ensemble Voting

| Level | ProfRate | Pairs | BestPnL | AvgPF | WR | AvgTrades |
|-------|---------|-------|---------|-------|-----|-----------|
| ensemble_5 | 35.7% | 19/19 | $+52,828 | 0.94 | 32.2% | 246 |
| ensemble_4 | 29.8% | 19/19 | $+26,740 | 0.91 | 31.8% | 309 |
| ensemble_2 | 26.4% | 19/19 | $+34,692 | 0.89 | 31.6% | 362 |
| ensemble_3 | 22.4% | 19/19 | $+29,832 | 0.87 | 31.3% | 345 |

## 5. Per-Pair Portfolio

- **Total PnL:** $+51,943.73 (+103.9%)
- **Trades:** 3196
- **Win Rate:** 38.5%
- **Profitable Pairs:** 19/19
- **Green Months:** 22/25

| Pair | Strategy | PnL (scaled) | PF | WR | MaxDD |
|------|----------|-------------|-----|-----|-------|
| SUIUSDT | mtf_adx_trending | $+5,656 | 1.97 | 20.3% | -16.2% |
| NEARUSDT | vs_ema_aligned | $+4,042 | 1.81 | 44.6% | -8.9% |
| ADAUSDT | regime_rsi_neutral | $+3,843 | 1.69 | 40.3% | -10.2% |
| DOGEUSDT | regime_rsi_neutral | $+3,809 | 2.03 | 14.9% | -10.3% |
| APTUSDT | mtf_adx_trending | $+3,143 | 1.71 | 51.7% | -8.1% |
| ARBUSDT | mtf_adx_trending | $+3,109 | 1.79 | 18.4% | -14.9% |
| XRPUSDT | mtf_adx_trending | $+2,932 | 1.67 | 31.6% | -6.2% |
| INJUSDT | vs_adx_trending | $+2,761 | 1.97 | 55.0% | -5.7% |
| ATOMUSDT | mtf_adx_trending | $+2,684 | 1.68 | 26.7% | -11.3% |
| OPUSDT | mtf_adx_trending | $+2,648 | 1.55 | 35.6% | -7.3% |
| SOLUSDT | regime_rsi_neutral | $+2,599 | 1.42 | 55.3% | -13.4% |
| AVAXUSDT | vs_ema_aligned | $+2,571 | 1.78 | 54.1% | -12.7% |
| LTCUSDT | vs_adx_trending | $+2,395 | 1.76 | 38.2% | -11.9% |
| ETHUSDT | vs_adx_trending | $+2,091 | 1.86 | 34.0% | -4.3% |
| DOTUSDT | mtf_adx_trending | $+1,768 | 1.37 | 52.5% | -10.2% |
| BNBUSDT | mtf_adx_trending | $+1,748 | 1.67 | 46.3% | -10.1% |
| FILUSDT | vs_macd_confirms | $+1,639 | 1.42 | 43.2% | -12.7% |
| LINKUSDT | mtf_adx_trending | $+1,474 | 1.44 | 43.7% | -13.7% |
| BTCUSDT | mtf_adx_trending | $+1,032 | 1.49 | 37.4% | -6.2% |

## 6. Production Recommendations

Based on Phase 2 analysis:

1. **Most robust strategy:** vs_ema_aligned (AvgDeg=-15.7%, TestPF=1.363)
   - Params: SL=3% TP=4% Trail=0% Lev=20x CD=12h
2. **Best combo:** vs_ema_aligned (BestPnL=$+69,085, ProfRate=68.2%)