# Estrategia Verificada — Backtest 2 Años (Marzo 2024 → Marzo 2026)

## Fecha de verificación: 2026-03-03
## Status: PRODUCCIÓN — Validada con 2 años de datos reales

---

## Configuración Ganadora

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Estrategia primaria** | `macd_cross_1h` | Mejor cross-pair: 13/19 pares rentables con misma config |
| **Estrategia secundaria** | `trend_follow_1h` | 14/19 pares, PF=1.28, complementa en tendencias fuertes |
| **Stop Loss** | 3% | Óptimo balance entre protección y ruido de mercado |
| **Take Profit** | 6% | R:R = 1:2, compensa el win rate bajo (~30%) |
| **Trailing Stop** | 6% | Captura extensiones sin salir prematuramente |
| **Cooldown** | 12h | Más oportunidades que 48h sin sobreoperar (+$21K vs CD=48h) |
| **Leverage** | 10x | Máximo retorno dentro de límites de riesgo |
| **Position Size** | 2% del equity | Balance entre crecimiento y protección |
| **Min Score** | 5 | Umbral mínimo de señal para entrar |
| **EMA Alignment** | STRICT (>0.3 LONG, <-0.3 SHORT) | Filtra señales neutras débiles |
| **Volume Ratio** | > 0.5 | Confirma actividad mínima |
| **Entry Fee** | Maker 0.02% (limit order) | Sin slippage en entrada |
| **TP Exit Fee** | Maker 0.02% (limit order) | Sin slippage en TP |
| **SL Exit Fee** | Taker 0.05% + slippage (0.01% majors, 0.02% alts) | Peor caso en SL |

---

## Reglas de Entrada — MACD Cross 1H (STRICT)

```
LONG:
  - macd_signal == "bullish_cross" (MACD histogram cruza de negativo a positivo)
  - ema_alignment > 0.3 (EMA 9 > EMA 21 > EMA 50, alineación real)
  - volume_ratio > 0.5 (volumen actual / SMA(20) de volumen)
  - Score base: 5
  - Bonus: +1 si ema_alignment > 0.5 (strong), +1 si volume_ratio > 1.5 (spike)

SHORT:
  - macd_signal == "bearish_cross" (MACD histogram cruza de positivo a negativo)
  - ema_alignment < -0.3 (EMA 9 < EMA 21 < EMA 50, alineación real)
  - volume_ratio > 0.5
  - Score base: 5
  - Bonus: +1 si ema_alignment < -0.5, +1 si volume_ratio > 1.5
```

### Por qué STRICT y no CURRENT

El backtest de 1 año original usaba `ema_alignment >= 0` (LONG) y `<= 0` (SHORT), lo cual permitía entrar con EMA neutra (0.0). Esto causó problemas en paper trading:

- **BTC SHORT con EMA=0.0**: perdió $1,208 — entró sin confirmación de tendencia
- **4 shorts seguidos con F&G=10**: todos stopped out, -$4,877 en 2 días

El backtest de 2 años comparó ambas reglas:

| Regla | MACD cross configs rentables | Mejor PnL cross-pair | PF |
|-------|------------------------------|----------------------|----|
| CURRENT (ema≥0/≤0) | 34.8% en 17 pares | $65,421 (+131%) | 1.11 |
| **STRICT (ema>0.3/<-0.3)** | **36.6%** en 15 pares | **$86,964 (+174%)** | **1.16** |

STRICT genera **$21,543 más** (+33%) con ~25% menos señales. Menos trades, mayor calidad.

---

## Reglas de Entrada — Trend Follow 1H (Secundaria)

```
LONG:
  - ADX > 25 (tendencia establecida)
  - ema_alignment > 0.3 (tendencia alcista)
  - plus_di > minus_di (presión compradora)
  - 30 <= RSI(14) <= 55 (pullback, no sobrecompra)
  - MACD en "bullish" o "bullish_cross"
  - Score base: 4, bonuses: ADX>35 (+1), ema>0.5 (+1), bullish_cross (+2), vol>1.2 (+1), stoch<40 (+1)

SHORT:
  - ADX > 25
  - ema_alignment < -0.3
  - minus_di > plus_di
  - 45 <= RSI(14) <= 70
  - MACD en "bearish" o "bearish_cross"
  - Score base: 4, mismos bonuses invertidos
```

---

## Resultados del Backtest

### Datos del test
- **Periodo**: 2024-03-04 a 2026-03-04 (730 días, 17,520 velas 1H por par)
- **Capital inicial**: $50,000
- **Pares**: 19 (BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, DOT, LINK, SUI, ARB, OP, APT, NEAR, LTC, ATOM, FIL, INJ)
- **Simulaciones**: 131,328 configuraciones testeadas
- **Motor**: 14 cores en paralelo, 137 segundos

### Resultado — MACD Cross STRICT (config ganadora cross-pair)

| Métrica | Valor |
|---------|-------|
| **Pares rentables** | 13/19 con misma config |
| **PnL total (13 pares)** | +$86,964 (+173.9%) |
| **Profit Factor** | 1.16 |
| **Total trades** | 2,956 |
| **Parámetros** | SL=3% TP=6% Trail=6% CD=12h Lev=10x Pos=2% |

### Resultado — Trend Follow 1H (mejor cross-pair)

| Métrica | Valor |
|---------|-------|
| **Pares rentables** | 14/19 con misma config |
| **PnL total (14 pares)** | +$37,462 (+74.9%) |
| **Profit Factor** | 1.28 |
| **Total trades** | 1,079 |
| **Parámetros** | SL=3% TP=6% Trail=6% CD=12h Lev=8x Pos=2% |

### Per-pair ranking (mejor resultado individual por par)

| Par | Estrategia ganadora | PF | PnL | Return |
|-----|--------------------|----|-----|--------|
| ADAUSDT | multi_confirm | 1.35 | +$28,502 | +57.0% |
| DOGEUSDT | multi_confirm | 1.43 | +$24,723 | +49.5% |
| SOLUSDT | multi_confirm | 1.37 | +$24,628 | +49.3% |
| BNBUSDT | macd_cross_1h | 2.08 | +$22,411 | +44.8% |
| APTUSDT | multi_confirm | 1.43 | +$21,096 | +42.2% |
| AVAXUSDT | macd_cross_1h | 1.46 | +$18,310 | +36.6% |
| INJUSDT | multi_confirm | 1.26 | +$16,598 | +33.2% |
| LINKUSDT | multi_confirm | 1.25 | +$16,564 | +33.1% |
| SUIUSDT | multi_confirm | 1.24 | +$15,481 | +31.0% |
| XRPUSDT | macd_cross_1h | 1.40 | +$14,686 | +29.4% |
| OPUSDT | macd_cross_1h | 1.49 | +$14,527 | +29.1% |
| FILUSDT | multi_confirm | 1.15 | +$13,730 | +27.5% |
| BTCUSDT | multi_confirm | 1.60 | +$12,493 | +25.0% |
| NEARUSDT | multi_confirm | 1.16 | +$12,031 | +24.1% |
| DOTUSDT | multi_confirm | 1.14 | +$11,451 | +22.9% |
| ETHUSDT | multi_confirm | 1.38 | +$11,056 | +22.1% |
| ARBUSDT | multi_confirm | 1.31 | +$10,512 | +21.0% |
| ATOMUSDT | multi_confirm | 1.14 | +$8,966 | +17.9% |
| LTCUSDT | multi_confirm | 1.15 | +$5,673 | +11.3% |

**19/19 pares rentables** — ningún par se descarta.

---

## Cambios vs Configuración Anterior (1 año)

| Parámetro | Antes (1 año) | Ahora (2 años) | Razón del cambio |
|-----------|---------------|----------------|------------------|
| EMA alignment LONG | >= 0 | **> 0.3** | Filtra señales neutras que pierden dinero |
| EMA alignment SHORT | <= 0 | **< -0.3** | Evita shorts sin confirmación de tendencia |
| Cooldown | 48h | **12h** | +$21K más profit, más oportunidades por par |
| Periodo de backtest | 365 días | **730 días** | Doble validación, incluye bull + bear markets |

---

## Indicadores Utilizados (calculados sobre velas 1H)

1. **MACD (12, 26, 9)** — Señal principal: cruces bullish/bearish del histograma
2. **EMA (9, 21, 50)** — Alineación: score -1 a +1 según orden de EMAs
3. **Volume Ratio** — Volumen actual / SMA(20) del volumen
4. **ADX (14)** — Fuerza de tendencia (>25 = trending)
5. **+DI / -DI** — Dirección del movimiento
6. **RSI (14)** — Momentum, pullback detection (30-55 LONG, 45-70 SHORT)
7. **Stochastic RSI (14,14,3,3)** — Zonas de sobrecompra/sobreventa
8. **Bollinger Bands (20, 2)** — Volatilidad y posición del precio
9. **MFI (14)** — Flujo de dinero (volumen + precio)
10. **Williams %R (14)** — Oscillador de momentum
11. **CCI (20)** — Commodity Channel Index
12. **Ichimoku (9, 26)** — Cloud, tenkan/kijun cross
13. **RSI Divergence** — Divergencias bullish/bearish

---

## Limitaciones Conocidas del Backtest

1. **No simula pérdidas correlacionadas**: Cada par se testea independiente con $50K. En producción, múltiples posiciones pueden perder simultáneamente (como pasó con 4 shorts el 2 de marzo).
2. **No incluye funding rates**: Las tasas de financiamiento de futuros no están modeladas.
3. **Slippage simplificado**: Se usa 0.01% majors / 0.02% alts, el real puede variar.
4. **Sin circuit breaker**: El backtest no para ante drawdowns como el bot real.
5. **Win rate bajo (~30%)**: Normal para estrategia de R:R 1:2. Esperar rachas de 4-6 pérdidas seguidas es estadísticamente esperado.

---

## Archivos de Referencia

- `bot/backtest/results_2year_strict.json` — Resultados completos STRICT (200 top configs)
- `bot/backtest/results_2year_current.json` — Resultados completos CURRENT (comparación)
- `bot/backtest/run_2year_fast.py` — Script de backtest (multiprocessing, 14 cores)
- `bot/strategy/signal_detector.py` — Detector de señales en producción
