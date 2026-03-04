# Estrategia Final a Probar: Volume Spike Opcion B

## Resultado del Mega Backtest (Phase 1 + Phase 2)
- **551K+ simulaciones** en Phase 1, **322K** en Phase 2
- **27 estrategias** base + **18 combos** x **19 pares** x **2 anos**
- Walk-forward validation: **0 overfit detectado** (15/15 ROBUST)

---

## Config Ganadora (Opcion B - Equilibrada)

```
Estrategia:      volume_spike
SL:              3%
TP:              4%
Trailing Stop:   6%
Cooldown:        48h
Leverage:        10x
Position Size:   2% del equity
Min Score:       5
```

### Metricas 2-Year Backtest ($50K inicial)
| Metrica | Valor |
|---------|-------|
| PnL Total | **+$231,185 (+462%)** |
| Profit Factor | **1.29** |
| Win Rate | **48%** |
| Green Months | **21/25 (84%)** |
| Pares Rentables | **19/19 (100%)** |
| Max Drawdown | ~15% |

### Comparacion vs Config Actual
| | Actual (macd_cross) | Nueva (volume_spike) | Mejora |
|---|---|---|---|
| PnL 2yr | $87K | $231K | **+165%** |
| PF | 1.16 | 1.29 | +11% |
| WR | ~35% | 48% | +13pp |
| Green Mo | — | 21/25 | — |

---

## Cambios Requeridos en Produccion

### 1. `bot/strategy/signal_detector.py` — Reemplazar estrategias

**ELIMINAR**: `macd_cross_1h` y `trend_follow_1h` como estrategias de entrada.

**AGREGAR**: `volume_spike` con la siguiente logica:

```python
def detect_volume_spike(self, snapshot: dict) -> dict | None:
    """Volume spike + directional bias."""
    vol_ratio = snapshot.get("volume_ratio", 0)
    adx = snapshot.get("adx")
    macd_sig = snapshot.get("macd_signal")
    ema_align = snapshot.get("ema_alignment")

    if adx is None or macd_sig is None or ema_align is None:
        return None
    if vol_ratio <= 2.5 or adx <= 20:
        return None

    score = 5
    if vol_ratio > 3.5: score += 1
    if vol_ratio > 5.0: score += 1

    if macd_sig in ("bullish", "bullish_cross") and ema_align > 0:
        return {
            "signal": "LONG",
            "strategy": "volume_spike",
            "score": score,
            "reason": f"Vol spike {vol_ratio:.1f}x + MACD bullish + EMA aligned"
        }
    elif macd_sig in ("bearish", "bearish_cross") and ema_align < 0:
        return {
            "signal": "SHORT",
            "strategy": "volume_spike",
            "score": score,
            "reason": f"Vol spike {vol_ratio:.1f}x + MACD bearish + EMA aligned"
        }
    return None
```

**Condiciones de entrada:**
- `volume_ratio > 2.5` (volumen actual 2.5x por encima de SMA)
- `ADX > 20` (hay algo de tendencia)
- `MACD` confirma direccion (bullish/bearish)
- `EMA alignment` confirma direccion (> 0 para LONG, < 0 para SHORT)
- Score minimo: 5 (filtro en engine)

### 2. `bot/config/settings.py` — Actualizar parametros

```python
# ANTES:
STOP_LOSS_PCT = 0.03      # 3% - MANTENER
TAKE_PROFIT_PCT = 0.06    # 6% - CAMBIAR
TRAILING_STOP_PCT = 0.06  # 6% - MANTENER
COOLDOWN_HOURS = 12       # 12h - CAMBIAR
LEVERAGE = 10             # 10x - MANTENER
POSITION_PCT = 0.02       # 2% - MANTENER

# DESPUES:
STOP_LOSS_PCT = 0.03      # 3% (sin cambio)
TAKE_PROFIT_PCT = 0.04    # 4% (era 6%)
TRAILING_STOP_PCT = 0.06  # 6% (sin cambio)
COOLDOWN_HOURS = 48       # 48h (era 12h)
LEVERAGE = 10             # 10x (sin cambio)
POSITION_PCT = 0.02       # 2% (sin cambio)
```

**Resumen de cambios en settings:**
- `TAKE_PROFIT_PCT`: 6% -> **4%** (TP mas cercano, WR sube de 28% a 48%)
- `COOLDOWN_HOURS`: 12h -> **48h** (menos trades, mayor calidad, PF sube a 1.29)

### 3. `bot/core/engine.py` — Ajustar flujo

- Cambiar referencia de `macd_cross_1h`/`trend_follow_1h` a `volume_spike`
- El cooldown de 48h se lee de settings, deberia funcionar automatico
- La logica de 1H candle check se mantiene igual (la senal se evalua cada 1H candle cerrada)

### 4. `bot/strategy/market_analyzer.py` — Sin cambios

Ya calcula todos los indicadores necesarios:
- `volume_ratio` (SMA de volumen)
- `adx`
- `macd_signal`
- `ema_alignment`

### 5. `bot/data/stream_manager.py` — Sin cambios

Ya tiene kline_1h stream para cada par.

### 6. `bot/ai/claude_trader.py` — Actualizar prompt

Cambiar la descripcion de la estrategia en el system prompt de Claude:
- Mencionar que ahora usamos `volume_spike` en vez de `macd_cross_1h`
- Ajustar las descripciones de cuando abrir/cerrar

---

## Bonus: Filtro de Session (Opcional)

Phase 2 mostro que **quitar Asian session** (08-24 UTC) mejora PF:
- volume_spike all_sessions: PF=1.245
- volume_spike no_asian: PF=1.250

Si se implementa, agregar en el engine:
```python
# Solo operar entre 08:00 y 24:00 UTC
from datetime import datetime, timezone
hour_utc = datetime.now(timezone.utc).hour
if hour_utc < 8:
    return  # Skip Asian session
```

Mejora marginal (+0.5% PF), implementar solo si se quiere optimizar al maximo.

---

## Validacion Walk-Forward

| Metrica | Resultado |
|---------|-----------|
| Metodo A (50/50) | ROBUST |
| Metodo B (Rolling 12m/3m) | ROBUST |
| Metodo C (Quarterly) | ROBUST |
| Degradation promedio | **Negativa** (test > train) |
| Veredicto | **NO HAY OVERFITTING** |

---

## Plan de Implementacion

1. Modificar `signal_detector.py` (agregar volume_spike, quitar macd/trend)
2. Modificar `settings.py` (TP=4%, CD=48h)
3. Ajustar `engine.py` (referencia a volume_spike)
4. Actualizar prompt de Claude en `claude_trader.py`
5. Reset paper trading balance a $200K
6. Correr 2-4 semanas en paper
7. Comparar resultados live vs backtest
8. Si OK: deploy a produccion real

---

## Archivos a Modificar (Checklist)

- [ ] `bot/strategy/signal_detector.py` — nueva estrategia
- [ ] `bot/config/settings.py` — TP=4%, CD=48h
- [ ] `bot/core/engine.py` — referencia volume_spike
- [ ] `bot/ai/claude_trader.py` — actualizar prompt
- [ ] `.env` en VPS — si hay overrides de TP/CD

## Archivos que NO cambian
- `bot/strategy/market_analyzer.py`
- `bot/strategy/indicators.py`
- `bot/data/stream_manager.py`
- `bot/data/candles.py`
- `bot/execution/paper_trader.py`
- `bot/risk/risk_manager.py`
