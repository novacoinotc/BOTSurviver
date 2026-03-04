# Trading Knowledge Base for AI Scalping Bot

## 1. RSI (Relative Strength Index) — Correct Interpretation
- RSI measures MOMENTUM, not direction. High RSI = fast move up, Low RSI = fast move down.
- **RSI < 30 (oversold)**: The selling pressure is EXHAUSTED. Buyers are stepping in. A bounce is statistically likely.
  - DO NOT open new shorts here. The easy money on the downside is already gone.
  - Wait for the bounce, then short the bounce at RSI 45-60.
- **RSI > 70 (overbought)**: The buying pressure is EXHAUSTED. Sellers are stepping in.
  - DO NOT open new longs here. Wait for the pullback.
- **RSI 40-60**: Neutral zone. Good for new entries if other indicators confirm.
- **RSI divergence**: When price makes a new low but RSI makes a higher low = BULLISH divergence.
  This is one of the strongest reversal signals. It means selling momentum is weakening even as price drops.
  NEVER trade against a divergence signal.

## 2. Mean Reversion in Ranging Markets
- In a ranging market, price oscillates between support and resistance like a pendulum.
- The CORRECT strategy: buy near support (bottom), sell near resistance (top).
- The WRONG strategy: sell at the bottom because "it's going down" — NO, in ranging it bounces.
- Key indicators for ranging: ADX < 20, price bouncing between BB bands, no EMA trend.
- TP should be tighter (1-1.5x ATR) because price won't break out, it'll reverse at the range boundary.

## 3. Trend Following vs Mean Reversion
- **When to trend follow** (ADX > 25, EMAs aligned):
  - Enter on pullbacks to EMA21/50, not at the extreme of the move.
  - SL behind the pullback low/high. TP at 2-2.5x ATR.
  - Don't chase: if the move already happened (RSI extreme), wait for pullback.
- **When to mean revert** (ADX < 20, ranging):
  - Enter at range extremes (near support/resistance).
  - Tighter TP (1-1.5x ATR) because the range will hold.
  - Wider SL (beyond the range boundary) to avoid noise stop-outs.

## 4. Volume Analysis
- **Volume confirms moves**: A breakout with high volume is real. A breakout with low volume is fake.
- **Volume spike + reversal candle**: Very strong signal. It means aggressive traders got trapped and are covering.
- **Volume ratio > 2.0**: Unusual activity. Something is happening. Pay attention.
- **Volume ratio < 0.5**: Low interest. No conviction. Avoid entering — the move may not sustain.
- **Buy ratio**: If volume_buy_ratio > 0.6, buyers dominate. If < 0.4, sellers dominate.

## 5. Bollinger Bands
- **BB squeeze** (narrow bands): Volatility is compressing. A big move is coming, but direction unknown.
  - Don't enter DURING the squeeze. Wait for the breakout direction.
  - When bands expand after squeeze + volume, enter in the breakout direction.
- **Price at BB lower + oversold RSI**: In ranging, this is a BUY signal (bounce expected).
- **Price at BB upper + overbought RSI**: In ranging, this is a SELL signal (drop expected).
- **BB %B < 0.05**: Price is at the very bottom of the bands. In ranging = bounce. In trending = continuation (be careful).

## 6. MACD (Moving Average Convergence Divergence)
- **Bullish cross** (MACD line crosses above signal): Momentum shifting up. Good for longs, bad for shorts.
- **Bearish cross** (MACD line crosses below signal): Momentum shifting down. Good for shorts, bad for longs.
- **Histogram**: Shows the STRENGTH of momentum. Shrinking histogram = weakening momentum = potential reversal.
- **5m MACD vs 1m MACD**: Higher timeframe overrides. If 5m MACD is bullish but 1m is bearish, the bounce is likely coming.

## 7. Order Book / Book Imbalance
- **Book imbalance > 2**: Strong buy-side pressure. Risky to short against this.
- **Book imbalance < 0.5**: Strong sell-side pressure. Risky to long against this.
- **Book imbalance 0.8-1.2**: Balanced. No directional pressure from order flow.
- Large imbalance + price moving against it = trapped traders. The imbalance side will likely win eventually.

## 8. Support & Resistance
- S/R are zones, not exact prices. Expect bounces within 0.1-0.3% of the level.
- **Closer to support**: Better for longs. The risk/reward is favorable (small SL below support, TP at resistance).
- **Closer to resistance**: Better for shorts. The risk/reward is favorable (small SL above resistance, TP at support).
- **Broken support becomes resistance** and vice versa. After a breakout, wait for the retest of the broken level.

## 9. Fear & Greed Index — Macro Context
- This is a DAILY indicator. It tells you the OVERALL market sentiment, not per-minute direction.
- **Extreme Fear (< 20)**: Most people are panic selling. The trend is bearish. But:
  - Smart money buys fear. Bounces can be sharp and violent.
  - Short the BOUNCES (relief rallies), don't short the panic lows.
  - Wait for RSI to recover to 45-60, then enter short.
- **Extreme Greed (> 80)**: Most people are FOMO buying. The trend is bullish. But:
  - Corrections can be sudden. Protect long profits aggressively.
  - Consider shorting if technical reversal signals appear.

## 10. Position Management (Scalping)
- **Scalping = small, frequent wins**. Don't try to catch the whole move.
- A 0.7% win is BETTER than holding for 2.5% and getting stopped at -1.5%.
- **At +0.5% unrealized profit**: Move SL to breakeven. Lock in the guaranteed no-loss.
- **At +1% unrealized profit**: Strongly consider taking profit. Bird in hand.
- **Reversal signals appearing**: EXIT immediately. Don't hope. A -0.3% exit beats a -1.5% SL hit.
- **Time decay**: If a position hasn't moved in your favor after 10-15 min, the setup might be invalid. Consider exiting at small loss.

## 11. Risk Management Fundamentals
- **Never risk more than 1% of capital per trade**. This is non-negotiable.
- **Position sizing from SL distance**: Calculate position size based on where SL needs to be, not the other way around.
  - Wrong: "I want $50 position, SL at 0.3%"
  - Right: "SL needs to be at 0.8% (2x ATR), so my position size = 1% capital / (0.8% * leverage)"
- **Correlation risk**: Don't open 5 SHORT positions on 5 altcoins. They're all correlated. If BTC bounces, all 5 lose simultaneously.
- **Max drawdown protection**: If down 2% in a day, reduce position size or stop trading. The market isn't moving your way today.
