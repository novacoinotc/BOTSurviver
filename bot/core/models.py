"""Pydantic models for signals, orders, positions, trades, and memory."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ActionType(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"
    ADJUST = "ADJUST"
    HOLD = "HOLD"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# --- Market Snapshot sent to Claude ---

class MarketSnapshot(BaseModel):
    pair: str
    price: float
    change_1m: float = 0.0
    change_5m: float = 0.0
    change_1h: float = 0.0
    rsi_7: Optional[float] = None
    rsi_14: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_pct: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_signal: Optional[str] = None
    vwap: Optional[float] = None
    price_vs_vwap: Optional[str] = None
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None  # ATR as % of price
    volume_delta_5m: Optional[float] = None
    book_imbalance: Optional[float] = None
    funding_rate: Optional[float] = None
    # New scalping indicators
    stoch_rsi_k: Optional[float] = None
    stoch_rsi_d: Optional[float] = None
    adx: Optional[float] = None  # trend strength (>25 = trending)
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    mfi: Optional[float] = None  # money flow index (volume-weighted RSI)
    bb_width: Optional[float] = None
    bb_squeeze: Optional[bool] = None  # tight bands = breakout imminent
    sentiment: Optional[dict] = None
    fear_greed: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Claude Decision ---

class TradeDecision(BaseModel):
    action: ActionType
    pair: str
    direction: Optional[Direction] = None
    leverage: Optional[int] = None
    position_size_pct: Optional[float] = None  # % of capital
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.0  # 0-1


# --- Position ---

class Position(BaseModel):
    id: str
    pair: str
    direction: Direction
    entry_price: float
    current_price: float = 0.0
    quantity: float
    leverage: int
    stop_loss: float
    take_profit: float
    margin_used: float  # USDT locked as margin
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_fee: float = 0.0
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    entry_reasoning: str = ""
    entry_indicators: Optional[dict] = None


# --- Closed Trade ---

class Trade(BaseModel):
    id: str
    pair: str
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    pnl: float  # net PnL after fees
    pnl_pct: float
    entry_fee: float
    exit_fee: float
    margin_used: float
    hold_time_minutes: float
    opened_at: datetime
    closed_at: datetime = Field(default_factory=datetime.utcnow)
    entry_reasoning: str = ""
    exit_reasoning: str = ""
    entry_indicators: Optional[dict] = None
    exit_indicators: Optional[dict] = None
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    sentiment_score: Optional[int] = None


# --- Trade Memory ---

class TradeMemory(BaseModel):
    id: Optional[int] = None
    trade_id: str
    pair: str
    direction: Direction
    pnl: float
    pnl_pct: float
    leverage: int
    hold_time_minutes: float
    market_regime: MarketRegime
    indicators_at_entry: dict
    sentiment_score: Optional[int] = None
    claude_reasoning: str
    lesson_learned: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- Learned Rule ---

class LearnedRule(BaseModel):
    id: Optional[int] = None
    rule: str
    source_trades: list[str] = Field(default_factory=list)  # trade IDs
    confidence: float = 0.5
    times_applied: int = 0
    times_successful: int = 0
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# --- Parameter Change ---

class ParameterChange(BaseModel):
    id: Optional[int] = None
    param_name: str
    old_value: float
    new_value: float
    reasoning: str
    performance_before: Optional[dict] = None
    performance_after: Optional[dict] = None
    reverted: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- API Cost ---

class ApiCost(BaseModel):
    id: Optional[int] = None
    service: str  # claude_haiku, claude_sonnet, cryptopanic
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float
    purpose: str  # trade_decision, daily_review, optimization, sentiment
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- Daily Stats ---

class DailyStats(BaseModel):
    date: str  # YYYY-MM-DD
    starting_balance: float
    ending_balance: float
    pnl_gross: float
    pnl_net: float  # after all fees + API costs
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_fees: float
    total_api_costs: float
    max_drawdown_pct: float
    best_trade_pnl: float
    worst_trade_pnl: float
    avg_hold_time_minutes: float
    sharpe_ratio: Optional[float] = None
