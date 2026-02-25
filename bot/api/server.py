"""FastAPI server exposing bot data to the dashboard."""

import logging
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Scalping Bot API", version="1.0.0")

cors_origins = [o.strip() for o in settings.cors_origin.split(",") if o.strip()]
cors_origins.extend(["http://localhost:3000", "http://localhost:3006"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine reference (set by main.py)
_engine = None


def set_engine(engine):
    global _engine
    _engine = engine


def get_engine():
    assert _engine is not None, "Engine not initialized"
    return _engine


# --- Endpoints ---

@app.get("/api/status")
async def get_status():
    """Balance, positions, uptime, circuit breaker status."""
    engine = get_engine()
    return engine.get_status()


@app.get("/api/trades")
async def get_trades(
    pair: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Trade history with pagination and filters."""
    engine = get_engine()
    trades = await engine.db.get_trades(pair=pair, status=status, limit=limit, offset=offset)
    total = await engine.db.count_trades(status=status)
    return {"trades": trades, "total": total, "limit": limit, "offset": offset}


@app.get("/api/trades/{trade_id}")
async def get_trade_detail(trade_id: str):
    """Single trade with full reasoning and indicators."""
    engine = get_engine()
    trade = await engine.db.get_trade_by_id(trade_id)
    if not trade:
        return {"error": "Trade not found"}, 404

    # Get associated memory
    memories = await engine.db.find_similar_trades(trade["pair"], "", limit=100)
    memory = next((m for m in memories if m.get("trade_id") == trade_id), None)

    return {"trade": trade, "memory": memory}


@app.get("/api/memory")
async def get_memory(limit: int = Query(default=50, le=200)):
    """Lessons learned and patterns."""
    engine = get_engine()
    memories = await engine.memory.get_recent_memories(limit)
    rules = await engine.memory.get_active_rules()
    stats = await engine.memory.get_stats()
    return {"memories": memories, "rules": rules, "stats": stats}


@app.get("/api/analytics")
async def get_analytics():
    """Calculated metrics: Sharpe, drawdown, win rate, per pair."""
    engine = get_engine()
    daily_stats = await engine.db.get_daily_stats(limit=30)
    all_trades = await engine.db.get_trades(status="closed", limit=1000)

    # Compute analytics
    total_trades = len(all_trades)
    winning = [t for t in all_trades if t["pnl"] > 0]
    losing = [t for t in all_trades if t["pnl"] <= 0]
    win_rate = len(winning) / total_trades if total_trades > 0 else 0

    total_pnl = sum(t["pnl"] for t in all_trades)
    avg_win = sum(t["pnl"] for t in winning) / len(winning) if winning else 0
    avg_loss = sum(t["pnl"] for t in losing) / len(losing) if losing else 0
    profit_factor = abs(sum(t["pnl"] for t in winning) / sum(t["pnl"] for t in losing)) if losing and sum(t["pnl"] for t in losing) != 0 else 0

    # PnL by pair
    pnl_by_pair = {}
    for t in all_trades:
        pair = t["pair"]
        if pair not in pnl_by_pair:
            pnl_by_pair[pair] = {"pnl": 0, "trades": 0, "wins": 0}
        pnl_by_pair[pair]["pnl"] += t["pnl"]
        pnl_by_pair[pair]["trades"] += 1
        if t["pnl"] > 0:
            pnl_by_pair[pair]["wins"] += 1

    # PnL by hour
    pnl_by_hour = {}
    for t in all_trades:
        if t.get("closed_at"):
            hour = t["closed_at"][11:13] if isinstance(t["closed_at"], str) else "00"
            if hour not in pnl_by_hour:
                pnl_by_hour[hour] = 0
            pnl_by_hour[hour] += t["pnl"]

    # Sharpe ratio approximation
    import numpy as np
    if all_trades:
        returns = [t["pnl_pct"] for t in all_trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Max drawdown from daily stats
    max_dd = min((s.get("max_drawdown_pct", 0) for s in daily_stats), default=0)

    return {
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "pnl_by_pair": pnl_by_pair,
        "pnl_by_hour": dict(sorted(pnl_by_hour.items())),
        "daily_stats": daily_stats,
    }


@app.get("/api/params")
async def get_params():
    """Current parameters and change history."""
    engine = get_engine()
    current = await engine.db.get_current_params()
    history = await engine.db.get_param_history(limit=100)
    return {"current": current, "history": history}


@app.get("/api/costs")
async def get_costs(since: Optional[str] = None):
    """API cost breakdown."""
    engine = get_engine()
    by_service = await engine.db.get_costs_by_service(since=since)
    total = await engine.db.get_total_api_cost(since=since)
    recent = await engine.db.get_api_costs(limit=50)

    # Add VPS prorated cost
    vps_monthly = 6.0  # $6/month
    vps_daily = vps_monthly / 30

    equity_summary = engine.position_manager.get_equity_summary()
    trading_pnl = equity_summary["total_pnl"]

    return {
        "total_api_cost": round(total, 4),
        "by_service": by_service,
        "vps_monthly": vps_monthly,
        "vps_daily": round(vps_daily, 4),
        "trading_pnl": round(trading_pnl, 4),
        "net_pnl": round(trading_pnl - total - vps_daily, 4),
        "recent_costs": recent,
    }


@app.get("/api/market")
async def get_market():
    """Current market snapshot for all pairs."""
    engine = get_engine()
    snapshots = engine.market_analyzer.get_all_snapshots(engine.pairs)
    summary = engine.market_analyzer.get_market_summary(engine.pairs)

    snapshot_dicts = {}
    for pair, snap in snapshots.items():
        d = snap.model_dump(exclude_none=True)
        d["timestamp"] = d["timestamp"].isoformat()
        snapshot_dicts[pair] = d

    return {"snapshots": snapshot_dicts, "summary": summary}


@app.get("/api/positions")
async def get_positions():
    """Open positions with unrealized PnL."""
    engine = get_engine()
    return {
        "positions": engine.position_manager.get_open_positions(),
        **engine.position_manager.get_equity_summary(),
    }
