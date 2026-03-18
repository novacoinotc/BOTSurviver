"use client";

import { useStatus, usePositions } from "@/hooks/use-agents";
import useSWR from "swr";
import { api } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  BarChart3,
  Target,
  ShieldAlert,
  Clock,
  Activity,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function OverviewPage() {
  const { data: status } = useStatus();
  const { data: positions } = usePositions();
  const { data: analytics } = useSWR("analytics", () => api.getAnalytics(), {
    refreshInterval: 30000,
  });
  const { data: trades } = useSWR("recent-trades", () => api.getTrades({ limit: 5 }), {
    refreshInterval: 10000,
  });

  const pnl = status?.total_pnl ?? 0;
  const isProfit = pnl >= 0;

  // Build equity curve from daily stats
  const equityCurve =
    analytics?.daily_stats
      ?.slice()
      .reverse()
      .map((d) => ({
        date: d.date,
        equity: d.ending_balance,
        pnl: d.pnl_net,
      })) ?? [];

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold">Overview</h1>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Balance</CardTitle>
            <Wallet className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-lg md:text-2xl font-bold font-mono">
              ${(status?.total_equity ?? 0).toFixed(2)}
            </p>
            <p className="text-xs text-muted-foreground">
              Free: ${(status?.free_margin ?? 0).toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total PnL</CardTitle>
            {isProfit ? (
              <TrendingUp className="w-4 h-4 text-green-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
          </CardHeader>
          <CardContent>
            <p className={`text-lg md:text-2xl font-bold font-mono ${isProfit ? "text-green-400" : "text-red-400"}`}>
              {isProfit ? "+" : ""}${pnl.toFixed(2)}
            </p>
            <p className="text-xs text-muted-foreground">
              {(status?.total_pnl_pct ?? 0).toFixed(2)}% return
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Win Rate</CardTitle>
            <Target className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-lg md:text-2xl font-bold font-mono">
              {((analytics?.win_rate ?? 0) * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">
              {analytics?.winning_trades ?? 0}W / {analytics?.losing_trades ?? 0}L
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Drawdown</CardTitle>
            <BarChart3 className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <p className="text-lg md:text-2xl font-bold font-mono text-red-400">
              {(status?.drawdown_pct ?? 0).toFixed(2)}%
            </p>
            <p className="text-xs text-muted-foreground">
              Max: {(analytics?.max_drawdown_pct ?? 0).toFixed(2)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Equity Curve */}
      {equityCurve.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Equity Curve</CardTitle>
          </CardHeader>
          <CardContent className="h-52 md:h-64 px-1 md:px-6">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={equityCurve}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={10} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="equity"
                  stroke="hsl(142, 71%, 45%)"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <div className="grid md:grid-cols-2 gap-6">
        {/* Open Positions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Open Positions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {(positions?.positions ?? []).length === 0 ? (
              <p className="text-sm text-muted-foreground">No open positions</p>
            ) : (
              <div className="space-y-3">
                {positions?.positions.map((pos) => {
                  const isPnlPositive = pos.unrealized_pnl >= 0;
                  return (
                    <div key={pos.id} className="p-3 rounded-lg bg-accent/50 space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="font-mono font-medium text-sm">{pos.pair}</span>
                          <Badge
                            variant="secondary"
                            className={`text-xs shrink-0 ${pos.direction === "LONG"
                              ? "bg-green-500/20 text-green-400"
                              : "bg-red-500/20 text-red-400"
                            }`}
                          >
                            {pos.direction} {pos.leverage}x
                          </Badge>
                        </div>
                        <p className={`font-mono font-medium text-sm shrink-0 ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
                          {isPnlPositive ? "+" : ""}${pos.unrealized_pnl.toFixed(2)}
                        </p>
                      </div>
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Entry: ${pos.entry_price.toFixed(2)}</span>
                        <span>{pos.hold_time_minutes.toFixed(0)}m</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Recent Trades
            </CardTitle>
          </CardHeader>
          <CardContent>
            {(trades?.trades ?? []).length === 0 ? (
              <p className="text-sm text-muted-foreground">No trades yet</p>
            ) : (
              <div className="space-y-3">
                {trades?.trades.map((trade) => {
                  const isPnlPositive = trade.pnl >= 0;
                  return (
                    <div key={trade.id} className="p-3 rounded-lg bg-accent/50 space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="font-mono font-medium text-sm">{trade.pair}</span>
                          <Badge
                            variant="secondary"
                            className={`text-xs shrink-0 ${trade.direction === "LONG"
                              ? "bg-green-500/20 text-green-400"
                              : "bg-red-500/20 text-red-400"
                            }`}
                          >
                            {trade.direction}
                          </Badge>
                        </div>
                        <span className={`font-mono font-medium text-sm shrink-0 ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
                          {isPnlPositive ? "+" : ""}${trade.pnl.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{trade.hold_time_minutes.toFixed(0)}m | {trade.leverage}x</span>
                        <span>{(trade.pnl_pct * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Circuit Breaker */}
      {status?.circuit_breaker?.active && (
        <Card className="border-red-500/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-400">
              <ShieldAlert className="w-5 h-5" />
              Circuit Breaker Active
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{status.circuit_breaker.reason}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
