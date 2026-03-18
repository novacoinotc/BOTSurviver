"use client";

import useSWR from "swr";
import { api } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function AnalyticsPage() {
  const { data } = useSWR("analytics", () => api.getAnalytics(), { refreshInterval: 30000 });

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Loading analytics...</p>
      </div>
    );
  }

  // PnL by pair chart data
  const pairData = Object.entries(data.pnl_by_pair)
    .map(([pair, info]) => ({
      pair: pair.replace("USDT", ""),
      pnl: info.pnl,
      trades: info.trades,
      winRate: info.trades > 0 ? ((info.wins / info.trades) * 100) : 0,
    }))
    .sort((a, b) => b.pnl - a.pnl);

  // PnL by hour chart data
  const hourData = Object.entries(data.pnl_by_hour).map(([hour, pnl]) => ({
    hour: `${hour}:00`,
    pnl,
  }));

  // Daily PnL
  const dailyPnl = data.daily_stats
    ?.slice()
    .reverse()
    .map((d) => ({
      date: d.date,
      pnl: d.pnl_net,
      trades: d.total_trades,
    })) ?? [];

  return (
    <div className="space-y-4 md:space-y-6">
      <h1 className="text-xl md:text-2xl font-bold flex items-center gap-2">
        <BarChart3 className="w-5 h-5 md:w-6 md:h-6" />
        Analytics
      </h1>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 md:gap-4">
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Total Trades</p>
            <p className="text-xl md:text-2xl font-bold font-mono">{data.total_trades}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Win Rate</p>
            <p className="text-xl md:text-2xl font-bold font-mono">{(data.win_rate * 100).toFixed(1)}%</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Profit Factor</p>
            <p className="text-xl md:text-2xl font-bold font-mono">{data.profit_factor.toFixed(2)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Sharpe Ratio</p>
            <p className="text-xl md:text-2xl font-bold font-mono">{data.sharpe_ratio.toFixed(2)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Max Drawdown</p>
            <p className="text-xl md:text-2xl font-bold font-mono text-red-400">{data.max_drawdown_pct.toFixed(2)}%</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Total PnL</p>
            <p className={`text-lg md:text-xl font-bold font-mono ${data.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
              {data.total_pnl >= 0 ? "+" : ""}${data.total_pnl.toFixed(2)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Avg Win</p>
            <p className="text-lg md:text-xl font-bold font-mono text-green-400">${data.avg_win.toFixed(2)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Avg Loss</p>
            <p className="text-lg md:text-xl font-bold font-mono text-red-400">${data.avg_loss.toFixed(2)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Win/Loss</p>
            <p className="text-lg md:text-xl font-bold font-mono">
              {data.winning_trades}/{data.losing_trades}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-4 md:gap-6">
        {/* PnL by Pair */}
        <Card>
          <CardHeader className="px-3 md:px-6">
            <CardTitle className="text-sm md:text-base">PnL by Pair</CardTitle>
          </CardHeader>
          <CardContent className="h-72 px-1 md:px-6">
            {pairData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={pairData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={10} />
                  <YAxis type="category" dataKey="pair" width={40} stroke="hsl(var(--muted-foreground))" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                    formatter={(value) => [`$${Number(value).toFixed(2)}`, "PnL"]}
                  />
                  <Bar dataKey="pnl" fill="hsl(142, 71%, 45%)">
                    {pairData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.pnl >= 0 ? "hsl(142, 71%, 45%)" : "hsl(0, 72%, 51%)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground text-center pt-8">No data yet</p>
            )}
          </CardContent>
        </Card>

        {/* Daily PnL */}
        <Card>
          <CardHeader className="px-3 md:px-6">
            <CardTitle className="text-sm md:text-base">Daily PnL</CardTitle>
          </CardHeader>
          <CardContent className="h-72 px-1 md:px-6">
            {dailyPnl.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dailyPnl}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={10} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="pnl">
                    {dailyPnl.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.pnl >= 0 ? "hsl(142, 71%, 45%)" : "hsl(0, 72%, 51%)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground text-center pt-8">No data yet</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* PnL by Hour */}
      {hourData.length > 0 && (
        <Card>
          <CardHeader className="px-3 md:px-6">
            <CardTitle className="text-sm md:text-base">PnL by Hour (UTC)</CardTitle>
          </CardHeader>
          <CardContent className="h-64 px-1 md:px-6">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="hour" stroke="hsl(var(--muted-foreground))" fontSize={10} />
                <YAxis stroke="hsl(var(--muted-foreground))" fontSize={10} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="pnl">
                  {hourData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.pnl >= 0 ? "hsl(142, 71%, 45%)" : "hsl(0, 72%, 51%)"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Per-Pair Detail */}
      <Card>
        <CardHeader className="px-3 md:px-6">
          <CardTitle className="text-sm md:text-base">Performance by Pair</CardTitle>
        </CardHeader>
        <CardContent className="px-3 md:px-6">
          <div className="space-y-2">
            {pairData.map((p) => (
              <div key={p.pair} className="flex items-center justify-between gap-2 p-2 rounded bg-accent/30">
                <span className="font-mono font-medium text-sm">{p.pair}</span>
                <div className="flex items-center gap-2 md:gap-4 text-xs md:text-sm">
                  <span className="text-muted-foreground">{p.trades}t</span>
                  <span className="text-muted-foreground hidden sm:inline">WR: {p.winRate.toFixed(0)}%</span>
                  <span className={`font-mono ${p.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {p.pnl >= 0 ? "+" : ""}${p.pnl.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
