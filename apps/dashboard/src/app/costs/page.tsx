"use client";

import useSWR from "swr";
import { api } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { DollarSign, TrendingUp, TrendingDown, Server, Bot, Newspaper } from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const SERVICE_COLORS: Record<string, string> = {
  claude_haiku: "hsl(210, 70%, 55%)",
  claude_sonnet: "hsl(280, 70%, 55%)",
  cryptopanic: "hsl(30, 70%, 55%)",
};

const SERVICE_ICONS: Record<string, typeof Bot> = {
  claude_haiku: Bot,
  claude_sonnet: Bot,
  cryptopanic: Newspaper,
};

export default function CostsPage() {
  const { data } = useSWR("costs", () => api.getCosts(), { refreshInterval: 60000 });

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Loading costs...</p>
      </div>
    );
  }

  const isNetPositive = data.net_pnl >= 0;

  // Pie chart data
  const pieData = data.by_service.map((s) => ({
    name: s.service,
    value: s.total_cost,
  }));
  pieData.push({ name: "VPS (daily)", value: data.vps_daily });

  return (
    <div className="space-y-4 md:space-y-6">
      <h1 className="text-xl md:text-2xl font-bold flex items-center gap-2">
        <DollarSign className="w-5 h-5 md:w-6 md:h-6" />
        Costs
      </h1>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 px-3 md:px-6">
            <CardTitle className="text-xs md:text-sm font-medium text-muted-foreground">Trading PnL</CardTitle>
            {data.trading_pnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-green-400 shrink-0" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400 shrink-0" />
            )}
          </CardHeader>
          <CardContent className="px-3 md:px-6">
            <p className={`text-lg md:text-2xl font-bold font-mono ${data.trading_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
              {data.trading_pnl >= 0 ? "+" : ""}${data.trading_pnl.toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 px-3 md:px-6">
            <CardTitle className="text-xs md:text-sm font-medium text-muted-foreground">API Costs</CardTitle>
            <Bot className="w-4 h-4 text-muted-foreground shrink-0" />
          </CardHeader>
          <CardContent className="px-3 md:px-6">
            <p className="text-lg md:text-2xl font-bold font-mono text-red-400">
              -${data.total_api_cost.toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2 px-3 md:px-6">
            <CardTitle className="text-xs md:text-sm font-medium text-muted-foreground">VPS/mo</CardTitle>
            <Server className="w-4 h-4 text-muted-foreground shrink-0" />
          </CardHeader>
          <CardContent className="px-3 md:px-6">
            <p className="text-lg md:text-2xl font-bold font-mono">
              ${data.vps_monthly.toFixed(0)}
            </p>
          </CardContent>
        </Card>

        <Card className={isNetPositive ? "border-green-500/30" : "border-red-500/30"}>
          <CardHeader className="flex flex-row items-center justify-between pb-2 px-3 md:px-6">
            <CardTitle className="text-xs md:text-sm font-medium text-muted-foreground">Net PnL</CardTitle>
            {isNetPositive ? (
              <TrendingUp className="w-4 h-4 text-green-400 shrink-0" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400 shrink-0" />
            )}
          </CardHeader>
          <CardContent className="px-3 md:px-6">
            <p className={`text-lg md:text-2xl font-bold font-mono ${isNetPositive ? "text-green-400" : "text-red-400"}`}>
              {isNetPositive ? "+" : ""}${data.net_pnl.toFixed(2)}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid md:grid-cols-2 gap-4 md:gap-6">
        {/* Cost Breakdown Pie */}
        <Card>
          <CardHeader className="px-3 md:px-6">
            <CardTitle className="text-sm md:text-base">Cost Breakdown</CardTitle>
          </CardHeader>
          <CardContent className="h-64 md:h-72 px-1 md:px-6">
            {pieData.length > 0 && pieData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={SERVICE_COLORS[entry.name] || `hsl(${index * 90}, 50%, 50%)`}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                    formatter={(value) => [`$${Number(value).toFixed(4)}`, "Cost"]}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground text-center pt-8">No cost data yet</p>
            )}
          </CardContent>
        </Card>

        {/* Service Details */}
        <Card>
          <CardHeader className="px-3 md:px-6">
            <CardTitle className="text-sm md:text-base">Cost by Service</CardTitle>
          </CardHeader>
          <CardContent className="px-3 md:px-6">
            <div className="space-y-3">
              {data.by_service.map((service) => {
                const Icon = SERVICE_ICONS[service.service] || DollarSign;
                return (
                  <div key={service.service} className="p-3 rounded-lg bg-accent/50">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Icon className="w-4 h-4 shrink-0" />
                        <span className="font-medium text-sm">{service.service}</span>
                      </div>
                      <span className="font-mono font-medium text-sm">${service.total_cost.toFixed(4)}</span>
                    </div>
                    <div className="flex flex-wrap gap-2 md:gap-4 mt-2 text-xs text-muted-foreground">
                      <span>{service.call_count} calls</span>
                      <span>{(service.total_tokens_in / 1000).toFixed(1)}K in</span>
                      <span>{(service.total_tokens_out / 1000).toFixed(1)}K out</span>
                    </div>
                  </div>
                );
              })}

              <div className="p-3 rounded-lg bg-accent/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Server className="w-4 h-4 shrink-0" />
                    <span className="font-medium text-sm">VPS</span>
                  </div>
                  <span className="font-mono font-medium text-sm">${data.vps_monthly.toFixed(2)}/mo</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent API Calls - mobile cards, desktop table */}
      <Card>
        <CardHeader className="px-3 md:px-6">
          <CardTitle className="text-sm md:text-base">Recent API Calls</CardTitle>
        </CardHeader>
        <CardContent className="px-3 md:px-6">
          {data.recent_costs.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">No API calls recorded yet</p>
          ) : (
            <>
              {/* Mobile: card layout */}
              <div className="md:hidden space-y-2">
                {data.recent_costs.slice(0, 20).map((cost) => (
                  <div key={cost.id} className="p-2 rounded bg-accent/30 text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <Badge variant="secondary" className="text-xs">{cost.service}</Badge>
                      <span className="font-mono">${cost.cost_usd.toFixed(6)}</span>
                    </div>
                    <div className="flex items-center justify-between text-muted-foreground">
                      <span>{cost.purpose}</span>
                      <span>{new Date(cost.created_at).toLocaleTimeString()}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Desktop: table */}
              <div className="hidden md:block overflow-x-auto -mx-6">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="py-2 px-6 font-medium text-muted-foreground">Time</th>
                      <th className="py-2 px-2 font-medium text-muted-foreground">Service</th>
                      <th className="py-2 px-2 font-medium text-muted-foreground">Purpose</th>
                      <th className="py-2 px-2 font-medium text-muted-foreground">Tokens</th>
                      <th className="py-2 px-6 font-medium text-muted-foreground">Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent_costs.slice(0, 30).map((cost) => (
                      <tr key={cost.id} className="border-b border-border/50">
                        <td className="py-2 px-6">{new Date(cost.created_at).toLocaleTimeString()}</td>
                        <td className="py-2 px-2"><Badge variant="secondary">{cost.service}</Badge></td>
                        <td className="py-2 px-2">{cost.purpose}</td>
                        <td className="py-2 px-2 font-mono">
                          {cost.tokens_in + cost.tokens_out > 0 ? `${cost.tokens_in}/${cost.tokens_out}` : "-"}
                        </td>
                        <td className="py-2 px-6 font-mono">${cost.cost_usd.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
