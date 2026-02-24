"use client";

import { useAgents, useStats } from "@/hooks/use-agents";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const COLORS = ["#22c55e", "#ef4444", "#eab308", "#3b82f6", "#a855f7"];

export default function AnalyticsPage() {
  const { data: stats } = useStats();
  const { data: agentsData } = useAgents();

  const agents = agentsData?.data ?? [];

  // Generation distribution
  const genData = Object.entries(stats?.agentsByGeneration ?? {}).map(
    ([gen, count]) => ({
      generation: `Gen ${gen}`,
      count,
    })
  );

  // Status distribution
  const statusData = [
    { name: "Alive", value: stats?.aliveAgents ?? 0 },
    { name: "Dead", value: stats?.deadAgents ?? 0 },
    { name: "Pending", value: stats?.pendingAgents ?? 0 },
  ].filter((d) => d.value > 0);

  // Top agents by crypto balance
  const topAgents = [...agents]
    .sort((a, b) => Number(b.cryptoBalance) - Number(a.cryptoBalance))
    .slice(0, 10)
    .map((a) => ({
      name: a.name,
      crypto: Number(a.cryptoBalance),
    }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Analytics</h1>
        <p className="text-muted-foreground text-sm">
          Ecosystem insights and statistics
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Total Agents</p>
            <p className="text-3xl font-bold">{stats?.totalAgents ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Ecosystem Balance</p>
            <p className="text-3xl font-bold text-yellow-400 font-mono">
              ${Number(stats?.totalEcosystemBalance ?? 0).toFixed(2)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Survival Rate</p>
            <p className="text-3xl font-bold text-green-400">
              {stats && stats.totalAgents > 0
                ? (
                    ((stats.aliveAgents) / stats.totalAgents) *
                    100
                  ).toFixed(0)
                : 0}
              %
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">Pending Requests</p>
            <p className="text-3xl font-bold text-blue-400">
              {stats?.pendingRequestsCount ?? 0}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Top Agents */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Top Agents by Crypto</CardTitle>
          </CardHeader>
          <CardContent>
            {topAgents.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={topAgents}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: "#999" }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis tick={{ fontSize: 11, fill: "#999" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                    }}
                  />
                  <Bar dataKey="crypto" fill="#22c55e" radius={[4, 4, 0, 0]} name="Crypto (USDT)" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-center py-12 text-muted-foreground">
                No data yet
              </p>
            )}
          </CardContent>
        </Card>

        {/* Status Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {statusData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {statusData.map((_, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[index % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-center py-12 text-muted-foreground">
                No data yet
              </p>
            )}
          </CardContent>
        </Card>

        {/* Generation Distribution */}
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle className="text-base">
              Agents by Generation
            </CardTitle>
          </CardHeader>
          <CardContent>
            {genData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={genData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="generation"
                    tick={{ fontSize: 12, fill: "#999" }}
                  />
                  <YAxis tick={{ fontSize: 12, fill: "#999" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1a1a",
                      border: "1px solid #333",
                    }}
                  />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-center py-12 text-muted-foreground">
                No data yet
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
