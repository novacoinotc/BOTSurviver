"use client";

import { useState } from "react";
import useSWR from "swr";
import { api, Trade } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ArrowLeftRight, ChevronLeft, ChevronRight } from "lucide-react";

function TradeCard({ trade, onClick }: { trade: Trade; onClick: () => void }) {
  const isPnlPositive = trade.pnl >= 0;
  return (
    <div
      className="p-3 border border-border rounded-lg space-y-2 cursor-pointer active:bg-accent/50"
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono font-medium text-sm">{trade.pair}</span>
          <Badge
            variant="secondary"
            className={`text-xs ${trade.direction === "LONG" ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}
          >
            {trade.direction}
          </Badge>
        </div>
        <Badge variant="secondary" className="text-xs">{trade.status}</Badge>
      </div>
      <div className="flex items-center justify-between text-sm">
        <div className="text-muted-foreground">
          <span className="font-mono">${trade.entry_price.toFixed(2)}</span>
          <span className="mx-1">→</span>
          <span className="font-mono">{trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : "—"}</span>
        </div>
        <span className={`font-mono font-medium ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
          {isPnlPositive ? "+" : ""}${trade.pnl.toFixed(2)}
        </span>
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{trade.leverage}x</span>
        <span className={`font-mono ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
          {(trade.pnl_pct * 100).toFixed(2)}%
        </span>
        <span>{trade.hold_time_minutes.toFixed(0)}m</span>
      </div>
    </div>
  );
}

export default function TradesPage() {
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState<"all" | "open" | "closed">("all");
  const limit = 20;

  const { data } = useSWR(
    `trades-${filter}-${page}`,
    () =>
      api.getTrades({
        status: filter === "all" ? undefined : filter,
        limit,
        offset: page * limit,
      }),
    { refreshInterval: 10000 }
  );

  const [selectedTrade, setSelectedTrade] = useState<string | null>(null);
  const { data: tradeDetail } = useSWR(
    selectedTrade ? `trade-${selectedTrade}` : null,
    () => (selectedTrade ? api.getTrade(selectedTrade) : null)
  );

  const trades = data?.trades ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / limit);

  return (
    <div className="space-y-4 md:space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl md:text-2xl font-bold flex items-center gap-2">
          <ArrowLeftRight className="w-5 h-5 md:w-6 md:h-6" />
          Trades
        </h1>
        <div className="flex gap-1 md:gap-2">
          {(["all", "open", "closed"] as const).map((f) => (
            <Button
              key={f}
              variant={filter === f ? "default" : "outline"}
              size="sm"
              className="text-xs md:text-sm px-2 md:px-3"
              onClick={() => { setFilter(f); setPage(0); }}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </Button>
          ))}
        </div>
      </div>

      {/* Mobile: Card layout */}
      <div className="md:hidden space-y-2">
        {trades.map((trade) => (
          <TradeCard
            key={trade.id}
            trade={trade}
            onClick={() => setSelectedTrade(trade.id)}
          />
        ))}
        {trades.length === 0 && (
          <p className="text-center text-muted-foreground py-8">No trades found</p>
        )}
      </div>

      {/* Desktop: Table layout */}
      <Card className="hidden md:block">
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Pair</TableHead>
                <TableHead>Direction</TableHead>
                <TableHead>Entry</TableHead>
                <TableHead>Exit</TableHead>
                <TableHead>Leverage</TableHead>
                <TableHead>PnL</TableHead>
                <TableHead>PnL %</TableHead>
                <TableHead>Hold</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades.map((trade) => {
                const isPnlPositive = trade.pnl >= 0;
                return (
                  <TableRow
                    key={trade.id}
                    className="cursor-pointer hover:bg-accent/50"
                    onClick={() => setSelectedTrade(trade.id)}
                  >
                    <TableCell className="font-mono font-medium">{trade.pair}</TableCell>
                    <TableCell>
                      <Badge
                        variant="secondary"
                        className={trade.direction === "LONG"
                          ? "bg-green-500/20 text-green-400"
                          : "bg-red-500/20 text-red-400"
                        }
                      >
                        {trade.direction}
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono">${trade.entry_price.toFixed(2)}</TableCell>
                    <TableCell className="font-mono">
                      {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : "-"}
                    </TableCell>
                    <TableCell>{trade.leverage}x</TableCell>
                    <TableCell className={`font-mono ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
                      {isPnlPositive ? "+" : ""}${trade.pnl.toFixed(4)}
                    </TableCell>
                    <TableCell className={`font-mono ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
                      {(trade.pnl_pct * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell>{trade.hold_time_minutes.toFixed(0)}m</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{trade.status}</Badge>
                    </TableCell>
                  </TableRow>
                );
              })}
              {trades.length === 0 && (
                <TableRow>
                  <TableCell colSpan={9} className="text-center text-muted-foreground py-8">
                    No trades found
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-xs md:text-sm text-muted-foreground">
            {page * limit + 1}-{Math.min((page + 1) * limit, total)} of {total}
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page >= totalPages - 1}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Trade Detail */}
      {selectedTrade && tradeDetail && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between text-base md:text-lg">
              <span>Trade: {tradeDetail.trade.pair}</span>
              <Button variant="ghost" size="sm" onClick={() => setSelectedTrade(null)}>
                Close
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Entry Reasoning</p>
                <p className="mt-1">{tradeDetail.trade.entry_reasoning || "N/A"}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Exit Reasoning</p>
                <p className="mt-1">{tradeDetail.trade.exit_reasoning || "N/A"}</p>
              </div>
            </div>
            {tradeDetail.memory && (
              <div>
                <p className="text-muted-foreground text-sm">Lesson Learned</p>
                <p className="mt-1 text-sm p-3 bg-accent/50 rounded-lg">
                  {tradeDetail.memory.lesson_learned || "Pending review..."}
                </p>
                {tradeDetail.memory.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {tradeDetail.memory.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            )}
            <div>
              <p className="text-muted-foreground text-sm">Market Regime</p>
              <Badge variant="secondary" className="mt-1">{tradeDetail.trade.market_regime}</Badge>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
