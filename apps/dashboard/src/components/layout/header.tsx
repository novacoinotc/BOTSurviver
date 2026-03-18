"use client";

import { useStatus } from "@/hooks/use-agents";
import { Badge } from "@/components/ui/badge";
import { Wallet, TrendingUp, TrendingDown, Activity, ShieldAlert, Clock } from "lucide-react";

export function Header() {
  const { data: status } = useStatus();

  const pnl = status?.total_pnl ?? 0;
  const pnlPct = status?.total_pnl_pct ?? 0;
  const isProfit = pnl >= 0;

  return (
    <header className="border-b border-border bg-card sticky top-0 z-10 px-4 md:px-6 py-2 md:py-0 md:h-14">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 md:gap-6 md:h-14 text-sm pl-10 md:pl-0">
        <div className="flex items-center gap-1.5">
          <Wallet className="w-4 h-4 text-green-400 shrink-0" />
          <span className="text-muted-foreground hidden sm:inline">Equity:</span>
          <span className="font-mono font-medium">
            ${(status?.total_equity ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {isProfit ? (
            <TrendingUp className="w-4 h-4 text-green-400 shrink-0" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-400 shrink-0" />
          )}
          <span className="text-muted-foreground hidden sm:inline">PnL:</span>
          <span className={`font-mono font-medium ${isProfit ? "text-green-400" : "text-red-400"}`}>
            {isProfit ? "+" : ""}${pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({pnlPct.toFixed(2)}%)
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <Activity className="w-4 h-4 text-blue-400 shrink-0" />
          <Badge variant="secondary" className="bg-blue-500/20 text-blue-400">
            {status?.open_positions ?? 0}
          </Badge>
        </div>

        {status?.circuit_breaker?.active && (
          <div className="flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-red-400 shrink-0" />
            <Badge variant="secondary" className="bg-red-500/20 text-red-400 text-xs">
              Circuit Breaker
            </Badge>
          </div>
        )}

        <div className="md:ml-auto flex items-center gap-1.5">
          <Clock className="w-4 h-4 text-muted-foreground shrink-0" />
          <span className="text-muted-foreground font-mono text-xs md:text-sm">
            {status?.mode === "paper" ? "PAPER" : "LIVE"} | {status?.market_regime ?? "..."}
          </span>
          <Badge
            variant="secondary"
            className={`text-xs ${status?.ws_connected ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}
          >
            {status?.ws_connected ? "Connected" : "Disconnected"}
          </Badge>
        </div>
      </div>
    </header>
  );
}
