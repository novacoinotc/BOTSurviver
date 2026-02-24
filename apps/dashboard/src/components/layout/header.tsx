"use client";

import { useStats } from "@/hooks/use-agents";
import { Badge } from "@/components/ui/badge";
import { Bot, Skull, Clock, Wallet, Inbox } from "lucide-react";

export function Header() {
  const { data: stats } = useStats();

  return (
    <header className="h-14 border-b border-border bg-card flex items-center px-6 gap-6 sticky top-0 z-10">
      <div className="flex items-center gap-2 text-sm">
        <Bot className="w-4 h-4 text-green-400" />
        <span className="text-muted-foreground">Alive:</span>
        <Badge variant="secondary" className="bg-green-500/20 text-green-400">
          {stats?.aliveAgents ?? 0}
        </Badge>
      </div>

      <div className="flex items-center gap-2 text-sm">
        <Skull className="w-4 h-4 text-red-400" />
        <span className="text-muted-foreground">Dead:</span>
        <Badge variant="secondary" className="bg-red-500/20 text-red-400">
          {stats?.deadAgents ?? 0}
        </Badge>
      </div>

      <div className="flex items-center gap-2 text-sm">
        <Wallet className="w-4 h-4 text-green-400" />
        <span className="text-muted-foreground">Crypto:</span>
        <span className="font-mono text-green-400">
          {Number(stats?.totalCryptoBalance ?? 0).toFixed(2)} USDT
        </span>
      </div>

      <div className="flex items-center gap-2 text-sm">
        <Inbox className="w-4 h-4 text-blue-400" />
        <span className="text-muted-foreground">Pending:</span>
        <Badge variant="secondary" className="bg-blue-500/20 text-blue-400">
          {stats?.pendingRequestsCount ?? 0}
        </Badge>
      </div>

      <div className="ml-auto flex items-center gap-2 text-sm">
        <Clock className="w-4 h-4 text-muted-foreground" />
        <span className="text-muted-foreground font-mono">
          {stats?.totalAgents ?? 0} total agents
        </span>
      </div>
    </header>
  );
}
