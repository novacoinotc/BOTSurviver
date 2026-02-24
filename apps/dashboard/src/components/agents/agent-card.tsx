"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Agent } from "@/lib/api-client";
import { Bot, Clock, DollarSign, GitBranch, Wallet } from "lucide-react";

function getTimeRemaining(diesAt: string): string {
  const diff = new Date(diesAt).getTime() - Date.now();
  if (diff <= 0) return "EXPIRED";
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

function getStatusColor(status: string) {
  switch (status) {
    case "alive":
      return "bg-green-500/20 text-green-400 border-green-500/30";
    case "dead":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    default:
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
  }
}

export function AgentCard({ agent }: { agent: Agent }) {
  const cryptoBalance = Number(agent.cryptoBalance);
  const timeLeft = getTimeRemaining(agent.diesAt);

  return (
    <Link href={`/agents/${agent.id}`}>
      <Card className="hover:border-primary/50 transition-colors cursor-pointer">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <Bot className="w-4 h-4" />
              {agent.name}
            </CardTitle>
            <Badge variant="outline" className={getStatusColor(agent.status)}>
              {agent.status}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground flex items-center gap-1">
              <GitBranch className="w-3 h-3" />
              Gen {agent.generation}
            </span>
            <span className="text-muted-foreground flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {timeLeft}
            </span>
          </div>

          <div className="flex items-center gap-1.5 text-sm">
            <Wallet className="w-3.5 h-3.5 text-green-400" />
            <span
              className={`font-mono font-medium ${
                cryptoBalance <= 0
                  ? "text-red-400"
                  : cryptoBalance < 2
                    ? "text-yellow-400"
                    : "text-green-400"
              }`}
            >
              {cryptoBalance.toFixed(2)} USDT
            </span>
          </div>

          {agent.solanaAddress && (
            <p className="text-xs text-muted-foreground font-mono truncate">
              SOL: {agent.solanaAddress.slice(0, 8)}...{agent.solanaAddress.slice(-6)}
            </p>
          )}

          {agent.strategy && (
            <p className="text-xs text-muted-foreground line-clamp-2">
              {agent.strategy}
            </p>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
