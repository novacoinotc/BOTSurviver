"use client";

import { useAgents } from "@/hooks/use-agents";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Bot, Wallet } from "lucide-react";
import Link from "next/link";
import type { Agent } from "@/lib/api-client";

function AgentNode({ agent, allAgents }: { agent: Agent; allAgents: Agent[] }) {
  const children = allAgents.filter((a) => a.parentId === agent.id);
  const cryptoBalance = Number(agent.cryptoBalance);

  return (
    <div className="flex flex-col items-center">
      <Link href={`/agents/${agent.id}`}>
        <Card
          className={`w-52 hover:border-primary/50 transition-colors cursor-pointer ${
            agent.status === "dead" ? "opacity-50" : ""
          }`}
        >
          <CardContent className="p-3 text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <Bot className="w-4 h-4" />
              <span className="font-medium text-sm">{agent.name}</span>
            </div>
            <div className="flex items-center justify-center gap-2 mb-1">
              <Badge
                variant="outline"
                className={
                  agent.status === "alive"
                    ? "bg-green-500/20 text-green-400 text-xs"
                    : "bg-red-500/20 text-red-400 text-xs"
                }
              >
                {agent.status}
              </Badge>
            </div>
            <div className="flex items-center justify-center gap-1 text-xs font-mono">
              <span className={`flex items-center gap-0.5 ${
                cryptoBalance <= 0 ? "text-red-400" : cryptoBalance < 2 ? "text-yellow-400" : "text-green-400"
              }`}>
                <Wallet className="w-3 h-3" />{cryptoBalance.toFixed(2)} USDT
              </span>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Gen {agent.generation}
            </p>
          </CardContent>
        </Card>
      </Link>

      {children.length > 0 && (
        <>
          <div className="w-px h-6 bg-border" />
          <div className="flex gap-4 relative">
            {children.length > 1 && (
              <div className="absolute top-0 left-1/2 -translate-x-1/2 h-px bg-border"
                   style={{ width: `${(children.length - 1) * 208}px` }} />
            )}
            {children.map((child) => (
              <div key={child.id} className="flex flex-col items-center">
                <div className="w-px h-6 bg-border" />
                <AgentNode agent={child} allAgents={allAgents} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function FamilyTreePage() {
  const { data } = useAgents();
  const agents = data?.data ?? [];
  const roots = agents.filter((a) => !a.parentId);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Family Tree</h1>
        <p className="text-muted-foreground text-sm">
          Visualize agent lineage and relationships
        </p>
      </div>

      <div className="overflow-auto pb-8">
        <div className="flex gap-8 justify-center min-w-max p-8">
          {roots.map((root) => (
            <AgentNode key={root.id} agent={root} allAgents={agents} />
          ))}
        </div>

        {agents.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            No agents yet
          </div>
        )}
      </div>
    </div>
  );
}
