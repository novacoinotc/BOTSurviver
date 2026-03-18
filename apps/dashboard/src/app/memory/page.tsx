"use client";

import useSWR from "swr";
import { api } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, BookOpen, Lightbulb, CheckCircle, XCircle } from "lucide-react";

export default function MemoryPage() {
  const { data } = useSWR("memory", () => api.getMemory(100), { refreshInterval: 30000 });

  const memories = data?.memories ?? [];
  const rules = data?.rules ?? [];
  const stats = data?.stats;

  return (
    <div className="space-y-6">
      <h1 className="text-xl md:text-2xl font-bold flex items-center gap-2">
        <Brain className="w-5 h-5 md:w-6 md:h-6" />
        Bot Memory
      </h1>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 md:gap-4">
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Total Memories</p>
            <p className="text-xl md:text-2xl font-bold font-mono">{stats?.total_memories ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">With Lessons</p>
            <p className="text-xl md:text-2xl font-bold font-mono">{stats?.memories_with_lessons ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Winning</p>
            <p className="text-2xl font-bold font-mono text-green-400">{stats?.winning_trades ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Losing</p>
            <p className="text-2xl font-bold font-mono text-red-400">{stats?.losing_trades ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 px-3 md:px-6">
            <p className="text-xs md:text-sm text-muted-foreground">Active Rules</p>
            <p className="text-2xl font-bold font-mono text-blue-400">{stats?.active_rules ?? 0}</p>
          </CardContent>
        </Card>
      </div>

      {/* Learned Rules */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-400" />
            Learned Rules
          </CardTitle>
        </CardHeader>
        <CardContent>
          {rules.length === 0 ? (
            <p className="text-sm text-muted-foreground">No rules learned yet. The bot will discover patterns after enough trades.</p>
          ) : (
            <div className="space-y-3">
              {rules.map((rule) => {
                const successRate = rule.times_applied > 0
                  ? (rule.times_successful / rule.times_applied * 100).toFixed(0)
                  : "N/A";
                return (
                  <div key={rule.id} className="p-3 rounded-lg bg-accent/50">
                    <p className="text-sm">{rule.rule}</p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                      <span>Confidence: {(rule.confidence * 100).toFixed(0)}%</span>
                      <span>Applied: {rule.times_applied}x</span>
                      <span>Success: {successRate}%</span>
                      <Badge variant="secondary" className={rule.active ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}>
                        {rule.active ? "Active" : "Disabled"}
                      </Badge>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trade Memories */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Trade Memories & Lessons
          </CardTitle>
        </CardHeader>
        <CardContent>
          {memories.length === 0 ? (
            <p className="text-sm text-muted-foreground">No trade memories yet</p>
          ) : (
            <div className="space-y-4">
              {memories.map((mem) => {
                const isPnlPositive = mem.pnl > 0;
                return (
                  <div key={mem.id} className="p-3 md:p-4 rounded-lg border border-border">
                    <div className="flex flex-wrap items-center gap-2">
                      {isPnlPositive ? (
                        <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-400 shrink-0" />
                      )}
                      <span className="font-mono font-medium text-sm">{mem.pair}</span>
                      <Badge
                        variant="secondary"
                        className={`text-xs ${mem.direction === "LONG"
                          ? "bg-green-500/20 text-green-400"
                          : "bg-red-500/20 text-red-400"
                        }`}
                      >
                        {mem.direction}
                      </Badge>
                      <span className={`font-mono text-sm ml-auto ${isPnlPositive ? "text-green-400" : "text-red-400"}`}>
                        {isPnlPositive ? "+" : ""}${mem.pnl.toFixed(2)} ({(mem.pnl_pct * 100).toFixed(2)}%)
                      </span>
                    </div>

                    {mem.claude_reasoning && (
                      <div className="mt-2">
                        <p className="text-xs text-muted-foreground">Reasoning</p>
                        <p className="text-sm mt-1">{mem.claude_reasoning}</p>
                      </div>
                    )}

                    {mem.lesson_learned && (
                      <div className="mt-2 p-2 bg-yellow-500/10 rounded border border-yellow-500/20">
                        <p className="text-xs text-yellow-400">Lesson Learned</p>
                        <p className="text-sm mt-1">{mem.lesson_learned}</p>
                      </div>
                    )}

                    {mem.tags.length > 0 && (
                      <div className="flex gap-1 mt-2">
                        {mem.tags.map((tag) => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    )}

                    <p className="text-xs text-muted-foreground mt-2">
                      {mem.leverage}x | {mem.hold_time_minutes.toFixed(0)}m | {new Date(mem.created_at).toLocaleString()}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
