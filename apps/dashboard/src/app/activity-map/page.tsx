"use client";

import { useEffect, useState, useMemo } from "react";
import { useAgents } from "@/hooks/use-agents";
import { api, type AgentLog, type Agent } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Brain,
  Lightbulb,
  Send,
  ArrowRightLeft,
  AlertTriangle,
  MessageSquare,
  Bot,
  ChevronDown,
  ChevronRight,
  RefreshCw,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Category classification
// ---------------------------------------------------------------------------

type Category = "pensamientos" | "estrategias" | "solicitudes" | "transacciones" | "errores" | "controlador";

const CATEGORY_META: Record<
  Category,
  { label: string; color: string; bgColor: string; borderColor: string; icon: React.ElementType }
> = {
  pensamientos: {
    label: "Pensamientos",
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/30",
    icon: Brain,
  },
  estrategias: {
    label: "Estrategias",
    color: "text-orange-400",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/30",
    icon: Lightbulb,
  },
  solicitudes: {
    label: "Solicitudes",
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    borderColor: "border-green-500/30",
    icon: Send,
  },
  transacciones: {
    label: "Transacciones",
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/30",
    icon: ArrowRightLeft,
  },
  errores: {
    label: "Errores",
    color: "text-red-400",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
    icon: AlertTriangle,
  },
  controlador: {
    label: "Controlador",
    color: "text-sky-400",
    bgColor: "bg-sky-500/10",
    borderColor: "border-sky-500/30",
    icon: MessageSquare,
  },
};

function classifyLog(log: AgentLog): Category {
  const level = log.level.toLowerCase();
  const msg = log.message.toLowerCase();

  if (level === "error") return "errores";
  if (level === "controller" || msg.includes("controller") || msg.includes("message from")) return "controlador";
  if (msg.includes("strategy") || msg.includes("estrategia") || msg.includes("plan")) return "estrategias";
  if (
    msg.includes("request") ||
    msg.includes("solicitud") ||
    msg.includes("approved") ||
    msg.includes("denied") ||
    msg.includes("aprobad") ||
    msg.includes("rechazad")
  )
    return "solicitudes";
  if (
    msg.includes("transaction") ||
    msg.includes("transfer") ||
    msg.includes("income") ||
    msg.includes("payment") ||
    msg.includes("balance") ||
    msg.includes("transacci")
  )
    return "transacciones";
  return "pensamientos";
}

// ---------------------------------------------------------------------------
// Types for grouped data
// ---------------------------------------------------------------------------

interface GroupedAgent {
  agent: Agent;
  categories: Record<Category, AgentLog[]>;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatTime(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleString("es-ES", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "ahora";
  if (mins < 60) return `hace ${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `hace ${hours}h`;
  const days = Math.floor(hours / 24);
  return `hace ${days}d`;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LogEntry({ log }: { log: AgentLog }) {
  const category = classifyLog(log);
  const meta = CATEGORY_META[category];

  return (
    <div className="flex items-start gap-2 py-1.5 px-2 rounded-md hover:bg-white/5 transition-colors group">
      <div className={`mt-0.5 w-1.5 h-1.5 rounded-full shrink-0 ${meta.color.replace("text-", "bg-")}`} />
      <div className="flex-1 min-w-0">
        <p className="text-xs leading-relaxed text-foreground/80 break-words">
          {log.message.length > 200 ? log.message.slice(0, 200) + "..." : log.message}
        </p>
        <span className="text-[10px] text-muted-foreground">{formatTime(log.createdAt)}</span>
      </div>
    </div>
  );
}

function CategoryBranch({
  category,
  logs,
}: {
  category: Category;
  logs: AgentLog[];
}) {
  const [expanded, setExpanded] = useState(logs.length <= 5);
  const meta = CATEGORY_META[category];
  const Icon = meta.icon;
  const displayLogs = expanded ? logs : logs.slice(0, 3);

  if (logs.length === 0) return null;

  return (
    <div className="relative">
      {/* Connecting line to parent */}
      <div className="absolute left-0 top-0 bottom-0 w-px bg-border" />

      <div className="ml-4 relative">
        {/* Horizontal connector */}
        <div className="absolute -left-4 top-3.5 w-4 h-px bg-border" />

        {/* Category header */}
        <button
          onClick={() => setExpanded(!expanded)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${meta.bgColor} ${meta.borderColor} border transition-colors hover:brightness-125 w-full text-left`}
        >
          <Icon className={`w-3.5 h-3.5 ${meta.color}`} />
          <span className={`text-xs font-medium ${meta.color}`}>
            {meta.label}
          </span>
          <Badge variant="outline" className={`ml-auto text-[10px] ${meta.color} border-current/20`}>
            {logs.length}
          </Badge>
          {logs.length > 3 && (
            expanded ? (
              <ChevronDown className="w-3 h-3 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-3 h-3 text-muted-foreground" />
            )
          )}
        </button>

        {/* Log entries */}
        <div className="mt-1 ml-2 space-y-0.5">
          {displayLogs.map((log) => (
            <LogEntry key={log.id} log={log} />
          ))}
          {!expanded && logs.length > 3 && (
            <button
              onClick={() => setExpanded(true)}
              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-2 py-1"
            >
              + {logs.length - 3} actividades mas...
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function AgentTree({ group }: { group: GroupedAgent }) {
  const { agent, categories } = group;
  const totalLogs = Object.values(categories).reduce((sum, logs) => sum + logs.length, 0);
  const lastActivity = Object.values(categories)
    .flat()
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())[0];

  const categoryOrder: Category[] = [
    "pensamientos",
    "estrategias",
    "solicitudes",
    "transacciones",
    "errores",
    "controlador",
  ];

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-3">
          {/* Agent center node */}
          <div className="relative">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${
                agent.status === "alive"
                  ? "border-green-500 bg-green-500/10"
                  : agent.status === "dead"
                  ? "border-red-500/50 bg-red-500/10 opacity-60"
                  : "border-yellow-500 bg-yellow-500/10"
              }`}
            >
              <Bot
                className={`w-5 h-5 ${
                  agent.status === "alive"
                    ? "text-green-400"
                    : agent.status === "dead"
                    ? "text-red-400"
                    : "text-yellow-400"
                }`}
              />
            </div>
            {agent.status === "alive" && (
              <span className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-green-500 rounded-full border-2 border-card animate-pulse" />
            )}
          </div>

          <div className="flex-1 min-w-0">
            <CardTitle className="text-sm font-semibold truncate">
              {agent.name}
            </CardTitle>
            <div className="flex items-center gap-2 mt-0.5">
              <Badge
                variant="outline"
                className={`text-[10px] ${
                  agent.status === "alive"
                    ? "bg-green-500/20 text-green-400"
                    : agent.status === "dead"
                    ? "bg-red-500/20 text-red-400"
                    : "bg-yellow-500/20 text-yellow-400"
                }`}
              >
                {agent.status === "alive" ? "Activo" : agent.status === "dead" ? "Muerto" : "Pendiente"}
              </Badge>
              <span className="text-[10px] text-muted-foreground">
                Gen {agent.generation}
              </span>
            </div>
          </div>

          <div className="text-right shrink-0">
            <p className="text-xs font-mono text-muted-foreground">
              {totalLogs} actividades
            </p>
            {lastActivity && (
              <p className="text-[10px] text-muted-foreground">
                {timeAgo(lastActivity.createdAt)}
              </p>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0 pb-4">
        {totalLogs === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            Sin actividad registrada
          </p>
        ) : (
          <div className="space-y-2">
            {categoryOrder.map((cat) => (
              <CategoryBranch
                key={cat}
                category={cat}
                logs={categories[cat]}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Summary bar
// ---------------------------------------------------------------------------

function SummaryBar({ groups }: { groups: GroupedAgent[] }) {
  const totals = useMemo(() => {
    const counts: Record<Category, number> = {
      pensamientos: 0,
      estrategias: 0,
      solicitudes: 0,
      transacciones: 0,
      errores: 0,
      controlador: 0,
    };
    for (const g of groups) {
      for (const cat of Object.keys(counts) as Category[]) {
        counts[cat] += g.categories[cat].length;
      }
    }
    return counts;
  }, [groups]);

  return (
    <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
      {(Object.keys(CATEGORY_META) as Category[]).map((cat) => {
        const meta = CATEGORY_META[cat];
        const Icon = meta.icon;
        return (
          <div
            key={cat}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${meta.bgColor} ${meta.borderColor}`}
          >
            <Icon className={`w-4 h-4 ${meta.color}`} />
            <div>
              <p className={`text-sm font-bold ${meta.color}`}>{totals[cat]}</p>
              <p className="text-[10px] text-muted-foreground">{meta.label}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function ActivityMapPage() {
  const { data: agentsData } = useAgents();
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | "all">("all");

  const agents = agentsData?.data ?? [];

  const fetchLogs = async () => {
    try {
      setLoading(true);
      setError(null);
      const params = selectedAgent !== "all" ? { agent_id: selectedAgent } : undefined;
      const result = await api.getLogs(params);
      setLogs(result.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error al cargar los registros");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    // Refetch every 30 seconds
    const interval = setInterval(fetchLogs, 30000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAgent]);

  // Group logs by agent, then by category
  const groups: GroupedAgent[] = useMemo(() => {
    const agentMap = new Map<string, Agent>();
    for (const a of agents) {
      agentMap.set(a.id, a);
    }

    const logsByAgent = new Map<string, AgentLog[]>();
    for (const log of logs) {
      if (!logsByAgent.has(log.agentId)) {
        logsByAgent.set(log.agentId, []);
      }
      logsByAgent.get(log.agentId)!.push(log);
    }

    const result: GroupedAgent[] = [];

    for (const [agentId, agentLogs] of logsByAgent) {
      const agent = agentMap.get(agentId);
      if (!agent) continue;

      const categories: Record<Category, AgentLog[]> = {
        pensamientos: [],
        estrategias: [],
        solicitudes: [],
        transacciones: [],
        errores: [],
        controlador: [],
      };

      // Sort logs by date descending (most recent first)
      const sorted = [...agentLogs].sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );

      for (const log of sorted) {
        const cat = classifyLog(log);
        categories[cat].push(log);
      }

      result.push({ agent, categories });
    }

    // Sort by most recent activity
    result.sort((a, b) => {
      const aLatest = Object.values(a.categories)
        .flat()
        .sort((x, y) => new Date(y.createdAt).getTime() - new Date(x.createdAt).getTime())[0];
      const bLatest = Object.values(b.categories)
        .flat()
        .sort((x, y) => new Date(y.createdAt).getTime() - new Date(x.createdAt).getTime())[0];
      if (!aLatest) return 1;
      if (!bLatest) return -1;
      return new Date(bLatest.createdAt).getTime() - new Date(aLatest.createdAt).getTime();
    });

    return result;
  }, [logs, agents]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Mapa de Actividad</h1>
          <p className="text-muted-foreground text-sm">
            Visualiza la evolucion y actividades de cada agente en tiempo real
          </p>
        </div>
        <button
          onClick={fetchLogs}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border text-sm text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Actualizar
        </button>
      </div>

      {/* Agent filter */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground">Filtrar por agente:</span>
        <button
          onClick={() => setSelectedAgent("all")}
          className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${
            selectedAgent === "all"
              ? "bg-primary text-primary-foreground border-primary"
              : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/30"
          }`}
        >
          Todos
        </button>
        {agents.map((agent) => (
          <button
            key={agent.id}
            onClick={() => setSelectedAgent(agent.id)}
            className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${
              selectedAgent === agent.id
                ? "bg-primary text-primary-foreground border-primary"
                : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/30"
            }`}
          >
            {agent.name}
          </button>
        ))}
      </div>

      {/* Summary */}
      {!loading && !error && <SummaryBar groups={groups} />}

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <div className="flex flex-col items-center gap-3">
            <RefreshCw className="w-6 h-6 text-muted-foreground animate-spin" />
            <p className="text-sm text-muted-foreground">Cargando actividades...</p>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardContent className="py-6 text-center">
            <AlertTriangle className="w-6 h-6 text-red-400 mx-auto mb-2" />
            <p className="text-sm text-red-400">{error}</p>
            <button
              onClick={fetchLogs}
              className="mt-3 text-xs text-muted-foreground hover:text-foreground underline"
            >
              Reintentar
            </button>
          </CardContent>
        </Card>
      )}

      {/* Activity map */}
      {!loading && !error && (
        <ScrollArea className="h-[calc(100vh-320px)]">
          {groups.length === 0 ? (
            <div className="text-center py-20 text-muted-foreground">
              <Bot className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p className="text-sm">Sin actividades registradas</p>
              <p className="text-xs mt-1">Las actividades de los agentes apareceran aqui</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pb-4">
              {groups.map((group) => (
                <AgentTree key={group.agent.id} group={group} />
              ))}
            </div>
          )}
        </ScrollArea>
      )}
    </div>
  );
}
