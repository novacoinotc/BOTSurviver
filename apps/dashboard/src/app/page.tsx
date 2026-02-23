"use client";

import { useAgents, useStats } from "@/hooks/use-agents";
import { AgentCard } from "@/components/agents/agent-card";
import { LiveFeed } from "@/components/live-feed";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api-client";
import { toast } from "sonner";
import { Zap, RefreshCw, ShieldCheck, ShieldOff } from "lucide-react";
import { useState, useEffect } from "react";

export default function OverviewPage() {
  const { data: agentsData, mutate: mutateAgents } = useAgents();
  const { mutate: mutateStats } = useStats();
  const [triggering, setTriggering] = useState(false);
  const [autoApprove, setAutoApprove] = useState(true);
  const [togglingAuto, setTogglingAuto] = useState(false);

  const agents = agentsData?.data ?? [];
  const alive = agents.filter((a) => a.status === "alive");
  const dead = agents.filter((a) => a.status === "dead");

  useEffect(() => {
    api.getAutoApprove().then((data) => setAutoApprove(data.enabled)).catch(() => {});
  }, []);

  const handleTrigger = async () => {
    setTriggering(true);
    try {
      await api.triggerCycle();
      toast.success("Ciclo de agentes activado");
      setTimeout(() => {
        mutateAgents();
        mutateStats();
      }, 3000);
    } catch {
      toast.error("Error al activar ciclo");
    } finally {
      setTriggering(false);
    }
  };

  const handleToggleAutoApprove = async () => {
    setTogglingAuto(true);
    try {
      const result = await api.setAutoApprove(!autoApprove);
      setAutoApprove(result.enabled);
      toast.success(
        result.enabled
          ? "Auto-aprobar activado: el agente opera libremente"
          : "Auto-aprobar desactivado: requiere aprobación manual"
      );
    } catch {
      toast.error("Error al cambiar auto-aprobar");
    } finally {
      setTogglingAuto(false);
    }
  };

  const handleRefresh = () => {
    mutateAgents();
    mutateStats();
  };

  return (
    <div className="flex gap-6">
      {/* Main content */}
      <div className="flex-1 space-y-6 min-w-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Panel de Agentes</h1>
            <p className="text-muted-foreground text-sm">
              Monitorea todos los agentes autónomos del ecosistema
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              variant={autoApprove ? "default" : "outline"}
              size="sm"
              onClick={handleToggleAutoApprove}
              disabled={togglingAuto}
              className={autoApprove ? "bg-green-600 hover:bg-green-700" : ""}
            >
              {autoApprove ? (
                <ShieldCheck className="w-4 h-4 mr-1" />
              ) : (
                <ShieldOff className="w-4 h-4 mr-1" />
              )}
              {autoApprove ? "Auto-Aprobar: ON" : "Auto-Aprobar: OFF"}
            </Button>
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw className="w-4 h-4 mr-1" />
              Actualizar
            </Button>
            <Button
              size="sm"
              onClick={handleTrigger}
              disabled={triggering}
            >
              <Zap className="w-4 h-4 mr-1" />
              {triggering ? "Ejecutando..." : "Forzar Ciclo"}
            </Button>
          </div>
        </div>

        {alive.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold mb-3 text-green-400">
              Vivos ({alive.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {alive.map((agent) => (
                <AgentCard key={agent.id} agent={agent} />
              ))}
            </div>
          </section>
        )}

        {dead.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold mb-3 text-red-400">
              Muertos ({dead.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {dead.map((agent) => (
                <AgentCard key={agent.id} agent={agent} />
              ))}
            </div>
          </section>
        )}

        {agents.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-lg">Sin agentes aún</p>
            <p className="text-sm">
              Ejecuta el seed para crear el agente Genesis
            </p>
          </div>
        )}
      </div>

      {/* Live Feed sidebar */}
      <Card className="w-80 shrink-0 p-4 hidden lg:block">
        <LiveFeed />
      </Card>
    </div>
  );
}
