"use client";

import { useSSE, type LiveEvent } from "@/hooks/use-sse";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Brain,
  Search,
  Cpu,
  Lightbulb,
  FileText,
  RefreshCw,
  Skull,
  Radio,
  CircleDot,
  Zap,
  MessageSquare,
  CheckCircle,
  Clock,
} from "lucide-react";

function getStatusIcon(status: string) {
  switch (status) {
    case "thinking":
      return <Brain className="w-3.5 h-3.5 text-purple-400 animate-pulse" />;
    case "building_context":
      return <Search className="w-3.5 h-3.5 text-blue-400 animate-pulse" />;
    case "calling_ai":
      return <Cpu className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />;
    case "thought_complete":
      return <Lightbulb className="w-3.5 h-3.5 text-yellow-400" />;
    case "strategy_updated":
      return <Zap className="w-3.5 h-3.5 text-orange-400" />;
    case "request_created":
      return <FileText className="w-3.5 h-3.5 text-green-400" />;
    case "request_auto_approved":
      return <CheckCircle className="w-3.5 h-3.5 text-green-400" />;
    case "request_pending":
      return <Clock className="w-3.5 h-3.5 text-yellow-400" />;
    case "controller_message":
      return <MessageSquare className="w-3.5 h-3.5 text-blue-400" />;
    case "idle":
      return <CircleDot className="w-3.5 h-3.5 text-gray-400" />;
    case "cycle_started":
      return <RefreshCw className="w-3.5 h-3.5 text-blue-400 animate-spin" />;
    case "cycle_complete":
      return <CircleDot className="w-3.5 h-3.5 text-green-400" />;
    case "reaper_ran":
      return <Skull className="w-3.5 h-3.5 text-red-400" />;
    default:
      return <Radio className="w-3.5 h-3.5 text-gray-400" />;
  }
}

function getStatusColor(status: string) {
  switch (status) {
    case "thinking":
    case "building_context":
    case "calling_ai":
      return "border-l-purple-500";
    case "thought_complete":
      return "border-l-yellow-500";
    case "strategy_updated":
      return "border-l-orange-500";
    case "request_created":
    case "request_auto_approved":
      return "border-l-green-500";
    case "request_pending":
      return "border-l-yellow-500";
    case "controller_message":
      return "border-l-blue-500";
    case "cycle_started":
      return "border-l-blue-500";
    case "cycle_complete":
      return "border-l-emerald-500";
    case "reaper_ran":
      return "border-l-red-500";
    default:
      return "border-l-gray-500";
  }
}

function timeAgo(timestamp: string) {
  const diff = Date.now() - new Date(timestamp).getTime();
  const secs = Math.floor(diff / 1000);
  if (secs < 5) return "ahora";
  if (secs < 60) return `hace ${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `hace ${mins}m`;
  return `hace ${Math.floor(mins / 60)}h`;
}

function EventItem({ event }: { event: LiveEvent }) {
  return (
    <div
      className={`flex items-start gap-2.5 px-3 py-2 border-l-2 ${getStatusColor(event.status)} bg-card/50 rounded-r-md`}
    >
      <div className="mt-0.5 shrink-0">{getStatusIcon(event.status)}</div>
      <div className="min-w-0 flex-1">
        <p className="text-sm leading-snug">{event.message}</p>
        {event.thought && (
          <p className="text-xs text-muted-foreground mt-1 italic line-clamp-2">
            &ldquo;{event.thought}&rdquo;
          </p>
        )}
      </div>
      <span className="text-[10px] text-muted-foreground shrink-0 mt-0.5">
        {timeAgo(event.timestamp)}
      </span>
    </div>
  );
}

export function LiveFeed() {
  const { events, connected } = useSSE();

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Radio className="w-4 h-4" />
          En Vivo
        </h3>
        <Badge
          variant="outline"
          className={
            connected
              ? "bg-green-500/20 text-green-400 text-[10px]"
              : "bg-red-500/20 text-red-400 text-[10px]"
          }
        >
          <span
            className={`w-1.5 h-1.5 rounded-full mr-1 inline-block ${
              connected ? "bg-green-400 animate-pulse" : "bg-red-400"
            }`}
          />
          {connected ? "Conectado" : "Desconectado"}
        </Badge>
      </div>

      <ScrollArea className="h-[400px]">
        <div className="space-y-1.5 pr-3">
          {events.length === 0 && (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Radio className="w-6 h-6 mx-auto mb-2 opacity-50" />
              <p>Esperando actividad...</p>
              <p className="text-xs mt-1">
                Los eventos aparecerán aquí en tiempo real
              </p>
            </div>
          )}
          {events.map((event) => (
            <EventItem key={event.id} event={event} />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
