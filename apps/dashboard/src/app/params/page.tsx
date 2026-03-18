"use client";

import useSWR from "swr";
import { api } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SlidersHorizontal, ArrowRight, RotateCcw } from "lucide-react";

const PARAM_LABELS: Record<string, string> = {
  default_leverage: "Default Leverage",
  default_position_pct: "Position Size (%)",
  max_open_positions: "Max Open Positions",
  min_score_to_enter: "Min Confidence",
  daily_loss_pause_pct: "Daily Loss Pause (%)",
};

const PARAM_FORMAT: Record<string, (v: number) => string> = {
  default_leverage: (v) => `${v}x`,
  default_position_pct: (v) => `${(v * 100).toFixed(2)}%`,
  max_open_positions: (v) => `${v}`,
  min_score_to_enter: (v) => `${(v * 100).toFixed(0)}%`,
  daily_loss_pause_pct: (v) => `${(v * 100).toFixed(1)}%`,
};

export default function ParamsPage() {
  const { data } = useSWR("params", () => api.getParams(), { refreshInterval: 30000 });

  const current = data?.current ?? {};
  const history = data?.history ?? [];

  return (
    <div className="space-y-4 md:space-y-6">
      <h1 className="text-xl md:text-2xl font-bold flex items-center gap-2">
        <SlidersHorizontal className="w-5 h-5 md:w-6 md:h-6" />
        Parameters
      </h1>

      {/* Current Parameters */}
      <Card>
        <CardHeader className="px-3 md:px-6">
          <CardTitle className="text-sm md:text-base">Current Parameters</CardTitle>
        </CardHeader>
        <CardContent className="px-3 md:px-6">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 md:gap-4">
            {Object.entries(current).map(([name, value]) => (
              <div key={name} className="p-3 rounded-lg bg-accent/50 text-center">
                <p className="text-xs text-muted-foreground">{PARAM_LABELS[name] || name}</p>
                <p className="text-lg md:text-xl font-bold font-mono mt-1">
                  {PARAM_FORMAT[name] ? PARAM_FORMAT[name](value) : value}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Change History */}
      <Card>
        <CardHeader className="px-3 md:px-6">
          <CardTitle className="text-sm md:text-base">Parameter Change History</CardTitle>
        </CardHeader>
        <CardContent className="px-3 md:px-6">
          {history.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No parameter changes yet. The optimizer will adjust parameters every 6 hours.
            </p>
          ) : (
            <div className="space-y-3">
              {history.map((change) => {
                const fmt = PARAM_FORMAT[change.param_name] || ((v: number) => `${v}`);
                return (
                  <div key={change.id} className="p-3 rounded-lg bg-accent/30 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-sm">
                        {PARAM_LABELS[change.param_name] || change.param_name}
                      </span>
                      {change.reverted ? (
                        <Badge variant="secondary" className="bg-red-500/20 text-red-400 text-xs shrink-0">
                          <RotateCcw className="w-3 h-3 mr-1" />
                          Reverted
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-green-500/20 text-green-400 text-xs shrink-0">
                          Active
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 font-mono text-sm">
                      <span className="text-muted-foreground">{fmt(change.old_value)}</span>
                      <ArrowRight className="w-3 h-3 shrink-0" />
                      <span className="font-medium">{fmt(change.new_value)}</span>
                    </div>
                    {change.reasoning && (
                      <p className="text-xs text-muted-foreground line-clamp-2">{change.reasoning}</p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      {new Date(change.created_at).toLocaleString()}
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
