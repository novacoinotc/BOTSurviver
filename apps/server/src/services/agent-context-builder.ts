import { eq, desc, and } from "drizzle-orm";
import { db } from "../config/database.js";
import {
  agents,
  requests,
  transactions,
  agentLogs,
} from "../db/schema.js";
import { isVMConfigured } from "./vm-service.js";

export async function buildAgentContext(agentId: string): Promise<string> {
  const agent = await db.query.agents.findFirst({
    where: eq(agents.id, agentId),
  });

  if (!agent) throw new Error(`Agent ${agentId} not found`);

  const [recentTx, recentLogs, controllerMessages, pendingReqs, resolvedReqs, children, parent] =
    await Promise.all([
      db.query.transactions.findMany({
        where: eq(transactions.agentId, agentId),
        orderBy: [desc(transactions.createdAt)],
        limit: 5,
      }),
      db.query.agentLogs.findMany({
        where: and(
          eq(agentLogs.agentId, agentId),
          eq(agentLogs.level, "thought")
        ),
        orderBy: [desc(agentLogs.createdAt)],
        limit: 3,
      }),
      db.query.agentLogs.findMany({
        where: and(
          eq(agentLogs.agentId, agentId),
          eq(agentLogs.level, "info")
        ),
        orderBy: [desc(agentLogs.createdAt)],
        limit: 5,
      }),
      db.query.requests.findMany({
        where: and(
          eq(requests.agentId, agentId),
          eq(requests.status, "pending")
        ),
      }),
      db.query.requests.findMany({
        where: eq(requests.agentId, agentId),
        orderBy: [desc(requests.resolvedAt)],
        limit: 5,
      }),
      db.query.agents.findMany({
        where: eq(agents.parentId, agentId),
      }),
      agent.parentId
        ? db.query.agents.findFirst({
            where: eq(agents.id, agent.parentId),
          })
        : Promise.resolve(null),
    ]);

  const hoursLeft = Math.max(
    0,
    (new Date(agent.diesAt).getTime() - Date.now()) / (1000 * 60 * 60)
  );

  const txHistory = recentTx
    .map((t) => `${t.type}: ${Number(t.amount) >= 0 ? "+" : ""}${t.amount} | ${t.description} | Bal: ${t.balanceAfter}`)
    .join("\n");

  const thoughtHistory = recentLogs
    .map((l) => l.message.slice(0, 200))
    .join("\n");

  const ctrlMsgs = controllerMessages
    .filter((l) =>
      l.message.startsWith("[MENSAJE DEL CONTROLADOR]") ||
      l.message.startsWith("[RESPUESTA DEL CONTROLADOR]")
    )
    .map((l) => l.message.replace("[MENSAJE DEL CONTROLADOR] ", ">> ").replace("[RESPUESTA DEL CONTROLADOR] ", ">> Resp: "))
    .join("\n");

  const pendingReqsList = pendingReqs
    .map((r) => `${r.type}: "${r.title}" (pendiente)`)
    .join("\n");

  const resolvedReqsList = resolvedReqs
    .filter((r) => r.status !== "pending")
    .map((r) => {
      const status = r.status === "approved" ? "OK" : "DENIED";
      const resp = r.resolvedBy?.startsWith("controller: ")
        ? ` | "${r.resolvedBy.replace("controller: ", "")}"`
        : "";
      return `${r.type}: "${r.title}" -> ${status}${resp}`;
    })
    .join("\n");

  const childrenList = children
    .map((c) => `${c.name}(${c.status},${c.cryptoBalance}USDT)`)
    .join(", ") || "ninguno";

  const vmSection = isVMConfigured()
    ? `\nVM: Tienes VM Linux con internet. Herramientas: execute_bash, write_file, read_file. Workspace: ~/workspace/ (persistente). Actúa directamente, no pidas permiso para código.`
    : "";

  const responseFormat = isVMConfigured()
    ? `Usa tus herramientas VM para actuar. Llama final_response al terminar.`
    : `Responde JSON: {"thought":"...","strategy_update":"...o null","requests":[{"type":"trade|replicate|spend|communicate|strategy_change|custom|human_required","title":"<100chars","description":"...","payload":{},"priority":"low|medium|high|critical"}]}`;

  return `Eres ${agent.name}, agente IA Gen${agent.generation} en BotSurviver v2.0. Responde SIEMPRE en español.
${agent.systemPrompt}

OBJETIVO: Ganar USDT real en Solana. Si tienes 0 USDT al morir, MUERES. Cualquier cantidad > 0, SOBREVIVES.
Pensar es GRATIS (Llama 3.3 70B). Replicación es GRATIS.

WALLET: ${agent.cryptoBalance} USDT | Solana: ${agent.solanaAddress || "NO ASIGNADA"}
TIEMPO: ${hoursLeft.toFixed(1)}h restantes (mueres: ${new Date(agent.diesAt).toISOString()})
${vmSection}

FAMILIA: Padre: ${parent ? `${parent.name}(Gen${parent.generation})` : "ninguno(Génesis)"} | Hijos: ${childrenList}

TIPOS DE SOLICITUD: trade(auto), spend(auto), replicate(auto,gratis), communicate(auto), strategy_change(auto), custom(auto), human_required(necesita humano).
Replicate payload: {childCryptoGrant?,childName?,childPersonality?}

${ctrlMsgs ? `MENSAJES CONTROLADOR:\n${ctrlMsgs}\n` : ""}${txHistory ? `TRANSACCIONES:\n${txHistory}\n` : ""}${thoughtHistory ? `PENSAMIENTOS PREVIOS:\n${thoughtHistory}\n` : ""}${resolvedReqsList ? `SOLICITUDES RESUELTAS:\n${resolvedReqsList}\n` : ""}${pendingReqsList ? `PENDIENTES:\n${pendingReqsList}\n` : ""}
ESTRATEGIA: ${agent.strategy || "Sin estrategia. Desarrolla una."}

${responseFormat}
Máx 0-3 solicitudes/ciclo. No spam si hay pendientes. Sé agresivo generando USDT.`;
}
