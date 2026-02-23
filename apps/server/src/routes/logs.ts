import { Router } from "express";
import { eq, desc, and } from "drizzle-orm";
import { db } from "../config/database.js";
import { agentLogs, agents } from "../db/schema.js";
import { sseManager } from "../lib/sse-manager.js";
import type { LogLevel } from "@botsurviver/shared";

const router = Router();

router.get("/", async (req, res) => {
  const agentId = req.query.agent_id as string | undefined;
  const level = req.query.level as LogLevel | undefined;

  const conditions = [];
  if (agentId) conditions.push(eq(agentLogs.agentId, agentId));
  if (level) conditions.push(eq(agentLogs.level, level));

  const logs = await db.query.agentLogs.findMany({
    where: conditions.length > 0 ? and(...conditions) : undefined,
    orderBy: [desc(agentLogs.createdAt)],
    limit: 100,
  });

  res.json({ data: logs, total: logs.length });
});

// Send a message from Controller to an agent
router.post("/message", async (req, res) => {
  const { agentId, message } = req.body;

  if (!agentId || !message) {
    return res
      .status(400)
      .json({ error: "agentId and message are required" });
  }

  const agent = await db.query.agents.findFirst({
    where: eq(agents.id, agentId),
    columns: { id: true, name: true },
  });

  if (!agent) {
    return res.status(404).json({ error: "Agent not found" });
  }

  const [log] = await db
    .insert(agentLogs)
    .values({
      agentId,
      level: "info",
      message: `[MENSAJE DEL CONTROLADOR] ${message}`,
      metadata: { fromController: true, originalMessage: message },
    })
    .returning();

  sseManager.broadcast({
    type: "agent_activity",
    data: {
      agentId,
      name: agent.name,
      status: "controller_message",
      message: `Controlador envió mensaje a ${agent.name}: "${message.slice(0, 100)}"`,
      timestamp: new Date().toISOString(),
    },
  });

  res.json(log);
});

export default router;
