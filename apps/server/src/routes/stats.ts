import { Router } from "express";
import { eq } from "drizzle-orm";
import { db } from "../config/database.js";
import { agents, requests } from "../db/schema.js";

const router = Router();

router.get("/", async (_req, res) => {
  const allAgents = await db.query.agents.findMany();

  const alive = allAgents.filter((a) => a.status === "alive");
  const dead = allAgents.filter((a) => a.status === "dead");
  const pending = allAgents.filter((a) => a.status === "pending");

  const totalCryptoBalance = alive.reduce(
    (sum, a) => sum + Number(a.cryptoBalance),
    0
  );

  const pendingRequests = await db.query.requests.findMany({
    where: eq(requests.status, "pending"),
    columns: { id: true },
  });

  const agentsByGeneration: Record<number, number> = {};
  for (const agent of allAgents) {
    agentsByGeneration[agent.generation] =
      (agentsByGeneration[agent.generation] || 0) + 1;
  }

  res.json({
    totalAgents: allAgents.length,
    aliveAgents: alive.length,
    deadAgents: dead.length,
    pendingAgents: pending.length,
    totalCryptoBalance: totalCryptoBalance.toFixed(8),
    totalEcosystemBalance: totalCryptoBalance.toFixed(8),
    pendingRequestsCount: pendingRequests.length,
    agentsByGeneration,
  });
});

export default router;
