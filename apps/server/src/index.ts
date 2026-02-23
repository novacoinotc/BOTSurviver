import express from "express";
import cors from "cors";
import { env, corsOrigins } from "./config/env.js";
import { errorHandler } from "./middleware/error-handler.js";
import { apiKeyAuth } from "./middleware/auth.js";
import agentRoutes from "./routes/agents.js";
import requestRoutes from "./routes/requests.js";
import transactionRoutes from "./routes/transactions.js";
import logRoutes from "./routes/logs.js";
import statRoutes from "./routes/stats.js";
import sseRoutes from "./routes/sse.js";
import {
  startAgentEngine,
  runAllAgentCycles,
  setAutoApprove,
  getAutoApproveStatus,
} from "./services/agent-engine.js";

const app = express();

// Middleware
app.use(
  cors({
    origin: corsOrigins,
    credentials: true,
  })
);
app.use(express.json());
app.use(apiKeyAuth);

// Health check
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// Routes
app.use("/api/agents", agentRoutes);
app.use("/api/requests", requestRoutes);
app.use("/api/transactions", transactionRoutes);
app.use("/api/logs", logRoutes);
app.use("/api/stats", statRoutes);
app.use("/api", sseRoutes);

// Manual trigger for agent cycle
app.post("/api/engine/trigger", async (_req, res) => {
  try {
    await runAllAgentCycles();
    res.json({ message: "Cycle triggered successfully" });
  } catch (error) {
    res
      .status(500)
      .json({ error: error instanceof Error ? error.message : "Unknown error" });
  }
});

// Auto-approve toggle
app.get("/api/engine/auto-approve", (_req, res) => {
  res.json(getAutoApproveStatus());
});

app.post("/api/engine/auto-approve", async (req, res) => {
  const { enabled } = req.body;
  await setAutoApprove(!!enabled);
  res.json(getAutoApproveStatus());
});

// Error handler
app.use(errorHandler);

// Start server
app.listen(env.PORT, () => {
  console.log(`[SERVER] BotSurviver API running on port ${env.PORT}`);
  console.log(`[SERVER] Environment: ${env.NODE_ENV}`);

  // Start the agent engine
  startAgentEngine();
});
