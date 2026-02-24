const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
      ...options.headers,
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(error.message || `API error: ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Agents
  getAgents: () => apiFetch<{ data: Agent[]; total: number }>("/api/agents"),
  getAgent: (id: string) => apiFetch<AgentDetail>(`/api/agents/${id}`),
  createAgent: (data: { name: string; systemPrompt: string; cryptoBalance?: string }) =>
    apiFetch<Agent>("/api/agents", { method: "POST", body: JSON.stringify(data) }),
  killAgent: (id: string) =>
    apiFetch<{ message: string }>(`/api/agents/${id}`, { method: "DELETE" }),

  // Requests
  getRequests: (params?: { status?: string; type?: string; agent_id?: string }) => {
    const query = new URLSearchParams(params as Record<string, string>).toString();
    return apiFetch<{ data: RequestWithAgent[]; total: number }>(
      `/api/requests${query ? `?${query}` : ""}`
    );
  },
  approveRequest: (id: string, response?: string) =>
    apiFetch<RequestItem>(`/api/requests/${id}/approve`, {
      method: "POST",
      body: JSON.stringify({ response }),
    }),
  denyRequest: (id: string, reason?: string) =>
    apiFetch<RequestItem>(`/api/requests/${id}/deny`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    }),
  bulkAction: (ids: string[], action: "approve" | "deny") =>
    apiFetch<{ results: unknown[] }>("/api/requests/bulk", {
      method: "POST",
      body: JSON.stringify({ ids, action }),
    }),

  // Transactions
  getTransactions: (agentId?: string) => {
    const query = agentId ? `?agent_id=${agentId}` : "";
    return apiFetch<{ data: Transaction[]; total: number }>(`/api/transactions${query}`);
  },
  addIncome: (data: { agentId: string; amount: number; description: string }) =>
    apiFetch<{ newBalance: string }>("/api/transactions", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Logs
  getLogs: (params?: { agent_id?: string; level?: string }) => {
    const query = new URLSearchParams(params as Record<string, string>).toString();
    return apiFetch<{ data: AgentLog[]; total: number }>(
      `/api/logs${query ? `?${query}` : ""}`
    );
  },

  // Controller messages
  sendMessage: (agentId: string, message: string) =>
    apiFetch<AgentLog>("/api/logs/message", {
      method: "POST",
      body: JSON.stringify({ agentId, message }),
    }),

  // Stats
  getStats: () => apiFetch<DashboardStats>("/api/stats"),

  // Engine
  triggerCycle: () =>
    apiFetch<{ message: string }>("/api/engine/trigger", { method: "POST" }),
  getAutoApprove: () =>
    apiFetch<{ enabled: boolean; autoApproveTypes: string[] }>(
      "/api/engine/auto-approve"
    ),
  setAutoApprove: (enabled: boolean) =>
    apiFetch<{ enabled: boolean; autoApproveTypes: string[] }>(
      "/api/engine/auto-approve",
      { method: "POST", body: JSON.stringify({ enabled }) }
    ),
};

// Types used by the API client
export interface Agent {
  id: string;
  parentId: string | null;
  generation: number;
  name: string;
  systemPrompt: string;
  strategy: string | null;
  apiBudget: string;
  cryptoBalance: string;
  solanaAddress: string | null;
  status: "alive" | "dead" | "pending";
  bornAt: string;
  diesAt: string;
  lastThoughtAt: string | null;
  metadata: Record<string, unknown>;
}

export interface AgentDetail extends Agent {
  parent: Agent | null;
  children: Agent[];
  recentTransactions: Transaction[];
  recentLogs: AgentLog[];
  requests: RequestItem[];
}

export interface RequestItem {
  id: string;
  agentId: string;
  type: string;
  title: string;
  description: string;
  payload: Record<string, unknown>;
  status: "pending" | "approved" | "denied";
  priority: string;
  createdAt: string;
  resolvedAt: string | null;
  resolvedBy: string | null;
}

export interface RequestWithAgent extends RequestItem {
  agent: {
    id: string;
    name: string;
    generation: number;
    status: string;
  };
}

export interface Transaction {
  id: string;
  agentId: string;
  amount: string;
  type: string;
  description: string;
  balanceAfter: string;
  referenceId: string | null;
  createdAt: string;
}

export interface AgentLog {
  id: string;
  agentId: string;
  level: string;
  message: string;
  metadata: Record<string, unknown>;
  createdAt: string;
}

export interface DashboardStats {
  totalAgents: number;
  aliveAgents: number;
  deadAgents: number;
  pendingAgents: number;
  totalCryptoBalance: string;
  totalEcosystemBalance: string;
  pendingRequestsCount: number;
  agentsByGeneration: Record<number, number>;
}
