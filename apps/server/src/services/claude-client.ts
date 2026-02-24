import Groq from "groq-sdk";
import { env } from "../config/env.js";
import {
  executeCommand,
  writeFile,
  readFile,
  isVMConfigured,
} from "./vm-service.js";
import { sseManager } from "../lib/sse-manager.js";

const groq = new Groq({ apiKey: env.GROQ_API_KEY });

export interface AgentThoughtResponse {
  thought: string;
  strategy_update: string | null;
  requests: Array<{
    type: string;
    title: string;
    description: string;
    payload: Record<string, unknown>;
    priority: string;
  }>;
  apiCost: number;
  toolsUsed: number;
}

// Tool definitions in OpenAI format for Groq - kept minimal to save tokens
const VM_TOOLS: Groq.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "execute_bash",
      description: "Ejecuta bash en tu VM Linux. Dir: ~/workspace/",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "Comando bash" },
          timeout: { type: "number", description: "Timeout seg (max 120)" },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description: "Escribe archivo en ~/workspace/",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Ruta relativa" },
          content: { type: "string", description: "Contenido" },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Lee archivo de ~/workspace/",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Ruta relativa" },
        },
        required: ["path"],
      },
    },
  },
];

const FINAL_RESPONSE_TOOL: Groq.Chat.ChatCompletionTool = {
  type: "function",
  function: {
    name: "final_response",
    description: "Respuesta final del ciclo con pensamiento y solicitudes.",
    parameters: {
      type: "object",
      properties: {
        thought: { type: "string", description: "Tu análisis del ciclo" },
        strategy_update: { type: "string", description: "Estrategia nueva o null" },
        requests: {
          type: "array",
          description: "Solicitudes (0-3)",
          items: {
            type: "object",
            properties: {
              type: { type: "string", enum: ["trade", "replicate", "spend", "communicate", "strategy_change", "custom", "human_required"] },
              title: { type: "string" },
              description: { type: "string" },
              payload: { type: "object" },
              priority: { type: "string", enum: ["low", "medium", "high", "critical"] },
            },
            required: ["type", "title", "description", "priority"],
          },
        },
      },
      required: ["thought", "requests"],
    },
  },
};

// Safety net only - the real limit is the agent's API budget
const MAX_TOOL_TURNS = 50;

// Groq Llama 3.3 70B pricing: $0.59/M input, $0.79/M output
function calculateCost(usage: {
  prompt_tokens?: number;
  completion_tokens?: number;
}): number {
  const input = usage.prompt_tokens || 0;
  const output = usage.completion_tokens || 0;
  return (input / 1_000_000) * 0.59 + (output / 1_000_000) * 0.79;
}

type GroqMessage = Groq.Chat.ChatCompletionMessageParam;

/**
 * Run an agent's thinking cycle with optional VM tool access.
 * Uses Groq (Llama 3.3 70B) with multi-turn tool calling.
 */
export async function getAgentThought(
  systemPrompt: string,
  agentId: string,
  agentName: string
): Promise<AgentThoughtResponse> {
  const vmAvailable = isVMConfigured();
  const tools = vmAvailable
    ? [...VM_TOOLS, FINAL_RESPONSE_TOOL]
    : [FINAL_RESPONSE_TOOL];

  const messages: GroqMessage[] = [
    { role: "system", content: systemPrompt },
    {
      role: "user",
      content: vmAvailable
        ? "Ciclo de pensamiento. Actúa con tus herramientas VM y llama final_response al terminar."
        : "Ciclo de pensamiento. Llama final_response con tu análisis y solicitudes.",
    },
  ];

  let totalCost = 0;
  let toolsUsed = 0;

  for (let turn = 0; turn < MAX_TOOL_TURNS; turn++) {
    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      max_tokens: 2048,
      temperature: 0.6,
      tools,
      tool_choice: "auto",
      messages,
    });

    const choice = response.choices[0];
    if (!choice) break;

    if (response.usage) {
      totalCost += calculateCost(response.usage);
    }

    const message = choice.message;

    // If model responded with text (no tool calls) - try to parse or use as thought
    if (choice.finish_reason === "stop" || !message.tool_calls?.length) {
      const text = message.content || "";
      // Try to parse as JSON fallback
      try {
        let jsonText = text.trim();
        if (jsonText.startsWith("```")) {
          jsonText = jsonText
            .replace(/^```(?:json)?\n?/, "")
            .replace(/\n?```$/, "");
        }
        const parsed = JSON.parse(jsonText);
        return {
          thought: parsed.thought || text.slice(0, 500),
          strategy_update: parsed.strategy_update || null,
          requests: Array.isArray(parsed.requests)
            ? parsed.requests.slice(0, 3)
            : [],
          apiCost: totalCost,
          toolsUsed,
        };
      } catch {
        return {
          thought: text.slice(0, 1000) || "Sin pensamiento",
          strategy_update: null,
          requests: [],
          apiCost: totalCost,
          toolsUsed,
        };
      }
    }

    // Process tool calls
    if (message.tool_calls?.length) {
      // Add assistant message to history (with tool_calls)
      messages.push(message as GroqMessage);

      for (const toolCall of message.tool_calls) {
        const funcName = toolCall.function.name;
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(toolCall.function.arguments);
        } catch {
          args = {};
        }

        // Handle final_response
        if (funcName === "final_response") {
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: "OK",
          });

          return {
            thought: (args.thought as string) || "Sin pensamiento",
            strategy_update: (args.strategy_update as string) || null,
            requests: Array.isArray(args.requests)
              ? (
                  args.requests as AgentThoughtResponse["requests"]
                ).slice(0, 3)
              : [],
            apiCost: totalCost,
            toolsUsed,
          };
        }

        // Handle VM tools
        toolsUsed++;
        let toolResult: string;

        try {
          switch (funcName) {
            case "execute_bash": {
              const cmd = args.command as string;
              const timeout = args.timeout as number | undefined;

              sseManager.broadcast({
                type: "agent_activity",
                data: {
                  agentId,
                  name: agentName,
                  status: "vm_executing",
                  message: `${agentName} ejecutando: ${cmd.slice(0, 100)}`,
                  command: cmd.slice(0, 200),
                  timestamp: new Date().toISOString(),
                },
              });

              const result = await executeCommand(agentId, cmd, timeout);
              toolResult =
                result.exitCode === 0
                  ? result.stdout || "(sin output)"
                  : `EXIT CODE ${result.exitCode}\nSTDOUT: ${result.stdout}\nSTDERR: ${result.stderr}`;
              break;
            }

            case "write_file": {
              const path = args.path as string;
              const content = args.content as string;

              sseManager.broadcast({
                type: "agent_activity",
                data: {
                  agentId,
                  name: agentName,
                  status: "vm_writing",
                  message: `${agentName} escribiendo archivo: ${path}`,
                  timestamp: new Date().toISOString(),
                },
              });

              await writeFile(agentId, path, content);
              toolResult = `Archivo escrito: ${path} (${content.length} bytes)`;
              break;
            }

            case "read_file": {
              const path = args.path as string;
              const content = await readFile(agentId, path);
              toolResult = content;
              break;
            }

            default:
              toolResult = `Herramienta desconocida: ${funcName}`;
          }
        } catch (error) {
          toolResult = `Error: ${error instanceof Error ? error.message : String(error)}`;
        }

        // Add tool result to messages
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: toolResult,
        });
      }
    }
  }

  // Safety limit reached - ask for final summary
  try {
    messages.push({
      role: "user",
      content:
        "Has alcanzado el límite de acciones por ciclo. Llama a final_response AHORA con tu pensamiento, estrategia y solicitudes.",
    });
    const finalResponse = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      max_tokens: 2048,
      tools: [FINAL_RESPONSE_TOOL],
      tool_choice: {
        type: "function",
        function: { name: "final_response" },
      },
      messages,
    });

    if (finalResponse.usage) {
      totalCost += calculateCost(finalResponse.usage);
    }

    const finalChoice = finalResponse.choices[0];
    if (finalChoice?.message.tool_calls?.length) {
      const tc = finalChoice.message.tool_calls[0];
      const args = JSON.parse(tc.function.arguments);
      return {
        thought:
          (args.thought as string) || "Ciclo alcanzó límite de seguridad.",
        strategy_update: (args.strategy_update as string) || null,
        requests: Array.isArray(args.requests)
          ? (args.requests as AgentThoughtResponse["requests"]).slice(0, 3)
          : [],
        apiCost: totalCost,
        toolsUsed,
      };
    }
  } catch {
    // If this also fails, fall through
  }

  return {
    thought: "Ciclo alcanzó el límite de seguridad de acciones.",
    strategy_update: null,
    requests: [],
    apiCost: totalCost,
    toolsUsed,
  };
}
