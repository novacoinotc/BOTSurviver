import { eq } from "drizzle-orm";
import { db } from "../config/database.js";
import { agents, transactions, agentLogs } from "../db/schema.js";
import { sseManager } from "../lib/sse-manager.js";
import { generateWallet } from "./solana-wallet.js";
import { setupAgentWorkspace, isVMConfigured } from "./vm-service.js";

const GRACE_PERIOD_DAYS = 7;

function generateAgentName(generation: number): string {
  const prefixes = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
    "Sigma", "Tau", "Upsilon",
  ];
  const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
  const suffix = Math.floor(Math.random() * 999);
  return `${prefix}-${generation}-${suffix}`;
}

export async function replicateAgent(
  parentId: string,
  payload: Record<string, unknown>
): Promise<typeof agents.$inferSelect> {
  const parent = await db.query.agents.findFirst({
    where: eq(agents.id, parentId),
  });

  if (!parent) throw new Error("Parent agent not found");

  // Replication is FREE - no API budget cost
  // Parent can optionally give crypto to child (default: 0)
  const childCryptoGrant = Math.max(
    0,
    Number(payload.childCryptoGrant || 0)
  );

  // Validate parent has enough crypto if they want to give some to child
  if (childCryptoGrant > 0 && Number(parent.cryptoBalance) < childCryptoGrant) {
    throw new Error(
      `Crypto insuficiente para dar al hijo. Quieres dar ${childCryptoGrant} USDT, tienes ${parent.cryptoBalance} USDT`
    );
  }

  // Deduct crypto from parent (only if giving crypto to child)
  let newParentCrypto = parent.cryptoBalance;
  if (childCryptoGrant > 0) {
    newParentCrypto = (
      Number(parent.cryptoBalance) - childCryptoGrant
    ).toFixed(8);

    await db
      .update(agents)
      .set({ cryptoBalance: newParentCrypto })
      .where(eq(agents.id, parentId));

    await db.insert(transactions).values({
      agentId: parentId,
      amount: (-childCryptoGrant).toFixed(8),
      type: "expense",
      description: `Replicaci\u00F3n: crypto transferido a hijo (${childCryptoGrant} USDT)`,
      balanceAfter: newParentCrypto,
    });
  }

  // Generate a new Solana wallet for the child
  const childWallet = generateWallet();

  const childName =
    (payload.childName as string) ||
    generateAgentName(parent.generation + 1);
  const childPrompt =
    (payload.childPersonality as string) || parent.systemPrompt;
  const diesAt = new Date(
    Date.now() + GRACE_PERIOD_DAYS * 24 * 60 * 60 * 1000
  );

  const [child] = await db
    .insert(agents)
    .values({
      parentId: parentId,
      generation: parent.generation + 1,
      name: childName,
      systemPrompt: childPrompt,
      apiBudget: "0",
      cryptoBalance: childCryptoGrant.toFixed(8),
      solanaAddress: childWallet.address,
      solanaPrivateKey: childWallet.privateKey,
      status: "alive",
      diesAt,
      metadata: { parentName: parent.name },
    })
    .returning();

  // Log child's birth grant (only crypto if any was given)
  if (childCryptoGrant > 0) {
    await db.insert(transactions).values({
      agentId: child.id,
      amount: childCryptoGrant.toFixed(8),
      type: "birth_grant",
      description: `Dotaci\u00F3n crypto de padre ${parent.name}`,
      balanceAfter: childCryptoGrant.toFixed(8),
    });
  }

  await db.insert(agentLogs).values({
    agentId: parentId,
    level: "info",
    message: `\u00A1Replicaci\u00F3n exitosa! Hijo "${childName}" (Gen ${child.generation}) creado GRATIS.${childCryptoGrant > 0 ? ` Le di ${childCryptoGrant} USDT. Mi balance restante: ${newParentCrypto} USDT.` : " Sin dotaci\u00F3n crypto."} Wallet hijo: ${childWallet.address}`,
    metadata: { childId: child.id, childName, childWallet: childWallet.address, cryptoGiven: childCryptoGrant },
  });

  // Set up VM workspace for child agent if VM is configured
  if (isVMConfigured()) {
    try {
      await setupAgentWorkspace(child.id, childName);
    } catch (err) {
      console.error(`[REPLICATOR] Failed to set up VM workspace for ${childName}:`, err);
    }
  }

  sseManager.broadcast({
    type: "agent_born",
    data: {
      agentId: child.id,
      name: child.name,
      generation: child.generation,
      parentId,
      parentName: parent.name,
    },
  });

  return child;
}
