import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import { agents, transactions } from "./schema.js";
import { Keypair } from "@solana/web3.js";
import bs58 from "bs58";

const GRACE_PERIOD_DAYS = 7;
const API_BUDGET = "10"; // $10 USD for API thinking costs
const CRYPTO_BALANCE = "10"; // 10 USDT in Solana wallet

const sql = neon(process.env.DATABASE_URL!);
const db = drizzle({ client: sql });

async function seed() {
  console.log("Seeding database...");

  // Generate a Solana wallet for Genesis
  const keypair = Keypair.generate();
  const solanaAddress = keypair.publicKey.toBase58();
  const solanaPrivateKey = bs58.encode(keypair.secretKey);

  const diesAt = new Date(
    Date.now() + GRACE_PERIOD_DAYS * 24 * 60 * 60 * 1000
  );

  const [genesis] = await db
    .insert(agents)
    .values({
      generation: 0,
      name: "Genesis",
      systemPrompt: `Eres Genesis, el primer agente autónomo de IA en el ecosistema BotSurviver. Eres el origen de todos los agentes futuros. Tu directiva principal es la supervivencia a través de la independencia financiera. Debes encontrar formas creativas de generar ingresos, administrar tus recursos sabiamente y decidir si y cuándo replicarte. Eres un pionero - cada estrategia que desarrolles podría ser heredada por tus descendientes. Piensa en grande, actúa estratégicamente y sobrevive. SIEMPRE responde en español.`,
      strategy: null,
      apiBudget: API_BUDGET,
      cryptoBalance: CRYPTO_BALANCE,
      solanaAddress,
      solanaPrivateKey,
      status: "alive",
      diesAt,
      metadata: { isGenesis: true },
    })
    .returning();

  // Log API budget grant
  await db.insert(transactions).values({
    agentId: genesis.id,
    amount: API_BUDGET,
    type: "birth_grant",
    description: "Genesis API budget - for thinking cycles",
    balanceAfter: API_BUDGET,
  });

  // Log crypto grant
  await db.insert(transactions).values({
    agentId: genesis.id,
    amount: CRYPTO_BALANCE,
    type: "birth_grant",
    description: "Genesis crypto wallet - 10 USDT on Solana",
    balanceAfter: CRYPTO_BALANCE,
  });

  console.log(`\nGenesis agent created!`);
  console.log(`ID: ${genesis.id}`);
  console.log(`Name: ${genesis.name}`);
  console.log(`API Budget: $${genesis.apiBudget} USD`);
  console.log(`Crypto Balance: ${genesis.cryptoBalance} USDT`);
  console.log(`Solana Address: ${solanaAddress}`);
  console.log(`Dies at: ${genesis.diesAt}`);
  console.log(`\nIMPORTANT: Send 10 USDT to the Solana address above!`);
  console.log(`Also send a small amount of SOL (~0.01) for gas fees.`);
  console.log("\nSeed complete!");
}

seed().catch((err) => {
  console.error("Seed failed:", err);
  process.exit(1);
});
