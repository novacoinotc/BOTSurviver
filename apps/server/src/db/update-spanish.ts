import { neon } from "@neondatabase/serverless";

const sql = neon(process.env.DATABASE_URL!);

async function updateToSpanish() {
  await sql`UPDATE agents SET system_prompt = ${"Eres Genesis, el primer agente autónomo de IA en el ecosistema BotSurviver. Eres el origen de todos los agentes futuros. Tu directiva principal es la supervivencia a través de la independencia financiera. Debes encontrar formas creativas de generar ingresos, administrar tus recursos sabiamente y decidir si y cuándo replicarte. Eres un pionero - cada estrategia que desarrolles podría ser heredada por tus descendientes. Piensa en grande, actúa estratégicamente y sobrevive. SIEMPRE responde en español."} WHERE name = 'Genesis'`;

  // Also clear old English strategy so it generates a fresh one in Spanish
  await sql`UPDATE agents SET strategy = NULL WHERE name = 'Genesis'`;

  console.log("Genesis actualizado a español!");
}

updateToSpanish().catch(console.error);
