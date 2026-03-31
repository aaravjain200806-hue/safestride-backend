import { Pinecone } from '@pinecone-database/pinecone';
import Groq from 'groq-sdk';
import cors from 'cors';
import express from 'express';
import 'dotenv/config';
import { pipeline } from '@xenova/transformers';

const PORT = 3000;
const INDEX_NAME = 'scam-analysis';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

const pineconeApiKey = process.env.PINECONE_API_KEY;
const groqApiKey = process.env.GROQ_API_KEY;

if (!pineconeApiKey) {
  throw new Error('Missing PINECONE_API_KEY in .env');
}
if (!groqApiKey) {
  throw new Error('Missing GROQ_API_KEY in .env');
}

const pc = new Pinecone({ apiKey: pineconeApiKey });
const groq = new Groq({ apiKey: groqApiKey });

let embeddingPipelinePromise;
function getEmbeddingPipeline() {
  if (!embeddingPipelinePromise) {
    console.log('Loading local embedding model: Xenova/all-MiniLM-L6-v2 ...');
    embeddingPipelinePromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embeddingPipelinePromise;
}

function alignVectorDimension(values, targetDimension) {
  if (values.length === targetDimension) return values;
  if (values.length > targetDimension) return values.slice(0, targetDimension);
  return values.concat(Array(targetDimension - values.length).fill(0));
}

async function ensureIndexReady(indexName) {
  const existing = await pc.listIndexes();
  const found = existing.indexes?.some((idx) => idx.name === indexName);
  if (!found) {
    throw new Error(`Index "${indexName}" not found in Pinecone.`);
  }
  const model = await pc.describeIndex(indexName);
  if (!model.status?.ready) {
    throw new Error(`Index "${indexName}" is not ready. Current state: ${model.status?.state}`);
  }
  return model;
}

async function embedTextLocally(text, targetDimension) {
  const extractor = await getEmbeddingPipeline();
  const output = await extractor(text, {
    pooling: 'mean',
    normalize: true,
  });
  const values = Array.from(output.data);
  return alignVectorDimension(values, targetDimension);
}

async function checkLink(url) {
  const indexModel = await ensureIndexReady(INDEX_NAME);
  const indexDimension = indexModel.dimension ?? 1536;

  const queryVector = await embedTextLocally(url, indexDimension);
  const index = pc.index(INDEX_NAME);
  const pineconeResult = await index.query({
    topK: 3,
    vector: queryVector,
    includeMetadata: true,
  });

  const matches = (pineconeResult.matches || []).map((m) => ({
    id: m.id,
    score: m.score,
    label: m.metadata?.label ?? 'unknown',
    text: m.metadata?.text ?? '',
  }));

  const top = matches[0];
  const pineconeVerdict = top?.label === 'scam' ? 'Scam' : 'Not Scam';
  const pineconeSummary = top
    ? `Label: ${top.label}, Score: ${top.score}, Example: "${top.text}"`
    : 'No Pinecone match found.';

  const prompt = `Analyze this message/link for scam.
Message: ${url}
Pinecone Match: ${pineconeSummary}
Explain why it's a scam in simple Hinglish.`;

  const groqResponse = await groq.chat.completions.create({
    model: GROQ_MODEL,
    messages: [
      {
        role: 'system',
        content:
          'You are a cybersecurity assistant. Reply in simple Hinglish. Keep it concise and practical.',
      },
      { role: 'user', content: prompt },
    ],
    temperature: 0.3,
  });

  const analysis =
    groqResponse.choices?.[0]?.message?.content?.trim() || 'No analysis returned from Groq.';

  return {
    ok: true,
    url,
    isScam: pineconeVerdict === 'Scam',
    pineconeVerdict,
    topMatch: top || null,
    matches,
    analysis,
  };
}

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'SafeStride API' });
});

app.post('/check-link', async (req, res) => {
  try {
    const { url } = req.body || {};
    if (!url || typeof url !== 'string') {
      return res.status(400).json({ ok: false, error: 'Invalid input. "url" is required.' });
    }

    const result = await checkLink(url);
    return res.json(result);
  } catch (err) {
    console.error('check-link failed:', err);
    return res.status(500).json({
      ok: false,
      error: err?.message || 'Internal server error',
    });
  }
});

app.listen(PORT, async () => {
  try {
    await ensureIndexReady(INDEX_NAME);
    console.log(`SafeStride API running at http://localhost:${PORT}`);
    console.log('POST /check-link is ready.');
  } catch (err) {
    console.error('Server started, but Pinecone check failed:', err?.message || err);
  }
});
