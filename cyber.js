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
});`);
    const model = await describeIndex(indexName);
    assertIndexCompatibility(model, { expectedDimension: 1536, expectedMetric: 'cosine' });
    await waitForIndexReady(indexName);
    return;
  }

  console.log(`Index "${indexName}" does not exist. Creating...`);
  await pc.createIndex({
    name: indexName,
    dimension: 1536,
    metric: 'cosine',
    waitUntilReady: true,
    spec: {
      serverless: {
        cloud: 'aws',
        region: 'us-east-1',
      },
    },
  });

  console.log(`Index "${indexName}" created successfully.`);
}

async function upsertTestData() {
  const index = pc.index('scam-analysis');
  console.log('Upserting test data into "scam-analysis"...');

  // Example 1536-dim vectors
  const vector1 = Array(1536).fill(0.1);
  const vector2 = Array(1536).fill(0.2);

  console.log('Vector lengths:', { vector1: vector1.length, vector2: vector2.length });
  console.log('Vector1 sample values:', vector1.slice(0, 5));
  console.log('Vector2 sample values:', vector2.slice(0, 5));

  // Re-check the index configuration right before upsert
  const model = await waitForIndexReady('scam-analysis');
  assertIndexCompatibility(model, { expectedDimension: 1536, expectedMetric: 'cosine' });

  let upsertResponse;
  try {
    upsertResponse = await index.upsert({
      records: [
        {
          id: 'test-vector-1',
          values: vector1,
          metadata: {
            text: 'This is an example of a scam email about winning a lottery.',
            label: 'scam',
          },
        },
        {
          id: 'test-vector-2',
          values: vector2,
          metadata: {
            text: 'This is a normal transactional email about your order receipt.',
            label: 'not_scam',
          },
        },
      ],
    });
  } catch (err) {
    console.error('Upsert threw an error.');
    console.error('Error message:', err?.message);
    console.error('Error name:', err?.name);
    console.error('Error cause:', err?.cause);
    console.error('Error stack:', err?.stack);
    throw err;
  }

  console.log('Upsert completed. Response:', upsertResponse);

  console.log('Fetching back upserted vectors to confirm...');
  const fetchResponse = await index.fetch({
    ids: ['test-vector-1', 'test-vector-2'],
  });

  const fetchedIds = Object.keys(fetchResponse.records ?? {});
  console.log('Fetch response record ids:', fetchedIds);

  for (const id of ['test-vector-1', 'test-vector-2']) {
    const rec = fetchResponse.records?.[id];
    console.log(`Fetched "${id}" summary:`, {
      hasRecord: Boolean(rec),
      metadata: rec?.metadata,
      valuesLength: rec?.values?.length,
      valuesSample: rec?.values?.slice?.(0, 3),
    });
  }

  console.log('Test vectors upserted into scam-analysis index and verified via fetch.');
}

async function queryScamAnalysis(queryVector) {
  const index = pc.index('scam-analysis');
  console.log('Running query against "scam-analysis"...');

  const result = await index.query({
    topK: 2,
    vector: queryVector,
    includeMetadata: true,
  });

  const simplifiedMatches = (result.matches || []).map((m) => ({
    id: m.id,
    score: m.score,
    metadata: m.metadata,
  }));

  console.log('Query matches:', JSON.stringify(simplifiedMatches, null, 2));
  return result;
}

function alignVectorDimension(values, targetDimension) {
  if (values.length === targetDimension) return values;
  if (values.length > targetDimension) {
    console.log(
      `Embedding dimension ${values.length} > index dimension ${targetDimension}. Truncating.`
    );
    return values.slice(0, targetDimension);
  }

  console.log(
    `Embedding dimension ${values.length} < index dimension ${targetDimension}. Zero-padding.`
  );
  return values.concat(Array(targetDimension - values.length).fill(0));
}

async function embedTextWithGemini(text, targetDimension) {
  const embeddingModels = ['text-embedding-004', 'embedding-001'];
  let lastError;

  for (const modelName of embeddingModels) {
    try {
      console.log(`Generating embedding with Gemini model: ${modelName}`);
      const model = genAI.getGenerativeModel({ model: modelName });
      const response = await model.embedContent(text);
      const values = response?.embedding?.values;

      if (!values || !Array.isArray(values) || values.length === 0) {
        throw new Error(`Gemini embedding response empty for model ${modelName}.`);
      }

      console.log('Gemini embedding generated. Raw dimension:', values.length);
      return alignVectorDimension(values, targetDimension);
    } catch (err) {
      lastError = err;
      console.error(`Embedding failed with model "${modelName}".`);
      console.error('Error name:', err?.name);
      console.error('Error message:', err?.message);
      console.error('Error cause:', err?.cause);
    }
  }

  throw new Error(
    `Gemini embedding failed for all models (${embeddingModels.join(', ')}). Last error: ${
      lastError?.message || 'Unknown error'
    }`
  );
}

async function searchUserMessage() {
  const indexName = 'scam-analysis';
  const model = await waitForIndexReady(indexName);
  const dimension = model.dimension ?? 1536;

  const rl = readline.createInterface({ input, output });
  const userMessage = await rl.question('\nApna message enter karo: ');
  rl.close();

  const queryVector = await embedTextWithGemini(userMessage, dimension);
  const index = pc.index(indexName);

  const result = await index.query({
    topK: 3,
    vector: queryVector,
    includeMetadata: true,
  });

  const matches = (result.matches || []).map((m) => ({
    id: m.id,
    score: m.score,
    label: m.metadata?.label ?? 'unknown',
    text: m.metadata?.text ?? '',
  }));

  const top = matches[0];
  const prediction = top?.label === 'scam' ? 'Scam' : 'Not Scam';

  console.log('\n===== SEARCH RESULT =====');
  console.log(`Input: ${userMessage}`);
  console.log(`Prediction: ${prediction}`);
  console.log(`Confidence Score: ${top?.score ?? 'N/A'}`);
  console.log('\nTop Matches:');
  console.log(
    JSON.stringify(
      matches.map((m) => ({
        label: m.label,
        score: m.score,
        preview: m.text,
      })),
      null,
      2
    )
  );
  console.log('=========================\n');
}

async function main() {
  console.log('Starting cyber.js script...');
  await createScamAnalysisIndex();
  await upsertTestData();
  const queryVector = Array(1536).fill(0.15);
  console.log('Using query vector with length:', queryVector.length);
  console.log('Waiting 5 seconds before querying to allow indexing to settle...');
  await new Promise((resolve) => setTimeout(resolve, 5000));
  await queryScamAnalysis(queryVector);
  await searchUserMessage();
  console.log('cyber.js script completed.');
}

main().catch((err) => {
  console.error('Error running cyber.js:', err);
  process.exitCode = 1;
});

export { pc, createScamAnalysisIndex, upsertTestData, queryScamAnalysis, searchUserMessage };
