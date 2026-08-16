const MAX_BATCH_SIZE = 1000;
const MAX_TOP_K = 20;
const MAX_METADATA_BYTES = 10 * 1024;
const CHUNK_SIZE = 100;

function corsHeaders(request) {
  const origin = request.headers.get("Origin");

  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin"
  };
}

function json(request, data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(request)
    }
  });
}

function errorMessage(error) {
  return error instanceof Error
    ? error.message
    : String(error);
}

function isPlainObject(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);

  return (
    prototype === Object.prototype ||
    prototype === null
  );
}

function metadataSize(metadata) {
  return new TextEncoder()
    .encode(JSON.stringify(metadata))
    .byteLength;
}

function validateMetadata(metadata) {
  if (!isPlainObject(metadata)) {
    return "metadata must be a plain JSON object";
  }

  for (const key of Object.keys(metadata)) {
    if (
      key.includes(".") ||
      key.includes('"') ||
      key.startsWith("$")
    ) {
      return `Invalid metadata key: ${key}`;
    }
  }

  if (metadataSize(metadata) > MAX_METADATA_BYTES) {
    return "metadata exceeds 10 KiB";
  }

  return null;
}

function validateQuestion(question) {
  if (!isPlainObject(question)) {
    return {
      valid: false,
      id: null,
      message: "question must be an object"
    };
  }

  if (
    typeof question.id !== "string" ||
    !question.id.trim()
  ) {
    return {
      valid: false,
      id: question.id ?? null,
      message: "id must be a non-empty string"
    };
  }

  if (
    typeof question.text !== "string" ||
    !question.text.trim()
  ) {
    return {
      valid: false,
      id: question.id,
      message: "text must be a non-empty string"
    };
  }

  const metadata = question.metadata ?? {};

  const metadataError = validateMetadata(metadata);

  if (metadataError) {
    return {
      valid: false,
      id: question.id,
      message: metadataError
    };
  }

  if (
    typeof metadata.subject !== "string" ||
    !metadata.subject.trim()
  ) {
    return {
      valid: false,
      id: question.id,
      message: "metadata.subject must be a non-empty string"
    };
  }

  return {
    valid: true,
    id: question.id,
    text: question.text,
    metadata,
    namespace: metadata.subject
  };
}

/*
 * Pinecone integrated-embedding upsert.
 *
 * `text` is sent to Pinecone.
 * Pinecone's llama-text-embed-v2 model performs the embedding.
 */
async function pineconeUpsert(env, namespace, records) {
  const url =
    `https://${env.PINECONE_HOST}` +
    `/records/namespaces/${encodeURIComponent(namespace)}/upsert`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Api-Key": env.PINECONE_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      records
    })
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result?.message ||
      result?.error ||
      `Pinecone returned HTTP ${response.status}`
    );
  }

  return result;
}

/*
 * Pinecone integrated-embedding search.
 *
 * Pinecone embeds `text` using llama-text-embed-v2
 * and searches the specified namespace.
 */
async function pineconeSearch(
  env,
  namespace,
  text,
  topK
) {
  const url =
    `https://${env.PINECONE_HOST}` +
    `/records/namespaces/${encodeURIComponent(namespace)}/search`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Api-Key": env.PINECONE_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query: {
        top_k: topK,
        inputs: {
          text
        }
      }
    })
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result?.message ||
      result?.error ||
      `Pinecone returned HTTP ${response.status}`
    );
  }

  return result;
}

/*
 * POST /questions
 */
async function handleQuestions(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return json(request, {
      error: "Malformed JSON"
    }, 400);
  }

  /*
   * Top-level validation.
   * No Pinecone writes occur if this fails.
   */
  if (!isPlainObject(body)) {
    return json(request, {
      error: "Request body must be a JSON object"
    }, 400);
  }

  if (!Array.isArray(body.questions)) {
    return json(request, {
      error: "questions must be an array"
    }, 400);
  }

  if (
    body.questions.length < 1 ||
    body.questions.length > MAX_BATCH_SIZE
  ) {
    return json(request, {
      error:
        `questions must contain between 1 and ${MAX_BATCH_SIZE} items`
    }, 400);
  }

  const results = new Array(
    body.questions.length
  );

  const validQuestions = [];

  /*
   * Validate each question independently.
   */
  body.questions.forEach((question, index) => {
    const validation =
      validateQuestion(question);

    if (!validation.valid) {
      results[index] = {
        id: validation.id,
        status: "error",
        message: validation.message
      };

      return;
    }

    validQuestions.push({
      index,
      id: validation.id,
      text: validation.text,
      metadata: validation.metadata,
      namespace: validation.namespace
    });
  });

  /*
   * Group questions by metadata.subject.
   *
   * Biology → Biology namespace
   * Chemistry → Chemistry namespace
   * Physics → Physics namespace
   */
  const groups = new Map();

  for (const question of validQuestions) {
    if (!groups.has(question.namespace)) {
      groups.set(question.namespace, []);
    }

    groups
      .get(question.namespace)
      .push(question);
  }

  /*
   * Process each namespace independently.
   */
  for (const [namespace, questions] of groups) {
    for (
      let offset = 0;
      offset < questions.length;
      offset += CHUNK_SIZE
    ) {
      const chunk = questions.slice(
        offset,
        offset + CHUNK_SIZE
      );

      /*
       * Pinecone integrated embedding expects
       * `_id` and `text`.
       *
       * Metadata is stored alongside the record.
       */
      const records = chunk.map(question => ({
        _id: question.id,
        text: question.text,
        ...question.metadata
      }));

      try {
        await pineconeUpsert(
          env,
          namespace,
          records
        );

        for (const question of chunk) {
          results[question.index] = {
            id: question.id,
            status: "ok"
          };
        }

      } catch (error) {
        const message =
          `Pinecone upsert failed: ${errorMessage(error)}`;

        for (const question of chunk) {
          results[question.index] = {
            id: question.id,
            status: "error",
            message
          };
        }
      }
    }
  }

  return json(request, {
    results,
    mutationId: crypto.randomUUID()
  });
}

/*
 * POST /search
 */
async function handleSearch(request, env) {
  let body;

  try {
    body = await request.json();
  } catch {
    return json(request, {
      error: "Malformed JSON"
    }, 400);
  }

  if (!isPlainObject(body)) {
    return json(request, {
      error: "Request body must be a JSON object"
    }, 400);
  }

  if (
    typeof body.text !== "string" ||
    !body.text.trim()
  ) {
    return json(request, {
      error: "text must be a non-empty string"
    }, 400);
  }

  if (
    body.topK !== undefined &&
    !Number.isInteger(body.topK)
  ) {
    return json(request, {
      error: "topK must be an integer"
    }, 400);
  }

  const topK = Math.min(
    Math.max(body.topK ?? 5, 1),
    MAX_TOP_K
  );

  /*
   * The client must provide the subject because
   * subject determines the Pinecone namespace.
   */
  if (
    typeof body.subject !== "string" ||
    !body.subject.trim()
  ) {
    return json(request, {
      error: "subject must be a non-empty string"
    }, 400);
  }

  let result;

  try {
    result = await pineconeSearch(
      env,
      body.subject,
      body.text,
      topK
    );
  } catch (error) {
    return json(request, {
      error:
        `Pinecone search failed: ${errorMessage(error)}`
    }, 502);
  }

  /*
   * Pinecone integrated search returns hits.
   */
  const matches = (result?.result?.hits || [])
    .map(hit => ({
      id: hit._id,
      score: hit._score,
      metadata: extractMetadata(hit.fields)
    }))
    .sort((a, b) => b.score - a.score);

  return json(request, {
    matches
  });
}

/*
 * Pinecone integrated-record responses contain the
 * text field alongside metadata fields.
 *
 * We return only the metadata to the client.
 */
function extractMetadata(fields) {
  if (!fields || typeof fields !== "object") {
    return {};
  }

  const metadata = { ...fields };

  delete metadata.text;

  return metadata;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    /*
     * CORS preflight.
     */
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(request)
      });
    }

    /*
     * POST /questions
     */
    if (url.pathname === "/questions") {
      if (request.method !== "POST") {
        return json(request, {
          error: "Method not allowed"
        }, 405);
      }

      return handleQuestions(request, env);
    }

    /*
     * POST /search
     */
    if (url.pathname === "/search") {
      if (request.method !== "POST") {
        return json(request, {
          error: "Method not allowed"
        }, 405);
      }

      return handleSearch(request, env);
    }

    /*
     * GET /health
     */
    if (url.pathname === "/health") {
      if (request.method !== "GET") {
        return json(request, {
          error: "Method not allowed"
        }, 405);
      }

      return json(request, {
        status: "ok",
        service: "utme-questions-worker"
      });
    }

    return json(request, {
      error: "Not found"
    }, 404);
  }
};
