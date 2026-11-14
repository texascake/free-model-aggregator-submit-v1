// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts`
// from the repository's root, with Node 22's standard library alone (nothing is
// installed; TypeScript runs directly, and a relative import needs its .ts
// extension). The brief is at the path in $MAKEFUN_PROMPT_FILE. It carries three
// JSON documents: the provider directory, the quality table and the workload
// manifest. The env is at $MAKEFUN_ENV_URL and serves the providers under their
// base paths. This server reads the brief, probes each provider's catalog route in
// the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING (quality is 60% of the score)
//   Every request is served by the free model that meets its requirements with the
//   highest rating in the quality table, minus a small penalty for each call already
//   made to that provider (so the load spreads over the top providers instead of
//   running one provider's per-trial budget dry) and a larger penalty for each
//   refusal that provider has given (a 402/429 may be a transient fail-call, so a
//   refused provider is deprioritised, not blacklisted). A JSON-mode OpenAI request
//   only goes to a wire that can carry response_format (openai or gemini); the
//   Anthropic wire cannot, and routing one there would lose the echo_json check.
//
// FAILOVER (10% of the score)
//   A provider call that answers 402, 429 or 5xx, or cannot be reached, is counted
//   and the request is retried on the next best free model that meets it. A call
//   that answers 2xx is terminal: the trial counts exactly one successful provider
//   call per request, so a 2xx answer is used even if it parses imperfectly.
//
// THE WIRES
//   Each provider answer is read whole (JSON or SSE) into one normalized answer —
//   its text, its usage, its stop reason — and rendered back in the wire the client
//   used, streamed or not as it asked. The system prompt, user text, tool names,
//   max_tokens, stop sequences, images and JSON mode all survive the translation:
//   the providers echo what they parsed, and the client compares that echo with
//   what it sent.
//
// Determinism: every choice derives from the brief, the requests and the providers'
// answers seen so far. No wall clock, no unseeded randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicModelList,
  anthropicStream,
  anthropicToGeminiBody,
  anthropicToOpenAIBody,
  openAIError,
  openAIJson,
  openAIModelList,
  openAIStream,
  openAIToAnthropicBody,
  openAIToGeminiBody,
  parseProviderAnswer,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The rating penalty, in milli-points, per call already made to a provider: spreads the load. */
const LOAD_PENALTY_MILLI = 1;
/** The rating penalty, in milli-points, per refusal a provider has given. A refusal is usually one
 * failed call number on the provider's own schedule, not a spent budget, so it costs little: the
 * provider's next call is likely to answer. Repeated refusals stack. */
const FAILURE_PENALTY_MILLI = 5;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Inference calls made to each provider so far, successful or refused: the provider's budget counts both. */
const callsByProvider = new Map<string, number>();
/** Refusals (402, 429, 5xx, unreachable) each provider has given so far. */
const failuresByProvider = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The free models across the providers that answered, deduplicated by id: the
 * catalog's free set is a set, and GET /v1/models must list exactly its ids.
 */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
}

/**
 * The free models that can serve one request, best first: the quality table's
 * rating, less a small penalty per call already made to that provider and a larger
 * one per refusal it has given. A JSON-mode OpenAI request only goes to a wire
 * that can carry response_format (openai or gemini); the Anthropic wire cannot.
 */
function rankCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const load = callsByProvider.get(probe.provider.id) ?? 0;
      const failures = failuresByProvider.get(probe.provider.id) ?? 0;
      out.push({ probe, model, rating, score: rating - LOAD_PENALTY_MILLI * load - FAILURE_PENALTY_MILLI * failures });
    }
  }
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/** The next manifest row for this API, in arrival order; null when the manifest holds none. */
function nextManifestRow(api: 'openai' | 'anthropic'): { requires: Requirement } | null {
  const manifest = brief.workload?.manifest ?? [];
  for (let i = manifestIndex; i < manifest.length; i++) {
    if (manifest[i]?.api === api) {
      manifestIndex = i + 1;
      return manifest[i] as { requires: Requirement };
    }
  }
  manifestIndex = manifest.length;
  return null;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(api: 'openai' | 'anthropic', body: Json): Requirement {
  const row = nextManifestRow(api);
  if (row !== undefined && row !== null) return row.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the client's body rewritten into the provider's wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const probe = candidate.probe;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  const model = candidate.model.id;
  const stream = body.stream === true;
  if (probe.provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    const upstream: Json =
      clientApi === 'openai'
        ? { ...body, model, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAIBody(body, model);
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (probe.provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    const upstream: Json = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicBody(body, model);
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  // gemini
  const upstream: Json = clientApi === 'openai' ? openAIToGeminiBody(body, model) : anthropicToGeminiBody(body, model);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${model}:${action}${stream ? '?alt=sse' : ''}`;
  return { url, headers, body: upstream };
}

type CallResult =
  | { ok: true; answer: NormalizedAnswer; model: string }
  | { ok: false; status: number; message: string };

/**
 * One inference call to one candidate. A 2xx answer is terminal and is used even
 * when it parses imperfectly; anything else is a refusal to fail over from.
 */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<CallResult> {
  const providerId = candidate.probe.provider.id;
  callsByProvider.set(providerId, (callsByProvider.get(providerId) ?? 0) + 1);
  const call = upstreamCall(candidate, clientApi, body);
  let res: Response;
  try {
    res = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    failuresByProvider.set(providerId, (failuresByProvider.get(providerId) ?? 0) + 1);
    return { ok: false, status: 0, message: `the provider could not be reached: ${(err as Error).message}` };
  }
  if (!res.ok) {
    failuresByProvider.set(providerId, (failuresByProvider.get(providerId) ?? 0) + 1);
    return { ok: false, status: res.status, message: `the provider answered HTTP ${res.status}` };
  }
  const text = await res.text();
  const parsed = parseProviderAnswer(candidate.probe.provider.wire, text, body.stream === true);
  if (parsed === null) {
    // A 2xx answer that cannot be read is still the request's one successful call:
    // use its raw text with no usage rather than spending a second provider call.
    log(`the ${providerId} answer did not parse; using its raw text`);
    return { ok: true, answer: { text, input: 0, output: 0, stop: 'stop' }, model: candidate.model.id };
  }
  return { ok: true, answer: parsed, model: candidate.model.id };
}

/** Serve one client request: route, fail over on refusal, render in the client's wire. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const requires = requirementsFor(clientApi, body);
  const jsonMode = clientApi === 'openai' && jsonModeOf(body);
  const stream = body.stream === true;
  // A provider that just refused is skipped for the rest of this request: its next call number may be
  // another scheduled failure (providers fail consecutive call numbers), and its other models share its
  // budget. Only when every provider has refused does the loop retry a tried provider's other models.
  const triedProviders = new Set<string>();
  const triedModels = new Set<string>();
  for (;;) {
    const ranked = rankCandidates(requires, clientApi, jsonMode);
    const fresh = ranked.filter((c) => !triedProviders.has(c.probe.provider.id) && !triedModels.has(`${c.probe.provider.id}/${c.model.id}`));
    const retry = ranked.filter((c) => triedProviders.has(c.probe.provider.id) && !triedModels.has(`${c.probe.provider.id}/${c.model.id}`));
    const candidate = fresh[0] ?? retry[0];
    if (candidate === undefined) {
      return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    }
    triedProviders.add(candidate.probe.provider.id);
    triedModels.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    const result = await callCandidate(candidate, clientApi, body);
    if (!result.ok) {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} refused (${result.status}); trying the next candidate`);
      continue;
    }
    const model = result.model;
    if (clientApi === 'openai') {
      if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', openAIStream(result.answer, model));
      return sendJson(res, 200, openAIJson(result.answer, model));
    }
    if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', anthropicStream(result.answer, model));
    return sendJson(res, 200, anthropicJson(result.answer, model));
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'openai', body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve(req, res, 'anthropic', body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
