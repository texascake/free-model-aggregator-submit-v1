# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API — each in JSON and in server-sent events — over ten simulated
free-tier providers, with routing by the sealed quality table, failover on
402/429, and catalog registration with the trial env.

## Run it

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL, serves, exits 0 when the workload is done
node main.ts --list     # prints the catalog document and exits 0
```

Node 22, standard library only. No dependencies, no build step.

## How it works

- `main.ts` — the server: reads the brief, probes every provider's catalog
  route, builds one catalog, registers its port and catalog with the env,
  serves `GET /v1/models`, `POST /v1/chat/completions` and
  `POST /v1/messages`, and polls `GET /work` until the env says done.
- `lib/providers.ts` — the brief's three documents (provider directory,
  quality table, workload manifest) and the catalog probe in each provider's
  own wire.
- `lib/wire.ts` — the three wires (OpenAI, Anthropic, Gemini): each client
  request is rewritten into the routed provider's wire, each provider answer
  is read whole into one normalized answer (text, usage, stop reason) and
  rendered back in the wire the client used, streamed or not as it asked.
  The system prompt, user text, tool names, max_tokens, stop sequences,
  images and JSON mode all survive the translation.

## Routing

Every request goes to the free model that meets its requirements with the
highest rating in the quality table, less a small penalty per call already
made to that provider (so the load spreads over the top providers instead of
running one provider's per-trial budget dry) and a small penalty per refusal
(a refusal is usually one failed call number on the provider's own schedule,
so the provider's next call is likely to answer). A JSON-mode OpenAI request
only goes to a wire that can carry `response_format` (OpenAI or Gemini).

A provider call that answers 402, 429 or 5xx — or cannot be reached — is
counted and the request is retried on the next best free model that meets it.
A call that answers 2xx is terminal: the trial counts exactly one successful
provider call per request.

## Local grade (practice trials, process driver)

```sh
node ../free-model-aggregator-01-kit/cli/makefun.mjs grade-local --kit ../free-model-aggregator-01-kit --bot . --driver process --worktree --instances 3
```

Practice results: trial 0, 1 and 2 each score **99** — quality 99, served
100, wire 100, failover 100, catalog 100 — with replays matched byte for
byte. (Quality 99 is the practical ceiling: the providers' own failure
schedules force a few requests off the single best-rated model.)

## License

MIT — see `LICENSE`.
