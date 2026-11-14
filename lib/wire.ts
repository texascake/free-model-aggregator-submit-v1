// The client wires and the three upstream wires, written for this entry.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. A provider may speak any of the three wires (openai, anthropic, gemini),
// so every client request is rewritten into the routed provider's wire, and every
// provider answer is read whole — JSON or SSE — into one normalized answer
// `{ text, input, output, stop }` and rendered back in the wire the client used.
//
// The providers in this trial always answer plain text (their answer is an echo of
// what they parsed), so no provider answer carries tool calls; the stop family is
// what the renderers map. What the client compares is the echo: the system prompt,
// the user text, the tool names, max_tokens, the stop sequences, the image and the
// JSON mode must all survive the translation, so each builder below carries every
// field the client's wire holds into the provider's wire in that wire's own shape.
//
// Everything here is a pure function of its arguments: no clock and no randomness,
// so the same body and the same provider answer give the same bytes on a replay.
// The model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One provider answer, read whole: its text, its usage, its stop reason as the provider named it. */
export interface NormalizedAnswer {
  text: string;
  input: number;
  output: number;
  stop: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The text slice one streamed delta carries, as the providers slice theirs. */
const STREAM_PIECE = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A number where one is expected, else 0. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A stop list from a string or a string array, else []. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((v): v is string => typeof v === 'string');
  return [];
}

/** One Anthropic image block as an OpenAI image_url part; null for anything else. */
function openAIImagePart(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/** One OpenAI image_url part as an Anthropic image block; null for anything else. */
function anthropicImageBlock(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/** One OpenAI image_url part as a Gemini part; null for anything else. */
function geminiImagePart(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
  }
  return { fileData: { fileUri: url } };
}

/** One Anthropic image block as a Gemini part; null for anything else. */
function geminiImagePartFromAnthropic(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** The stop family of a stop reason: 0 stop, 1 length, 2 tool use, 3 refusal. */
export function stopFamilyOf(stop: string): number {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 1;
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 2;
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 3;
  return 0;
}

/** A stop reason in one wire's spelling for a stop family. */
export function clientStop(family: number, wire: 'openai' | 'anthropic'): string {
  if (wire === 'openai') return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

// ---------------------------------------------------------------------------
// Client OpenAI -> upstream Anthropic
// ---------------------------------------------------------------------------

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: the system
 * messages become the system prompt, image_url parts become image blocks, tool
 * messages become tool_result blocks, assistant tool_calls become tool_use blocks,
 * and max_tokens, temperature, top_p, stop sequences, tools and tool_choice pass
 * through. JSON mode has no Anthropic spelling: callers must not route a JSON-mode
 * OpenAI request to an Anthropic-wire provider.
 */
export function openAIToAnthropicBody(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length === 0) return;
    messages.push({ role: 'user', content: pendingToolResults });
    pendingToolResults = [];
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: toolResultText(raw.content),
      });
      continue;
    }
    flushToolResults();
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    // A user message: text and image parts, in order.
    const content: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') content.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') content.push({ type: 'text', text: part.text });
        else {
          const image = anthropicImageBlock(part);
          if (image !== null) content.push(image);
        }
      }
    }
    if (content.length > 0) messages.push({ role: 'user', content });
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Client Anthropic -> upstream OpenAI
// ---------------------------------------------------------------------------

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks become
 * tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAIBody(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      else {
        const image = openAIImagePart(block);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Client OpenAI / Anthropic -> upstream Gemini
// ---------------------------------------------------------------------------

/** OpenAI function tools as Gemini functionDeclarations. */
function geminiDeclarationsFromOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiDeclarationsFromAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/** A Gemini generationConfig from the client's cap, stops, JSON mode and sampling fields. */
function geminiConfig(maxTokens: unknown, stops: string[], jsonMode: boolean, temperature: unknown, topP: unknown): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system
 * messages become the systemInstruction, user parts become user contents (text and
 * inline images), assistant tool_calls become functionCall parts, tool messages
 * become functionResponse parts on the nearest user content, and the output cap,
 * stop sequences, JSON mode and sampling land in generationConfig.
 */
export function openAIToGeminiBody(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const lastUserContent = (): Json | null => {
    for (let i = contents.length - 1; i >= 0; i--) {
      if (contents[i]?.role === 'user') return contents[i] as Json;
    }
    return null;
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      const part = { functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { result: toolResultText(raw.content) } } };
      const target = lastUserContent();
      if (target !== null && contents[contents.length - 1] === target) (target.parts as Json[]).push(part);
      else contents.push({ role: 'user', parts: [part] });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else {
          const image = geminiImagePart(part);
          if (image !== null) parts.push(image);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const declarations = geminiDeclarationsFromOpenAI(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  out.generationConfig = geminiConfig(maxTokens, stopList(body.stop), format === 'json_object' || format === 'json_schema', body.temperature, body.top_p);
  void model;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt
 * becomes the systemInstruction, user blocks become user contents, tool_use blocks
 * become functionCall parts, tool_result blocks become functionResponse parts, and
 * the output cap, stop sequences and sampling land in generationConfig. Anthropic
 * has no JSON mode to carry, so none is set.
 */
export function anthropicToGeminiBody(body: Json, model: string): Json {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (assistant && block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      if (!assistant && block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            response: { result: toolResultText(block.content) },
          },
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (!assistant) {
        const image = geminiImagePartFromAnthropic(block);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = geminiDeclarationsFromAnthropic(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((v): v is string => typeof v === 'string') : [];
  out.generationConfig = geminiConfig(body.max_tokens, stops, false, body.temperature, body.top_p);
  void model;
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers -> one normalized answer
// ---------------------------------------------------------------------------

/** One SSE frame's lines as `{ event, data }` pairs; data-only frames carry event null. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The text of a Chat Completions message: a string, or its text parts joined. */
function openAIText(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  return textOf(message.content);
}

/** A Chat Completions answer (JSON or SSE) as one normalized answer; null when it cannot be read. */
export function parseOpenAIAnswer(text: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(body)) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: openAIText(message),
      input: numberOrZero(usage.prompt_tokens),
      output: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    };
  }
  let answer = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') answer += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return sawAny ? { text: answer, input, output, stop } : null;
}

/** The text of an Anthropic content list: its text blocks joined with no separator. */
function anthropicText(content: unknown): string {
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A Messages answer (JSON or SSE) as one normalized answer; null when it cannot be read. */
export function parseAnthropicAnswer(text: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(body)) return null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: anthropicText(body.content),
      input: numberOrZero(usage.input_tokens),
      output: numberOrZero(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    };
  }
  let answer = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      answer += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return sawAny ? { text: answer, input, output, stop } : null;
}

/** The text of a Gemini candidate: its parts' text joined with no separator. */
function geminiText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini generateContent answer (JSON or SSE) as one normalized answer; null when it cannot be read. */
export function parseGeminiAnswer(text: string, stream: boolean): NormalizedAnswer | null {
  if (!stream) {
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(body)) return null;
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) return null;
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text: geminiText(candidate),
      input: numberOrZero(usage.promptTokenCount),
      output: numberOrZero(usage.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    };
  }
  let answer = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  let sawAny = false;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    sawAny = true;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      answer += geminiText(candidate);
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return sawAny ? { text: answer, input, output, stop } : null;
}

/** One provider answer, in its own wire, as one normalized answer; null when it cannot be read. */
export function parseProviderAnswer(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): NormalizedAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text, stream);
  if (wire === 'anthropic') return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

// ---------------------------------------------------------------------------
// One normalized answer -> the client's wire
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The text split into pieces of at most STREAM_PIECE characters. */
function pieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) out.push(text.slice(at, at + STREAM_PIECE));
  if (out.length === 0) out.push('');
  return out;
}

/** A normalized answer as a Chat Completions object. */
export function openAIJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: clientStop(stopFamilyOf(answer.stop), 'openai'), logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A normalized answer as a Chat Completions event stream. */
export function openAIStream(answer: NormalizedAnswer, model: string): string {
  const stop = clientStop(stopFamilyOf(answer.stop), 'openai');
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of pieces(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, stop)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as a Messages object. */
export function anthropicJson(answer: NormalizedAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: clientStop(stopFamilyOf(answer.stop), 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** A normalized answer as a Messages event stream. */
export function anthropicStream(answer: NormalizedAnswer, model: string): string {
  const stop = clientStop(stopFamilyOf(answer.stop), 'anthropic');
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of pieces(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
