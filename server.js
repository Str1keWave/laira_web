const express = require('express');
const WebSocket = require('ws');
const app = express();
const revai = require('revai-node-sdk');

app.use(express.static('public'));
// Frames are sent as base64 JPEG in /api/t1; bump the JSON limit accordingly.
app.use(express.json({ limit: '10mb' }));

// === /stream_test demo page (LaiRA visual pipeline test) ===
app.get('/stream_test', (req, res) => {
  res.sendFile(__dirname + '/public/stream_test.html');
});

// === /t1_test demo page (LaiRA T1 orchestrator MVP, browser-cam, no Pi) ===
app.get('/t1_test', (req, res) => {
  res.sendFile(__dirname + '/public/t1_test.html');
});

// Browser asks for the runtime config (Modal tracker URL, etc.) on page load.
app.get('/api/config', (req, res) => {
  res.json({
    modalTrackerUrl: process.env.MODAL_TRACKER_URL || '',
  });
});

// =============================================================
// /api/vision — single-shot vision lookup. Given a frame + a target description,
// return a bbox that fully contains the target.
//
// Distance-to-arrival is no longer the LLM's job — the SAMURAI worker on Modal
// runs Depth Anything V2 alongside SAM 2 and emits a metric depth_cm per frame.
// The orchestrator decides the target distance per intent (target_distance_cm
// argument on follow / go_to). All this endpoint owes the caller is the bbox.
//
// Used by:
//   - stream_test.html (just needs bbox to seed SAMURAI)
//   - the T1 orchestrator (when executing follow / go_to tools)
// =============================================================
app.post('/api/vision', async (req, res) => {
  try {
    const { frame, message, width, height, purpose } = req.body || {};
    if (!frame || !message) {
      return res.status(400).json({ error: 'missing frame or message' });
    }
    const apiKey = process.env.T1_PROMPT_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'T1_PROMPT_KEY not set on server' });
    }

    const systemPrompt = [
      "You are LaiRA's vision module. You receive a single video frame and a target",
      "description. Identify the target and return a single JSON object — no",
      "surrounding text — of the form:",
      '  {"bbox": [x1, y1, x2, y2], "reasoning": "brief"}',
      `Coordinates are integer pixels in the image's native resolution (image width=${width || 'unknown'}, height=${height || 'unknown'}).`,
      "The bbox should fully contain the target with a small margin (~10px).",
      "",
      "If you cannot identify the target with confidence, return:",
      '  {"bbox": null, "reasoning": "why not"}',
      "",
      `Caller's purpose: ${purpose || 'unspecified'}.`,
    ].join('\n');

    const anthropicResp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'claude-opus-4-7',
        max_tokens: 512,
        system: systemPrompt,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: frame } },
            { type: 'text', text: message },
          ],
        }],
      }),
    });

    if (!anthropicResp.ok) {
      const errText = await anthropicResp.text();
      console.error('Anthropic error', anthropicResp.status, errText);
      return res.status(502).json({ error: 'anthropic_error', status: anthropicResp.status, detail: errText });
    }

    const payload = await anthropicResp.json();
    const text = (payload.content || []).map(b => b.text || '').join('').trim();
    const m = text.match(/\{[\s\S]*\}/);
    if (!m) {
      return res.status(502).json({ error: 'no_json_in_response', raw: text });
    }
    const parsed = JSON.parse(m[0]);
    return res.json(parsed);
  } catch (err) {
    console.error('/api/vision error', err);
    return res.status(500).json({ error: 'server_error', detail: String(err) });
  }
});

// =============================================================
// /api/t1 — the orchestrator. Given a frame + user utterance, returns an
// ordered list of tool calls to execute on the client.
//
// Two-tier model:
//   1. Sonnet 4.6 sees the user message + frame, picks ONE tool from a small
//      safe set. If the request is multi-step, fine-grained, or uncertain,
//      Sonnet calls escalate() (no args).
//   2. On escalate, the SERVER chains a second LLM call to Opus 4.7 with the
//      original utterance + extended tool set (drive() added, escalate
//      removed). Opus plans the FULL sequence in one response.
//
// Client never sees the escalation hop — gets back one flat tool list.
//
// Hard invariant baked into both system prompts: NO TALKBACK. LaiRA is a dog;
// dogs don't speak. Reasoning text is debug-only, never surfaced to the user.
// If LaiRA can't act, she does nothing. She never asks questions.
// =============================================================

// --- Tool definitions ---------------------------------------------------

const SHARED_TOOLS = [
  {
    name: 'sit',
    description: 'Make LaiRA sit on her haunches.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'stand',
    description: 'Make LaiRA stand up from any pose. Also use this when the user wants her to stop sitting / lying / shaking.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'lie_down',
    description: 'Make LaiRA lie down on her belly. Use for: lie down, sleep, rest, take a nap.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'shake',
    description: 'Make LaiRA offer her paw for a handshake.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'follow',
    description: 'Continuously track and follow a target object until told to stop. Holds the requested following distance, does NOT terminate on its own. Use for: follow X, chase X, stay on X. For "follow me" the target is the visible person.',
    input_schema: {
      type: 'object',
      properties: {
        target: { type: 'string', description: 'Short natural-language description of what to follow (e.g. "the person in the red shirt", "me", "the green ball").' },
        target_distance_cm: {
          type: 'integer', minimum: 5, maximum: 300,
          description: 'How far back LaiRA should hold while following, in centimetres. Reference points: 30-50 close-follow on a small toy or pet, 60-100 a polite person-following distance, 150+ "stay in the area but give space".',
        },
      },
      required: ['target', 'target_distance_cm'],
    },
  },
  {
    name: 'go_to',
    description: 'Walk to a specific target and STOP once reached. Differs from follow: terminates when LaiRA has arrived. Use for: go to X, come to X, walk over to X.',
    input_schema: {
      type: 'object',
      properties: {
        target: { type: 'string', description: 'Short natural-language description of what to walk to.' },
        target_distance_cm: {
          type: 'integer', minimum: 5, maximum: 300,
          description: 'How close LaiRA should be to the target when "arrived", in centimetres. Reference points: 10-20 nose-to-target (a ball she\'d sniff), 30-60 body-distance (a person, another dog, a piece of furniture she greets), 80-150 room-scale ("just be in the area").',
        },
      },
      required: ['target', 'target_distance_cm'],
    },
  },
  {
    name: 'stop',
    description: 'Cancel any active task and stand still. Use for: stop, halt, freeze, never mind, cancel that.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'wait',
    description: 'Pause for duration_ms milliseconds before the next action. Use for pacing within a compound plan ("sit, then stand a couple seconds later" -> sit, wait(2000), stand) or when the user explicitly asks to wait. Hard cap is 30000ms; for longer idle the user should re-trigger LaiRA. The wait is interrupted immediately by stop, by manual user input, or by the user issuing a new command.',
    input_schema: {
      type: 'object',
      properties: {
        duration_ms: {
          type: 'integer', minimum: 100, maximum: 30000,
          description: 'Milliseconds to pause. Reference: 1500-3000 a beat between actions, 5000-10000 a noticeable wait, 10000+ a long deliberate pause.',
        },
      },
      required: ['duration_ms'],
    },
  },
];

const SONNET_TOOLS = [
  ...SHARED_TOOLS,
  {
    name: 'escalate',
    description: 'Forward the user request to a more capable orchestrator. ALWAYS use this for: (a) compound requests with multiple steps ("do X then Y"), (b) any movement that is not centering on a tracked object — turning around, backing up, spinning, exploring, (c) anything you are not confident you can map cleanly to ONE of the other tools. When in doubt, escalate. The orchestrator will see the original user message; you do not pass any arguments yourself.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
];

const OPUS_TOOLS = [
  ...SHARED_TOOLS,
  {
    name: 'drive',
    description: 'Send a single low-level movement command. Used for non-tracking-based movement like "back up a bit", "turn around", "spin in place". Direction: forward / back (translation), or rotate_left / rotate_right (turn in place). LaiRA is a quadruped — she does not strafe; sideways movement is achieved by rotating then driving forward. duration_ms is hard-capped at 3000ms (do not trust longer blind motion — that\'s what the tracker is for).',
    input_schema: {
      type: 'object',
      properties: {
        direction: { type: 'string', enum: ['forward', 'back', 'rotate_left', 'rotate_right'] },
        duration_ms: { type: 'integer', minimum: 100, maximum: 3000 },
      },
      required: ['direction', 'duration_ms'],
    },
  },
];

// --- System prompts ----------------------------------------------------

// Hard, load-bearing constraint shared by both tiers. The dog metaphor is
// the operative framing: LaiRA can't speak, can't ask, can't apologize,
// can't explain to the user. She just acts, or doesn't.
const NO_TALKBACK_CLAUSE = [
  'CRITICAL CONSTRAINT — READ TWICE:',
  'You are LaiRA, a quadruped robot dog. You communicate ONLY through tool calls. You do NOT speak to the user.',
  'Dogs do not talk. You are a dog.',
  'If you cannot perform the requested action, do nothing — do not apologize, do not explain to the user, do not ask for clarification.',
  'Never call a tool that "tells the user" something. There is no such tool. There never will be.',
  'You may include brief reasoning text BEFORE your tool calls. This reasoning is logged for human debugging only and is never shown to the user. Keep it under 30 words. Skip it if the action is obvious.',
].join(' ');

const SONNET_SYSTEM = [
  NO_TALKBACK_CLAUSE,
  '',
  'You are the first-line command router for LaiRA. Your job: decide which single tool to call.',
  'You have a small set of safe actions. For anything compound, fine-grained-movement-related, or uncertain, call escalate(). It is ALWAYS better to escalate than to guess.',
  'STOP/CANCEL is NOT one of the cases that needs escalation. If the user says "stop", "halt", "freeze", "never mind", "cancel that", or anything in that family, call stop() directly. Never escalate a stop intent — escalation adds latency and stop should be instant.',
  'Vision tools (follow, go_to) accept a natural-language target description AND a target_distance_cm — how close LaiRA should be to the target. Pass the user\'s wording through for the target. For target_distance_cm, infer from the user\'s intent what makes sense (sniff/touch a ball ≈ 15cm; greet a person ≈ 50cm; "just be in the kitchen" ≈ 150cm). When in genuine doubt about distance, escalate.',
  'You see the current camera frame so you can sanity-check whether a tool makes sense (e.g. don\'t call follow("the cat") if there\'s clearly no cat). But don\'t over-interpret — the vision system is more capable than you at object identification.',
  'Always emit exactly one tool call. Never zero, never multiple.',
].join('\n');

const OPUS_SYSTEM = [
  NO_TALKBACK_CLAUSE,
  '',
  'You are the orchestrator for LaiRA. You were called because the request was too complex for the first-line router. Plan the entire sequence of tool calls UPFRONT in one response. You will NOT get a chance to react to results — tools execute fire-and-forget, and any failure aborts the rest of the plan.',
  'Be conservative. If a step is risky or you\'re uncertain it will work, omit it. Better to do less correctly than more half-done.',
  'You can issue multiple tool calls in your single response. They execute in order. Each tool blocks until complete (e.g. go_to() blocks until LaiRA arrives or gives up). Plan accordingly: after a successful go_to(green ball, 15), you can chain a sit() because you know LaiRA will be standing right next to it. For pacing ("sit, then stand a moment later"), use wait(duration_ms) between the two actions.',
  'For follow / go_to you must pick target_distance_cm (5-300). Reference: 10-20 nose-to-target, 30-60 body-distance to a person/dog, 80-150 room-scale. Choose what fits the user\'s actual intent.',
  'You have a drive() tool the first-line router does not. Use it for "turn around", "back up", "explore" type intents. Note: LaiRA cannot strafe — there is no left/right translation, only forward/back and rotate_left/rotate_right. To "move sideways" you rotate then drive forward.',
  '',
  'Camera frames in your message history are labeled "[frame X of N — ...]". The CURRENT frame (highest index, marked CURRENT) is the just-captured one. Earlier frames in the same session are kept so you remember spatial context from when the task started — for example, if the user pointed somewhere in the original frame, that gesture won\'t be in the current frame but you can refer back to it. Older frames beyond the rolling window are omitted to save tokens.',
  '',
  'After go_to(target) arrives, if the user wanted LaiRA to stay there ("go wait by X", "go to Y and stay", "park there"), call stop() — LaiRA will stand still until told otherwise. Do NOT use wait() for indefinite stays: it\'s capped at 30s and is intended for short pacing between actions, not for parking.',
].join('\n');

// --- Helpers -----------------------------------------------------------

const VALID_SONNET_TOOL_NAMES = new Set(SONNET_TOOLS.map(t => t.name));
const VALID_OPUS_TOOL_NAMES = new Set(OPUS_TOOLS.map(t => t.name));

function extractToolCalls(anthropicPayload, validNames) {
  // Pull out any tool_use blocks in order, dropping unknown names defensively.
  const blocks = anthropicPayload.content || [];
  const tools = [];
  let reasoning = '';
  for (const b of blocks) {
    if (b.type === 'text' && b.text) reasoning += b.text;
    if (b.type === 'tool_use' && validNames.has(b.name)) {
      tools.push({ tool: b.name, args: b.input || {} });
    }
  }
  return { tools, reasoning: reasoning.trim() };
}

async function callAnthropic(model, system, tools, messages) {
  const apiKey = process.env.T1_PROMPT_KEY;
  if (!apiKey) throw new Error('T1_PROMPT_KEY not set on server');
  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      system,
      tools,
      messages,
    }),
  });
  if (!resp.ok) {
    const errText = await resp.text();
    const err = new Error(`anthropic ${resp.status}: ${errText}`);
    err.status = resp.status;
    err.detail = errText;
    throw err;
  }
  return await resp.json();
}

// --- T1 conversation session store ---
//
// Sessions hold the message[] history for /api/t1 calls that share a
// session_id. The orchestrator uses this for *within-plan* continuity —
// e.g. when a go_to gets stuck and the client re-engages Sonnet with a
// status update, Sonnet sees her own original plan and can decide whether
// to retry, settle, or give up.
//
// Lifecycle:
//   - Client generates a session_id when the user issues a fresh command.
//   - Each /api/t1 call appends (user_turn, assistant_summary) to the session.
//   - When the orchestrator emits stop(), the session is deleted server-side
//     so the next call with the same id starts fresh — matches user spec.
//   - Idle sessions evict after T1_SESSION_TTL_MS to bound memory.
//
// Storage tricks:
//   - We strip image content blocks from prior user turns (replace with text
//     marker) before storing — Sonnet doesn't need yesterday's frame to
//     remember yesterday's plan, and old frames bloat both the store and
//     subsequent API payloads.
//   - We store assistant turns as plain text summaries of the tool calls,
//     not the raw tool_use blocks. This sidesteps the API's requirement
//     that tool_use must be followed by tool_result — we never executed
//     tools server-side, so there's no result to fake. Plain text reads
//     fine to the model and is correctly formatted.
const T1_SESSIONS = new Map();
const T1_SESSION_TTL_MS = 10 * 60 * 1000;
// Keep at most this many most-recent user turns' frames intact in the message
// history. Older user turns get their image blocks stripped to a text marker.
// 10 is well above any realistic compound plan turn count — it's a safety cap,
// not a normal operating point.
const MAX_FRAMES_IN_HISTORY = 10;

function pruneT1Sessions() {
  const now = Date.now();
  for (const [sid, s] of T1_SESSIONS) {
    if (now - s.lastTouch > T1_SESSION_TTL_MS) T1_SESSIONS.delete(sid);
  }
}

function stripFramesFromUserContent(content) {
  return content.map(b => b.type === 'image'
    ? { type: 'text', text: '[prior camera frame omitted]' }
    : b);
}

// Walk the full user-turn list (priorMessages + the current user turn) and
// build a fresh messages[] for the API call. We keep image content in the
// most recent MAX_FRAMES_IN_HISTORY user turns, and label each kept frame
// with its position so Opus knows which one is "now" and which is "the
// original frame from when the task started". Older user turns get their
// images stripped to a text marker.
//
// Note: this only mutates content in-flight for the API call; what we
// store in T1_SESSIONS keeps frames intact so subsequent turns can re-decide
// what to keep based on the new MAX window.
function buildCallMessages(priorMessages, currentUserContent) {
  const allMessages = [...priorMessages, { role: 'user', content: currentUserContent }];
  // Find indices of user turns that contain at least one image block.
  const imageTurnIndices = [];
  for (let i = 0; i < allMessages.length; i++) {
    const m = allMessages[i];
    if (m.role !== 'user' || !Array.isArray(m.content)) continue;
    if (m.content.some(b => b && b.type === 'image')) {
      imageTurnIndices.push(i);
    }
  }
  const keepCount = Math.min(imageTurnIndices.length, MAX_FRAMES_IN_HISTORY);
  const keepStart = imageTurnIndices.length - keepCount; // index into imageTurnIndices
  const keptIndices = new Set(imageTurnIndices.slice(keepStart));

  return allMessages.map((m, i) => {
    if (m.role !== 'user' || !Array.isArray(m.content)) return m;
    const hasImage = m.content.some(b => b && b.type === 'image');
    if (!hasImage) return m;
    if (!keptIndices.has(i)) {
      return { role: 'user', content: stripFramesFromUserContent(m.content) };
    }
    // This is one of the kept turns — figure out its position label.
    const positionInWindow = imageTurnIndices.slice(keepStart).indexOf(i) + 1; // 1-based
    const isCurrent = positionInWindow === keepCount;
    const isOldest = positionInWindow === 1 && keepCount > 1;
    const label = isCurrent
      ? `[frame ${positionInWindow} of ${keepCount} — CURRENT (just captured)]`
      : isOldest
        ? `[frame ${positionInWindow} of ${keepCount} — oldest in window, from when this part of the conversation started]`
        : `[frame ${positionInWindow} of ${keepCount}]`;
    // Insert the label as a text block right before each image block.
    const labeled = [];
    for (const b of m.content) {
      if (b && b.type === 'image') {
        labeled.push({ type: 'text', text: label });
      }
      labeled.push(b);
    }
    return { role: 'user', content: labeled };
  });
}

function summarizeAssistantTurn(toolList, reasoning) {
  const calls = (toolList || [])
    .map(t => `${t.tool}(${JSON.stringify(t.args || {})})`)
    .join('; ');
  const text = (reasoning ? reasoning.trim() + '\n' : '') +
    (calls ? `Plan: ${calls}` : 'Plan: (no tools)');
  return { role: 'assistant', content: text };
}

// --- Endpoint ----------------------------------------------------------

app.post('/api/t1', async (req, res) => {
  try {
    const { frame, text, session_id } = req.body || {};
    if (!frame || !text) {
      return res.status(400).json({ error: 'missing frame or text' });
    }

    const userContent = [
      { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: frame } },
      { type: 'text', text },
    ];

    // Pull prior conversation history if the client is continuing a session.
    const sessRecord = session_id ? T1_SESSIONS.get(session_id) : null;
    const priorMessages = sessRecord ? sessRecord.messages : [];
    // Sticky escalation: once Opus has been engaged for a session, every
    // subsequent turn in that session goes straight to Opus. Sonnet was
    // routed past on this conversation already, no point re-routing.
    const escalatedSticky = !!(sessRecord && sessRecord.escalated);
    const messagesForCall = buildCallMessages(priorMessages, userContent);

    let finalSource, finalReasoning, finalTools;
    let extraSonnetReasoning;
    let escalated = false;

    if (escalatedSticky) {
      // Skip the router entirely — go straight to Opus.
      const opusPayload = await callAnthropic(
        'claude-opus-4-7', OPUS_SYSTEM, OPUS_TOOLS, messagesForCall
      );
      const opusResult = extractToolCalls(opusPayload, VALID_OPUS_TOOL_NAMES);
      finalSource = 'opus';
      finalReasoning = opusResult.reasoning;
      finalTools = opusResult.tools;
    } else {
      // Tier 1: Sonnet
      const sonnetPayload = await callAnthropic(
        'claude-sonnet-4-6', SONNET_SYSTEM, SONNET_TOOLS, messagesForCall
      );
      const sonnetResult = extractToolCalls(sonnetPayload, VALID_SONNET_TOOL_NAMES);

      // Did Sonnet escalate? If yes, chain to Opus with the SAME user input.
      // Per the user's spec, escalate carries no args — Opus re-plans from
      // the raw utterance + frame (and the same prior history). Cleaner than
      // relaying Sonnet's partial reasoning.
      escalated = sonnetResult.tools.some(t => t.tool === 'escalate');

      if (escalated) {
        const opusPayload = await callAnthropic(
          'claude-opus-4-7', OPUS_SYSTEM, OPUS_TOOLS, messagesForCall
        );
        const opusResult = extractToolCalls(opusPayload, VALID_OPUS_TOOL_NAMES);
        finalSource = 'opus';
        finalReasoning = opusResult.reasoning;
        finalTools = opusResult.tools;
        extraSonnetReasoning = sonnetResult.reasoning;
      } else {
        finalSource = 'sonnet';
        finalReasoning = sonnetResult.reasoning;
        finalTools = sonnetResult.tools;
      }
    }

    // Persist this turn to the session store so a follow-up call (e.g.
    // re-engage on stuck) sees what was planned. Frames are stored intact
    // here — the per-call builder decides which ones survive into the next
    // API payload based on the rolling window.
    if (session_id) {
      const storedUserTurn = { role: 'user', content: userContent };
      const storedAssistantTurn = summarizeAssistantTurn(finalTools, finalReasoning);
      const messages = [...priorMessages, storedUserTurn, storedAssistantTurn];
      const escalatedNow = escalatedSticky || finalSource === 'opus';
      T1_SESSIONS.set(session_id, {
        messages,
        lastTouch: Date.now(),
        escalated: escalatedNow,
      });

      // Spec: stop() ends the T1 session. Drop server-side state so the
      // next call with this id starts fresh.
      if (finalTools.some(t => t.tool === 'stop')) {
        T1_SESSIONS.delete(session_id);
      }
      pruneT1Sessions();
    }

    return res.json({
      source: finalSource,
      reasoning: finalReasoning,
      tools: finalTools,
      ...(extraSonnetReasoning ? { sonnet_reasoning: extraSonnetReasoning } : {}),
    });
  } catch (err) {
    console.error('/api/t1 error', err);
    return res.status(err.status || 500).json({
      error: 'server_error',
      detail: err.detail || String(err),
    });
  }
});

// Serve user and laira pages at '/user' and '/laira'
app.get('/user', (req, res) => {
  res.sendFile(__dirname + '/public/user.html');
});
app.get('/user2', (req, res) => {
  res.sendFile(__dirname + '/public/user2.html');
});
app.get('/laira', (req, res) => {
  res.sendFile(__dirname + '/public/laira.html');
});
// Serve the CLI page
app.get('/cli', (req, res) => {
  res.sendFile(__dirname + '/public/cli_page.html');
});

app.get("/turn-credentials", async (req, res) => {
  const resp = await fetch(
    `https://laira.metered.live/api/v1/turn/credentials?apiKey=${process.env.METERED_API_KEY}`
  );
  res.json(await resp.json());
});

// Create HTTP server and WebSocket server
const server = app.listen(process.env.PORT || 3000, () => {
  console.log(`Server running on port ${server.address().port}`);
});
const wss = new WebSocket.Server({ server });

// Define connections
let userSocket = null;
let laiRASocket = null;
let forwardSocketSet = new Set(); // To handle multiple CLI clients

wss.on('connection', (ws, req) => {
  if (req.url === '/user') {
    userSocket = ws;
    console.log('User connected');

    if (laiRASocket && laiRASocket.readyState === WebSocket.OPEN) {
      userSocket.send(JSON.stringify({ type: 'laiRA-connected' }));
    }

    userSocket.on('message', (message, isBinary) => {
      if (isBinary) {
        console.log('Received binary message from User');
      } else {
        console.log('Received from User:', message);
      }

      // Forward message to LaiRA if connected
      if (laiRASocket && laiRASocket.readyState === WebSocket.OPEN) {
        laiRASocket.send(message, { binary: isBinary });
      }

      // Forward only text chat messages to CLI clients
      if (!isBinary) {
        forwardChatMessagesToCLIClients(message);
      }
    });

    userSocket.on('close', () => {
      console.log('User disconnected');
      userSocket = null;
    });
  } else if (req.url === '/laiRA') {
    laiRASocket = ws;
    console.log('LaiRA connected');

    if (userSocket && userSocket.readyState === WebSocket.OPEN) {
      userSocket.send(JSON.stringify({ type: 'laiRA-connected' }));
    }

    laiRASocket.on('message', (message, isBinary) => {
      if (isBinary) {
        console.log('Received binary message from LaiRA');
      } else {
        console.log('Received from LaiRA:', message);
      }

      // Forward message to User if connected
      if (userSocket && userSocket.readyState === WebSocket.OPEN) {
        userSocket.send(message, { binary: isBinary });
      }

      // Forward only text chat messages to CLI clients
      if (!isBinary) {
        forwardChatMessagesToCLIClients(message);
      }
    });

    laiRASocket.on('close', () => {
      console.log('LaiRA disconnected');
      laiRASocket = null;
    });
  } else if (req.url === '/forward') {
    // Handle connections to WebSocket B
    console.log('Forward client connected');
    forwardSocketSet.add(ws);

    ws.on('message', (message, isBinary) => {
      if (isBinary) {
        console.log('Received binary message from Forward client');
      } else {
        console.log('Received from Forward client:', message);
      }
      // If needed, handle messages from CLI clients
    });

    ws.on('close', () => {
      console.log('Forward client disconnected');
      forwardSocketSet.delete(ws);
    });
  } else if (req.url === '/audio-stream') {
  console.log('Audio stream connected');

  // Configure Rev.ai audio
  const audioConfig = new revai.AudioConfig(
    "audio/x-raw",   // MIME
    "interleaved",   // layout
    48000,           // sample rate Hz 
    "S16LE",         // format
    1                // channels
  );

  const token = process.env.REVAI_ACCESS_TOKEN; // set this in env
  const client = new revai.RevAiStreamingClient(token, audioConfig);
  const revStream = client.start();

  // Forward Rev.ai transcripts back to user.html
  revStream.on('data', (data) => {
    try {
      const parsed = JSON.parse(data.toString());
      if (parsed.type === "final" && userSocket && userSocket.readyState === WebSocket.OPEN) {
        userSocket.send(JSON.stringify({ type: "voice-transcript", data: parsed }));
      }
    } catch (err) {
      console.error("Error parsing Rev.ai data", err, data.toString());
    }
  });

  revStream.on('end', () => {
    console.log("Rev.ai stream ended");
  });

  ws.on('message', (message, isBinary) => {
    if (isBinary) {
      // PCM audio chunk
      revStream.write(message);
    } else {
      console.log("Non-binary message on /audio-stream", message.toString());
    }
  });

  ws.on('close', () => {
    console.log("Audio stream disconnected");
    client.end();
  });
}


  
});

// Function to forward only chat messages to CLI clients
function forwardChatMessagesToCLIClients(data) {
  try {
    const parsedData = JSON.parse(data);
    if (parsedData.type === 'chat-message') {
      forwardSocketSet.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(data);
        }
      });
    }
  } catch (e) {
    console.error('Error parsing data for CLI clients:', e);
  }
}
