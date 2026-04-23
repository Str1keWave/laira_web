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
// /api/vision — single-shot vision lookup. Given a frame + a target description
// + a purpose, return a bbox that fully contains the target and (for go_to) an
// estimate of what fraction of the frame the target should occupy when LaiRA
// has "arrived" next to it.
//
// Used by:
//   - stream_test.html (pre-T1: just needs bbox to seed SAMURAI)
//   - the T1 orchestrator (when executing follow / go_to tools)
//
// Was /api/t1 in earlier iterations. Renamed because /api/t1 now belongs to
// the orchestrator endpoint below.
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

    // done_area_frac is the LLM's world-knowledge estimate of "how much of
    // the frame should this target occupy when LaiRA is right next to it."
    // It feeds the arrived-detector for go_to. We always ask for it (even
    // for follow / identify) since it's cheap and lets the caller reuse
    // the same vision call across modes.
    const systemPrompt = [
      "You are LaiRA's vision module. You receive a single video frame and a target",
      "description. Identify the target and return a single JSON object — no",
      "surrounding text — of the form:",
      '  {"bbox": [x1, y1, x2, y2], "done_area_frac": 0.XX, "reasoning": "brief"}',
      `Coordinates are integer pixels in the image's native resolution (image width=${width || 'unknown'}, height=${height || 'unknown'}).`,
      "The bbox should fully contain the target with a small margin (~10px).",
      "",
      "done_area_frac is YOUR ESTIMATE of what fraction of the frame the target",
      "should occupy when LaiRA (a small quadruped robot dog with a forward-facing",
      "camera ~30cm above the ground) has walked right up to it. Calibration:",
      "  - tennis ball / mug / small toy:        ~0.05 - 0.10",
      "  - a chair, a backpack, a small box:     ~0.20 - 0.30",
      "  - a person standing:                    ~0.35 - 0.50",
      "  - a sofa or large object she'd nose:    ~0.40 - 0.60",
      "Use 0.30 as a fallback if you're unsure.",
      "",
      "If you cannot identify the target with confidence, return:",
      '  {"bbox": null, "done_area_frac": null, "reasoning": "why not"}',
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
    // Backfill default done_area_frac if missing or invalid — we promised the
    // caller this field always has a usable number for non-null bboxes.
    if (parsed.bbox && (parsed.done_area_frac == null || isNaN(parsed.done_area_frac))) {
      parsed.done_area_frac = 0.30;
    }
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
    description: 'Continuously track and follow a target object until told to stop. Holds a comfortable following distance, does NOT terminate on its own. Use for: follow X, chase X, stay on X. For "follow me" the target is the visible person.',
    input_schema: {
      type: 'object',
      properties: {
        target: { type: 'string', description: 'Short natural-language description of what to follow (e.g. "the person in the red shirt", "me", "the green ball").' },
      },
      required: ['target'],
    },
  },
  {
    name: 'go_to',
    description: 'Walk to a specific target and STOP once reached. Differs from follow: terminates when LaiRA has arrived. Use for: go to X, come to X, walk over to X.',
    input_schema: {
      type: 'object',
      properties: {
        target: { type: 'string', description: 'Short natural-language description of what to walk to.' },
      },
      required: ['target'],
    },
  },
  {
    name: 'stop',
    description: 'Cancel any active task and stand still. Use for: stop, halt, freeze, never mind, cancel that.',
    input_schema: { type: 'object', properties: {}, required: [] },
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
    description: 'Send a single low-level movement command. Used for non-tracking-based movement like "back up a bit", "turn around", "spin in place". Direction: forward/back/left/right (translation), or rotate_left/rotate_right (turn in place). duration_ms is hard-capped at 3000.',
    input_schema: {
      type: 'object',
      properties: {
        direction: { type: 'string', enum: ['forward', 'back', 'left', 'right', 'rotate_left', 'rotate_right'] },
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
  'Vision tools (follow, go_to) accept a natural-language target description. Pass the user\'s wording through — don\'t identify the target yourself, the vision system will handle it.',
  'You see the current camera frame so you can sanity-check whether a tool makes sense (e.g. don\'t call follow("the cat") if there\'s clearly no cat). But don\'t over-interpret — the vision system is more capable than you at object identification.',
  'Always emit exactly one tool call. Never zero, never multiple.',
].join('\n');

const OPUS_SYSTEM = [
  NO_TALKBACK_CLAUSE,
  '',
  'You are the orchestrator for LaiRA. You were called because the request was too complex for the first-line router. Plan the entire sequence of tool calls UPFRONT in one response. You will NOT get a chance to react to results — tools execute fire-and-forget, and any failure aborts the rest of the plan.',
  'Be conservative. If a step is risky or you\'re uncertain it will work, omit it. Better to do less correctly than more half-done.',
  'You can issue multiple tool calls in your single response. They execute in order. Each tool blocks until complete (e.g. go_to() blocks until LaiRA arrives or gives up). Plan accordingly: after a successful go_to(green ball), you can chain a sit() because you know LaiRA will be standing next to it.',
  'You have a drive() tool the first-line router does not. Use it for "turn around", "back up", "explore" type intents.',
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

async function callAnthropic(model, system, tools, content) {
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
      messages: [{ role: 'user', content }],
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

// --- Endpoint ----------------------------------------------------------

app.post('/api/t1', async (req, res) => {
  try {
    const { frame, text } = req.body || {};
    if (!frame || !text) {
      return res.status(400).json({ error: 'missing frame or text' });
    }

    const userContent = [
      { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: frame } },
      { type: 'text', text },
    ];

    // Tier 1: Sonnet
    const sonnetPayload = await callAnthropic(
      'claude-sonnet-4-6', SONNET_SYSTEM, SONNET_TOOLS, userContent
    );
    const sonnetResult = extractToolCalls(sonnetPayload, VALID_SONNET_TOOL_NAMES);

    // Did Sonnet escalate? If yes, chain to Opus with the SAME user input.
    // Per the user's spec, escalate carries no args — Opus re-plans from
    // the raw utterance + frame. Cleaner than relaying Sonnet's partial reasoning.
    const escalated = sonnetResult.tools.some(t => t.tool === 'escalate');

    if (escalated) {
      const opusPayload = await callAnthropic(
        'claude-opus-4-7', OPUS_SYSTEM, OPUS_TOOLS, userContent
      );
      const opusResult = extractToolCalls(opusPayload, VALID_OPUS_TOOL_NAMES);
      return res.json({
        source: 'opus',
        sonnet_reasoning: sonnetResult.reasoning,
        reasoning: opusResult.reasoning,
        tools: opusResult.tools,
      });
    }

    return res.json({
      source: 'sonnet',
      reasoning: sonnetResult.reasoning,
      tools: sonnetResult.tools,
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
