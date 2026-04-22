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

// Browser asks for the runtime config (Modal tracker URL, etc.) on page load.
app.get('/api/config', (req, res) => {
  res.json({
    modalTrackerUrl: process.env.MODAL_TRACKER_URL || '',
  });
});

// T1: given a JPEG frame + a user message, ask Claude Opus for a target bbox.
// Returns { bbox: [x1,y1,x2,y2] | null, reasoning: string }.
app.post('/api/t1', async (req, res) => {
  try {
    const { frame, message, width, height } = req.body || {};
    if (!frame || !message) {
      return res.status(400).json({ error: 'missing frame or message' });
    }
    const apiKey = process.env.T1_PROMPT_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'T1_PROMPT_KEY not set on server' });
    }

    const systemPrompt = [
      "You are LaiRA's vision T1 module. You receive a single video frame and a",
      "tracking command from the user. Identify the target object the user is",
      "referring to, and return a single JSON object — no surrounding text — of",
      "the form:",
      '  {"bbox": [x1, y1, x2, y2], "reasoning": "brief explanation"}',
      "Coordinates are integer pixels in the image's native resolution",
      `(image width=${width || 'unknown'}, height=${height || 'unknown'}).`,
      "The bbox should fully contain the target with a small margin (~10px).",
      "If you cannot identify the target with confidence, return:",
      '  {"bbox": null, "reasoning": "why not"}',
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
    // Extract first {...} JSON object from the response.
    const m = text.match(/\{[\s\S]*\}/);
    if (!m) {
      return res.status(502).json({ error: 'no_json_in_response', raw: text });
    }
    const parsed = JSON.parse(m[0]);
    return res.json(parsed);
  } catch (err) {
    console.error('/api/t1 error', err);
    return res.status(500).json({ error: 'server_error', detail: String(err) });
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
