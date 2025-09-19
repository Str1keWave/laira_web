const express = require('express');
const WebSocket = require('ws');
const app = express();
const revai = require('revai-node-sdk');

app.use(express.static('public'));

// Serve user and laira pages at '/user' and '/laira'
app.get('/user', (req, res) => {
  res.sendFile(__dirname + '/public/user.html');
});
app.get('/laira', (req, res) => {
  res.sendFile(__dirname + '/public/laira.html');
});
// Serve the CLI page
app.get('/cli', (req, res) => {
  res.sendFile(__dirname + '/public/cli_page.html');
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
