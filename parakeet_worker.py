"""
LaiRA Parakeet streaming STT — Modal serverless GPU.

Architecture: NVIDIA Parakeet TDT 0.6B v2 (English) running on an L4. Pi
opens a WebSocket session at wake-fire time, streams 16kHz mono int16
PCM chunks as binary frames during the command utterance, then sends a
JSON close message when its VAD detects end-of-speech. Server runs
Parakeet on the accumulated audio and returns the final transcript.

Why "stream-during-speak, batch-on-close" instead of true streaming
partials: Parakeet has RTF ~0.05 on L4 (20x realtime), so a 3-second
command transcribes in ~150ms — the streaming-partials machinery
(cache-aware buffered inference, partial token emission, etc.) buys
maybe 100ms over batch-at-close, at the cost of a much messier
implementation. For voice commands where we only act on the FINAL
transcript anyway, batch is the right tradeoff. The WebSocket protocol
leaves room to add partials later if we ever want them.

Why Parakeet over Whisper: comparable WER (~6.4% on Open ASR) and
specifically trained on 36k hours of non-speech audio for hallucination
suppression — directly addresses the "Whisper invents Korean from quiet
audio" failure mode we kept hitting on Pi's webcam mic. Whisper invents
plausible content from silence; Parakeet returns nothing.

Protocol (over /ws):
  c->s JSON  {"type":"open",  "session_id":str, "sample_rate":int}
  s->c JSON  {"type":"open_ok", "session_id":str}
  c->s BIN   raw int16 little-endian PCM at sample_rate
  c->s JSON  {"type":"close", "session_id":str}
  s->c JSON  {"type":"final", "session_id":str, "text":str,
              "duration_ms":int, "infer_ms":int, "n_chunks":int}
  s->c JSON  {"type":"error", "session_id":str, "error":str}

Deploy:    python3 -m modal deploy parakeet_worker.py
Healthz:   GET  https://<workspace>--laira-parakeet-parakeetservice-fastapi-app.modal.run/healthz
Websocket: WS   wss://<workspace>--laira-parakeet-parakeetservice-fastapi-app.modal.run/ws
"""
import io
import json
import os
import time
import wave

import modal

app = modal.App("laira-parakeet")

# Parakeet TDT 0.6B v2 — English-only, currently top of the Open ASR
# Leaderboard's English cluster. RNN-T decoder is streaming-friendly and
# was trained with non-speech-noise augmentation specifically to avoid
# the "hallucinate from silence" failure that's been killing us on
# Whisper. ~600M params; fits comfortably on L4 in fp16.
PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

# We standardize on 16kHz mono on the Pi side (laira_stt's resampler
# already produces this). Parakeet's training data is also 16kHz, so
# feeding it the native rate avoids a resample step.
TARGET_SR = 16000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "libsndfile1")
    .pip_install(
        # Pinning numpy explicitly to a 1.26.x line — pip was otherwise
        # pulling a numpy-1.20-era wheel-less version that needed
        # building from source and failed on missing build tools.
        "numpy==1.26.4",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile",
        "Cython",
        "packaging",
        "fastapi[standard]",
    )
    .pip_install(
        # NeMo ASR brings in the Parakeet TDT model loader + decoder.
        # Installed as a separate step so its dep resolution doesn't
        # disturb the numpy/torch we already locked in. 2.4.0 is the
        # current 2.x line that ships Parakeet TDT 0.6B v2 cleanly.
        "nemo_toolkit[asr]==2.4.0",
    )
    .run_commands(
        # Pre-download the model so cold-start doesn't spend ~30s pulling
        # weights on first request. Stored under HF cache, persists in the
        # image layer.
        "python -c \"import nemo.collections.asr as nemo_asr; "
        f"nemo_asr.models.ASRModel.from_pretrained('{PARAKEET_MODEL_NAME}')\"",
    )
)

with image.imports():
    import numpy as np
    import torch
    import nemo.collections.asr as nemo_asr


@app.cls(
    # L4 is the same GPU class SAMURAI uses — keeps our Modal infra
    # uniform. RTF is 0.05 here (20x realtime), so the GPU spends most
    # of its time idle even on back-to-back commands; an L4 is overkill
    # but the smallest GPU class Modal offers that has the bf16/fp16
    # support NeMo wants.
    gpu="L4",
    image=image,
    scaledown_window=300,   # 5 min keep-warm after last request
    max_containers=1,        # single shared container; Pi only sends one session at a time
    timeout=600,
)
@modal.concurrent(max_inputs=4)
class ParakeetService:
    @modal.enter()
    def setup(self):
        # Load Parakeet TDT 0.6B v2 onto the L4. fp16 is fine for
        # transcription — quality is indistinguishable from fp32 and
        # ~half the memory + faster kernels.
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            PARAKEET_MODEL_NAME
        )
        self.asr_model.to("cuda")
        self.asr_model.eval()
        # NeMo's Parakeet config supports bfloat16 inference. We use
        # autocast at call-time rather than converting the model weights
        # in case we ever need fp32 fallback.

        # Per-session state: each entry is a list of audio chunk arrays
        # (numpy int16 at TARGET_SR) accumulated since the session was
        # opened. Cleared on close.
        self.sessions: dict = {}

        # Warmup: run one transcribe on dummy audio so the first user-
        # facing call doesn't pay JIT/cudnn-autotune latency.
        try:
            dummy = (np.random.randn(TARGET_SR).astype(np.float32) * 0.05)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                _ = self.asr_model.transcribe([dummy], batch_size=1, verbose=False)
            print("[ParakeetService] model warmed")
        except Exception as e:
            print(f"[ParakeetService] warmup failed: {e!r}")
        print("[ParakeetService] Parakeet TDT 0.6B v2 ready")

    def _transcribe(self, pcm_int16_concat, sample_rate):
        """Run Parakeet on a complete audio array. Returns transcript string."""
        if pcm_int16_concat.size == 0:
            return ""
        # NeMo expects float32 PCM normalized to [-1, 1].
        audio_f32 = pcm_int16_concat.astype(np.float32) / 32768.0
        # Resample if Pi sent at a non-target rate. (Should never happen
        # in normal flow — Pi is configured for 16kHz — but cheap defense.)
        if sample_rate != TARGET_SR:
            try:
                import torchaudio
                t = torch.from_numpy(audio_f32).unsqueeze(0)
                t = torchaudio.functional.resample(t, sample_rate, TARGET_SR)
                audio_f32 = t.squeeze(0).numpy()
            except Exception as e:
                print(f"[transcribe] resample failed: {e!r}; sending raw")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # transcribe accepts a list of audio arrays. Returns a list
            # of Hypothesis objects (or strings, depending on NeMo
            # version) — we normalize to string below.
            results = self.asr_model.transcribe(
                [audio_f32],
                batch_size=1,
                verbose=False,
            )
        if not results:
            return ""
        first = results[0]
        # NeMo 2.x returns objects with a .text attribute; older returns plain strings.
        if hasattr(first, "text"):
            return first.text.strip()
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, list) and first:
            inner = first[0]
            if hasattr(inner, "text"):
                return inner.text.strip()
            if isinstance(inner, str):
                return inner.strip()
        # Unknown shape — log and return empty rather than crashing.
        print(f"[transcribe] unexpected result shape: {type(first)} {first!r:.200s}")
        return ""

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/healthz")
        def healthz():
            return {"ok": True, "model": PARAKEET_MODEL_NAME,
                    "sessions": len(self.sessions)}

        @api.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            current_sid = None
            current_sr = TARGET_SR
            try:
                while True:
                    msg = await websocket.receive()
                    # FastAPI delivers either {"text":...} or {"bytes":...}.
                    if "text" in msg and msg["text"] is not None:
                        try:
                            data = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "error": "bad json"}
                            ))
                            continue
                        typ = data.get("type")
                        sid = data.get("session_id", "default")

                        if typ == "open":
                            current_sid = sid
                            current_sr = int(data.get("sample_rate", TARGET_SR))
                            # Drop any stale session under this id (shouldn't
                            # happen since each WS is one session, but safe).
                            self.sessions[sid] = {
                                "chunks": [],
                                "total_samples": 0,
                                "opened_at": time.time(),
                                "sr": current_sr,
                            }
                            await websocket.send_text(json.dumps(
                                {"type": "open_ok", "session_id": sid}
                            ))

                        elif typ == "close":
                            sess = self.sessions.pop(sid, None)
                            if sess is None:
                                await websocket.send_text(json.dumps(
                                    {"type": "error", "session_id": sid,
                                     "error": "no session"}
                                ))
                                continue
                            chunks = sess["chunks"]
                            sr = sess["sr"]
                            total_samples = sess["total_samples"]
                            duration_ms = int(total_samples * 1000 / max(sr, 1))
                            t0 = time.time()
                            try:
                                if chunks:
                                    pcm = np.concatenate(chunks)
                                else:
                                    pcm = np.zeros(0, dtype=np.int16)
                                text = self._transcribe(pcm, sr)
                            except Exception as e:
                                print(f"[transcribe] EXC sid={sid}: {e!r}")
                                await websocket.send_text(json.dumps(
                                    {"type": "error", "session_id": sid,
                                     "error": f"transcribe_exc: {e}"}
                                ))
                                continue
                            infer_ms = int((time.time() - t0) * 1000)
                            print(f"[transcribe] sid={sid} duration={duration_ms}ms "
                                  f"chunks={len(chunks)} infer={infer_ms}ms text={text!r}")
                            await websocket.send_text(json.dumps(
                                {"type": "final", "session_id": sid,
                                 "text": text, "duration_ms": duration_ms,
                                 "infer_ms": infer_ms, "n_chunks": len(chunks)}
                            ))
                            current_sid = None

                        else:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "session_id": sid,
                                 "error": f"unknown type {typ}"}
                            ))
                    elif "bytes" in msg and msg["bytes"] is not None:
                        # Audio chunk. Must follow an "open" frame.
                        if current_sid is None or current_sid not in self.sessions:
                            # Drop silently — Pi might still be flushing
                            # late chunks after a close, no point erroring.
                            continue
                        pcm = np.frombuffer(msg["bytes"], dtype="<i2")
                        sess = self.sessions[current_sid]
                        sess["chunks"].append(pcm)
                        sess["total_samples"] += pcm.size
                    else:
                        # Empty-ish frame; ignore.
                        continue
            except WebSocketDisconnect:
                # Client dropped. Best-effort drop the session if still
                # registered so memory doesn't leak.
                if current_sid is not None:
                    self.sessions.pop(current_sid, None)

        return api
