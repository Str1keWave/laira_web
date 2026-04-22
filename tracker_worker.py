"""
LaiRA tracker service — Modal serverless GPU.

MVP design: SAM 2 image predictor with a rolling box prompt.
Each track call uses the previous bbox (expanded by 30%) as a box prompt
to SAM 2's image predictor and returns the tightened mask's bbox.

This is NOT true SAMURAI memory-bank tracking — it's per-frame image
segmentation conditioned on the previous bbox. Fast (~150ms on T4),
stateless beyond "last bbox," and good enough to validate the pipeline.

Upgrade path: replace _segment with streaming SAM 2 video predictor
that maintains a per-session memory bank across frames.

Protocol (JSON over websocket /ws):
  c->s: {"type":"init",  "session_id":str, "frame":b64jpg, "bbox":[x1,y1,x2,y2]}
  s->c: {"type":"init_ok",   "session_id":str, "bbox":[x,y,w,h]}
  s->c: {"type":"init_fail", "session_id":str, "error":str}
  c->s: {"type":"track", "session_id":str, "frame":b64jpg}
  s->c: {"type":"bbox",  "session_id":str, "bbox":[x,y,w,h]|null, "ms":int}
  c->s: {"type":"close", "session_id":str}

Deploy:    modal deploy tracker_worker.py
Healthz:   GET  https://<workspace>--laira-tracker-trackerservice-fastapi-app.modal.run/healthz
Websocket: WS   wss://<workspace>--laira-tracker-trackerservice-fastapi-app.modal.run/ws
"""
import base64
import json
import time

import modal

app = modal.App("laira-tracker")

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
)
CHECKPOINT_PATH = "/root/checkpoints/sam2.1_hiera_base_plus.pt"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy<2",
        "opencv-python-headless",
        "pillow",
        "fastapi[standard]",
        "hydra-core",
        "iopath",
        "tqdm",
    )
    .run_commands(
        # Clone to a non-`sam2` directory name so SAM 2's own
        # "are-you-running-from-the-repo-dir" check doesn't trip at import time.
        "git clone https://github.com/facebookresearch/sam2.git /opt/sam2-src",
        "cd /opt/sam2-src && pip install -e .",
        f"mkdir -p /root/checkpoints && wget -q -O {CHECKPOINT_PATH} {CHECKPOINT_URL}",
    )
    .workdir("/opt/sam2-src")
)

with image.imports():
    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor


@app.cls(
    gpu="T4",
    image=image,
    scaledown_window=300,   # keep warm 5 min after last request
    max_containers=1,       # single shared container for the demo
    timeout=600,
)
@modal.concurrent(max_inputs=10)
class TrackerService:
    @modal.enter()
    def setup(self):
        torch.set_float32_matmul_precision("high")
        self.sam = build_sam2(CFG, CHECKPOINT_PATH, device="cuda")
        self.predictor = SAM2ImagePredictor(self.sam)
        self.sessions: dict = {}
        # warm up so the first real call isn't a 3s outlier
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            self.predictor.set_image(dummy)
            self.predictor.predict(
                box=np.array([100, 100, 300, 300], dtype=np.float32),
                multimask_output=False,
            )
        print("[TrackerService] ready")

    def _decode(self, b64_jpg: str) -> "np.ndarray":
        raw = base64.b64decode(b64_jpg)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _segment(self, rgb, box_xyxy):
        """Run SAM 2 image predictor with a box prompt; return (x,y,w,h) or None."""
        with torch.inference_mode():
            self.predictor.set_image(rgb)
            masks, _, _ = self.predictor.predict(
                box=np.array(box_xyxy, dtype=np.float32),
                multimask_output=False,
            )
        m = masks[0] > 0
        nz = np.argwhere(m)
        if len(nz) == 0:
            return None
        y1, x1 = nz.min(axis=0)
        y2, x2 = nz.max(axis=0)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

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
            return {"ok": True, "sessions": len(self.sessions)}

        @api.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    msg = await websocket.receive_text()
                    data = json.loads(msg)
                    typ = data.get("type")
                    sid = data.get("session_id", "default")

                    if typ == "init":
                        rgb = self._decode(data["frame"])
                        if rgb is None:
                            await websocket.send_text(json.dumps(
                                {"type": "init_fail", "session_id": sid, "error": "decode"}
                            ))
                            continue
                        x1, y1, x2, y2 = data["bbox"]
                        bbox = self._segment(rgb, (x1, y1, x2, y2))
                        if bbox is None:
                            await websocket.send_text(json.dumps(
                                {"type": "init_fail", "session_id": sid, "error": "no mask"}
                            ))
                            continue
                        self.sessions[sid] = {
                            "last_bbox": bbox,
                            "h": rgb.shape[0],
                            "w": rgb.shape[1],
                        }
                        await websocket.send_text(json.dumps(
                            {"type": "init_ok", "session_id": sid, "bbox": list(bbox)}
                        ))

                    elif typ == "track":
                        if sid not in self.sessions:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "session_id": sid, "error": "no session"}
                            ))
                            continue
                        sess = self.sessions[sid]
                        rgb = self._decode(data["frame"])
                        if rgb is None:
                            await websocket.send_text(json.dumps(
                                {"type": "bbox", "session_id": sid, "bbox": None, "ms": 0}
                            ))
                            continue
                        H, W = rgb.shape[:2]
                        x, y, w, h = sess["last_bbox"]
                        mx, my = int(w * 0.3), int(h * 0.3)
                        prompt = (
                            max(0, x - mx),
                            max(0, y - my),
                            min(W, x + w + mx),
                            min(H, y + h + my),
                        )
                        t0 = time.time()
                        bbox = self._segment(rgb, prompt)
                        dt_ms = int((time.time() - t0) * 1000)
                        if bbox is not None:
                            sess["last_bbox"] = bbox
                        await websocket.send_text(json.dumps(
                            {"type": "bbox", "session_id": sid,
                             "bbox": list(bbox) if bbox else None, "ms": dt_ms}
                        ))

                    elif typ == "close":
                        self.sessions.pop(sid, None)
                        await websocket.send_text(json.dumps(
                            {"type": "closed", "session_id": sid}
                        ))

                    else:
                        await websocket.send_text(json.dumps(
                            {"type": "error", "error": f"unknown type {typ}"}
                        ))
            except WebSocketDisconnect:
                pass

        return api
