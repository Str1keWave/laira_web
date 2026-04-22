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
  s->c: {"type":"init_ok",   "session_id":str, "bbox":[x,y,w,h], "debug":{...}}
  s->c: {"type":"init_fail", "session_id":str, "error":str, "debug":{...}}
  c->s: {"type":"track", "session_id":str, "frame":b64jpg,
         "bbox":[x,y,w,h] | omitted,    # if provided, used as prompt seed
         "raw": true | omitted}         # diagnostic: skip output growth clamp
  s->c: {"type":"bbox",  "session_id":str, "bbox":[x,y,w,h]|null,
         "ms":int, "debug":{score,coverage,n_components,raw_bbox,
                            largest_blob_bbox,seed_bbox,prompt_xyxy,raw_mode}}
  c->s: {"type":"stream", "session_id":str, "enabled":bool}
  s->c: {"type":"stream_ack", "session_id":str, "enabled":bool}
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
        """Run SAM 2 image predictor with a box prompt.

        Returns (bbox_xywh, debug_dict) or (None, debug_dict).
        debug_dict carries diagnostic info for the client overlay:
          score, coverage, n_components, raw_bbox, largest_blob_bbox.
        """
        H, W = rgb.shape[:2]
        img_area = H * W
        with torch.inference_mode():
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(
                box=np.array(box_xyxy, dtype=np.float32),
                multimask_output=False,
            )
        m = (masks[0] > 0).astype(np.uint8)
        mask_area = int(m.sum())
        coverage = mask_area / img_area if img_area else 0.0
        score = float(scores[0]) if len(scores) else 0.0

        debug = {
            "score": round(score, 3),
            "coverage": round(coverage, 4),
            "n_components": 0,
            "raw_bbox": None,
            "largest_blob_bbox": None,
        }

        if mask_area == 0:
            print(f"[segment] empty mask box={tuple(int(v) for v in box_xyxy)}")
            return None, debug
        if coverage > 0.70:
            print(f"[segment] REJECT mask: coverage {coverage:.2%} > 70%")
            return None, debug

        # Raw bbox = min/max of all mask pixels (current behavior).
        nz = np.argwhere(m)
        y1, x1 = nz.min(axis=0)
        y2, x2 = nz.max(axis=0)
        raw_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        debug["raw_bbox"] = raw_bbox

        # Connected-component analysis: largest blob's bbox can be much
        # smaller than raw_bbox if there are stray pixels. This is the
        # diagnostic that'll tell us whether stray pixels are the culprit.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        # label 0 is background; skip it
        if n_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            biggest = int(np.argmax(areas)) + 1
            bx = int(stats[biggest, cv2.CC_STAT_LEFT])
            by = int(stats[biggest, cv2.CC_STAT_TOP])
            bw = int(stats[biggest, cv2.CC_STAT_WIDTH])
            bh = int(stats[biggest, cv2.CC_STAT_HEIGHT])
            largest_blob_bbox = (bx, by, bw, bh)
        else:
            largest_blob_bbox = raw_bbox
        debug["n_components"] = max(0, n_labels - 1)
        debug["largest_blob_bbox"] = largest_blob_bbox

        # Loud log if raw bbox is much bigger than largest-blob bbox →
        # stray-pixel inflation is happening.
        rw, rh = raw_bbox[2], raw_bbox[3]
        lw, lh = largest_blob_bbox[2], largest_blob_bbox[3]
        if rw > lw * 1.3 or rh > lh * 1.3:
            print(f"[segment] STRAY-PIXEL inflation: "
                  f"raw={raw_bbox} largest_blob={largest_blob_bbox} "
                  f"n_comp={debug['n_components']}")

        print(f"[segment] box={tuple(int(v) for v in box_xyxy)} "
              f"score={score:.3f} coverage={coverage:.2%} "
              f"n_comp={debug['n_components']} "
              f"raw={raw_bbox} largest={largest_blob_bbox}")

        return raw_bbox, debug

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
                        bbox, debug = self._segment(rgb, (x1, y1, x2, y2))
                        if bbox is None:
                            await websocket.send_text(json.dumps(
                                {"type": "init_fail", "session_id": sid,
                                 "error": "no mask", "debug": debug}
                            ))
                            continue
                        self.sessions[sid] = {
                            "last_bbox": bbox,
                            "h": rgb.shape[0],
                            "w": rgb.shape[1],
                            "stream": False,
                        }
                        await websocket.send_text(json.dumps(
                            {"type": "init_ok", "session_id": sid,
                             "bbox": list(bbox), "debug": debug}
                        ))

                    elif typ == "stream":
                        # Diagnostic: client toggles continuous SAM mode.
                        # Server-side this is mostly a flag for logging; the
                        # browser still drives frame cadence via track msgs.
                        if sid in self.sessions:
                            self.sessions[sid]["stream"] = bool(data.get("enabled"))
                            print(f"[stream] sid={sid} -> "
                                  f"{self.sessions[sid]['stream']}")
                        await websocket.send_text(json.dumps(
                            {"type": "stream_ack", "session_id": sid,
                             "enabled": bool(data.get("enabled"))}
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
                        # Prefer the browser's current local-tracker bbox as the
                        # prompt seed (constant size, no feedback). Fall back to
                        # the server's last SAM bbox if the browser didn't send one.
                        seed_bbox = data.get("bbox") or sess["last_bbox"]
                        x, y, w, h = seed_bbox
                        # raw=true → diagnostic mode: skip output clamp so the
                        # client can see SAM's natural behavior.
                        raw_mode = bool(data.get("raw"))
                        # Cap expansion at a small absolute pixel value so a large
                        # bbox doesn't expand into a whole-frame prompt.
                        mx = min(int(w * 0.15), 24)
                        my = min(int(h * 0.15), 24)
                        prompt = (
                            max(0, x - mx),
                            max(0, y - my),
                            min(W, x + w + mx),
                            min(H, y + h + my),
                        )
                        t0 = time.time()
                        bbox, debug = self._segment(rgb, prompt)
                        # Output growth clamp (skipped in raw mode for diagnosis).
                        if bbox is not None and not raw_mode:
                            bx, by, bw, bh = bbox
                            max_w = int(w * 1.3)
                            max_h = int(h * 1.3)
                            if bw > max_w or bh > max_h:
                                print(f"[track] CLAMP growth: "
                                      f"({bw}x{bh}) -> cap ({max_w}x{max_h}) "
                                      f"from seed ({w}x{h})")
                                cx, cy = bx + bw // 2, by + bh // 2
                                bw = min(bw, max_w)
                                bh = min(bh, max_h)
                                bx = max(0, cx - bw // 2)
                                by = max(0, cy - bh // 2)
                                bbox = (bx, by, bw, bh)
                        dt_ms = int((time.time() - t0) * 1000)
                        if bbox is not None:
                            sess["last_bbox"] = bbox
                        # Echo the prompt & seed so the client can overlay them.
                        debug["seed_bbox"] = list(seed_bbox)
                        debug["prompt_xyxy"] = list(prompt)
                        debug["raw_mode"] = raw_mode
                        await websocket.send_text(json.dumps(
                            {"type": "bbox", "session_id": sid,
                             "bbox": list(bbox) if bbox else None,
                             "ms": dt_ms, "debug": debug}
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
