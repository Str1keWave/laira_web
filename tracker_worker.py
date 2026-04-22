"""
LaiRA tracker service — Modal serverless GPU.

Architecture (current): SAMURAI on top of SAM 2.1 video predictor.
Each session maintains a true memory bank — every track call appends the new
frame to the inference state and runs one propagation step. SAMURAI's
motion-aware (Kalman-filtered) memory selection picks the most useful prior
frames as conditioning. This is the actual tracker, not per-frame stateless
segmentation.

The SAMURAI video predictor was written for offline use: `init_state` requires
a video folder and `propagate_in_video` iterates the whole thing. We adapt it
for streaming by:
  1. seeding init_state from a temp dir with the first frame,
  2. preprocessing each new frame the same way the official loader does,
  3. appending to inference_state["images"] and bumping num_frames,
  4. calling propagate_in_video(start_frame_idx=N, max_frame_num_to_track=1)
     and pulling the single yielded mask.

This works because all the predictor really cares about is that
inference_state["images"][frame_idx] exists and num_frames is correct;
the memory bank does the rest.

Protocol (JSON over websocket /ws):
  c->s: {"type":"init",  "session_id":str, "frame":b64jpg, "bbox":[x1,y1,x2,y2]}
  s->c: {"type":"init_ok",   "session_id":str, "bbox":[x,y,w,h], "debug":{...}}
  s->c: {"type":"init_fail", "session_id":str, "error":str, "debug":{...}}
  c->s: {"type":"track", "session_id":str, "frame":b64jpg,
         "raw": true | omitted}      # diagnostic: skip output growth clamp
  s->c: {"type":"bbox",  "session_id":str, "bbox":[x,y,w,h]|null,
         "ms":int, "debug":{...}}
  c->s: {"type":"stream", "session_id":str, "enabled":bool}
  s->c: {"type":"stream_ack", "session_id":str, "enabled":bool}
  c->s: {"type":"close", "session_id":str}

Note: client may still send "bbox" in track messages — it's ignored now.
SAMURAI tracks via memory bank, not via per-frame box prompts.

Deploy:    modal deploy tracker_worker.py
Healthz:   GET  https://<workspace>--laira-tracker-trackerservice-fastapi-app.modal.run/healthz
Websocket: WS   wss://<workspace>--laira-tracker-trackerservice-fastapi-app.modal.run/ws
"""
import base64
import json
import os
import tempfile
import time

import modal

app = modal.App("laira-tracker")

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
)
CHECKPOINT_PATH = "/root/checkpoints/sam2.1_hiera_base_plus.pt"
# SAMURAI ships its own configs that wire the Kalman-augmented memory selection.
CFG = "configs/samurai/sam2.1_hiera_b+.yaml"

# Pruning windows. Sessions are unbounded in length; per-session memory is
# bounded by these constants instead of by a hard frame cap.
#
# IMAGES_KEEP_RECENT — number of recent raw frames to retain in
# state["images"] (each ~12MB on CPU). We always keep frame 0 too as the
# conditioning anchor. SAMURAI's memory bank attends over the cached
# maskmem_features (in output_dict), NOT raw images, so old images can be
# dropped immediately after the frame is processed. Keeping a small window
# is purely defensive in case the predictor ever re-fetches.
IMAGES_KEEP_RECENT = 3
# OUTPUT_KEEP — how many recent non-conditioning per-frame outputs to keep
# in the memory bank. The model only uses ~7 of these via SAMURAI's
# motion-aware selection, but keep a generous window so selection has room
# to reach further back when motion is stable. Each entry is ~0.5MB.
OUTPUT_KEEP = 60

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
        "loguru",
        "scipy",        # SAMURAI's Kalman filter imports scipy.linalg
    )
    .run_commands(
        # SAMURAI is a fork of SAM 2 with Kalman-filtered memory selection.
        # The actual sam2 Python package lives at samurai/sam2/sam2/.
        "git clone https://github.com/yangchris11/samurai.git /opt/samurai-src",
        # Skip the CUDA extension build — it's only needed for an optional
        # mask post-processing step and adds ~3 min to image build for nothing
        # we care about right now.
        "cd /opt/samurai-src/sam2 && SAM2_BUILD_CUDA=0 pip install -e .",
        f"mkdir -p /root/checkpoints && wget -q -O {CHECKPOINT_PATH} {CHECKPOINT_URL}",
    )
    # Workdir is the parent of the sam2 package dir so Hydra finds configs and
    # SAM 2's "are you running from inside the package" check doesn't trip.
    .workdir("/opt/samurai-src/sam2")
)

with image.imports():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor


# ---------- frame preprocessing ----------
# SAM 2 normalizes frames with ImageNet mean/std after resize to image_size².
# We replicate that here so a streaming frame matches what load_video_frames
# would produce. Source: sam2/utils/misc.py::_load_img_as_tensor.
_SAM2_MEAN = (0.485, 0.456, 0.406)
_SAM2_STD = (0.229, 0.224, 0.225)


def _frame_to_sam2_tensor(rgb_np, image_size):
    """RGB ndarray (H, W, 3) uint8 → preprocessed tensor (3, image_size, image_size).

    Output sits on CPU; predictor moves to GPU per-frame as needed.
    """
    pil = Image.fromarray(rgb_np)
    pil = pil.resize((image_size, image_size))
    arr = (np.array(pil) / 255.0).astype(np.float32)
    img = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    mean = torch.tensor(_SAM2_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_SAM2_STD, dtype=torch.float32).view(3, 1, 1)
    return (img - mean) / std


def _mask_to_bbox(mask_2d):
    """Binary mask (H, W) → ((x, y, w, h), debug) using largest connected component.

    Returns (None, debug) for empty masks.
    """
    m = mask_2d.astype(np.uint8)
    H, W = m.shape
    debug = {
        "score": None,         # filled by caller (SAMURAI iou/kf scores)
        "coverage": 0.0,
        "n_components": 0,
        "raw_bbox": None,
        "largest_blob_bbox": None,
    }
    area = int(m.sum())
    if area == 0:
        return None, debug
    debug["coverage"] = round(area / (H * W), 4)

    nz = np.argwhere(m)
    y1, x1 = nz.min(axis=0)
    y2, x2 = nz.max(axis=0)
    raw_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    debug["raw_bbox"] = raw_bbox

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
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

    # Use largest blob — stray-pixel inflation was the suspected failure mode
    # under the old image-predictor regime. Even with SAMURAI's better masks,
    # this is the safer choice for downstream use.
    return largest_blob_bbox, debug


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
        self.predictor = build_sam2_video_predictor(CFG, CHECKPOINT_PATH, device="cuda")
        # Per-session state: each entry is the SAMURAI inference_state dict
        # produced by predictor.init_state(), plus our bookkeeping.
        self.sessions: dict = {}

        # Warmup: exercise the full streaming pipeline once on a dummy frame
        # so the first real session doesn't pay JIT/kernel-compile latency
        # AND so any breakage in the dict-images path fails at boot, not on
        # the first user-facing request.
        with tempfile.TemporaryDirectory() as tdir:
            dummy = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(tdir, "00000.jpg"),
                        cv2.cvtColor(dummy, cv2.COLOR_RGB2BGR))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = self.predictor.init_state(tdir, offload_video_to_cpu=True,
                                                  offload_state_to_cpu=True)
                self.predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=0,
                    box=np.array([100, 100, 300, 300], dtype=np.float32),
                )
                # Convert images to dict and run one streaming-style frame
                # through propagate_in_video. This validates the dict path.
                state["images"] = {0: state["images"][0]}
                state["images"][1] = _frame_to_sam2_tensor(
                    dummy, self.predictor.image_size
                )
                state["num_frames"] = 2
                gen = self.predictor.propagate_in_video(
                    state, start_frame_idx=1, max_frame_num_to_track=1,
                )
                next(gen)
        print("[TrackerService] SAMURAI ready (streaming path validated)")

    # ---------- helpers ----------
    def _decode(self, b64_jpg: str):
        raw = base64.b64decode(b64_jpg)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _new_session(self, sid, first_rgb, bbox_xyxy):
        """Initialize a SAMURAI inference state from the first frame + box prompt.

        Returns (bbox_xywh, debug) or (None, debug).
        """
        H, W = first_rgb.shape[:2]
        with tempfile.TemporaryDirectory() as tdir:
            # init_state needs a video folder; give it a 1-frame "video".
            jpg_path = os.path.join(tdir, "00000.jpg")
            cv2.imwrite(jpg_path, cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = self.predictor.init_state(
                    tdir, offload_video_to_cpu=True, offload_state_to_cpu=True
                )
                _, _, masks = self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=0,
                    box=np.array(bbox_xyxy, dtype=np.float32),
                )

        # masks is (B, 1, H_orig, W_orig) — float logits at video resolution.
        mask = (masks[0, 0].cpu().numpy() > 0)
        bbox, debug = _mask_to_bbox(mask)
        debug["score"] = None  # add_new_points_or_box doesn't expose iou here
        if bbox is None:
            return None, debug

        # Convert state["images"] from a (1, 3, S, S) tensor into a dict
        # keyed by frame_idx. The predictor only does state["images"][i] in
        # _get_image_feature, which works on either a tensor or a dict — but
        # a dict lets us drop old entries without renumbering frame indices.
        state["images"] = {0: state["images"][0]}

        self.sessions[sid] = {
            "state": state,
            "next_frame_idx": 1,   # 0 was the init frame
            "h": H, "w": W,
            "last_bbox": bbox,
            "stream": False,
        }
        return bbox, debug

    def _prune_session(self, state, current_frame_idx):
        """Drop old per-frame storage to bound per-session RAM.

        Safe because:
          - state["images"][K] is only re-read by _get_image_feature on
            cache miss, and we only ever process frame indices forward.
            After frame K is propagated, its image isn't needed again.
          - The memory bank attends over maskmem_features stored in
            output_dict, not raw images, so dropping old images doesn't
            affect tracking quality.
          - We never prune frame 0's storage — it's the conditioning anchor.
          - non_cond_frame_outputs entries older than the model's effective
            memory window are dead weight.
        """
        # 1) Raw images: keep frame 0 + a small recent window.
        images = state["images"]
        keep_set = {0, *range(
            max(0, current_frame_idx - IMAGES_KEEP_RECENT + 1),
            current_frame_idx + 1,
        )}
        for k in [k for k in images if k not in keep_set]:
            del images[k]

        # 2) Output dict bookkeeping: drop non-conditioning entries older
        # than OUTPUT_KEEP frames. cond_frame_outputs (frame 0) untouched.
        cutoff = current_frame_idx - OUTPUT_KEEP
        if cutoff > 0:
            ncfo = state["output_dict"]["non_cond_frame_outputs"]
            for k in [k for k in ncfo if k < cutoff]:
                del ncfo[k]
            for obj_dict in state["output_dict_per_obj"].values():
                ncfo_obj = obj_dict["non_cond_frame_outputs"]
                for k in [k for k in ncfo_obj if k < cutoff]:
                    del ncfo_obj[k]
            fat = state["frames_already_tracked"]
            for k in [k for k in fat if k < cutoff]:
                del fat[k]

    def _track_frame(self, sid, rgb, raw_mode):
        """Append a frame to the session and run one propagation step.

        Returns (bbox_xywh|None, debug_dict).
        """
        sess = self.sessions[sid]
        state = sess["state"]
        frame_idx = sess["next_frame_idx"]

        # state["images"] is now a dict {frame_idx: tensor(3, S, S)} after
        # _new_session converted it from the init tensor. Insert the new
        # frame and bump num_frames so propagate_in_video's range() is right.
        new_tensor = _frame_to_sam2_tensor(rgb, self.predictor.image_size)
        state["images"][frame_idx] = new_tensor
        state["num_frames"] = frame_idx + 1
        sess["next_frame_idx"] = frame_idx + 1

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            gen = self.predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            )
            try:
                _, _, video_res_masks = next(gen)
            except StopIteration:
                return None, {"score": None, "coverage": 0.0,
                              "n_components": 0, "raw_bbox": None,
                              "largest_blob_bbox": None}

        # Prune old per-frame storage now that this frame is processed.
        self._prune_session(state, frame_idx)

        mask = (video_res_masks[0, 0].cpu().numpy() > 0)
        bbox, debug = _mask_to_bbox(mask)

        # Pull SAMURAI's iou + kalman scores out of the most recent
        # non-cond output (they live on the last frame's compact_current_out).
        per_obj = state["output_dict_per_obj"]
        if per_obj:
            obj_dict = next(iter(per_obj.values()))
            entry = obj_dict["non_cond_frame_outputs"].get(frame_idx)
            if entry is not None:
                iou = entry.get("object_score_logits")
                kf = entry.get("kf_score")
                if iou is not None:
                    try:
                        debug["score"] = round(float(iou.flatten()[0].item()), 3)
                    except Exception:
                        pass
                if kf is not None:
                    try:
                        debug["kf_score"] = round(float(np.asarray(kf).flatten()[0]), 3)
                    except Exception:
                        pass

        # Output growth clamp (skipped in raw mode for diagnosis). The seed
        # for this clamp is the previous bbox.
        if bbox is not None and not raw_mode and sess["last_bbox"] is not None:
            px, py, pw, ph = sess["last_bbox"]
            bx, by, bw, bh = bbox
            max_w = int(pw * 1.5)   # SAMURAI legitimately tracks deformation;
            max_h = int(ph * 1.5)   # be looser than the old image-predictor cap
            if bw > max_w or bh > max_h:
                print(f"[track] CLAMP growth: ({bw}x{bh}) -> "
                      f"cap ({max_w}x{max_h}) from prev ({pw}x{ph})")
                cx, cy = bx + bw // 2, by + bh // 2
                bw = min(bw, max_w)
                bh = min(bh, max_h)
                bx = max(0, cx - bw // 2)
                by = max(0, cy - bh // 2)
                bbox = (bx, by, bw, bh)

        if bbox is not None:
            sess["last_bbox"] = bbox

        debug["frame_idx"] = frame_idx
        debug["raw_mode"] = raw_mode
        return bbox, debug

    def _close_session(self, sid):
        sess = self.sessions.pop(sid, None)
        if sess is not None:
            # Drop reference to predictor state so Python can GC the tensors.
            sess["state"] = None

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
            return {"ok": True, "tracker": "samurai-video",
                    "sessions": len(self.sessions)}

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
                        # Drop any stale session under this id first.
                        self._close_session(sid)
                        rgb = self._decode(data["frame"])
                        if rgb is None:
                            await websocket.send_text(json.dumps(
                                {"type": "init_fail", "session_id": sid,
                                 "error": "decode"}
                            ))
                            continue
                        x1, y1, x2, y2 = data["bbox"]
                        try:
                            bbox, debug = self._new_session(sid, rgb, (x1, y1, x2, y2))
                        except Exception as e:
                            print(f"[init] EXC sid={sid}: {e!r}")
                            await websocket.send_text(json.dumps(
                                {"type": "init_fail", "session_id": sid,
                                 "error": f"init_exc: {e}"}
                            ))
                            continue
                        if bbox is None:
                            await websocket.send_text(json.dumps(
                                {"type": "init_fail", "session_id": sid,
                                 "error": "no mask", "debug": debug}
                            ))
                            continue
                        print(f"[init] sid={sid} bbox={bbox} "
                              f"n_comp={debug['n_components']}")
                        await websocket.send_text(json.dumps(
                            {"type": "init_ok", "session_id": sid,
                             "bbox": list(bbox), "debug": debug}
                        ))

                    elif typ == "stream":
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
                                {"type": "error", "session_id": sid,
                                 "error": "no session"}
                            ))
                            continue
                        sess = self.sessions[sid]
                        rgb = self._decode(data["frame"])
                        if rgb is None:
                            await websocket.send_text(json.dumps(
                                {"type": "bbox", "session_id": sid,
                                 "bbox": None, "ms": 0}
                            ))
                            continue
                        raw_mode = bool(data.get("raw"))
                        t0 = time.time()
                        try:
                            bbox, debug = self._track_frame(sid, rgb, raw_mode)
                        except Exception as e:
                            print(f"[track] EXC sid={sid}: {e!r}")
                            await websocket.send_text(json.dumps(
                                {"type": "bbox", "session_id": sid,
                                 "bbox": None, "ms": 0,
                                 "debug": {"error": f"track_exc: {e}"}}
                            ))
                            continue
                        dt_ms = int((time.time() - t0) * 1000)
                        debug["ms"] = dt_ms
                        # seed_bbox for client overlay continuity (the previous
                        # bbox is what the client should think of as the seed).
                        debug["seed_bbox"] = (
                            list(sess["last_bbox"]) if sess["last_bbox"] else None
                        )
                        await websocket.send_text(json.dumps(
                            {"type": "bbox", "session_id": sid,
                             "bbox": list(bbox) if bbox else None,
                             "ms": dt_ms, "debug": debug}
                        ))

                    elif typ == "close":
                        self._close_session(sid)
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
