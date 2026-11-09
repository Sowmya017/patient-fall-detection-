"""
Offline video processing pipeline — wraps existing YOLO / MobileNetV3 / MediaPipe / CTR-GCN
models for Streamlit dashboard use. All models are loaded lazily and kept alive for the
lifetime of the VideoProcessor instance (cache_resource in app.py).
"""
import base64
import os
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SRC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))

from mediapipe_functions import mediapipe_to_ntu25
from role_classifier import MobileNetRoleClassifier
from ctrgcn_model import Model
import graph.ntu_rgb_d  # noqa: F401 — registers graph class used by Model

WINDOW_SIZE    = 72
MIN_FRAMES     = 15    # minimum skeleton frames before first inference
INFERENCE_FPS  = 18.0  # model training frame rate — subsample input to this rate
WINDOW_STRIDE  = WINDOW_SIZE // 4  # 18 frames — shift between multi-window candidates
CLASS_MAP = {0: "non_fall", 1: "fall"}

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 12), (12, 13), (13, 14),
    (0, 16), (16, 17), (17, 18),
    (2, 4), (4, 5), (5, 6),
    (2, 8), (8, 9), (9, 10),
]

ROLE_COLORS = {
    "patient":       (0, 200, 0),
    "medical_staff": (0, 165, 255),
    "other":         (200, 200, 0),
    "person":        (180, 180, 180),
    "unknown":       (128, 128, 128),
}

POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pose_model(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose landmarker model to {path} …")
        urllib.request.urlretrieve(POSE_LANDMARKER_URL, str(path))
    return str(path)


def _frame_to_b64(frame: np.ndarray, max_w: int = 640) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FallEvent:
    track_id: Any
    role: str
    start_frame: int
    end_frame: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    peak_probability: float = 0.0
    snapshot_b64: Optional[str] = None


@dataclass
class TrackSummary:
    track_id: Any
    role: str
    role_conf: float
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    total_frames: int = 0
    fall_frames: int = 0
    warning_frames: int = 0
    max_fall_prob: float = 0.0
    num_fall_events: int = 0


@dataclass
class ProcessingResult:
    output_video_path: str
    total_frames: int
    fps: float
    duration: float
    width: int
    height: int
    fall_events: List[FallEvent] = field(default_factory=list)
    track_summaries: Dict[Any, TrackSummary] = field(default_factory=dict)
    timeline: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    processing_fps: float = 0.0


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------

class VideoProcessor:
    def __init__(
        self,
        yolo_weights: str,
        role_weights: str,
        fall_weights: str,
        pose_model: str,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_weights = yolo_weights
        self.role_weights = role_weights
        self.fall_weights = fall_weights
        self.pose_model_path = _ensure_pose_model(Path(pose_model))

        self._yolo_obj     = None
        self._role_cls_obj = None
        self._fall_model_obj = None
        self._pose_lmk_obj = None

    # -- lazy loaders --------------------------------------------------------

    def _get_yolo(self):
        if self._yolo_obj is None:
            from ultralytics import YOLO
            self._yolo_obj = YOLO(self.yolo_weights)
        return self._yolo_obj

    def _get_role_cls(self):
        if self._role_cls_obj is None:
            self._role_cls_obj = MobileNetRoleClassifier(self.role_weights, device=self.device)
        return self._role_cls_obj

    def _get_fall_model(self):
        if self._fall_model_obj is None:
            m = Model(
                num_class=2, num_point=25, num_person=2, in_channels=3,
                graph="graph.ntu_rgb_d.Graph",
                graph_args={"labeling_mode": "spatial"},
            )
            ckpt = torch.load(self.fall_weights, map_location=self.device, weights_only=False)
            m.load_state_dict(ckpt)
            m.to(self.device)
            m.eval()
            self._fall_model_obj = m
        return self._fall_model_obj

    def _get_pose_lmk(self):
        if self._pose_lmk_obj is None:
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self.pose_model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_lmk_obj = mp_vision.PoseLandmarker.create_from_options(opts)
        return self._pose_lmk_obj

    # -- inference helpers ---------------------------------------------------

    @staticmethod
    def _preprocess_window(seq: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(seq) < MIN_FRAMES:
            return None
        # For short sequences tile (repeat) the frames to fill WINDOW_SIZE so the
        # model sees real motion rather than a long static-pose padding.
        if len(seq) < WINDOW_SIZE:
            times = (WINDOW_SIZE + len(seq) - 1) // len(seq)
            seq = (list(seq) * times)[:WINDOW_SIZE]
        x = np.array(seq[:WINDOW_SIZE], dtype=np.float32)
        if not np.any(np.abs(x) > 1e-8):
            return None
        x -= x[:, 0:1, :]
        bones = [(0, 1), (1, 2), (2, 3), (0, 12), (0, 16)]
        lengths = np.stack([np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in bones], axis=1)
        scale = np.median(lengths[lengths > 1e-6]) if np.any(lengths > 1e-6) else 1.0
        x /= scale
        fp = np.transpose(x, (2, 0, 1))[:, :, :, None]
        return np.concatenate([fp, np.zeros_like(fp)], axis=3).astype(np.float32)

    @torch.no_grad()
    def _predict_fall(self, seq: List[np.ndarray]) -> Tuple[str, float, float]:
        data = self._preprocess_window(seq)
        if data is None:
            return "no_pose", 0.0, 0.0
        t = torch.tensor(data).unsqueeze(0).to(self.device)
        probs = torch.softmax(self._get_fall_model()(t), dim=1)[0].cpu().numpy()
        p_fall = float(probs[1])
        return CLASS_MAP[int(np.argmax(probs))], float(probs[np.argmax(probs)]), p_fall

    def _predict_fall_multi(self, seq: List[np.ndarray]) -> Tuple[str, float, float]:
        """Run CTR-GCN on up to 3 staggered windows; return the max fall probability.

        Fast falls may peak in the middle of the buffer. Shifting windows by WINDOW_STRIDE
        ensures at least one window captures the fall near its centre.
        """
        n = len(seq)
        if n < MIN_FRAMES:
            return "no_pose", 0.0, 0.0
        max_fp = 0.0
        for shift in range(3):
            offset = shift * WINDOW_STRIDE
            if n - offset < MIN_FRAMES:
                break
            window = seq[max(0, n - WINDOW_SIZE - offset): n - offset if offset else None]
            _, _, fp = self._predict_fall(window)
            if fp > max_fp:
                max_fp = fp
        label = CLASS_MAP[1] if max_fp >= 0.5 else CLASS_MAP[0]
        conf  = max_fp if label == "fall" else (1.0 - max_fp)
        return label, conf, max_fp

    # -- drawing helpers -----------------------------------------------------

    @staticmethod
    def _draw_text(frame, text, xy, color, scale=0.55, thick=1):
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _draw_skeleton(self, frame, kpts, box, color):
        x1, y1, x2, y2 = box
        cw, ch = max(1, x2 - x1), max(1, y2 - y1)
        pts = [(int(x1 + k[0] * cw), int(y1 + k[1] * ch)) for k in kpts]
        for i, j in SKELETON_EDGES:
            cv2.line(frame, pts[i], pts[j], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)

    @staticmethod
    def _clip_box(box, w, h):
        x1, y1, x2, y2 = (int(v) for v in box)
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    @staticmethod
    def _new_state(frame_idx: int) -> Dict:
        return dict(
            sequence=deque(maxlen=WINDOW_SIZE),
            prev_kpts=None,
            role_label="person", role_conf=0.0,
            motion_label="warming_up", motion_conf=0.0,
            fall_prob=0.0, fall_streak=0, alarm_active=False,
            last_seen=frame_idx, last_role_frame=-10_000, last_motion_frame=-10_000,
            in_fall_event=False, event_start_frame=0, event_start_time=0.0,
            event_peak_prob=0.0, event_snapshot=None,
            # FPS normalisation accumulator (subsample to INFERENCE_FPS)
            pose_acc=0.0,
            # Physical signal histories
            aspect_history=deque(maxlen=24),   # bbox height/width ratio
            hip_y_history=deque(maxlen=24),    # absolute hip y (0-1 in frame)
        )

    # -- main entry ----------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_path: str,
        fall_threshold: float = 0.90,
        warning_threshold: float = 0.50,
        role_threshold: float = 0.50,
        infer_every: int = 6,
        role_every: int = 10,
        max_pose_targets: int = 3,
        alarm_confirmation_seconds: float = 0.60,
        disable_role_classifier: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ProcessingResult:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        # For short videos (< 2 s) allow a single prediction to confirm the alarm
        is_short = (total_frames / fps) < 2.0
        req_streak = max(1 if is_short else 2,
                         int(round((fps / max(1, infer_every)) * alarm_confirmation_seconds)))

        # Try H.264; fall back to mp4v for browsers
        for fourcc_str in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (W, H))
            if writer.isOpened():
                break

        yolo      = self._get_yolo()
        pose_lmk  = self._get_pose_lmk()
        role_cls  = None if disable_role_classifier else self._get_role_cls()

        track_state: Dict[Any, Dict] = {}
        fall_events: List[FallEvent] = []
        track_summaries: Dict[Any, TrackSummary] = {}
        timeline: List[Dict] = []
        sample_every = max(1, int(fps))

        t0 = time.perf_counter()
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame_time = frame_idx / fps
            annotated = frame.copy()

            # ---- YOLO tracking ----
            yolo_res = yolo.track(source=frame, persist=True, classes=[0],
                                  conf=0.25, iou=0.45, imgsz=960, verbose=False)[0]
            detections = []
            if yolo_res.boxes is not None and len(yolo_res.boxes):
                boxes_xy = yolo_res.boxes.xyxy.cpu().numpy()
                raw_ids  = (yolo_res.boxes.id.cpu().numpy().astype(int)
                            if yolo_res.boxes.id is not None
                            else [f"d{i}" for i in range(len(boxes_xy))])
                for idx, (bv, rid) in enumerate(zip(boxes_xy, raw_ids)):
                    tid = int(rid) if isinstance(rid, (int, np.integer)) else rid
                    x1, y1, x2, y2 = self._clip_box(bv, W, H)
                    if (x2 - x1) * (y2 - y1) == 0:
                        continue
                    s = track_state.setdefault(tid, self._new_state(frame_idx))
                    s["last_seen"] = frame_idx

                    # Role classification
                    if role_cls is None:
                        s["role_label"], s["role_conf"] = "person", 1.0
                    elif frame_idx - s["last_role_frame"] >= role_every:
                        s["role_label"], s["role_conf"] = role_cls.predict_crop(frame[y1:y2, x1:x2])
                        s["last_role_frame"] = frame_idx

                    is_patient = (role_cls is None or
                                  (s["role_label"] == "patient" and s["role_conf"] >= role_threshold))
                    detections.append(dict(tid=tid, box=(x1,y1,x2,y2), state=s, is_patient=is_patient))

                    if tid not in track_summaries:
                        track_summaries[tid] = TrackSummary(
                            track_id=tid, role=s["role_label"], role_conf=s["role_conf"],
                            first_seen_time=frame_time)
                    ts = track_summaries[tid]
                    ts.last_seen_time = frame_time
                    ts.total_frames  += 1
                    ts.role, ts.role_conf = s["role_label"], s["role_conf"]

            # ---- Sort: patients first, largest bounding box first ----
            detections.sort(key=lambda d: (not d["is_patient"],
                                           -((d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))))
            pose_targets = detections[:max_pose_targets]

            frame_max_fp = 0.0
            any_alarm    = False

            for det in pose_targets:
                x1, y1, x2, y2 = det["box"]
                s = det["state"]
                tid = det["tid"]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res    = pose_lmk.detect(mp_img)

                # ---- Physical signals (bbox aspect ratio & hip velocity) ----
                bbox_aspect = (y2 - y1) / max(1, x2 - x1)  # tall>1 standing, <1 fallen
                s["aspect_history"].append(bbox_aspect)

                if res.pose_landmarks:
                    mp_kpts  = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks[0]], np.float32)
                    ctr_kpts = mediapipe_to_ntu25(mp_kpts, s["prev_kpts"])
                    s["prev_kpts"] = ctr_kpts.copy()
                    sk_color = (0, 0, 255) if s["alarm_active"] else (0, 255, 255)
                    self._draw_skeleton(annotated, ctr_kpts, (x1, y1, x2, y2), sk_color)

                    # Track absolute hip y (joint 0) in frame-normalised coords
                    hip_abs_y = (y1 + float(ctr_kpts[0, 1]) * (y2 - y1)) / max(1, H)
                    s["hip_y_history"].append(hip_abs_y)

                    # FPS normalisation: only add to skeleton buffer at ~INFERENCE_FPS
                    s["pose_acc"] += INFERENCE_FPS / fps
                    if s["pose_acc"] >= 1.0:
                        s["sequence"].append(ctr_kpts)
                        s["pose_acc"] -= 1.0
                else:
                    s["pose_acc"] += INFERENCE_FPS / fps
                    if s["pose_acc"] >= 1.0:
                        s["sequence"].append(np.zeros((25, 3), np.float32))
                        s["pose_acc"] -= 1.0

                # CTR-GCN inference
                if (len(s["sequence"]) >= MIN_FRAMES and
                        frame_idx - s["last_motion_frame"] >= infer_every):
                    label, conf, fp = self._predict_fall_multi(list(s["sequence"]))

                    # ---- Physical signal boosts (multiplicative — safe against false alarms) ----
                    aspects = list(s["aspect_history"])
                    if len(aspects) >= 8:
                        aspect_delta = aspects[-1] - aspects[0]  # negative = getting wider
                        # Lying down right now
                        if bbox_aspect < 0.90:
                            fp = min(1.0, fp * 1.50)
                        # Rapidly transitioning from tall to wide (active fall)
                        elif aspect_delta < -0.40:
                            fp = min(1.0, fp * 1.35)

                    hip_ys = list(s["hip_y_history"])
                    if len(hip_ys) >= 6:
                        # Rapid downward hip movement (positive y = downward in image)
                        hip_velocity = hip_ys[-1] - hip_ys[-6]
                        if hip_velocity > 0.07:
                            fp = min(1.0, fp * 1.30)
                    s["motion_label"], s["motion_conf"], s["fall_prob"] = label, conf, fp
                    s["last_motion_frame"] = frame_idx

                    if fp >= fall_threshold:
                        s["fall_streak"] += 1
                    else:
                        # Close open fall event
                        if s["in_fall_event"]:
                            ev = FallEvent(
                                track_id=tid, role=s["role_label"],
                                start_frame=s["event_start_frame"], end_frame=frame_idx,
                                start_time=s["event_start_time"], end_time=frame_time,
                                duration=frame_time - s["event_start_time"],
                                peak_probability=s["event_peak_prob"],
                                snapshot_b64=s["event_snapshot"],
                            )
                            fall_events.append(ev)
                            track_summaries[tid].num_fall_events += 1
                            s["in_fall_event"] = False
                        s["fall_streak"] = 0
                        s["alarm_active"] = False

                    s["alarm_active"] = s["fall_streak"] >= req_streak

                    if s["alarm_active"] and not s["in_fall_event"]:
                        s["in_fall_event"] = True
                        s["event_start_frame"] = frame_idx
                        s["event_start_time"]  = frame_time
                        s["event_peak_prob"]   = fp
                        s["event_snapshot"]    = _frame_to_b64(annotated)
                    elif s["in_fall_event"] and fp > s["event_peak_prob"]:
                        s["event_peak_prob"]  = fp
                        s["event_snapshot"]   = _frame_to_b64(annotated)

                frame_max_fp = max(frame_max_fp, s["fall_prob"])
                any_alarm = any_alarm or s["alarm_active"]

                # Update track summary
                ts = track_summaries.get(tid)
                if ts:
                    ts.max_fall_prob = max(ts.max_fall_prob, s["fall_prob"])
                    if s["alarm_active"]:
                        ts.fall_frames += 1
                    elif s["fall_prob"] >= warning_threshold:
                        ts.warning_frames += 1

            # ---- Draw bounding boxes ----
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                s = det["state"]
                fp = s["fall_prob"]
                if s["alarm_active"]:
                    color, label = (0, 0, 255), "FALL"
                elif fp >= warning_threshold and s["motion_label"] != "warming_up":
                    color, label = (0, 165, 255), "WARNING"
                else:
                    color = ROLE_COLORS.get(s["role_label"], (180, 180, 180))
                    label = s["motion_label"].upper()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                self._draw_text(annotated, f"ID:{det['tid']} {s['role_label']}", (x1, max(16, y1-22)), color)
                if s["motion_label"] != "warming_up":
                    self._draw_text(annotated, f"{label} {fp:.2f}", (x1, max(32, y1-6)), color)

            # ---- FALL overlay banner ----
            if any_alarm:
                ov = annotated.copy()
                cv2.rectangle(ov, (0, 0), (W, 56), (0, 0, 180), -1)
                cv2.addWeighted(ov, 0.55, annotated, 0.45, 0, annotated)
                self._draw_text(annotated, "!! FALL DETECTED !!",
                                (W // 2 - 150, 40), (255, 255, 255), scale=1.0, thick=2)

            # ---- Header ----
            self._draw_text(annotated,
                            f"Frame {frame_idx}  People:{len(detections)}  {frame_time:.1f}s",
                            (10, 25), (220, 220, 220), scale=0.55)

            writer.write(annotated)

            # ---- Timeline sample (1/sec) ----
            if frame_idx % sample_every == 0:
                timeline.append(dict(
                    frame=frame_idx, time=round(frame_time, 2),
                    people=len(detections),
                    max_fall_prob=round(frame_max_fp, 3),
                    alarm=any_alarm,
                    warning=any(d["state"]["fall_prob"] >= warning_threshold and
                                not d["state"]["alarm_active"] for d in detections),
                ))

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames)

            # ---- Stale track cleanup (> 3 s unseen) ----
            stale_cutoff = frame_idx - int(fps * 3)
            for tid in list(track_state):
                if track_state[tid]["last_seen"] < stale_cutoff:
                    s = track_state.pop(tid)
                    if s["in_fall_event"]:
                        ev = FallEvent(
                            track_id=tid, role=s["role_label"],
                            start_frame=s["event_start_frame"], end_frame=frame_idx,
                            start_time=s["event_start_time"], end_time=frame_time,
                            duration=frame_time - s["event_start_time"],
                            peak_probability=s["event_peak_prob"],
                            snapshot_b64=s["event_snapshot"],
                        )
                        fall_events.append(ev)
                        if tid in track_summaries:
                            track_summaries[tid].num_fall_events += 1

        # Close remaining open events at end of video
        for tid, s in track_state.items():
            if s["in_fall_event"]:
                ev = FallEvent(
                    track_id=tid, role=s["role_label"],
                    start_frame=s["event_start_frame"], end_frame=frame_idx,
                    start_time=s["event_start_time"], end_time=duration,
                    duration=duration - s["event_start_time"],
                    peak_probability=s["event_peak_prob"],
                    snapshot_b64=s["event_snapshot"],
                )
                fall_events.append(ev)
                if tid in track_summaries:
                    track_summaries[tid].num_fall_events += 1

        cap.release()
        writer.release()

        proc_fps = frame_idx / max(0.001, time.perf_counter() - t0)

        stats = dict(
            total_frames   = frame_idx,
            total_people   = len(track_summaries),
            total_falls    = len(fall_events),
            warnings       = sum(1 for ts in track_summaries.values()
                                 if ts.warning_frames > 0 and ts.num_fall_events == 0),
            patients       = sum(1 for ts in track_summaries.values() if ts.role == "patient"),
            medical_staff  = sum(1 for ts in track_summaries.values() if ts.role == "medical_staff"),
            other          = sum(1 for ts in track_summaries.values()
                                 if ts.role in ("other", "person", "unknown")),
            processing_fps = round(proc_fps, 2),
            video_fps      = fps,
            duration       = round(duration, 2),
        )

        return ProcessingResult(
            output_video_path=output_path,
            total_frames=frame_idx, fps=fps, duration=duration, width=W, height=H,
            fall_events=fall_events, track_summaries=track_summaries,
            timeline=timeline, stats=stats, processing_fps=proc_fps,
        )
