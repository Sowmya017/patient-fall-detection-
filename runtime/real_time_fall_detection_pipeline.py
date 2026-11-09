import argparse
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))

from ctrgcn_model import Model
import graph.ntu_rgb_d
from mediapipe_functions import mediapipe_to_ntu25
from role_classifier import MobileNetRoleClassifier


WINDOW_SIZE = 72
CLASS_MAP = {0: "non_fall", 1: "fall"}
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 12), (12, 13), (13, 14),
    (0, 16), (16, 17), (17, 18), (2, 4), (4, 5), (5, 6),
    (2, 8), (8, 9), (9, 10),
]

POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def parse_source(value):
    if str(value).isdigit():
        return int(value)
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full camera pipeline: YOLO tracking, MobileNet role classification, MediaPipe pose, and CTR-GCN fall detection."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index such as 0, RTSP/HTTP stream URL, or video path.",
    )
    parser.add_argument("--yolo-weights", default="model_weights/patient_detection_yolov8n.pt")
    parser.add_argument("--role-weights", default="model_weights/role_classification_mobilenetv3_best.pt")
    parser.add_argument("--fall-weights", default="model_weights/fall_motion_ctrgcn_72f_impact.pt")
    parser.add_argument(
        "--disable-role-classifier",
        action="store_true",
        help="Skip MobileNet role classification and run fall detection on tracked people directly.",
    )
    parser.add_argument("--output", default="", help="Optional output video path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO image size.")
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=0.90,
        help="Fall probability threshold. Higher values reduce false alarms on long streams.",
    )
    parser.add_argument(
        "--alarm-confirmations",
        type=int,
        default=0,
        help="Fixed consecutive fall predictions needed for alarm. Use 0 for automatic FPS-based confirmation.",
    )
    parser.add_argument(
        "--alarm-confirmation-seconds",
        type=float,
        default=0.60,
        help="When --alarm-confirmations is 0, require this many seconds of consecutive fall predictions.",
    )
    parser.add_argument(
        "--role-threshold",
        type=float,
        default=0.50,
        help="Minimum MobileNet confidence needed to trust the patient label.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=6,
        help="Run CTR-GCN every N frames after the 72-frame buffer is full.",
    )
    parser.add_argument(
        "--role-every",
        type=int,
        default=10,
        help="Refresh MobileNet role prediction every N frames per track.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=1,
        help="Maximum patient tracks to run MediaPipe/CTR-GCN on per frame.",
    )
    parser.add_argument(
        "--include-non-patient",
        action="store_true",
        help="Also run fall detection on people not classified as patient.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show the live window. Useful for headless runs or saving only.",
    )
    parser.add_argument(
        "--fps-log-every",
        type=float,
        default=5.0,
        help="Print measured processing FPS every N seconds. Use 0 to disable console FPS logs.",
    )
    parser.add_argument(
        "--pose-model",
        default="model_weights/pose_landmarker_full.task",
        help="Path to MediaPipe Pose Landmarker .task model file. Downloaded automatically if missing.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def ensure_pose_model(path):
    path = resolve_path(path)
    if not path.exists():
        print(f"Downloading MediaPipe pose landmarker model to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_LANDMARKER_URL, str(path))
        print("Download complete.")
    return str(path)


def get_alarm_confirmations(args, processing_fps, camera_fps):
    if args.alarm_confirmations > 0:
        return args.alarm_confirmations

    effective_fps = processing_fps if processing_fps > 0 else camera_fps
    if effective_fps <= 0:
        effective_fps = 18.0

    prediction_rate = effective_fps / max(1, args.infer_every)
    return max(2, int(round(prediction_rate * args.alarm_confirmation_seconds)))


def load_fall_model(weights_path, device):
    model = Model(
        num_class=2,
        num_point=25,
        num_person=2,
        in_channels=3,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def preprocess_window(sequence):
    x = np.array(sequence, dtype=np.float32)
    if len(x) != WINDOW_SIZE:
        return None
    if not np.any(np.abs(x) > 1e-8):
        return None

    root = x[:, 0, :]
    x = x - root[:, None, :]

    bones = [(0, 1), (1, 2), (2, 3), (0, 12), (0, 16)]
    lengths = [np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in bones]
    lengths = np.stack(lengths, axis=1)
    scale = np.median(lengths[lengths > 1e-6]) if np.any(lengths > 1e-6) else 1.0
    x = x / scale

    first_person = np.transpose(x, (2, 0, 1))[:, :, :, None]
    second_person = np.zeros_like(first_person)
    return np.concatenate([first_person, second_person], axis=3).astype(np.float32)


@torch.no_grad()
def predict_fall(model, sequence, device):
    data = preprocess_window(sequence)
    if data is None:
        return "no_pose", 0.0, 0.0

    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    fall_probability = float(probabilities[1])
    pred = int(np.argmax(probabilities))
    return CLASS_MAP[pred], float(probabilities[pred]), fall_probability


def clip_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def draw_text(frame, text, origin, color, scale=0.6, thickness=2):
    x, y = origin
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_crop_skeleton(frame, keypoints, box, color):
    x1, y1, x2, y2 = box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    points = []
    for point in keypoints:
        px = int(x1 + point[0] * crop_w)
        py = int(y1 + point[1] * crop_h)
        points.append((px, py))

    for i, j in SKELETON_EDGES:
        cv2.line(frame, points[i], points[j], color, 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def get_track_id(raw_id, fallback_index):
    if raw_id is None:
        return f"det_{fallback_index}"
    return int(raw_id)


def main():
    args = parse_args()
    if args.alarm_confirmations < 0:
        raise ValueError("--alarm-confirmations must be 0 for automatic mode or 1 or greater for fixed mode.")
    if args.alarm_confirmation_seconds <= 0:
        raise ValueError("--alarm-confirmation-seconds must be greater than 0.")
    if args.infer_every < 1:
        raise ValueError("--infer-every must be 1 or greater.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The full camera pipeline needs ultralytics. Install project dependencies with "
            "`pip install -r requirements.txt` from C:\\GP\\FallDetection_Final."
        ) from exc

    yolo = YOLO(str(resolve_path(args.yolo_weights)))
    role_classifier = None
    if not args.disable_role_classifier:
        role_classifier = MobileNetRoleClassifier(resolve_path(args.role_weights), device=device)
    fall_model = load_fall_model(resolve_path(args.fall_weights), device)

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    track_state = {}
    frame_index = 0
    last_status = "warming_up"
    last_fall_probability = 0.0
    run_start_time = time.perf_counter()
    fps_window_start = run_start_time
    fps_window_frames = 0
    measured_fps = 0.0
    average_fps = 0.0

    pose_model_path = ensure_pose_model(args.pose_model)
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_start_time = time.perf_counter()
            frame_index += 1
            fps_window_frames += 1
            annotated = frame.copy()
            results = yolo.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            detections = []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else [None] * len(boxes)

                for det_index, (box_values, raw_id) in enumerate(zip(boxes, ids)):
                    box = clip_box(box_values, width, height)
                    if box_area(box) <= 0:
                        continue

                    track_id = get_track_id(raw_id, det_index)
                    state = track_state.setdefault(
                        track_id,
                        {
                            "sequence": deque(maxlen=WINDOW_SIZE),
                            "previous_keypoints": None,
                            "role_label": "person",
                            "role_conf": 0.0,
                            "motion_label": "warming_up",
                            "motion_conf": 0.0,
                            "fall_probability": 0.0,
                            "fall_streak": 0,
                            "alarm_active": False,
                            "required_alarm_confirmations": 1,
                            "last_seen": frame_index,
                            "last_role_frame": -10_000,
                            "last_motion_frame": -10_000,
                        },
                    )
                    state["last_seen"] = frame_index

                    if role_classifier is None:
                        state["role_label"] = "person"
                        state["role_conf"] = 1.0
                        state["last_role_frame"] = frame_index
                    elif frame_index - state["last_role_frame"] >= args.role_every:
                        x1, y1, x2, y2 = box
                        crop = frame[y1:y2, x1:x2]
                        state["role_label"], state["role_conf"] = role_classifier.predict_crop(crop)
                        state["last_role_frame"] = frame_index
                    is_patient = (
                        role_classifier is None
                        or (state["role_label"] == "patient" and state["role_conf"] >= args.role_threshold)
                    )

                    detections.append(
                        {
                            "track_id": track_id,
                            "box": box,
                            "state": state,
                            "is_patient": is_patient,
                        }
                    )

            detections.sort(key=lambda item: (not item["is_patient"], -box_area(item["box"])))
            pose_targets = [
                item for item in detections if args.include_non_patient or item["is_patient"]
            ][: max(1, args.max_patients)]

            for item in pose_targets:
                box = item["box"]
                state = item["state"]
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = pose_landmarker.detect(mp_image)
                if result.pose_landmarks:
                    mp_keypoints = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]],
                        dtype=np.float32,
                    )
                    ctr_keypoints = mediapipe_to_ntu25(mp_keypoints, state["previous_keypoints"])
                    state["previous_keypoints"] = ctr_keypoints.copy()
                    state["sequence"].append(ctr_keypoints)
                    draw_crop_skeleton(annotated, ctr_keypoints, box, (0, 255, 255))
                else:
                    state["sequence"].append(np.zeros((25, 3), dtype=np.float32))

                if len(state["sequence"]) == WINDOW_SIZE and frame_index - state["last_motion_frame"] >= args.infer_every:
                    label, confidence, fall_probability = predict_fall(fall_model, list(state["sequence"]), device)
                    required_confirmations = get_alarm_confirmations(args, measured_fps or average_fps, fps)
                    is_fall_prediction = fall_probability >= args.fall_threshold
                    if is_fall_prediction:
                        state["fall_streak"] += 1
                    else:
                        state["fall_streak"] = 0
                        state["alarm_active"] = False

                    state["required_alarm_confirmations"] = required_confirmations
                    state["alarm_active"] = state["fall_streak"] >= required_confirmations
                    state["motion_label"] = "fall" if is_fall_prediction else "non_fall"
                    state["motion_conf"] = confidence
                    state["fall_probability"] = fall_probability
                    state["last_motion_frame"] = frame_index
                    last_status = "alarm" if state["alarm_active"] else state["motion_label"]
                    last_fall_probability = fall_probability

            active_ids = {item["track_id"] for item in detections}
            for track_id in list(track_state.keys()):
                if track_id not in active_ids and frame_index - track_state[track_id]["last_seen"] > 90:
                    del track_state[track_id]

            for item in detections:
                box = item["box"]
                state = item["state"]
                x1, y1, x2, y2 = box
                fall_probability = state["fall_probability"]
                required_confirmations = state["required_alarm_confirmations"]
                is_alarm = state["fall_streak"] >= required_confirmations
                state["alarm_active"] = is_alarm
                color = (0, 0, 255) if is_alarm else ((0, 180, 0) if item["is_patient"] else (180, 180, 0))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                draw_text(
                    annotated,
                    f"ID {item['track_id']} | {state['role_label']} {state['role_conf']:.2f}",
                    (x1, max(22, y1 - 28)),
                    color,
                )
                draw_text(
                    annotated,
                    f"{state['motion_label']} p_fall={fall_probability:.2f} streak={state['fall_streak']}/{required_confirmations}",
                    (x1, max(44, y1 - 8)),
                    color,
                )

            header_color = (0, 0, 255) if last_status == "alarm" else (0, 220, 0)
            now = time.perf_counter()
            elapsed = max(now - run_start_time, 1e-6)
            average_fps = frame_index / elapsed
            window_elapsed = now - fps_window_start
            if window_elapsed >= 1.0:
                measured_fps = fps_window_frames / window_elapsed
                fps_window_start = now
                fps_window_frames = 0

            if args.fps_log_every > 0 and elapsed >= args.fps_log_every:
                log_bucket = int(elapsed // args.fps_log_every)
                previous_bucket = int(max(elapsed - max(now - frame_start_time, 1e-6), 0) // args.fps_log_every)
                if log_bucket != previous_bucket:
                    print(
                        f"frame={frame_index} processing_fps={measured_fps:.2f} "
                        f"average_fps={average_fps:.2f} camera_fps={fps:.2f}",
                        flush=True,
                    )

            draw_text(
                annotated,
                f"Full pipeline | frame {frame_index} | status={last_status} p_fall={last_fall_probability:.2f}",
                (20, 35),
                header_color,
                scale=0.8,
            )
            draw_text(
                annotated,
                f"FPS process={measured_fps:.1f} avg={average_fps:.1f} camera={fps:.1f}",
                (20, 68),
                (255, 255, 255),
                scale=0.7,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Patient Fall Detection Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        pose_landmarker.close()

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("Full camera pipeline stopped.")
    print(f"Frames processed: {frame_index}")
    print(f"Average processing FPS: {average_fps:.2f}")
    print(f"Camera reported FPS: {fps:.2f}")
    if args.output:
        print(f"Output video: {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
