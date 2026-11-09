import argparse
import os
import sys
from collections import deque
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import mediapipe as mp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from ctrgcn_model import Model
import graph.ntu_rgb_d
from mediapipe_functions import mediapipe_to_ntu25
from pose_motion_classification import EDGES, TARGET_T, preprocess_window


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(path):
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def list_images(sequence_dir):
    paths = [
        path
        for path in Path(sequence_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: natural_key(p.name))


def load_model(weights_path, device):
    model = Model(
        num_class=2,
        num_point=25,
        num_person=2,
        in_channels=3,
        graph="graph.ntu_rgb_d.Graph",
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def draw_skeleton(frame, keypoints):
    h, w = frame.shape[:2]
    for i, j in EDGES:
        pt1 = tuple((keypoints[i, :2] * [w, h]).astype(int))
        pt2 = tuple((keypoints[j, :2] * [w, h]).astype(int))
        cv2.line(frame, pt1, pt2, (0, 220, 90), 2)
    for point in keypoints:
        pt = tuple((point[:2] * [w, h]).astype(int))
        cv2.circle(frame, pt, 3, (40, 80, 255), -1)


def draw_panel(frame, state):
    h, w = frame.shape[:2]
    panel_h = 155
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    status_color = (0, 255, 0)
    if state["alert"]:
        status_color = (0, 0, 255)
    elif state["latest_prob"] >= state["threshold"]:
        status_color = (0, 190, 255)

    lines = [
        f"Real-time config: 18 FPS | rolling window 72 frames = 4.0 sec | predict every 6 frames = 0.33 sec",
        f"Frame {state['frame_idx']}/{state['total_frames']} | buffer {state['buffer_len']}/72 | warmup: {state['warmup_left']:.1f}s left",
        f"Latest CTR-GCN fall probability: {state['latest_prob']:.3f} | threshold {state['threshold']:.2f} | confirmations {state['confirmations']}/{state['required_confirmations']}",
        f"Decision: {state['decision']}",
    ]

    y = 27
    for line in lines:
        color = status_color if line.startswith("Decision") else (235, 235, 235)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 32

    bar_x, bar_y, bar_w, bar_h = 18, panel_h - 25, w - 36, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    fill = int(bar_w * min(1.0, state["buffer_len"] / 72.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 180, 255), -1)


def draw_probability_trace(frame, probabilities, threshold):
    h, w = frame.shape[:2]
    chart_w, chart_h = min(360, w - 30), 95
    x0, y0 = w - chart_w - 18, h - chart_h - 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + chart_w, y0 + chart_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + chart_w, y0 + chart_h), (150, 150, 150), 1)

    thresh_y = int(y0 + chart_h - threshold * chart_h)
    cv2.line(frame, (x0, thresh_y), (x0 + chart_w, thresh_y), (0, 190, 255), 1)
    cv2.putText(frame, "fall prob", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    if len(probabilities) < 2:
        return

    vals = probabilities[-60:]
    points = []
    for i, value in enumerate(vals):
        x = int(x0 + (i / max(len(vals) - 1, 1)) * chart_w)
        y = int(y0 + chart_h - np.clip(value, 0, 1) * chart_h)
        points.append((x, y))
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(frame, p1, p2, (0, 80, 255), 2)


@torch.no_grad()
def predict(model, sequence_buffer, device):
    data = preprocess_window(list(sequence_buffer))
    if data is None:
        return 0.0
    tensor = torch.tensor(data).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    return float(probabilities[1].item())


def parse_args():
    parser = argparse.ArgumentParser(description="Render real-time rolling-window fall detection demo.")
    parser.add_argument(
        "--sequence-dir",
        default=r"F:\GP_Dataset\Subjects\Subject12\Activity02\Trial3\Subject12Activity2Trial3Camera1",
    )
    parser.add_argument("--weights", default="model_weights/fall_motion_ctrgcn_72f_impact.pt")
    parser.add_argument("--output", default="evaluation/results/runtime_visualizations/fall_realtime_demo.mp4")
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--window", type=int, default=72)
    parser.add_argument("--predict-every", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--confirmations", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = list_images(args.sequence_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {args.sequence_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {image_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    sequence_buffer = deque(maxlen=args.window)
    previous_keypoints = None
    latest_prob = 0.0
    confirmation_count = 0
    alert = False
    prob_trace = []

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for frame_number, image_path in enumerate(image_paths, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                mp_keypoints = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                ctr_keypoints = mediapipe_to_ntu25(mp_keypoints, previous_keypoints)
                previous_keypoints = ctr_keypoints.copy()
                sequence_buffer.append(ctr_keypoints)
                draw_skeleton(frame, ctr_keypoints)
            else:
                sequence_buffer.append(np.zeros((25, 3), dtype=np.float32))

            should_predict = len(sequence_buffer) == args.window and frame_number % args.predict_every == 0
            if should_predict:
                latest_prob = predict(model, sequence_buffer, device)
                prob_trace.append(latest_prob)
                if latest_prob >= args.threshold:
                    confirmation_count += 1
                else:
                    confirmation_count = 0
                alert = confirmation_count >= args.confirmations
            elif len(sequence_buffer) < args.window:
                prob_trace.append(latest_prob)

            if len(sequence_buffer) < args.window:
                decision = "warming up"
            elif alert:
                decision = "FALL ALERT"
            elif latest_prob >= args.threshold:
                decision = "fall candidate"
            else:
                decision = "non_fall"

            state = {
                "frame_idx": frame_number,
                "total_frames": len(image_paths),
                "buffer_len": len(sequence_buffer),
                "warmup_left": max(args.window - len(sequence_buffer), 0) / args.fps,
                "latest_prob": latest_prob,
                "threshold": args.threshold,
                "confirmations": confirmation_count,
                "required_confirmations": args.confirmations,
                "alert": alert,
                "decision": decision,
            }
            draw_panel(frame, state)
            draw_probability_trace(frame, prob_trace, args.threshold)
            writer.write(frame)

    writer.release()
    print(f"Rendered {len(image_paths)} frames at {args.fps} FPS")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
