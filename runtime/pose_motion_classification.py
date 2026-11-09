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

RUNTIME_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_ROOT))

from ctrgcn_model import Model
import graph.ntu_rgb_d
from mediapipe_functions import mediapipe_to_ntu25


TARGET_T = 72
CLASS_MAP = {0: "non_fall", 1: "fall"}
EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 12), (12, 13), (13, 14),
    (0, 16), (16, 17), (17, 18), (2, 4), (4, 5), (5, 6),
    (2, 8), (8, 9), (9, 10),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MediaPipe pose estimation and CTR-GCN motion classification."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--weights",
        default="model_weights/fall_motion_ctrgcn_72f_impact.pt",
        help="CTR-GCN model checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default="runtime_outputs/pose_motion_classification.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the video while processing. Press Esc to stop.",
    )
    return parser.parse_args()


def preprocess_window(sequence):
    x = np.array(sequence, dtype=np.float32)
    mask = np.any(np.abs(x) > 1e-8, axis=(1, 2))
    x = x[mask]

    if len(x) == 0:
        return None

    root = x[:, 0, :]
    x = x - root[:, None, :]

    bones = [(0, 1), (1, 2), (2, 3), (0, 12), (0, 16)]
    lengths = [np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in bones]
    lengths = np.stack(lengths, axis=1)
    scale = np.median(lengths[lengths > 1e-6]) if np.any(lengths > 1e-6) else 1.0
    x = x / scale

    if len(x) < TARGET_T:
        pad = np.repeat(x[-1][None, ...], TARGET_T - len(x), axis=0)
        x = np.concatenate([x, pad], axis=0)
    elif len(x) > TARGET_T:
        idx = np.linspace(0, len(x) - 1, TARGET_T).astype(int)
        x = x[idx]

    first_person = np.transpose(x, (2, 0, 1))[:, :, :, None]
    empty_second_person = np.zeros_like(first_person)
    two_person_input = np.concatenate([first_person, empty_second_person], axis=3)
    return two_person_input.astype(np.float32)


def draw_skeleton(frame, keypoints):
    height, width = frame.shape[:2]

    for i, j in EDGES:
        pt1 = tuple((keypoints[i, :2] * [width, height]).astype(int))
        pt2 = tuple((keypoints[j, :2] * [width, height]).astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    for point in keypoints:
        pt = tuple((point[:2] * [width, height]).astype(int))
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)


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


def main():
    args = parse_args()
    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(weights_path), device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    sequence = deque(maxlen=TARGET_T)
    previous_keypoints = None
    prediction = "warming_up"
    confidence = 0.0
    frame_count = 0

    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                mp_keypoints = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                ctr_keypoints = mediapipe_to_ntu25(mp_keypoints, previous_keypoints)
                previous_keypoints = ctr_keypoints.copy()
                sequence.append(ctr_keypoints)
                draw_skeleton(frame, ctr_keypoints)
            else:
                sequence.append(np.zeros((25, 3), dtype=np.float32))

            if len(sequence) >= TARGET_T:
                data = preprocess_window(sequence)
                if data is not None:
                    tensor = torch.tensor(data).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(tensor)
                        probabilities = torch.softmax(logits, dim=1)
                        pred = torch.argmax(probabilities, dim=1).item()
                    prediction = CLASS_MAP[pred]
                    confidence = probabilities[0][pred].item()

            color = (0, 255, 0) if prediction == "non_fall" else (0, 0, 255)
            cv2.putText(
                frame,
                f"{prediction} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_count += 1

            if args.show:
                cv2.imshow("Pose Estimation and Motion Classification", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("Pose estimation and motion classification complete.")
    print(f"Frames processed: {frame_count}")
    print(f"Last prediction: {prediction} ({confidence:.2f})")
    print(f"Output video: {output_path}")


if __name__ == "__main__":
    main()
