import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from mediapipe_functions import mediapipe_to_ntu25
from ctrgcn_model import Model
import graph.ntu_rgb_d


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_NAMES = ["non_fall", "fall"]


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def find_image_sequence_dirs(root):
    root = Path(root)
    sequence_dirs = []
    for current, _, files in os.walk(root):
        if any(Path(name).suffix.lower() in IMAGE_EXTENSIONS for name in files):
            sequence_dirs.append(Path(current))
    return sorted(sequence_dirs, key=lambda p: natural_key(p.as_posix()))


def list_images(sequence_dir, frame_stride=1, max_frames=0):
    images = [
        path
        for path in Path(sequence_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images = sorted(images, key=lambda p: natural_key(p.name))
    if frame_stride > 1:
        images = images[::frame_stride]
    if max_frames and len(images) > max_frames:
        indices = np.linspace(0, len(images) - 1, max_frames).astype(int)
        images = [images[i] for i in indices]
    return images


def sample_id_from_path(value):
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def load_model(weights_path, device):
    model = Model(
        num_class=2,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def extract_skeleton_sequence(image_paths, pose):
    sequence = []
    previous_keypoints = None
    detected_frames = 0

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            sequence.append(np.zeros((25, 3), dtype=np.float32))
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
            sequence.append(ctr_keypoints)
            detected_frames += 1
        else:
            sequence.append(np.zeros((25, 3), dtype=np.float32))

    return np.array(sequence, dtype=np.float32), detected_frames


def load_or_extract_skeletons(sequence_dir, image_paths, pose, cache_dir, frame_stride, max_frames):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sample_id_from_path(
        f"{Path(sequence_dir).resolve()}|stride={frame_stride}|max={max_frames}|mp25"
    )
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["sequence"].astype(np.float32), int(cached["detected_frames"]), True

    compatible_cache_path = cache_dir / f"{sample_id_from_path(f'{sequence_dir}|stride={frame_stride}|max={max_frames}')}.npz"
    if compatible_cache_path.exists():
        cached = np.load(compatible_cache_path, allow_pickle=True)
        return cached["sequence"].astype(np.float32), int(cached["detected_frames"]), True

    sequence, detected_frames = extract_skeleton_sequence(image_paths, pose)
    np.savez_compressed(
        cache_path,
        sequence=sequence,
        detected_frames=np.array(detected_frames, dtype=np.int32),
        source=str(sequence_dir),
    )
    return sequence, detected_frames, False


def preprocess_72_window(window):
    x = np.array(window, dtype=np.float32)
    if x.shape[0] != 72:
        raise ValueError(f"Expected 72 frames, got {x.shape[0]}")

    valid = np.any(np.abs(x) > 1e-8, axis=(1, 2))
    if not np.any(valid):
        return None

    root = x[:, 0, :]
    x = x - root[:, None, :]
    bones = [(0, 1), (1, 2), (2, 3), (0, 12), (0, 16)]
    lengths = [np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in bones]
    lengths = np.stack(lengths, axis=1)
    scale = np.median(lengths[lengths > 1e-6]) if np.any(lengths > 1e-6) else 1.0
    x = x / scale

    first_person = np.transpose(x, (2, 0, 1))[:, :, :, None]
    empty_second_person = np.zeros_like(first_person)
    return np.concatenate([first_person, empty_second_person], axis=3).astype(np.float32)


def window_starts(num_frames, window_size, window_stride):
    if num_frames <= 0:
        return []
    if num_frames <= window_size:
        return [0]
    starts = list(range(0, num_frames - window_size + 1, window_stride))
    last_start = num_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.no_grad()
def predict_sliding_windows(model, sequence, device, window_size=72, window_stride=18, batch_size=64):
    valid_frames = int(np.any(np.abs(sequence) > 1e-8, axis=(1, 2)).sum())
    if len(sequence) == 0 or valid_frames == 0:
        return empty_prediction()

    starts = window_starts(len(sequence), window_size, window_stride)
    batches = []
    kept_starts = []
    for start in starts:
        window = sequence[start:start + window_size]
        if len(window) < window_size:
            pad = np.repeat(window[-1][None, ...], window_size - len(window), axis=0)
            window = np.concatenate([window, pad], axis=0)
        data = preprocess_72_window(window)
        if data is not None:
            batches.append(data)
            kept_starts.append(start)

    if not batches:
        return empty_prediction()

    fall_probs = []
    for i in range(0, len(batches), batch_size):
        tensor = torch.tensor(np.stack(batches[i:i + batch_size]), dtype=torch.float32).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        fall_probs.extend(float(value) for value in probs)

    best_idx = int(np.argmax(fall_probs))
    max_prob = float(np.max(fall_probs))
    mean_prob = float(np.mean(fall_probs))
    final_prob = float(fall_probs[-1])
    pred_index = int(max_prob >= 0.5)
    return {
        "pred_label": CLASS_NAMES[pred_index],
        "pred_index": pred_index,
        "fall_probability_max": max_prob,
        "fall_probability_mean": mean_prob,
        "fall_probability_final": final_prob,
        "best_window_start": int(kept_starts[best_idx]),
        "best_window_end": int(kept_starts[best_idx] + window_size - 1),
        "num_windows": len(fall_probs),
    }


def empty_prediction():
    return {
        "pred_label": "no_pose",
        "pred_index": -1,
        "fall_probability_max": 0.0,
        "fall_probability_mean": 0.0,
        "fall_probability_final": 0.0,
        "best_window_start": -1,
        "best_window_end": -1,
        "num_windows": 0,
    }


def collect_samples(nonfall_roots, fall_roots):
    samples = []
    for root in nonfall_roots:
        for sequence_dir in find_image_sequence_dirs(root):
            samples.append({"path": sequence_dir, "true_label": "non_fall", "true_index": 0})
    for root in fall_roots:
        for sequence_dir in find_image_sequence_dirs(root):
            samples.append({"path": sequence_dir, "true_label": "fall", "true_index": 1})
    return samples


def compute_metrics(results, elapsed_seconds):
    valid = results[results["pred_index"] >= 0].copy()
    y_true = valid["true_index"].astype(int).to_numpy()
    y_pred = valid["pred_index"].astype(int).to_numpy()
    y_score = valid["fall_probability_max"].astype(float).to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    try:
        roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        roc_auc = None

    return {
        "num_samples": int(len(results)),
        "num_valid_samples": int(len(valid)),
        "num_no_pose_samples": int((results["pred_index"] < 0).sum()),
        "support_non_fall": int(support[0]),
        "support_fall": int(support[1]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_non_fall": float(precision[0]),
        "recall_non_fall": float(recall[0]),
        "f1_non_fall": float(f1[0]),
        "precision_fall": float(precision[1]),
        "recall_fall": float(recall[1]),
        "f1_fall": float(f1[1]),
        "macro_f1": float(macro_f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def write_report(metrics, args, out_dir):
    lines = [
        "# 72-Frame Sliding-Window Fall Evaluation",
        "",
        "## Setup",
        "",
        f"- Weights: `{args.weights}`",
        f"- Non-fall roots: `{'; '.join(args.nonfall_roots)}`",
        f"- Fall roots: `{'; '.join(args.fall_roots)}`",
        f"- Window size: {args.window_size} frames",
        f"- Window stride: {args.window_stride} frames",
        "- Aggregation: video/sample is fall if any window has fall probability >= 0.50",
        "",
        "## Metrics",
        "",
        f"- Evaluated samples: {metrics['num_samples']}",
        f"- Valid samples: {metrics['num_valid_samples']}",
        f"- No-pose samples: {metrics['num_no_pose_samples']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Fall precision: {metrics['precision_fall']:.4f}",
        f"- Fall recall: {metrics['recall_fall']:.4f}",
        f"- Fall F1: {metrics['f1_fall']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- ROC AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'n/a'}",
        "",
        "## Confusion Matrix",
        "",
        "| True \\ Predicted | non_fall | fall |",
        "|---|---:|---:|",
        f"| non_fall | {metrics['confusion_matrix'][0][0]} | {metrics['confusion_matrix'][0][1]} |",
        f"| fall | {metrics['confusion_matrix'][1][0]} | {metrics['confusion_matrix'][1][1]} |",
        "",
        "## Files",
        "",
        "- `per_sample_results.csv`",
        "- `metrics.json`",
    ]
    report_path = out_dir / "sliding_window_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run 72-frame sliding-window CTR-GCN evaluation.")
    parser.add_argument(
        "--nonfall-roots",
        nargs="+",
        default=[r"F:\GP_Dataset\activities2", r"F:\GP_Dataset\activities6-11"],
    )
    parser.add_argument("--fall-roots", nargs="+", default=[r"F:\GP_Dataset\Subjects"])
    parser.add_argument("--weights", default="model_weights/fall_motion_ctrgcn_72f_impact.pt")
    parser.add_argument("--out-dir", default="evaluation/results/fall_motion_ctrgcn_72f_sliding_window")
    parser.add_argument("--cache-dir", default="runtime_outputs/evaluation_cache/fall_motion_ctrgcn_72f_sliding_window")
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--window-stride", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    partial_path = out_dir / "per_sample_results.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    samples = collect_samples(args.nonfall_roots, args.fall_roots)
    if args.limit:
        samples = samples[:args.limit]

    rows = []
    processed_paths = set()
    if partial_path.exists():
        existing = pd.read_csv(partial_path)
        rows = existing.to_dict("records")
        processed_paths = set(existing["path"].astype(str))
        print(f"Resuming from {len(rows)} completed samples.", flush=True)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for index, sample in enumerate(samples, 1):
            if str(sample["path"]) in processed_paths:
                continue

            image_paths = list_images(sample["path"], args.frame_stride, args.max_frames)
            sequence, detected_frames, cache_hit = load_or_extract_skeletons(
                sample["path"], image_paths, pose, cache_dir, args.frame_stride, args.max_frames
            )
            prediction = predict_sliding_windows(
                model,
                sequence,
                device,
                window_size=args.window_size,
                window_stride=args.window_stride,
                batch_size=args.batch_size,
            )
            rows.append(
                {
                    "sample_index": index,
                    "path": str(sample["path"]),
                    "true_label": sample["true_label"],
                    "true_index": sample["true_index"],
                    **prediction,
                    "num_frames": len(image_paths),
                    "num_pose_frames": detected_frames,
                    "pose_detection_rate": detected_frames / max(len(image_paths), 1),
                    "cache_hit": cache_hit,
                }
            )
            row = rows[-1]
            pd.DataFrame([row]).to_csv(
                partial_path,
                mode="a",
                index=False,
                header=not partial_path.exists(),
            )
            print(
                f"[{index}/{len(samples)}] {sample['true_label']} -> {prediction['pred_label']} "
                f"max_p_fall={prediction['fall_probability_max']:.3f} "
                f"best={prediction['best_window_start']}-{prediction['best_window_end']} "
                f"windows={prediction['num_windows']} poses={detected_frames}/{len(image_paths)}"
                f" cache={cache_hit}",
                flush=True,
            )

    results = pd.read_csv(partial_path)
    elapsed = time.time() - start_time
    metrics = compute_metrics(results, elapsed)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path = write_report(metrics, args, out_dir)
    print(f"Saved report: {report_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
