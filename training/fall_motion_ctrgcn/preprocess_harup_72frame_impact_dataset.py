import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from mediapipe_functions import mediapipe_to_ntu25


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
FALL_ACTIVITIES = {1, 2, 3, 4, 5}
NON_FALL_ACTIVITIES = {6, 7, 8, 9, 10, 11}
ACCEL_COLS = {
    "ankle": [1, 2, 3],
    "right_pocket": [8, 9, 10],
    "belt": [15, 16, 17],
    "neck": [22, 23, 24],
    "wrist": [29, 30, 31],
}


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def parse_sequence_metadata(sequence_dir):
    name = Path(sequence_dir).name
    match = re.search(r"Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)", name)
    if not match:
        return None
    return {
        "subject": int(match.group(1)),
        "activity": int(match.group(2)),
        "trial": int(match.group(3)),
        "camera": int(match.group(4)),
    }


def parse_frame_timestamp(path):
    # Frame names use 2018-07-04T12_04_17.738369.png.
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}(?:\.\d+)?", Path(path).stem)
    if not match:
        raise ValueError(f"Could not parse timestamp from frame name: {path}")
    stamp = match.group(0).replace("_", ":", 2)
    return pd.to_datetime(stamp)


def find_sequence_dirs(roots):
    sequence_dirs = []
    seen = set()
    for root in roots:
        for current, _, files in os.walk(root):
            if any(Path(name).suffix.lower() in IMAGE_EXTENSIONS for name in files):
                path = Path(current)
                key = str(path.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    sequence_dirs.append(path)
    return sorted(sequence_dirs, key=lambda p: natural_key(p.as_posix()))


def list_frames(sequence_dir):
    frames = [
        path
        for path in Path(sequence_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(frames, key=lambda p: natural_key(p.name))


def load_sensor_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=[1])
    df["TimeStamps"] = pd.to_datetime(df["TimeStamps"])
    return df


def acceleration_magnitude(trial_df, sensor):
    values = trial_df.iloc[:, ACCEL_COLS[sensor]].astype(float).to_numpy()
    return np.linalg.norm(values, axis=1)


def select_center_time(trial_df, activity, sensor):
    tag_mask = trial_df["Tag"].to_numpy() == activity
    if activity in FALL_ACTIVITIES and np.any(tag_mask):
        mag = acceleration_magnitude(trial_df, sensor)
        impact_index = int(np.argmax(np.where(tag_mask, mag, -np.inf)))
        return {
            "center_time": trial_df["TimeStamps"].iloc[impact_index],
            "center_index": impact_index,
            "center_type": f"impact_peak_{sensor}",
            "center_magnitude": float(mag[impact_index]),
            "tag_start_time": trial_df.loc[tag_mask, "TimeStamps"].iloc[0],
            "tag_end_time": trial_df.loc[tag_mask, "TimeStamps"].iloc[-1],
        }

    if np.any(tag_mask):
        tag_times = trial_df.loc[tag_mask, "TimeStamps"].reset_index(drop=True)
        center_time = tag_times.iloc[0] + (tag_times.iloc[-1] - tag_times.iloc[0]) / 2
        center_index = int(np.argmin(np.abs((trial_df["TimeStamps"] - center_time).dt.total_seconds())))
        return {
            "center_time": center_time,
            "center_index": center_index,
            "center_type": "activity_tag_midpoint",
            "center_magnitude": None,
            "tag_start_time": tag_times.iloc[0],
            "tag_end_time": tag_times.iloc[-1],
        }

    center_index = len(trial_df) // 2
    return {
        "center_time": trial_df["TimeStamps"].iloc[center_index],
        "center_index": center_index,
        "center_type": "trial_midpoint_no_matching_tag",
        "center_magnitude": None,
        "tag_start_time": pd.NaT,
        "tag_end_time": pd.NaT,
    }


def choose_frames_at_18fps(frames, center_time, clip_frames, fps):
    frame_times = pd.DatetimeIndex([parse_frame_timestamp(path) for path in frames])
    start_time = center_time - pd.Timedelta(seconds=clip_frames / fps / 2)
    target_times = [start_time + pd.Timedelta(seconds=i / fps) for i in range(clip_frames)]

    selected = []
    selected_indices = []
    for target_time in target_times:
        deltas = np.abs((frame_times - target_time).total_seconds())
        idx = int(np.argmin(deltas))
        selected.append(frames[idx])
        selected_indices.append(idx)

    return selected, selected_indices, target_times


def preprocess_ctrgcn_input(sequence):
    x = np.array(sequence, dtype=np.float32)
    mask = np.any(np.abs(x) > 1e-8, axis=(1, 2))
    valid = x[mask]
    if len(valid) == 0:
        first_person = np.zeros((3, len(x), 25, 1), dtype=np.float32)
        return np.concatenate([first_person, first_person], axis=3)

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


def extract_skeletons(frame_paths, pose):
    sequence = []
    previous_keypoints = None
    pose_frames = 0

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
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
            pose_frames += 1
        else:
            sequence.append(np.zeros((25, 3), dtype=np.float32))

    return np.array(sequence, dtype=np.float32), pose_frames


def safe_timestamp(value):
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def save_sample(out_path, keypoints, ctrgcn_input, selected_frames, selected_indices, target_times, metadata):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        ctrgcn_input=ctrgcn_input,
        label=np.array(metadata["label_index"], dtype=np.int64),
        label_name=np.array(metadata["label_name"]),
        subject=np.array(metadata["subject"], dtype=np.int64),
        activity=np.array(metadata["activity"], dtype=np.int64),
        trial=np.array(metadata["trial"], dtype=np.int64),
        camera=np.array(metadata["camera"], dtype=np.int64),
        selected_frame_paths=np.array([str(path) for path in selected_frames]),
        selected_frame_indices=np.array(selected_indices, dtype=np.int64),
        target_times=np.array([pd.Timestamp(t).isoformat() for t in target_times]),
    )


def sequence_slug(sequence_dir):
    return re.sub(r"[^a-zA-Z0-9]+", "_", Path(sequence_dir).name).strip("_").lower()


def process_dataset(args):
    out_root = Path(args.out_root)
    clips_root = out_root / "clips"
    out_root.mkdir(parents=True, exist_ok=True)

    sensor_df = load_sensor_csv(args.csv)
    grouped = {key: group.reset_index(drop=True) for key, group in sensor_df.groupby(["Subject", "Activity", "Trial"])}
    sequence_dirs = find_sequence_dirs(args.fall_roots + args.nonfall_roots)
    if args.limit:
        sequence_dirs = sequence_dirs[:args.limit]

    rows = []
    skipped = []
    start = time.time()

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:
        for idx, sequence_dir in enumerate(sequence_dirs, start=1):
            meta = parse_sequence_metadata(sequence_dir)
            if meta is None:
                skipped.append({"path": str(sequence_dir), "reason": "could_not_parse_metadata"})
                continue

            activity = meta["activity"]
            if activity in FALL_ACTIVITIES:
                label_name = "fall"
                label_index = 1
            elif activity in NON_FALL_ACTIVITIES:
                label_name = "non_fall"
                label_index = 0
            else:
                skipped.append({"path": str(sequence_dir), "reason": f"unsupported_activity_{activity}"})
                continue

            trial_key = (meta["subject"], meta["activity"], meta["trial"])
            trial_df = grouped.get(trial_key)
            if trial_df is None:
                skipped.append({"path": str(sequence_dir), "reason": "no_sensor_rows"})
                continue

            frames = list_frames(sequence_dir)
            if not frames:
                skipped.append({"path": str(sequence_dir), "reason": "no_frames"})
                continue

            center = select_center_time(trial_df, activity, args.impact_sensor)
            selected_frames, selected_indices, target_times = choose_frames_at_18fps(
                frames,
                center["center_time"],
                args.clip_frames,
                args.fps,
            )
            keypoints, pose_frames = extract_skeletons(selected_frames, pose)
            ctrgcn_input = preprocess_ctrgcn_input(keypoints)

            sample_id = f"{sequence_slug(sequence_dir)}_{label_name}"
            out_path = clips_root / label_name / f"{sample_id}.npz"
            save_sample(
                out_path,
                keypoints,
                ctrgcn_input,
                selected_frames,
                selected_indices,
                target_times,
                {
                    **meta,
                    "label_name": label_name,
                    "label_index": label_index,
                },
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "label": label_name,
                    "label_index": label_index,
                    "subject": meta["subject"],
                    "activity": meta["activity"],
                    "trial": meta["trial"],
                    "camera": meta["camera"],
                    "source_dir": str(sequence_dir),
                    "clip_path": str(out_path),
                    "center_type": center["center_type"],
                    "center_time": safe_timestamp(center["center_time"]),
                    "center_sensor_row": center["center_index"],
                    "center_magnitude": center["center_magnitude"],
                    "tag_start_time": safe_timestamp(center["tag_start_time"]),
                    "tag_end_time": safe_timestamp(center["tag_end_time"]),
                    "window_start_time": safe_timestamp(target_times[0]),
                    "window_end_time": safe_timestamp(target_times[-1]),
                    "selected_source_frame_start": selected_indices[0],
                    "selected_source_frame_end": selected_indices[-1],
                    "unique_source_frames": len(set(selected_indices)),
                    "clip_frames": args.clip_frames,
                    "fps": args.fps,
                    "pose_frames": pose_frames,
                    "pose_detection_rate": pose_frames / args.clip_frames,
                    "ctrgcn_shape": "x".join(map(str, ctrgcn_input.shape)),
                }
            )

            print(
                f"[{idx}/{len(sequence_dirs)}] {sample_id} "
                f"center={center['center_type']} poses={pose_frames}/{args.clip_frames}"
            )

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "csv": args.csv,
        "fall_roots": args.fall_roots,
        "nonfall_roots": args.nonfall_roots,
        "out_root": str(out_root),
        "fps": args.fps,
        "clip_frames": args.clip_frames,
        "seconds_before_center": args.clip_frames / args.fps / 2,
        "seconds_after_center": args.clip_frames / args.fps / 2,
        "impact_sensor": args.impact_sensor,
        "num_samples": int(len(manifest)),
        "num_fall": int((manifest["label"] == "fall").sum()) if not manifest.empty else 0,
        "num_non_fall": int((manifest["label"] == "non_fall").sum()) if not manifest.empty else 0,
        "skipped": skipped,
        "elapsed_seconds": time.time() - start,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved summary: {out_root / 'summary.json'}")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create 72-frame 18 FPS impact-centered skeleton clips.")
    parser.add_argument("--csv", default=r"C:\Users\Victus\Downloads\CompleteDataSet (1).csv")
    parser.add_argument("--fall-roots", nargs="+", default=[r"F:\GP_Dataset\Subjects"])
    parser.add_argument(
        "--nonfall-roots",
        nargs="+",
        default=[r"F:\GP_Dataset\activities2", r"F:\GP_Dataset\activities6-11"],
    )
    parser.add_argument("--out-root", default=r"F:\GP_Dataset\Fall72_18fps_ImpactCentered")
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--clip-frames", type=int, default=72)
    parser.add_argument("--impact-sensor", choices=sorted(ACCEL_COLS), default="wrist")
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Optional small-run limit for testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
