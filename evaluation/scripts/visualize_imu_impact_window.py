import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
ACCEL_COLS = {
    "ankle": [1, 2, 3],
    "right_pocket": [8, 9, 10],
    "belt": [15, 16, 17],
    "neck": [22, 23, 24],
    "wrist": [29, 30, 31],
}


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def parse_frame_timestamp(path):
    stamp = Path(path).stem.replace("_", ":", 2)
    return pd.to_datetime(stamp)


def list_frames(frame_dir):
    frames = [
        path
        for path in Path(frame_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(frames, key=lambda p: natural_key(p.name))


def load_trial(csv_path, subject, activity, trial):
    df = pd.read_csv(csv_path, skiprows=[1])
    df = df[
        (df["Subject"] == subject)
        & (df["Activity"] == activity)
        & (df["Trial"] == trial)
    ].copy()
    if df.empty:
        raise ValueError(f"No rows for Subject={subject}, Activity={activity}, Trial={trial}")
    df["TimeStamps"] = pd.to_datetime(df["TimeStamps"])
    df = df.reset_index(drop=True)
    return df


def compute_acc_magnitude(df, sensor):
    values = df.iloc[:, ACCEL_COLS[sensor]].astype(float).to_numpy()
    return np.linalg.norm(values, axis=1)


def compute_window_indices(frame_times, impact_time, window_size):
    deltas = np.abs((frame_times - impact_time).total_seconds())
    impact_frame_idx = int(np.argmin(deltas))
    start_idx = max(0, impact_frame_idx - window_size // 2)
    end_idx = start_idx + window_size - 1
    if end_idx >= len(frame_times):
        end_idx = len(frame_times) - 1
        start_idx = max(0, end_idx - window_size + 1)
    return impact_frame_idx, start_idx, end_idx


def draw_text(frame, text, xy, scale=0.58, color=(245, 245, 245), thickness=1):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame, state, chart):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 170), (18, 18, 18), -1)
    cv2.rectangle(overlay, (0, h - 170), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    draw_text(frame, "CSV-guided fall window localization", (18, 28), 0.75, (255, 255, 255), 2)
    draw_text(frame, f"Trial: Subject {state['subject']} | Activity {state['activity']} | Trial {state['trial']} | Camera frames: {state['num_frames']}", (18, 58))
    draw_text(frame, f"Fall interval from CSV: Tag == Activity ({state['fall_start']} -> {state['fall_end']})", (18, 86), color=(0, 220, 255))
    draw_text(frame, f"Impact estimate: max {state['sensor']} acceleration magnitude inside fall interval at {state['impact_time']} | magnitude={state['impact_mag']:.2f}g", (18, 114), color=(0, 170, 255))
    draw_text(frame, f"Selected 72-frame clip: frames {state['window_start'] + 1}-{state['window_end'] + 1}, centered near impact frame {state['impact_frame'] + 1}", (18, 142), color=(120, 255, 120))

    chart_x, chart_y, chart_w, chart_h = 26, h - 135, w - 52, 92
    cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (140, 140, 140), 1)

    n = len(chart["mag"])
    mag = chart["mag"]
    mag_min = float(np.nanmin(mag))
    mag_max = float(np.nanmax(mag))
    denom = max(mag_max - mag_min, 1e-6)

    def x_at(index):
        return int(chart_x + (index / max(n - 1, 1)) * chart_w)

    def y_at(value):
        return int(chart_y + chart_h - ((value - mag_min) / denom) * chart_h)

    # 72-frame selected window.
    win_x1 = x_at(state["window_start"])
    win_x2 = x_at(state["window_end"])
    cv2.rectangle(frame, (win_x1, chart_y), (win_x2, chart_y + chart_h), (40, 90, 40), -1)

    # CSV fall interval.
    fall_x1 = x_at(chart["fall_start_idx"])
    fall_x2 = x_at(chart["fall_end_idx"])
    cv2.rectangle(frame, (fall_x1, chart_y), (fall_x2, chart_y + chart_h), (0, 120, 160), 2)

    points = [(x_at(i), y_at(v)) for i, v in enumerate(mag)]
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(frame, p1, p2, (255, 255, 255), 1)

    impact_x = x_at(state["impact_frame"])
    cv2.line(frame, (impact_x, chart_y), (impact_x, chart_y + chart_h), (0, 0, 255), 2)

    current_x = x_at(state["frame_idx"])
    cv2.line(frame, (current_x, chart_y - 8), (current_x, chart_y + chart_h + 8), (255, 0, 255), 2)

    draw_text(frame, "white: acceleration magnitude | cyan box: Tag fall interval | green band: chosen 72 frames | red: impact | magenta: current frame", (chart_x, h - 18), 0.48)

    if state["window_start"] <= state["frame_idx"] <= state["window_end"]:
        badge = "INSIDE SELECTED 72-FRAME TRAINING/EVAL CLIP"
        color = (80, 255, 80)
    else:
        badge = "OUTSIDE SELECTED 72-FRAME CLIP"
        color = (180, 180, 180)
    draw_text(frame, badge, (18, 165), 0.54, color, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Render CSV Tag + IMU impact fall-window visualization.")
    parser.add_argument("--csv", default=r"C:\Users\Victus\Downloads\CompleteDataSet (1).csv")
    parser.add_argument("--frames", default=r"F:\GP_Dataset\Subjects\Subject01\Activity01\Trial1\Subject1Activity1Trial1Camera1")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--activity", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--sensor", choices=sorted(ACCEL_COLS), default="wrist")
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--output", default="evaluation/results/runtime_visualizations/subject1_activity1_trial1_camera1.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    frames = list_frames(args.frames)
    if not frames:
        raise FileNotFoundError(f"No image frames found in {args.frames}")

    df = load_trial(args.csv, args.subject, args.activity, args.trial)
    mag = compute_acc_magnitude(df, args.sensor)
    fall_mask = df["Tag"].to_numpy() == args.activity
    if not np.any(fall_mask):
        raise ValueError(f"No Tag == Activity rows found for activity {args.activity}")

    fall_indices = np.where(fall_mask)[0]
    fall_start_idx = int(fall_indices[0])
    fall_end_idx = int(fall_indices[-1])
    masked_mag = np.where(fall_mask, mag, -np.inf)
    impact_sensor_idx = int(np.argmax(masked_mag))
    impact_time = df["TimeStamps"].iloc[impact_sensor_idx]

    frame_times = pd.DatetimeIndex([parse_frame_timestamp(path) for path in frames])
    impact_frame_idx, window_start, window_end = compute_window_indices(frame_times, impact_time, args.window_size)

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    chart = {
        "mag": mag,
        "fall_start_idx": fall_start_idx,
        "fall_end_idx": fall_end_idx,
    }

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        state = {
            "subject": args.subject,
            "activity": args.activity,
            "trial": args.trial,
            "num_frames": len(frames),
            "fall_start": df["TimeStamps"].iloc[fall_start_idx].strftime("%H:%M:%S.%f")[:-3],
            "fall_end": df["TimeStamps"].iloc[fall_end_idx].strftime("%H:%M:%S.%f")[:-3],
            "impact_time": impact_time.strftime("%H:%M:%S.%f")[:-3],
            "impact_mag": float(mag[impact_sensor_idx]),
            "sensor": args.sensor,
            "impact_frame": impact_frame_idx,
            "window_start": window_start,
            "window_end": window_end,
            "frame_idx": i,
        }
        draw_overlay(frame, state, chart)
        writer.write(frame)

    writer.release()
    print(f"Fall tag interval rows: {fall_start_idx}-{fall_end_idx}")
    print(f"Impact sensor row: {impact_sensor_idx}, timestamp: {impact_time}, mag: {mag[impact_sensor_idx]:.3f}")
    print(f"Impact frame: {impact_frame_idx + 1}")
    print(f"Selected 72-frame window: {window_start + 1}-{window_end + 1}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
