import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO person detection on a video stream.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--weights", default="model_weights/patient_detection_yolov8n.pt")
    parser.add_argument("--out-dir", default="evaluation/results/patient_detection_yolo")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=15)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(
            str(out_dir / "patient_detection_yolo_annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    total_frames = 0
    no_detection_frames = 0
    person_counts = []
    confidences = []

    with tqdm(desc="frames") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False,
            )[0]

            person_count = 0
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                for box, class_id, score in zip(boxes, classes, scores):
                    if class_id != 0:
                        continue
                    person_count += 1
                    confidences.append(float(score))
                    if writer is not None:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                        cv2.putText(
                            frame,
                            f"person {score:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 180, 0),
                            2,
                            cv2.LINE_AA,
                        )

            if person_count == 0:
                no_detection_frames += 1
            person_counts.append(person_count)
            total_frames += 1

            if writer is not None:
                writer.write(frame)
            progress.update(1)

    cap.release()
    if writer is not None:
        writer.release()

    metrics = {
        "video": str(args.video),
        "weights": str(args.weights),
        "frames": total_frames,
        "no_detection_frames": no_detection_frames,
        "no_detection_rate": no_detection_frames / max(total_frames, 1),
        "mean_person_count_per_frame": float(np.mean(person_counts)) if person_counts else 0.0,
        "median_person_count_per_frame": float(np.median(person_counts)) if person_counts else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "median_confidence": float(np.median(confidences)) if confidences else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
