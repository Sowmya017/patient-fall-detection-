import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

SRC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))

from role_classifier import MobileNetRoleClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run patient/person detection and tracking on a video."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument(
        "--weights",
        default="model_weights/patient_detection_yolov8n.pt",
        help="YOLO weights path.",
    )
    parser.add_argument(
        "--output",
        default="runtime_outputs/patient_detection_tracking.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--role-weights",
        default="model_weights/role_classification_mobilenetv3_best.pt",
        help="MobileNet role-classification checkpoint path.",
    )
    parser.add_argument(
        "--disable-role-classifier",
        action="store_true",
        help="Only run person detection/tracking without role classification.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the video while processing. Press Esc to stop.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    role_classifier = None
    if not args.disable_role_classifier:
        role_classifier = MobileNetRoleClassifier(args.role_weights)

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

    frame_count = 0
    tracked_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]

        annotated = frame.copy()
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else [None] * len(boxes)
            )

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int)
                tracked_count += 1
                role_label = "person"
                role_conf = 0.0
                if role_classifier is not None:
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    role_label, role_conf = role_classifier.predict_crop(crop)

                id_text = f"ID {track_id}" if track_id is not None else "ID ?"
                label = f"{id_text} | {role_label} ({role_conf:.2f})"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 0),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(annotated)
        frame_count += 1

        if args.show:
            cv2.imshow("Patient Detection and Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print("Detection and tracking complete.")
    print(f"Frames processed: {frame_count}")
    print(f"Tracked detections drawn: {tracked_count}")
    print(f"Output video: {output_path}")


if __name__ == "__main__":
    main()
