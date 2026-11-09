import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def main(config_path="training/patient_detection_yolo/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    detector_cfg = cfg["detector"]
    test_cfg = cfg["test"]

    model = YOLO(detector_cfg["weights"])

    cap = cv2.VideoCapture(test_cfg["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        test_cfg["out_video"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    zero_frames = 0
    total_frames = 0
    confs = []

    pbar = tqdm()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(
            source=frame,
            imgsz=test_cfg["imgsz"],
            conf=test_cfg["conf"],
            iou=test_cfg["iou"],
            max_det=test_cfg["max_det"],
            verbose=False
        )[0]

        det_count = 0

        if res.boxes is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()

            for bb, c, cfi in zip(xyxy, cls, cf):
                if c != 0:
                    continue
                det_count += 1
                confs.append(cfi)
                x1, y1, x2, y2 = bb.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if det_count == 0:
            zero_frames += 1

        total_frames += 1
        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    print("\n==== Test Summary ====")
    print(f"Frames: {total_frames}")
    print(f"No detection frames: {zero_frames} ({100*zero_frames/total_frames:.2f}%)")
    if confs:
        print(f"Mean confidence: {np.mean(confs):.3f}")
        print(f"Median confidence: {np.median(confs):.3f}")
    print("Output video:", test_cfg["out_video"])


if __name__ == "__main__":
    main()
