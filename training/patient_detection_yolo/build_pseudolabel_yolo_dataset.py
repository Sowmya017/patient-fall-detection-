import os
import cv2
import yaml
import math
import random
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def list_videos(videos_dir):
    vids = []
    for root, _, files in os.walk(videos_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                vids.append(os.path.join(root, f))
    vids.sort()
    return vids


def xyxy_to_yolo_norm(x1, y1, x2, y2, w, h):
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def main(config_path="C:/GP/YOLO_Tracker/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    detector_cfg = cfg["detector"]
    filter_cfg = cfg["filtering"]

    random.seed(dataset_cfg["seed"])

    videos = list_videos(dataset_cfg["videos_dir"])
    print(f"Found {len(videos)} videos")

    # Split by video
    idx = list(range(len(videos)))
    random.shuffle(idx)
    n_val = max(1, int(len(videos) * dataset_cfg["val_ratio"]))
    val_set = set(idx[:n_val])

    # Output dirs
    img_train = os.path.join(dataset_cfg["out_root"], "images/train")
    img_val   = os.path.join(dataset_cfg["out_root"], "images/val")
    lab_train = os.path.join(dataset_cfg["out_root"], "labels/train")
    lab_val   = os.path.join(dataset_cfg["out_root"], "labels/val")

    for d in [img_train, img_val, lab_train, lab_val]:
        ensure_dir(d)

    # Write YAML dataset file
    dataset_yaml = {
        "path": dataset_cfg["out_root"],
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"}
    }
    with open(os.path.join(dataset_cfg["out_root"], "person.yaml"), "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    model = YOLO(detector_cfg["weights"])

    for i, vid_path in enumerate(videos):
        split = "val" if i in val_set else "train"
        img_dir = img_val if split == "val" else img_train
        lab_dir = lab_val if split == "val" else lab_train

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / dataset_cfg["fps"])))

        frame_idx = 0
        vid_stem = os.path.splitext(os.path.basename(vid_path))[0]

        pbar = tqdm(desc=f"{split} | {vid_stem}", total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            h, w = frame.shape[:2]
            res = model.predict(
                source=frame,
                imgsz=detector_cfg["imgsz"],
                conf=detector_cfg["conf"],
                iou=detector_cfg["iou"],
                max_det=detector_cfg["max_det"],
                verbose=False
            )[0]

            img_name = f"{vid_stem}_{frame_idx:07d}.jpg"
            lab_name = f"{vid_stem}_{frame_idx:07d}.txt"

            cv2.imwrite(os.path.join(img_dir, img_name), frame)

            lines = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()

                for bb, c, cf in zip(xyxy, cls, confs):
                    if c != 0 or cf < detector_cfg["conf"]:
                        continue

                    area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    if area < filter_cfg["min_box_area"]:
                        continue

                    cx, cy, bw, bh = xyxy_to_yolo_norm(*bb, w, h)
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(os.path.join(lab_dir, lab_name), "w") as f:
                f.write("\n".join(lines))

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

    print("Dataset building complete.")


if __name__ == "__main__":
    main()
