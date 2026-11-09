import os
import re
import cv2
import json
import yaml
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from JointsMapping import *

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_metadata_json(out_path: str, meta: dict):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def list_sample_folders(root_dir: str, image_extensions, min_images_per_sample=2, recursive=True):
    """
    A sample folder is any folder containing at least min_images_per_sample image files.
    """
    exts = {e.lower() for e in image_extensions}
    sample_dirs = []

    if recursive:
        for current_root, _, files in os.walk(root_dir):
            image_files = [
                f for f in files
                if os.path.splitext(f)[1].lower() in exts
            ]
            if len(image_files) >= min_images_per_sample:
                sample_dirs.append(current_root)
    else:
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if not os.path.isdir(full):
                continue
            files = os.listdir(full)
            image_files = [
                f for f in files
                if os.path.splitext(f)[1].lower() in exts
            ]
            if len(image_files) >= min_images_per_sample:
                sample_dirs.append(full)

    sample_dirs.sort(key=natural_key)
    return sample_dirs


def list_images_in_sample(sample_dir: str, image_extensions, sort_mode="natural"):
    exts = {e.lower() for e in image_extensions}
    image_paths = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if os.path.splitext(f)[1].lower() in exts
    ]

    if sort_mode == "natural":
        image_paths.sort(key=natural_key)
    else:
        image_paths.sort()

    return image_paths


def safe_rel_stem(path: str, root_dir: str) -> str:
    """
    Example:
      F:/HAR-UP/Subject1/Activity2/Trial1
    ->
      Subject1__Activity2__Trial1
    """
    rel = os.path.relpath(path, root_dir)
    return rel.replace("\\", "__").replace("/", "__").replace(":", "")



def extract_sequence(image_paths, mp_cfg, fps):
    mp_pose = mp.solutions.pose

    keypoints = []
    frame_has_pose = []

    prev_ctr = None

    if len(image_paths) == 0:
        raise RuntimeError("No images found")

    # Get frame size
    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError("Cannot read first image")

    height, width = first.shape[:2]

    with mp_pose.Pose(
        static_image_mode=mp_cfg["static_image_mode"],
        model_complexity=mp_cfg["model_complexity"],
        smooth_landmarks=mp_cfg["smooth_landmarks"],
        enable_segmentation=mp_cfg["enable_segmentation"],
        min_detection_confidence=mp_cfg["min_detection_confidence"],
        min_tracking_confidence=mp_cfg["min_tracking_confidence"],
    ) as pose:

        for path in image_paths:
            frame = cv2.imread(path)

            if frame is None:
                keypoints.append(np.zeros((25, 3), dtype=np.float32))
                frame_has_pose.append(0)
                continue

            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                mp_kpts = np.array(
                    [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark],
                    dtype=np.float32
                )

                ctr = mediapipe_to_ntu25(mp_kpts, prev_ctr)
                prev_ctr = ctr.copy()

              

                keypoints.append(ctr)
                frame_has_pose.append(1)
            else:
                keypoints.append(np.zeros((25, 3), dtype=np.float32))
                frame_has_pose.append(0)

    return (
        np.array(keypoints, dtype=np.float32),
        np.array(frame_has_pose, dtype=np.uint8),
        float(fps),
        width,
        height
    )



def save_sequence_npz(
    out_path: str,
    keypoints: np.ndarray,
    frame_has_pose: np.ndarray,
    fps: float,
    width: int,
    height: int,
    sample_path: str,
    image_paths
):
    ensure_dir(os.path.dirname(out_path))
    np.savez_compressed(
        out_path,
        keypoints=keypoints,                 # [T, 25, 3]
        frame_has_pose=frame_has_pose,       # [T]
        fps=np.array([fps], dtype=np.float32),
        frame_width=np.array([width], dtype=np.int32),
        frame_height=np.array([height], dtype=np.int32),
        sample_path=np.array([sample_path]),
        num_frames=np.array([len(image_paths)], dtype=np.int32),
        image_paths=np.array(image_paths, dtype=object),
    )


import cv2
import numpy as np
import os

import os
from math import ceil

def group_samples_by_length(sample_dirs, min_images_per_sample=2, n_groups=8, image_extensions=None):
    """
    Groups samples into n_groups of approximately equal total length.
    
    Args:
        sample_dirs: list of sample folder paths
        min_images_per_sample: minimum images to consider a sample
        n_groups: number of groups (e.g., number of workers)
        image_extensions: list of valid image extensions, e.g., [".jpg", ".png"]
    
    Returns:
        List of n_groups, each is a list of sample folder paths
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".png", ".jpeg"]
    
    # 1️⃣ Compute length of each sample
    def sample_length(path):
        count = 0
        for f in os.listdir(path):
            if os.path.splitext(f)[1].lower() in image_extensions:
                count += 1
        return max(count, min_images_per_sample)
    
    samples_with_length = [(p, sample_length(p)) for p in sample_dirs]
    
    # 2️⃣ Sort by descending length
    samples_with_length.sort(key=lambda x: x[1], reverse=True)
    
    # 3️⃣ Greedy assignment to balance total length per group
    groups = [[] for _ in range(n_groups)]
    group_lengths = [0] * n_groups
    
    for sample, length in samples_with_length:
        # assign to group with minimal current total length
        idx = group_lengths.index(min(group_lengths))
        groups[idx].append(sample)
        group_lengths[idx] += length
    
    return groups

def process_sample(args):
    sample_dir, input_root, skeleton_root, overlay_root, cfg = args

    try:
        mp_cfg = cfg["mediapipe"]
        fps = float(cfg["dataset"]["fps"])
        discovery_cfg = cfg["sample_discovery"]
        image_extensions = discovery_cfg["image_extensions"]
        sort_mode = cfg["runtime"]["sort_mode"]

        stem = safe_rel_stem(sample_dir, input_root)
        out_npz = os.path.join(skeleton_root, stem + ".npz")

        image_paths = list_images_in_sample(
            sample_dir=sample_dir,
            image_extensions=image_extensions,
            sort_mode=sort_mode
        )

        if len(image_paths) < discovery_cfg["min_images_per_sample"]:
            return ("failed", sample_dir, "not enough images")

        # Extract skeleton sequence
        keypoints, frame_has_pose, fps_out, width, height = extract_sequence(
            image_paths=image_paths,
            mp_cfg=mp_cfg,
            fps=fps
        )

        # Save skeleton npz
        save_sequence_npz(
            out_path=out_npz,
            keypoints=keypoints,
            frame_has_pose=frame_has_pose,
            fps=fps_out,
            width=width,
            height=height,
            sample_path=sample_dir,
            image_paths=image_paths
        )

        # --- SAVE OVERLAY VIDEO ---
        if overlay_root is not None:
            ensure_dir(overlay_root)
            overlay_path = os.path.join(overlay_root, stem + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(overlay_path, fourcc, fps_out, (width, height))

            # Optional: define edges (CTR-GCN style)
            edges = [
                (0,1),(1,2),(2,3),(0,12),(12,13),(13,14),(0,16),(16,17),(17,18),
                (2,4),(4,5),(5,6),(2,8),(8,9),(9,10),
                (12,16),(4,8),(6,10),(14,18)
            ]

            for idx, img_path in enumerate(image_paths):
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                kp = keypoints[idx]
                for (i,j) in edges:
                    pt1 = tuple((kp[i,:2]*[width, height]).astype(int))
                    pt2 = tuple((kp[j,:2]*[width, height]).astype(int))
                    cv2.line(img, pt1, pt2, (0,255,0), 2)
                for k in range(kp.shape[0]):
                    h, w, _ = img.shape
                    pt = tuple((kp[k, :2] * [w, h]).astype(int))
                    cv2.circle(img, pt, 3, (0,0,255), -1)
                writer.write(img)
            writer.release()

        return ("ok", sample_dir, len(image_paths), int(frame_has_pose.sum()))

    except Exception as e:
        return ("failed", sample_dir, str(e))



def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    input_root = cfg["input"]["root_dir"]
    fps = float(cfg["dataset"]["fps"])
    skeleton_root = cfg["output"]["skeleton_root"]
    save_overlay = bool(cfg["output"]["save_overlay"])
    overlay_root = cfg["output"]["overlay_root"]
    save_json_metadata = bool(cfg["output"]["save_json_metadata"])
    summary_filename = cfg["output"]["summary_filename"]

    discovery_cfg = cfg["sample_discovery"]
    image_extensions = discovery_cfg["image_extensions"]
    min_images_per_sample = int(discovery_cfg["min_images_per_sample"])
    recursive = bool(discovery_cfg["recursive"])
    runtime_cfg = cfg["runtime"]
    skip_existing = bool(runtime_cfg["skip_existing"])
    overwrite_summary = bool(runtime_cfg["overwrite_summary"])
    verbose_errors = bool(runtime_cfg["verbose_errors"])

    # -----------------------------
    # Discover sample folders
    # -----------------------------
    sample_dirs = list_sample_folders(
        root_dir=input_root,
        image_extensions=image_extensions,
        min_images_per_sample=min_images_per_sample,
        recursive=recursive
    )

    if len(sample_dirs) == 0:
        print(f"No sample folders found under: {input_root}")
        return

    print(f"Found {len(sample_dirs)} sample folders.")

    # -----------------------------
    # Prepare tasks (skip existing)
    # -----------------------------
    tasks = []
    for sample_dir in sample_dirs:
        stem = safe_rel_stem(sample_dir, input_root)
        out_npz = os.path.join(skeleton_root, stem + ".npz")
        if skip_existing and os.path.exists(out_npz):
            continue
        tasks.append((sample_dir, input_root, skeleton_root, overlay_root, cfg))

    print(f"[INFO] Processing {len(tasks)} samples (skipped {len(sample_dirs) - len(tasks)})")

    # -----------------------------
    # Group tasks by length to balance workers
    # -----------------------------
    max_workers = 4
    sample_dirs_to_group = [t[0] for t in tasks]
    groups = group_samples_by_length(sample_dirs_to_group, n_groups=max_workers)

    # Flatten groups into task list in grouped order
    grouped_tasks = []
    for g in groups:
        for s in g:
            # find full task tuple
            task_tuple = next(t for t in tasks if t[0] == s)
            grouped_tasks.append(task_tuple)

    # -----------------------------
    # Parallel execution
    # -----------------------------
    results = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, t) for t in grouped_tasks]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(("failed", "unknown", str(e)))

    # -----------------------------
    # Process results
    # -----------------------------
    summary = {
        "config_path": config_path,
        "input_root": input_root,
        "output_root": skeleton_root,
        "num_samples_found": len(sample_dirs),
        "fps_used": fps,
        "processed": 0,
        "skipped": len(sample_dirs) - len(tasks),
        "failed": 0,
        "files": []
    }

    for r in results:
        if r[0] == "ok":
            summary["processed"] += 1
            summary["files"].append({
                "sample_dir": r[1],
                "status": "ok",
                "num_frames": r[2],
                "num_frames_with_pose": r[3]
            })
        else:
            summary["failed"] += 1
            summary["files"].append({
                "sample_dir": r[1],
                "status": "failed",
                "error": r[2]
            })
            if verbose_errors:
                print(f"[ERROR] {r[1]}: {r[2]}")

    # -----------------------------
    # Save summary
    # -----------------------------
    summary_path = os.path.join(skeleton_root, summary_filename)
    if overwrite_summary or not os.path.exists(summary_path):
        save_metadata_json(summary_path, summary)

    print("\nDone.")
    print(f"Processed: {summary['processed']}")
    print(f"Skipped:   {summary['skipped']}")
    print(f"Failed:    {summary['failed']}")
    print(f"Summary:   {summary_path}")

def saving_summary():
    import os
    import json
    from pathlib import Path

    skeleton_root = "C:/GP/HAR-UP/FinalSkeletonExtractedSamples"
    summary_path = os.path.join(skeleton_root, "summary.json")

    files = []
    processed = 0

    for name in os.listdir(skeleton_root):
        if name.endswith(".npz"):
            processed += 1
            files.append({
                "output_npz": os.path.join(skeleton_root, name),
                "status": "ok"
            })

    summary = {
        "config_path": str(Path(__file__).parent / "config.yaml") if "__file__" in globals() else "config.yaml",
        "input_root": r"F:/GP_Dataset",
        "output_root": skeleton_root,
        "num_samples_found": processed,
        "fps_used": 18.0,
        "processed": processed,
        "skipped": 0,
        "failed": 0,
        "files": files
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", summary_path)


def debug_shapes(npz_path: str):
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)

    print("\n=== SHAPE DEBUG ===")
    for key in data.files:
        try:
            print(f"{key}: {data[key].shape}")
        except AttributeError:
            # For scalars or non-array objects
            print(f"{key}: (scalar or object)")
    print("===================\n")

def debug_skeleton_semantics(npz_path: str, visualize: bool = False):
    """
    Checks temporal consistency and semantic validity of a preprocessed skeleton sequence.
    
    Args:
        npz_path: path to the .npz file
        visualize: if True, optionally visualize joint trajectories (requires matplotlib)
    """
    import numpy as np
    import os

    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["keypoints"]   # [T, 25, 3]
    frame_has_pose = data["frame_has_pose"]  # [T]

    T, V, C = keypoints.shape
    print(f"\n=== SEMANTIC DEBUG ===")
    print(f"Frames: {T}, Joints: {V}, Channels: {C}")

    # 1. Check for all-zero frames
    zero_frames = np.sum(np.all(keypoints == 0, axis=(1, 2)))
    print(f"Frames with no detected pose (all zeros): {zero_frames}/{T}")

    # 2. Check for NaNs or Infs
    if np.isnan(keypoints).any():
        print("[WARNING] NaN values detected in keypoints!")
    if np.isinf(keypoints).any():
        print("[WARNING] Inf values detected in keypoints!")

    # 3. Temporal motion analysis
    if T > 1:
        # Compute L2 distance of each joint from previous frame
        diffs = np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=2)  # [T-1, V]
        mean_diffs = np.mean(diffs, axis=1)
        max_diffs = np.max(diffs, axis=1)

        print(f"Mean per-frame joint movement: min={mean_diffs.min():.4f}, max={mean_diffs.max():.4f}, mean={mean_diffs.mean():.4f}")
        print(f"Max joint movement per frame: min={max_diffs.min():.4f}, max={max_diffs.max():.4f}, mean={max_diffs.mean():.4f}")

        # Flag frames with unusually large jumps
        threshold = 0.5  # normalized unit (tweak as needed)
        jump_frames = np.where(max_diffs > threshold)[0]
        if len(jump_frames) > 0:
            print(f"[WARNING] {len(jump_frames)} frames have joint jumps > {threshold}")
        else:
            print("No abnormal joint jumps detected.")

    # 4. Pose detection ratio
    pose_ratio = np.mean(frame_has_pose)
    print(f"Pose detection ratio: {pose_ratio:.4f}")

    # 5. Optional: visualize trajectories of a few joints
    if visualize:
        try:
            import matplotlib.pyplot as plt
            joints_to_plot = [0, 1, 2]  # example: first three joints
            for j in joints_to_plot:
                plt.plot(keypoints[:, j, 0], label=f'joint {j} x')
                plt.plot(keypoints[:, j, 1], label=f'joint {j} y')
            plt.title("Joint trajectories (x/y)")
            plt.xlabel("Frame")
            plt.ylabel("Normalized position")
            plt.legend()
            plt.show()
        except ImportError:
            print("[INFO] matplotlib not installed; skipping visualization.")

    print("=== SEMANTIC DEBUG DONE ===\n")

if __name__ == "__main__":
    # main("C:/GP/MediaPipe_Preprocessing/config.yaml")
   
    # debug_skeleton_semantics(r"C:\GP\HAR-UP\PreprocessedDataset\Subject06__Activity10__Trial1__Subject6Activity10Trial1Camera1.npz", visualize=True)    
    # debug_shapes(r"C:\GP\HAR-UP\PreprocessedDataset\Subject06__Activity10__Trial1__Subject6Activity10Trial1Camera1.npz")
    saving_summary()