import os
import re
import random
import argparse
from typing import Dict, List, Iterable, Tuple, Any

import numpy as np
import yaml
from tqdm import tqdm

NTU_NAME_RE = re.compile(r"^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})\.skeleton$")


# ----------------------------
# Utils
# ----------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def parse_ntu_filename(fname: str) -> Dict[str, int]:
    m = NTU_NAME_RE.match(os.path.basename(fname))
    if not m:
        return None
    setup, camera, subject, rep, action = map(int, m.groups())
    return {"setup": setup, "camera": camera, "subject": subject, "rep": rep, "action": action}

def iter_skeleton_files(roots: List[str], recursive: bool) -> Iterable[str]:
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory not found: {root}")
        if recursive:
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    if fn.endswith(".skeleton"):
                        yield os.path.join(dp, fn)
        else:
            for fn in os.listdir(root):
                if fn.endswith(".skeleton"):
                    yield os.path.join(root, fn)


# ----------------------------
# NTU .skeleton parser
# ----------------------------
def _safe_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return 0

def _safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0

def read_skeleton_xyz_select2(path: str, num_joints: int = 25, max_bodies: int = 2) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads an NTU .skeleton file and returns:
      x: (T, M, V, 3) with M=max_bodies (2), V=25, xyz only.
      debug: small dict with parsing stats

    Body selection:
      - Track bodies by bodyID across frames.
      - Score each body by total count of "valid joints" across frames.
        We treat a joint as valid if trackingState != 0 (last token in joint line) AND xyz not all zeros.
      - Select top-2 bodies by score. Missing bodies are zeros.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()

    idx = 0
    if idx >= len(lines):
        return np.zeros((0, max_bodies, num_joints, 3), dtype=np.float32), {"empty": True}

    T = _safe_int(lines[idx]); idx += 1
    if T <= 0:
        return np.zeros((0, max_bodies, num_joints, 3), dtype=np.float32), {"T": T, "empty": True}

    # Store per bodyID: list of frames data (T, V, 3) + valid joint counts
    bodies_xyz: Dict[str, np.ndarray] = {}      # bodyID -> (T, V, 3)
    bodies_valid: Dict[str, np.ndarray] = {}    # bodyID -> (T,) valid joint count per frame

    # We’ll lazily create arrays when a bodyID first appears
    def ensure_body(body_id: str):
        if body_id not in bodies_xyz:
            bodies_xyz[body_id] = np.zeros((T, num_joints, 3), dtype=np.float32)
            bodies_valid[body_id] = np.zeros((T,), dtype=np.int32)

    frames_read = 0
    for t in range(T):
        if idx >= len(lines):
            break
        nb = _safe_int(lines[idx]); idx += 1

        frames_read += 1

        for _b in range(nb):
            if idx >= len(lines): break

            # body info line: contains bodyID as first token in most NTU files
            body_info = lines[idx].split(); idx += 1
            body_id = body_info[0] if len(body_info) > 0 else f"body{_b}"
            ensure_body(body_id)

            if idx >= len(lines): break
            jc = _safe_int(lines[idx]); idx += 1  # joint count (usually 25)

            valid_count = 0
            for j in range(jc):
                if idx >= len(lines): break
                parts = lines[idx].split(); idx += 1

                if j >= num_joints:
                    continue

                # NTU joint line has many values; first 3 are x,y,z
                x = _safe_float(parts[0]) if len(parts) > 0 else 0.0
                y = _safe_float(parts[1]) if len(parts) > 1 else 0.0
                z = _safe_float(parts[2]) if len(parts) > 2 else 0.0

                # trackingState is usually the last token
                tracking_state = _safe_int(parts[-1]) if len(parts) > 0 else 0

                bodies_xyz[body_id][t, j, 0] = x
                bodies_xyz[body_id][t, j, 1] = y
                bodies_xyz[body_id][t, j, 2] = z

                if tracking_state != 0 and not (x == 0.0 and y == 0.0 and z == 0.0):
                    valid_count += 1

            bodies_valid[body_id][t] = valid_count

    # Score each body by total valid joints
    body_scores = []
    for bid, valid_vec in bodies_valid.items():
        score = int(valid_vec.sum())
        body_scores.append((score, bid))
    body_scores.sort(reverse=True)

    selected = [bid for (_s, bid) in body_scores[:max_bodies]]

    # Build output (T, M, V, 3)
    out = np.zeros((T, max_bodies, num_joints, 3), dtype=np.float32)
    for m, bid in enumerate(selected):
        out[:, m, :, :] = bodies_xyz[bid]

    debug = {
        "T": T,
        "frames_read": frames_read,
        "num_bodies_seen": len(bodies_xyz),
        "selected_body_ids": selected,
        "top_scores": body_scores[:max_bodies],
    }
    return out, debug


# ----------------------------
# Optional normalization (off by default)
# ----------------------------
def normalize_skeleton_tmvc(x_tmvc: np.ndarray, center_joint: int = 1, scale_joint: int = 0, eps: float = 1e-6) -> np.ndarray:
    """
    x_tmvc: (T, M, V, 3)
    Center by center_joint and scale by distance to scale_joint, per frame per body.
    """
    x = x_tmvc.copy()
    T, M, V, C = x.shape
    for t in range(T):
        for m in range(M):
            body = x[t, m]
            if np.allclose(body, 0):
                continue
            center = body[center_joint].copy()
            body = body - center
            scale = np.linalg.norm(body[scale_joint])
            if scale > eps:
                body = body / scale
            x[t, m] = body
    return x


# ----------------------------
# Shape conversions / pad-clip
# ----------------------------
def to_ctvm(x_tmvc: np.ndarray) -> np.ndarray:
    """
    (T, M, V, C) -> (C, T, V, M)
    Here C=3 for xyz.
    """
    return np.transpose(x_tmvc, (3, 0, 2, 1)).copy()

def pad_or_clip_ctvm(x_ctvm: np.ndarray, T_out: int, pad_mode: str = "zeros") -> np.ndarray:
    """
    x_ctvm: (C,T,V,M) -> (C,T_out,V,M)
    For CTR-GCN eval, safest deterministic behavior is:
      - clip: take first T_out frames
      - pad: zeros or repeat_last
    """
    C, T, V, M = x_ctvm.shape
    if T == T_out:
        return x_ctvm
    if T > T_out:
        return x_ctvm[:, :T_out, :, :]

    pad = T_out - T
    if pad_mode == "repeat_last" and T > 0:
        last = x_ctvm[:, -1:, :, :]
        pad_tensor = np.repeat(last, pad, axis=1)
    else:
        pad_tensor = np.zeros((C, pad, V, M), dtype=x_ctvm.dtype)

    return np.concatenate([x_ctvm, pad_tensor], axis=1)


# ----------------------------
# Main preprocessing
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg["project"]["seed"]))

    data_cfg = cfg["data"]
    pp = cfg["preprocess"]

    roots = data_cfg["roots"]
    recursive = bool(data_cfg.get("recursive", True))
    dedupe = bool(data_cfg.get("dedupe_by_filename", True))

    num_joints = int(data_cfg.get("num_joints", 25))
    max_bodies = int(data_cfg.get("max_bodies", 2))

    num_classes = int(data_cfg.get("num_classes", 120))
    filter_action_max = int(data_cfg.get("filter_action_max", num_classes))

    out_dir = data_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    T_out = int(pp.get("T", 300))
    pad_mode = str(pp.get("pad_mode", "zeros")).lower()

    dtype_str = str(pp.get("dtype", "float16")).lower()
    dtype = np.float16 if dtype_str == "float16" else np.float32

    normalize = bool(pp.get("normalize", False))
    center_joint = int(pp.get("center_joint", 1))
    scale_joint = int(pp.get("scale_joint", 0))

    # ----------------------------
    # Pass 1: collect file paths + labels + metadata
    # ----------------------------
    file_paths: List[str] = []
    labels: List[int] = []
    infos: List[dict] = []
    seen = set()

    for p in iter_skeleton_files(roots, recursive):
        fn = os.path.basename(p)
        if dedupe and fn in seen:
            continue
        seen.add(fn)

        meta = parse_ntu_filename(fn)
        if meta is None:
            continue
        if meta["action"] > filter_action_max:
            continue

        file_paths.append(p)
        labels.append(meta["action"] - 1)  # 0-based
        infos.append({
            **meta,
            "filename": fn,
            "path": p,
        })

    N = len(file_paths)
    if N == 0:
        raise RuntimeError("No .skeleton files found after filtering.")

    # ----------------------------
    # Create memmap: (N, 3, 300, 25, 2)
    # ----------------------------
    X_path = os.path.join(out_dir, "X.kmp")
    X = np.memmap(X_path, mode="w+", dtype=dtype, shape=(N, 3, T_out, num_joints, max_bodies))

    bad = 0
    debug_keep = []  # store a few debug samples to help sanity-check

    for i, pth in enumerate(tqdm(file_paths, desc="Preprocess -> CTR-GCN memmap")):
        try:
            x_tmvc, dbg = read_skeleton_xyz_select2(
                pth, num_joints=num_joints, max_bodies=max_bodies
            )  # (T, M, V, 3)

            if normalize:
                x_tmvc = normalize_skeleton_tmvc(
                    x_tmvc, center_joint=center_joint, scale_joint=scale_joint
                )

            x_ctvm = to_ctvm(x_tmvc)                 # (3, T, 25, 2)
            x_ctvm = pad_or_clip_ctvm(x_ctvm, T_out, pad_mode=pad_mode)
            X[i] = x_ctvm.astype(dtype, copy=False)

            if len(debug_keep) < 10:
                debug_keep.append({"i": i, "file": os.path.basename(pth), **dbg})

        except Exception as e:
            bad += 1
            X[i] = np.zeros((3, T_out, num_joints, max_bodies), dtype=dtype)

        if i % 5000 == 0:
            X.flush()

    X.flush()

    # ----------------------------
    # Save side files
    # ----------------------------
    y_path = os.path.join(out_dir, "y.npy")
    np.save(y_path, np.asarray(labels, dtype=np.int64))

    meta_path = os.path.join(out_dir, "meta.npy")
    np.save(
        meta_path,
        {
            "shape": (N, 3, T_out, num_joints, max_bodies),
            "dtype": "float16" if dtype == np.float16 else "float32",
            "layout": "NCTVM",
            "description": "NTU .skeleton -> CTR-GCN format (N,3,T,25,2), xyz only",
            "pad_mode": pad_mode,
            "normalize": normalize,
        },
        allow_pickle=True
    )

    info_path = os.path.join(out_dir, "info.npy")
    np.save(info_path, np.asarray(infos, dtype=object), allow_pickle=True)

    # Extra run info (human-friendly)
    runinfo_path = os.path.join(out_dir, "runinfo.npy")
    np.save(
        runinfo_path,
        {
            "N": N,
            "bad_rows": bad,
            "T": T_out,
            "V": num_joints,
            "M": max_bodies,
            "dtype": "float16" if dtype == np.float16 else "float32",
            "debug_samples": debug_keep,
            "filter_action_max": filter_action_max,
        },
        allow_pickle=True
    )

    print("\nDone.")
    print("Output dir:", out_dir)
    print("X.kmp:", X_path)
    print("y.npy:", y_path)
    print("meta.npy:", meta_path)
    print("info.npy:", info_path)
    print("Bad rows (zero-filled):", bad)
    print("Memmap shape:", (N, 3, T_out, num_joints, max_bodies))


if __name__ == "__main__":
    main()