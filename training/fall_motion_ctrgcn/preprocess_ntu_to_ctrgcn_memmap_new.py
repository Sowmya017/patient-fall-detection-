import os
import re
import numpy as np
from tqdm import tqdm

# ============================================================
# Hardcoded hyperparameters (NO config file)
# ============================================================

# Where your .skeleton files are
ROOT_DIRS = [
    r"C:\GP\Dataset\nturgbd_skeletons_s001_to_s017",
    r"C:\GP\Dataset\nturgbd_skeletons_s018_to_s032",
    # r"C:\GP\Dataset",
]
RECURSIVE = True
DEDUPE_BY_FILENAME = True

# Output folder
OUT_DIR = r"C:\GP\cache\ntu_ctrgcn_preprocessed_new"

# CTR-GCN expected geometry
C = 3
T_OUT = 300
V = 25
M = 2

# Storage
DTYPE = np.float16  # float16 is faster/smaller; float32 also works

# ============================================================
# Official NTU120 split definitions used in CTR-GCN seq_transformation.py
# - CSub: train_ids list, test_ids = complement in [1..106]
# - CSet: train_ids even setups, test_ids odd setups (1..32)
# ============================================================
CSUB_TRAIN_IDS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81,
    82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
}
CSUB_TEST_IDS = {i for i in range(1, 107) if i not in CSUB_TRAIN_IDS}

CSET_TRAIN_IDS = {i for i in range(1, 33) if i % 2 == 0}  # even setups
CSET_TEST_IDS = {i for i in range(1, 33) if i % 2 == 1}   # odd setups

# ============================================================
# Filename parsing
# ============================================================
NTU_NAME_RE = re.compile(r"^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})\.skeleton$")


def parse_ntu_filename(fname: str):
    m = NTU_NAME_RE.match(os.path.basename(fname))
    if not m:
        return None
    setup, camera, subject, rep, action = map(int, m.groups())
    return {"setup": setup, "camera": camera, "subject": subject, "rep": rep, "action": action}


def iter_skeleton_files(roots, recursive=True):
    for root in roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Directory not found: {root}")
        if recursive:
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    if fn.endswith(".skeleton"):
                        yield os.path.join(dp, fn)
        else:
            for fn in os.listdir(root):
                if fn.endswith(".skeleton"):
                    yield os.path.join(root, fn)


# ============================================================
# Core: .skeleton -> (T, 150) = (T, 2 bodies * 25 joints * 3 coords)
# We keep only xyz. We store at most 2 bodies in file order.
# ============================================================
def read_skeleton_to_T150(path: str, V=25, M=2) -> np.ndarray:
    with open(path, "r") as f:
        lines = f.read().splitlines()

    idx = 0
    if idx >= len(lines):
        return np.zeros((0, M * V * 3), dtype=np.float32)

    T = int(lines[idx]); idx += 1
    out = np.zeros((T, M * V * 3), dtype=np.float32)

    for t in range(T):
        if idx >= len(lines):
            break
        num_bodies = int(lines[idx]); idx += 1

        for b in range(num_bodies):
            if idx >= len(lines): break

            # body info line (ignored except to skip it)
            idx += 1

            if idx >= len(lines): break
            num_joints = int(lines[idx]); idx += 1

            # Read joints
            # Each joint line has many fields; first 3 are x,y,z
            # We keep only first V joints, and only first M bodies.
            joint_xyz = np.zeros((V, 3), dtype=np.float32)
            for j in range(num_joints):
                if idx >= len(lines): break
                parts = lines[idx].split(); idx += 1
                if j < V and len(parts) >= 3:
                    joint_xyz[j, 0] = float(parts[0])
                    joint_xyz[j, 1] = float(parts[1])
                    joint_xyz[j, 2] = float(parts[2])

            if b < M:
                # Place into vector at block [b]
                start = b * V * 3
                out[t, start:start + V * 3] = joint_xyz.reshape(-1)

    return out


# ============================================================
# CTR-GCN seq_translation (key step)
# From CTR-GCN's NTU120 seq_transformation.py:
# - find "real first frame" of actor1 (first frame where first 75 dims are non-zero)
# - origin = joint-2 coords of actor1 => dims [3:6]
# - subtract origin tiled across all joints (25 or 50 joints)
# ============================================================
def seq_translation_T150(ske_T150: np.ndarray) -> np.ndarray:
    if ske_T150.shape[0] == 0:
        return ske_T150

    T = ske_T150.shape[0]
    # Determine if actor2 exists at all
    num_bodies = 2 if ske_T150.shape[1] == 150 else 1  # we always pass 150, so 2 here
    # Find real first frame of actor1
    i = 0
    while i < T:
        if np.any(ske_T150[i, :75] != 0):
            break
        i += 1
    if i == T:
        # no valid frame for actor1
        return ske_T150

    origin = np.copy(ske_T150[i, 3:6])  # joint-2 of actor1
    # subtract origin for every frame
    # for 2 bodies => tile origin 50 times (50 joints * 3 dims = 150)
    tiled = np.tile(origin, 50).astype(np.float32)
    out = ske_T150.astype(np.float32, copy=True)
    out -= tiled[None, :]
    return out


# ============================================================
# Align frames to T_OUT (default 300)
# - If only one body (actor2 all zeros), duplicate actor1 into actor2
# - Pad zeros if shorter, clip if longer
# ============================================================
def align_and_duplicate_T150(ske_T150: np.ndarray, T_out=300) -> np.ndarray:
    # ensure shape (T,150)
    if ske_T150.ndim != 2 or ske_T150.shape[1] != 150:
        raise ValueError(f"Expected (T,150), got {ske_T150.shape}")

    T = ske_T150.shape[0]

    # If actor2 is entirely missing, duplicate actor1 into actor2 for all frames
    # (this matches CTR-GCN align_frames behavior for single-person sequences) :contentReference[oaicite:1]{index=1}
    if T > 0 and np.all(ske_T150[:, 75:] == 0):
        ske_T150 = np.hstack([ske_T150[:, :75], ske_T150[:, :75]]).astype(np.float32, copy=False)

    # Clip / pad to T_out
    out = np.zeros((T_out, 150), dtype=np.float32)
    if T == 0:
        return out
    if T >= T_out:
        out[:] = ske_T150[:T_out]
    else:
        out[:T] = ske_T150
    return out


# ============================================================
# Convert (T,150) -> (3, T, 25, 2)
# ============================================================
def T150_to_CTVM(ske_T150: np.ndarray) -> np.ndarray:
    # (T,150) -> (T,2,25,3) -> (3,T,25,2)
    T = ske_T150.shape[0]
    tmv3 = ske_T150.reshape(T, M, V, 3)
    ctvM = np.transpose(tmv3, (3, 0, 2, 1)).copy()
    return ctvM


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pass 1: collect all valid file paths + labels + metadata
    file_paths = []
    labels = []
    infos = []
    seen = set()

    for p in iter_skeleton_files(ROOT_DIRS, RECURSIVE):
        fn = os.path.basename(p)
        if DEDUPE_BY_FILENAME and fn in seen:
            continue
        seen.add(fn)

        meta = parse_ntu_filename(fn)
        if meta is None:
            continue

        # NTU120 actions: A001..A120
        if not (1 <= meta["action"] <= 120):
            continue

        file_paths.append(p)
        labels.append(meta["action"] - 1)  # 0..119
        infos.append({**meta, "filename": fn, "path": p})

    N = len(file_paths)
    if N == 0:
        raise RuntimeError("No .skeleton files found (check ROOT_DIRS).")

    print(f"Found {N} samples. Writing memmap to: {OUT_DIR}")

    # Create memmap: (N,3,300,25,2)
    X_path = os.path.join(OUT_DIR, "X.kmp")
    X = np.memmap(X_path, mode="w+", dtype=DTYPE, shape=(N, C, T_OUT, V, M))

    bad = 0
    for i, pth in enumerate(tqdm(file_paths, desc="Preprocess NTU -> CTR-GCN")):
        try:
            ske = read_skeleton_to_T150(pth, V=V, M=M)      # (T,150)
            ske = seq_translation_T150(ske)                 # origin shift (joint-2 of actor1)
            ske = align_and_duplicate_T150(ske, T_out=T_OUT)  # (300,150)
            x = T150_to_CTVM(ske)                           # (3,300,25,2)
            X[i] = x.astype(DTYPE, copy=False)
        except Exception:
            bad += 1
            X[i] = np.zeros((C, T_OUT, V, M), dtype=DTYPE)

        if i % 5000 == 0:
            X.flush()
    X.flush()

    # Save labels and metadata
    y_path = os.path.join(OUT_DIR, "y.npy")
    np.save(y_path, np.asarray(labels, dtype=np.int64))

    info_path = os.path.join(OUT_DIR, "info.npy")
    np.save(info_path, np.asarray(infos, dtype=object), allow_pickle=True)

    meta_path = os.path.join(OUT_DIR, "meta.npy")
    np.save(
        meta_path,
        {
            "shape": (N, C, T_OUT, V, M),
            "dtype": "float16" if DTYPE == np.float16 else "float32",
            "layout": "NCTVM",
            "T": T_OUT,
            "V": V,
            "M": M,
            "C": C,
            "notes": "CTR-GCN-style preprocessing: seq_translation (origin=joint2 of actor1 real first frame), duplicate single body, pad/clip to T=300",
            "bad_rows": bad,
        },
        allow_pickle=True
    )

    # Save split indices (CSub and CSet) consistent with CTR-GCN's get_indices logic :contentReference[oaicite:2]{index=2}
    subjects = np.array([m["subject"] for m in infos], dtype=np.int32)
    setups = np.array([m["setup"] for m in infos], dtype=np.int32)

    csub_test_idx = np.where(np.isin(subjects, list(CSUB_TEST_IDS)))[0].astype(np.int64)
    csub_train_idx = np.where(np.isin(subjects, list(CSUB_TRAIN_IDS)))[0].astype(np.int64)

    cset_test_idx = np.where(np.isin(setups, list(CSET_TEST_IDS)))[0].astype(np.int64)
    cset_train_idx = np.where(np.isin(setups, list(CSET_TRAIN_IDS)))[0].astype(np.int64)

    np.save(os.path.join(OUT_DIR, "csub_train_idx.npy"), csub_train_idx)
    np.save(os.path.join(OUT_DIR, "csub_test_idx.npy"), csub_test_idx)
    np.save(os.path.join(OUT_DIR, "cset_train_idx.npy"), cset_train_idx)
    np.save(os.path.join(OUT_DIR, "cset_test_idx.npy"), cset_test_idx)

    print("\nDone.")
    print("X.kmp:", X_path)
    print("y.npy:", y_path)
    print("info.npy:", info_path)
    print("meta.npy:", meta_path)
    print("Bad rows (zero-filled):", bad)
    print("Memmap shape:", (N, C, T_OUT, V, M))
    print("CSub test samples:", len(csub_test_idx), "| CSet test samples:", len(cset_test_idx))


if __name__ == "__main__":
    main()