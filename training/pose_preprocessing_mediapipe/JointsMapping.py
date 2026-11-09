import numpy as np

# Mediapipe 33 keypoints indices
MP = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32
}

def midpoint(a, b):
    return (a + b) / 2

def smooth_joint(curr, prev, alpha=0.85):
    if prev is None:
        return curr
    return alpha * prev + (1 - alpha) * curr

def mediapipe_to_ntu25(mp_kpts, prev=None):
    """
    Converts Mediapipe 33 landmarks to NTU-RGB+D 25-joint format
    fully compatible with CTR-GCN adjacency.
    """
    ctr = np.zeros((25, 3), dtype=np.float32)

    # -------- Spine / Core --------
    hip_center = midpoint(mp_kpts[MP["left_hip"]], mp_kpts[MP["right_hip"]])
    shoulder_center = midpoint(mp_kpts[MP["left_shoulder"]], mp_kpts[MP["right_shoulder"]])

    ctr[0] = hip_center                     # spine_base
    ctr[1] = 0.6 * hip_center + 0.4 * shoulder_center  # spine_mid
    ctr[2] = shoulder_center                # neck
    ctr[3] = mp_kpts[MP["nose"]]           # head

    # -------- Arms --------
    ctr[4] = mp_kpts[MP["left_shoulder"]]
    ctr[5] = mp_kpts[MP["left_elbow"]]
    ctr[6] = mp_kpts[MP["left_wrist"]]
    ctr[7] = mp_kpts[MP["left_index"]]      # left hand tip

    ctr[8] = mp_kpts[MP["right_shoulder"]]
    ctr[9] = mp_kpts[MP["right_elbow"]]
    ctr[10] = mp_kpts[MP["right_wrist"]]
    ctr[11] = mp_kpts[MP["right_index"]]   # right hand tip

    # -------- Legs --------
    ctr[12] = mp_kpts[MP["left_hip"]]
    ctr[13] = mp_kpts[MP["left_knee"]]
    ctr[14] = mp_kpts[MP["left_ankle"]]
    ctr[15] = mp_kpts[MP["left_foot_index"]]

    ctr[16] = mp_kpts[MP["right_hip"]]
    ctr[17] = mp_kpts[MP["right_knee"]]
    ctr[18] = mp_kpts[MP["right_ankle"]]
    ctr[19] = mp_kpts[MP["right_foot_index"]]

    # -------- Spine shoulder & center joints --------
    ctr[20] = shoulder_center               # spine_shoulder (NTU 25)
    ctr[21] = ctr[7]                        # left_hand_tip (duplicate if needed)
    ctr[22] = mp_kpts[MP["left_thumb"]]     # left_thumb
    ctr[23] = ctr[11]                       # right_hand_tip
    ctr[24] = mp_kpts[MP["right_thumb"]]    # right_thumb

    # -------- Temporal smoothing --------
    if prev is not None:
        for i in range(5, 20):  # joints that tend to move/jitter
            ctr[i] = smooth_joint(ctr[i], prev[i])

    return ctr


# import numpy as np

# MP = {
#     "nose": 0,
#     "left_shoulder": 11,
#     "right_shoulder": 12,
#     "left_elbow": 13,
#     "right_elbow": 14,
#     "left_wrist": 15,
#     "right_wrist": 16,
#     "left_hip": 23,
#     "right_hip": 24,
#     "left_knee": 25,
#     "right_knee": 26,
#     "left_ankle": 27,
#     "right_ankle": 28,
# }

# def midpoint(a, b):
#     return (a + b) / 2

# def weighted_center(hip_center, shoulder_center):
#     """
#     More weight to hips (better fall detection)
#     """
#     return 0.6 * hip_center + 0.4 * shoulder_center


# def smooth_joint(current, previous, alpha=0.7):
#     """
#     Reduce jitter (important for motion stream)
#     """
#     if previous is None:
#         return current
#     return alpha * previous + (1 - alpha) * current


# def mediapipe_to_ctrgcn_optimized(mp, prev=None):
#     ctr = np.zeros((25, 3))

#     # ===== Core Landmarks =====
#     hip_center = midpoint(mp[MP["left_hip"]], mp[MP["right_hip"]])
#     shoulder_center = midpoint(mp[MP["left_shoulder"]], mp[MP["right_shoulder"]])

#     # 🔥 Improved Spine Modeling
#     spine_base = hip_center
#     spine_mid = weighted_center(hip_center, shoulder_center)
#     neck = shoulder_center
#     head = mp[MP["nose"]]

#     ctr[0] = spine_base
#     ctr[1] = spine_mid
#     ctr[2] = neck
#     ctr[3] = head

#     # ===== Arms (less important → slightly smoothed) =====
#     ctr[4] = mp[MP["left_shoulder"]]
#     ctr[5] = mp[MP["left_elbow"]]
#     ctr[6] = mp[MP["left_wrist"]]

#     ctr[8] = mp[MP["right_shoulder"]]
#     ctr[9] = mp[MP["right_elbow"]]
#     ctr[10] = mp[MP["right_wrist"]]

#     # ===== Legs (VERY IMPORTANT for falls) =====
#     ctr[12] = mp[MP["left_hip"]]
#     ctr[13] = mp[MP["left_knee"]]
#     ctr[14] = mp[MP["left_ankle"]]

#     ctr[16] = mp[MP["right_hip"]]
#     ctr[17] = mp[MP["right_knee"]]
#     ctr[18] = mp[MP["right_ankle"]]

#     # ===== Stabilization =====
#     if prev is not None:
#         for i in [6, 10, 14, 18]:  # wrists + ankles (noisy joints)
#             ctr[i] = smooth_joint(ctr[i], prev[i])

#     # ===== Symmetry Fix (very important) =====
#     ctr[7] = ctr[6]    # left hand
#     ctr[11] = ctr[10]  # right hand
#     ctr[15] = ctr[14]  # left foot
#     ctr[19] = ctr[18]  # right foot

#     # ===== Fill Remaining Joints =====
#     for i in range(20, 25):
#         ctr[i] = spine_base

#     return ctr