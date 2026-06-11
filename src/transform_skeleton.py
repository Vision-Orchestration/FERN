"""
transform_skeleton.py
Geometric transformation: rotate a skeleton from an angled camera
back to front-view equivalent using R_y(-theta).
"""
import numpy as np

JOINT_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
N_JOINTS = 10


def rotation_y(theta_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Y-axis (vertical)."""
    t = np.radians(theta_deg)
    return np.array([
        [ np.cos(t), 0, np.sin(t)],
        [ 0,         1, 0        ],
        [-np.sin(t), 0, np.cos(t)],
    ], dtype=np.float32)


def transform_to_front(
    skeleton: np.ndarray,
    camera_angle_deg: float = 45.0,
    zero_z_after: bool = True,
) -> np.ndarray:
    """
    Transform (T, 30) skeleton from angled camera to front-view equivalent.

    Parameters
    ----------
    skeleton         : (T, 30) — feature columns only, no metadata
    camera_angle_deg : angle of camera relative to subject front (degrees)
                       positive = camera is to the subject's right
    zero_z_after     : True  -> set z=0 after transform (matches training data)
                       False -> keep real depth (for Phase 2 / stereo)

    Returns
    -------
    (T, 30) — front-equivalent skeleton
    """
    if skeleton.shape[1] != N_JOINTS * 3:
        raise ValueError(
            f"Expected {N_JOINTS * 3} feature cols, got {skeleton.shape[1]}"
        )
    T = skeleton.shape[0]
    R = rotation_y(-camera_angle_deg)

    joints   = skeleton.reshape(T, N_JOINTS, 3)
    rotated  = (R @ joints.transpose(0, 2, 1)).transpose(0, 2, 1)

    if zero_z_after:
        rotated[:, :, 2] = 0.0

    return rotated.reshape(T, N_JOINTS * 3)


def transform_csv(
    csv_path: str,
    output_path: str,
    camera_angle_deg: float = 45.0,
    zero_z_after: bool = True,
) -> dict:
    """
    Load a skeleton CSV, apply transform, write to output_path.
    Preserves all non-feature columns (frame_idx, pose_detected, metadata).
    Returns stats dict.
    """
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(csv_path)

    feature_cols = [
        c for c in df.columns
        if any(c.startswith(j) for j in JOINT_NAMES)
        and c.endswith(("_x", "_y", "_z"))
    ]
    if len(feature_cols) != N_JOINTS * 3:
        # may already be subset — try all x/y/z columns excluding metadata
        all_xyz = [c for c in df.columns if c.endswith(("_x","_y","_z"))
                   and not c.startswith("mid_hip") and not c.startswith("torso")]
        if len(all_xyz) >= N_JOINTS * 3:
            feature_cols = all_xyz[:N_JOINTS * 3]
        else:
            raise ValueError(
                f"{csv_path}: found {len(feature_cols)} feature cols, "
                f"expected {N_JOINTS * 3}"
            )

    raw = df[feature_cols].values.astype(np.float32)

    nan_mask  = np.isnan(raw)
    raw_clean = np.where(nan_mask, 0.0, raw)

    transformed = transform_to_front(raw_clean, camera_angle_deg, zero_z_after)

    transformed[nan_mask] = np.nan

    df[feature_cols] = transformed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_no_detect = int((df["pose_detected"] == 0).sum()) \
                  if "pose_detected" in df.columns else 0
    return {
        "input":      csv_path,
        "output":     output_path,
        "frames":     len(df),
        "no_detect":  n_no_detect,
        "detect_pct": 100 * (1 - n_no_detect / max(len(df), 1)),
    }


if __name__ == "__main__":
    dummy = np.random.randn(60, 30).astype(np.float32)
    out   = transform_to_front(dummy, camera_angle_deg=45.0, zero_z_after=True)
    assert out.shape == (60, 30), "Shape mismatch"
    assert np.all(out[:, 2::3] == 0.0), "z not zeroed"
    print("transform_skeleton.py: smoke test passed.")
