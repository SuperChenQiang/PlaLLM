from os import PathLike
import numpy as np
from scipy.io import loadmat


def _pose(annotation: dict, path: str | PathLike[str]) -> np.ndarray:
    if "Pose_Para" not in annotation:
        raise ValueError(f"{path}: missing Pose_Para annotation")
    pose = np.asarray(annotation["Pose_Para"], dtype=np.float64).reshape(-1)[:3]
    if pose.size != 3 or not np.isfinite(pose).all():
        raise ValueError(f"{path}: Pose_Para must contain finite pitch/yaw/roll values")
    return pose


def _landmarks(annotation: dict, path: str | PathLike[str]) -> np.ndarray:
    if "pt2d" not in annotation:
        raise ValueError(f"{path}: missing pt2d annotation")
    points = np.asarray(annotation["pt2d"], dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] != 2
        or points.shape[1] == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{path}: pt2d must be a finite, nonempty (2, N) array")
    if np.any(np.ptp(points, axis=1) <= 0):
        raise ValueError(f"{path}: facial landmarks must span a nonzero area")
    return points


def load_pose_annotation(path: str | PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read landmarks and [pitch, yaw, roll] in radians in one disk read."""
    annotation = loadmat(path, variable_names=["Pose_Para", "pt2d"])
    return _landmarks(annotation, path), _pose(annotation, path)


def get_ypr_from_mat(mat_path: str | PathLike[str]) -> np.ndarray:
    """Return [pitch, yaw, roll] in radians (legacy function name)."""
    return _pose(loadmat(mat_path, variable_names=["Pose_Para"]), mat_path)


def get_pt2d_from_mat(mat_path: str | PathLike[str]) -> np.ndarray:
    """Return the annotation's 2D landmarks, shaped (2, N)."""
    return _landmarks(loadmat(mat_path, variable_names=["pt2d"]), mat_path)


def get_R(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Return the right-handed rotation Rz(roll) @ Ry(yaw) @ Rx(pitch).

    All input angles are in radians. The result is a (3, 3) float64 array.
    """
    cp, cy, cr = np.cos([pitch, yaw, roll])
    sp, sy, sr = np.sin([pitch, yaw, roll])
    return np.array(
        [
            [cr * cy, cr * sy * sp - sr * cp, cr * sy * cp + sr * sp],
            [sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp],
            [-sy, cy * sp, cy * cp],
        ],
        dtype=np.float64,
    )
