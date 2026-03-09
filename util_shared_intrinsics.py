"""
util_shared_intrinsics -- Load Shared Pinhole Intrinsics
========================================================

Loads the post-undistortion K matrix produced by MOD_EpisodeEnUndistort
(Step 1) so that the Pi3X SLAM / VO pipeline (Step 2) can use the real
camera intrinsics instead of the heuristic ``focal = max(H, W) * 1.2``.

The undistorted_intrinsics.yaml file contains:
    fx, fy  : focal lengths in pixels (at the undistorted resolution)
    cx, cy  : principal point (at the undistorted resolution)
    image_width, image_height : undistorted image dimensions
    fov, focal_scale          : original undistortion parameters

Usage::

    from util_shared_intrinsics import load_shared_intrinsics

    K = load_shared_intrinsics(input_base, source="auto")
    # K is (3, 3) numpy array or None
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_shared_intrinsics(
    input_base: Path,
    source: str = "auto",
    explicit_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load shared pinhole intrinsics from the upstream undistortion step.

    Args:
        input_base: EPISODING_* directory (parent of episode directories).
        source: ``"auto"`` to search input_base for
            ``undistorted_intrinsics.yaml``, ``"path"`` to use
            *explicit_path*.
        explicit_path: Path to undistorted_intrinsics.yaml (used when
            source="path").

    Returns:
        (3, 3) numpy array with the K matrix, or ``None`` if not
        found / disabled.
    """
    if source == "path":
        if explicit_path is None:
            raise ValueError(
                "shared_intrinsics.source='path' but shared_intrinsics.path "
                "is null -- provide an explicit path to the intrinsics YAML"
            )
        yaml_path = Path(explicit_path)
    else:
        yaml_path = Path(input_base) / "undistorted_intrinsics.yaml"

    if not yaml_path.is_file():
        logger.warning(
            "Shared intrinsics file not found: %s -- "
            "falling back to estimated intrinsics (focal = max(H,W)*1.2)",
            yaml_path,
        )
        return None

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data["cx"])
    cy = float(data["cy"])

    K = np.array([
        [fx,  0.0, cx ],
        [0.0, fy,  cy ],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    logger.info(
        "Loaded shared intrinsics from %s: "
        "fx=%.2f fy=%.2f cx=%.2f cy=%.2f (image %dx%d)",
        yaml_path.name,
        fx, fy, cx, cy,
        int(data.get("image_width", 0)),
        int(data.get("image_height", 0)),
    )
    return K


def scale_intrinsics(
    K: np.ndarray,
    original_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Scale a (3, 3) intrinsics matrix from one resolution to another.

    When Pi3X resizes images via ``pixel_limit`` before inference, the K
    from undistortion (at the original undistorted resolution) must be
    scaled by the same ratio so that fx, fy, cx, cy match the resized
    pixel coordinates.

    Args:
        K: (3, 3) intrinsics matrix at *original_hw* resolution.
        original_hw: (H, W) of the original undistorted images.
        target_hw: (H, W) after pixel-limit resizing.

    Returns:
        New (3, 3) intrinsics matrix scaled to *target_hw*.
    """
    orig_H, orig_W = original_hw
    tgt_H, tgt_W = target_hw

    if orig_H == tgt_H and orig_W == tgt_W:
        return K.copy()

    scale_x = tgt_W / orig_W
    scale_y = tgt_H / orig_H

    K_scaled = K.copy()
    K_scaled[0, :] *= scale_x
    K_scaled[1, :] *= scale_y

    logger.info(
        "Scaled intrinsics from %dx%d -> %dx%d "
        "(scale_x=%.4f, scale_y=%.4f): "
        "fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
        orig_W, orig_H, tgt_W, tgt_H,
        scale_x, scale_y,
        K_scaled[0, 0], K_scaled[1, 1],
        K_scaled[0, 2], K_scaled[1, 2],
    )
    return K_scaled


def build_K_4x4(K_3x3: np.ndarray, N: int) -> np.ndarray:
    """Embed a (3, 3) K into (N, 4, 4) homogeneous projection matrices.

    Produces the same format as ``estimate_intrinsics_4x4()`` in
    util_common.py -- a batch of 4x4 identity matrices with the top-left
    3x3 block replaced by K.

    Args:
        K_3x3: (3, 3) intrinsics matrix.
        N: Number of frames (all share the same K).

    Returns:
        (N, 4, 4) numpy array.
    """
    K_4x4 = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
    K_4x4[:, :3, :3] = K_3x3
    return K_4x4
