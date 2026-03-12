"""UTIL_calib_loader -- Single entry point for loading rig calibration.

Handles the mapping between calibration camera indices (as stored in the
Kalibr camchain YAML) and output camera indices (as they appear in
undistorted images and COLMAP reconstructions).

The undistort pipeline remaps cameras via stack_to_calib_map:
    stack_pos 0 (output cam0) -> calib cam2
    stack_pos 1 (output cam1) -> calib cam1
    stack_pos 2 (output cam2) -> calib cam0
    stack_pos 3 (output cam3) -> calib cam3

This means "cam0" in a COLMAP image name is NOT calibration cam0 --
it's calibration cam2. This module applies that mapping so callers get
transforms and baselines in output camera space.

If the supplier changes the calibration or the mapping, only this file
needs to be updated.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default mapping: output camera index -> calibration camera name.
# RopeCap v1: stacked image order (top to bottom) is cam2, cam1, cam0, cam3.
DEFAULT_STACK_TO_CALIB_MAP: dict[int, str] = {
    0: "cam2",
    1: "cam1",
    2: "cam0",
    3: "cam3",
}


def _invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


@dataclass
class OutputRigCalibration:
    """Rig geometry in OUTPUT camera space (matching COLMAP image names).

    All transforms and baselines use the output camera indices (0-3),
    which correspond to the image names _cam0_, _cam1_, etc.
    """
    # Baselines in output camera space
    front_baseline_m: float    # output cam0 - output cam1 (the front stereo pair)
    back_baseline_m: float     # output cam2 - output cam3
    left_baseline_m: float     # output cam0 - output cam2
    right_baseline_m: float    # output cam1 - output cam3

    # All pairwise distances [6 pairs]
    all_baselines: dict[tuple[int, int], float] = field(repr=False, default_factory=dict)

    # Transforms: T_output_camX_from_output_cam0
    T_cam1_from_cam0: np.ndarray = field(repr=False, default_factory=lambda: np.eye(4))
    T_cam2_from_cam0: np.ndarray = field(repr=False, default_factory=lambda: np.eye(4))
    T_cam3_from_cam0: np.ndarray = field(repr=False, default_factory=lambda: np.eye(4))

    # The mapping that was applied
    stack_to_calib_map: dict[int, str] = field(repr=False, default_factory=dict)


def load_rig_calibration(
    camchain_path: str,
    stack_to_calib_map: Optional[dict[int, str]] = None,
) -> OutputRigCalibration:
    """Load rig calibration and remap to output camera space.

    Args:
        camchain_path: Path to Kalibr camchain YAML.
        stack_to_calib_map: Mapping from output camera index to calibration
            camera name. If None, uses DEFAULT_STACK_TO_CALIB_MAP.

    Returns:
        OutputRigCalibration with transforms and baselines in output camera
        space (matching COLMAP image names).

    Raises:
        ValueError: If cam0-cam1 (output space) is not the shortest baseline
            among all pairs -- this indicates a mapping error.
    """
    import yaml
    import re as _re

    if stack_to_calib_map is None:
        stack_to_calib_map = DEFAULT_STACK_TO_CALIB_MAP

    with open(camchain_path) as f:
        calib = yaml.safe_load(f)

    # --- Step 1: Parse the raw camchain (calibration space) ---
    cam_ids = sorted(
        int(k[3:]) for k in calib if k.startswith("cam") and k[3:].isdigit()
    )

    # Detect chain root
    root = None
    for cid in cam_ids:
        if "T_cn_cnm1" not in calib[f"cam{cid}"]:
            root = cid
            break
    if root is None:
        raise ValueError("No chain root found (all cameras have T_cn_cnm1)")

    # Get YAML key order for chain
    with open(camchain_path) as f:
        raw = f.read()
    chain_order = [
        int(m.group(1)) for m in _re.finditer(r'^cam(\d+)\s*:', raw, _re.MULTILINE)
    ]
    if chain_order[0] != root:
        raise ValueError(
            f"Expected chain root cam{root} first in YAML, "
            f"got cam{chain_order[0]}. Order: {chain_order}")

    # Build T_calib_camX_from_calib_root
    T_calib_from_root: dict[int, np.ndarray] = {root: np.eye(4)}
    for i in range(1, len(chain_order)):
        cid = chain_order[i]
        prev = chain_order[i - 1]
        T_cn_cnm1 = np.array(calib[f"cam{cid}"]["T_cn_cnm1"])
        T_calib_from_root[cid] = T_cn_cnm1 @ T_calib_from_root[prev]

    logger.info("Camchain: order=%s root=cam%d (%d cameras)",
                chain_order, root, len(chain_order))

    # --- Step 2: Build mapping from output index to calibration index ---
    # stack_to_calib_map: {output_idx: "camN"} -> {output_idx: N}
    out_to_calib: dict[int, int] = {}
    for out_idx, calib_name in stack_to_calib_map.items():
        calib_idx = int(calib_name.replace("cam", ""))
        out_to_calib[out_idx] = calib_idx

    logger.info("Camera mapping (output -> calib): %s",
                {f"out_cam{k}": f"calib_cam{v}" for k, v in sorted(out_to_calib.items())})

    # --- Step 3: Compute transforms in output camera space ---
    # T_output_camX_from_output_cam0 =
    #   T_calib[map[X]]_from_root @ inv(T_calib[map[0]]_from_root)

    calib_idx_0 = out_to_calib[0]
    if calib_idx_0 not in T_calib_from_root:
        raise ValueError(f"Output cam0 maps to calib cam{calib_idx_0} which is not in the chain")

    T_root_from_calib0 = _invert_se3(T_calib_from_root[calib_idx_0])

    def _T_outX_from_out0(out_x: int) -> np.ndarray:
        calib_x = out_to_calib[out_x]
        return T_calib_from_root[calib_x] @ T_root_from_calib0

    out_cams = sorted(out_to_calib.keys())

    T_out = {}
    for x in out_cams:
        T_out[x] = _T_outX_from_out0(x)

    # --- Step 4: Compute all pairwise distances ---
    all_baselines: dict[tuple[int, int], float] = {}
    for i in range(len(out_cams)):
        for j in range(i + 1, len(out_cams)):
            ci, cj = out_cams[i], out_cams[j]
            T_j_from_i = T_out[cj] @ _invert_se3(T_out[ci])
            dist = float(np.linalg.norm(T_j_from_i[:3, 3]))
            all_baselines[(ci, cj)] = dist

    front = all_baselines.get((0, 1), 0.0)
    back = all_baselines.get((2, 3), 0.0)
    left = all_baselines.get((0, 2), 0.0)
    right = all_baselines.get((1, 3), 0.0)

    logger.info("Output-space baselines: front(cam0-1)=%.4f m, back(cam2-3)=%.4f m, "
                "left(cam0-2)=%.4f m, right(cam1-3)=%.4f m",
                front, back, left, right)

    # --- Step 5: Safety check -- cam0-cam1 should be the shortest pair ---
    min_pair = min(all_baselines, key=all_baselines.get)
    min_dist = all_baselines[min_pair]

    if min_pair != (0, 1):
        raise ValueError(
            f"FRONT PAIR CHECK FAILED: output cam0-cam1 baseline ({front:.4f} m) "
            f"is NOT the shortest pair. Shortest is cam{min_pair[0]}-cam{min_pair[1]} "
            f"({min_dist:.4f} m). This likely means the stack_to_calib_map is wrong "
            f"or the calibration file has changed. Current mapping: {stack_to_calib_map}")

    logger.info("Front pair check OK: cam0-cam1 (%.4f m) is shortest baseline", front)

    # --- Step 6: Log all pairwise for debugging ---
    for (ci, cj), dist in sorted(all_baselines.items()):
        ci_calib = out_to_calib[ci]
        cj_calib = out_to_calib[cj]
        logger.debug("  out_cam%d-out_cam%d (calib cam%d-cam%d): %.6f m",
                      ci, cj, ci_calib, cj_calib, dist)

    return OutputRigCalibration(
        front_baseline_m=front,
        back_baseline_m=back,
        left_baseline_m=left,
        right_baseline_m=right,
        all_baselines=all_baselines,
        T_cam1_from_cam0=T_out.get(1, np.eye(4)),
        T_cam2_from_cam0=T_out.get(2, np.eye(4)),
        T_cam3_from_cam0=T_out.get(3, np.eye(4)),
        stack_to_calib_map=stack_to_calib_map,
    )
