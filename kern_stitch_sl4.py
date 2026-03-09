"""
kern_stitch_sl4 - SL(4) projective inter-submap alignment backend.

Estimates a 15-DoF projective transformation H in SL(4) between
overlapping submaps:

    H_overlap = inv(P_prev) @ P_curr @ H_scale

where:
  - P = K @ inv(cam2world) is the full projection matrix built from
    Pi3X's estimated intrinsics and cam2world poses
  - H_scale is a uniform scale estimated via median distance ratio

The SL(4) transform captures the full projective distortion between
submaps, which matters when intrinsics are unreliable.

Interface:
    align_submaps(pts_curr, pts_prev, **cfg) -> (T_rel, info)
"""

import time
import numpy as np
from kern_scale import estimate_rigid_kabsch


def _estimate_scale_pairwise(X, Y):
    """Median distance-ratio scale:  scale = median(||Y|| / ||X||)."""
    x_dists = np.linalg.norm(X, axis=1)
    y_dists = np.linalg.norm(Y, axis=1)
    valid = x_dists > 1e-12
    scales = y_dists[valid] / x_dists[valid]
    return float(np.median(scales))


def _projective_transform(H, pts):
    """Apply 4x4 projective transform:  p' = (H @ [p; 1])[:3] / w."""
    N = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N, 1))])
    out = pts_h @ H.T  # (N, 4)
    w = out[:, 3:4]
    safe_w = np.where(np.abs(w) < 1e-12, 1e-12 * np.sign(w + 1e-15), w)
    return out[:, :3] / safe_w


def _projective_error(H, pts_src, pts_dst):
    """Per-point Euclidean error after projective transfer."""
    projected = _projective_transform(H, pts_src)
    return np.linalg.norm(projected - pts_dst, axis=1)


def _decompose_sl4(H):
    """Extract approximate scale, rotation angle, translation from SL(4)."""
    sR = H[:3, :3]
    t = H[:3, 3]
    h44 = H[3, 3]

    det_sR = np.linalg.det(sR)
    s = np.cbrt(abs(det_sR)) if abs(det_sR) > 1e-15 else 1.0
    R_approx = sR / max(s, 1e-12)

    # Nearest rotation via SVD polar decomposition
    U, _, Vt = np.linalg.svd(R_approx)
    R_nearest = U @ Vt
    if np.linalg.det(R_nearest) < 0:
        R_nearest = U @ np.diag([1, 1, -1]) @ Vt

    angle_rad = np.arccos(np.clip((np.trace(R_nearest) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = float(np.degrees(angle_rad))
    t_eff = t / max(abs(h44), 1e-12)
    t_norm = float(np.linalg.norm(t_eff))

    proj_vec = H[3, :3]
    proj_mag = float(np.linalg.norm(proj_vec) / max(abs(h44), 1e-12))

    return s, angle_deg, t_eff, t_norm, proj_mag


def align_submaps(
    pts_curr: np.ndarray,
    pts_prev: np.ndarray,
    proj_mat_curr: np.ndarray = None,
    proj_mat_prev: np.ndarray = None,
    inlier_thresh: float = 0.5,
    max_ransac_iter: int = 5000,
    max_eval_pts: int = 50000,
) -> tuple:
    """Estimate SL(4) projective alignment from curr-local to prev-local.

    Two paths:
      A) Analytic (projmat) — when full 4x4 projection matrices
         P = K @ inv(cam2world) are provided (built from Pi3X outputs).
         Computes P_temp = inv(P_prev) @ P_curr and a median-distance-ratio
         scale correction.
      B) DLT + RANSAC — robust projective estimation directly from 3D-3D
         point correspondences.  Used when proj_mats are not available, or
         when they are pure intrinsics (detected automatically).

    Args:
        pts_curr: (N, 3) points in current submap local frame.
        pts_prev: (N, 3) corresponding points in prior submap local frame.
        proj_mat_curr: (4, 4) full projection matrix of the overlap frame in
                       curr submap.  Must include extrinsics, not just K.
        proj_mat_prev: (4, 4) full projection matrix of the overlap frame in
                       prev submap.  Must include extrinsics, not just K.
        inlier_thresh: Distance threshold for inlier classification.
        max_ransac_iter: Max RANSAC iterations for DLT path.
        max_eval_pts: Subsample to this many points for RANSAC residual eval.

    Returns:
        (T_rel, info) where:
            T_rel: 4x4 SL(4) matrix mapping curr -> prev (projective)
            info:  dict with diagnostic fields
    """
    _t0 = time.perf_counter()

    use_projmat = False
    if proj_mat_curr is not None and proj_mat_prev is not None:
        # Detect degenerate case: if proj_mats are pure intrinsics (same K),
        # inv(K) @ K = I and the analytic path degenerates.  Check the norm.
        P_temp = np.linalg.inv(proj_mat_prev) @ proj_mat_curr
        off_identity = np.linalg.norm(P_temp - np.eye(4))
        if off_identity > 1e-6:
            use_projmat = True

    _t_detect = time.perf_counter()

    if use_projmat:
        # Analytic path: full-projection-matrix-based alignment
        # P = K @ inv(cam2world) from Pi3X outputs
        # Transform current points through relative projection
        t1 = _projective_transform(P_temp, pts_curr)
        t2 = pts_prev

        # Scale estimation via median distance ratio
        scale = _estimate_scale_pairwise(t1, t2)
        H_scale = np.diag([scale, scale, scale, 1.0])

        # Compose: H_overlap maps curr -> prev
        T_rel = P_temp @ H_scale

        # Normalize to SL(4): det(T_rel) = 1
        det = np.linalg.det(T_rel)
        if abs(det) > 1e-15:
            sign = 1.0 if det > 0 else -1.0
            T_rel = sign * T_rel / (abs(det) ** 0.25)

        # Evaluate quality
        errors = _projective_error(T_rel, pts_curr, pts_prev)
        inlier_mask = errors < inlier_thresh
        n_inliers = int(inlier_mask.sum())
        path_used = 'projmat'

    else:
        # DLT + RANSAC from kern_sl4_solver
        from _RefCodes.MOD_InterGroupPoseEstimation.kern_sl4_solver import (
            ransac_sl4,
        )
        # Subsample for RANSAC efficiency (full set used for final eval)
        n_full = len(pts_curr)
        if n_full > max_eval_pts:
            sub_idx = np.random.choice(n_full, max_eval_pts, replace=False)
            pts_curr_sub = pts_curr[sub_idx]
            pts_prev_sub = pts_prev[sub_idx]
        else:
            pts_curr_sub = pts_curr
            pts_prev_sub = pts_prev

        _t_sub = time.perf_counter()
        T_rel, ransac_info = ransac_sl4(
            pts_curr_sub, pts_prev_sub,
            threshold=inlier_thresh,
            max_ransac_iter=max_ransac_iter,
            refine=True,
        )
        _t_ransac = time.perf_counter()
        # Evaluate on full set
        errors = _projective_error(T_rel, pts_curr, pts_prev)
        inlier_mask = errors < inlier_thresh
        n_inliers = int(inlier_mask.sum())
        scale = 1.0  # embedded in H
        path_used = 'dlt_ransac'
        _t_eval = time.perf_counter()
        print(f"    [SL4-DLT] n={n_full} sub={len(pts_curr_sub)} | "
              f"subsample={_t_sub-_t_detect:.3f}s "
              f"ransac={_t_ransac-_t_sub:.3f}s "
              f"full_eval={_t_eval-_t_ransac:.3f}s")

    _t_align = time.perf_counter()

    n_pts = len(pts_curr)
    inlier_ratio = n_inliers / max(n_pts, 1)

    # Kabsch cross-check on inliers
    kabsch_rmsd = np.nan
    if n_inliers >= 4:
        projected = _projective_transform(T_rel, pts_curr)
        _, _, _, kabsch_rmsd = estimate_rigid_kabsch(
            projected[inlier_mask], pts_prev[inlier_mask])
    _t_kabsch = time.perf_counter()

    # Decompose for diagnostics
    s_approx, rot_deg, t_eff, t_norm, proj_mag = _decompose_sl4(T_rel)
    _t_decompose = time.perf_counter()

    print(f"    [SL4] path={path_used} detect={_t_detect-_t0:.3f}s "
          f"align={_t_align-_t_detect:.3f}s "
          f"kabsch={_t_kabsch-_t_align:.3f}s "
          f"decompose={_t_decompose-_t_kabsch:.3f}s "
          f"total={_t_decompose-_t0:.3f}s")

    info = {
        'backend': 'sl4',
        'path': path_used,
        's': float(s_approx),
        'scale_est': float(scale) if use_projmat else float(s_approx),
        'R': None,  # not directly decomposable from SL(4)
        't': t_eff,
        'n_inliers': n_inliers,
        'inlier_ratio': inlier_ratio,
        'inlier_mask': inlier_mask,
        'kabsch_rmsd': float(kabsch_rmsd),
        'rot_deg': rot_deg,
        't_norm': t_norm,
        'proj_magnitude': proj_mag,
    }
    return T_rel, info


def format_diagnostics(info: dict, edge_label: str, n_overlap_frames: int,
                       n_pts: int, is_loop_closure: bool) -> str:
    """Format alignment diagnostics for console output."""
    box_w = 72
    bar = '=' * box_w
    lines = [
        bar,
        f"  STITCH {edge_label}  "
        f"({n_overlap_frames} overlap frames, {n_pts} pts)  "
        f"{'[LC]' if is_loop_closure else ''}",
        f"  SL(4) [{info.get('path','?')}]:  s~{info['s']:.6f}  rot~{info['rot_deg']:.4f}deg  "
        f"|t|~{info['t_norm']:.6f}  "
        f"proj={info['proj_magnitude']:.6f}  "
        f"inliers={info['n_inliers']}/{n_pts} ({info['inlier_ratio']:.1%})",
    ]
    if not np.isnan(info['kabsch_rmsd']):
        lines.append(f"  Kabsch SE(3) RMSD: {info['kabsch_rmsd']:.6f}")
    lines.append(bar)
    return '\n'.join(lines)


def make_stitch_record(info: dict, edge_label: str,
                       submap_id_prev: int, submap_id_curr: int,
                       is_loop_closure: bool, n_pts: int) -> dict:
    """Create a stitch record dict for end-of-run reporting."""
    return {
        'edge': edge_label,
        'is_lc': is_loop_closure,
        'submap_prev': submap_id_prev,
        'submap_curr': submap_id_curr,
        'sim3_s': info['s'],  # approximate scale for reporting consistency
        'inliers': info['n_inliers'],
        'total_pts': n_pts,
        'kabsch_rmsd': info['kabsch_rmsd'],
        'rot_deg': info['rot_deg'],
        't_norm': info['t_norm'],
        'backend': 'sl4',
        'proj_magnitude': info['proj_magnitude'],
    }
