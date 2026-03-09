"""
kern_stitch_sim3 - SIM(3) inter-submap alignment backend.

Estimates a 7-DoF similarity transform (scale + rotation + translation)
between overlapping submaps using Umeyama + RANSAC.

    p_prev = s * R @ p_curr + t

Returns T_rel as a 4x4 matrix [sR, t; 0, 1] that maps curr-local
points to prev-local points.

Used when camera intrinsics are well-estimated (shared K), so submaps
live in a metric/similarity-equivalent frame.

Interface:
    align_submaps(pts_curr, pts_prev, **cfg) -> (T_rel, info)
"""

import time
import numpy as np
from kern_scale import estimate_sim3_ransac, estimate_rigid_kabsch


def align_submaps(
    pts_curr: np.ndarray,
    pts_prev: np.ndarray,
    inlier_thresh: float = 0.5,
    n_iter: int = 1000,
    min_samples: int = 4,
) -> tuple:
    """Estimate SIM(3) alignment from curr-local to prev-local.

    Args:
        pts_curr: (N, 3) points in current submap local frame.
        pts_prev: (N, 3) corresponding points in prior submap local frame.
        inlier_thresh: RANSAC inlier distance threshold.
        n_iter: RANSAC iterations.
        min_samples: Minimum correspondences per RANSAC hypothesis.

    Returns:
        (T_rel, info) where:
            T_rel: 4x4 SIM(3) matrix  [sR, t; 0, 1]
            info:  dict with diagnostic fields:
                s, R, t, n_inliers, inlier_ratio, inlier_mask,
                kabsch_rmsd, rot_deg, t_norm
    """
    _t0 = time.perf_counter()
    s, R, t_vec, T_rel, inlier_mask = estimate_sim3_ransac(
        pts_curr, pts_prev,
        n_iter=n_iter,
        inlier_thresh=inlier_thresh,
        min_samples=min_samples,
    )
    _t_ransac = time.perf_counter()
    n_inliers = int(inlier_mask.sum())
    inlier_ratio = n_inliers / max(len(inlier_mask), 1)

    # Kabsch SE(3) cross-check on inliers
    kabsch_rmsd = np.nan
    if n_inliers >= 4:
        _, _, _, kabsch_rmsd = estimate_rigid_kabsch(
            pts_curr[inlier_mask], pts_prev[inlier_mask])
    _t_kabsch = time.perf_counter()

    # Decompose for diagnostics
    from kern_stitch_debug import decompose_sim3
    rel_s, rel_rot_deg, rel_t, rel_t_norm = decompose_sim3(T_rel)
    _t_decompose = time.perf_counter()

    print(f"    [SIM3] ransac={_t_ransac-_t0:.3f}s "
          f"kabsch={_t_kabsch-_t_ransac:.3f}s "
          f"decompose={_t_decompose-_t_kabsch:.3f}s "
          f"total={_t_decompose-_t0:.3f}s")

    info = {
        'backend': 'sim3',
        's': float(s),
        'R': R,
        't': t_vec,
        'n_inliers': n_inliers,
        'inlier_ratio': inlier_ratio,
        'inlier_mask': inlier_mask,
        'kabsch_rmsd': float(kabsch_rmsd),
        'rot_deg': float(rel_rot_deg),
        't_norm': float(rel_t_norm),
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
        f"  SIM(3): s={info['s']:.6f}  rot={info['rot_deg']:.4f}deg  "
        f"|t|={info['t_norm']:.6f}  "
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
        'sim3_s': info['s'],
        'inliers': info['n_inliers'],
        'total_pts': n_pts,
        'kabsch_rmsd': info['kabsch_rmsd'],
        'rot_deg': info['rot_deg'],
        't_norm': info['t_norm'],
        'backend': 'sim3',
    }
