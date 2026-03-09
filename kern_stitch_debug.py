"""
kern_stitch_debug - Per-stitch and cumulative debug artifact generation.

Writes per-stitch diagnostics:
  stitch_NNNN/info.json     -- alignment stats (scale, residual, inlier ratio)
  stitch_NNNN/merged.ply    -- merged point clouds of adjacent submaps
  stitch_NNNN/colmap/       -- COLMAP text model for the pair
  stitch_NNNN/overlap_poses.json   -- overlap camera poses (group A, B, Umeyama, aligned B→A)
  stitch_NNNN/overlap_poses_3d.html -- Plotly 3D visualization of overlap poses

And cumulative snapshots after each graph optimisation:
  cumulative_NNNN/cumulative.ply  -- all submaps merged
  cumulative_NNNN/colmap/         -- COLMAP text model

Extracted from essn_submap.SubmapProcessor to keep the ESSN under 500 LOC.
"""

import html as html_mod
import json
import os
from typing import Optional, Tuple, List

import numpy as np
import open3d as o3d
import torch

from termcolor import colored
from util_colmap import write_colmap_txt


def decompose_sim3(H):
    """Decompose a 4x4 SIM(3) matrix into scale, rotation (axis-angle deg), translation."""
    M = H[:3, :3]
    s = np.cbrt(np.linalg.det(M))
    R = M / max(abs(s), 1e-12)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    angle_rad = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    t = H[:3, 3]
    t_norm = np.linalg.norm(t)
    return s, angle_deg, t, t_norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_submap_world_data(submap, graph, max_pts: Optional[int] = 100000):
    """Get a submap's points, colors, and cam2world poses in world frame.

    max_pts: None = no downsampling (e.g. for small runs with 2 submaps).
    Returns:
        (points, colors, cam2world_poses, image_names)
    """
    pts = submap.get_points_in_world_frame(graph)
    colors = submap.get_points_colors()
    if pts is None or len(pts) == 0:
        pts = np.zeros((0, 3))
        colors = np.zeros((0, 3), dtype=np.uint8)
    else:
        valid = np.isfinite(pts).all(axis=1)
        n_common = min(len(pts), len(colors))
        pts = pts[:n_common][valid[:n_common]]
        colors = colors[:n_common][valid[:n_common]]
        if max_pts is not None and len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts, colors = pts[idx], colors[idx]

    poses_w = submap.get_all_poses_world(graph)
    names = (list(submap.img_names)
             if submap.img_names
             else [f"frame_{i}" for i in range(len(poses_w))])
    return pts, colors, poses_w, names


def get_submap_local_points_colors(submap):
    """Get a submap's points and colors in submap-local (canonical) frame, confidence-filtered."""
    point_list = []
    color_list = []
    for i in range(len(submap.pointclouds)):
        pts = submap.pointclouds[i].reshape(-1, 3)
        col = submap.colors[i].reshape(-1, 3)
        mask = submap.conf_masks[i].reshape(-1) > submap.conf_threshold
        point_list.append(pts[mask])
        color_list.append(col[mask])
    if not point_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    pts = np.vstack(point_list)
    colors = np.vstack(color_list)
    n_common = min(len(pts), len(colors))
    return pts[:n_common], colors[:n_common]


def _frame_hw(submap) -> Tuple[int, int]:
    """Return (H, W) of a submap's stored frames, or (0, 0) if unavailable."""
    frames = submap.get_all_frames()
    if frames is None:
        return 0, 0
    if isinstance(frames, torch.Tensor):
        _, _, H, W = frames.shape
    else:
        H, W = frames.shape[2], frames.shape[3]
    return H, W


def _homo_apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N,3) points; returns (N,3)."""
    n = pts.shape[0]
    ones = np.ones((n, 1), dtype=pts.dtype)
    homo = np.hstack([pts, ones])  # (N, 4)
    out = (H @ homo.T).T  # (N, 4)
    return out[:, :3]


# ---------------------------------------------------------------------------
# Per-stitch debug
# ---------------------------------------------------------------------------

def save_stitch_debug(
    stitch_debug_dir: str,
    seq: int,
    submap_id_curr: int,
    frame_id_curr: int,
    submap_id_prev: int,
    frame_id_prev: int,
    scale_factor: float,
    good_mask: np.ndarray,
    pts_curr: np.ndarray,
    pts_prev: np.ndarray,
    map_store,
    graph,
    H_overlap_alt: Optional[np.ndarray] = None,
    overlapping_node_id_prev: Optional[int] = None,
    max_pts_per_submap: Optional[int] = 100000,
):
    """Save JSON + merged PLY + COLMAP model for one inter-submap stitch.

    Produces per stitch directory stitch_NNNN/:
      info.json             -- machine-readable alignment stats
      merged.ply            -- full point clouds (primary alignment) in world frame
      merged_umeyama.ply    -- if H_overlap_alt given: merge using Umeyama-style transform
      merged_sl4.ply        -- if H_overlap_alt given: merge using SL4-style transform
      colmap/               -- COLMAP model (cameras, images, points3D)

    When H_overlap_alt and overlapping_node_id_prev are provided, also writes an
    alternate merged PLY so you can compare both Umeyama and SL4 stitching.
    max_pts_per_submap: None = no downsampling (e.g. for 2-submap runs).
    """
    if not stitch_debug_dir:
        return

    stitch_dir = os.path.join(stitch_debug_dir, f"stitch_{seq:04d}")
    os.makedirs(stitch_dir, exist_ok=True)

    # --- JSON info ---
    pts_prior_overlap = pts_prev.copy()
    pts_curr_overlap = pts_curr * scale_factor
    residual = np.linalg.norm(pts_prior_overlap - pts_curr_overlap, axis=1)

    info = {
        "stitch_index": seq,
        "submap_prev": int(submap_id_prev),
        "frame_prev": int(frame_id_prev),
        "submap_curr": int(submap_id_curr),
        "frame_curr": int(frame_id_curr),
        "scale_factor": float(scale_factor),
        "overlap": {
            "good_mask_count": int(good_mask.sum()),
            "good_mask_total": int(len(good_mask)),
            "good_mask_ratio": float(good_mask.sum() / max(len(good_mask), 1)),
        },
        "residual": {
            "min": float(residual.min()),
            "median": float(np.median(residual)),
            "mean": float(residual.mean()),
            "max": float(residual.max()),
            "std": float(residual.std()),
        },
        "prior_pts_range": [float(pts_prior_overlap.min()), float(pts_prior_overlap.max())],
        "curr_pts_range": [float(pts_curr_overlap.min()), float(pts_curr_overlap.max())],
    }
    with open(os.path.join(stitch_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=2)

    # --- Merged PLY: full point clouds of both submaps in world frame ---
    prior_submap = map_store.get_submap(submap_id_prev)
    current_submap = map_store.get_submap(submap_id_curr)

    pts_a, col_a, poses_a, names_a = get_submap_world_data(prior_submap, graph)
    pts_b, col_b, poses_b, names_b = get_submap_world_data(current_submap, graph)

    if len(pts_a) > 0 or len(pts_b) > 0:
        merged_pts = np.vstack([p for p in [pts_a, pts_b] if len(p) > 0])
        merged_col = np.vstack([c for c in [col_a, col_b] if len(c) > 0])
    else:
        merged_pts = np.zeros((0, 3))
        merged_col = np.zeros((0, 3), dtype=np.uint8)

    # --- COLMAP model for the pair ---
    colmap_dir = os.path.join(stitch_dir, "colmap")
    all_poses = []
    all_names = []
    if len(poses_a) > 0:
        all_poses.append(poses_a)
        all_names.extend(names_a)
    if len(poses_b) > 0:
        all_poses.append(poses_b)
        all_names.extend(names_b)
    if all_poses:
        combined_poses = np.concatenate(all_poses, axis=0)
        empty_pts = np.zeros((0, 3))
        empty_col = np.zeros((0, 3), dtype=np.uint8)
        H, W = _frame_hw(prior_submap)
        write_colmap_txt(colmap_dir, combined_poses, all_names,
                         empty_pts, empty_col, H, W)

    n_a, n_b = len(pts_a), len(pts_b)
    print(f"  [StitchDebug] stitch_{seq:04d}/  "
          f"({n_a}+{n_b}={n_a+n_b} pts)  "
          f"colmap/ ({len(all_names)} cams)  "
          f"residual mean={residual.mean():.4f}", flush=True)


# ---------------------------------------------------------------------------
# Cumulative snapshot
# ---------------------------------------------------------------------------

def save_cumulative_debug(stitch_debug_dir: str, seq: int, map_store, graph):
    """Save cumulative PLY + COLMAP of all submaps after graph optimisation."""
    if not stitch_debug_dir:
        return

    cum_dir = os.path.join(stitch_debug_dir, f"cumulative_{seq:04d}")
    os.makedirs(cum_dir, exist_ok=True)

    all_pts, all_colors = [], []
    all_poses, all_names = [], []
    H = W = 0

    for submap in map_store.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        pts, colors, poses_w, names = get_submap_world_data(submap, graph, max_pts=50000)
        if len(pts) > 0:
            all_pts.append(pts)
            all_colors.append(colors)
        if len(poses_w) > 0:
            all_poses.append(poses_w)
            all_names.extend(names)
        if H == 0:
            H, W = _frame_hw(submap)

    if not all_pts and not all_poses:
        return

    combined_pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))
    combined_col = np.vstack(all_colors) if all_colors else np.zeros((0, 3), dtype=np.uint8)

    # COLMAP (cameras + images only, no points — keeps output small)
    if all_poses:
        combined_poses = np.concatenate(all_poses, axis=0)
        colmap_dir = os.path.join(cum_dir, "colmap")
        empty_pts = np.zeros((0, 3))
        empty_col = np.zeros((0, 3), dtype=np.uint8)
        write_colmap_txt(colmap_dir, combined_poses, all_names, empty_pts, empty_col, H, W)

    print(f"  [StitchDebug] cumulative_{seq:04d}/  "
          f"colmap/ ({len(all_names)} cams, {len(all_pts)} submaps)", flush=True)


# ---------------------------------------------------------------------------
# End-of-run stitch alignment report
# ---------------------------------------------------------------------------

def print_stitch_report(records):
    """Print end-of-run alignment report from stitch records.

    Args:
        records: list of dicts from SubmapProcessor.stitch_records, each with:
            edge, is_lc, sim3_s, inliers, total_pts, kabsch_rmsd, rot_deg, t_norm
    """
    if not records:
        print("[StitchReport] No stitches recorded.")
        return

    sep = '=' * 90
    thin = '-' * 90
    print(f"\n{sep}")
    print("  STITCH ALIGNMENT REPORT  —  per-submap SIM(3)")
    print(sep)

    print(thin)
    hdr = (f"{'Edge':<20} {'Ty':<3} "
           f"{'SIM3 s':>9} {'Inl':>7} "
           f"{'Kab RMSD':>9} "
           f"{'rot°':>10} {'|t|':>10}")
    print(hdr)
    print(thin)

    scales, kabsch_rmsds, rots, tnorms = [], [], [], []

    for r in records:
        lc_tag = 'LC' if r['is_lc'] else 'SQ'
        inl_str = (f"{r['inliers']}/{r['total_pts']}"
                   if r['total_pts'] > 0 else '---')
        def _f(v, w=9): return f'{v:{w}.4f}' if not np.isnan(v) else '---'.rjust(w)

        row = (f"{r['edge']:<20} {lc_tag:<3} "
               f"{_f(r['sim3_s'])} {inl_str:>7} "
               f"{_f(r['kabsch_rmsd'])} "
               f"{_f(r.get('rot_deg', np.nan), 10)} "
               f"{_f(r.get('t_norm', np.nan), 10)}")
        print(row)

        if not np.isnan(r['sim3_s']): scales.append(r['sim3_s'])
        if not np.isnan(r['kabsch_rmsd']): kabsch_rmsds.append(r['kabsch_rmsd'])
        if not np.isnan(r.get('rot_deg', np.nan)): rots.append(r['rot_deg'])
        if not np.isnan(r.get('t_norm', np.nan)): tnorms.append(r['t_norm'])

    print(thin)
    def _stats(label, arr):
        a = np.array(arr)
        print(f"  {label:<22}: mean={a.mean():.6f}  std={a.std():.6f}  "
              f"min={a.min():.6f}  max={a.max():.6f}  (n={len(a)})")

    if scales: _stats('SIM(3) scale', scales)
    if rots: _stats('Rotation (deg)', rots)
    if tnorms: _stats('Translation |t|', tnorms)
    if kabsch_rmsds:
        _stats('Kabsch RMSD', kabsch_rmsds)
        k = np.array(kabsch_rmsds)
        if k.max() > 0.05:
            print(colored('  WARNING: High Kabsch RMSD — pose rigidity may be violated', 'red'))
    print(f"\n{sep}\n")
