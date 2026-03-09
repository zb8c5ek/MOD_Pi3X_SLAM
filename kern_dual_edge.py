"""
kern_dual_edge -- Dual-edge VO group registration.

For each VO group, compute two independent alignment edges:

1. **Temporal edge** (``group_i ↔ group_{i+1}``):
   Uses overlapping frame poses between adjacent groups.

2. **Submap anchor edge** (``group_i ↔ submap_j``):
   Uses shared keyframe poses between the group and a SLAM submap.

Both edges are computed via SIM(3) Umeyama alignment on corresponding
camera positions.  The kernel reports quality metrics for each and
selects the better edge (or both when both are good).

Interface::

    results = compute_dual_edges(
        exported_groups, scene_graph, kf_timeline, kf_window,
        sim3_thresh=0.5,
    )
    # results["group_000"]["temporal_edge"]  -> dict or None
    # results["group_000"]["submap_anchor_edge"] -> dict or None
    # results["group_000"]["selected_edge"] -> "temporal" | "submap_anchor" | "both" | "none"
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kern_dual_edge")

_FRAME_IDX_RE = re.compile(r"^(\d{6})_")

# Quality thresholds
RMSD_GOOD = 0.05
RMSD_WARNING = 0.15
RMSD_BAD = 0.30


def _classify_rmsd(rmsd: float) -> str:
    if rmsd <= RMSD_GOOD:
        return "good"
    elif rmsd <= RMSD_WARNING:
        return "warning"
    elif rmsd <= RMSD_BAD:
        return "bad"
    return "failed"


def _frame_idx_from_path(p: str) -> Optional[int]:
    m = _FRAME_IDX_RE.match(os.path.basename(p))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Build per-group and per-submap pose lookups
# ---------------------------------------------------------------------------

def _build_group_pose_map(
    exported_groups: list,
    map_store,
    graph,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Map group_idx -> {frame_idx: cam2world_4x4} from COLMAP exports.

    The poses come from the SLAM submap's world-frame poses (which is what
    ``export_per_submap_colmap`` writes).  We reuse the same submap iteration
    order.
    """
    group_poses: Dict[int, Dict[int, np.ndarray]] = {}
    group_idx = 0

    for submap in map_store.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses_world = submap.get_all_poses_world(graph)
        fidx_map: Dict[int, np.ndarray] = {}
        for i, img_path in enumerate(submap.img_names):
            fidx = _frame_idx_from_path(str(img_path))
            if fidx is not None and i < len(poses_world):
                fidx_map[fidx] = poses_world[i]
        group_poses[group_idx] = fidx_map
        group_idx += 1

    return group_poses


def _build_submap_pose_map(
    scene_graph: dict,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Map submap_id -> {frame_idx: cam2world_global_4x4} from scene graph."""
    sm_poses: Dict[int, Dict[int, np.ndarray]] = {}
    for sm in scene_graph.get("submaps", []):
        sid = sm["submap_id"]
        fidx_map: Dict[int, np.ndarray] = {}
        for kf in sm.get("keyframes", []):
            fidx = kf.get("frame_idx")
            pose = kf.get("pose_cam2world_global")
            if fidx is not None and pose is not None:
                fidx_map[fidx] = np.array(pose, dtype=np.float64)
        sm_poses[sid] = fidx_map
    return sm_poses


# ---------------------------------------------------------------------------
# SIM(3) alignment on camera positions
# ---------------------------------------------------------------------------

def _align_poses_sim3(
    poses_a: Dict[int, np.ndarray],
    poses_b: Dict[int, np.ndarray],
    shared_fidxs: List[int],
    inlier_thresh: float = 0.5,
) -> Optional[dict]:
    """Align camera positions from two pose sets via SIM(3).

    Extracts the translation component (cam2world[:3, 3]) of each shared
    frame, then runs Umeyama + RANSAC.

    Args:
        poses_a: {frame_idx: 4x4} source poses.
        poses_b: {frame_idx: 4x4} target poses.
        shared_fidxs: Frame indices present in both.
        inlier_thresh: RANSAC inlier distance threshold.

    Returns:
        Alignment info dict, or None if too few correspondences.
    """
    if len(shared_fidxs) < 3:
        return None

    pts_a = np.array([poses_a[f][:3, 3] for f in shared_fidxs])
    pts_b = np.array([poses_b[f][:3, 3] for f in shared_fidxs])

    try:
        from kern_scale import estimate_sim3_ransac, estimate_rigid_kabsch
    except ImportError:
        logger.warning("kern_scale not available; skipping alignment")
        return None

    s, R, t_vec, T_rel, inlier_mask = estimate_sim3_ransac(
        pts_a, pts_b,
        n_iter=500,
        inlier_thresh=inlier_thresh,
        min_samples=3,
    )

    n_inliers = int(inlier_mask.sum())
    if n_inliers < 3:
        return {
            "status": "failed",
            "reason": f"only {n_inliers} inliers",
            "n_correspondences": len(shared_fidxs),
            "n_inliers": n_inliers,
        }

    _, _, _, kabsch_rmsd = estimate_rigid_kabsch(
        pts_a[inlier_mask], pts_b[inlier_mask])

    from kern_stitch_debug import decompose_sim3
    rel_s, rel_rot_deg, rel_t, rel_t_norm = decompose_sim3(T_rel)

    return {
        "status": _classify_rmsd(kabsch_rmsd),
        "kabsch_rmsd": float(kabsch_rmsd),
        "scale": float(rel_s),
        "rot_deg": float(rel_rot_deg),
        "t_norm": float(rel_t_norm),
        "n_correspondences": len(shared_fidxs),
        "n_inliers": n_inliers,
        "shared_frames": shared_fidxs,
    }


# ---------------------------------------------------------------------------
# Main: compute dual edges for all groups
# ---------------------------------------------------------------------------

def compute_dual_edges(
    exported_groups: list,
    scene_graph: dict,
    kf_timeline: List[int],
    kf_to_submap: Dict[int, int],
    map_store,
    graph,
    kf_window: int = 3,
    sim3_thresh: float = 0.5,
    strategy: str = "prefer_slam",
) -> Dict[str, dict]:
    """Compute dual-edge registration for all VO groups.

    Args:
        exported_groups:  List of (group_idx, group_dir, img_paths).
        scene_graph:      SLAM scene graph dict.
        kf_timeline:      Sorted SLAM keyframe frame indices.
        kf_to_submap:     frame_idx -> submap_id mapping.
        map_store:        GraphMap (for pose access).
        graph:            PoseGraph (for world-frame poses).
        kf_window:        ±K keyframe window size.
        sim3_thresh:      RANSAC inlier threshold.
        strategy:         Edge selection: "prefer_slam", "pick_best", "fuse".

    Returns:
        Dict keyed by group name, each containing:
        - temporal_edge: alignment info to next group (or None)
        - submap_anchor_edge: alignment info to SLAM submap (or None)
        - selected_edge: which edge was selected
        - selection_reason: why
    """
    from essn_vo import compute_kf_window_for_group

    group_poses = _build_group_pose_map(exported_groups, map_store, graph)
    submap_poses = _build_submap_pose_map(scene_graph)

    results: Dict[str, dict] = {}
    n_groups = len(exported_groups)

    for i, (group_idx, group_dir, img_paths) in enumerate(exported_groups):
        group_name = f"group_{group_idx:03d}"
        g_poses = group_poses.get(group_idx, {})
        g_fidxs = sorted(g_poses.keys())

        entry: dict = {
            "temporal_edge": None,
            "submap_anchor_edge": None,
            "selected_edge": "none",
            "selection_reason": "no edges available",
        }

        # --- A) Temporal edge: overlap with next group ---
        if i + 1 < n_groups:
            next_idx = exported_groups[i + 1][0]
            next_poses = group_poses.get(next_idx, {})
            shared = sorted(set(g_fidxs) & set(next_poses.keys()))
            if shared:
                temporal = _align_poses_sim3(
                    g_poses, next_poses, shared, sim3_thresh)
                if temporal:
                    temporal["target"] = f"group_{next_idx:03d}"
                    entry["temporal_edge"] = temporal

        # --- B) Submap anchor edge: kf_window keyframes ---
        kf_pre, kf_post = compute_kf_window_for_group(
            g_fidxs, kf_timeline, kf_window)
        anchor_kfs = kf_pre + kf_post

        # Also include any KFs inside the group that are in a SLAM submap
        kfs_inside = [f for f in g_fidxs if f in kf_to_submap]
        all_anchor_candidates = sorted(set(anchor_kfs + kfs_inside))

        if all_anchor_candidates:
            # Find the best submap to anchor to (most shared KFs)
            submap_counts: Dict[int, int] = {}
            for fidx in all_anchor_candidates:
                sid = kf_to_submap.get(fidx)
                if sid is not None:
                    submap_counts[sid] = submap_counts.get(sid, 0) + 1

            if submap_counts:
                best_sid = max(submap_counts, key=submap_counts.get)
                sm_poses = submap_poses.get(best_sid, {})
                shared = sorted(set(g_poses.keys()) & set(sm_poses.keys()))
                if shared:
                    anchor = _align_poses_sim3(
                        g_poses, sm_poses, shared, sim3_thresh)
                    if anchor:
                        anchor["target_submap"] = f"submap_{best_sid:03d}"
                        anchor["target_submap_id"] = best_sid
                        entry["submap_anchor_edge"] = anchor

        # --- C) Edge selection ---
        te = entry["temporal_edge"]
        sa = entry["submap_anchor_edge"]

        te_ok = te and te.get("status") not in ("failed", None)
        sa_ok = sa and sa.get("status") not in ("failed", None)

        if strategy == "prefer_slam":
            if sa_ok:
                entry["selected_edge"] = "submap_anchor"
                entry["selection_reason"] = (
                    f"SLAM anchor RMSD={sa['kabsch_rmsd']:.4f} "
                    f"({sa['status']})")
            elif te_ok:
                entry["selected_edge"] = "temporal"
                entry["selection_reason"] = (
                    f"temporal fallback RMSD={te['kabsch_rmsd']:.4f} "
                    f"({te['status']})")
            else:
                entry["selected_edge"] = "none"
                entry["selection_reason"] = "both edges failed or unavailable"
        elif strategy == "pick_best":
            if te_ok and sa_ok:
                if sa["kabsch_rmsd"] <= te["kabsch_rmsd"]:
                    entry["selected_edge"] = "submap_anchor"
                    entry["selection_reason"] = (
                        f"lower RMSD ({sa['kabsch_rmsd']:.4f} vs "
                        f"{te['kabsch_rmsd']:.4f})")
                else:
                    entry["selected_edge"] = "temporal"
                    entry["selection_reason"] = (
                        f"lower RMSD ({te['kabsch_rmsd']:.4f} vs "
                        f"{sa['kabsch_rmsd']:.4f})")
            elif sa_ok:
                entry["selected_edge"] = "submap_anchor"
                entry["selection_reason"] = "only SLAM anchor available"
            elif te_ok:
                entry["selected_edge"] = "temporal"
                entry["selection_reason"] = "only temporal available"
        else:
            if te_ok and sa_ok:
                entry["selected_edge"] = "both"
                entry["selection_reason"] = (
                    f"fuse: temporal RMSD={te['kabsch_rmsd']:.4f}, "
                    f"SLAM RMSD={sa['kabsch_rmsd']:.4f}")
            elif sa_ok:
                entry["selected_edge"] = "submap_anchor"
                entry["selection_reason"] = "only SLAM anchor available"
            elif te_ok:
                entry["selected_edge"] = "temporal"
                entry["selection_reason"] = "only temporal available"

        results[group_name] = entry

        _log_edge(group_name, entry)

    return results


def _log_edge(group_name: str, entry: dict):
    """Print a concise one-liner per group."""
    sel = entry["selected_edge"]
    reason = entry["selection_reason"]
    te = entry.get("temporal_edge")
    sa = entry.get("submap_anchor_edge")

    parts = [f"  [{group_name}]"]
    if te:
        parts.append(f"T:{te.get('kabsch_rmsd', 0):.4f}({te.get('status', '?')})")
    else:
        parts.append("T:--")
    if sa:
        parts.append(f"S:{sa.get('kabsch_rmsd', 0):.4f}({sa.get('status', '?')})")
    else:
        parts.append("S:--")
    parts.append(f"-> {sel}")

    logger.info(" | ".join(parts))
    print(" | ".join(parts), flush=True)


def print_dual_edge_summary(results: Dict[str, dict]):
    """Print a summary table of dual-edge registration results."""
    n = len(results)
    if n == 0:
        return

    n_temporal = sum(1 for r in results.values() if r["selected_edge"] == "temporal")
    n_slam = sum(1 for r in results.values() if r["selected_edge"] == "submap_anchor")
    n_both = sum(1 for r in results.values() if r["selected_edge"] == "both")
    n_none = sum(1 for r in results.values() if r["selected_edge"] == "none")

    te_rmsds = [r["temporal_edge"]["kabsch_rmsd"]
                for r in results.values()
                if r["temporal_edge"] and "kabsch_rmsd" in r["temporal_edge"]]
    sa_rmsds = [r["submap_anchor_edge"]["kabsch_rmsd"]
                for r in results.values()
                if r["submap_anchor_edge"] and "kabsch_rmsd" in r["submap_anchor_edge"]]

    print(f"\n{'='*60}")
    print(f"DUAL-EDGE REGISTRATION SUMMARY  ({n} groups)")
    print(f"{'='*60}")
    print(f"  Selected: temporal={n_temporal}  slam_anchor={n_slam}  "
          f"both={n_both}  none={n_none}")
    if te_rmsds:
        print(f"  Temporal RMSD:  mean={np.mean(te_rmsds):.4f}  "
              f"std={np.std(te_rmsds):.4f}  "
              f"min={np.min(te_rmsds):.4f}  max={np.max(te_rmsds):.4f}")
    if sa_rmsds:
        print(f"  SLAM Anchor RMSD: mean={np.mean(sa_rmsds):.4f}  "
              f"std={np.std(sa_rmsds):.4f}  "
              f"min={np.min(sa_rmsds):.4f}  max={np.max(sa_rmsds):.4f}")
    print(f"{'='*60}\n")
