"""
essn_vo -- Visual Odometry data organiser (temporal grouping + staging).

This ESSN handles the "VO" half of the VO-en-SLAM pipeline:

1. Discover episodes and images under an EPISODING root.
2. Split timestamps into temporal **groups** (fixed-size chunks with overlap).
3. After SLAM has run, stage per-group images from EPISODING into the output
   directory and write a ``grouping_manifest.json`` for downstream MODs
   (MOD_KabschAnalyzer, MOD_InterGroupPoseEstimation, etc.).

The ESSN does *not* perform any 3D reconstruction itself -- that is done by
``essn_slam.Pi3xSLAM``.  It owns only the data layout contract.

Output layout produced::

    episode_dir/
      group_000/                 <-- per-group staged images
        cam0/p+0_y+30_r+0/...
        cam1/p+0_y-30_r+0/...
        sparse/0/                <-- COLMAP per-group (written by util_colmap)
      group_001/
        ...
      grouping_manifest.json
"""

import bisect
import datetime
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("essn_vo")

EPISODE_RE = re.compile(r"^episode_\d+")
UNDISTORT_RE = re.compile(r"^undistort_fov")
_FRAME_IDX_RE = re.compile(r"^(\d{6})_")


# ---------------------------------------------------------------------------
# Episode / image discovery
# ---------------------------------------------------------------------------

def discover_episodes(
    input_base: Path,
    episode_filter: Optional[List[str]] = None,
) -> List[Path]:
    """Find ``episode_*`` directories under the EPISODING root."""
    episodes = sorted(
        d for d in input_base.iterdir()
        if d.is_dir() and EPISODE_RE.match(d.name)
    )
    if episode_filter:
        episodes = [e for e in episodes if e.name in episode_filter]
    if not episodes:
        logger.warning("No episode directories found in %s", input_base)
    return episodes


def find_undistort_dir(episode_dir: Path) -> Path:
    """Return the ``undistort_fov*`` sub-directory (or episode_dir itself)."""
    for d in sorted(episode_dir.iterdir()):
        if d.is_dir() and UNDISTORT_RE.match(d.name):
            return d
    return episode_dir


# ---------------------------------------------------------------------------
# Keyframe timeline extraction from SLAM scene graph
# ---------------------------------------------------------------------------

def extract_kf_timeline(scene_graph: dict) -> List[int]:
    """Extract a sorted list of SLAM keyframe frame indices.

    Args:
        scene_graph: Dict returned by ``export_slam_scene_graph()``.

    Returns:
        Sorted list of unique frame indices across all submaps.
    """
    kf_indices = set()
    for sm in scene_graph.get("submaps", []):
        for kf in sm.get("keyframes", []):
            fidx = kf.get("frame_idx")
            if fidx is not None:
                kf_indices.add(fidx)
    return sorted(kf_indices)


def kf_to_submap_map(scene_graph: dict) -> Dict[int, int]:
    """Map each keyframe frame_idx to its containing submap_id.

    When a keyframe appears in multiple submaps (overlap), the first
    (lowest submap_id) is kept.
    """
    mapping: Dict[int, int] = {}
    for sm in scene_graph.get("submaps", []):
        sid = sm["submap_id"]
        for kf in sm.get("keyframes", []):
            fidx = kf.get("frame_idx")
            if fidx is not None and fidx not in mapping:
                mapping[fidx] = sid
    return mapping


def compute_kf_window_for_group(
    group_frame_indices: List[int],
    kf_timeline: List[int],
    K: int,
) -> Tuple[List[int], List[int]]:
    """Compute the ±K keyframe window extensions for a VO group.

    Args:
        group_frame_indices: Sorted frame indices of the group's core frames.
        kf_timeline:        Sorted global SLAM keyframe frame indices.
        K:                  Number of KFs to extend on each side.

    Returns:
        (kf_pre, kf_post): Lists of keyframe frame indices to prepend / append.
        These are keyframes OUTSIDE the group's core range.
    """
    if not group_frame_indices or not kf_timeline or K <= 0:
        return [], []

    t_start = group_frame_indices[0]
    t_end = group_frame_indices[-1]

    core_set = set(group_frame_indices)

    # Find KFs strictly before t_start
    idx_start = bisect.bisect_left(kf_timeline, t_start)
    kf_pre = [f for f in kf_timeline[max(0, idx_start - K):idx_start]
              if f not in core_set]

    # Find KFs strictly after t_end
    idx_end = bisect.bisect_right(kf_timeline, t_end)
    kf_post = [f for f in kf_timeline[idx_end:idx_end + K]
               if f not in core_set]

    return kf_pre, kf_post


# ---------------------------------------------------------------------------
# Per-group image staging
# ---------------------------------------------------------------------------

def _extract_frame_indices(img_paths: List[str]) -> List[int]:
    """Extract sorted unique frame indices from image filenames."""
    indices = set()
    for p in img_paths:
        m = _FRAME_IDX_RE.match(os.path.basename(p))
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def stage_group_images(
    episode_out_dir: Path,
    undistort_dir: Path,
    exported_groups: list,
    use_symlinks: bool = True,
    vo_cameras: Optional[List[str]] = None,
    vo_camera_angles: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict]:
    """Stage per-group images from EPISODING into the output directory.

    Creates::

        episode_out_dir/group_NNN/cam0/p+0_y+30_r+0/000002_..._cam0_p+0_y+30_r+0.jpg

    Args:
        episode_out_dir: Episode directory in the VO-en-SLAM output.
        undistort_dir:   ``undistort_fov*/`` directory from EPISODING.
        exported_groups: From ``export_per_submap_colmap``:
            list of ``(idx, dir, img_paths)`` tuples.
        use_symlinks:    Symlinks (True) or file copies (False).
        vo_cameras:      If set, only stage these cameras (e.g. ["cam0","cam1"]).
        vo_camera_angles: If set, only stage these angles per camera.

    Returns:
        Dict keyed by group name with ``frame_indices``, ``files``,
        ``num_frames``.
    """
    all_cameras = sorted(
        d.name for d in undistort_dir.iterdir()
        if d.is_dir() and d.name.startswith("cam")
    )
    cameras = [c for c in all_cameras if c in vo_cameras] if vo_cameras else all_cameras

    angles_per_cam: Dict[str, List[str]] = {}
    for cam in cameras:
        cam_dir = undistort_dir / cam
        all_angles = sorted(d.name for d in cam_dir.iterdir() if d.is_dir())
        if vo_camera_angles and cam in vo_camera_angles:
            angles_per_cam[cam] = [a for a in all_angles
                                   if a in vo_camera_angles[cam]]
        else:
            angles_per_cam[cam] = all_angles

    all_files_by_frame: Dict[int, Dict[str, Dict[str, Path]]] = {}
    for cam in cameras:
        for angle in angles_per_cam[cam]:
            angle_dir = undistort_dir / cam / angle
            for img in angle_dir.iterdir():
                if not img.is_file():
                    continue
                m = _FRAME_IDX_RE.match(img.name)
                if not m:
                    continue
                fidx = int(m.group(1))
                all_files_by_frame.setdefault(fidx, {}).setdefault(cam, {})[angle] = img

    staging_info: Dict[str, Dict] = {}

    for group_idx, _group_dir, img_full_paths in exported_groups:
        group_name = f"group_{group_idx:03d}"
        frame_indices = _extract_frame_indices(img_full_paths)

        group_out = episode_out_dir / group_name
        files_manifest: Dict[str, Dict[str, List[str]]] = {}
        n_staged = 0

        for fidx in frame_indices:
            cam_files = all_files_by_frame.get(fidx, {})
            for cam in cameras:
                angle_files = cam_files.get(cam, {})
                for angle in angles_per_cam[cam]:
                    src = angle_files.get(angle)
                    if src is None:
                        continue
                    dst_dir = group_out / cam / angle
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / src.name
                    if dst.exists():
                        continue
                    if use_symlinks:
                        try:
                            dst.symlink_to(src)
                        except OSError:
                            shutil.copy2(str(src), str(dst))
                    else:
                        shutil.copy2(str(src), str(dst))
                    n_staged += 1
                    files_manifest.setdefault(cam, {}).setdefault(angle, []).append(src.name)

        for cam in files_manifest:
            for angle in files_manifest[cam]:
                files_manifest[cam][angle].sort()

        staging_info[group_name] = {
            "frame_indices": frame_indices,
            "files": files_manifest,
            "num_frames": len(frame_indices),
        }
        logger.info("  [Stage] %s: %d frames, %d images staged",
                     group_name, len(frame_indices), n_staged)

    return staging_info


# ---------------------------------------------------------------------------
# Grouping manifest
# ---------------------------------------------------------------------------

def write_grouping_manifest(
    episode_dir: Path,
    exported_groups: list,
    staging_info: Dict[str, Dict],
    undistort_dir: Optional[Path],
    image_paths: list,
    group_size: int,
    overlap: int,
    stride: int,
    kf_window: int = 0,
    kf_timeline: Optional[List[int]] = None,
    registration_results: Optional[Dict[str, dict]] = None,
):
    """Write ``grouping_manifest.json`` with dual-edge registration data.

    Args:
        kf_window:             ±K keyframe window parameter.
        kf_timeline:           Sorted SLAM keyframe indices (for window calc).
        registration_results:  Per-group dict from ``kern_dual_edge``.
    """
    cameras: List[str] = []
    poses: List[str] = []
    if undistort_dir and undistort_dir.is_dir():
        cameras = sorted(
            d.name for d in undistort_dir.iterdir()
            if d.is_dir() and d.name.startswith("cam")
        )
        if cameras:
            first_cam = undistort_dir / cameras[0]
            poses = sorted(d.name for d in first_cam.iterdir() if d.is_dir())

    kf_tl = kf_timeline or []

    groups = []
    for group_idx, _group_dir, img_full_paths in exported_groups:
        group_name = f"group_{group_idx:03d}"
        info = staging_info.get(group_name, {})
        frame_indices = info.get("frame_indices", [])

        kf_pre, kf_post = compute_kf_window_for_group(
            frame_indices, kf_tl, kf_window) if kf_tl else ([], [])

        entry = {
            "name": group_name,
            "position_start": 0,
            "position_end": max(len(frame_indices) - 1, 0),
            "frame_start": frame_indices[0] if frame_indices else 0,
            "frame_end": frame_indices[-1] if frame_indices else 0,
            "frame_indices": frame_indices,
            "num_frames": len(frame_indices),
            "kf_window_pre": kf_pre,
            "kf_window_post": kf_post,
            "files": info.get("files", {}),
        }

        if registration_results and group_name in registration_results:
            entry["registration"] = registration_results[group_name]

        groups.append(entry)

    manifest = {
        "episode": episode_dir.name,
        "source_undistort": undistort_dir.name if undistort_dir else "",
        "source_path": str(undistort_dir) if undistort_dir else "",
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "group_size": group_size,
            "overlap_frames": overlap,
            "keyframe_window": kf_window,
            "stride_frame": stride,
        },
        "slam_kf_timeline": kf_tl,
        "cameras": cameras,
        "poses": poses,
        "total_frames": len(image_paths),
        "num_groups": len(groups),
        "groups": groups,
    }
    manifest_path = episode_dir / "grouping_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s", manifest_path)
