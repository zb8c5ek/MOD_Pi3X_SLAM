"""
UTIL_IO_Discovery - Image discovery and loading for the Pi3X SLAM pipeline.

Supports three data layouts:

  Layout A -- grouped (episode data):
    data_root/
      group_000/
        cam0/
          p+0_y+0_r+0/
            000002_..._cam0_p+0_y+0_r+0.jpg
          p+0_y+30_r+0/
            ...
        cam1/
          ...
      group_001/
        ...

  Layout B -- flat cam_orientation dirs:
    data_root/
      cam0_p+0_y+0_r+0/
        frame_000000.jpg
      cam1_p+0_y+0_r+0/
        ...

  Layout C -- flat images:
    data_root/
      images/
        img_0001.jpg
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

DIR_CAM_RE = re.compile(
    r'^(cam\d+)_'
    r'(p[+-]?\d+_y[+-]?\d+_r[+-]?\d+)$'
)

GROUP_RE = re.compile(r'^group_\d+$')


def _matches_filter(cam_id, orientation, cameras, camera_angles):
    """Check if a cam/orientation pair passes the user's filter."""
    if cam_id not in cameras:
        return False
    if cam_id in camera_angles:
        return orientation in camera_angles[cam_id]
    return True


def _discover_grouped(data_root, cameras, camera_angles, stride):
    """Layout A: group_XXX/camN/orientation/*.jpg"""
    group_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and GROUP_RE.match(d.name)
    ])
    if not group_dirs:
        return None

    # view_key -> sorted list of files across all groups
    view_file_lists: Dict[str, List[Path]] = {}

    for group_dir in group_dirs:
        for cam_dir in sorted(group_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            cam_id = cam_dir.name
            if cam_id not in cameras:
                continue
            for orient_dir in sorted(cam_dir.iterdir()):
                if not orient_dir.is_dir():
                    continue
                orientation = orient_dir.name
                if not _matches_filter(cam_id, orientation, cameras, camera_angles):
                    continue
                view_key = f"{cam_id}_{orientation}"
                files = sorted([
                    f for f in orient_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                ])
                if view_key not in view_file_lists:
                    view_file_lists[view_key] = []
                view_file_lists[view_key].extend(files)

    if not view_file_lists:
        return None

    # Apply stride per view
    for vk in view_file_lists:
        view_file_lists[vk] = view_file_lists[vk][::stride]

    for vk in sorted(view_file_lists):
        print(f"  {vk}: {len(view_file_lists[vk])} images ({len(group_dirs)} groups)")

    return view_file_lists


def _discover_flat_cam(data_root, cameras, camera_angles, stride):
    """Layout B: camN_ORIENTATION/*.jpg directly under data_root."""
    view_file_lists: Dict[str, List[Path]] = {}

    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        m = DIR_CAM_RE.match(entry.name)
        if not m:
            continue
        cam_id, orientation = m.group(1), m.group(2)
        if not _matches_filter(cam_id, orientation, cameras, camera_angles):
            continue
        view_key = f"{cam_id}_{orientation}"
        files = sorted([
            f for f in entry.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        files = files[::stride]
        view_file_lists[view_key] = files
        print(f"  {view_key}: {len(files)} images")

    return view_file_lists if view_file_lists else None


CAM_DIR_RE = re.compile(r'^cam\d+$')

def _discover_nested_cam(data_root, cameras, camera_angles, stride):
    """Layout D: camN/ORIENTATION/*.jpg -- nested cam dirs from undistort output."""
    view_file_lists: Dict[str, List[Path]] = {}

    for cam_dir in sorted(data_root.iterdir()):
        if not cam_dir.is_dir() or not CAM_DIR_RE.match(cam_dir.name):
            continue
        cam_id = cam_dir.name
        if cam_id not in cameras:
            continue
        for orient_dir in sorted(cam_dir.iterdir()):
            if not orient_dir.is_dir():
                continue
            orientation = orient_dir.name
            if not _matches_filter(cam_id, orientation, cameras, camera_angles):
                continue
            view_key = f"{cam_id}_{orientation}"
            files = sorted([
                f for f in orient_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ])
            files = files[::stride]
            if files:
                view_file_lists[view_key] = files
                print(f"  {view_key}: {len(files)} images")

    return view_file_lists if view_file_lists else None


def _interleave(view_file_lists):
    """Interleave images by index across all views for temporal consistency."""
    view_keys = sorted(view_file_lists.keys())
    max_len = max(len(files) for files in view_file_lists.values())

    all_paths = []
    for i in range(max_len):
        for vk in view_keys:
            files = view_file_lists[vk]
            if i < len(files):
                all_paths.append(str(files[i]))

    print(f"Total images discovered: {len(all_paths)} from {len(view_keys)} views")
    return all_paths


def discover_images(
    data_root: str,
    cameras: List[str],
    camera_angles: Dict[str, List[str]],
    stride: int = 1,
) -> List[str]:
    """
    Discover and return sorted image paths from data_root.

    Tries Layout A (grouped), then B (flat cam dirs), then C (flat images).
    Images are interleaved by timestamp across all selected views.

    Args:
        data_root: Root directory containing image data.
        cameras: Camera IDs to include, e.g. ["cam0", "cam1"].
        camera_angles: Per-camera orientation filter.
        stride: Frame stride (1 = every frame).

    Returns:
        List of image file paths.
    """
    data_root = Path(data_root)

    # Layout A: group_XXX/camN/orientation/
    view_files = _discover_grouped(data_root, cameras, camera_angles, stride)
    if view_files:
        return _interleave(view_files)

    # Layout D: camN/orientation/ (nested cam dirs from undistort)
    view_files = _discover_nested_cam(data_root, cameras, camera_angles, stride)
    if view_files:
        return _interleave(view_files)

    # Layout B: camN_orientation/
    view_files = _discover_flat_cam(data_root, cameras, camera_angles, stride)
    if view_files:
        return _interleave(view_files)

    # Layout C: flat images/ or loose files
    images_dir = data_root / "images"
    if images_dir.exists():
        flat_files = sorted([
            f for f in images_dir.glob("*")
            if f.suffix.lower() in IMAGE_EXTS
        ])
        if flat_files:
            result = [str(f) for f in flat_files[::stride]]
            print(f"Total images discovered: {len(result)} (flat images/ dir)")
            return result

    flat_files = sorted([
        f for f in data_root.glob("*")
        if f.suffix.lower() in IMAGE_EXTS
    ])
    if flat_files:
        result = [str(f) for f in flat_files[::stride]]
        print(f"Total images discovered: {len(result)} (flat root dir)")
        return result

    raise ValueError(f"No images found in {data_root}")


# ======================================================================
# Structured multi-view discovery (for multi-view keyframe selection)
# ======================================================================

def discover_timestamps(
    data_root: str,
    cameras: List[str],
    camera_angles: Dict[str, List[str]],
    stride: int = 1,
) -> 'tuple[list[dict[str, str]], list[str]]':
    """Discover images grouped by timestamp across all views.

    Returns:
        (timestamps, view_keys)
        timestamps: list of dicts, one per timestamp.
            Each dict maps view_key -> image_path.
            e.g. [{"cam0_p+0_y+0_r+0": "path/to/img.jpg", ...}, ...]
        view_keys: sorted list of all view keys.
    """
    data_root = Path(data_root)

    view_files = _discover_grouped(data_root, cameras, camera_angles, stride)
    if view_files is None:
        view_files = _discover_nested_cam(data_root, cameras, camera_angles, stride)
    if view_files is None:
        view_files = _discover_flat_cam(data_root, cameras, camera_angles, stride)
    if view_files is None:
        raise ValueError(
            f"discover_timestamps requires multi-view layout (A, B, or D) in {data_root}"
        )

    view_keys = sorted(view_files.keys())
    max_len = max(len(files) for files in view_files.values())

    timestamps = []
    for i in range(max_len):
        ts = {}
        for vk in view_keys:
            files = view_files[vk]
            if i < len(files):
                ts[vk] = str(files[i])
        if ts:
            timestamps.append(ts)

    n_views = len(view_keys)
    n_ts = len(timestamps)
    n_total = sum(len(ts) for ts in timestamps)
    print(f"Timestamps: {n_ts}, Views: {n_views}, Total images: {n_total}")
    return timestamps, view_keys
