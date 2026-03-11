"""
BURNPIPE_VO_Pipeline -- Thin orchestrator for VO-en-SLAM pipeline.

Called by LORDPIPE_StreamLine.py (Step 2) or standalone::

    python MOD_Pi3X_SLAM/_WegwerfSkript_VO/BURNPIPE_VO_Pipeline.py <step2_vo.yaml>

Delegates to two ESSNs:

* **essn_vo**   -- episode discovery, temporal grouping, image staging,
                   grouping manifest (backward-compat with downstream MODs).
* **essn_slam** -- keyframe selection, submap stitching, loop closure,
                   global reconstruction.

Output layout::

    VO-en-SLAM_EPISODING_{datetime}/
    ├── pipeline_report.json
    ├── VO/                                  <-- VO temporal groups
    │   └── episode_{name}/
    │       ├── grouping_manifest.json
    │       ├── group_000/
    │       │   ├── cam0/p+0_y+30_r+0/*.jpg
    │       │   └── sparse/0/cameras,images,points3D.txt
    │       └── group_001/
    └── SLAM/                                <-- SLAM keyframe submaps
        └── episode_{name}/
            ├── scene_graph.json
            ├── visualization_all.rrd
            ├── KeyFrames/cam0/p+0_y+30_r+0/  <-- all unique KFs
            ├── colmap/
            ├── kf_debug/
            ├── stitch_debug/
            └── submap_000/cam0/p+0_y+30_r+0/ <-- per-submap KFs
"""

import sys
import os
import re
import json
import time
import bisect
import shutil
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: add module root to sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent        # _WegwerfSkript_VO/
_MODULE_ROOT = _SCRIPT_DIR.parent                    # MOD_Pi3X_SLAM/

for p in (str(_MODULE_ROOT), str(_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from essn_slam import SLAMConfig, Pi3xSLAM
from essn_vo import (
    discover_episodes,
    find_undistort_dir,
    stage_group_images,
    write_grouping_manifest,
    extract_kf_timeline,
    kf_to_submap_map,
)
from kern_dual_edge import compute_dual_edges, print_dual_edge_summary
from util_pipeline_report import generate_pipeline_summary_html
from util_shared_intrinsics import load_shared_intrinsics
from util_colmap import (
    export_per_submap_colmap,
    export_per_submap_colmap_slam,
    export_all_colmap,
    export_slam_scene_graph,
)
from IO_UTIL_Discovery import discover_timestamps, discover_images

logger = logging.getLogger("BURNPIPE_VO_Pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
_FRAME_IDX_RE = re.compile(r"^(\d{6})_")
_CAM_ANGLE_RE = re.compile(r"_(cam\d+)_(p[+-]\d+_y[+-]\d+_r[+-]\d+)\.\w+$")


# ============================================================================
# Config loading
# ============================================================================

def load_vo_config(yaml_path: str) -> Dict[str, Any]:
    """Load step2_vo.yaml and return the raw config dict."""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def _parse_cameras(input_cfg: dict) -> Tuple[List[str], Dict[str, List[str]]]:
    """Parse cameras config into (camera_list, camera_angles_dict)."""
    raw = input_cfg.get('cameras', {})
    if isinstance(raw, dict):
        cameras = sorted(raw.keys())
        camera_angles = {cam: [raw[cam]] for cam in cameras}
    elif isinstance(raw, list):
        cameras = list(raw)
        camera_angles = input_cfg.get('camera_angles', {})
    else:
        cameras = ['cam0']
        camera_angles = {}
    return cameras, camera_angles


# ============================================================================
# Image discovery (wraps IO_UTIL_Discovery)
# ============================================================================

def _discover_episode_images(
    image_root: Path,
    cameras: List[str],
    camera_angles: Dict[str, List[str]],
    stride: int = 1,
) -> Tuple[Optional[list], list, bool]:
    """Discover images, trying multi-view timestamps then flat paths.

    Returns:
        (timestamps_or_None, view_keys_or_flat_paths, is_multi_view)
    """
    try:
        timestamps, view_keys = discover_timestamps(
            data_root=str(image_root),
            cameras=cameras,
            camera_angles=camera_angles,
            stride=stride,
        )
        return timestamps, view_keys, True
    except ValueError:
        pass
    try:
        flat_paths = discover_images(
            data_root=str(image_root),
            cameras=cameras,
            camera_angles=camera_angles,
            stride=stride,
        )
        return None, flat_paths, False
    except ValueError:
        pass
    logger.error("No images found in %s", image_root)
    return None, [], False


# ============================================================================
# Output directory creation
# ============================================================================

def _create_vo_output_dir(input_base: Path, output_base: Optional[str]) -> Path:
    """Create ``VO-en-SLAM_EPISODING_{datetime}/`` output directory."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_base:
        vo_dir = Path(output_base)
    else:
        parent = input_base.parent
        vo_dir = parent / f"VO-en-SLAM_EPISODING_{ts}"
    vo_dir.mkdir(parents=True, exist_ok=True)
    return vo_dir


def _create_episode_outputs(
    vo_output_dir: Path, episode_name: str,
) -> Tuple[Path, Path]:
    """Create per-episode VO and SLAM output directories.

    Layout::

        vo_output_dir/
          VO/episode_name/group_000/ ...
          SLAM/episode_name/submap_000/ ...

    Returns:
        (vo_episode_dir, slam_episode_dir)
    """
    vo_ep = vo_output_dir / "VO" / episode_name
    slam_ep = vo_output_dir / "SLAM" / episode_name
    vo_ep.mkdir(parents=True, exist_ok=True)
    slam_ep.mkdir(parents=True, exist_ok=True)
    return vo_ep, slam_ep


# ============================================================================
# Config translation: step2_vo.yaml -> SLAMConfig
# ============================================================================

def _build_slam_config(
    vo_cfg: Dict[str, Any],
    slam_output_dir: Path,
    shared_K: Optional[np.ndarray],
    shared_K_hw: Optional[Tuple[int, int]],
) -> SLAMConfig:
    """Translate step2_vo.yaml sections into a SLAMConfig.

    Reads from ``vo_cfg["slam"]`` for SLAM-specific params (matching
    Pi3X-SLAM's native config structure), with ``vo_cfg["processing"]``
    and ``vo_cfg["model"]`` for shared params.
    """
    paths = vo_cfg.get('paths', {})
    model = vo_cfg.get('model', {})
    processing = vo_cfg.get('processing', {})
    slam = vo_cfg.get('slam', {})

    device = model.get('device', 'cuda')
    dtype = model.get('dtype', 'float16')
    pixel_limit = processing.get('pixel_limit', 255000)
    conf_threshold = processing.get('conf_threshold', 50.0)
    if conf_threshold < 1.0:
        conf_threshold = conf_threshold * 100.0
    conf_min_abs = processing.get('conf_min_abs', 0.0)

    waft_ckpt = slam.get('waft_ckpt_path') or paths.get('waft_ckpt_path', '')

    return SLAMConfig(
        ckpt_path=paths.get('ckpt_path', ''),
        device=device,
        dtype=dtype,
        pixel_limit=pixel_limit,
        submap_size=slam.get('submap_size', 5),
        overlap_window=slam.get('overlap_window', 1),
        max_submaps=slam.get('max_submaps', 0),
        max_timestamps=slam.get('max_timestamps', 0),
        conf_threshold=conf_threshold,
        conf_min_abs=conf_min_abs,
        keyframe_method=slam.get('keyframe_method', 'waft'),
        waft_ckpt_path=waft_ckpt or None,
        use_keyframe_selection=slam.get('use_keyframe_selection', True),
        min_disparity=slam.get('min_disparity', 50.0),
        shadow_keyframe_method=slam.get('shadow_keyframe_method', 'lk'),
        max_loops=slam.get('max_loops', 1),
        lc_retrieval_threshold=slam.get('lc_retrieval_threshold', 0.95),
        lc_conf_threshold=slam.get('lc_conf_threshold', 0.25),
        sim3_inlier_thresh=slam.get('sim3_inlier_thresh', 0.5),
        kf_save_debug_images=slam.get('kf_save_debug_images', False),
        rerun_save_path=slam.get('rerun_save_path', None),
        rerun_max_points=slam.get('rerun_max_points', 0),
        output_dir=str(slam_output_dir),
        shared_intrinsics=shared_K,
        shared_intrinsics_hw=shared_K_hw,
    )


# ============================================================================
# SLAM scene graph export (keyframe images by submap)
# ============================================================================

def _export_slam_outputs(
    slam_dir: Path,
    sp,
    all_frame_paths: list,
    use_symlinks: bool = True,
    undistort_dir: Path = None,
) -> dict:
    """Export SLAM scene graph + keyframe images into ``_slam/``.

    Creates::

        slam_dir/
        ├── scene_graph.json
        ├── KeyFrames/
        │   └── cam{N}/angle/<all cam/angle images at KF timestamps>
        └── submap_NNN/
            └── cam0/p+0_y+30_r+0/<keyframe images for this submap>

    KeyFrames/ collects images for ALL cameras and angles available in
    ``undistort_dir`` at keyframe timestamps -- not just the subset used
    by SLAM.  Falls back to SLAM-only images if ``undistort_dir`` is None.
    """
    slam_dir.mkdir(parents=True, exist_ok=True)

    sg = export_slam_scene_graph(sp.map, sp.graph, sp.stitch_records)

    for sm in sg["submaps"]:
        sm_dir = slam_dir / f"submap_{sm['submap_id']:03d}"
        sm_dir.mkdir(parents=True, exist_ok=True)
        for kf in sm["keyframes"]:
            origin = Path(kf["origin_path"])
            if not origin.is_file():
                continue
            m = _CAM_ANGLE_RE.search(kf["image_name"])
            if m:
                cam, angle = m.group(1), m.group(2)
                dst = sm_dir / cam / angle / kf["image_name"]
            else:
                dst = sm_dir / kf["image_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            if use_symlinks:
                try:
                    dst.symlink_to(origin)
                except OSError:
                    shutil.copy2(str(origin), str(dst))
            else:
                shutil.copy2(str(origin), str(dst))

    n_kf_images = sum(sm["num_keyframes"] for sm in sg["submaps"])
    logger.info("  Staged %d keyframe images across %d submaps",
                n_kf_images, len(sg["submaps"]))

    # Rename LC submap directories with _lc suffix for clarity
    for lc_sid in sg.get("lc_submap_ids", []):
        lc_dir = slam_dir / f"submap_{lc_sid:03d}"
        lc_dir_renamed = slam_dir / f"submap_{lc_sid:03d}_lc"
        if lc_dir.is_dir() and not lc_dir_renamed.exists():
            lc_dir.rename(lc_dir_renamed)
    n_lc_renamed = sum(
        1 for lc_sid in sg.get("lc_submap_ids", [])
        if (slam_dir / f"submap_{lc_sid:03d}_lc").is_dir()
    )
    if n_lc_renamed:
        logger.info("  Renamed %d LC submap dirs with _lc suffix", n_lc_renamed)

    # Unified KeyFrames/ folder: collect ALL cam/angle images at keyframe
    # timestamps from the undistort dir (not just SLAM cameras).
    kf_dir = slam_dir / "KeyFrames"
    kf_dir.mkdir(parents=True, exist_ok=True)
    n_kf_staged = 0

    kf_frame_indices = set()
    for sm in sg["submaps"]:
        for kf in sm["keyframes"]:
            if kf.get("frame_idx") is not None:
                kf_frame_indices.add(kf["frame_idx"])

    if undistort_dir and undistort_dir.is_dir():
        for cam_dir in sorted(undistort_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            for angle_dir in sorted(cam_dir.iterdir()):
                if not angle_dir.is_dir():
                    continue
                cam, angle = cam_dir.name, angle_dir.name
                for img in sorted(angle_dir.iterdir()):
                    if img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    m_idx = _FRAME_IDX_RE.match(img.name)
                    if not m_idx or int(m_idx.group(1)) not in kf_frame_indices:
                        continue
                    dst = kf_dir / cam / angle / img.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        continue
                    if use_symlinks:
                        try:
                            dst.symlink_to(img)
                        except OSError:
                            shutil.copy2(str(img), str(dst))
                    else:
                        shutil.copy2(str(img), str(dst))
                    n_kf_staged += 1
        logger.info("  KeyFrames/: %d images (%d timestamps × all cams/angles from %s)",
                    n_kf_staged, len(kf_frame_indices), undistort_dir)
    else:
        # Fallback: only SLAM keyframe images (no undistort_dir available)
        seen_kf = set()
        for sm in sg["submaps"]:
            for kf in sm["keyframes"]:
                img_name = kf["image_name"]
                if img_name in seen_kf:
                    continue
                seen_kf.add(img_name)
                origin = Path(kf["origin_path"])
                if not origin.is_file():
                    continue
                m = _CAM_ANGLE_RE.search(img_name)
                if m:
                    cam, angle = m.group(1), m.group(2)
                    dst = kf_dir / cam / angle / img_name
                else:
                    dst = kf_dir / img_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    continue
                if use_symlinks:
                    try:
                        dst.symlink_to(origin)
                    except OSError:
                        shutil.copy2(str(origin), str(dst))
                else:
                    shutil.copy2(str(origin), str(dst))
                n_kf_staged += 1
        logger.warning("  KeyFrames/: %d images (SLAM cams only, no undistort_dir)",
                       n_kf_staged)

    # Non-KF → KF association
    kf_frame_set = set()
    kf_frame_sorted = []
    for sm in sg["submaps"]:
        for kf in sm["keyframes"]:
            if kf["frame_idx"] is not None:
                kf_frame_set.add(kf["frame_idx"])
    kf_frame_sorted = sorted(kf_frame_set)

    all_frame_indices = set()
    for p in all_frame_paths:
        m = _FRAME_IDX_RE.match(os.path.basename(p))
        if m:
            all_frame_indices.add(int(m.group(1)))
    all_frames_sorted = sorted(all_frame_indices)

    frames = []
    for fidx in all_frames_sorted:
        if fidx in kf_frame_set:
            frames.append({"frame_idx": fidx, "is_keyframe": True})
        else:
            pos = bisect.bisect_left(kf_frame_sorted, fidx)
            prev_kf = kf_frame_sorted[pos - 1] if pos > 0 else None
            next_kf = kf_frame_sorted[pos] if pos < len(kf_frame_sorted) else None
            frames.append({
                "frame_idx": fidx,
                "is_keyframe": False,
                "prev_kf": prev_kf,
                "next_kf": next_kf,
            })

    sg["frames"] = frames
    sg["summary"]["num_total_frames"] = len(all_frames_sorted)
    sg["summary"]["num_keyframes"] = len(kf_frame_set)
    sg["summary"]["num_non_keyframes"] = len(all_frames_sorted) - len(kf_frame_set)

    sg_path = slam_dir / "scene_graph.json"
    with open(sg_path, "w", encoding="utf-8") as f:
        json.dump(sg, f, indent=2, ensure_ascii=False)
    logger.info("  Wrote %s", sg_path)

    return sg


# ============================================================================
# Debug-lite output (git-friendly alignment diagnostics)
# ============================================================================

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _extract_sim3_scale(T_s):
    """Extract SIM3 scale from a 4x4 [sR,t;0,1] matrix."""
    sR = T_s[:3, :3]
    return float(np.cbrt(np.linalg.det(sR)))


def _write_debug_lite(slam_ep_out, sp, slam_config_dict=None):
    """Write lightweight alignment diagnostics for git upload.

    Creates::

        slam_ep_out/_debug_lite/
        ├── config.json               # SLAM config snapshot
        ├── stitch_records.json        # every alignment record (SIM3/SL4)
        ├── submap_transforms.json     # per-submap T_s + extracted scale
        ├── submap_summary.json        # per-submap image lists, frame IDs
        ├── overlap_detection.json     # per-edge overlap frame pairs
        └── graph_factors.json         # per-factor post-opt error

    All files are small JSON — no binary data, no images.
    """
    import gtsam
    from gtsam.symbol_shorthand import X

    lite_dir = Path(slam_ep_out) / "_debug_lite"
    lite_dir.mkdir(parents=True, exist_ok=True)

    map_store = sp.map
    graph = sp.graph

    # ── 1. Config ──
    if slam_config_dict:
        safe_cfg = {}
        for k, v in slam_config_dict.items():
            if isinstance(v, np.ndarray):
                safe_cfg[k] = v.tolist()
            elif hasattr(v, '__dict__'):
                safe_cfg[k] = str(v)
            else:
                try:
                    json.dumps(v)
                    safe_cfg[k] = v
                except (TypeError, ValueError):
                    safe_cfg[k] = str(v)
        with open(lite_dir / "config.json", "w") as f:
            json.dump(safe_cfg, f, indent=2)

    # ── 2. Stitch records (raw from make_stitch_record) ──
    _dump = lambda obj, fh: json.dump(obj, fh, indent=2, cls=_NumpyEncoder)

    with open(lite_dir / "stitch_records.json", "w") as f:
        _dump(sp.stitch_records, f)

    shadow = getattr(sp, "shadow_records", None)
    if shadow:
        with open(lite_dir / "shadow_stitch_records.json", "w") as f:
            _dump(shadow, f)

    # ── 3. Per-submap T_s transforms + scale ──
    transforms = []
    for submap in map_store.ordered_submaps_by_key():
        sid = submap.get_id()
        is_lc = submap.get_lc_status()
        try:
            T_s = graph.get_submap_transform(sid)
            scale = _extract_sim3_scale(T_s)
            T_list = T_s.tolist()
        except Exception:
            T_list = None
            scale = None
        transforms.append({
            "submap_id": sid,
            "is_lc": bool(is_lc),
            "scale": scale,
            "T_s": T_list,
        })
    with open(lite_dir / "submap_transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

    # ── 4. Per-submap summary ──
    summaries = []
    for submap in map_store.ordered_submaps_by_key():
        sid = submap.get_id()
        basenames = [os.path.basename(n) for n in submap.img_names]
        fids = submap.get_frame_ids()
        n_pts_raw = 0
        n_pts_conf = 0
        if submap.pointclouds is not None:
            for idx in range(len(submap.pointclouds)):
                n_raw = int(np.prod(submap.pointclouds[idx].shape[:-1]))
                n_pts_raw += n_raw
                if submap.conf_masks is not None:
                    n_pts_conf += int(
                        (submap.conf_masks[idx] > submap.conf_threshold).sum())
        summaries.append({
            "submap_id": sid,
            "is_lc": bool(submap.get_lc_status()),
            "num_images": len(basenames),
            "image_basenames": basenames,
            "frame_ids": [int(f) for f in fids] if fids is not None else [],
            "conf_threshold": float(submap.conf_threshold)
                if submap.conf_threshold is not None else None,
            "points_raw": n_pts_raw,
            "points_after_conf": n_pts_conf,
        })
    with open(lite_dir / "submap_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    # ── 5. Overlap detection (re-derive from image names) ──
    seq_submaps = [s for s in summaries if not s["is_lc"]]
    overlap_info = []
    for i in range(len(seq_submaps) - 1):
        sa, sb = seq_submaps[i], seq_submaps[i + 1]
        names_a = set(sa["image_basenames"])
        shared = [n for n in sb["image_basenames"] if n in names_a]
        overlap_info.append({
            "submap_a": sa["submap_id"],
            "submap_b": sb["submap_id"],
            "num_images_a": sa["num_images"],
            "num_images_b": sb["num_images"],
            "shared_images": len(shared),
            "shared_basenames": shared,
        })
    # LC submaps: list which sequential submaps their images come from
    lc_submaps = [s for s in summaries if s["is_lc"]]
    for lcs in lc_submaps:
        connected = []
        for name in lcs["image_basenames"]:
            for ss in seq_submaps:
                if name in ss["image_basenames"]:
                    connected.append({
                        "image": name,
                        "seq_submap": ss["submap_id"],
                    })
                    break
        overlap_info.append({
            "lc_submap": lcs["submap_id"],
            "num_images": lcs["num_images"],
            "connections": connected,
        })
    with open(lite_dir / "overlap_detection.json", "w") as f:
        json.dump(overlap_info, f, indent=2)

    # ── 6. Graph factor errors (post-optimisation) ──
    factors = []
    try:
        for i in range(graph.graph.size()):
            factor = graph.graph.at(i)
            keys = []
            for k in factor.keys():
                keys.append(int(gtsam.symbolIndex(k)))
            try:
                err = float(factor.error(graph.values))
            except Exception:
                err = None
            factors.append({
                "factor_idx": i,
                "submap_keys": keys,
                "error": err,
            })
    except Exception as e:
        factors.append({"error": f"Could not iterate factors: {e}"})
    with open(lite_dir / "graph_factors.json", "w") as f:
        json.dump(factors, f, indent=2)

    logger.info("  Wrote _debug_lite/ (%d files) -> %s", 6, lite_dir)
    return lite_dir


# ============================================================================
# Pipeline report
# ============================================================================

def _write_pipeline_report(
    vo_output_dir: Path,
    episodes_processed: list,
    total_time: float,
    vo_cfg: Dict[str, Any],
    shared_K_loaded: bool,
):
    """Write ``pipeline_report.json`` at the output root."""
    report = {
        "module": "MOD_Pi3X_SLAM",
        "pipeline": "VO-en-SLAM",
        "timestamp": datetime.datetime.now().isoformat(),
        "total_time_s": total_time,
        "episodes_processed": len(episodes_processed),
        "episodes": episodes_processed,
        "shared_intrinsics": {
            "enabled": vo_cfg.get('shared_intrinsics', {}).get('enabled', False),
            "loaded": shared_K_loaded,
        },
        "slam_features": {
            "keyframe_selection": True,
            "loop_closure": True,
            "pose_graph_optimization": True,
        },
        "config_snapshot": {k: v for k, v in vo_cfg.items()
                           if k != 'shared_intrinsics'},
    }
    report_path = vo_output_dir / "pipeline_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", report_path)


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(config_path: str) -> int:
    """Run the VO-en-SLAM pipeline.

    1. Discover episodes + images  (essn_vo)
    2. Run Pi3X SLAM               (essn_slam)
    3. Export per-group COLMAP      (util_colmap)
    4. Export SLAM scene graph      (_slam/)
    5. Stage per-group images       (essn_vo)
    6. Write grouping manifest      (essn_vo)

    Args:
        config_path: Path to step2_vo.yaml.

    Returns:
        Exit code (0 = success).
    """
    t_start = time.time()

    # 1. Load config
    logger.info("Loading VO config from %s", config_path)
    vo_cfg = load_vo_config(config_path)
    paths_cfg = vo_cfg.get('paths', {})
    input_base = Path(paths_cfg['input_base'])
    grouping = vo_cfg.get('grouping', {})
    input_cfg = vo_cfg.get('input', {})
    filters_cfg = vo_cfg.get('filters', {})

    # VO cameras (for grouping + staging -- typically 2 stereo views)
    vo_cameras, vo_camera_angles = _parse_cameras(input_cfg)
    vo_stride = input_cfg.get('stride_frame', 1)

    # SLAM cameras (for reconstruction -- more angles for better coverage)
    # SLAM always uses stride=1: keyframe selection handles frame decimation.
    slam_input_cfg = vo_cfg.get('slam_input', {})
    if slam_input_cfg:
        slam_cameras, slam_camera_angles = _parse_cameras(slam_input_cfg)
    else:
        slam_cameras, slam_camera_angles = vo_cameras, vo_camera_angles

    group_size = grouping.get('group_size', 5)
    overlap = grouping.get('overlap_frames', 1)
    kf_window = grouping.get('keyframe_window', 3)

    logger.info("Input base: %s", input_base)
    logger.info("VO cameras: %s (stride=%d)", vo_cameras, vo_stride)
    logger.info("SLAM cameras: %s, angles: %s (stride=1, always)",
                slam_cameras, slam_camera_angles)
    logger.info("Group size: %d, overlap: %d, kf_window: ±%d",
                group_size, overlap, kf_window)

    # 2. Load shared intrinsics
    si_cfg = vo_cfg.get('shared_intrinsics', {})
    shared_K = None
    shared_K_hw = None
    if si_cfg.get('enabled', False):
        shared_K = load_shared_intrinsics(
            input_base=input_base,
            source=si_cfg.get('source', 'auto'),
            explicit_path=si_cfg.get('path'),
        )
        if shared_K is not None:
            yaml_path = input_base / "undistorted_intrinsics.yaml"
            if yaml_path.is_file():
                with open(yaml_path) as f:
                    intr_data = yaml.safe_load(f)
                shared_K_hw = (
                    int(intr_data.get('image_height', 0)),
                    int(intr_data.get('image_width', 0)),
                )
            logger.info("Shared K loaded, original_hw=%s", shared_K_hw)
        else:
            logger.warning(
                "shared_intrinsics.enabled=true but file not found; "
                "falling back to estimated intrinsics"
            )

    # 3. Create output directory
    vo_output_dir = _create_vo_output_dir(
        input_base, paths_cfg.get('output_base'))
    logger.info("Output: %s", vo_output_dir)

    # 4. Discover episodes (essn_vo)
    episodes = discover_episodes(input_base, filters_cfg.get('episodes'))
    if not episodes:
        logger.info("No episode dirs found; treating input_base as a single episode")
        episodes = [input_base]

    episodes_processed = []

    # 5. Process each episode
    for episode_dir in episodes:
        episode_name = episode_dir.name
        logger.info("=" * 60)
        logger.info("Processing episode: %s", episode_name)

        # 5a. Find image directory (essn_vo)
        image_root = find_undistort_dir(episode_dir)
        logger.info("Image root: %s", image_root)

        # 5b. Discover images for SLAM (multi-angle)
        timestamps, view_keys_or_paths, is_multi_view = \
            _discover_episode_images(
                image_root, slam_cameras, slam_camera_angles)

        if is_multi_view:
            n_images = sum(len(ts) for ts in timestamps)
            logger.info("SLAM: %d timestamps, %d total images (multi-view)",
                        len(timestamps), n_images)
        else:
            if not view_keys_or_paths:
                logger.warning("No images for episode %s, skipping", episode_name)
                continue
            n_images = len(view_keys_or_paths)
            logger.info("SLAM: %d images (flat)", n_images)

        # 5c. Create output directories (VO/ and SLAM/ side by side)
        vo_ep_out, slam_ep_out = _create_episode_outputs(
            vo_output_dir, episode_name)
        logger.info("VO output:   %s", vo_ep_out)
        logger.info("SLAM output: %s", slam_ep_out)

        # 5d. Build SLAM config (outputs go into SLAM/episode/)
        slam_config = _build_slam_config(
            vo_cfg, slam_ep_out, shared_K, shared_K_hw)

        # 5e. Run Pi3X SLAM (essn_slam)
        logger.info("Running Pi3X SLAM...")
        view_keys = view_keys_or_paths if is_multi_view else []
        slam = Pi3xSLAM(slam_config, view_keys=view_keys)

        if is_multi_view:
            slam.run_timestamps(timestamps)
        else:
            slam.run(view_keys_or_paths)

        # 5f. Export per-group COLMAP into VO/episode/group_NNN/sparse/
        logger.info("Exporting per-group COLMAP models...")
        sp = slam.submap_processor
        max_export_pts = vo_cfg.get('output', {}).get('max_export_pts', 500000)
        exported_groups = export_per_submap_colmap(
            str(vo_ep_out), sp.map, sp.graph, shared_K=shared_K,
            max_export_pts=max_export_pts)
        logger.info("Exported %d groups", len(exported_groups))

        # 5f-2. Export per-submap COLMAP into SLAM/episode/submap_NNN/sparse/
        logger.info("Exporting per-submap COLMAP models (SLAM)...")
        exported_submaps = export_per_submap_colmap_slam(
            str(slam_ep_out), sp.map, sp.graph, shared_K=shared_K,
            max_export_pts=max_export_pts)
        logger.info("Exported %d submaps", len(exported_submaps))

        # 5f-3. Export unified COLMAP model (all cameras + all points + PLY)
        logger.info("Exporting unified COLMAP model...")
        unified_colmap_dir = slam_ep_out / "colmap"
        unified_result = export_all_colmap(
            str(unified_colmap_dir), sp.map, sp.graph, shared_K=shared_K,
            max_export_pts=max_export_pts)
        if unified_result:
            logger.info("Unified model: %d images, %d points -> %s",
                        unified_result["num_images"],
                        unified_result["num_points"],
                        unified_result["path"])

        # 5g. Collect all frame paths
        all_paths = []
        if is_multi_view:
            for ts in timestamps:
                all_paths.extend(ts.values())
        else:
            all_paths = list(view_keys_or_paths)

        # 5h. Export SLAM scene graph + keyframe submap staging into SLAM/episode/
        logger.info("Exporting SLAM scene graph + keyframe manifest...")
        use_symlinks = vo_cfg.get('output', {}).get('use_symlinks', True)
        sg = _export_slam_outputs(
            slam_ep_out, sp, all_paths, use_symlinks=use_symlinks,
            undistort_dir=image_root)
        logger.info("  Scene graph: %d submaps, %d edges, %d LCs",
                     sg["summary"]["num_submaps"],
                     sg["summary"]["num_kf_edges"],
                     sg["summary"]["num_loop_closures"])

        # 5i. Debug-lite alignment diagnostics (git-friendly)
        _write_debug_lite(slam_ep_out, sp,
                          slam_config_dict=slam_config.__dict__
                          if hasattr(slam_config, '__dict__') else None)

        # 5j. Dual-edge registration (kern_dual_edge)
        kf_timeline = extract_kf_timeline(sg)
        kf_submap = kf_to_submap_map(sg)
        registration_results = {}
        if exported_groups and kf_timeline:
            logger.info("Computing dual-edge registration "
                        "(kf_window=±%d, %d keyframes)...",
                        kf_window, len(kf_timeline))
            registration_results = compute_dual_edges(
                exported_groups=exported_groups,
                scene_graph=sg,
                kf_timeline=kf_timeline,
                kf_to_submap=kf_submap,
                map_store=sp.map,
                graph=sp.graph,
                kf_window=kf_window,
            )
            print_dual_edge_summary(registration_results)
        else:
            logger.info("Skipping dual-edge registration "
                        "(groups=%d, kfs=%d)",
                        len(exported_groups), len(kf_timeline))

        # 5k. Stage per-group images from EPISODING into VO/episode/ (VO cameras only)
        undistort_dir = image_root
        staging_info = {}
        if undistort_dir and undistort_dir.is_dir():
            logger.info("Staging group images (VO cameras: %s) from %s",
                        vo_cameras, undistort_dir)
            staging_info = stage_group_images(
                vo_ep_out, undistort_dir, exported_groups,
                use_symlinks=use_symlinks,
                vo_cameras=vo_cameras,
                vo_camera_angles=vo_camera_angles,
            )
        else:
            logger.warning("Undistort dir not found; skipping image staging")

        # 5l. Write grouping manifest with dual-edge data into VO/episode/
        write_grouping_manifest(
            vo_ep_out, exported_groups, staging_info,
            undistort_dir, all_paths,
            group_size, overlap, vo_stride,
            kf_window=kf_window,
            kf_timeline=kf_timeline,
            registration_results=registration_results,
        )

        # 5m. Collect keyframe agreement stats
        kf_agreement = getattr(slam, 'kf_agreement', None)

        # 5n. Read back manifest for the report
        manifest_path = vo_ep_out / "grouping_manifest.json"
        manifest_data = None
        if manifest_path.is_file():
            with open(manifest_path) as f:
                manifest_data = json.load(f)

        # 5o. Generate HTML summary at top level
        logger.info("Generating pipeline_summary.html...")
        html_path = generate_pipeline_summary_html(
            output_dir=vo_output_dir,
            scene_graph=sg,
            exported_groups=exported_groups,
            registration_results=registration_results,
            manifest_data=manifest_data,
            slam_report=None,
            kf_agreement=kf_agreement,
            elapsed_s=time.time() - t_start,
            total_images=n_images,
        )
        logger.info("  HTML report: %s", html_path)

        # 5p. MASt3R remap (Tier 1: submaps, Tier 2: keyframes)
        remap_cfg = vo_cfg.get('remap', {})
        if remap_cfg.get('enabled', False):
            from kern_remap import remap_submaps, remap_keyframes
            _colmap_exe = remap_cfg.get('colmap_exe', 'colmap')
            _remap_image_size = remap_cfg.get('image_size', 512)
            _remap_min_matches = remap_cfg.get('min_num_matches', 15)
            _remap_timeout = remap_cfg.get('mapper_timeout', 300)

            # Load MASt3R model for remapping
            logger.info("Loading MASt3R model for remapping...")
            _mast3r_ckpt_dir = remap_cfg.get('mast3r_ckpt_dir', '')
            _mast3r_model_name = remap_cfg.get(
                'model_name',
                'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
            _mast3r_dir = remap_cfg.get('mast3r_dir', '')
            _dust3r_dir = remap_cfg.get('dust3r_dir', '')
            for _p in (_mast3r_dir, _dust3r_dir):
                if _p and _p not in sys.path:
                    sys.path.insert(0, _p)
            from mast3r.model import AsymmetricMASt3R
            _ckpt = str(Path(_mast3r_ckpt_dir) / (_mast3r_model_name + '.pth'))
            _remap_model = AsymmetricMASt3R.from_pretrained(_ckpt).eval().cuda()
            logger.info("MASt3R model loaded for remap")

            # Tier 1: submap remap
            logger.info("Running Tier 1 remap (submaps)...")
            t1_result = remap_submaps(
                slam_dir=slam_ep_out,
                model=_remap_model,
                colmap_exe=_colmap_exe,
                image_size=_remap_image_size,
                min_num_matches=_remap_min_matches,
                mapper_timeout=_remap_timeout,
            )
            logger.info("Tier 1 complete: %d submaps", t1_result["num_submaps"])

            # Tier 2: keyframes remap
            logger.info("Running Tier 2 remap (keyframes, all cams/angles)...")
            t2_result = remap_keyframes(
                slam_dir=slam_ep_out,
                model=_remap_model,
                colmap_exe=_colmap_exe,
                image_size=_remap_image_size,
                min_num_matches=_remap_min_matches,
                mapper_timeout=remap_cfg.get('kf_mapper_timeout', 600),
            )
            logger.info("Tier 2 complete: %s", "ok" if t2_result.get("success") else "failed")

            del _remap_model
            import torch
            torch.cuda.empty_cache()

        episodes_processed.append({
            "name": episode_name,
            "num_images": n_images,
            "num_groups": len(exported_groups),
            "num_submaps": sp.map.get_num_submaps(),
            "num_loop_closures": sp.graph.get_num_loops(),
            "dual_edge_summary": {
                "n_temporal": sum(1 for r in registration_results.values()
                                 if r["selected_edge"] == "temporal"),
                "n_slam_anchor": sum(1 for r in registration_results.values()
                                     if r["selected_edge"] == "submap_anchor"),
                "n_both": sum(1 for r in registration_results.values()
                              if r["selected_edge"] == "both"),
                "n_none": sum(1 for r in registration_results.values()
                              if r["selected_edge"] == "none"),
            } if registration_results else {},
        })

    # 6. Write pipeline report
    total_time = time.time() - t_start
    _write_pipeline_report(
        vo_output_dir, episodes_processed, total_time,
        vo_cfg, shared_K is not None)

    logger.info("=" * 60)
    logger.info("VO-en-SLAM pipeline complete in %.1fs", total_time)
    logger.info("  Episodes: %d", len(episodes_processed))
    logger.info("  Output: %s", vo_output_dir)
    logger.info("  Report: %s", vo_output_dir / "pipeline_summary.html")
    logger.info("=" * 60)

    return 0


# ============================================================================
# Entry point
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python BURNPIPE_VO_Pipeline.py <step2_vo.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)

    if not os.path.isfile(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    exit_code = run_pipeline(config_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
