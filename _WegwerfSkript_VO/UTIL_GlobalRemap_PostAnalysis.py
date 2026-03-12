#!/usr/bin/env python3
"""
UTIL_GlobalRemap_PostAnalysis.py -- Rigorous analysis of GlobalRemap COLMAP output
==================================================================================

Loads the best COLMAP model from ``colmap_project/``, cross-references with
the MASt3R attempted pairs and COLMAP DB verified matches, and produces a
detailed registration report broken down by camera, angle, timestamp, and
pair type.

Optionally visualizes the result in Rerun with the 3D point cloud, registered
camera poses, and the SLAM trajectory overlaid for comparison.

Usage::

    cd D:\\RopediaGeoEngine
    micromamba run -n sfm3rV2 python ^
        MOD_Pi3X_SLAM\\_WegwerfSkript_VO\\UTIL_GlobalRemap_PostAnalysis.py ^
        E:\\...\\GlobalRemap_episode_001_full_t0000-0970_20260311_214411_soft-spire ^
        [--rerun] [--save viz.rrd]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("post_analysis")

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODULE_ROOT.parent
for p in (str(_PROJECT_ROOT), str(_MODULE_ROOT), str(_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Regex to parse image relpath: "cam0_p+0_y+30_r+0/000000_..._cam0_p+0_y+30_r+0.jpg"
_IMG_RE = re.compile(
    r"^(cam\d+)_(p[+-]\d+_y[+-]\d+_r[+-]\d+)/(\d{6})_"
)


# =============================================================================
# Data structures
# =============================================================================

class ImageInfo:
    """Parsed identity of a staged image."""
    __slots__ = ("relpath", "cam", "angle", "frame_idx", "subfolder")

    def __init__(self, relpath: str):
        self.relpath = relpath
        m = _IMG_RE.match(relpath)
        if not m:
            raise ValueError(f"Cannot parse image relpath: {relpath}")
        self.cam = m.group(1)
        self.angle = m.group(2)
        self.frame_idx = m.group(3)
        self.subfolder = f"{self.cam}_{self.angle}"


def parse_image_relpath(relpath: str) -> Optional[ImageInfo]:
    try:
        return ImageInfo(relpath)
    except ValueError:
        return None


# =============================================================================
# Data loading
# =============================================================================

def load_staged_images(staged_dir: Path) -> List[ImageInfo]:
    """Enumerate all images in staged_images/ and parse their identity."""
    images = []
    for sub in sorted(staged_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                relpath = f"{sub.name}/{f.name}"
                info = parse_image_relpath(relpath)
                if info:
                    images.append(info)
    return images


def load_reconstruction(sparse_dir: Path):
    """Load pycolmap reconstruction from sparse/0/."""
    import pycolmap
    return pycolmap.Reconstruction(str(sparse_dir))


def load_pairs(pairs_txt: Path) -> List[Tuple[str, str]]:
    """Load attempted pairs from pairs.txt."""
    pairs = []
    with open(pairs_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_db_image_ids(db_path: Path) -> Dict[str, int]:
    """Map image name -> COLMAP image_id from the database."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT image_id, name FROM images").fetchall()
    conn.close()
    return {name: iid for iid, name in rows}


def load_verified_pairs(db_path: Path) -> Set[Tuple[int, int]]:
    """Load verified pair IDs from two_view_geometries (rows > 0)."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"
    ).fetchall()
    conn.close()
    verified = set()
    for pair_id, nrows in rows:
        id1 = pair_id >> 32
        id2 = pair_id & 0xFFFFFFFF
        verified.add((min(id1, id2), max(id1, id2)))
    return verified


def load_all_db_pairs(db_path: Path) -> Dict[Tuple[int, int], int]:
    """Load all pairs from two_view_geometries with their inlier counts."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT pair_id, rows FROM two_view_geometries"
    ).fetchall()
    conn.close()
    result = {}
    for pair_id, nrows in rows:
        id1 = pair_id >> 32
        id2 = pair_id & 0xFFFFFFFF
        result[(min(id1, id2), max(id1, id2))] = nrows
    return result


# =============================================================================
# Section 1: Registration Breakdown
# =============================================================================

def analyze_registration(
    recon, staged_images: List[ImageInfo]
) -> Dict[str, Any]:
    """Analyze which images got registered, broken down by cam/angle/timestamp."""

    registered_names: Set[str] = set()
    for img in recon.images.values():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        registered_names.add(img.name)

    all_by_subfolder: Dict[str, List[ImageInfo]] = defaultdict(list)
    all_by_cam: Dict[str, List[ImageInfo]] = defaultdict(list)
    all_by_angle: Dict[str, List[ImageInfo]] = defaultdict(list)
    all_by_frame: Dict[str, List[ImageInfo]] = defaultdict(list)

    for info in staged_images:
        all_by_subfolder[info.subfolder].append(info)
        all_by_cam[info.cam].append(info)
        all_by_angle[info.angle].append(info)
        all_by_frame[info.frame_idx].append(info)

    def _count_reg(images: List[ImageInfo]) -> int:
        return sum(1 for i in images if i.relpath in registered_names)

    # Per cam+angle
    per_subfolder = {}
    for sf in sorted(all_by_subfolder):
        imgs = all_by_subfolder[sf]
        n_reg = _count_reg(imgs)
        per_subfolder[sf] = {
            "registered": n_reg, "total": len(imgs),
            "pct": round(100 * n_reg / max(len(imgs), 1), 1),
        }

    # Per cam
    per_cam = {}
    for cam in sorted(all_by_cam):
        imgs = all_by_cam[cam]
        n_reg = _count_reg(imgs)
        per_cam[cam] = {
            "registered": n_reg, "total": len(imgs),
            "pct": round(100 * n_reg / max(len(imgs), 1), 1),
        }

    # Per angle
    per_angle = {}
    for angle in sorted(all_by_angle):
        imgs = all_by_angle[angle]
        n_reg = _count_reg(imgs)
        per_angle[angle] = {
            "registered": n_reg, "total": len(imgs),
            "pct": round(100 * n_reg / max(len(imgs), 1), 1),
        }

    # Per timestamp
    per_timestamp = {}
    low_coverage_timestamps = []
    for fidx in sorted(all_by_frame):
        imgs = all_by_frame[fidx]
        n_reg = _count_reg(imgs)
        pct = round(100 * n_reg / max(len(imgs), 1), 1)
        per_timestamp[fidx] = {
            "registered": n_reg, "total": len(imgs), "pct": pct,
        }
        if pct < 25.0:
            low_coverage_timestamps.append(fidx)

    return {
        "total_staged": len(staged_images),
        "total_registered": len(registered_names),
        "pct_registered": round(100 * len(registered_names) / max(len(staged_images), 1), 1),
        "per_subfolder": per_subfolder,
        "per_cam": per_cam,
        "per_angle": per_angle,
        "per_timestamp": per_timestamp,
        "low_coverage_timestamps": low_coverage_timestamps,
        "registered_names": registered_names,
    }


# =============================================================================
# Section 2: 3D Point Contribution
# =============================================================================

def analyze_3d_contribution(recon) -> Dict[str, Any]:
    """Analyze 3D point contribution per image, track lengths, reproj errors."""

    # Per-image triangulated point counts
    per_image_triangulated: Dict[str, int] = {}
    for img in recon.images.values():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        n_tri = sum(1 for p2d in img.points2D if p2d.has_point3D())
        per_image_triangulated[img.name] = n_tri

    # Aggregate by subfolder
    by_subfolder: Dict[str, List[int]] = defaultdict(list)
    for name, n_tri in per_image_triangulated.items():
        info = parse_image_relpath(name)
        if info:
            by_subfolder[info.subfolder].append(n_tri)

    mean_tri_per_subfolder = {}
    for sf in sorted(by_subfolder):
        vals = by_subfolder[sf]
        mean_tri_per_subfolder[sf] = {
            "mean": round(np.mean(vals), 1),
            "median": round(float(np.median(vals)), 1),
            "min": int(np.min(vals)),
            "max": int(np.max(vals)),
            "count": len(vals),
        }

    # Track length distribution
    track_lengths = []
    reproj_errors = []
    for pt in recon.points3D.values():
        track_lengths.append(len(pt.track.elements))
        reproj_errors.append(pt.error)

    track_arr = np.array(track_lengths) if track_lengths else np.array([0])
    error_arr = np.array(reproj_errors) if reproj_errors else np.array([0.0])

    track_hist = {
        "mean": round(float(np.mean(track_arr)), 2),
        "median": round(float(np.median(track_arr)), 2),
        "p25": round(float(np.percentile(track_arr, 25)), 2),
        "p75": round(float(np.percentile(track_arr, 75)), 2),
        "p95": round(float(np.percentile(track_arr, 95)), 2),
        "max": int(np.max(track_arr)),
    }
    error_hist = {
        "mean": round(float(np.mean(error_arr)), 3),
        "median": round(float(np.median(error_arr)), 3),
        "p75": round(float(np.percentile(error_arr, 75)), 3),
        "p95": round(float(np.percentile(error_arr, 95)), 3),
    }

    return {
        "mean_triangulated_per_subfolder": mean_tri_per_subfolder,
        "track_length_stats": track_hist,
        "reproj_error_stats": error_hist,
        "num_points3d": len(recon.points3D),
    }


# =============================================================================
# Section 3: Pair Analysis
# =============================================================================

def classify_pair(info_a: ImageInfo, info_b: ImageInfo) -> str:
    """Classify a pair as temporal, stereo, or cross."""
    same_subfolder = (info_a.subfolder == info_b.subfolder)
    same_timestamp = (info_a.frame_idx == info_b.frame_idx)
    if same_subfolder and not same_timestamp:
        return "temporal"
    elif same_timestamp and not same_subfolder:
        return "stereo"
    else:
        return "cross"


def cam_pair_key(cam_a: str, cam_b: str) -> str:
    """Canonical cam pair key, e.g. 'cam0-cam1'."""
    return f"{min(cam_a, cam_b)}-{max(cam_a, cam_b)}"


def angle_pair_key(angle_a: str, angle_b: str) -> str:
    """Canonical angle pair key."""
    return f"{min(angle_a, angle_b)}--{max(angle_a, angle_b)}"


def analyze_pairs(
    attempted_pairs: List[Tuple[str, str]],
    db_path: Path,
    registered_names: Set[str],
) -> Dict[str, Any]:
    """Analyze attempted vs verified vs both-registered pairs."""

    name_to_id = load_db_image_ids(db_path)
    all_db_pairs = load_all_db_pairs(db_path)

    # Counters by geometry type
    by_geometry: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "attempted": 0, "verified": 0,
        "both_reg": 0, "one_reg": 0, "neither_reg": 0,
        "total_inliers": 0,
    })
    by_cam_pair: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "attempted": 0, "verified": 0,
        "both_reg": 0, "one_reg": 0, "neither_reg": 0,
        "total_inliers": 0,
    })
    by_angle_pair: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "attempted": 0, "verified": 0,
        "both_reg": 0, "one_reg": 0, "neither_reg": 0,
    })

    for path_a, path_b in attempted_pairs:
        info_a = parse_image_relpath(path_a)
        info_b = parse_image_relpath(path_b)
        if not info_a or not info_b:
            continue

        geom = classify_pair(info_a, info_b)
        cpk = cam_pair_key(info_a.cam, info_b.cam)
        apk = angle_pair_key(info_a.angle, info_b.angle)

        a_reg = path_a in registered_names
        b_reg = path_b in registered_names

        id_a = name_to_id.get(path_a)
        id_b = name_to_id.get(path_b)
        is_verified = False
        n_inliers = 0
        if id_a is not None and id_b is not None:
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            n_inliers = all_db_pairs.get(pair_key, 0)
            is_verified = n_inliers > 0

        for bucket in (by_geometry[geom], by_cam_pair[cpk]):
            bucket["attempted"] += 1
            if is_verified:
                bucket["verified"] += 1
                bucket["total_inliers"] += n_inliers
            if a_reg and b_reg:
                bucket["both_reg"] += 1
            elif a_reg or b_reg:
                bucket["one_reg"] += 1
            else:
                bucket["neither_reg"] += 1

        by_angle_pair[apk]["attempted"] += 1
        if is_verified:
            by_angle_pair[apk]["verified"] += 1
        if a_reg and b_reg:
            by_angle_pair[apk]["both_reg"] += 1
        elif a_reg or b_reg:
            by_angle_pair[apk]["one_reg"] += 1
        else:
            by_angle_pair[apk]["neither_reg"] += 1

    return {
        "by_geometry": dict(by_geometry),
        "by_cam_pair": dict(by_cam_pair),
        "by_angle_pair": dict(by_angle_pair),
    }


# =============================================================================
# Section 4: Connectivity Matrix
# =============================================================================

def build_connectivity_matrix(
    attempted_pairs: List[Tuple[str, str]],
    db_path: Path,
    registered_names: Set[str],
) -> Dict[str, Any]:
    """Build cam-cam connectivity matrix."""

    name_to_id = load_db_image_ids(db_path)
    verified_set = load_verified_pairs(db_path)

    all_cams: Set[str] = set()
    for pa, pb in attempted_pairs:
        ia, ib = parse_image_relpath(pa), parse_image_relpath(pb)
        if ia:
            all_cams.add(ia.cam)
        if ib:
            all_cams.add(ib.cam)
    cams = sorted(all_cams)

    matrix_verified: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in cams} for c in cams}
    matrix_both_reg: Dict[str, Dict[str, int]] = {c: {c2: 0 for c2 in cams} for c in cams}

    for path_a, path_b in attempted_pairs:
        info_a = parse_image_relpath(path_a)
        info_b = parse_image_relpath(path_b)
        if not info_a or not info_b:
            continue

        id_a = name_to_id.get(path_a)
        id_b = name_to_id.get(path_b)
        is_verified = False
        if id_a is not None and id_b is not None:
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            is_verified = pair_key in verified_set

        ca, cb = info_a.cam, info_b.cam
        if is_verified:
            matrix_verified[ca][cb] += 1
            matrix_verified[cb][ca] += 1

        a_reg = path_a in registered_names
        b_reg = path_b in registered_names
        if a_reg and b_reg:
            matrix_both_reg[ca][cb] += 1
            matrix_both_reg[cb][ca] += 1

    return {
        "cameras": cams,
        "verified_matches": matrix_verified,
        "both_registered": matrix_both_reg,
    }


# =============================================================================
# Console output
# =============================================================================

def print_registration_report(reg: Dict[str, Any]):
    """Print registration breakdown to console."""
    print("\n" + "=" * 70)
    print("SECTION 1: REGISTRATION BREAKDOWN")
    print("=" * 70)
    print(f"  Total staged: {reg['total_staged']}")
    print(f"  Total registered: {reg['total_registered']} ({reg['pct_registered']}%)")

    print(f"\n  {'Cam+Angle':<30s}  {'Reg':>5s} / {'Tot':>5s}  {'%':>7s}")
    print("  " + "-" * 55)
    for sf, d in sorted(reg["per_subfolder"].items()):
        print(f"  {sf:<30s}  {d['registered']:>5d} / {d['total']:>5d}  {d['pct']:>6.1f}%")

    print(f"\n  {'Camera':<30s}  {'Reg':>5s} / {'Tot':>5s}  {'%':>7s}")
    print("  " + "-" * 55)
    for cam, d in sorted(reg["per_cam"].items()):
        print(f"  {cam:<30s}  {d['registered']:>5d} / {d['total']:>5d}  {d['pct']:>6.1f}%")

    print(f"\n  {'Angle':<30s}  {'Reg':>5s} / {'Tot':>5s}  {'%':>7s}")
    print("  " + "-" * 55)
    for angle, d in sorted(reg["per_angle"].items()):
        print(f"  {angle:<30s}  {d['registered']:>5d} / {d['total']:>5d}  {d['pct']:>6.1f}%")

    # Timestamp coverage summary (not all 116 rows, just stats)
    ts_data = reg["per_timestamp"]
    ts_pcts = [v["pct"] for v in ts_data.values()]
    if ts_pcts:
        print(f"\n  Timestamp coverage ({len(ts_data)} timestamps):")
        print(f"    Mean:   {np.mean(ts_pcts):.1f}%")
        print(f"    Median: {np.median(ts_pcts):.1f}%")
        print(f"    Min:    {np.min(ts_pcts):.1f}%")
        print(f"    Max:    {np.max(ts_pcts):.1f}%")
        if reg["low_coverage_timestamps"]:
            print(f"    Low coverage (<25%): {len(reg['low_coverage_timestamps'])} timestamps")
            for fidx in reg["low_coverage_timestamps"][:10]:
                d = ts_data[fidx]
                print(f"      frame {fidx}: {d['registered']}/{d['total']} ({d['pct']}%)")
            if len(reg["low_coverage_timestamps"]) > 10:
                print(f"      ... and {len(reg['low_coverage_timestamps']) - 10} more")


def print_3d_contribution_report(contrib: Dict[str, Any]):
    """Print 3D point contribution analysis."""
    print("\n" + "=" * 70)
    print("SECTION 2: 3D POINT CONTRIBUTION")
    print("=" * 70)
    print(f"  Total 3D points: {contrib['num_points3d']}")

    tl = contrib["track_length_stats"]
    print(f"\n  Track length:  mean={tl['mean']}  median={tl['median']}  "
          f"p75={tl['p75']}  p95={tl['p95']}  max={tl['max']}")

    re_ = contrib["reproj_error_stats"]
    print(f"  Reproj error:  mean={re_['mean']}  median={re_['median']}  "
          f"p75={re_['p75']}  p95={re_['p95']}")

    print(f"\n  {'Cam+Angle':<30s}  {'Mean':>6s}  {'Med':>6s}  {'Min':>5s}  {'Max':>5s}  {'N':>4s}")
    print("  " + "-" * 65)
    for sf, d in sorted(contrib["mean_triangulated_per_subfolder"].items()):
        print(f"  {sf:<30s}  {d['mean']:>6.1f}  {d['median']:>6.1f}  "
              f"{d['min']:>5d}  {d['max']:>5d}  {d['count']:>4d}")


def print_pair_report(pair_stats: Dict[str, Any]):
    """Print pair analysis report."""
    print("\n" + "=" * 70)
    print("SECTION 3: PAIR ANALYSIS (Attempted -> Verified -> Both Registered)")
    print("=" * 70)

    def _print_table(title: str, data: Dict[str, Dict[str, int]]):
        print(f"\n  {title}:")
        print(f"  {'Key':<35s} {'Attempted':>9s} {'Verified':>9s} {'BothReg':>9s} "
              f"{'VerifRate':>9s} {'RegRate':>9s}")
        print("  " + "-" * 90)
        for key in sorted(data, key=lambda k: data[k].get("attempted", 0), reverse=True):
            d = data[key]
            att = d["attempted"]
            ver = d["verified"]
            br = d["both_reg"]
            vr = f"{100*ver/max(att,1):.1f}%" if att else "N/A"
            rr = f"{100*br/max(att,1):.1f}%" if att else "N/A"
            print(f"  {key:<35s} {att:>9d} {ver:>9d} {br:>9d} {vr:>9s} {rr:>9s}")

    _print_table("By pair geometry", pair_stats["by_geometry"])
    _print_table("By camera pair", pair_stats["by_cam_pair"])
    _print_table("By angle pair", pair_stats["by_angle_pair"])


def print_connectivity_report(conn: Dict[str, Any]):
    """Print inter-camera connectivity matrix."""
    print("\n" + "=" * 70)
    print("SECTION 4: INTER-CAMERA CONNECTIVITY MATRIX")
    print("=" * 70)
    cams = conn["cameras"]

    print("\n  Verified matches:")
    print(f"  {'':>8s}", end="")
    for c in cams:
        print(f"  {c:>8s}", end="")
    print()
    for ci in cams:
        print(f"  {ci:>8s}", end="")
        for cj in cams:
            print(f"  {conn['verified_matches'][ci][cj]:>8d}", end="")
        print()

    print("\n  Both-registered pairs:")
    print(f"  {'':>8s}", end="")
    for c in cams:
        print(f"  {c:>8s}", end="")
    print()
    for ci in cams:
        print(f"  {ci:>8s}", end="")
        for cj in cams:
            print(f"  {conn['both_registered'][ci][cj]:>8d}", end="")
        print()


# =============================================================================
# Section 5: Rerun Visualization
# =============================================================================

CAM_COLORS = {
    "cam0": [31, 119, 180, 255],
    "cam1": [255, 127, 14, 255],
    "cam2": [44, 160, 44, 255],
    "cam3": [214, 39, 40, 255],
}
UNREG_COLOR = [160, 160, 160, 80]
SUBMAP_COLORS = [
    [31, 119, 180, 255], [255, 127, 14, 255], [44, 160, 44, 255],
    [214, 39, 40, 255], [148, 103, 189, 255], [140, 86, 75, 255],
    [227, 119, 194, 255], [127, 127, 127, 255], [188, 189, 34, 255],
    [23, 190, 207, 255],
]


def build_frustum_lines(position, rotation_matrix, scale=0.08):
    """Build 8 line segments for a camera frustum pyramid."""
    hw, hh, d = scale * 0.5, scale * 0.4, scale
    corners_cam = np.array([[-hw, -hh, d], [hw, -hh, d], [hw, hh, d], [-hw, hh, d]])
    corners_world = (rotation_matrix @ corners_cam.T).T + position
    lines = []
    for c in corners_world:
        lines.append([position, c])
    for i in range(4):
        lines.append([corners_world[i], corners_world[(i + 1) % 4]])
    return np.array(lines)


def _get_cam_from_world(image):
    """Extract 4x4 cam-from-world matrix, handling pycolmap API variants."""
    cfw = image.cam_from_world
    if callable(cfw):
        rigid = cfw()
        m = rigid.matrix
        return m() if callable(m) else np.array(m)
    elif hasattr(cfw, "matrix"):
        m = cfw.matrix
        return m() if callable(m) else np.array(m)
    return np.array(cfw)


def run_rerun_visualization(
    recon,
    staged_images: List[ImageInfo],
    registered_names: Set[str],
    scene_graph_json: Optional[Path],
    save_path: str = "",
):
    """Visualize COLMAP reconstruction + SLAM trajectory in Rerun."""
    import rerun as rr

    rr.init("global_remap_analysis", spawn=not save_path)
    if save_path:
        rr.save(save_path)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # -- 3D point cloud --
    pts_xyz = []
    pts_rgb = []
    for pt in recon.points3D.values():
        pts_xyz.append(pt.xyz)
        pts_rgb.append(pt.color)
    if pts_xyz:
        pts_xyz = np.array(pts_xyz)
        pts_rgb = np.array(pts_rgb, dtype=np.uint8)
        rr.log("world/point_cloud",
               rr.Points3D(pts_xyz, colors=pts_rgb, radii=[0.003]),
               static=True)
        logger.info("Logged %d 3D points", len(pts_xyz))

    # -- Registered cameras (colored by cam, sorted by timestamp for trajectory) --
    cam_frame_poses: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for img in recon.images.values():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        info = parse_image_relpath(img.name)
        if not info:
            continue

        center = np.array(img.projection_center())
        T = _get_cam_from_world(img)
        R_world = np.linalg.inv(T[:3, :3])
        cam_frame_poses[info.cam].append((int(info.frame_idx), center, R_world))

        color = CAM_COLORS.get(info.cam, [200, 200, 200, 255])
        frustum = build_frustum_lines(center, R_world, scale=0.06)
        rr.log(f"world/cameras/{info.cam}/{info.subfolder}/{info.frame_idx}",
               rr.LineStrips3D(frustum, colors=[color] * len(frustum), radii=[0.001]),
               static=True)

    # Log per-cam trajectory as a line strip sorted by frame_idx
    for cam_name in sorted(cam_frame_poses):
        entries = sorted(cam_frame_poses[cam_name], key=lambda e: e[0])
        positions = np.array([e[1] for e in entries])
        color = CAM_COLORS.get(cam_name, [200, 200, 200, 255])
        if len(positions) >= 2:
            rr.log(f"world/cameras/{cam_name}/trajectory",
                   rr.LineStrips3D([positions], colors=[color], radii=[0.003]),
                   static=True)
        rr.log(f"world/cameras/{cam_name}/positions",
               rr.Points3D(positions, colors=[color] * len(positions), radii=[0.008]),
               static=True)

        # Time-varying camera pose for scrubbing
        for fidx, center, R_world in entries:
            rr.set_time_sequence("frame_idx", fidx)
            rr.log(f"world/cameras/{cam_name}/current",
                   rr.Points3D([center], colors=[color], radii=[0.015]))
            frustum = build_frustum_lines(center, R_world, scale=0.08)
            rr.log(f"world/cameras/{cam_name}/current_frustum",
                   rr.LineStrips3D(frustum, colors=[color] * len(frustum), radii=[0.002]))

    logger.info("Logged %d registered camera poses",
                sum(len(v) for v in cam_frame_poses.values()))

    # -- SLAM trajectory from scene graph --
    if scene_graph_json and scene_graph_json.is_file():
        with open(scene_graph_json, "r", encoding="utf-8") as f:
            sg = json.load(f)

        submaps = sg.get("submaps", [])
        all_slam_positions = []  # for unified trajectory line

        for i, submap in enumerate(submaps):
            sid = submap.get("submap_id", i)
            color = SUBMAP_COLORS[i % len(SUBMAP_COLORS)]
            kf_entries = []
            for kf in submap.get("keyframes", []):
                pose = kf.get("pose_cam2world_global")
                fidx = kf.get("frame_idx")
                if pose is None or fidx is None:
                    continue
                # pose is a 4x4 nested list: [[r00,r01,r02,tx],[r10,...],...]
                mat = np.array(pose)
                pos = mat[:3, 3]
                kf_entries.append((fidx, pos))

            kf_entries.sort(key=lambda e: e[0])

            if kf_entries:
                positions = np.array([e[1] for e in kf_entries])
                all_slam_positions.extend(kf_entries)

                if len(positions) >= 2:
                    rr.log(f"world/slam_trajectory/submap_{sid:03d}",
                           rr.LineStrips3D([positions], colors=[color], radii=[0.005]),
                           static=True)
                rr.log(f"world/slam_trajectory/submap_{sid:03d}/keyframes",
                       rr.Points3D(positions, colors=[color] * len(positions), radii=[0.012]),
                       static=True)

                # Time-varying SLAM pose for scrubbing alongside COLMAP cameras
                for fidx, pos in kf_entries:
                    rr.set_time_sequence("frame_idx", fidx)
                    rr.log(f"world/slam_trajectory/current",
                           rr.Points3D([pos], colors=[[255, 255, 0, 255]], radii=[0.02]))

        # Unified trajectory line across all submaps (sorted by frame_idx)
        if len(all_slam_positions) >= 2:
            all_slam_positions.sort(key=lambda e: e[0])
            unified = np.array([e[1] for e in all_slam_positions])
            rr.log("world/slam_trajectory/unified",
                   rr.LineStrips3D([unified], colors=[[255, 255, 0, 180]], radii=[0.002]),
                   static=True)

        logger.info("Logged SLAM trajectory (%d submaps, %d keyframes)",
                    len(submaps), len(all_slam_positions))

    logger.info("Rerun visualization complete")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-analysis of GlobalRemap COLMAP output",
    )
    parser.add_argument("global_remap_dir", type=str,
                        help="Path to GlobalRemap output directory")
    parser.add_argument("--scene-graph", type=str, default=None,
                        help="Path to scene_graph.json (auto-resolved if omitted)")
    parser.add_argument("--rerun", action="store_true",
                        help="Launch Rerun visualization")
    parser.add_argument("--save", type=str, default="",
                        help="Save Rerun .rrd to file instead of spawning viewer")
    args = parser.parse_args()

    remap_dir = Path(args.global_remap_dir).resolve()
    if not remap_dir.is_dir():
        logger.error("GlobalRemap dir not found: %s", remap_dir)
        return 1

    # Resolve paths
    colmap_project = remap_dir / "colmap_project"
    sparse_dir = colmap_project / "sparse" / "0"
    db_path = colmap_project / "database.db"
    staged_dir = remap_dir / "staged_images"
    pairs_txt = remap_dir / "mast3r_db" / "pairs.txt"
    summary_json = remap_dir / "global_remap_summary.json"

    for p, label in [(sparse_dir, "sparse model"), (db_path, "database"),
                     (staged_dir, "staged_images"), (pairs_txt, "pairs.txt")]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return 1

    # Auto-resolve scene graph
    scene_graph_json = None
    if args.scene_graph:
        scene_graph_json = Path(args.scene_graph).resolve()
    elif summary_json.is_file():
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        episode = summary.get("episode", "")
        vo_folder = remap_dir.parent
        sg_candidate = None
        for d in vo_folder.iterdir():
            if d.is_dir() and d.name.startswith("VO-en-SLAM"):
                sg_candidate = d / "SLAM" / episode / "scene_graph.json"
                break
        if sg_candidate and sg_candidate.is_file():
            scene_graph_json = sg_candidate
            logger.info("Auto-resolved scene graph: %s", scene_graph_json)

    logger.info("=" * 60)
    logger.info("GlobalRemap Post-Analysis")
    logger.info("=" * 60)
    logger.info("  Dir: %s", remap_dir)

    # -- Load data --
    logger.info("Loading reconstruction...")
    recon = load_reconstruction(sparse_dir)
    logger.info("  %d images, %d 3D points", recon.num_reg_images(), recon.num_points3D())

    logger.info("Loading staged images...")
    staged_images = load_staged_images(staged_dir)
    logger.info("  %d staged images", len(staged_images))

    logger.info("Loading pairs...")
    attempted_pairs = load_pairs(pairs_txt)
    logger.info("  %d attempted pairs", len(attempted_pairs))

    # -- Section 1: Registration --
    logger.info("Analyzing registration...")
    reg = analyze_registration(recon, staged_images)
    print_registration_report(reg)

    # -- Section 2: 3D contribution --
    logger.info("Analyzing 3D point contribution...")
    contrib = analyze_3d_contribution(recon)
    print_3d_contribution_report(contrib)

    # -- Section 3: Pair analysis --
    logger.info("Analyzing pairs...")
    pair_stats = analyze_pairs(attempted_pairs, db_path, reg["registered_names"])
    print_pair_report(pair_stats)

    # -- Section 4: Connectivity --
    logger.info("Building connectivity matrix...")
    connectivity = build_connectivity_matrix(
        attempted_pairs, db_path, reg["registered_names"])
    print_connectivity_report(connectivity)

    # -- Save JSON --
    output_json = {
        "registration": {k: v for k, v in reg.items() if k != "registered_names"},
        "contribution_3d": contrib,
        "pair_analysis": pair_stats,
        "connectivity": connectivity,
    }
    json_path = remap_dir / "post_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    logger.info("\nJSON report: %s", json_path)

    # -- Section 5: Rerun --
    if args.rerun or args.save:
        logger.info("Launching Rerun visualization...")
        run_rerun_visualization(
            recon, staged_images, reg["registered_names"],
            scene_graph_json, save_path=args.save,
        )

    print("\n" + "=" * 60)
    print("POST-ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  JSON: {json_path}")
    if args.rerun or args.save:
        print(f"  Rerun: {'saved to ' + args.save if args.save else 'viewer spawned'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
