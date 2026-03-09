"""
util_colmap - COLMAP format I/O utilities.

Functions for reading/writing COLMAP text format files:
- cameras.txt
- images.txt
- points3D.txt

Also provides high-level export helpers that iterate over a GraphMap
to produce a complete COLMAP model from the SLAM pipeline.
"""

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (qw, qx, qy, qz).
    
    Args:
        R: Rotation matrix (3, 3)
        
    Returns:
        List [qw, qx, qy, qz]
    """
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # returns [qx, qy, qz, qw]
    return [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]  # [qw, qx, qy, qz]


def write_cameras_txt(output_path, num_images, height, width, focal_length=None, shared_K=None):
    """
    Write cameras.txt in COLMAP text format.
    
    Args:
        output_path: Directory to write cameras.txt
        num_images: Number of cameras/images
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Focal length in pixels (estimated if None)
        shared_K: Optional (3, 3) real K matrix. When provided, fx, fy,
            cx, cy are taken from K instead of heuristic estimation.
    """
    if shared_K is not None:
        fx = float(shared_K[0, 0])
        fy = float(shared_K[1, 1])
        cx = float(shared_K[0, 2])
        cy = float(shared_K[1, 2])
    elif focal_length is not None:
        fx = fy = focal_length
        cx, cy = width / 2.0, height / 2.0
    else:
        fx = fy = max(height, width) * 1.2
        cx, cy = width / 2.0, height / 2.0

    filepath = os.path.join(output_path, 'cameras.txt')
    with open(filepath, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {num_images}\n")
        for i in range(num_images):
            f.write(f"{i} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def write_images_txt(output_path, poses, image_names):
    """
    Write images.txt in COLMAP text format.
    
    Args:
        output_path: Directory to write images.txt
        poses: Camera poses (N, 4, 4) - cam2world matrices
        image_names: List of image filenames
    """
    filepath = os.path.join(output_path, 'images.txt')
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}, mean observations per image: 0\n")
        
        for i, (pose, name) in enumerate(zip(poses, image_names)):
            R_c2w = pose[:3, :3]
            t_c2w = pose[:3, 3]
            
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
            tx, ty, tz = t_w2c
            
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}\n")
            f.write("\n")


def write_points3d_txt(output_path, points, colors=None):
    """
    Write points3D.txt in COLMAP text format.
    
    Args:
        output_path: Directory to write points3D.txt
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) in [0, 255], default gray if None
    """
    filepath = os.path.join(output_path, 'points3D.txt')
    with open(filepath, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        
        for i, pt in enumerate(points):
            x, y, z = pt
            if colors is not None:
                r, g, b = colors[i]
            else:
                r, g, b = 128, 128, 128
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} -1\n")


def write_colmap_txt(output_path, poses, image_names, points, colors, height, width,
                     focal_length=None, shared_K=None):
    """
    Write complete COLMAP text format output.
    
    Args:
        output_path: Directory to write COLMAP files
        poses: Camera poses (N, 4, 4) - cam2world matrices
        image_names: List of image filenames
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) in [0, 255]
        height: Image height
        width: Image width
        focal_length: Focal length (estimated if None)
        shared_K: Optional (3, 3) real K matrix for cameras.txt.
    """
    os.makedirs(output_path, exist_ok=True)
    write_cameras_txt(output_path, len(poses), height, width, focal_length, shared_K=shared_K)
    write_images_txt(output_path, poses, image_names)
    write_points3d_txt(output_path, points, colors)


# ---------------------------------------------------------------------------
# High-level pipeline exports
# ---------------------------------------------------------------------------

def export_all_colmap(colmap_output_path, map_store, graph, max_export_pts=500000):
    """Export all submap poses and points to COLMAP text format.

    Args:
        colmap_output_path: Directory to write COLMAP files. Skipped if None.
        map_store: GraphMap containing all submaps.
        graph: PoseGraph with optimised transforms.
        max_export_pts: Subsample points if total exceeds this.
    """
    if not colmap_output_path:
        return
    print(f"Exporting COLMAP to {colmap_output_path}")

    all_poses = []
    all_names = []
    all_points = []
    all_colors = []

    for submap in map_store.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses_world = submap.get_all_poses_world(graph)
        all_poses.append(poses_world)
        all_names.extend(submap.img_names)

        points = submap.get_points_in_world_frame(graph)
        if points is not None and len(points) > 0:
            all_points.append(points)
            all_colors.append(submap.get_points_colors())

    if not all_poses:
        print("No poses to export")
        return

    poses = np.concatenate(all_poses, axis=0)
    points = np.zeros((0, 3))
    colors = np.zeros((0, 3), dtype=np.uint8)

    H = W = 0
    first_submap = next(map_store.ordered_submaps_by_key())
    if first_submap.frames is not None:
        if isinstance(first_submap.frames, torch.Tensor):
            _, _, H, W = first_submap.frames.shape
        else:
            H, W = first_submap.frames.shape[2], first_submap.frames.shape[3]

    write_colmap_txt(colmap_output_path, poses, all_names, points, colors, H, W)
    print(f"COLMAP export: {len(poses)} images (points skipped for size)")


def export_per_submap_colmap(vo_results_dir, map_store, graph, shared_K=None,
                             max_export_pts=500000):
    """Export each submap as a separate COLMAP model for VO-EPISODING output.

    Creates the per-group directory structure expected by downstream modules:
        vo_results_dir/group_NNN/sparse/0/{cameras,images,points3D}.txt

    Args:
        vo_results_dir: Base directory (e.g. episode/_vo_results/run_XXX/).
        map_store: GraphMap containing all submaps.
        graph: PoseGraph with optimised transforms.
        shared_K: Optional (3, 3) real K matrix for cameras.txt.
        max_export_pts: Max points per group before subsampling.

    Returns:
        List of (group_idx, group_dir, img_full_paths) tuples.
        img_full_paths are the original full image paths from the submap,
        needed by callers for image staging and manifest generation.
    """
    if not vo_results_dir:
        return []

    exported = []
    group_idx = 0

    for submap in map_store.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue

        group_name = f"group_{group_idx:03d}"
        sparse_dir = os.path.join(vo_results_dir, group_name, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)

        poses_world = submap.get_all_poses_world(graph)
        img_full_paths = list(submap.img_names)
        img_names = [os.path.basename(n) for n in img_full_paths]

        points = submap.get_points_in_world_frame(graph)
        colors = submap.get_points_colors()
        if points is None:
            points = np.zeros((0, 3))
            colors = np.zeros((0, 3), dtype=np.uint8)
        elif len(points) > max_export_pts:
            idx = np.random.choice(len(points), max_export_pts, replace=False)
            points = points[idx]
            colors = colors[idx]

        H = W = 0
        if submap.frames is not None:
            if isinstance(submap.frames, torch.Tensor):
                _, _, H, W = submap.frames.shape
            else:
                H, W = submap.frames.shape[2], submap.frames.shape[3]

        write_colmap_txt(sparse_dir, poses_world, img_names, points, colors,
                         H, W, shared_K=shared_K)

        exported.append((group_idx, os.path.join(vo_results_dir, group_name),
                         img_full_paths))
        print(f"  [VO Export] {group_name}: {len(poses_world)} images, "
              f"{len(points)} points -> {sparse_dir}")
        group_idx += 1

    return exported


def export_poses(log_poses_path, map_store, graph):
    """Export optimised poses to a log file.

    Args:
        log_poses_path: Output file path. Skipped if None.
        map_store: GraphMap containing all submaps.
        graph: PoseGraph with optimised transforms.
    """
    if not log_poses_path:
        return
    map_store.write_poses_to_file(log_poses_path, graph)


# ---------------------------------------------------------------------------
# Scene graph + keyframe manifest export
# ---------------------------------------------------------------------------

import re as _re

_FRAME_IDX_RE = _re.compile(r"^(\d{6})_")
_TIMESTAMP_NS_RE = _re.compile(r"^\d+_\d+_(\d+)_cam")


def _parse_image_meta(basename):
    """Extract frame_idx and timestamp_ns from an image basename."""
    fidx = ts_ns = None
    m = _FRAME_IDX_RE.match(basename)
    if m:
        fidx = int(m.group(1))
    m = _TIMESTAMP_NS_RE.match(basename)
    if m:
        ts_ns = int(m.group(1))
    return fidx, ts_ns


def export_slam_scene_graph(map_store, graph, stitch_records=None):
    """Export the full SLAM scene graph organised by submap.

    Each submap entry carries its keyframes with:
      - frame index, timestamp (ns), origin file path
      - cam2world pose in both submap-local and global frame

    Edges are: intra-submap sequential, inter-submap overlap, loop closure.
    Registration quality comes from ``stitch_records``.

    Args:
        map_store: GraphMap with all submaps.
        graph: PoseGraph with optimised transforms.
        stitch_records: List of stitch record dicts
            (from ``kern_stitch_sim3.make_stitch_record``).

    Returns:
        Serialisable dict.
    """
    submaps_out = []
    group_idx = 0
    submap_id_to_group = {}
    lc_submaps = []
    img_name_to_submap = {}

    for submap in map_store.ordered_submaps_by_key():
        sid = submap.get_id()
        if submap.get_lc_status():
            lc_submaps.append(submap)
            continue

        group_name = f"group_{group_idx:03d}"
        submap_id_to_group[sid] = group_name

        poses_world = submap.get_all_poses_world(graph)
        poses_local = submap.get_all_poses()

        keyframes = []
        for i, img_path in enumerate(submap.img_names):
            basename = os.path.basename(img_path)
            fidx, ts_ns = _parse_image_meta(basename)

            pose_g = poses_world[i].tolist() if i < len(poses_world) else None
            pose_l = (poses_local[i].tolist()
                      if poses_local is not None and i < len(poses_local)
                      else None)

            keyframes.append({
                "index_in_submap": i,
                "frame_idx": fidx,
                "timestamp_ns": ts_ns,
                "image_name": basename,
                "origin_path": str(img_path),
                "pose_cam2world_global": pose_g,
                "pose_cam2world_local": pose_l,
            })
            img_name_to_submap[basename] = (group_name, sid, fidx)

        submaps_out.append({
            "submap_id": sid,
            "group": group_name,
            "num_keyframes": len(keyframes),
            "keyframes": keyframes,
        })
        group_idx += 1

    # ── KF edges ──

    kf_edges = []

    # Intra-submap: consecutive KFs share the Pi3X reconstruction.
    for sm in submaps_out:
        kfs = sm["keyframes"]
        for i in range(len(kfs) - 1):
            kf_edges.append({
                "type": "sequential",
                "kf_a": kfs[i]["frame_idx"],
                "kf_b": kfs[i + 1]["frame_idx"],
                "submap_id": sm["submap_id"],
                "group": sm["group"],
            })

    # Inter-submap overlap: shared image basenames.
    for i in range(len(submaps_out) - 1):
        sa, sb = submaps_out[i], submaps_out[i + 1]
        names_a = {kf["image_name"] for kf in sa["keyframes"]}
        shared = [kf for kf in sb["keyframes"] if kf["image_name"] in names_a]
        if shared:
            kf_edges.append({
                "type": "overlap",
                "group_a": sa["group"],
                "group_b": sb["group"],
                "submap_a": sa["submap_id"],
                "submap_b": sb["submap_id"],
                "shared_frame_indices": sorted({kf["frame_idx"] for kf in shared}),
                "num_shared": len(shared),
            })

    # Loop closure: trace LC submaps to their connected non-LC submaps.
    lc_edges = []
    for lc_sub in lc_submaps:
        connections = []
        for name in lc_sub.img_names:
            bn = os.path.basename(name)
            if bn in img_name_to_submap:
                connections.append(img_name_to_submap[bn])
        if len(connections) >= 2:
            lc_edge = {
                "type": "loop_closure",
                "group_a": connections[0][0],
                "group_b": connections[1][0],
                "submap_a": connections[0][1],
                "submap_b": connections[1][1],
                "kf_a": connections[0][2],
                "kf_b": connections[1][2],
                "lc_submap_id": lc_sub.get_id(),
            }
            lc_edges.append(lc_edge)
            kf_edges.append(lc_edge)

    # ── Registration quality ──

    registration = []
    if stitch_records:
        for rec in stitch_records:
            registration.append({
                "edge": rec.get("edge", ""),
                "is_loop_closure": rec.get("is_lc", False),
                "submap_prev": rec.get("submap_prev"),
                "submap_curr": rec.get("submap_curr"),
                "group_prev": submap_id_to_group.get(rec.get("submap_prev"), ""),
                "group_curr": submap_id_to_group.get(rec.get("submap_curr"), ""),
                "sim3_scale": rec.get("sim3_s"),
                "kabsch_rmsd": rec.get("kabsch_rmsd"),
                "rotation_deg": rec.get("rot_deg"),
                "translation_norm": rec.get("t_norm"),
                "num_inliers": rec.get("inliers"),
                "total_points": rec.get("total_pts"),
                "backend": rec.get("backend"),
            })

    return {
        "submaps": submaps_out,
        "kf_edges": kf_edges,
        "loop_closures": lc_edges,
        "registration": registration,
        "summary": {
            "num_submaps": len(submaps_out),
            "num_kf_edges": len(kf_edges),
            "num_loop_closures": len(lc_edges),
            "num_submaps_total": map_store.get_num_submaps(),
            "num_graph_nodes": graph.get_num_nodes(),
            "num_graph_loops": graph.get_num_loops(),
        },
    }
