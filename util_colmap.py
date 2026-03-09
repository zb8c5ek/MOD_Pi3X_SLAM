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
        List of (group_idx, group_dir) tuples that were exported.
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
        img_names = [os.path.basename(n) for n in submap.img_names]

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

        exported.append((group_idx, os.path.join(vo_results_dir, group_name)))
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
