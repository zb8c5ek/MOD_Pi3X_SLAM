"""
kern_submap - Submap data structure for batched frame storage.

Each Submap holds a group of keyframes processed together by Pi3X:
poses, dense point clouds, confidence maps, retrieval embeddings.

Pi3X outputs cam2world poses and points in a shared world frame *local*
to this submap.  The PoseGraph stores a single SIM(3) transform T_s per
submap that maps local → global:

    p_global = T_s @ p_local
    cam2world_global = T_s @ cam2world_local
"""

import re
import os
import torch
import numpy as np
import open3d as o3d


class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.poses = None          # (S, 4, 4) cam2world in submap-local frame
        self.frames = None         # (S, 3, H, W) image tensors
        self.proj_mats = None      # (S, 4, 4) K embedded in 4x4
        self.retrieval_vectors = None
        self.colors = None         # (S, H, W, 3) uint8
        self.conf = None           # (S, H, W)
        self.conf_masks = None     # (S, H, W)
        self.conf_threshold = None
        self.pointclouds = None    # (S, H, W, 3) submap-local coords
        self.voxelized_points = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None
        self.is_lc_submap = False
        self.img_names = []
        self.semantic_vectors = []

    def set_lc_status(self, is_lc_submap):
        self.is_lc_submap = is_lc_submap

    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics_4x4, conf_min_abs=0.0):
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        pct_thresh = np.percentile(self.conf, conf_threshold_percentile) + 1e-6
        self.conf_threshold = max(pct_thresh, conf_min_abs)
        self.proj_mats = intrinsics_4x4

    def set_img_names(self, img_names):
        self.img_names = img_names

    def add_all_frames(self, frames):
        self.frames = frames

    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def get_lc_status(self):
        return self.is_lc_submap

    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold

    def get_conf_masks_frame(self, index):
        return self.conf_masks[index]

    def get_frame_at_index(self, index):
        return self.frames[index, ...]

    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    def get_img_names_at_index(self, index):
        return self.img_names[index]

    def get_all_frames(self):
        return self.frames

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_all_poses(self):
        """Return raw cam2world poses in submap-local frame, shape (S, 4, 4)."""
        return self.poses

    # ------------------------------------------------------------------
    # World-frame queries (require graph for T_s)
    # ------------------------------------------------------------------

    def get_all_poses_world(self, graph):
        """Return cam2world poses in the global frame.

        For each frame i:  cam2world_global[i] = T_s @ cam2world_local[i]

        Args:
            graph: PoseGraph with optimised submap transforms.

        Returns:
            (S, 4, 4) cam2world poses in global frame.
        """
        T_s = graph.get_submap_transform(self.submap_id)
        poses_global = []
        for i in range(len(self.poses)):
            poses_global.append(T_s @ self.poses[i])
        return np.stack(poses_global, axis=0)

    def get_first_pose_world(self, graph):
        """First frame's cam2world in global frame."""
        T_s = graph.get_submap_transform(self.submap_id)
        return T_s @ self.poses[0]

    def get_last_pose_world(self, graph):
        """Last non-LC frame's cam2world in global frame."""
        T_s = graph.get_submap_transform(self.submap_id)
        idx = self.last_non_loop_frame_index
        return T_s @ self.poses[idx]

    def get_frame_pointcloud(self, pose_index):
        """Return raw point cloud for one frame in submap-local coords."""
        return self.pointclouds[pose_index]

    def get_points_in_world_frame(self, graph):
        """Return confidence-filtered points from all frames in global frame.

        Applies the single submap-level SIM(3) transform T_s to all points:
            p_global = T_s @ p_local

        Args:
            graph: PoseGraph with optimised submap transforms.

        Returns:
            (M, 3) array of filtered points in global frame, or None if empty.
        """
        T_s = graph.get_submap_transform(self.submap_id)
        sR = T_s[:3, :3]
        t = T_s[:3, 3]

        points_all = []
        n_total_pass = 0
        for index in range(len(self.pointclouds)):
            points = self.pointclouds[index]
            conf_mask = self.conf_masks[index] > self.conf_threshold
            n_pass = int(conf_mask.sum())
            n_total_pass += n_pass

            pts_local = points[conf_mask]  # (K, 3)
            if len(pts_local) > 0:
                pts_global = (sR @ pts_local.T).T + t
                points_all.append(pts_global)

        if n_total_pass == 0 and self.conf_masks is not None and len(self.conf_masks) > 0:
            cm0 = self.conf_masks[0]
            print(f"  [DBG submap {self.submap_id}] conf_threshold={self.conf_threshold:.6f} "
                  f"conf_masks[0] min={cm0.min():.6f} max={cm0.max():.6f} "
                  f"shape={cm0.shape} dtype={cm0.dtype}", flush=True)

        if points_all:
            return np.vstack(points_all)
        return None

    def get_points_list_in_world_frame(self, graph):
        """Return per-frame point clouds in global frame (for debug).

        Returns:
            (point_list, frame_id_list, frame_conf_mask)
        """
        T_s = graph.get_submap_transform(self.submap_id)
        sR = T_s[:3, :3]
        t = T_s[:3, 3]

        point_list = []
        frame_id_list = []
        frame_conf_mask = []
        for index in range(len(self.pointclouds)):
            points = self.pointclouds[index]
            pts_flat = points.reshape(-1, 3)
            pts_global = (sR @ pts_flat.T).T + t
            point_list.append(pts_global.reshape(points.shape))
            frame_id_list.append(self.frame_ids[index])
            conf_mask = self.conf_masks[index] > self.conf_threshold
            frame_conf_mask.append(conf_mask)
        return point_list, frame_id_list, frame_conf_mask

    # ------------------------------------------------------------------
    # Voxelized points
    # ------------------------------------------------------------------

    def get_voxel_points_in_world_frame(self, graph, voxel_size, nb_points=8,
                                        factor_for_outlier_rejection=2.0):
        """Return voxelized points in global frame."""
        if self.voxelized_points is None:
            if voxel_size <= 0.0:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")
            points = self.filter_data_by_confidence(self.pointclouds)
            points_flat = points.reshape(-1, 3)
            colors = self.filter_data_by_confidence(self.colors)
            colors_flat = colors.reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_flat)
            pcd.colors = o3d.utility.Vector3dVector(colors_flat)
            self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
            if nb_points > 0:
                self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(
                    nb_points=nb_points, radius=voxel_size * factor_for_outlier_rejection
                )

        # Transform voxelized local points to global frame
        T_s = graph.get_submap_transform(self.submap_id)
        sR = T_s[:3, :3]
        t = T_s[:3, 3]

        pts_local = np.asarray(self.voxelized_points.points)
        pts_global = (sR @ pts_local.T).T + t

        voxelized_pcd = o3d.geometry.PointCloud()
        voxelized_pcd.points = o3d.utility.Vector3dVector(pts_global)
        voxelized_pcd.colors = self.voxelized_points.colors
        return voxelized_pcd

    # ------------------------------------------------------------------
    # Setters and misc
    # ------------------------------------------------------------------

    def set_frame_ids(self, file_paths):
        """Extract numeric frame IDs from file paths."""
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks

    def set_all_semantic_vectors(self, semantic_vectors):
        self.semantic_vectors = semantic_vectors

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])

    def get_frame_ids(self):
        return self.frame_ids

    def filter_data_by_confidence(self, data):
        init_conf_mask = self.conf > self.conf_threshold
        return data[init_conf_mask]

    def get_points_colors(self):
        colors = self.filter_data_by_confidence(self.colors)
        return colors.reshape(-1, 3)

    def get_all_semantic_vectors(self):
        return self.semantic_vectors

    def get_points_in_mask(self, frame_index, mask, graph):
        points = self.get_points_list_in_world_frame(graph)[0][frame_index]
        points_flat = points.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        return points_flat[mask_flat]
