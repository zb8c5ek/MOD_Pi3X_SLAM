"""
kern_map - GraphMap: high-level map storing all submaps.

Provides submap registration, image retrieval queries,
and export (poses, point clouds) functionality.
"""

import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from util_common import cosine_similarity


class GraphMap:
    def __init__(self):
        self.submaps = dict()
        self.non_lc_submap_ids = []

    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)

    def get_largest_key(self, ignore_loop_closure_submaps=False):
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys) if non_lc_keys else None
        return max(self.submaps.keys())

    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for idx, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)

            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]
            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        """Find the closest frame across all past submaps using L2 distance on SALAD descriptors."""
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue
            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue
            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for idx, embedding in enumerate(submap_embeddings):
                    score = torch.linalg.norm(embedding - query_vector)
                    scores.append(score.item())

                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]
                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(
                self.submaps[detected_loop.detected_submap_id].get_frame_at_index(
                    detected_loop.detected_submap_frame
                )
            )
        return frames

    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def get_all_submap_transforms(self, graph):
        """Return per-submap local-to-global SIM(3) transforms."""
        transforms = []
        for submap in self.ordered_submaps_by_key():
            transforms.append(graph.get_submap_transform(submap.get_id()))
        return np.stack(transforms)

    def get_all_cam2world_global(self, graph):
        """Return all cam2world poses in global frame, shape (N, 4, 4)."""
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, kitti_format=False):
        """Write cam2world poses to file (TUM or KITTI format)."""
        all_poses = self.get_all_cam2world_global(graph)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            count = 0
            for submap in self.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                for frame_index, frame_id in enumerate(frame_ids):
                    cam2world = all_poses[count]
                    count += 1
                    rot = cam2world[:3, :3]
                    t = cam2world[:3, 3]
                    x, y, z = t
                    if kitti_format:
                        output = cam2world.flatten()[:12]
                        output = np.array([float(frame_id), *output])
                    else:
                        quaternion = R.from_matrix(rot).as_quat()
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def write_points_to_file(self, graph, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame(graph)
            if pcd is None or len(pcd) == 0:
                continue
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        if not pcd_all:
            print("[kern_map] No points to write — all submaps returned empty.")
            return
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)
