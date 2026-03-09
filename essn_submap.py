"""
essn_submap - Submap processing, graph management, loop closure, and visualization.

Owns the Pi3X model, the GTSAM pose graph, the SALAD image retrieval system,
the Viser viewer, and the COLMAP exporter.  Processes one submap at a time
through the full reconstruction pipeline.

Extracted from essn_slam.py so the orchestrator (essn_slam.Pi3xSLAM) stays thin.

Delegates to:
  kern_inference      -- model loading, inference, Pi3X→Submap conversion
  kern_stitch_debug   -- per-stitch and cumulative debug artifacts
  util_colmap         -- COLMAP text format export
"""

from typing import Optional
import os
import time

import numpy as np
import torch
import open3d as o3d
from termcolor import colored

from kern_inference import (
    load_model, load_images_from_paths, run_inference, pi3x_to_submap_data,
)
from util_common import Accumulator, StageProfiler, compute_image_embeddings
from kern_submap import Submap
from kern_map import GraphMap
from kern_graph import PoseGraph
import kern_stitch_sim3
import kern_stitch_sl4
from kern_loop_closure import ImageRetrieval
from kern_stitch_debug import (
    save_stitch_debug, save_cumulative_debug, print_stitch_report,
)
from util_viewer import Viewer
from util_colmap import export_all_colmap, export_poses as _export_poses
from util_report import ESSNReport, setup_mod_logger
from util_step_report import generate_step_report


class SubmapProcessor:
    """Processes submap batches through the full SLAM backend.

    Pipeline per submap:
      1. Load & preprocess keyframe images
      2. Run Pi3X inference -> cam2world, world points, confidence
      3. Convert to Submap format (invert poses, estimate K, sigmoid conf)
      4. Compute SALAD embeddings for loop closure retrieval
      5. Detect loop closure candidates via L2 distance search
      6. Verify loop closures by running Pi3X on the 2-frame pair
      7. Add submap + edges to the GTSAM pose graph
      8. Optimize the graph
      9. Update 3D visualization
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        pixel_limit: int = 255000,
        patch_size: int = 14,
        conf_threshold: float = 50.0,
        conf_min_abs: float = 0.0,
        max_loops: int = 1,
        lc_retrieval_threshold: float = 0.95,
        lc_conf_threshold: float = 0.25,
        vis_voxel_size: Optional[float] = None,
        viewer_port: int = 8080,
        viewer_max_points: Optional[int] = 10000,
        colmap_output_path: Optional[str] = None,
        log_poses_path: Optional[str] = None,
        stitch_debug_dir: Optional[str] = None,
        sim3_inlier_thresh: float = 0.5,
        alignment_mode: str = "sim3",
        shadow_sl4: bool = False,
        shared_intrinsics=None,
        shared_intrinsics_hw=None,
    ):
        self.device = device
        self.dtype_torch = torch.float16 if dtype == "float16" else torch.bfloat16
        self.pixel_limit = pixel_limit
        self.patch_size = patch_size
        self.conf_threshold = conf_threshold
        self.conf_min_abs = conf_min_abs
        self.max_loops = max_loops
        self.lc_retrieval_threshold = lc_retrieval_threshold
        self.lc_conf_threshold = lc_conf_threshold
        self.vis_voxel_size = vis_voxel_size
        self.colmap_output_path = colmap_output_path
        self.log_poses_path = log_poses_path
        self.stitch_debug_dir = stitch_debug_dir
        self.viewer_max_points = viewer_max_points
        self.sim3_inlier_thresh = sim3_inlier_thresh
        self.alignment_mode = alignment_mode
        self.shadow_sl4 = shadow_sl4
        self.shared_intrinsics = shared_intrinsics
        self.shared_intrinsics_hw = shared_intrinsics_hw
        if stitch_debug_dir:
            os.makedirs(stitch_debug_dir, exist_ok=True)

        # Select stitch backend
        if alignment_mode == "sim3":
            self._stitch_backend = kern_stitch_sim3
        elif alignment_mode == "sl4":
            self._stitch_backend = kern_stitch_sl4
        else:
            raise ValueError(f"Unknown alignment_mode '{alignment_mode}'. Choose 'sim3' or 'sl4'.")

        # Load Pi3X model
        print("Loading Pi3X model...")
        self.model, load_time = load_model(ckpt_path, device)
        print(f"Pi3X model loaded in {load_time:.2f}s")

        # SLAM components
        self.viewer = Viewer(port=viewer_port)
        self.map = GraphMap()
        self.graph = PoseGraph(mode=alignment_mode)
        self.image_retrieval = ImageRetrieval(device=device)

        self.current_working_submap = None
        self._submap_seq = 0
        self.stitch_records = []   # per-stitch diagnostic records
        self.shadow_records = []   # shadow SL4 per-stitch records (when shadow_sl4=True)

        # Shadow trajectory: chain shadow T_rel into global transforms
        # Mirrors graph chaining but uses shadow SL4 alignment instead
        self._shadow_transforms = {}  # submap_id -> 4x4 global transform

        # Logger and report
        self.logger = setup_mod_logger("essn_submap", log_dir=stitch_debug_dir)
        self.report = ESSNReport("essn_submap")

        # Timers
        self.inference_timer = Accumulator()
        self.loop_closure_timer = Accumulator()
        self.clip_timer = Accumulator()

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _set_point_cloud(self, points, colors, name, point_size):
        if self.vis_voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(self.vis_voxel_size)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        pcd_name = "pcd_" + name
        handle = self.viewer.server.scene.add_point_cloud(
            name=pcd_name, points=points, colors=colors,
            point_size=point_size, point_shape="circle", precision="float32",
        )
        self.viewer.point_cloud_handles[pcd_name] = handle

    def _set_submap_point_cloud(self, submap):
        try:
            points = submap.get_points_in_world_frame(self.graph)
            colors = submap.get_points_colors()
            if points is None or len(points) == 0:
                print(f"[Viewer] WARNING: submap {submap.get_id()} has no points after filtering")
                return
            valid = np.isfinite(points).all(axis=1)
            if not valid.all():
                points = points[valid]
                colors = colors[valid]
            n = len(points)
            if self.viewer_max_points is None or self.viewer_max_points == 0:
                max_pts = n
            else:
                max_pts = min(n, max(int(self.viewer.gui_max_points.value), self.viewer_max_points))
            if n > max_pts:
                idx = np.random.choice(n, max_pts, replace=False)
                idx.sort()
                points = points[idx]
                colors = colors[idx]
                print(f"[Viewer] Downsampled submap {submap.get_id()}: {n} -> {max_pts} points")
            name = str(submap.get_id())
            print(f"[Viewer] Adding point cloud for submap {name}: {len(points)} points, range=[{points.min():.2f}, {points.max():.2f}]")
            self._set_point_cloud(points, colors, name, 0.002)
        except Exception as e:
            print(f"[Viewer] ERROR in point cloud for submap {submap.get_id()}: {e}")
            import traceback; traceback.print_exc()

    def _set_submap_poses(self, submap):
        try:
            extrinsics = submap.get_all_poses_world(self.graph)
            images = submap.get_all_frames()
            print(f"[Viewer] Visualizing {extrinsics.shape[0]} camera poses for submap {submap.get_id()}")
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())
        except Exception as e:
            print(f"[Viewer] ERROR in poses for submap {submap.get_id()}: {e}")
            import traceback; traceback.print_exc()

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self._set_submap_point_cloud(submap)
            self._set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self._set_submap_point_cloud(submap)
        self._set_submap_poses(submap)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def print_stitch_report(self):
        """Delegate to kern_stitch_debug.print_stitch_report."""
        print_stitch_report(self.stitch_records)

    # ------------------------------------------------------------------
    # Edge / graph management
    # ------------------------------------------------------------------

    @staticmethod
    def _find_overlap_pairs(prev_submap, curr_submap):
        """Find overlapping frame indices by matching image filenames.

        Returns:
            List of (prev_frame_idx, curr_frame_idx) for all shared images.
        """
        curr_name_to_idx = {name: i for i, name in enumerate(curr_submap.img_names)}
        pairs = []
        for prev_idx, prev_name in enumerate(prev_submap.img_names):
            if prev_name in curr_name_to_idx:
                pairs.append((prev_idx, curr_name_to_idx[prev_name]))
        return pairs

    def _aggregate_overlap_points(self, current_submap, prior_submap, overlap_pairs):
        """Aggregate confidence-filtered point correspondences from overlap pairs.

        Returns:
            (pts_curr, pts_prev) or (None, None) if insufficient points.
        """
        t0 = time.perf_counter()
        all_pts_curr = []
        all_pts_prev = []
        total_raw = 0
        total_kept = 0
        for prev_idx, curr_idx in overlap_pairs:
            current_conf = current_submap.get_conf_masks_frame(curr_idx)
            prior_conf = prior_submap.get_conf_masks_frame(prev_idx)
            good_mask = ((prior_conf > prior_submap.get_conf_threshold()) *
                         (current_conf > current_submap.get_conf_threshold())).reshape(-1)
            total_raw += len(good_mask)
            if np.sum(good_mask) < 10:
                good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                if np.sum(good_mask) < 10:
                    continue
            pts_c = current_submap.get_frame_pointcloud(curr_idx).reshape(-1, 3)[good_mask]
            pts_p = prior_submap.get_frame_pointcloud(prev_idx).reshape(-1, 3)[good_mask]
            total_kept += len(pts_c)
            all_pts_curr.append(pts_c)
            all_pts_prev.append(pts_p)

        if not all_pts_curr:
            return None, None
        t1 = time.perf_counter()
        result_curr = np.vstack(all_pts_curr)
        result_prev = np.vstack(all_pts_prev)
        t2 = time.perf_counter()
        print(f"    [AGGREGATE] {len(overlap_pairs)} frames, {total_raw} raw -> "
              f"{total_kept} kept | loop={t1-t0:.3f}s vstack={t2-t1:.3f}s")
        return result_curr, result_prev

    def _add_edge(self, submap_id_curr, submap_id_prev=None,
                  frame_id_curr=0, frame_id_prev=None,
                  overlap_pairs=None, is_loop_closure=False):
        """Add a submap node to the graph and align it to the previous submap.

        One graph node per submap.  The node stores T_s: local → global.

        For the first submap:  T_s = Identity, pinned with an anchor prior.
        For subsequent submaps:  alignment on overlapping points
        gives T_rel (curr-local → prev-local), then:
            T_curr = T_prev @ T_rel

        The alignment backend (SIM(3) or SL(4)) is selected by self.alignment_mode.
        When self.shadow_sl4 is True and the primary is SIM(3), the SL(4) backend
        also runs on the same points and prints a comparison (does not affect the graph).
        """
        assert not (is_loop_closure and submap_id_prev is None)
        current_submap = self.map.get_submap(submap_id_curr)

        if submap_id_prev is not None:
            prior_submap = self.map.get_submap(submap_id_prev)

            if overlap_pairs is None:
                overlap_pairs = [(frame_id_prev, frame_id_curr)]

            t0 = time.perf_counter()
            pts_curr, pts_prev = self._aggregate_overlap_points(
                current_submap, prior_submap, overlap_pairs)
            t_agg = time.perf_counter() - t0

            if pts_curr is None:
                print(colored(f"  [WARN] No valid overlap points for "
                              f"s{submap_id_prev}->s{submap_id_curr}", 'red'))
                if not is_loop_closure:
                    self.graph.add_submap_node(submap_id_curr, np.eye(4))
                return

            n_pts = len(pts_curr)
            n_overlap_frames = len(overlap_pairs)
            edge_label = 'LC' if is_loop_closure else f's{submap_id_prev}->s{submap_id_curr}'
            backend = self._stitch_backend
            print(f"  [TIMING] aggregate_points: {t_agg:.3f}s ({n_pts} pts from {n_overlap_frames} frames)")

            # --- Primary alignment ---
            align_kwargs = dict(inlier_thresh=self.sim3_inlier_thresh)
            if self.alignment_mode == "sl4":
                # Build full projection matrices: P = K @ inv(cam2world)
                # proj_mats stores K (intrinsics); cam2world from poses
                rep_prev_idx, rep_curr_idx = overlap_pairs[0]
                K_curr = current_submap.proj_mats[rep_curr_idx]
                K_prev = prior_submap.proj_mats[rep_prev_idx]
                pose_curr = current_submap.get_all_poses()[rep_curr_idx]
                pose_prev = prior_submap.get_all_poses()[rep_prev_idx]
                align_kwargs['proj_mat_curr'] = K_curr @ np.linalg.inv(pose_curr)
                align_kwargs['proj_mat_prev'] = K_prev @ np.linalg.inv(pose_prev)

            t0 = time.perf_counter()
            T_rel, info = backend.align_submaps(pts_curr, pts_prev, **align_kwargs)
            t_align = time.perf_counter() - t0
            print(f"  [TIMING] {self.alignment_mode}_alignment: {t_align:.3f}s")

            # Print diagnostics
            diag_str = backend.format_diagnostics(
                info, edge_label, n_overlap_frames, n_pts, is_loop_closure)
            print(colored(diag_str, 'cyan'))

            # Store record
            self.stitch_records.append(backend.make_stitch_record(
                info, edge_label, submap_id_prev, submap_id_curr,
                is_loop_closure, n_pts))

            # --- Shadow SL(4) comparison ---
            if self.shadow_sl4 and self.alignment_mode == "sim3":
                t0_shadow = time.perf_counter()
                shadow_T_rel = None
                try:
                    rep_prev_idx, rep_curr_idx = overlap_pairs[0]
                    K_c = current_submap.proj_mats[rep_curr_idx]
                    K_p = prior_submap.proj_mats[rep_prev_idx]
                    pose_c = current_submap.get_all_poses()[rep_curr_idx]
                    pose_p = prior_submap.get_all_poses()[rep_prev_idx]
                    sl4_kwargs = dict(
                        inlier_thresh=self.sim3_inlier_thresh,
                        proj_mat_curr=K_c @ np.linalg.inv(pose_c),
                        proj_mat_prev=K_p @ np.linalg.inv(pose_p),
                    )
                    shadow_T_rel, shadow_info = kern_stitch_sl4.align_submaps(
                        pts_curr, pts_prev, **sl4_kwargs)
                    shadow_diag = kern_stitch_sl4.format_diagnostics(
                        shadow_info, f"SHADOW {edge_label}",
                        n_overlap_frames, n_pts, is_loop_closure)
                    print(colored(shadow_diag, 'magenta'))

                    # Store shadow record
                    self.shadow_records.append(kern_stitch_sl4.make_stitch_record(
                        shadow_info, f"SHADOW {edge_label}",
                        submap_id_prev, submap_id_curr,
                        is_loop_closure, n_pts))

                    # Chain shadow trajectory: T_curr_shadow = T_prev_shadow @ shadow_T_rel
                    if not is_loop_closure and shadow_T_rel is not None:
                        T_prev_shadow = self._shadow_transforms.get(
                            submap_id_prev, np.eye(4))
                        self._shadow_transforms[submap_id_curr] = (
                            T_prev_shadow @ shadow_T_rel)
                except Exception as e:
                    print(colored(f"  [SHADOW SL4] Failed: {e}", 'red'))
                print(f"  [TIMING] shadow_sl4: {time.perf_counter()-t0_shadow:.3f}s")

            # Chain: T_curr_global = T_prev_global @ T_rel
            t0_graph = time.perf_counter()
            T_prev = self.graph.get_submap_transform(submap_id_prev)
            T_curr_global = T_prev @ T_rel

            if not is_loop_closure:
                self.graph.add_submap_node(submap_id_curr, T_curr_global)

                t0_debug = time.perf_counter()
                rep_prev, rep_curr = overlap_pairs[0]
                rep_conf_curr = current_submap.get_conf_masks_frame(rep_curr)
                rep_conf_prev = prior_submap.get_conf_masks_frame(rep_prev)
                rep_mask = ((rep_conf_prev > prior_submap.get_conf_threshold()) *
                            (rep_conf_curr > current_submap.get_conf_threshold())).reshape(-1)
                save_stitch_debug(
                    self.stitch_debug_dir, self._submap_seq,
                    submap_id_curr, rep_curr,
                    submap_id_prev, rep_prev,
                    info.get('s', 1.0), rep_mask, pts_curr, pts_prev,
                    self.map, self.graph)
                print(f"  [TIMING] stitch_debug_io: {time.perf_counter()-t0_debug:.3f}s")

            self.graph.add_between_factor(submap_id_prev, submap_id_curr, T_rel)
            print(f"  [TIMING] graph_ops: {time.perf_counter()-t0_graph:.3f}s")
        else:
            # First submap: identity transform, anchored
            T_identity = np.eye(4)
            self.graph.add_submap_node(submap_id_curr, T_identity)
            self.graph.add_anchor(submap_id_curr, T_identity)
            # Shadow trajectory: first submap is also identity
            if self.shadow_sl4:
                self._shadow_transforms[submap_id_curr] = T_identity.copy()

    # ------------------------------------------------------------------
    # Submap registration
    # ------------------------------------------------------------------

    def _add_points(self, data, detected_loops, lc_data=None):
        """Register a submap in the map and graph from Pi3X-converted data."""
        reg_prof = StageProfiler()
        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()

        with reg_prof("add_data"):
            self.current_working_submap.add_all_poses(data['cam2world'])
            self.current_working_submap.add_all_points(
                data['world_points'], data['colors'], data['conf'],
                self.conf_threshold, data['K_4x4'], self.conf_min_abs
            )
            self.current_working_submap.set_conf_masks(data['conf'])
            self.map.add_submap(self.current_working_submap)

        if submap_id_prev is not None:
            prev_submap = self.map.get_submap(submap_id_prev)
            with reg_prof("find_overlap"):
                overlap_pairs = self._find_overlap_pairs(prev_submap, self.current_working_submap)
            if overlap_pairs:
                print(f"  [OVERLAP] {len(overlap_pairs)} shared frames between "
                      f"submap {submap_id_prev} and {submap_id_curr}")
            else:
                print(colored(f"  [WARN] No overlapping images found by name between "
                              f"submap {submap_id_prev} and {submap_id_curr}. "
                              f"Falling back to last/first frame.", 'yellow'))
            with reg_prof("add_edge"):
                self._add_edge(submap_id_curr, submap_id_prev,
                               overlap_pairs=overlap_pairs if overlap_pairs else None,
                               frame_id_curr=0,
                               frame_id_prev=prev_submap.get_last_non_loop_frame_index(),
                               is_loop_closure=False)
        else:
            self._add_edge(submap_id_curr, is_loop_closure=False)

        reg_prof.print_summary(f"Registration s{submap_id_curr}")

        for index, loop in enumerate(detected_loops):
            if lc_data is None:
                continue
            assert loop.query_submap_id == self.current_working_submap.get_id()

            lc_submap_num = self.map.get_largest_key() + 1
            print(f"Creating loop closure submap with id {lc_submap_num}")
            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(lc_data['frames'])
            lc_submap.set_frame_ids(lc_data['frame_names'])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(lc_data['cam2world'])
            lc_submap.add_all_points(
                lc_data['world_points'], lc_data['colors'], lc_data['conf'],
                self.conf_threshold, lc_data['K_4x4'], self.conf_min_abs
            )
            lc_submap.set_conf_masks(lc_data['conf'])
            self.map.add_submap(lc_submap)

            # LC submap connects query → LC → detected
            self._add_edge(lc_submap_num, loop.query_submap_id,
                           frame_id_curr=0, frame_id_prev=loop.query_submap_frame,
                           is_loop_closure=False)
            self._add_edge(loop.detected_submap_id, lc_submap_num,
                           frame_id_curr=loop.detected_submap_frame, frame_id_prev=1,
                           is_loop_closure=True)

    # ------------------------------------------------------------------
    # Loop closure verification
    # ------------------------------------------------------------------

    def _verify_loop_closure(self, query_frame, retrieved_frame):
        """Verify a loop closure candidate by running Pi3X on the 2-frame pair.

        Uses mean sigmoid confidence as a proxy for geometric overlap:
        high confidence means the model successfully reconstructed both frames
        in a consistent coordinate system.

        Returns:
            (accepted, lc_data) -- lc_data is None when rejected.
        """
        lc_pair = torch.stack([query_frame, retrieved_frame])
        results, _ = run_inference(self.model, lc_pair, self.device, self.dtype_torch)

        mean_conf = torch.sigmoid(results['conf']).mean().item()
        print(f"  LC verification: mean confidence = {mean_conf:.4f} (threshold = {self.lc_conf_threshold})")

        if mean_conf < self.lc_conf_threshold:
            print(colored("  Loop closure rejected: low confidence", "red"))
            return False, None

        data = pi3x_to_submap_data(
            results, lc_pair,
            shared_intrinsics=self.shared_intrinsics,
            original_hw=self.shared_intrinsics_hw,
        )
        return True, data

    # ------------------------------------------------------------------
    # Process one submap
    # ------------------------------------------------------------------

    def process_submap(self, image_paths, clip_model=None, clip_preprocess=None):
        """Run the full SLAM pipeline for one submap batch.

        Args:
            image_paths: list of keyframe file paths for this submap.
            clip_model: optional CLIP model for semantic search.
            clip_preprocess: optional CLIP preprocessor.

        Returns:
            True if a loop closure was detected and verified.
        """
        device = self.device
        prof = StageProfiler()

        # 1. Load and preprocess images
        with prof("load_images"):
            with self.inference_timer:
                images, (H, W), _ = load_images_from_paths(
                    image_paths, self.pixel_limit, self.patch_size
                )
                images = images.to(device)
        print(f"[essn_submap] INFO: Loaded {len(image_paths)} images, shape: {images.shape}", flush=True)

        # 2. Create submap and compute embeddings
        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + 1

        print(f"Creating submap {new_pcd_num}")
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_paths)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)

        with prof("salad_embeddings"):
            new_submap.set_all_retrieval_vectors(
                self.image_retrieval.get_all_submap_embeddings(new_submap)
            )
        new_submap.set_img_names(image_paths)

        with prof("clip_embeddings"):
            with self.clip_timer:
                if clip_model is not None and clip_preprocess is not None:
                    image_embs = compute_image_embeddings(clip_model, clip_preprocess, image_paths)
                    new_submap.set_all_semantic_vectors(image_embs)

        self.current_working_submap = new_submap

        # 3. Run Pi3X inference
        with prof("pi3x_inference"):
            with self.inference_timer:
                results, inf_time = run_inference(self.model, images, device, self.dtype_torch)
        print(f"[essn_submap] INFO: Submap {new_pcd_num}: {len(image_paths)} images, "
              f"inference={inf_time:.2f}s", flush=True)

        # 4. Convert to submap format
        with prof("pi3x_to_submap"):
            data = pi3x_to_submap_data(
                results, images,
                shared_intrinsics=self.shared_intrinsics,
                original_hw=self.shared_intrinsics_hw,
            )

        conf = data['conf']
        pct_thresh = np.percentile(conf, self.conf_threshold)
        final_thresh = max(pct_thresh, self.conf_min_abs)
        n_survive = int((conf > final_thresh).sum())
        n_total = conf.size
        print(f"  [CONF] min={conf.min():.4f} med={np.median(conf):.4f} "
              f"max={conf.max():.4f} | p{self.conf_threshold:.0f}={pct_thresh:.4f} "
              f"floor={self.conf_min_abs} | thr={final_thresh:.4f} "
              f"| keep {n_survive}/{n_total} ({100*n_survive/n_total:.1f}%)",
              flush=True)

        # 5. Loop closure detection
        detected_loops = []
        lc_data = None
        if self.max_loops > 0:
            with prof("loop_closure_search"):
                with self.loop_closure_timer:
                    detected_loops = self.image_retrieval.find_loop_closures(
                        self.map, new_submap,
                        max_loop_closures=self.max_loops,
                        max_similarity_thres=self.lc_retrieval_threshold
                    )

            if detected_loops:
                print(colored(f"Loop closure candidates: {len(detected_loops)}", "yellow"))
                loop = detected_loops[0]
                query_frame = new_submap.get_frame_at_index(loop.query_submap_frame)
                retrieved_frame = self.map.get_frames_from_loops(detected_loops)[0]

                with prof("loop_closure_verify"):
                    accepted, lc_results = self._verify_loop_closure(query_frame, retrieved_frame)
                if accepted:
                    self.graph.increment_loop_closure()
                    lc_data = lc_results
                    lc_data['frames'] = torch.stack([query_frame, retrieved_frame])
                    lc_data['frame_names'] = [
                        new_submap.get_img_names_at_index(loop.query_submap_frame),
                        self.map.get_submap(loop.detected_submap_id).get_img_names_at_index(loop.detected_submap_frame)
                    ]
                else:
                    detected_loops = []

        # 6. Move tensors to CPU and free GPU
        with prof("gpu_cleanup"):
            for key in results:
                if isinstance(results[key], torch.Tensor):
                    results[key] = results[key].cpu()
            torch.cuda.empty_cache()

        # 7. Add submap to map + graph (includes SIM3 alignment)
        with prof("submap_registration"):
            self._add_points(data, detected_loops, lc_data)

        # 8. Optimize pose graph
        with prof("graph_optimize"):
            self.graph.optimize()

        # 8b. Stitch debug: cumulative point cloud
        self._submap_seq += 1
        with prof("stitch_debug_io"):
            save_cumulative_debug(self.stitch_debug_dir, self._submap_seq, self.map, self.graph)

        # 8c. Per-step HTML report with 3D trajectories
        with prof("step_report"):
            report_path = generate_step_report(
                step_idx=self._submap_seq,
                essn_name="essn_submap",
                output_dir=os.path.join(self.stitch_debug_dir, "step_reports")
                    if self.stitch_debug_dir else None,
                graph=self.graph,
                map_store=self.map,
                stitch_records=self.stitch_records,
                shadow_records=self.shadow_records if self.shadow_sl4 else None,
                shadow_transforms=self._shadow_transforms if self.shadow_sl4 else None,
            )
            if report_path:
                print(f"  [REPORT] {report_path}")

        # 9. Update visualization
        loop_detected = len(detected_loops) > 0
        with prof("viewer_update"):
            if loop_detected:
                self.update_all_submap_vis()
            else:
                self.update_latest_submap_vis()

        # 10. Print profiling summary
        prof.print_summary(f"Submap {new_pcd_num}")

        # 11. Update ESSN report
        self.report.set_metric("submaps_processed", self._submap_seq)
        self.report.set_metric("total_inference_time_s", self.inference_timer.total_time)
        self.report.set_metric("total_loop_closures", self.graph.get_num_loops())
        self.report.set_metric("total_submaps_in_map", self.map.get_num_submaps())
        self.logger.info(
            f"Submap {new_pcd_num}: {len(image_paths)} images, "
            f"inference={inf_time:.2f}s, loop_detected={loop_detected}"
        )

        return loop_detected

    # ------------------------------------------------------------------
    # Exports (delegated to util_colmap)
    # ------------------------------------------------------------------

    def export_colmap(self):
        """Export all submap poses and points to COLMAP text format."""
        export_all_colmap(self.colmap_output_path, self.map, self.graph)

    def export_poses(self):
        """Export optimized poses to a log file."""
        _export_poses(self.log_poses_path, self.map, self.graph)
