"""
util_rerun_logger - Rerun-based 3D visualization for Pi3X SLAM.

Replaces the Viser-based util_viewer.py.  Logs point clouds, camera
frustums with images, and OBB wireframes into a Rerun recording that
can be saved as .rrd for later playback or streamed live.

Usage from essn_submap::

    logger = RerunLogger(save_path="vis.rrd")  # or "" for live viewer
    logger.log_point_cloud("submap_0", points, colors)
    logger.log_submap_poses(submap_id=0, extrinsics=..., images=...)
"""

import numpy as np
import torch

try:
    import rerun as rr
    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False


class RerunLogger:
    """Thin wrapper around the Rerun SDK for SLAM visualization."""

    def __init__(self, save_path: str = "", application_id: str = "pi3x_slam"):
        """
        Args:
            save_path: Where to write the .rrd recording.
                       "" = spawn a live Rerun viewer window.
                       "path/to/file.rrd" = save to disk (no viewer).
            application_id: Rerun application identifier.
        """
        if not _HAS_RERUN:
            raise ImportError(
                "rerun-sdk is required for visualization.  "
                "Install with: pip install rerun-sdk"
            )

        rr.init(application_id, spawn=not save_path)
        if save_path:
            rr.save(save_path)
            print(f"[Rerun] Recording to {save_path}")
        else:
            print("[Rerun] Spawned live viewer")

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

        self._submap_colors = {}
        np.random.seed(100)
        self._palette = np.random.randint(0, 256, size=(250, 3), dtype=np.uint8)

    def _color_for_submap(self, submap_id: int) -> list:
        if submap_id not in self._submap_colors:
            idx = len(self._submap_colors) + 1
            self._submap_colors[submap_id] = self._palette[idx % len(self._palette)].tolist()
        return self._submap_colors[submap_id]

    # ------------------------------------------------------------------
    # Point clouds
    # ------------------------------------------------------------------

    def log_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        max_points: int = 0,
        voxel_size: float = 0.0,
    ):
        """Log a coloured point cloud under ``world/pcd_{name}``.

        Args:
            name: Entity name suffix (e.g. submap id).
            points: (N, 3) float32.
            colors: (N, 3) uint8.
            max_points: Downsample to this many points (0 = no limit).
            voxel_size: Open3D voxel downsampling size (0 = skip).
        """
        if voxel_size > 0:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(voxel_size)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        n = len(points)
        if max_points > 0 and n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
            idx.sort()
            points = points[idx]
            colors = colors[idx]

        rr.set_time("submap", sequence=int(name) if name.isdigit() else 0)
        rr.log(
            f"world/pcd_{name}",
            rr.Points3D(points, colors=colors, radii=np.full(len(points), 0.002)),
        )

    # ------------------------------------------------------------------
    # Camera poses + frustums
    # ------------------------------------------------------------------

    def log_submap_poses(
        self,
        submap_id: int,
        extrinsics: np.ndarray,
        images: np.ndarray,
        n_views: int = 1,
        view_keys: list = None,
    ):
        """Log camera poses, thumbnail images, and trajectory for one submap.

        Frames are interleaved as [cam0_kf0, cam1_kf0, cam0_kf1, cam1_kf1, ...]
        so ``n_views`` is used to de-interleave into keyframe × camera.

        Args:
            submap_id: Integer submap identifier.
            extrinsics: (S, 4, 4) cam2world matrices.
            images: (S, 3, H, W) image tensors (float 0-1 or uint8).
            n_views: Number of physical cameras per keyframe timestamp.
            view_keys: Optional camera names (length n_views).  Falls back
                to ``["cam0", "cam1", ...]``.
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        if not np.isfinite(extrinsics).all():
            print(f"[Rerun] WARNING: submap {submap_id} extrinsics contain NaN/inf, skipping")
            return

        color = self._color_for_submap(submap_id)
        rr.set_time("submap", sequence=submap_id)

        S = extrinsics.shape[0]
        n_v = max(n_views, 1)
        n_kf = S // n_v

        cam_names = list(view_keys) if view_keys and len(view_keys) == n_v else [
            f"cam{v}" for v in range(n_v)
        ]
        # Shorten long angle-encoded view keys (e.g. "cam0_p+0_y+30_r+0" -> "cam0")
        cam_labels = [k.split("_")[0] if "_" in k else k for k in cam_names]

        centers_per_cam = {v: [] for v in range(n_v)}

        for img_id in range(S):
            kf_idx = img_id // n_v
            view_idx = img_id % n_v
            cam_label = cam_labels[view_idx]

            if n_v > 1:
                entity = f"world/submap_{submap_id}/kf_{kf_idx:03d}/{cam_label}"
            else:
                entity = f"world/submap_{submap_id}/kf_{kf_idx:03d}"

            cam2world = extrinsics[img_id]  # 4x4
            t = cam2world[:3, 3]
            sR = cam2world[:3, :3]
            s = np.cbrt(np.linalg.det(sR))
            R = sR / max(abs(s), 1e-12)
            centers_per_cam[view_idx].append(t.astype(np.float32))

            rr.log(
                entity,
                rr.Transform3D(
                    translation=t,
                    mat3x3=R,
                ),
            )

            img = images[img_id]  # (3, H, W)
            if img.dtype in (np.float32, np.float64):
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            else:
                img = img.transpose(1, 2, 0)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fx = fy

            rr.log(
                f"{entity}/image",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    width=w,
                    height=h,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.07,
                ),
            )
            rr.log(f"{entity}/image", rr.Image(img))

        for v in range(n_v):
            pts = centers_per_cam[v]
            if len(pts) < 2:
                continue
            trajectory = np.stack(pts)
            cam_label = cam_labels[v]
            traj_entity = (f"world/traj_{submap_id}/{cam_label}"
                           if n_v > 1
                           else f"world/traj_{submap_id}")
            rr.log(
                traj_entity,
                rr.LineStrips3D(
                    [trajectory],
                    colors=[color],
                    radii=[0.004],
                ),
            )

    # ------------------------------------------------------------------
    # Unified trajectory across all submaps
    # ------------------------------------------------------------------

    def log_unified_trajectory(
        self,
        all_centers: list,
        n_views: int = 1,
        cam_labels: list = None,
    ):
        """Log a single continuous trajectory line across all submaps.

        Args:
            all_centers: List of (submap_id, centers_per_cam) tuples, where
                centers_per_cam is ``{view_idx: list_of_float32_xyz}``.
                Must be sorted by submap_id.
            n_views: Number of physical cameras per timestamp.
            cam_labels: Camera names (length n_views).
        """
        if not all_centers:
            return

        n_v = max(n_views, 1)
        labels = cam_labels or [f"cam{v}" for v in range(n_v)]

        for v in range(n_v):
            pts = []
            for _sid, cpc in all_centers:
                pts.extend(cpc.get(v, []))
            if len(pts) < 2:
                continue
            trajectory = np.stack(pts)
            entity = (f"world/trajectory_unified/{labels[v]}"
                      if n_v > 1
                      else "world/trajectory_unified")
            rr.log(
                entity,
                rr.LineStrips3D(
                    [trajectory],
                    colors=[[255, 255, 0]],
                    radii=[0.006],
                ),
            )

    def log_all_trajectories(
        self,
        all_centers: list,
        n_views: int = 1,
        cam_labels: list = None,
    ):
        """Re-log per-submap trajectory lines and the unified trajectory.

        Lightweight alternative to full ``log_submap_poses`` -- only updates
        line geometry, no images or Transform3D entities.

        Args:
            all_centers: Same format as ``log_unified_trajectory``.
            n_views: Number of physical cameras per timestamp.
            cam_labels: Camera names (length n_views).
        """
        if not all_centers:
            return

        n_v = max(n_views, 1)
        labels = cam_labels or [f"cam{v}" for v in range(n_v)]

        for sid, cpc in all_centers:
            color = self._color_for_submap(sid)
            rr.set_time("submap", sequence=sid)
            for v in range(n_v):
                pts = cpc.get(v, [])
                if len(pts) < 2:
                    continue
                trajectory = np.stack(pts)
                traj_entity = (f"world/traj_{sid}/{labels[v]}"
                               if n_v > 1
                               else f"world/traj_{sid}")
                rr.log(
                    traj_entity,
                    rr.LineStrips3D(
                        [trajectory],
                        colors=[color],
                        radii=[0.004],
                    ),
                )

        latest_sid = all_centers[-1][0] if all_centers else 0
        rr.set_time("submap", sequence=latest_sid)
        self.log_unified_trajectory(all_centers, n_views, labels)

    # ------------------------------------------------------------------
    # OBB wireframes
    # ------------------------------------------------------------------

    def log_obb(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        rotation: np.ndarray,
        color: tuple = (255, 0, 0),
        name: str = "",
    ):
        """Log an oriented bounding box as wireframe line segments."""
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
        ], dtype=np.float32)

        corners_world = (rotation @ corners_local.T).T + center

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        segments = np.array(
            [[corners_world[i], corners_world[j]] for i, j in edges],
            dtype=np.float32,
        )

        entity = f"world/obb_{name}" if name else "world/obb"
        rr.log(entity, rr.LineStrips3D(segments, colors=[color] * len(segments), radii=[0.002]))
