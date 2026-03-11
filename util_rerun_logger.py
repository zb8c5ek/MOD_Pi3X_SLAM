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
    ):
        """Log camera poses, thumbnail images, and trajectory for one submap.

        Args:
            submap_id: Integer submap identifier.
            extrinsics: (S, 4, 4) cam2world matrices.
            images: (S, 3, H, W) image tensors (float 0-1 or uint8).
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        if not np.isfinite(extrinsics).all():
            print(f"[Rerun] WARNING: submap {submap_id} extrinsics contain NaN/inf, skipping")
            return

        color = self._color_for_submap(submap_id)
        rr.set_time("submap", sequence=submap_id)

        S = extrinsics.shape[0]
        centers = np.empty((S, 3), dtype=np.float32)

        for img_id in range(S):
            cam2world = extrinsics[img_id]  # 4x4
            entity = f"world/submap_{submap_id}/cam_{img_id:03d}"

            t = cam2world[:3, 3]
            R = cam2world[:3, :3]
            centers[img_id] = t

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
                    image_plane_distance=0.07,
                ),
            )
            rr.log(f"{entity}/image", rr.Image(img))

        if S >= 2:
            rr.log(
                f"world/traj_{submap_id}",
                rr.LineStrips3D(
                    [centers],
                    colors=[color],
                    radii=[0.004],
                ),
            )

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
