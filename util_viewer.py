"""
kern_viewer - Viser-based 3D visualization server.

Renders camera frustums with images, point clouds with adjustable
display settings, OBB wireframes, and walkthrough animation.

Ported from vggt_slam/viewer.py.
"""

import time
from typing import Dict, List
import numpy as np
import torch
import viser
import viser.transforms as viser_tf


class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(
            titlebar_content=None,
            control_layout="collapsible",
            dark_mode=True,
        )
        self.server.scene.configure_default_lights(enabled=True, cast_shadow=True)
        self.server.scene.set_up_direction("-y")

        self.gui_point_size = self.server.gui.add_slider(
            "Point Size", min=0.0005, max=0.02, step=0.0005, initial_value=0.002
        )
        self.gui_point_size.on_update(self._on_update_point_size)

        self.gui_max_points = self.server.gui.add_slider(
            "Max Points / Submap", min=1000, max=200000, step=1000, initial_value=10000
        )

        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        self.btn_walkthrough = self.server.gui.add_button("Play Walkthrough")
        self.btn_walkthrough.on_click(lambda _: self.run_walkthrough())

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}
        self.point_cloud_handles: Dict[str, viser.PointCloudHandle] = {}

        num_rand_colors = 250
        np.random.seed(100)
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)
        self.submap_id_to_color = dict()
        self.obj_id = 0

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 4, 4) cam2world matrices
        images_:    (S, 3, H, W) image tensors
        """
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if not np.isfinite(extrinsics).all():
            print(f"[Viewer] WARNING: submap {submap_id} extrinsics contain NaN/inf, skipping")
            return

        if submap_id not in self.submap_frames:
            next_id = len(self.submap_id_to_color) + 1
            self.submap_id_to_color[submap_id] = next_id
        self.submap_frames[submap_id] = []
        self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.07,
                image=img,
                format="jpeg",
                jpeg_quality=95,
                line_width=2.0,
                color=self.random_colors[self.submap_id_to_color[submap_id]]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

    def _on_update_point_size(self, _) -> None:
        new_size = self.gui_point_size.value
        for handle in self.point_cloud_handles.values():
            handle.point_size = new_size

    def visualize_obb(self, center, extent, rotation, color=(255, 0, 0), line_width=2.0):
        """Visualize an oriented bounding box as wireframe."""
        dx, dy, dz = extent / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz],
        ], dtype=np.float32)

        corners_world = (rotation @ corners_local.T).T + center

        edges_idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        segments = []
        for (i, j) in edges_idx:
            segments.append(corners_world[i])
            segments.append(corners_world[j])
        segments = np.array(segments, dtype=np.float32).reshape(-1, 2, 3)

        name = f"obb_{self.obj_id}"
        self.obj_id += 1
        self.server.scene.add_line_segments(
            name=name, points=segments, colors=color,
            line_width=line_width, visible=True
        )

    def run_walkthrough(self, fps: float = 20.0):
        """Walk through all frames in chronological order."""
        sorted_submap_ids = sorted(self.submap_frames.keys())
        if not sorted_submap_ids:
            print("No frames found to walk through.")
            return

        clients = self.server.get_clients()
        if not clients:
            print("No clients connected to perform walkthrough.")
            return

        print("Starting walkthrough of updated poses...")
        for sub_id in sorted_submap_ids:
            frames = self.submap_frames[sub_id]
            for frame_handle in frames:
                current_pos = frame_handle.position
                current_wxyz = frame_handle.wxyz
                for client in clients.values():
                    client.camera.position = current_pos
                    client.camera.wxyz = current_wxyz
                time.sleep(1.0 / fps)
