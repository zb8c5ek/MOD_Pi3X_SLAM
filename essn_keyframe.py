"""
essn_keyframe - Multi-view keyframe selection API.

Manages one optical-flow tracker per camera-angle view. At each timestamp,
computes per-view disparity independently (each view tracks against its own
last keyframe), then aggregates across views.

When a timestamp IS a keyframe, ALL view trackers update their reference.
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from kern_keyframe import (
    FrameTracker, WAFTFrameTracker, load_waft_model, WAFT_AVAILABLE,
)


class KeyframeSelector:
    """Multi-view keyframe selection.

    Usage (multi-view)::

        sel = KeyframeSelector("waft", waft_ckpt=..., view_keys=[...])
        for ts in timestamps:
            images = {vk: cv2.imread(ts[vk]) for vk in ts}
            is_kf, disps = sel.check_timestamp(images, min_disparity)
            if is_kf:
                paths = list(ts.values())  # all views at this timestamp

    Usage (single-view, backward compat)::

        sel = KeyframeSelector("lk")
        is_kf = sel.is_keyframe(cv2.imread(path), 50.0)
    """

    METHODS = ("lk", "waft")

    def __init__(
        self,
        method: str = "lk",
        waft_ckpt: str = None,
        device: str = "cuda",
        debug_dir: str = None,
        view_keys: List[str] = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.METHODS}")

        self._method = method
        self._device = device
        self._debug_dir = debug_dir
        self._waft_wrapper = None
        self._trackers: Dict[str, object] = {}

        if method == "waft":
            if not WAFT_AVAILABLE:
                raise RuntimeError("WAFT not installed. See 3rdParty/MANIFEST.md")
            if not waft_ckpt:
                raise ValueError("waft_ckpt required when method='waft'")
            print(f"[KF] Loading WAFT from {waft_ckpt}", flush=True)
            self._waft_wrapper = load_waft_model(waft_ckpt, device)

        if view_keys:
            for vk in view_keys:
                self._get_or_create_tracker(vk)

        if debug_dir:
            print(f"[KF] Debug images -> {debug_dir}", flush=True)
        print(f"[KF] '{method}' backend, {len(self._trackers)} views", flush=True)

    @property
    def method(self) -> str:
        return self._method

    # ------------------------------------------------------------------

    def _get_or_create_tracker(self, view_key: str):
        if view_key in self._trackers:
            return self._trackers[view_key]

        vk_dir = os.path.join(self._debug_dir, view_key) if self._debug_dir else None

        if self._method == "waft":
            t = WAFTFrameTracker(self._waft_wrapper, self._device,
                                 debug_dir=vk_dir, view_key=view_key)
        else:
            t = FrameTracker(debug_dir=vk_dir, view_key=view_key)

        self._trackers[view_key] = t
        return t

    # ------------------------------------------------------------------
    # Multi-view API
    # ------------------------------------------------------------------

    def check_timestamp(
        self,
        view_images: Dict[str, np.ndarray],
        min_disparity: float,
    ) -> Tuple[bool, Dict[str, float]]:
        """Evaluate one timestamp across all views.

        Args:
            view_images: {view_key: bgr_uint8} for every view at this timestamp.
            min_disparity: Pixel threshold. Timestamp is a keyframe when the
                MAX per-view disparity exceeds this.

        Returns:
            (is_keyframe, {view_key: disparity})
        """
        disparities: Dict[str, float] = {}
        flow_vis_map: Dict[str, object] = {}

        for vk, img in view_images.items():
            tracker = self._get_or_create_tracker(vk)
            disp, flow_vis = tracker.compute_disparity(img, min_disparity)
            disparities[vk] = disp
            flow_vis_map[vk] = flow_vis

        max_disp = max(disparities.values()) if disparities else 0.0

        first_timestamp = all(
            self._trackers[vk].last_kf_bgr is None
            for vk in view_images
        )
        is_kf = first_timestamp or (max_disp > min_disparity)

        for vk, img in view_images.items():
            tracker = self._trackers[vk]
            tracker.save_debug(img, disparities[vk], flow_vis_map[vk],
                               min_disparity, is_kf)

        if is_kf:
            for vk, img in view_images.items():
                self._trackers[vk].set_keyframe(img)

        return is_kf, disparities

    # ------------------------------------------------------------------
    # Single-view backward-compat API
    # ------------------------------------------------------------------

    def is_keyframe(self, image: np.ndarray, min_disparity: float) -> bool:
        return self.check_timestamp({"_default": image}, min_disparity)[0]
