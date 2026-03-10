"""
essn_keyframe - Multi-view keyframe selection API.

Manages one optical-flow tracker per camera-angle view. At each timestamp,
computes per-view disparity independently (each view tracks against its own
last keyframe), then aggregates across views.

When a timestamp IS a keyframe, ALL view trackers update their reference.

Shadow mode
-----------
When ``shadow_method`` is set, a parallel set of trackers runs the alternate
backend on every frame.  The shadow trackers observe the same images and
follow the same keyframe cadence (they update their reference whenever the
primary decides "keyframe") but their disparity values are *not* used for the
actual keyframe decision.  After all timestamps have been processed, call
``get_agreement_stats()`` to get a summary comparing primary vs shadow.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from kern_keyframe import (
    FrameTracker, WAFTFrameTracker, load_waft_model, WAFT_AVAILABLE,
)


class KeyframeSelector:
    """Multi-view keyframe selection with optional shadow comparison.

    Usage (multi-view)::

        sel = KeyframeSelector("waft", waft_ckpt=..., shadow_method="lk",
                               view_keys=[...])
        for ts in timestamps:
            images = {vk: cv2.imread(ts[vk]) for vk in ts}
            is_kf, disps = sel.check_timestamp(images, min_disparity)
            if is_kf:
                paths = list(ts.values())

        stats = sel.get_agreement_stats()

    Usage (single-view, backward compat)::

        sel = KeyframeSelector("lk")
        is_kf = sel.is_keyframe(cv2.imread(path), 50.0)
    """

    METHODS = ("lk", "waft")

    def __init__(
        self,
        method: str = "waft",
        waft_ckpt: str = None,
        device: str = "cuda",
        debug_dir: str = None,
        view_keys: List[str] = None,
        shadow_method: Optional[str] = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {self.METHODS}")

        self._method = method
        self._device = device
        self._debug_dir = debug_dir
        self._waft_wrapper = None
        self._trackers: Dict[str, object] = {}

        # Shadow comparison state
        self._shadow_method = shadow_method if shadow_method != method else None
        self._shadow_waft_wrapper = None
        self._shadow_trackers: Dict[str, object] = {}
        self._agreement_log: List[Dict] = []

        # Load primary WAFT model if needed
        if method == "waft":
            if not WAFT_AVAILABLE:
                raise RuntimeError("WAFT not installed. See 3rdParty/MANIFEST.md")
            if not waft_ckpt:
                raise ValueError("waft_ckpt required when method='waft'")
            print(f"[KF] Loading WAFT from {waft_ckpt}", flush=True)
            self._waft_wrapper = load_waft_model(waft_ckpt, device)

        # Load shadow WAFT model if shadow is WAFT (and primary is not)
        if self._shadow_method == "waft":
            if not WAFT_AVAILABLE:
                print("[KF] WARNING: shadow method 'waft' requested but WAFT not available, disabling shadow")
                self._shadow_method = None
            elif not waft_ckpt:
                print("[KF] WARNING: shadow method 'waft' but no waft_ckpt, disabling shadow")
                self._shadow_method = None
            else:
                self._shadow_waft_wrapper = load_waft_model(waft_ckpt, device)

        # Pre-create trackers for known views
        if view_keys:
            for vk in view_keys:
                self._get_or_create_tracker(vk)
                if self._shadow_method:
                    self._get_or_create_shadow_tracker(vk)

        # Timing accumulators (seconds)
        self._t_compute = 0.0
        self._t_debug_write = 0.0
        self._t_shadow = 0.0
        self._t_set_kf = 0.0
        self._n_calls = 0

        if debug_dir:
            print(f"[KF] Debug images -> {debug_dir}", flush=True)
        shadow_tag = f", shadow={self._shadow_method}" if self._shadow_method else ""
        print(f"[KF] '{method}' backend, {len(self._trackers)} views{shadow_tag}", flush=True)

    @property
    def method(self) -> str:
        return self._method

    @property
    def shadow_method(self) -> Optional[str]:
        return self._shadow_method

    # ------------------------------------------------------------------
    # Tracker creation
    # ------------------------------------------------------------------

    def _create_tracker(self, method: str, waft_wrapper, view_key: str, debug_dir: str = None):
        vk_dir = os.path.join(debug_dir, view_key) if debug_dir else None
        if method == "waft":
            return WAFTFrameTracker(waft_wrapper, self._device,
                                    debug_dir=vk_dir, view_key=view_key)
        else:
            return FrameTracker(debug_dir=vk_dir, view_key=view_key)

    def _get_or_create_tracker(self, view_key: str):
        if view_key in self._trackers:
            return self._trackers[view_key]
        t = self._create_tracker(self._method, self._waft_wrapper, view_key, self._debug_dir)
        self._trackers[view_key] = t
        return t

    def _get_or_create_shadow_tracker(self, view_key: str):
        if view_key in self._shadow_trackers:
            return self._shadow_trackers[view_key]
        shadow_debug = os.path.join(self._debug_dir, "_shadow") if self._debug_dir else None
        t = self._create_tracker(self._shadow_method, self._shadow_waft_wrapper, view_key, shadow_debug)
        self._shadow_trackers[view_key] = t
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
        self._n_calls += 1

        # --- Primary trackers (compute_disparity) ---
        t0 = time.perf_counter()
        disparities: Dict[str, float] = {}
        flow_vis_map: Dict[str, object] = {}

        for vk, img in view_images.items():
            tracker = self._get_or_create_tracker(vk)
            disp, flow_vis = tracker.compute_disparity(img, min_disparity)
            disparities[vk] = disp
            flow_vis_map[vk] = flow_vis

        t1 = time.perf_counter()
        self._t_compute += t1 - t0

        max_disp = max(disparities.values()) if disparities else 0.0

        first_timestamp = all(
            self._trackers[vk].last_kf_bgr is None
            for vk in view_images
        )
        is_kf = first_timestamp or (max_disp > min_disparity)

        # Save primary debug images
        t2 = time.perf_counter()
        for vk, img in view_images.items():
            tracker = self._trackers[vk]
            tracker.save_debug(img, disparities[vk], flow_vis_map[vk],
                               min_disparity, is_kf)
        t3 = time.perf_counter()
        self._t_debug_write += t3 - t2

        # Update primary reference on keyframe
        if is_kf:
            t4 = time.perf_counter()
            for vk, img in view_images.items():
                self._trackers[vk].set_keyframe(img)
            self._t_set_kf += time.perf_counter() - t4

        # --- Shadow trackers ---
        if self._shadow_method:
            t5 = time.perf_counter()
            shadow_disps: Dict[str, float] = {}
            for vk, img in view_images.items():
                st = self._get_or_create_shadow_tracker(vk)
                s_disp, _ = st.compute_disparity(img, min_disparity)
                shadow_disps[vk] = s_disp

            shadow_max = max(shadow_disps.values()) if shadow_disps else 0.0
            shadow_first = all(
                self._shadow_trackers[vk].last_kf_bgr is None
                for vk in view_images
            )
            shadow_is_kf = shadow_first or (shadow_max > min_disparity)

            if is_kf:
                for vk, img in view_images.items():
                    self._shadow_trackers[vk].set_keyframe(img)

            self._t_shadow += time.perf_counter() - t5

            self._agreement_log.append({
                "primary_kf": is_kf,
                "shadow_kf": shadow_is_kf,
                "primary_max_disp": round(max_disp, 2),
                "shadow_max_disp": round(shadow_max, 2),
            })

        return is_kf, disparities

    def get_timing_summary(self) -> Dict:
        """Return accumulated timing breakdown across all check_timestamp calls."""
        total = self._t_compute + self._t_debug_write + self._t_shadow + self._t_set_kf
        n = max(self._n_calls, 1)
        return {
            "n_calls": self._n_calls,
            "compute_s": self._t_compute,
            "debug_write_s": self._t_debug_write,
            "shadow_s": self._t_shadow,
            "set_kf_s": self._t_set_kf,
            "total_tracked_s": total,
            "avg_per_ts_ms": total / n * 1000,
            "avg_compute_ms": self._t_compute / n * 1000,
            "avg_debug_ms": self._t_debug_write / n * 1000,
            "avg_shadow_ms": self._t_shadow / n * 1000,
        }

    # ------------------------------------------------------------------
    # Agreement statistics
    # ------------------------------------------------------------------

    def get_agreement_stats(self) -> Dict:
        """Return a summary of primary vs shadow keyframe agreement.

        Returns dict with keys: total, agree_kf, agree_skip,
        primary_only_kf, shadow_only_kf, agreement_pct,
        primary_method, shadow_method.
        """
        if not self._agreement_log:
            return {}

        total = len(self._agreement_log)
        agree_kf = sum(1 for e in self._agreement_log if e["primary_kf"] and e["shadow_kf"])
        agree_skip = sum(1 for e in self._agreement_log if not e["primary_kf"] and not e["shadow_kf"])
        primary_only = sum(1 for e in self._agreement_log if e["primary_kf"] and not e["shadow_kf"])
        shadow_only = sum(1 for e in self._agreement_log if not e["primary_kf"] and e["shadow_kf"])
        agreement_pct = round((agree_kf + agree_skip) / total * 100, 1) if total else 0.0

        return {
            "primary_method": self._method,
            "shadow_method": self._shadow_method,
            "total_timestamps": total,
            "agree_kf": agree_kf,
            "agree_skip": agree_skip,
            "primary_only_kf": primary_only,
            "shadow_only_kf": shadow_only,
            "agreement_pct": agreement_pct,
        }

    # ------------------------------------------------------------------
    # Single-view backward-compat API
    # ------------------------------------------------------------------

    def is_keyframe(self, image: np.ndarray, min_disparity: float) -> bool:
        return self.check_timestamp({"_default": image}, min_disparity)[0]
