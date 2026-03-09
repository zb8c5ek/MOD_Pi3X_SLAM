"""
kern_keyframe - Optical flow keyframe selection backends.

Two backends share the same API:

  compute_disparity(image)  -> float   (never updates reference)
  set_keyframe(image)       -> None    (updates reference to this frame)

This split lets multi-view callers aggregate disparities across views
before deciding which frames become the new reference.

  FrameTracker      -- Lucas-Kanade sparse flow (cv2). CPU-only.
  WAFTFrameTracker  -- WAFT dense flow (GPU). Requires load_waft_model().

WAFT is optional: imports are guarded so the module works in LK-only mode.
"""

import os
import sys
import argparse

import numpy as np
import cv2
import torch


# ======================================================================
# Debug visualization helpers
# ======================================================================

def _flow_to_color(flow_uv):
    """Convert (2, H, W) optical flow to a BGR color image (HSV wheel)."""
    u, v = flow_uv[0], flow_uv[1]
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_colorbar(height, width, vmin, vmax):
    """Vertical colorbar: green(high) -> red(low) with tick labels."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = 1.0 - y / max(height - 1, 1)
        hue = int(val * 120)
        bar[y, :] = np.array([hue, 255, 255], dtype=np.uint8)
    bar = cv2.cvtColor(bar, cv2.COLOR_HSV2BGR)

    n_ticks = 5
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        y = int(frac * (height - 1))
        val = vmax - frac * (vmax - vmin)
        cv2.line(bar, (0, y), (8, y), (255, 255, 255), 1)
        cv2.putText(bar, f"{val:.0f}", (10, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return bar


def _build_debug_canvas(ref_image, cur_image, flow_vis, disp, threshold,
                         is_kf, frame_idx, kf_count, view_key=""):
    """4-panel debug image: REF | CURRENT | FLOW | colorbar."""
    h, w = cur_image.shape[:2]

    if ref_image is None:
        ref_panel = np.zeros_like(cur_image)
        cv2.putText(ref_panel, "no ref", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    else:
        ref_panel = cv2.resize(ref_image, (w, h)) if ref_image.shape[:2] != (h, w) else ref_image.copy()

    if flow_vis is None:
        flow_panel = np.zeros_like(cur_image)
    else:
        flow_panel = cv2.resize(flow_vis, (w, h)) if flow_vis.shape[:2] != (h, w) else flow_vis

    cbar_w = max(60, w // 8)
    cbar = _make_colorbar(h, cbar_w, 0, max(threshold * 2, disp * 1.5, 1))

    tag = "KF" if is_kf else "skip"
    color = (0, 255, 0) if is_kf else (0, 128, 255)

    for panel, label in [(ref_panel, "REF (last KF)"), (cur_image, "CURRENT"),
                          (flow_panel, "FLOW")]:
        cv2.putText(panel, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    canvas = np.hstack([ref_panel, cur_image.copy(), flow_panel, cbar])
    info = f"f{frame_idx} kf#{kf_count} disp={disp:.1f}/{threshold:.0f} [{tag}]"
    if view_key:
        info = f"[{view_key}] {info}"
    cv2.putText(canvas, info, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas


# ======================================================================
# LK backend (always available)
# ======================================================================

class FrameTracker:
    """Sparse optical flow using Lucas-Kanade (cv2)."""

    def __init__(self, debug_dir=None, view_key=""):
        self.last_kf_bgr = None
        self.kf_pts = None
        self.kf_gray = None
        self.debug_dir = debug_dir
        self.view_key = view_key
        self._frame_idx = 0
        self._kf_count = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def set_keyframe(self, image):
        """Set *image* as the new reference keyframe."""
        self.last_kf_bgr = image.copy()
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.kf_pts = cv2.goodFeaturesToTrack(
            self.kf_gray, maxCorners=1000, qualityLevel=0.01,
            minDistance=8, blockSize=7,
        )
        self._kf_count += 1

    def compute_disparity(self, image, threshold_for_debug=50.0):
        """Compute mean disparity against last keyframe. Does NOT update ref.

        Returns: (disparity: float, flow_vis: np.ndarray or None)
        """
        self._frame_idx += 1

        if self.last_kf_bgr is None or self.kf_pts is None or len(self.kf_pts) < 10:
            return 0.0, None

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, curr_gray, self.kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        status = status.flatten()
        good_kf = self.kf_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_kf) < 10:
            return 0.0, None

        displacement = np.linalg.norm(good_next - good_kf, axis=1)
        mean_disp = float(np.mean(displacement))

        flow_vis = np.zeros_like(image)
        for p1, p2 in zip(good_kf, good_next):
            p1t = tuple(p1.ravel().astype(int))
            p2t = tuple(p2.ravel().astype(int))
            cv2.arrowedLine(flow_vis, p1t, p2t, color=(0, 255, 0), thickness=1, tipLength=0.3)

        return mean_disp, flow_vis

    def save_debug(self, image, disp, flow_vis, threshold, is_kf):
        if not self.debug_dir:
            return
        tag = "KF" if is_kf else "skip"
        canvas = _build_debug_canvas(
            self.last_kf_bgr, image, flow_vis, disp, threshold, is_kf,
            self._frame_idx, self._kf_count, self.view_key)
        cv2.imwrite(os.path.join(self.debug_dir, f"{self._frame_idx:06d}_{tag}.jpg"), canvas)


# ======================================================================
# WAFT backend (optional -- requires 3rdParty/WAFT)
# ======================================================================

WAFT_AVAILABLE = False
_waft_import_error = None
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _waft_root = os.path.join(_this_dir, "3rdParty", "WAFT")
    if os.path.isdir(_waft_root) and _waft_root not in sys.path:
        sys.path.insert(0, _waft_root)

    from model import fetch_model as _waft_fetch_model
    from utils.utils import load_ckpt as _waft_load_ckpt
    from inference_tools import InferenceWrapper as _WAFTInferenceWrapper
    WAFT_AVAILABLE = True
except Exception as _e:
    _waft_import_error = _e


def _make_waft_args():
    args = argparse.Namespace()
    args.algorithm = "waft-a1"
    args.dav2_backbone = "vits"
    args.network_backbone = "vits"
    args.use_var = True
    args.var_min = 0
    args.var_max = 10
    args.iters = 5
    args.image_size = [432, 960]
    args.scale = 0
    return args


def load_waft_model(ckpt_path, device='cuda'):
    """Load the WAFT model and return an InferenceWrapper."""
    if not WAFT_AVAILABLE:
        raise RuntimeError(
            f"WAFT not available. Import failed: {_waft_import_error}\n"
            "Check 3rdParty/WAFT/ and install dependencies (see 3rdParty/MANIFEST.md)"
        )
    args = _make_waft_args()
    model = _waft_fetch_model(args)
    _waft_load_ckpt(model, ckpt_path)
    model = model.to(device).eval()

    wrapper = _WAFTInferenceWrapper(
        model, scale=0, train_size=args.image_size,
        pad_to_train_size=True, tiling=False,
    )
    return wrapper


def _bgr_to_tensor(image, device='cuda'):
    """Convert a BGR uint8 numpy image to a float32 (1, 3, H, W) RGB tensor in [0, 255]."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


class WAFTFrameTracker:
    """Dense optical flow using WAFT."""

    def __init__(self, waft_wrapper, device='cuda', debug_dir=None, view_key=""):
        self.wrapper = waft_wrapper
        self.device = device
        self.last_kf_tensor = None
        self.last_kf_bgr = None
        self.debug_dir = debug_dir
        self.view_key = view_key
        self._frame_idx = 0
        self._kf_count = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    def set_keyframe(self, image):
        """Set *image* as the new reference keyframe."""
        self.last_kf_tensor = _bgr_to_tensor(image, self.device)
        self.last_kf_bgr = image.copy()
        self._kf_count += 1

    def compute_disparity(self, image, threshold_for_debug=50.0):
        """Compute mean flow magnitude against last keyframe. Does NOT update ref.

        Returns: (disparity: float, flow_color: np.ndarray or None)
        """
        self._frame_idx += 1

        if self.last_kf_tensor is None:
            return 0.0, None

        current_tensor = _bgr_to_tensor(image, self.device)
        with torch.no_grad():
            output = self.wrapper.calc_flow(self.last_kf_tensor, current_tensor)

        flow = output['flow'][-1]  # (1, 2, H, W)
        mean_disp = flow.norm(dim=1).mean().item()

        flow_np = flow[0].cpu().numpy()
        flow_color = _flow_to_color(flow_np)

        return mean_disp, flow_color

    def save_debug(self, image, disp, flow_vis, threshold, is_kf):
        if not self.debug_dir:
            return
        tag = "KF" if is_kf else "skip"
        canvas = _build_debug_canvas(
            self.last_kf_bgr, image, flow_vis, disp, threshold, is_kf,
            self._frame_idx, self._kf_count, self.view_key)
        cv2.imwrite(os.path.join(self.debug_dir, f"{self._frame_idx:06d}_{tag}.jpg"), canvas)
