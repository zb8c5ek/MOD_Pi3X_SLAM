"""
kern_utils - Shared model-agnostic utilities for the Pi3X SLAM pipeline.

Ported from vggt_slam/slam_utils.py with additions for Pi3X intrinsics estimation.
"""

import os
import re
import time
import numpy as np
import scipy
import matplotlib
from PIL import Image
import torch
from torchvision import transforms as TF


def estimate_intrinsics(H, W):
    """
    Estimate a pinhole intrinsic matrix from image dimensions.
    Uses focal = max(H,W) * 1.2 with principal point at image center.
    Returns (3, 3) numpy array.
    """
    focal = max(H, W) * 1.2
    K = np.array([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def estimate_intrinsics_4x4(H, W, N=1):
    """
    Estimate intrinsics and return as (N, 4, 4) array suitable for Submap.proj_mats.
    """
    K = estimate_intrinsics(H, W)
    K_4x4 = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
    K_4x4[:, :3, :3] = K
    return K_4x4


def slice_with_overlap(lst, n, k):
    """Sliding window over list with overlap k. Returns list of sublists."""
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)
    return result


def sort_images_by_number(image_paths):
    """Sort image paths by the numeric part of the filename."""
    def extract_number(path):
        filename = os.path.basename(path)
        match = re.search(r'\d+(?:\.\d+)?(?=\.[^.]+$)', filename)
        return float(match.group()) if match else float('inf')
    return sorted(image_paths, key=extract_number)


def downsample_images(image_names, downsample_factor):
    """Keep every downsample_factor-th image."""
    return image_names[::downsample_factor]


def decompose_camera(P, no_inverse=False):
    """
    Decompose a 3x4 or 4x4 camera projection matrix P into
    intrinsics K, rotation R, translation t, and scale.
    """
    if P.shape[0] != 3:
        P = P / P[-1, -1]
        P = P[0:3, :]
    assert P.shape == (3, 4)

    M = P[:, :3]
    K, R = scipy.linalg.rq(M)

    if K[0, 0] < 0:
        K[:, 0] *= -1
        R[0, :] *= -1
    if K[1, 1] < 0:
        K[:, 1] *= -1
        R[1, :] *= -1
    if K[2, 2] < 0:
        K[:, 2] *= -1
        R[2, :] *= -1

    scale = K[2, 2]
    if not no_inverse:
        R = np.linalg.inv(R)
        t = -R @ np.linalg.inv(K) @ P[:, 3]
    else:
        t = np.linalg.inv(K) @ P[:, 3]
    K = K / scale

    if np.linalg.det(R) < 0:
        R = -R

    return K, R, t, scale


def normalize_to_sl4(H):
    """Normalize a 4x4 homography matrix H to unit determinant (SL4)."""
    det = np.linalg.det(H)
    if det == 0:
        raise ValueError("Homography matrix is singular and cannot be normalized.")
    scale = det ** (1 / 4)
    return H / scale


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def compute_image_embeddings(model, preprocess, image_paths, batch_size=64, device="cuda"):
    """Compute CLIP image embeddings for a list of image paths."""
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))

    imgs = torch.stack(imgs).to(device)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size]
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0).numpy()


def compute_text_embeddings(clip_model, tokenizer, text, device="cuda"):
    """Compute CLIP text embedding for a query string."""
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().numpy()


def compute_obb_from_points(points: np.ndarray):
    """
    Compute an oriented bounding box (OBB) via PCA for a Nx3 point cloud.
    Returns (center, extent, rotation).
    """
    assert points.ndim == 2 and points.shape[1] == 3
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Point cloud is empty or invalid")

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotation = eigvecs
    points_local = centered @ rotation
    min_corner = points_local.min(axis=0)
    max_corner = points_local.max(axis=0)
    extent = max_corner - min_corner
    center_local = 0.5 * (min_corner + max_corner)
    center_world = centroid + center_local @ rotation.T

    return center_world, extent, rotation


class Accumulator:
    """Context-manager timer for profiling code blocks."""
    def __init__(self):
        self.total_time = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.total_time += (time.perf_counter() - self.start)


class StageProfiler:
    """Lightweight per-stage profiler for the SLAM pipeline.

    Usage::

        prof = StageProfiler()
        with prof("load_images"):
            ...
        with prof("inference"):
            ...
        prof.print_summary("Submap 3")
    """

    def __init__(self):
        self._stages = {}       # name -> list of durations
        self._order = []        # insertion order
        self._current = None
        self._start = None

    class _Context:
        def __init__(self, profiler, name):
            self._profiler = profiler
            self._name = name

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            dt = time.perf_counter() - self._t0
            self._profiler._record(self._name, dt)

    def __call__(self, stage_name: str):
        """Return a context manager that times the named stage."""
        return self._Context(self, stage_name)

    def _record(self, name, dt):
        if name not in self._stages:
            self._stages[name] = []
            self._order.append(name)
        self._stages[name].append(dt)

    def print_summary(self, label: str = "Profile"):
        """Print a compact profiling table for the current run."""
        if not self._stages:
            return
        total = sum(sum(v) for v in self._stages.values())
        bar = '-' * 64
        print(f"\n{bar}")
        print(f"  {label}  —  stage profiling  (total {total:.2f}s)")
        print(f"  {'Stage':<28} {'Calls':>5} {'Total':>8} {'Mean':>8} {'%':>6}")
        print(bar)
        for name in self._order:
            dts = self._stages[name]
            n = len(dts)
            s = sum(dts)
            mean = s / n
            pct = 100.0 * s / total if total > 0 else 0
            print(f"  {name:<28} {n:>5} {s:>7.2f}s {mean:>7.3f}s {pct:>5.1f}%")
        print(bar)

    def get_total(self, stage_name: str) -> float:
        """Total seconds spent in a stage across all calls."""
        return sum(self._stages.get(stage_name, [0.0]))

    def reset(self):
        self._stages.clear()
        self._order.clear()
