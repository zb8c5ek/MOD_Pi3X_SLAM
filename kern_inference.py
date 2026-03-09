"""
kern_inference - Pi3X model loading and inference utilities.

Functions for:
- Loading Pi3X model from checkpoint
- Running inference on image tensors
- Loading images from multi-camera directories (interleaved)
- Loading images from arbitrary file paths (for SLAM keyframe flow)
- Filtering and subsampling point clouds

Based on MOD_Pi3X_VisualOdometry/kern_inference.py with SLAM helpers added.
"""

import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pi3.models.pi3x import Pi3X
from pi3.utils.geometry import depth_edge


def load_model(
    ckpt_path: Union[str, Path],
    device: str = 'cuda',
) -> Tuple[Pi3X, float]:
    """
    Load Pi3X model from checkpoint.
    Returns (model, load_time_seconds).
    """
    from safetensors.torch import load_file

    start_time = time.time()
    model = Pi3X().to(device).eval()
    model.load_state_dict(load_file(ckpt_path), strict=False)
    load_time = time.time() - start_time

    return model, load_time


def load_images_multicam(
    imgs_root: Union[str, Path],
    cam_dirs: List[str],
    interval: int = 1,
    pixel_limit: int = 255000,
    patch_size: int = 14,
    verbose: bool = True,
) -> Tuple[torch.Tensor, List[str], Tuple[int, int], float]:
    """
    Load and preprocess images from multiple camera directories with interleaving.
    Images are interleaved: cam0[0], cam1[0], cam0[1], cam1[1], ...

    Returns (images_tensor, image_names, (H, W), load_time_seconds).
    """
    imgs_root = Path(imgs_root)
    start_time = time.time()

    cam_file_lists = {}
    for cam_dir in cam_dirs:
        cam_path = imgs_root / cam_dir
        if not cam_path.exists():
            print(f"Warning: Camera directory not found: {cam_path}")
            continue

        cam_files = sorted([
            f for f in cam_path.glob("*")
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        cam_files = cam_files[::interval]
        cam_file_lists[cam_dir] = cam_files
        if verbose:
            print(f"Found {len(cam_files)} images in {cam_dir}/")

    if not cam_file_lists:
        raise ValueError("No images found in any camera directory!")

    all_images = []
    image_names = []

    max_len = max(len(files) for files in cam_file_lists.values())
    for i in range(max_len):
        for cam_dir in cam_dirs:
            if cam_dir in cam_file_lists and i < len(cam_file_lists[cam_dir]):
                img_file = cam_file_lists[cam_dir][i]
                img = Image.open(img_file).convert("RGB")
                all_images.append(img)
                image_names.append(f"{cam_dir}/{img_file.name}")

    if verbose:
        print(f"Total images loaded: {len(all_images)} (interleaved from {len(cam_file_lists)} cameras)")

    first_img = all_images[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target = int(round(W_orig * scale / patch_size) * patch_size)
    H_target = int(round(H_orig * scale / patch_size) * patch_size)
    if verbose:
        print(f"Resizing images to: ({W_target}, {H_target})")

    transform = transforms.Compose([
        transforms.Resize((H_target, W_target)),
        transforms.ToTensor(),
    ])

    img_tensors = [transform(img) for img in all_images]
    imgs = torch.stack(img_tensors)

    load_time = time.time() - start_time
    return imgs, image_names, (H_target, W_target), load_time


def load_images_from_paths(
    image_paths: List[str],
    pixel_limit: int = 255000,
    patch_size: int = 14,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    """
    Load and preprocess images from arbitrary file paths.
    Used by the SLAM keyframe flow where images arrive one at a time.

    Returns (images_tensor, (H, W), load_time_seconds).
    - images_tensor: (N, 3, H, W) in [0, 1]
    """
    start_time = time.time()

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    W_orig, H_orig = pil_images[0].size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target = int(round(W_orig * scale / patch_size) * patch_size)
    H_target = int(round(H_orig * scale / patch_size) * patch_size)

    if verbose:
        print(f"Resizing {len(pil_images)} images to: ({W_target}, {H_target})")

    transform = transforms.Compose([
        transforms.Resize((H_target, W_target)),
        transforms.ToTensor(),
    ])

    imgs = torch.stack([transform(img) for img in pil_images])
    load_time = time.time() - start_time
    return imgs, (H_target, W_target), load_time


def run_inference(
    model: Pi3X,
    imgs: torch.Tensor,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Run Pi3X inference on images.

    Returns (results_dict, inference_time_seconds).
    Results dict contains:
    - 'points': (1, N, H, W, 3) world coordinates
    - 'local_points': (1, N, H, W, 3) camera-local coordinates
    - 'conf': (1, N, H, W, 1) confidence logits
    - 'camera_poses': (1, N, 4, 4) cam2world poses
    - 'rays': (1, N, H, W, 3) ray directions
    - 'metric': (1,) scale factor
    """
    imgs = imgs.to(device)

    start_time = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            results = model(imgs[None])
    torch.cuda.synchronize()
    inference_time = time.time() - start_time

    return results, inference_time


def pi3x_to_submap_data(results, images, shared_intrinsics=None, original_hw=None):
    """Convert Pi3X inference output to the format expected by Submap.

    Pi3X returns:
        camera_poses: (1, N, 4, 4) cam2world
        points:       (1, N, H, W, 3) world coords
        conf:         (1, N, H, W, 1) logits

    Submap needs:
        world_to_cam: (N, 4, 4)
        world_points: (N, H, W, 3)
        conf:         (N, H, W) after sigmoid
        colors:       (N, H, W, 3) uint8
        K_4x4:        (N, 4, 4)

    Args:
        results: Pi3X inference output dict.
        images: (N, 3, H, W) tensor or numpy array.
        shared_intrinsics: Optional (3, 3) real K matrix from upstream
            undistortion. When provided, replaces the heuristic
            focal = max(H,W)*1.2 estimate.
        original_hw: Optional (H, W) of the original undistorted images.
            Required for scaling K when pixel_limit resizing changes the
            inference resolution. If None, K is used as-is.
    """
    from util_common import estimate_intrinsics_4x4

    cam2world = results['camera_poses'][0].float().cpu().numpy()
    world_to_cam = np.linalg.inv(cam2world)

    world_points = results['points'][0].float().cpu().numpy()

    conf_logits = results['conf'][..., 0][0]
    conf = torch.sigmoid(conf_logits).float().cpu().numpy()

    if isinstance(images, torch.Tensor):
        colors = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    else:
        colors = images

    N, H, W = conf.shape

    if shared_intrinsics is not None:
        from util_shared_intrinsics import scale_intrinsics, build_K_4x4
        if original_hw is not None:
            K_3x3 = scale_intrinsics(shared_intrinsics, original_hw, (H, W))
        else:
            K_3x3 = shared_intrinsics.copy()
        K_4x4 = build_K_4x4(K_3x3, N)
    else:
        K_4x4 = estimate_intrinsics_4x4(H, W, N=1)
        K_4x4 = np.broadcast_to(K_4x4, (N, 4, 4)).copy()

    return {
        'world_to_cam': world_to_cam,
        'cam2world': cam2world,
        'world_points': world_points,
        'conf': conf,
        'colors': colors,
        'K_4x4': K_4x4,
    }


def filter_points(
    results: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    conf_threshold: float = 0.1,
    edge_rtol: float = 0.03,
    subsample_ratio: float = 1.0,
    subsample_min_points: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Filter and subsample point cloud from inference results.

    Returns (points, colors, total_before_subsample).
    """
    conf_scores = torch.sigmoid(results['conf'][..., 0])
    masks = conf_scores > conf_threshold
    non_edge = ~depth_edge(results['local_points'][..., 2], rtol=edge_rtol)
    masks = torch.logical_and(masks, non_edge)[0]

    points_all = results['points'][0]
    colors_all = imgs.permute(0, 2, 3, 1)

    points_masked = points_all[masks]
    colors_masked = colors_all[masks]

    total_before = len(points_masked)

    if subsample_ratio < 1.0 and total_before > 0:
        n_keep = int(total_before * subsample_ratio)
        if subsample_min_points > 0:
            if n_keep < subsample_min_points:
                n_keep = min(subsample_min_points, total_before)
        if n_keep < total_before:
            indices = torch.randperm(total_before)[:n_keep]
            points_masked = points_masked[indices]
            colors_masked = colors_masked[indices]

    return points_masked, colors_masked, total_before
