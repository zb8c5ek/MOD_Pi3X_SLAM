"""kern_remap -- Two-tier MASt3R remapping for SLAM submaps and keyframes.

Tier 1 (submap remap):
    For each submap, run MASt3R matching on its images (SLAM cameras only),
    build a COLMAP database, run the mapper, and dump .npz match caches.

Tier 2 (keyframe remap):
    For the unified KeyFrames/ folder (all cameras × all angles at KF
    timestamps), load pre-computed matches from Tier 1 where applicable,
    run MASt3R only on new pairs, then build a combined COLMAP reconstruction.

Pure algorithm -- no config loading, no report generation.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images_relpath(root: Path) -> List[str]:
    """Recursively collect image relative paths under *root*."""
    relpaths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                relpaths.append(
                    str(Path(dirpath, fn).relative_to(root)).replace("\\", "/")
                )
    return sorted(relpaths)


# =========================================================================
# Tier 1 -- Per-submap remap
# =========================================================================

def remap_submaps(
    slam_dir: Path,
    model,
    colmap_exe: str,
    image_size: int = 512,
    camera_model: str = "PINHOLE",
    min_num_matches: int = 15,
    mapper_timeout: int = 300,
) -> Dict[str, Any]:
    """Run MASt3R + COLMAP on every ``submap_NNN/`` inside *slam_dir*.

    For each submap:
      1. Collects images from ``submap_NNN/cam*/angle/*.jpg``.
      2. Runs ``create_db_from_folder_with_structure()`` with ``dump_dir``
         so that all pairwise matches are cached as ``.npz`` files.
      3. Runs COLMAP mapper to produce ``submap_NNN/sparse/``.

    Args:
        slam_dir:    SLAM episode output dir (contains submap_NNN/).
        model:       Loaded MASt3R model.
        colmap_exe:  Path to COLMAP binary.
        image_size:  MASt3R inference resolution.
        camera_model: COLMAP camera model name.
        min_num_matches: COLMAP mapper min match threshold.
        mapper_timeout:  COLMAP mapper timeout in seconds.

    Returns:
        Summary dict with per-submap results.
    """
    from KernLib_M3RSfM.kern_scene_graph_sfm import create_db_from_folder_with_structure
    from KernLib_M3RSfM.kern_colmap_mapper import run_mapper

    submap_dirs = sorted(
        d for d in slam_dir.iterdir()
        if d.is_dir() and d.name.startswith("submap_")
    )

    results = {}
    for sm_dir in submap_dirs:
        sm_name = sm_dir.name
        logger.info("[Tier1] Remapping %s ...", sm_name)

        filelist = _collect_images_relpath(sm_dir)
        if len(filelist) < 2:
            logger.warning("[Tier1] %s: <2 images, skipping", sm_name)
            results[sm_name] = {"success": False, "reason": "too few images"}
            continue

        dump_dir = sm_dir / "_matches"
        db_output = sm_dir / "_remap"

        db_result = create_db_from_folder_with_structure(
            root_path=sm_dir,
            filelist_relpath=filelist,
            dp_output=db_output,
            model=model,
            image_size=image_size,
            camera_model=camera_model,
            share_intrinsics_by_subfolder=True,
            dump_dir=dump_dir,
        )

        if not db_result.get("success"):
            logger.warning("[Tier1] %s: DB creation failed: %s",
                           sm_name, db_result.get("error"))
            results[sm_name] = db_result
            continue

        db_path = db_result["database_path"]
        sparse_output = sm_dir / "sparse"
        sparse_output.mkdir(parents=True, exist_ok=True)

        mapper_result = run_mapper(
            colmap_exe=colmap_exe,
            db_path=db_path,
            image_path=str(sm_dir),
            output_path=str(sparse_output),
            mapper_options={
                "min_num_matches": min_num_matches,
                "multiple_models": False,
            },
            timeout=mapper_timeout,
        )

        db_result["mapper"] = mapper_result
        results[sm_name] = db_result
        logger.info("[Tier1] %s: %d images, %d pairs, mapper=%s",
                    sm_name, db_result["num_images"], db_result["num_pairs"],
                    "ok" if mapper_result.get("success") else "failed")

    return {"submaps": results, "num_submaps": len(submap_dirs)}


# =========================================================================
# Tier 2 -- KeyFrames remap (all cameras × all angles)
# =========================================================================

def remap_keyframes(
    slam_dir: Path,
    model,
    colmap_exe: str,
    image_size: int = 512,
    camera_model: str = "PINHOLE",
    min_num_matches: int = 15,
    mapper_timeout: int = 600,
) -> Dict[str, Any]:
    """Run MASt3R + COLMAP on ``KeyFrames/`` with pre-computed match reuse.

    1. Collects images from ``KeyFrames/cam{N}/angle/*.jpg``.
    2. Gathers all ``.npz`` match caches from ``submap_NNN/_matches/``
       (Tier 1 output).  Pairs already matched there are reused.
    3. Runs MASt3R only on new pairs (cross-camera, cross-angle).
    4. Builds a COLMAP database from all matches and runs the mapper.

    Args:
        slam_dir:    SLAM episode output dir (contains KeyFrames/, submap_NNN/).
        model:       Loaded MASt3R model.
        colmap_exe:  Path to COLMAP binary.
        image_size:  MASt3R inference resolution.
        camera_model: COLMAP camera model name.
        min_num_matches: COLMAP mapper min match threshold.
        mapper_timeout:  COLMAP mapper timeout in seconds.

    Returns:
        Summary dict with DB and mapper results.
    """
    from KernLib_M3RSfM.kern_scene_graph_sfm import create_db_from_folder_with_structure
    from KernLib_M3RSfM.kern_colmap_mapper import run_mapper

    kf_dir = slam_dir / "KeyFrames"
    if not kf_dir.is_dir():
        return {"success": False, "error": "KeyFrames/ directory not found"}

    filelist = _collect_images_relpath(kf_dir)
    if len(filelist) < 2:
        return {"success": False, "error": f"Too few images ({len(filelist)})"}

    logger.info("[Tier2] KeyFrames remap: %d images", len(filelist))

    # Collect all Tier 1 match caches into a single directory via symlinks
    # so the matching function can find them.
    kf_dump_dir = kf_dir / "_matches"
    kf_dump_dir.mkdir(parents=True, exist_ok=True)

    # Build a mapping from submap relpaths -> keyframe relpaths.
    # Submap images: cam0/p+0_y+30_r+0/000042_..._cam0_p+0_y+30_r+0.jpg
    # KeyFrame images: cam0/p+0_y+30_r+0/000042_..._cam0_p+0_y+30_r+0.jpg
    # Same structure! So we can reuse .npz caches directly by copying/linking.
    n_reused = 0
    for sm_dir in sorted(slam_dir.iterdir()):
        if not sm_dir.is_dir() or not sm_dir.name.startswith("submap_"):
            continue
        match_dir = sm_dir / "_matches"
        if not match_dir.is_dir():
            continue
        for npz_file in match_dir.glob("*.npz"):
            dst = kf_dump_dir / npz_file.name
            if not dst.exists():
                try:
                    dst.symlink_to(npz_file)
                except OSError:
                    import shutil
                    shutil.copy2(str(npz_file), str(dst))
                n_reused += 1

    logger.info("[Tier2] Collected %d pre-computed match files from Tier 1",
                n_reused)

    db_output = kf_dir / "_remap"
    db_result = create_db_from_folder_with_structure(
        root_path=kf_dir,
        filelist_relpath=filelist,
        dp_output=db_output,
        model=model,
        image_size=image_size,
        camera_model=camera_model,
        share_intrinsics_by_subfolder=True,
        dump_dir=kf_dump_dir,
    )

    if not db_result.get("success"):
        logger.warning("[Tier2] DB creation failed: %s", db_result.get("error"))
        return db_result

    db_path = db_result["database_path"]
    sparse_output = kf_dir / "sparse"
    sparse_output.mkdir(parents=True, exist_ok=True)

    mapper_result = run_mapper(
        colmap_exe=colmap_exe,
        db_path=db_path,
        image_path=str(kf_dir),
        output_path=str(sparse_output),
        mapper_options={
            "min_num_matches": min_num_matches,
            "multiple_models": False,
        },
        timeout=mapper_timeout,
    )

    db_result["mapper"] = mapper_result
    db_result["n_reused_from_tier1"] = n_reused
    logger.info("[Tier2] KeyFrames: %d images, %d pairs (%d from Tier1 cache), "
                "mapper=%s",
                db_result["num_images"], db_result["num_pairs"], n_reused,
                "ok" if mapper_result.get("success") else "failed")

    return db_result
