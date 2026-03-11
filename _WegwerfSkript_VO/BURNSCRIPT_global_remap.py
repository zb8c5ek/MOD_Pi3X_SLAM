#!/usr/bin/env python3
"""
BURNSCRIPT_global_remap.py -- Global MASt3R remap on SLAM KeyFrames
====================================================================

Reads the ``global-remapping`` section from a LORD config YAML and runs
MASt3R matching + COLMAP reconstruction on SLAM keyframe images.

Pair generation uses the SLAM scene graph (submap adjacency + loop
closures) so that only spatially related frames are matched -- avoiding
the O(N^2) all-pairs cost.

Usage:
    cd D:\\RopediaGeoEngine
    micromamba run -n sfm3rV2 python MOD_Pi3X_SLAM\\_WegwerfSkript_VO\\BURNSCRIPT_global_remap.py ^
        StreamLines\\_lordConfigs\\_Experiment\\lord_config_slam_integration_0309.yaml

Auto-discovers the VO-en-SLAM output from ``overrides.vo_episoding_folder``,
finds the SLAM episode, and runs the global remap.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("global_remap")

# Bootstrap: project root on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_ROOT = _SCRIPT_DIR.parent          # MOD_Pi3X_SLAM/
_PROJECT_ROOT = _MODULE_ROOT.parent        # RopediaGeoEngine/

for p in (str(_PROJECT_ROOT), str(_MODULE_ROOT), str(_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
# Config loading
# =============================================================================

def load_lord_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_slam_episode(cfg: Dict) -> Dict[str, Path]:
    """Discover SLAM episode dir, KeyFrames dir, and scene_graph.json.

    After the SLAM pipeline runs, ``KeyFrames/`` contains images for ALL
    cameras and angles at keyframe timestamps (populated from the EPISODING
    undistort folder).  This function locates that directory.

    Returns:
        Dict with keys: slam_episode_dir, keyframes_dir, scene_graph_json,
        episode_name.
    """
    overrides = cfg.get("overrides", {})
    vo_folder = overrides.get("vo_episoding_folder")
    if not vo_folder:
        raise ValueError(
            "overrides.vo_episoding_folder is required to locate SLAM output"
        )
    vo_path = Path(vo_folder)
    slam_dir = vo_path / "SLAM"
    if not slam_dir.is_dir():
        raise FileNotFoundError(f"SLAM dir not found: {slam_dir}")

    episodes = sorted(
        d for d in slam_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    if not episodes:
        raise FileNotFoundError(f"No episode_* dirs in {slam_dir}")

    ep_dir = episodes[0]
    sg_json = ep_dir / "scene_graph.json"
    if not sg_json.is_file():
        raise FileNotFoundError(f"scene_graph.json not found: {sg_json}")

    kf_dir = ep_dir / "KeyFrames"
    if not kf_dir.is_dir():
        raise FileNotFoundError(
            f"KeyFrames/ not found: {kf_dir}  "
            f"(re-run SLAM pipeline to populate all cams/angles)"
        )

    return {
        "slam_episode_dir": ep_dir,
        "keyframes_dir": kf_dir,
        "scene_graph_json": sg_json,
        "episode_name": ep_dir.name,
    }


# =============================================================================
# Image staging (per-camera, multi-angle rig format)
# =============================================================================


def collect_and_stage(
    keyframes_dir: Path,
    rigs: List[Dict],
    output_dir: Path,
) -> Tuple[Path, List[str]]:
    """Collect images from KeyFrames/ folder for the requested rig.

    ``KeyFrames/`` already contains ALL cameras and angles at keyframe
    timestamps (populated by the SLAM pipeline from the EPISODING
    undistort folder).  This function selects the cam/angle combos
    requested by the rig config and stages them for MASt3R.

    Rig format (per-camera angle lists)::

        - name: "lite-surround"
          cam0: ["p+0_y+30_r+0", "p-30_y+0_r+0"]
          cam1: ["p+0_y-30_r+0"]
          cam2: ["p+0_y+30_r+0"]

    Stages into ``output_dir/staged_images/{cam}_{angle}/filename.jpg``
    so ``share_intrinsics_by_subfolder`` groups them correctly.
    """
    staging_dir = output_dir / "staged_images"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    filelist: List[str] = []
    n_staged = 0

    for rig in rigs:
        for key, val in rig.items():
            if key == "name":
                continue
            cam = key
            angles = val if isinstance(val, list) else [val]

            for angle in angles:
                src_dir = keyframes_dir / cam / angle
                if not src_dir.is_dir():
                    logger.warning("Source dir not found: %s", src_dir)
                    continue

                subfolder = f"{cam}_{angle}"
                dst_dir = staging_dir / subfolder
                dst_dir.mkdir(parents=True, exist_ok=True)

                for img_file in sorted(src_dir.iterdir()):
                    if img_file.suffix.lower() not in IMAGE_EXTS:
                        continue
                    dst_file = dst_dir / img_file.name
                    if not dst_file.exists():
                        shutil.copy2(str(img_file), str(dst_file))
                    filelist.append(f"{subfolder}/{img_file.name}")
                    n_staged += 1

    logger.info(
        "Staged %d images from %d rig(s) into %s",
        n_staged, len(rigs), staging_dir,
    )
    return staging_dir, sorted(filelist)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Global MASt3R remap on SLAM KeyFrames",
    )
    parser.add_argument("config", type=str, help="Path to LORD config YAML")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        logger.error("Config not found: %s", config_path)
        return 1

    cfg = load_lord_config(config_path)
    remap_cfg = cfg.get("global-remapping", {})
    if not remap_cfg:
        logger.error("No 'global-remapping' section in config")
        return 1

    t0 = time.time()

    # ── Resolve paths ──
    paths = resolve_slam_episode(cfg)
    ep_dir = paths["slam_episode_dir"]
    kf_dir = paths["keyframes_dir"]
    sg_json = paths["scene_graph_json"]
    logger.info("=" * 60)
    logger.info("Global MASt3R Remap")
    logger.info("=" * 60)
    logger.info("  Episode   : %s", paths["episode_name"])
    logger.info("  KeyFrames : %s", kf_dir)
    logger.info("  SceneGraph: %s", sg_json)

    # ── Output dir ──
    processing_dir = Path(cfg["global"]["processing_dir"])
    # Find the run dir (same as the one containing VO output)
    vo_folder = Path(cfg["overrides"]["vo_episoding_folder"])
    run_dir = vo_folder.parent
    output_dir = run_dir / f"GlobalRemap_{ep_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Output    : %s", output_dir)

    # ── Stage images ──
    logger.info("\n--- Step 1: Stage Images ---")
    rigs = remap_cfg.get("rigs", [])
    if not rigs:
        logger.error("No rigs defined in global-remapping.rigs")
        return 1
    staging_dir, filelist = collect_and_stage(kf_dir, rigs, output_dir)
    logger.info("  %d images staged", len(filelist))
    if len(filelist) < 2:
        logger.error("Too few images (%d)", len(filelist))
        return 1

    # ── Setup MASt3R ──
    logger.info("\n--- Step 2: Load MASt3R Model ---")
    hubs = cfg.get("model_hubs", {})
    mast3r_dir = remap_cfg.get("mast3r_dir", hubs.get("mast3r_dir", ""))
    dust3r_dir = remap_cfg.get("dust3r_dir", hubs.get("dust3r_dir", ""))

    from KernLib_M3RSfM import load_model, release_model, setup_mast3r_paths
    from KernLib_M3RSfM.kern_matching_strategy import parse as parse_matching_strategy

    setup_mast3r_paths(mast3r_dir, dust3r_dir)

    model_cfg = remap_cfg.get("model", {})
    model = load_model(
        checkpoint_dir=model_cfg.get("checkpoint_dir", hubs.get("mast3r_ckpt_dir", "")),
        model_name=model_cfg.get("name", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"),
    )

    match_raw = remap_cfg.get("matching", {}).get("combined", [{"type": "num_pts", "value": 300}])
    matching_strategy = parse_matching_strategy(match_raw)
    logger.info("  Model loaded, strategy=%s", matching_strategy)

    # ── Generate pairs from SLAM scene graph ──
    logger.info("\n--- Step 3: Generate Pairs ---")
    strategy_cfg = remap_cfg.get("strategy", {})
    strategy_name = strategy_cfg.get("name", "submap-window")

    if strategy_name == "submap-window":
        from kern_global_remap_pairs import generate_submap_window_pairs
        custom_pairs = generate_submap_window_pairs(
            scene_graph_json=str(sg_json),
            filelist_relpath=filelist,
            window_size=strategy_cfg.get("window_size", 1),
            include_lc=strategy_cfg.get("included_lc", True),
            temporal_window=strategy_cfg.get("temporal_window", 1),
        )
    else:
        logger.warning("Unknown strategy '%s', using all-pairs", strategy_name)
        custom_pairs = None

    n_full = len(filelist) * (len(filelist) - 1) // 2
    n_pairs = len(custom_pairs) if custom_pairs else n_full
    logger.info("  %d pairs (vs %d full, %.1fx reduction)",
                n_pairs, n_full, n_full / max(n_pairs, 1))

    # ── MASt3R matching + DB ──
    logger.info("\n--- Step 4: MASt3R Matching + COLMAP DB ---")
    from KernLib_M3RSfM.kern_scene_graph_sfm import create_db_from_folder_with_structure

    img_cfg = remap_cfg.get("image", {})
    inf_cfg = remap_cfg.get("inference", {})
    mapper_cfg = remap_cfg.get("mapper", {})
    match_cfg = remap_cfg.get("matching", {})

    db_output = output_dir / "mast3r_db"
    db_result = create_db_from_folder_with_structure(
        root_path=staging_dir,
        filelist_relpath=filelist,
        dp_output=db_output,
        model=model,
        matching_strategy=matching_strategy,
        image_size=img_cfg.get("size", 512),
        camera_model=mapper_cfg.get("camera_model", "PINHOLE"),
        share_intrinsics_by_subfolder=mapper_cfg.get("share_intrinsics_by_subfolder", True),
        batch_size=inf_cfg.get("batch_size", 16),
        custom_pairs=custom_pairs,
        min_len_track=match_cfg.get("min_len_track", 3),
        subsample=match_cfg.get("subsample", 8),
        pixel_tol=match_cfg.get("pixel_tol", 5),
    )

    logger.info("  [KernLib] DB result: success=%s, pairs=%s, time=%.1fs",
                db_result.get("success"), db_result.get("num_pairs"),
                db_result.get("processing_time", 0))

    # ── Step 4b: MOD_Mapper3r A/B comparison ──
    logger.info("\n--- Step 4b: MOD_Mapper3r (A/B comparison) ---")
    mapper3r_result = {"success": False, "num_pairs": 0, "processing_time": 0}
    try:
        mapper3r_root = _PROJECT_ROOT / "_Deprecated" / "MOD_Mapper3r-master"
        mapper3r_fp = str(mapper3r_root / "feature_process")
        if mapper3r_fp not in sys.path:
            sys.path.insert(0, mapper3r_fp)
        if str(mapper3r_root) not in sys.path:
            sys.path.insert(0, str(mapper3r_root))

        from essn_create_db import (
            create_db_from_folder_with_structure as mapper3r_create_db,
        )

        mapper3r_output = output_dir / "mapper3r_db"
        mapper3r_result = mapper3r_create_db(
            root_path=staging_dir,
            filelist_relpath=filelist,
            dp_output=mapper3r_output,
            model=model,
            matching_strategy=matching_strategy,
            image_size=img_cfg.get("size", 512),
            camera_model=mapper_cfg.get("camera_model", "PINHOLE"),
            share_intrinsics_by_subfolder=mapper_cfg.get("share_intrinsics_by_subfolder", True),
            batch_size=inf_cfg.get("batch_size", 16),
            custom_pairs=custom_pairs,
        )
        logger.info("  [Mapper3r] DB result: success=%s, pairs=%s, time=%.1fs",
                    mapper3r_result.get("success"), mapper3r_result.get("num_pairs"),
                    mapper3r_result.get("processing_time", 0))
    except Exception as e:
        logger.warning("  [Mapper3r] A/B comparison failed (non-fatal): %s", e, exc_info=True)

    release_model(model)

    if not db_result.get("success"):
        logger.error("MASt3R matching failed: %s", db_result.get("error"))
        return 1

    # ── COLMAP mapper ──
    colmap_exe = remap_cfg.get("colmap_exe", "colmap")
    mapper_timeout = remap_cfg.get("processing", {}).get("mapping_timeout", 600)

    if mapper_timeout and mapper_timeout > 0:
        logger.info("\n--- Step 5: COLMAP Mapper ---")
        from KernLib_M3RSfM.kern_colmap_mapper import run_mapper

        sparse_dir = output_dir / "sparse"
        mapper_result = run_mapper(
            colmap_exe=colmap_exe,
            db_path=db_result["database_path"],
            image_path=str(staging_dir),
            output_path=str(sparse_dir),
            mapper_options={
                "min_num_matches": mapper_cfg.get("min_num_matches", 150),
                "multiple_models": mapper_cfg.get("multiple_models", False),
                "extract_colors": mapper_cfg.get("extract_colors", True),
            },
            timeout=mapper_timeout,
        )
        logger.info("  Mapper: success=%s, registered=%s, points3d=%s",
                    mapper_result.get("success"),
                    mapper_result.get("num_registered"),
                    mapper_result.get("num_points3d"))
    else:
        logger.info("\n--- Step 5: COLMAP Mapper SKIPPED (mapping_timeout=0) ---")
        mapper_result = {"success": False, "num_registered": 0, "num_points3d": 0}

    # ── Report ──
    logger.info("\n--- Step 6: Generate Report ---")
    from UTILREPORT_global_remap import generate_report

    pairs_txt = db_output / "pairs.txt"
    report_html = output_dir / "global_remap_report.html"
    try:
        generate_report(
            scene_graph_json=sg_json,
            pairs_txt=pairs_txt,
            output_html=report_html,
            custom_pairs=custom_pairs,
            title=f"Global Remap: {ep_dir.name}",
        )
        logger.info("  Report: %s", report_html)
    except Exception as e:
        logger.warning("Report generation failed (non-fatal): %s", e)
        report_html = None

    # ── Summary ──
    elapsed = time.time() - t0
    summary = {
        "episode": ep_dir.name,
        "strategy": strategy_cfg,
        "rigs": rigs,
        "num_images": len(filelist),
        "matching_kernlib": {
            "strategy": str(matching_strategy),
            "num_pairs": db_result.get("num_pairs", 0),
            "matching_time": db_result.get("processing_time", 0),
        },
        "matching_mapper3r": {
            "num_pairs": mapper3r_result.get("num_pairs", 0),
            "matching_time": mapper3r_result.get("processing_time", 0),
            "success": mapper3r_result.get("success", False),
        },
        "mapper": {
            "success": mapper_result.get("success", False),
            "num_registered": mapper_result.get("num_registered", 0),
            "num_points3d": mapper_result.get("num_points3d", 0),
            "recon_dir": mapper_result.get("recon_dir"),
        },
        "total_time": round(elapsed, 1),
        "report_html": str(report_html) if report_html else None,
    }

    summary_path = output_dir / "global_remap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("GLOBAL REMAP COMPLETE")
    logger.info("=" * 60)
    logger.info("  Images:         %d", len(filelist))
    logger.info("  Input pairs:    %d", n_pairs)
    logger.info("  ── KernLib  ──  verified=%s  time=%.1fs",
                db_result.get("num_pairs", 0),
                db_result.get("processing_time", 0))
    logger.info("  ── Mapper3r ──  verified=%s  time=%.1fs",
                mapper3r_result.get("num_pairs", 0),
                mapper3r_result.get("processing_time", 0))
    logger.info("  Registered:     %d / %d",
                mapper_result.get("num_registered", 0), len(filelist))
    logger.info("  3D Points:      %d", mapper_result.get("num_points3d", 0))
    logger.info("  Total time:     %.1fs", elapsed)
    logger.info("  Output:         %s", output_dir)
    if mapper_result.get("recon_dir"):
        logger.info("  Recon:          %s", mapper_result["recon_dir"])
    if report_html:
        logger.info("  Report:         %s", report_html)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
