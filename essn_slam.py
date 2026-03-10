"""
essn_slam - Thin Pi3X Visual SLAM orchestrator.

Wires essn_keyframe (keyframe selection) and essn_submap (submap processing)
together in the main run loop. Owns the SLAMConfig and top-level statistics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import datetime
import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch

from util_common import Accumulator, StageProfiler
from util_report import ESSNReport, setup_mod_logger
from essn_keyframe import KeyframeSelector
from essn_submap import SubmapProcessor


def _capture_git_info(repo_dir: str = None) -> dict:
    """Capture current git branch, commit hash, dirty status."""
    info = {"branch": "unknown", "commit": "unknown", "dirty": False, "error": None}
    try:
        kw = dict(cwd=repo_dir, capture_output=True, text=True, timeout=5)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], **kw)
        info["branch"] = branch.stdout.strip() if branch.returncode == 0 else "unknown"
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kw)
        info["commit"] = commit.stdout.strip() if commit.returncode == 0 else "unknown"
        commit_full = subprocess.run(["git", "rev-parse", "HEAD"], **kw)
        info["commit_full"] = commit_full.stdout.strip() if commit_full.returncode == 0 else "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], **kw)
        info["dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else False
    except Exception as e:
        info["error"] = str(e)
    return info


@dataclass
class SLAMConfig:
    """Configuration for the Pi3X SLAM pipeline."""

    # Model
    ckpt_path: str = r"D:\_HUBs\HuggingFace\Pi3X\model.safetensors"
    device: str = "cuda"
    dtype: str = "float16"

    # Image preprocessing
    pixel_limit: int = 255000
    patch_size: int = 14

    # Submap batching
    submap_size: int = 16
    overlap_window: int = 1
    # Max submaps to process (0 = unlimited). Useful for long sequences.
    max_submaps: int = 0

    # Confidence filtering (percentile: e.g. 90.0 means keep top 10%)
    conf_threshold: float = 50.0
    # Absolute sigmoid confidence floor (0.0-1.0). Points below this are
    # always discarded regardless of percentile. E.g. 0.9 = keep only >90% confident.
    conf_min_abs: float = 0.0

    # Keyframe selection
    keyframe_method: str = "waft"         # "waft" (default) or "lk"
    waft_ckpt_path: Optional[str] = None  # required when keyframe_method="waft"
    min_disparity: float = 50.0
    use_keyframe_selection: bool = True
    shadow_keyframe_method: Optional[str] = "lk"  # run this backend alongside primary for comparison (None = off)
    kf_debug_dir: Optional[str] = None    # save flow visualizations when set

    # Loop closure
    max_loops: int = 1
    lc_retrieval_threshold: float = 0.95
    lc_conf_threshold: float = 0.25

    # Alignment
    alignment_mode: str = "sim3"     # "sim3" (7-DoF) or "sl4" (15-DoF projective)
    shadow_sl4: bool = False         # run SL(4) as shadow comparison alongside primary
    sim3_inlier_thresh: float = 0.5  # RANSAC inlier threshold for alignment

    # Visualization
    vis_voxel_size: Optional[float] = None
    viewer_port: int = 0
    viewer_max_points: Optional[int] = 10000  # 0 or None = no downsampling

    # Debug
    stitch_debug_dir: Optional[str] = None  # save per-stitch PLY overlap + cumulative

    # Output
    colmap_output_path: Optional[str] = None
    log_poses_path: Optional[str] = None
    output_dir: Optional[str] = None  # when set, all outputs go directly here (no automatic subfolder)

    # Shared intrinsics from upstream undistortion step (3x3 numpy or None)
    shared_intrinsics: Optional[object] = None
    # Original (H, W) of undistorted images before pixel_limit resizing
    shared_intrinsics_hw: Optional[tuple] = None

    # Source config file path (for backup into run directory)
    config_yaml_path: Optional[str] = None

    def __post_init__(self):
        needs_waft = (self.keyframe_method == "waft" or
                      self.shadow_keyframe_method == "waft")
        if needs_waft and not self.waft_ckpt_path:
            _mod_dir = os.path.dirname(os.path.abspath(__file__))
            _default = os.path.join(_mod_dir, "3rdParty", "WAFT", "ckpts", "tar-c-t.pth")
            if os.path.isfile(_default):
                self.waft_ckpt_path = _default


class Pi3xSLAM:
    """Top-level SLAM orchestrator.

    Delegates keyframe selection to essn_keyframe.KeyframeSelector and
    per-submap processing to essn_submap.SubmapProcessor.
    """

    def __init__(self, config: SLAMConfig, view_keys: List[str] = None):
        self.config = config
        self.run_id = datetime.datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
        self.run_dirs = {}
        self.report = ESSNReport("essn_slam")

        self._stamp_output_dirs()
        self._save_run_meta()

        # Logger
        log_dir = next((d for d in self.run_dirs.values() if d), None)
        self.logger = setup_mod_logger("essn_slam", log_dir=log_dir)

        self.keyframe_selector = KeyframeSelector(
            method=config.keyframe_method,
            waft_ckpt=config.waft_ckpt_path,
            device=config.device,
            debug_dir=config.kf_debug_dir,
            view_keys=view_keys or [],
            shadow_method=config.shadow_keyframe_method,
        )

        # Submap processor (Pi3X model, graph, loop closure, viewer)
        self.submap_processor = SubmapProcessor(
            ckpt_path=config.ckpt_path,
            device=config.device,
            dtype=config.dtype,
            pixel_limit=config.pixel_limit,
            patch_size=config.patch_size,
            conf_threshold=config.conf_threshold,
            conf_min_abs=config.conf_min_abs,
            max_loops=config.max_loops,
            lc_retrieval_threshold=config.lc_retrieval_threshold,
            lc_conf_threshold=config.lc_conf_threshold,
            vis_voxel_size=config.vis_voxel_size,
            viewer_port=config.viewer_port,
            viewer_max_points=config.viewer_max_points,
            colmap_output_path=config.colmap_output_path,
            log_poses_path=config.log_poses_path,
            stitch_debug_dir=config.stitch_debug_dir,
            sim3_inlier_thresh=config.sim3_inlier_thresh,
            alignment_mode=config.alignment_mode,
            shadow_sl4=config.shadow_sl4,
            shared_intrinsics=config.shared_intrinsics,
            shared_intrinsics_hw=config.shared_intrinsics_hw,
        )

    # ------------------------------------------------------------------
    # Run directory setup
    # ------------------------------------------------------------------

    # Paths that are files (not directories) — their parent is created, not themselves.
    _FILE_ATTRS = {"log_poses_path"}

    def _stamp_output_dirs(self):
        """Set up output directories.

        When output_dir is given, write directly there -- caller owns the path.
        Otherwise fall back to appending RUN_{datetime} to each individual path.
        """
        c = self.config
        if getattr(c, "output_dir", None):
            run_root = c.output_dir
            for attr, subdir in (
                ("kf_debug_dir", "kf_debug"),
                ("stitch_debug_dir", "stitch_debug"),
                ("colmap_output_path", "colmap"),
                ("log_poses_path", "log_poses.txt"),
            ):
                path = os.path.join(run_root, subdir)
                setattr(c, attr, path)
                self.run_dirs[attr] = path
        else:
            for attr in ("kf_debug_dir", "stitch_debug_dir", "colmap_output_path", "log_poses_path"):
                base = getattr(c, attr, None)
                if base:
                    stamped = os.path.join(base, self.run_id)
                    setattr(c, attr, stamped)
                    self.run_dirs[attr] = stamped

    def _save_run_meta(self):
        """Save run_meta.json + config backup into each output directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        git_info = _capture_git_info(script_dir)

        meta = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "git": git_info,
            "config": {k: v for k, v in asdict(self.config).items()
                       if not k.startswith("_")},
        }

        # Collect directories to write run_meta into (skip file-paths like log_poses.txt)
        dir_targets = set()
        for attr, d in self.run_dirs.items():
            if not d:
                continue
            if attr in self._FILE_ATTRS:
                os.makedirs(os.path.dirname(d), exist_ok=True)
            else:
                dir_targets.add(d)

        for d in dir_targets:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "run_meta.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)

            if self.config.config_yaml_path and os.path.isfile(self.config.config_yaml_path):
                shutil.copy2(self.config.config_yaml_path, os.path.join(d, "config_backup.yaml"))

        print(f"[Run] {self.run_id}  git={git_info['branch']}@{git_info['commit']}"
              f"{'*' if git_info['dirty'] else ''}", flush=True)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Phase 1: keyframe selection (upfront, full pass)
    # ------------------------------------------------------------------

    def _select_all_keyframes(self, timestamps, pipeline_prof):
        """Run keyframe selection over ALL timestamps before SLAM.

        Returns the list of keyframe timestamp dicts.
        After this call the KeyframeSelector is no longer needed and the
        caller can free its GPU memory.
        """
        n_ts = len(timestamps)
        kf_timestamps: List[Dict[str, str]] = []
        progress_interval = max(1, n_ts // 20)

        print(f"\n{'='*60}", flush=True)
        print(f"Phase 1: Keyframe selection  ({n_ts} timestamps)", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()

        if not self.config.use_keyframe_selection:
            print("  Keyframe selection disabled — using all timestamps", flush=True)
            return list(timestamps)

        for idx, ts in enumerate(timestamps):
            with pipeline_prof("keyframe_selection"):
                view_images = {}
                skip = False
                for vk, path in ts.items():
                    img = cv2.imread(path)
                    if img is None:
                        print(f"Warning: could not read {path}, skipping ts",
                              flush=True)
                        skip = True
                        break
                    view_images[vk] = img
                if skip:
                    continue

                is_kf, disps = self.keyframe_selector.check_timestamp(
                    view_images, self.config.min_disparity)
                if is_kf:
                    kf_timestamps.append(ts)

            if idx % progress_interval == 0 or idx == n_ts - 1:
                print(f"  [KF select] ts {idx+1}/{n_ts}  "
                      f"kf={len(kf_timestamps)}  "
                      f"({len(kf_timestamps)/max(idx+1,1)*100:.0f}% rate)",
                      flush=True)

        elapsed = time.time() - t0
        print(f"\n  Keyframe selection complete: "
              f"{len(kf_timestamps)}/{n_ts} timestamps selected "
              f"({len(kf_timestamps)/max(n_ts,1)*100:.1f}%) in {elapsed:.1f}s",
              flush=True)

        return kf_timestamps

    def _free_keyframe_selector(self):
        """Release the keyframe selector and its GPU memory."""
        if hasattr(self, 'keyframe_selector') and self.keyframe_selector is not None:
            kf_agreement = self.keyframe_selector.get_agreement_stats()
            del self.keyframe_selector
            self.keyframe_selector = None
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  [GPU] Freed keyframe selector models", flush=True)
            return kf_agreement
        return None

    # ------------------------------------------------------------------
    # Phase 2: SLAM processing
    # ------------------------------------------------------------------

    def _run_slam_on_keyframes(self, kf_timestamps, n_views, pipeline_prof):
        """Batch keyframes into submaps and run SLAM."""
        submap_size = self.config.submap_size
        overlap = self.config.overlap_window
        target_kf_timestamps = submap_size + overlap
        sp = self.submap_processor

        n_kf = len(kf_timestamps)
        print(f"\n{'='*60}", flush=True)
        print(f"Phase 2: SLAM processing  "
              f"({n_kf} kf-timestamps, submap_size={submap_size}, "
              f"overlap={overlap})", flush=True)
        print(f"{'='*60}", flush=True)

        submap_count = 0
        batch: List[Dict[str, str]] = []

        for i, ts in enumerate(kf_timestamps):
            batch.append(ts)

            is_last = (i == n_kf - 1)
            if len(batch) >= target_kf_timestamps or (is_last and batch):
                submap_count += 1
                all_paths = self._interleave_kf_timestamps(batch)
                print(f"\n--- Submap {submap_count}: {len(batch)} kf-timestamps, "
                      f"{len(all_paths)} images ---", flush=True)
                with pipeline_prof("submap_processing"):
                    sp.process_submap(all_paths)
                batch = batch[-overlap:] if overlap > 0 else []

                if 0 < self.config.max_submaps <= submap_count:
                    print(f"\nReached max_submaps limit "
                          f"({self.config.max_submaps}).", flush=True)
                    break

        return submap_count

    # ------------------------------------------------------------------
    # Main entry: two-phase run
    # ------------------------------------------------------------------

    def run_timestamps(self, timestamps: List[Dict[str, str]]):
        """Run the full SLAM pipeline on multi-view timestamp data.

        Two-phase execution:
          Phase 1 — Select ALL keyframes upfront (WAFT/LK on GPU).
          Phase 2 — Free KF selector, run SLAM submaps (Pi3X + SALAD).
        """
        n_ts = len(timestamps)
        n_views = len(timestamps[0]) if timestamps else 0
        print(f"Starting Pi3X SLAM: {n_ts} timestamps x {n_views} views",
              flush=True)

        total_start = time.time()
        pipeline_prof = StageProfiler()

        # Phase 1: keyframe selection (full pass)
        kf_timestamps = self._select_all_keyframes(timestamps, pipeline_prof)
        kf_agreement = self._free_keyframe_selector()

        # Phase 2: SLAM processing
        submap_count = self._run_slam_on_keyframes(
            kf_timestamps, n_views, pipeline_prof)

        # Summary
        sp = self.submap_processor
        total_time = time.time() - total_start
        kf_ts_count = len(kf_timestamps)
        total_images = kf_ts_count * n_views

        print(f"\n{'='*60}")
        print(f"Pi3X SLAM complete")
        print(f"  Keyframe method: {self.config.keyframe_method}")
        print(f"  Keyframe timestamps: {kf_ts_count} / {n_ts}")
        print(f"  Images processed: {total_images} ({n_views} views/ts)")
        print(f"  Submaps created: {sp.map.get_num_submaps()}")
        print(f"  Loop closures: {sp.graph.get_num_loops()}")
        print(f"  Total time: {total_time:.2f}s")
        if total_images > 0:
            print(f"  Avg Pi3X inference/image: "
                  f"{sp.inference_timer.total_time / total_images:.4f}s")
            print(f"  Avg FPS (images): {total_images / total_time:.2f}")

        if kf_agreement:
            print(f"  --- Keyframe agreement "
                  f"({kf_agreement['primary_method']} vs "
                  f"{kf_agreement['shadow_method']} shadow) ---")
            print(f"    Total timestamps: {kf_agreement['total_timestamps']}")
            print(f"    Both KF: {kf_agreement['agree_kf']}   "
                  f"Both skip: {kf_agreement['agree_skip']}   "
                  f"{kf_agreement['primary_method']}-only KF: "
                  f"{kf_agreement['primary_only_kf']}   "
                  f"{kf_agreement['shadow_method']}-only KF: "
                  f"{kf_agreement['shadow_only_kf']}")
            print(f"    Agreement: {kf_agreement['agreement_pct']}%")
        print(f"{'='*60}")

        pipeline_prof.print_summary("Pipeline")

        sp.export_colmap()
        sp.export_poses()

        # Finalize ESSN reports
        self.report.set_metric("total_timestamps", n_ts)
        self.report.set_metric("keyframe_timestamps", kf_ts_count)
        self.report.set_metric("total_images", total_images)
        self.report.set_metric("num_views", n_views)
        self.report.set_metric("submaps_created", sp.map.get_num_submaps())
        self.report.set_metric("loop_closures", sp.graph.get_num_loops())
        self.report.set_metric("total_time_s", total_time)
        self.report.set_metric("keyframe_method", self.config.keyframe_method)
        if kf_agreement:
            self.report.set_metric("keyframe_agreement", kf_agreement)
        if total_images > 0:
            self.report.set_metric("avg_inference_per_image_s",
                                   sp.inference_timer.total_time / total_images)
            self.report.set_metric("avg_fps_images", total_images / total_time)
        self.report.set_status("success")

        # Save reports to first available output dir
        report_dir = next((d for d in self.run_dirs.values() if d), None)
        if report_dir:
            self.report.save(report_dir)
            sp.report.set_status("success")
            sp.report.save(report_dir)
            self.logger.info(f"Reports saved to {report_dir}")

        self.report.print_summary()
        sp.report.print_summary()
        sp.print_stitch_report()

    @staticmethod
    def _interleave_kf_timestamps(kf_timestamps: List[Dict[str, str]]) -> List[str]:
        """Flatten keyframe timestamps into interleaved path list.

        Order: for each timestamp, emit all views in sorted view_key order.
        """
        paths = []
        for ts in kf_timestamps:
            for vk in sorted(ts.keys()):
                paths.append(ts[vk])
        return paths

    # ------------------------------------------------------------------
    # Legacy flat-list API (single-view or pre-interleaved)
    # ------------------------------------------------------------------

    def run(self, image_paths: List[str]):
        """Run on a flat list of image paths (single-view or pre-interleaved)."""
        from util_common import sort_images_by_number
        print(f"Starting Pi3X SLAM on {len(image_paths)} images", flush=True)
        image_paths = sort_images_by_number(image_paths)

        submap_size = self.config.submap_size
        overlap = self.config.overlap_window
        target_batch = submap_size + overlap

        keyframe_paths: List[str] = []
        keyframe_count = 0
        submap_count = 0
        total_start = time.time()
        keyframe_timer = Accumulator()
        sp = self.submap_processor

        for idx, img_path in enumerate(image_paths):
            is_last = (idx == len(image_paths) - 1)
            if self.config.use_keyframe_selection:
                with keyframe_timer:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    if self.keyframe_selector.is_keyframe(img, self.config.min_disparity):
                        keyframe_paths.append(img_path)
                        keyframe_count += 1
            else:
                keyframe_paths.append(img_path)
                keyframe_count += 1

            if len(keyframe_paths) >= target_batch or (is_last and keyframe_paths):
                submap_count += 1
                print(f"\n--- Submap {submap_count}: {len(keyframe_paths)} keyframes ---")
                sp.process_submap(keyframe_paths)
                keyframe_paths = keyframe_paths[-overlap:] if overlap > 0 else []
                if 0 < self.config.max_submaps <= submap_count:
                    print(f"\nReached max_submaps limit ({self.config.max_submaps}).")
                    break

        total_time = time.time() - total_start
        print(f"\n{'='*60}\nPi3X SLAM complete\n  Keyframes: {keyframe_count}")
        print(f"  Submaps: {sp.map.get_num_submaps()}")
        print(f"  Time: {total_time:.2f}s\n{'='*60}")
        sp.export_colmap()
        sp.export_poses()

    # ------------------------------------------------------------------
    # Viewer keepalive
    # ------------------------------------------------------------------

    def wait_for_exit(self):
        """Block until Ctrl+C, keeping the Viser viewer alive."""
        port = self.config.viewer_port
        print(f"\n=== Viser viewer ready at http://localhost:{port} ===")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down.")
