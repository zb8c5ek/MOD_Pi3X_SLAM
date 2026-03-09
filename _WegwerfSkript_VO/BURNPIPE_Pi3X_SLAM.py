"""
BURNPIPE_Pi3X_SLAM - Entry script for the Pi3X SLAM pipeline.

Usage:
    python BURNPIPE_Pi3X_SLAM.py _configs_slam/config_slam_test.yaml

Or via the PowerShell wrapper:
    .\\BURNSCRIPT_ENTRY_SLAM.ps1
"""

import sys
import os

# Add module root to path so we can import kern_* and essn_*
_script_dir = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_script_dir)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from UTIL4Burn_LoadYaml import load_slam_config
from UTIL_IO_Discovery import discover_timestamps, discover_images
from essn_slam import SLAMConfig, Pi3xSLAM


def main():
    if len(sys.argv) < 2:
        print("Usage: python BURNPIPE_Pi3X_SLAM.py <config_yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isabs(config_path):
        config_path = os.path.join(_script_dir, config_path)

    print(f"Loading config from {config_path}")
    cfg = load_slam_config(config_path)

    # Discover images -- structured by timestamp
    print(f"Discovering images in {cfg['data_root']}")
    try:
        timestamps, view_keys = discover_timestamps(
            data_root=cfg['data_root'],
            cameras=cfg['cameras'],
            camera_angles=cfg.get('camera_angles', {}),
            stride=cfg.get('stride_frame', 1),
        )
        multi_view = True
    except ValueError:
        image_paths = discover_images(
            data_root=cfg['data_root'],
            cameras=cfg['cameras'],
            camera_angles=cfg.get('camera_angles', {}),
            stride=cfg.get('stride_frame', 1),
        )
        multi_view = False
        view_keys = []
        print(f"Found {len(image_paths)} images (flat)")

    # Build SLAMConfig
    slam_config = SLAMConfig(
        ckpt_path=cfg['ckpt_path'],
        device=cfg['device'],
        dtype=cfg['dtype'],
        pixel_limit=cfg['pixel_limit'],
        submap_size=cfg['submap_size'],
        overlap_window=cfg['overlap_window'],
        max_submaps=cfg.get('max_submaps', 0),
        conf_threshold=cfg['conf_threshold'],
        conf_min_abs=cfg.get('conf_min_abs', 0.0),
        keyframe_method=cfg.get('keyframe_method', 'lk'),
        waft_ckpt_path=cfg.get('waft_ckpt_path'),
        min_disparity=cfg['min_disparity'],
        use_keyframe_selection=cfg['use_keyframe_selection'],
        max_loops=cfg['max_loops'],
        lc_retrieval_threshold=cfg['lc_retrieval_threshold'],
        lc_conf_threshold=cfg['lc_conf_threshold'],
        sim3_inlier_thresh=cfg.get('sim3_inlier_thresh', 0.5),
        kf_debug_dir=cfg.get('kf_debug_dir'),
        stitch_debug_dir=cfg.get('stitch_debug_dir'),
        colmap_output_path=cfg.get('colmap_output_path'),
        log_poses_path=cfg.get('log_poses_path'),
        output_dir=cfg.get('output_dir'),
        viewer_port=cfg.get('viewer_port', 8080),
        viewer_max_points=cfg.get('viewer_max_points', 10000),
        config_yaml_path=config_path,
    )

    # Run SLAM
    slam = Pi3xSLAM(slam_config, view_keys=view_keys)
    if multi_view:
        slam.run_timestamps(timestamps)
    else:
        slam.run(image_paths)
    slam.wait_for_exit()


if __name__ == "__main__":
    main()
