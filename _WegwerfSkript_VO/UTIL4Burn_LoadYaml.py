"""
UTIL4Burn_LoadYaml - YAML config loader for Pi3X SLAM pipeline.

Loads a YAML config file and flattens it into a dict compatible
with essn_slam.SLAMConfig.
"""

import yaml
import os


def load_slam_config(yaml_path: str) -> dict:
    """
    Load YAML config and flatten into a dict for SLAMConfig.

    Expected YAML structure:
        paths:
            data_root: "..."
            ckpt_path: "..."
        input:
            cameras: [cam0, cam1]
            camera_angles:
                cam0: [p+0_y+0_r+0, ...]
                cam1: [p+0_y+0_r+0, ...]
            stride_frame: 1
        model:
            device: cuda
            dtype: float16
        processing:
            pixel_limit: 255000
            conf_threshold: 50.0
        slam:
            submap_size: 16
            overlap_window: 1
            min_disparity: 50.0
            use_keyframe_selection: true
            max_loops: 1
            lc_retrieval_threshold: 0.95
            lc_conf_threshold: 0.25
        output:
            colmap_output_path: "_output_colmap"
            log_poses_path: null
            viewer_port: 8080
    """
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = {}

    # Paths
    paths = raw.get('paths', {})
    cfg['data_root'] = paths.get('data_root', '')
    cfg['ckpt_path'] = paths.get('ckpt_path', r"D:\_HUBs\HuggingFace\Pi3X\model.safetensors")

    # Input
    inp = raw.get('input', {})
    cfg['cameras'] = inp.get('cameras', ['cam0'])
    cfg['camera_angles'] = inp.get('camera_angles', {})
    cfg['stride_frame'] = inp.get('stride_frame', 1)

    # Model
    model = raw.get('model', {})
    cfg['device'] = model.get('device', 'cuda')
    cfg['dtype'] = model.get('dtype', 'float16')

    # Processing
    proc = raw.get('processing', {})
    cfg['pixel_limit'] = proc.get('pixel_limit', 255000)
    cfg['conf_threshold'] = proc.get('conf_threshold', 50.0)
    cfg['conf_min_abs'] = proc.get('conf_min_abs', 0.0)

    # SLAM
    slam = raw.get('slam', {})
    cfg['submap_size'] = slam.get('submap_size', 16)
    cfg['overlap_window'] = slam.get('overlap_window', 1)
    cfg['max_submaps'] = slam.get('max_submaps', 0)
    cfg['keyframe_method'] = slam.get('keyframe_method', 'lk')
    cfg['waft_ckpt_path'] = slam.get('waft_ckpt_path', None)
    cfg['min_disparity'] = slam.get('min_disparity', 50.0)
    cfg['use_keyframe_selection'] = slam.get('use_keyframe_selection', True)
    cfg['max_loops'] = slam.get('max_loops', 1)
    cfg['lc_retrieval_threshold'] = slam.get('lc_retrieval_threshold', 0.95)
    cfg['lc_conf_threshold'] = slam.get('lc_conf_threshold', 0.25)
    cfg['sim3_inlier_thresh'] = slam.get('sim3_inlier_thresh', 0.5)
    cfg['alignment_primary'] = slam.get('alignment_primary', 'umeyama').lower()
    kf_debug = slam.get('kf_debug_dir', None)
    if kf_debug and not os.path.isabs(kf_debug):
        kf_debug = os.path.join(cfg['data_root'], kf_debug)
    cfg['kf_debug_dir'] = kf_debug

    stitch_debug = slam.get('stitch_debug_dir', None)
    if stitch_debug and not os.path.isabs(stitch_debug):
        stitch_debug = os.path.join(cfg['data_root'], stitch_debug)
    cfg['stitch_debug_dir'] = stitch_debug

    # Output: optional single output_dir unifies all outputs (RUN_{datetime} created inside it)
    out = raw.get('output', {})
    output_dir = out.get('output_dir', None) or out.get('output_root', None)
    if output_dir:
        base = output_dir if os.path.isabs(output_dir) else os.path.join(cfg['data_root'], output_dir)
        cfg['output_dir'] = base
        cfg['kf_debug_dir'] = os.path.join(base, 'kf_debug')
        cfg['stitch_debug_dir'] = os.path.join(base, 'stitch_debug')
        cfg['colmap_output_path'] = os.path.join(base, 'colmap')
        cfg['log_poses_path'] = os.path.join(base, 'log_poses.txt')
    else:
        cfg['output_dir'] = None
        colmap_out = out.get('colmap_output_path', None)
        if colmap_out and not os.path.isabs(colmap_out):
            colmap_out = os.path.join(cfg['data_root'], colmap_out)
        cfg['colmap_output_path'] = colmap_out

        log_path = out.get('log_poses_path', None)
        if log_path and not os.path.isabs(log_path):
            log_path = os.path.join(cfg['data_root'], log_path)
        cfg['log_poses_path'] = log_path

    cfg['viewer_port'] = out.get('viewer_port', 8080)
    # 0 or null = no downsampling in viewer; otherwise max points per submap
    cfg['viewer_max_points'] = out.get('viewer_max_points', 10000)

    return cfg
