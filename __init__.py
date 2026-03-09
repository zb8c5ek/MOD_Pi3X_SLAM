"""
MOD_Pi3X_SLAM - Pi3X Visual SLAM Module for RopediaGeoEngine
=============================================================

Dense visual SLAM replacing MOD_Pi3X_VisualOdometry (Step 2) in the
LORDPIPE pipeline.  Uses Pi3X for geometry estimation, GTSAM SIM(3)/SL(4)
factor graph for global pose optimisation, SALAD for loop closure
detection, and WAFT/LK for keyframe selection.

Deployment
----------
This module is self-contained.  Add it as a git submodule to
RopediaGeoEngine at ``MOD_Pi3X_SLAM/`` and update the LORDPIPE
``MOD_SCRIPTS`` entry::

    "vo_pipeline": Path("MOD_Pi3X_SLAM/_WegwerfSkript_VO/BURNPIPE_VO_Pipeline.py"),

Public API for other RopediaGeoEngine modules
----------------------------------------------
::

    from MOD_Pi3X_SLAM.util_shared_intrinsics import (
        load_shared_intrinsics,
        scale_intrinsics,
        build_K_4x4,
    )
"""
