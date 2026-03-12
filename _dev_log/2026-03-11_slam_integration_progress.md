# SLAM Integration & Global Remap — Progress Report

**Date**: 2026-03-11
**Branch**: `claude/slam-integration-debug-M67U1`
**Config**: `lord_config_slam_integration_0311.yaml`

---

## 1. Critical Bug Fixes

### 1.1 Missing `db.commit()` — 0 Matches in COLMAP DB

**File**: `KernLib_M3RSfM/kern_scene_graph_sfm.py`

The COLMAP database was showing `Loading matches... 0` despite MASt3R producing valid matches. Root cause: Python's `sqlite3` module uses implicit transactions — `export_matches()` wrote data inside an uncommitted transaction that was silently rolled back when the connection closed.

**Fix**: Added explicit `colmap_db.commit()` after `export_matches()` (line ~319) and after `_update_camera_model` (line ~362).

### 1.2 Real DB ID Mapping

**File**: `KernLib_M3RSfM/kern_scene_graph_sfm.py`

The `image_to_colmap` mapping assumed sequential IDs. Fixed to query actual COLMAP image/camera IDs from the database, preventing ID mismatches between MASt3R's internal indices and COLMAP's auto-assigned IDs.

### 1.3 Mapper Timeout Interpretation

**Files**: `BURNSCRIPT_global_remap.py`, `kern_colmap_runner.py`, `kern_colmap_mapper.py`

`mapping_timeout: 0` in config was interpreted as "skip the mapper entirely." Changed semantics: `0` now means **unlimited timeout** (passes `None` to subprocess). Type signatures updated to `Optional[int]`.

---

## 2. Architectural Changes

### 2.1 Run Stamp Architecture (Deterministic Output Folders)

**File**: `StreamLines/LORDScripts/LORDPIPE_StreamLine.py`

Replaced dangerous `discover_newest_folder` auto-discovery with a deterministic **run ID** system. Each pipeline invocation generates a unique `timestamp_adjective-noun` identifier (Docker-style naming), e.g. `20260311_214411_soft-spire`.

All downstream steps inherit this ID, producing predictable folder names:
```
RUN_20260311_214411_soft-spire/
├── EPISODING_20260311_214411_soft-spire/
├── VO-en-SLAM_20260311_214411_soft-spire/
└── GlobalRemap_episode_001_full_t0000-0970_20260311_214411_soft-spire/
```

### 2.2 Dynamic Staging via `stereo_pairs`

**File**: `BURNSCRIPT_global_remap.py`

Removed the static top-level `rigs` config section. Image staging is now dynamically derived from `strategy.temporal_stitch.stereo_pairs` and `strategy.lc_stitch.stereo_pairs` via `_derive_rig_from_strategy()`. This ensures that what gets staged always matches what gets paired.

### 2.3 Multi-Model Selection

**File**: `BURNSCRIPT_global_remap.py`

When `mapper.multiple_models: true`, COLMAP may produce multiple reconstruction models. Added `_enumerate_models()` to scan all models, select the one with the most registered images, and `_assemble_colmap_project()` to organize the best model into a clean `colmap_project/` output folder.

---

## 3. Pair Generation Enhancements

### 3.1 `window_timestamp_stride` Parameter

**File**: `MOD_Pi3X_SLAM/kern_global_remap_pairs.py`

New config parameter under `temporal_stitch` and `lc_stitch`. Instead of only pairing consecutive timestamps (stride=1), `window_timestamp_stride: 2` pairs every other timestamp, spreading temporal connections over wider intervals within submap joints. This improves global coverage without increasing pair count proportionally.

### 3.2 Per-Joint Pair Output

**File**: `MOD_Pi3X_SLAM/kern_global_remap_pairs.py`, `BURNSCRIPT_global_remap.py`

Introduced `PairResult` class that extends `List[Tuple[str,str]]` with `.joints` and `.summary` metadata. Each submap-joint (e.g., `ts_sm000+sm001`, `lc_sm003_sm005`) now:
- Tracks its own pair list with regime, involved submaps, timestamp counts, stereo/temporal pair counts
- Writes a separate pair file to `pairs_per_joint/` for easy debugging
- Produces a `_joints_summary.json`

### 3.3 4-Camera Pair Coverage

**Config**: `lord_config_slam_integration_0311.yaml`

Initial runs showed cam2 and cam3 at **0% registration** — no cross-camera pairs involving these cameras were generated because `temporal_stitch.stereo_pairs` only had `cam0-cam1`.

**Fix**: Added `cam0-cam2` and `cam1-cam3` to `temporal_stitch.stereo_pairs`, bridging all four cameras into the matching graph.

---

## 4. Reporting & Analysis

### 4.1 Pre-Matching HTML Report

**File**: `BURNSCRIPT_global_remap.py`, `UTILREPORT_global_remap.py`

Added an early report generation step (Step 3b) immediately after pair generation, before the expensive MASt3R matching. This allows visual verification of pair structure before committing to a 20+ minute matching run.

### 4.2 Joint Pair Visualization in Report

**File**: `UTILREPORT_global_remap.py`

The HTML report now includes:
- **Pair Generation by Submap Joint** table (regime, submaps, timestamp count, stereo/temporal pair counts)
- **Representative Joint Pair Graphs** — 3D Plotly graphs showing keyframes as nodes and stereo/temporal pairs as edges, one per regime (largest joint)

### 4.3 Post-Analysis Script

**File**: `MOD_Pi3X_SLAM/_WegwerfSkript_VO/UTIL_GlobalRemap_PostAnalysis.py` (new)

Comprehensive post-reconstruction analysis tool with CLI interface:

| Section | Content |
|---------|---------|
| **Registration Breakdown** | Stats by cam+angle, camera, angle, timestamp; identifies low-coverage timestamps |
| **3D Point Contribution** | Triangulated points per image, track length distribution, reprojection errors |
| **Pair Analysis** | Attempted vs verified vs both-registered pairs by geometry type, camera pair, angle pair |
| **Inter-Camera Connectivity** | 4×4 matrices for verified matches and both-registered pairs |

Outputs `post_analysis.json` for downstream consumption.

### 4.4 Rerun 3D Visualization

**File**: `UTIL_GlobalRemap_PostAnalysis.py`

When invoked with `--rerun`, visualizes:
- 3D point cloud from COLMAP reconstruction
- Registered camera frustums (color-coded by camera)
- COLMAP camera trajectories (sorted by frame index, as `LineStrips3D`)
- SLAM trajectory from `scene_graph.json` (keyframes sorted by timestamp)
- Time-varying camera poses for interactive scrubbing via `rr.set_time_sequence`

---

## 5. Run Results Summary

### Run `20260311_214411_soft-spire` (2-camera, stride=1)

- 928 staged images, 6260 pairs
- 5513 verified matches (MASt3R), mapper produced reconstruction
- ~50% registration rate; cam2/cam3 at 0% (no pairs generated for them)

### Run `4cam_stride2` (4-camera, stride=2)

- Extended stereo pairs to all 4 cameras
- Per-joint pair files confirmed correct pair generation for cam0-cam2, cam1-cam3
- Awaiting final reconstruction results

---

## 6. Files Modified

| File | Change Type |
|------|------------|
| `KernLib_M3RSfM/kern_scene_graph_sfm.py` | Bug fix (commit, ID mapping) |
| `KernLib_M3RSfM/kern_colmap_runner.py` | Timeout signature |
| `KernLib_M3RSfM/kern_colmap_mapper.py` | Timeout signature |
| `BURNSCRIPT_global_remap.py` | Dynamic staging, run ID, multi-model, per-joint pairs, pre-report |
| `LORDPIPE_StreamLine.py` | Run stamp architecture, step 4 rewiring |
| `BURNSCRIPT_TEMPLATE_Undistort.py` | Run ID folder naming |
| `kern_global_remap_pairs.py` | `window_timestamp_stride`, `PairResult`, per-joint tracking |
| `UTILREPORT_global_remap.py` | Joint table, joint graphs, pre-match report |
| `UTIL_GlobalRemap_PostAnalysis.py` | New file — post-analysis + Rerun viz |
| `lord_config_slam_integration_0311.yaml` | New config — 4-camera, stride=2 |
