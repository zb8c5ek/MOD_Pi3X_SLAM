# Dual-Edge VO Group Registration

## Problem Statement

VO groups (temporal chunks of an episode) need to be aligned to a common
coordinate frame for downstream quality analysis (Kabsch) and inter-group
pose estimation.  Currently, alignment relies solely on temporal overlap
between adjacent groups -- a fragile sequential chain where errors
accumulate.

MOD_Pi3X_SLAM now produces a globally-optimised SLAM solution (submaps +
loop closures) alongside the VO groups.  We can exploit this to provide a
second, independent alignment path.

## Core Idea

Each VO group can be registered via **two independent edge types**:

| Edge type | Notation | How it works |
|-----------|----------|--------------|
| **Temporal overlap** | `group_i ↔ group_{i+1}` | Adjacent groups share `overlap_frames` timestamps. Corresponding 3D points from those shared frames give a SIM(3) alignment between groups. |
| **Submap anchor** | `group_i ↔ submap_j` | Extend each group's frame set by ±K SLAM keyframes (the **keyframe window**). Those keyframes exist in both the group's COLMAP reconstruction AND in a SLAM submap. Corresponding points give a SIM(3) alignment from group-local to SLAM-global. |

For each group, **evaluate both edge types** and pick the better one (or
use both as redundant constraints).

---

## Architecture

### 1. Parameters

```yaml
# essn_compo_vo.yaml
grouping:
  group_size: 20
  overlap_frames: 5      # temporal overlap (existing)
  keyframe_window: 3        # ±K SLAM keyframes to include per group (new)
```

### 2. Pipeline Flow (within BURNPIPE_VO_Pipeline)

```
 ┌─────────────────────────────────────────────────────────────┐
 │  1. Discover episodes + images             (essn_vo)        │
 │  2. Run SLAM (full episode)                (essn_slam)      │
 │     → submaps, keyframe timeline, global poses              │
 │  3. Build VO groups                        (essn_vo)        │
 │     For each group:                                         │
 │       a) Temporal frames: group_size contiguous timestamps   │
 │       b) Overlap frames: overlap_frames shared with prev/next│
 │       c) KF window frames: ±K nearest SLAM keyframes        │
 │          beyond the group boundary                          │
 │     → each group has: core + overlap + kf_window frames     │
 │  4. Export per-group COLMAP                (util_colmap)     │
 │  5. Compute dual-edge alignment            (kern_dual_edge) │
 │     For each group:                                         │
 │       a) Temporal edge: align overlap region to prev group   │
 │       b) Submap anchor: align kf_window region to SLAM submap│
 │       c) Report both qualities, pick/fuse the better one    │
 │  6. Stage images + write manifest          (essn_vo)        │
 │  7. Write enriched grouping_manifest.json                   │
 │     → includes per-group registration edges + quality        │
 └─────────────────────────────────────────────────────────────┘
```

### 3. Keyframe Window Mechanics

Given SLAM keyframe timeline: `[kf_0, kf_1, kf_2, ..., kf_N]`

For a VO group covering timestamps `[t_start, ..., t_end]`:

1. Find the SLAM keyframes that fall **within** this range:
   `kf_inside = [kf_i | t_start <= kf_i <= t_end]`

2. Find the first keyframe *before* `t_start` on the SLAM timeline,
   then extend backwards by K:
   `kf_pre = SLAM_keyframes[max(0, idx_first_inside - K) : idx_first_inside]`

3. Find the first keyframe *after* `t_end` on the SLAM timeline,
   then extend forwards by K:
   `kf_post = SLAM_keyframes[idx_last_inside + 1 : idx_last_inside + 1 + K]`

4. The group's full frame set becomes:
   `frames = kf_pre + core_temporal_frames + kf_post`

**Edge cases:**
- First group: `kf_pre` may be empty or shorter than K → that's fine.
- Last group: `kf_post` may be empty or shorter than K → that's fine.
- Sparse KF region: fewer than K keyframes available → use whatever exists.
- Keyframe *inside* the group: these are natural anchor points with
  poses in both the group COLMAP and SLAM global frame.

### 4. Dual-Edge Quality Assessment

For each group, compute alignment quality for both edge types:

#### A. Temporal Edge (group ↔ group)

- Source: 3D points from `overlap_frames` shared between `group_i` and
  `group_{i+1}`.
- Method: SIM(3) Umeyama (same as current submap stitching).
- Metrics: Kabsch RMSD, scale `s`, rotation angle, translation norm,
  inlier count.

#### B. Submap Anchor Edge (group ↔ SLAM submap)

- Source: 3D points from `kf_window` keyframes that exist in both the
  group's COLMAP and a SLAM submap.
- Identify which SLAM submap(s) contain the anchor keyframes.
- Extract corresponding 3D points from both the group's reconstruction
  and the SLAM submap (via image names / frame indices).
- Method: SIM(3) Umeyama (same algorithm).
- Metrics: same as temporal.

#### C. Edge Selection / Fusion

For each group, three strategies (configurable):

1. **Pick-best**: Use whichever edge has lower Kabsch RMSD.
2. **Prefer-SLAM**: Always use submap anchor when available and quality
   is acceptable (RMSD < threshold); fall back to temporal if not.
3. **Fuse**: When both edges are good, average the transforms (weighted
   by quality).  When one fails, use the other.

Default: **prefer-SLAM** (since it's globally consistent).

### 5. Enriched Grouping Manifest

`grouping_manifest.json` gains per-group registration metadata:

```json
{
  "groups": [
    {
      "name": "group_000",
      "frame_indices": [2, 5, 8, 11, 14],
      "kf_window_pre": [],
      "kf_window_post": [17, 20, 23],
      "registration": {
        "temporal_edge": {
          "target": "group_001",
          "kabsch_rmsd": 0.032,
          "scale": 1.002,
          "rot_deg": 2.1,
          "n_correspondences": 1523,
          "status": "good"
        },
        "submap_anchor_edge": {
          "target_submap": "submap_000",
          "kabsch_rmsd": 0.018,
          "scale": 0.998,
          "rot_deg": 1.4,
          "n_correspondences": 3210,
          "status": "good"
        },
        "selected_edge": "submap_anchor",
        "selection_reason": "lower RMSD (0.018 vs 0.032)"
      }
    }
  ]
}
```

### 6. New / Modified Modules

| Module | Change | Description |
|--------|--------|-------------|
| `essn_compo_vo.yaml` | Modify | Add `keyframe_window` parameter |
| `essn_vo.py` | Modify | Frame set builder: extend groups with ±K KF window frames |
| `kern_dual_edge.py` | **New** | Dual-edge alignment kernel: computes both edge types, selects/fuses |
| `BURNPIPE_VO_Pipeline.py` | Modify | Wire new step between COLMAP export and manifest writing |
| `util_colmap.py` | Modify (minor) | `export_per_submap_colmap` may need to handle extended frame sets |

### 7. Data Flow Diagram

```
SLAM run
  │
  ├─ keyframe_timeline: [kf_0, kf_1, ..., kf_N]  (frame indices)
  ├─ submap_j → {kf_indices, poses_global, points_3d}
  │
  │  VO group building
  │    │
  │    ├─ core frames:     [t_start ... t_end]  from group_size
  │    ├─ overlap frames:  [t_end-overlap+1 ... t_end]  shared with next
  │    ├─ kf_pre:          [-K keyframes before t_start]
  │    ├─ kf_post:         [+K keyframes after t_end]
  │    └─ full_frames = kf_pre ∪ core ∪ kf_post
  │
  │  Per-group COLMAP reconstruction
  │    │
  │    └─ group_i: poses, points, image_names  (in group-local frame)
  │
  │  Dual-edge alignment
  │    │
  │    ├─ Temporal:     match overlap images in group_i ∩ group_{i+1}
  │    │                → SIM(3)_temporal
  │    │
  │    ├─ Submap anchor: match kf_window images in group_i ∩ submap_j
  │    │                → SIM(3)_slam
  │    │
  │    └─ Select / fuse → T_group_to_global
  │
  └─ grouping_manifest.json  (with registration edges + quality)
```

### 8. Implementation Order

1. **Add `keyframe_window` param** to `essn_compo_vo.yaml` and config chain.
2. **Extend `essn_vo.py`**: after SLAM completes, query the scene graph
   for the keyframe timeline, build extended frame sets per group.
3. **Create `kern_dual_edge.py`**: alignment kernel that takes
   (group COLMAP, SLAM submap, overlap region) and computes both edge
   types with quality metrics.
4. **Wire into BURNPIPE**: call `kern_dual_edge` after per-group COLMAP
   export, before manifest writing.
5. **Enrich manifest**: add per-group `registration` block.
6. **Update downstream** (MOD_KabschAnalyzer): read the registration
   metadata and use the selected transform for quality assessment.

### 9. Quality Classification (downstream)

The Kabsch analyzer can now use per-group registration quality:

| RMSD range | Classification | Action |
|------------|---------------|--------|
| ≤ 0.05 | **good** | Accept |
| 0.05 - 0.15 | **warning** | Flag for review |
| 0.15 - 0.30 | **bad** | Route to remapping |
| > 0.30 or failed | **failed** | Route to Call4Revision |

When both edges exist, report agreement between them as an additional
confidence signal (if temporal and SLAM anchor agree → high confidence).

### 10. Future Extensions

- **Per-image pose transfer**: For non-keyframe images in a group,
  interpolate pose from bracketing SLAM keyframes (already in
  scene_graph.json `frames[].prev_kf / next_kf`).
- **Weighted graph optimisation**: Instead of pick-best, solve a full
  pose graph over all groups with both edge types as constraints
  (weighted by inverse RMSD).
- **Adaptive keyframe window**: Increase K for groups with poor temporal
  edges; decrease for groups already well-constrained.
