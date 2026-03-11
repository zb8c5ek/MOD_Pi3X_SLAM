"""kern_global_remap_pairs -- Two-regime pair generation for global remap.

Reads a SLAM ``scene_graph.json`` and generates MASt3R matching pairs using
two complementary regimes:

**TemporalStitch** (sequential neighbor submaps):
  - Stereo pairs: configurable cross-camera angle pairings at each timestamp.
  - Temporal chains: ALL cam/angle subfolders linked across consecutive
    timestamps within a configurable ``window_size``.

**LCStitch** (loop-closure-connected submaps):
  - Stereo pairs: configurable (typically SLAM-camera-centric).
  - Temporal chains: only ``slam_cameras`` subfolders get temporal links.
  - ``window_size`` controls temporal depth (0 = all timestamps in group).

Both regimes use ``stereo_pairs`` entries that specify which camera pairs to
link and, optionally, which angles per camera.  When angles are omitted for a
camera in a stereo entry, ALL angles from the ``rigs`` config are used.

Pure algorithm: reads JSON + config dicts, returns list of (relpath_a, relpath_b).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_FRAME_IDX_RE = re.compile(r"(\d{6})_")


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def _build_submap_fidxs(submaps: list) -> Dict[int, Set[int]]:
    """Map submap_id -> set of frame indices."""
    result: Dict[int, Set[int]] = {}
    for sm in submaps:
        sid = sm["submap_id"]
        result[sid] = set(kf["frame_idx"] for kf in sm["keyframes"])
    return result


def _build_sequential_neighbors(
    submaps: list,
    kf_edges: list,
    submap_fidxs: Dict[int, Set[int]],
) -> Dict[int, Set[int]]:
    """Sequential neighbor links (overlap edges + shared frame indices)."""
    neighbors: Dict[int, Set[int]] = {}

    for e in kf_edges:
        if e.get("type") == "overlap":
            a, b = e.get("submap_a"), e.get("submap_b")
            if a is not None and b is not None:
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)

    sid_list = sorted(submap_fidxs.keys())
    for i in range(len(sid_list)):
        for j in range(i + 1, len(sid_list)):
            sa, sb = sid_list[i], sid_list[j]
            if submap_fidxs[sa] & submap_fidxs[sb]:
                neighbors.setdefault(sa, set()).add(sb)
                neighbors.setdefault(sb, set()).add(sa)

    return neighbors


def _expand_neighbors(
    neighbors: Dict[int, Set[int]],
    seed: int,
    hops: int,
) -> Set[int]:
    """BFS expansion from *seed* up to *hops* steps.  hops=0 → {seed} only."""
    visited = {seed}
    frontier = {seed}
    for _ in range(hops):
        nxt: Set[int] = set()
        for sid in frontier:
            for nb in neighbors.get(sid, set()):
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return visited


# ---------------------------------------------------------------------------
# Stereo-pair expansion
# ---------------------------------------------------------------------------

def _resolve_stereo_pairs(
    stereo_cfg: List[Dict[str, Any]],
    rig_angles: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Expand stereo_pairs config into a flat list of (subfolder_a, subfolder_b).

    Each stereo entry has ``cameras: [camA, camB]`` and optional per-camera
    angle lists.  Missing angle lists default to ALL angles from ``rig_angles``.
    The output is a cross-product of cam_angle subfolders.

    Example::

        entry = {"cameras": ["cam0","cam1"], "cam0": ["a1","a2"], "cam1": ["b1"]}
        → [("cam0_a1","cam1_b1"), ("cam0_a2","cam1_b1")]
    """
    subfolder_pairs: List[Tuple[str, str]] = []

    for entry in stereo_cfg:
        cams = entry.get("cameras", [])
        if len(cams) != 2:
            logger.warning("stereo_pairs entry needs exactly 2 cameras, got %s", cams)
            continue
        cam_a, cam_b = cams[0], cams[1]
        angles_a = entry.get(cam_a, rig_angles.get(cam_a, []))
        angles_b = entry.get(cam_b, rig_angles.get(cam_b, []))

        for ang_a, ang_b in product(angles_a, angles_b):
            sf_a = f"{cam_a}_{ang_a}"
            sf_b = f"{cam_b}_{ang_b}"
            if sf_a != sf_b:
                subfolder_pairs.append((sf_a, sf_b))

    return subfolder_pairs


# ---------------------------------------------------------------------------
# Index-building helpers
# ---------------------------------------------------------------------------

def _build_file_indices(
    filelist_relpath: List[str],
) -> Tuple[Dict[int, List[str]], Dict[str, Dict[int, str]], Set[str]]:
    """Parse relpaths into lookup structures.

    Returns:
        fidx_to_relpaths: frame_idx → [relpath, ...]
        subfolder_fidx_rp: subfolder → {frame_idx: relpath}
        all_subfolders:    set of all subfolder names
    """
    fidx_to_relpaths: Dict[int, List[str]] = defaultdict(list)
    subfolder_fidx_rp: Dict[str, Dict[int, str]] = defaultdict(dict)

    for rp in filelist_relpath:
        parts = rp.split("/")
        filename = parts[-1]
        subfolder = parts[0] if len(parts) >= 2 else ""
        m = _FRAME_IDX_RE.match(filename)
        if m:
            fidx = int(m.group(1))
            fidx_to_relpaths[fidx].append(rp)
            subfolder_fidx_rp[subfolder][fidx] = rp

    all_subfolders = set(subfolder_fidx_rp.keys())
    return dict(fidx_to_relpaths), dict(subfolder_fidx_rp), all_subfolders


# ---------------------------------------------------------------------------
# Core pair-addition with dedup
# ---------------------------------------------------------------------------

class _PairCollector:
    """Accumulates deduplicated (relpath_a, relpath_b) pairs with counters."""

    def __init__(self):
        self._seen: Set[Tuple[str, str]] = set()
        self.pairs: List[Tuple[str, str]] = []
        self.n_stereo = 0
        self.n_temporal = 0

    def add(self, a: str, b: str, kind: str = "stereo") -> bool:
        key = (a, b) if a < b else (b, a)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.pairs.append(key)
        if kind == "stereo":
            self.n_stereo += 1
        else:
            self.n_temporal += 1
        return True


# ---------------------------------------------------------------------------
# Temporal-chain helper
# ---------------------------------------------------------------------------

def _add_temporal_chains(
    collector: _PairCollector,
    subfolders: Set[str],
    subfolder_fidx_rp: Dict[str, Dict[int, str]],
    fidxs: Set[int],
    window_size: int,
):
    """Add same-subfolder temporal pairs for *subfolders* within *fidxs*.

    ``window_size`` = number of forward timestamps to pair with.
    0 means pair ALL timestamps in the set with each other (full mesh within
    the same subfolder).
    """
    for sf in subfolders:
        fidx_map = subfolder_fidx_rp.get(sf)
        if not fidx_map:
            continue
        ordered = sorted(f for f in fidx_map if f in fidxs)
        if len(ordered) < 2:
            continue

        if window_size == 0:
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    collector.add(fidx_map[ordered[i]], fidx_map[ordered[j]], "temporal")
        else:
            for i, fidx_a in enumerate(ordered):
                for step in range(1, window_size + 1):
                    if i + step >= len(ordered):
                        break
                    collector.add(fidx_map[fidx_a], fidx_map[ordered[i + step]], "temporal")


# ---------------------------------------------------------------------------
# Stereo-pair application at timestamps
# ---------------------------------------------------------------------------

def _add_stereo_at_timestamps(
    collector: _PairCollector,
    resolved_stereo: List[Tuple[str, str]],
    subfolder_fidx_rp: Dict[str, Dict[int, str]],
    fidxs: Set[int],
):
    """At each timestamp in *fidxs*, add stereo pairs between subfolders."""
    for fidx in sorted(fidxs):
        for sf_a, sf_b in resolved_stereo:
            rp_a = subfolder_fidx_rp.get(sf_a, {}).get(fidx)
            rp_b = subfolder_fidx_rp.get(sf_b, {}).get(fidx)
            if rp_a and rp_b:
                collector.add(rp_a, rp_b, "stereo")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pairs(
    scene_graph_json: str,
    filelist_relpath: List[str],
    temporal_stitch_cfg: Dict[str, Any],
    lc_stitch_cfg: Dict[str, Any],
    rig_angles: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Two-regime pair generation.

    Args:
        scene_graph_json:   Path to scene_graph.json.
        filelist_relpath:   Staged image relative paths.
        temporal_stitch_cfg: Config dict for TemporalStitch regime.
        lc_stitch_cfg:      Config dict for LCStitch regime.
        rig_angles:         {cam_name: [angle, ...]} from rigs config,
                            used as default when stereo_pairs omit angles.

    Returns:
        Deduplicated list of (relpath_a, relpath_b) pairs.
    """
    sg = json.load(open(scene_graph_json, "r", encoding="utf-8"))
    submaps = sg["submaps"]
    kf_edges = sg.get("kf_edges", [])
    loop_closures = sg.get("loop_closures", [])

    submap_fidxs = _build_submap_fidxs(submaps)
    seq_neighbors = _build_sequential_neighbors(submaps, kf_edges, submap_fidxs)

    fidx_to_relpaths, subfolder_fidx_rp, all_subfolders = _build_file_indices(
        filelist_relpath,
    )

    collector = _PairCollector()

    # ------------------------------------------------------------------
    # Regime 1: TemporalStitch (sequential neighbors)
    # ------------------------------------------------------------------
    ts_stereo_cfg = temporal_stitch_cfg.get("stereo_pairs", [])
    ts_window = temporal_stitch_cfg.get("window_size", 1)
    ts_resolved = _resolve_stereo_pairs(ts_stereo_cfg, rig_angles)

    n_ts_before = len(collector.pairs)

    for sid in sorted(submap_fidxs.keys()):
        group_sids = _expand_neighbors(seq_neighbors, sid, hops=1)
        group_fidxs: Set[int] = set()
        for gsid in group_sids:
            group_fidxs |= submap_fidxs.get(gsid, set())

        _add_stereo_at_timestamps(collector, ts_resolved, subfolder_fidx_rp, group_fidxs)
        _add_temporal_chains(collector, all_subfolders, subfolder_fidx_rp, group_fidxs, ts_window)

    n_ts = len(collector.pairs) - n_ts_before

    # ------------------------------------------------------------------
    # Regime 2: LCStitch (loop-closure connections)
    # ------------------------------------------------------------------
    lc_stereo_cfg = lc_stitch_cfg.get("stereo_pairs", [])
    lc_window = lc_stitch_cfg.get("window_size", 0)
    lc_resolved = _resolve_stereo_pairs(lc_stereo_cfg, rig_angles)

    slam_cameras_cfg = lc_stitch_cfg.get("slam_cameras", {})
    slam_subfolders: Set[str] = set()
    for cam, angles in slam_cameras_cfg.items():
        for ang in angles:
            slam_subfolders.add(f"{cam}_{ang}")

    n_lc_before = len(collector.pairs)

    for lc in loop_closures:
        sa = lc.get("submap_a", lc.get("submap_id_a"))
        sb = lc.get("submap_b", lc.get("submap_id_b"))
        if sa is None or sb is None:
            continue
        lc_fidxs = (submap_fidxs.get(sa, set()) | submap_fidxs.get(sb, set()))
        if not lc_fidxs:
            continue

        _add_stereo_at_timestamps(collector, lc_resolved, subfolder_fidx_rp, lc_fidxs)

        temporal_sfs = slam_subfolders if slam_subfolders else all_subfolders
        _add_temporal_chains(collector, temporal_sfs, subfolder_fidx_rp, lc_fidxs, lc_window)

    n_lc = len(collector.pairs) - n_lc_before

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_seq_links = sum(len(v) for v in seq_neighbors.values()) // 2
    logger.info(
        "Two-regime pairs: %d submaps, %d seq-links, %d LCs | "
        "TemporalStitch=%d (window=%d) | LCStitch=%d (window=%d) | "
        "total=%d unique (%d stereo + %d temporal)",
        len(submap_fidxs), n_seq_links, len(loop_closures),
        n_ts, ts_window,
        n_lc, lc_window,
        len(collector.pairs), collector.n_stereo, collector.n_temporal,
    )
    return collector.pairs
