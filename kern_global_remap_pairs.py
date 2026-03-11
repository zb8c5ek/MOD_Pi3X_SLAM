"""kern_global_remap_pairs -- Submap-aware pair generation for global remap.

Reads a SLAM ``scene_graph.json`` and generates MASt3R matching pairs
based on submap adjacency with structured pairing:

1. **Intra-timestamp (stereo):** All C(K,2) cam/angle combos at each
   keyframe timestamp.  With K=8 views this gives 28 pairs per timestamp.

2. **Cross-timestamp (temporal chains):** Only **same cam_angle** at
   adjacent timestamps within the connected submap group.  A configurable
   ``temporal_window`` controls how many future timestamps each frame is
   paired with (default 1 = immediate next timestamp only).

When ``include_lc`` is True, loop-closure-connected submaps are treated
as distance-1 neighbors before BFS expansion.

Pure algorithm: reads JSON, returns list of (relpath_a, relpath_b) pairs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

_FRAME_IDX_RE = re.compile(r"(\d{6})_")


def _build_adjacency(
    submaps: list,
    kf_edges: list,
    loop_closures: list,
    include_lc: bool,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Build submap adjacency and per-submap frame index sets.

    Returns:
        (neighbors, submap_fidxs) where neighbors maps submap_id to a set
        of adjacent submap_ids, and submap_fidxs maps submap_id to a set
        of frame indices.
    """
    submap_fidxs: Dict[int, Set[int]] = {}
    for sm in submaps:
        sid = sm["submap_id"]
        submap_fidxs[sid] = set(kf["frame_idx"] for kf in sm["keyframes"])

    neighbors: Dict[int, Set[int]] = {}

    for e in kf_edges:
        if e.get("type") == "overlap":
            a = e.get("submap_a")
            b = e.get("submap_b")
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

    if include_lc:
        for lc in loop_closures:
            sa = lc.get("submap_a", lc.get("submap_id_a"))
            sb = lc.get("submap_b", lc.get("submap_id_b"))
            if sa is not None and sb is not None:
                neighbors.setdefault(sa, set()).add(sb)
                neighbors.setdefault(sb, set()).add(sa)

    return neighbors, submap_fidxs


def _expand_neighbors(
    neighbors: Dict[int, Set[int]],
    seed: int,
    window_size: int,
) -> Set[int]:
    """BFS expansion from *seed* up to *window_size* hops."""
    visited = {seed}
    frontier = {seed}
    for _ in range(window_size):
        next_frontier: Set[int] = set()
        for sid in frontier:
            for nb in neighbors.get(sid, set()):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def generate_submap_window_pairs(
    scene_graph_json: str,
    filelist_relpath: List[str],
    window_size: int = 1,
    include_lc: bool = True,
    temporal_window: int = 1,
) -> List[Tuple[str, str]]:
    """Generate MASt3R pairs using submap adjacency + structured pairing.

    For each connected submap group (BFS up to ``window_size`` hops):

    - **Intra-timestamp:** all C(K,2) cam/angle pairs at each timestamp
      (multi-view stereo constraint).
    - **Cross-timestamp:** only same cam_angle subfolder at adjacent
      timestamps within the group (temporal continuity).  The number of
      forward timestamps to pair with is controlled by ``temporal_window``.

    Args:
        scene_graph_json: Path to scene_graph.json from SLAM output.
        filelist_relpath: Image relative paths (staged images).
        window_size:      Submap adjacency hops (1 = immediate neighbors).
        include_lc:       Treat LC-connected submaps as distance-1 neighbors.
        temporal_window:  Number of forward timestamps for same-cam-angle
                          temporal chains (default 1).

    Returns:
        Deduplicated list of (relpath_a, relpath_b) pairs.
    """
    sg = json.load(open(scene_graph_json, "r", encoding="utf-8"))
    submaps = sg["submaps"]
    kf_edges = sg.get("kf_edges", [])
    loop_closures = sg.get("loop_closures", [])

    neighbors, submap_fidxs = _build_adjacency(
        submaps, kf_edges, loop_closures, include_lc,
    )

    # Map relpath -> (frame_idx, subfolder)
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

    seen: Set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []

    def _add_pair(a: str, b: str):
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    n_intra = 0
    n_temporal = 0

    for sid in sorted(submap_fidxs.keys()):
        connected_sids = _expand_neighbors(neighbors, sid, window_size)
        connected_fidxs = set()
        for csid in connected_sids:
            connected_fidxs |= submap_fidxs.get(csid, set())

        connected_fidxs_sorted = sorted(connected_fidxs)

        # 1) Intra-timestamp: all cam/angle combos at each timestamp
        for fidx in connected_fidxs_sorted:
            rps = fidx_to_relpaths.get(fidx, [])
            for a, b in combinations(rps, 2):
                before = len(pairs)
                _add_pair(a, b)
                n_intra += len(pairs) - before

        # 2) Cross-timestamp: same subfolder only, within temporal_window
        connected_set = set(connected_fidxs_sorted)
        for subfolder, fidx_map in subfolder_fidx_rp.items():
            fidxs_in_group = sorted(f for f in fidx_map if f in connected_set)
            for i, fidx_a in enumerate(fidxs_in_group):
                rp_a = fidx_map[fidx_a]
                for j in range(1, temporal_window + 1):
                    if i + j >= len(fidxs_in_group):
                        break
                    fidx_b = fidxs_in_group[i + j]
                    rp_b = fidx_map[fidx_b]
                    before = len(pairs)
                    _add_pair(rp_a, rp_b)
                    n_temporal += len(pairs) - before

    n_lc = len(loop_closures)
    n_links = sum(len(v) for v in neighbors.values()) // 2
    logger.info(
        "Submap-window pairs: %d submaps, %d neighbor links, %d LCs, "
        "window=%d, temporal_window=%d -> %d unique pairs "
        "(%d stereo + %d temporal)",
        len(submap_fidxs), n_links, n_lc, window_size, temporal_window,
        len(pairs), n_intra, n_temporal,
    )
    return pairs
