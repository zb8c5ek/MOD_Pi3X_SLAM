"""kern_global_remap_pairs -- Submap-aware pair generation for global remap.

Reads a SLAM ``scene_graph.json`` and generates MASt3R matching pairs
based on submap adjacency.  Pairs are generated between frames that share
a submap or belong to neighboring submaps (overlap + loop closure).

Strategy ``submap-window``:
    For each submap S, collect all frame indices in S plus frame indices
    in submaps within ``window_size`` hops on the adjacency graph.  All
    cross-camera/angle pairs within each such connected frame set are
    generated.  When ``include_lc`` is True, loop-closure-connected
    submaps are also treated as neighbors (distance = 1).

Pure algorithm: reads JSON, returns list of (relpath_a, relpath_b) pairs.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
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

    # From overlap edges
    for e in kf_edges:
        if e.get("type") == "overlap":
            a = e.get("submap_a")
            b = e.get("submap_b")
            if a is not None and b is not None:
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)

    # From frame_idx overlap (submaps sharing keyframe timestamps)
    sid_list = sorted(submap_fidxs.keys())
    for i in range(len(sid_list)):
        for j in range(i + 1, len(sid_list)):
            sa, sb = sid_list[i], sid_list[j]
            if submap_fidxs[sa] & submap_fidxs[sb]:
                neighbors.setdefault(sa, set()).add(sb)
                neighbors.setdefault(sb, set()).add(sa)

    # Loop closures as direct neighbors
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
) -> List[Tuple[str, str]]:
    """Generate MASt3R pairs using submap adjacency from scene_graph.json.

    For each submap, pairs are generated between all frames in that submap
    and all frames in submaps within ``window_size`` hops (including
    loop-closure connections if ``include_lc``).

    Args:
        scene_graph_json: Path to scene_graph.json from SLAM output.
        filelist_relpath: Image relative paths (staged images).
        window_size:      Number of adjacency hops to include (1 = immediate
                          neighbors, 2 = neighbors of neighbors, etc.)
        include_lc:       Treat loop-closure-connected submaps as distance-1
                          neighbors before BFS expansion.

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

    # Map filelist relpath -> frame_idx
    fidx_to_relpaths: Dict[int, List[str]] = defaultdict(list)
    for rp in filelist_relpath:
        filename = rp.split("/")[-1]
        m = _FRAME_IDX_RE.match(filename)
        if m:
            fidx_to_relpaths[int(m.group(1))].append(rp)

    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def _add_pair(a: str, b: str):
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    for sid in sorted(submap_fidxs.keys()):
        connected_sids = _expand_neighbors(neighbors, sid, window_size)
        connected_fidxs = set()
        for csid in connected_sids:
            connected_fidxs |= submap_fidxs.get(csid, set())

        # Sort for deterministic ordering
        connected_fidxs_sorted = sorted(connected_fidxs)

        # Intra-timestamp: stereo/multi-angle pairs at each frame_idx
        for fidx in connected_fidxs_sorted:
            rps = fidx_to_relpaths.get(fidx, [])
            for a, b in combinations(rps, 2):
                _add_pair(a, b)

        # Cross-timestamp: pair every image at one timestamp with every
        # image at a different timestamp within the connected group
        for i, fidx_a in enumerate(connected_fidxs_sorted):
            rps_a = fidx_to_relpaths.get(fidx_a, [])
            for fidx_b in connected_fidxs_sorted[i + 1:]:
                rps_b = fidx_to_relpaths.get(fidx_b, [])
                for ra in rps_a:
                    for rb in rps_b:
                        _add_pair(ra, rb)

    n_lc = len(loop_closures)
    n_links = sum(len(v) for v in neighbors.values()) // 2
    logger.info(
        "Submap-window pairs: %d submaps, %d neighbor links, %d LCs, "
        "window=%d -> %d unique pairs",
        len(submap_fidxs), n_links, n_lc, window_size, len(pairs),
    )
    return pairs
