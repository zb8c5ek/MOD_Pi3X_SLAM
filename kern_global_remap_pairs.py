"""kern_global_remap_pairs -- Two-regime pair generation for global remap.

Reads a SLAM ``scene_graph.json`` and generates MASt3R matching pairs using
two complementary regimes:

**TemporalStitch** (sequential neighbor submaps):
  - Stereo pairs: configurable cross-camera angle pairings at each timestamp.
  - Temporal chains: ALL cam/angle subfolders linked across consecutive
    timestamps within a configurable ``window_size``.
  - ``window_timestamp_stride``: step size between paired timestamps
    (default 1 = consecutive; 2 = every other, reaching further).

**LCStitch** (loop-closure-connected submaps):
  - Stereo pairs: configurable (typically SLAM-camera-centric).
  - Temporal chains: only ``slam_cameras`` subfolders get temporal links.
  - ``window_size`` controls temporal depth (0 = all timestamps in group).

Both regimes use ``stereo_pairs`` entries that specify which camera pairs to
link and, optionally, which angles per camera.  When angles are omitted for a
camera in a stereo entry, ALL angles from the ``rigs`` config are used.

Pure algorithm: reads JSON + config dicts, returns PairResult (list-compatible
with per-joint metadata).
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


class PairResult:
    """List-compatible result that also carries per-joint metadata.

    Iterating / indexing / len() work on the combined pair list, so existing
    callers that pass ``PairResult`` as ``custom_pairs`` continue to work.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        joints: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ):
        self.pairs = pairs
        self.joints = joints
        self.summary = summary

    # --- list-like interface so callers can use this as List[Tuple] ---
    def __iter__(self):
        return iter(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __bool__(self):
        return bool(self.pairs)


# ---------------------------------------------------------------------------
# Temporal-chain helper
# ---------------------------------------------------------------------------

def _add_temporal_chains(
    collector: _PairCollector,
    subfolders: Set[str],
    subfolder_fidx_rp: Dict[str, Dict[int, str]],
    fidxs: Set[int],
    window_size: int,
    timestamp_stride: int = 1,
):
    """Add same-subfolder temporal pairs for *subfolders* within *fidxs*.

    ``window_size`` = number of forward timestamps to pair with.
    0 means pair ALL timestamps in the set with each other (full mesh).

    ``timestamp_stride`` = step size between paired timestamps.  stride=1
    pairs consecutive timestamps (i, i+1, i+2, ...); stride=2 pairs every
    other (i, i+2, i+4, ...), reaching further with the same pair count.
    """
    stride = max(1, timestamp_stride)
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
                    target = i + step * stride
                    if target >= len(ordered):
                        break
                    collector.add(fidx_map[fidx_a], fidx_map[ordered[target]], "temporal")


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
) -> PairResult:
    """Two-regime pair generation with per-joint tracking.

    Args:
        scene_graph_json:   Path to scene_graph.json.
        filelist_relpath:   Staged image relative paths.
        temporal_stitch_cfg: Config dict for TemporalStitch regime.
        lc_stitch_cfg:      Config dict for LCStitch regime.
        rig_angles:         {cam_name: [angle, ...]} from rigs config,
                            used as default when stereo_pairs omit angles.

    Returns:
        PairResult -- iterates as List[Tuple[str,str]] for backward
        compatibility; also carries ``.joints`` and ``.summary``.
    """
    with open(scene_graph_json, "r", encoding="utf-8") as f:
        sg = json.load(f)
    submaps = sg["submaps"]
    kf_edges = sg.get("kf_edges", [])
    loop_closures = sg.get("loop_closures", [])

    submap_fidxs = _build_submap_fidxs(submaps)
    seq_neighbors = _build_sequential_neighbors(submaps, kf_edges, submap_fidxs)

    fidx_to_relpaths, subfolder_fidx_rp, all_subfolders = _build_file_indices(
        filelist_relpath,
    )

    collector = _PairCollector()
    joints: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Regime 1: TemporalStitch (sequential neighbors)
    # ------------------------------------------------------------------
    ts_stereo_cfg = temporal_stitch_cfg.get("stereo_pairs", [])
    ts_window = temporal_stitch_cfg.get("window_size", 1)
    ts_stride = temporal_stitch_cfg.get("window_timestamp_stride", 1)
    ts_resolved = _resolve_stereo_pairs(ts_stereo_cfg, rig_angles)

    n_ts_total = 0
    seen_ts_groups: Set[frozenset] = set()

    for sid in sorted(submap_fidxs.keys()):
        group_sids = _expand_neighbors(seq_neighbors, sid, hops=1)
        group_key = frozenset(group_sids)
        if group_key in seen_ts_groups:
            continue
        seen_ts_groups.add(group_key)

        group_fidxs: Set[int] = set()
        for gsid in group_sids:
            group_fidxs |= submap_fidxs.get(gsid, set())

        n_before = len(collector.pairs)
        _add_stereo_at_timestamps(collector, ts_resolved, subfolder_fidx_rp, group_fidxs)
        _add_temporal_chains(collector, all_subfolders, subfolder_fidx_rp,
                             group_fidxs, ts_window, ts_stride)
        n_joint = len(collector.pairs) - n_before

        if n_joint > 0:
            joint_pairs = collector.pairs[n_before:]
            sids_sorted = sorted(group_sids)
            label = "ts_" + "+".join(f"sm{s:03d}" for s in sids_sorted)
            joints.append({
                "label": label,
                "regime": "temporal_stitch",
                "submaps": sids_sorted,
                "n_timestamps": len(group_fidxs),
                "n_pairs": n_joint,
                "n_stereo": sum(1 for a, b in joint_pairs
                                if a.split("/")[0] != b.split("/")[0]
                                and _FRAME_IDX_RE.match(a.rsplit("/", 1)[-1]).group(1) ==
                                    _FRAME_IDX_RE.match(b.rsplit("/", 1)[-1]).group(1)),
                "n_temporal": 0,  # filled below
                "pairs": joint_pairs,
            })
            joints[-1]["n_temporal"] = n_joint - joints[-1]["n_stereo"]
            n_ts_total += n_joint

    # ------------------------------------------------------------------
    # Regime 2: LCStitch (loop-closure connections)
    # ------------------------------------------------------------------
    lc_stereo_cfg = lc_stitch_cfg.get("stereo_pairs", [])
    lc_window = lc_stitch_cfg.get("window_size", 0)
    lc_stride = lc_stitch_cfg.get("window_timestamp_stride", 1)
    lc_resolved = _resolve_stereo_pairs(lc_stereo_cfg, rig_angles)

    slam_cameras_cfg = lc_stitch_cfg.get("slam_cameras", {})
    slam_subfolders: Set[str] = set()
    for cam, angles in slam_cameras_cfg.items():
        for ang in angles:
            slam_subfolders.add(f"{cam}_{ang}")

    n_lc_total = 0

    for lc in loop_closures:
        sa = lc.get("submap_a", lc.get("submap_id_a"))
        sb = lc.get("submap_b", lc.get("submap_id_b"))
        if sa is None or sb is None:
            continue
        lc_fidxs = (submap_fidxs.get(sa, set()) | submap_fidxs.get(sb, set()))
        if not lc_fidxs:
            continue

        n_before = len(collector.pairs)
        _add_stereo_at_timestamps(collector, lc_resolved, subfolder_fidx_rp, lc_fidxs)

        temporal_sfs = slam_subfolders if slam_subfolders else all_subfolders
        _add_temporal_chains(collector, temporal_sfs, subfolder_fidx_rp,
                             lc_fidxs, lc_window, lc_stride)
        n_joint = len(collector.pairs) - n_before

        if n_joint > 0:
            joint_pairs = collector.pairs[n_before:]
            label = f"lc_sm{sa:03d}_sm{sb:03d}"
            joints.append({
                "label": label,
                "regime": "lc_stitch",
                "submaps": sorted({sa, sb}),
                "n_timestamps": len(lc_fidxs),
                "n_pairs": n_joint,
                "n_stereo": 0,
                "n_temporal": n_joint,
                "pairs": joint_pairs,
            })
            n_lc_total += n_joint

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_seq_links = sum(len(v) for v in seq_neighbors.values()) // 2
    summary = {
        "n_submaps": len(submap_fidxs),
        "n_seq_links": n_seq_links,
        "n_loop_closures": len(loop_closures),
        "n_ts_joints": sum(1 for j in joints if j["regime"] == "temporal_stitch"),
        "n_lc_joints": sum(1 for j in joints if j["regime"] == "lc_stitch"),
        "n_ts_pairs": n_ts_total,
        "n_lc_pairs": n_lc_total,
        "ts_window": ts_window,
        "ts_stride": ts_stride,
        "lc_window": lc_window,
        "lc_stride": lc_stride,
        "total": len(collector.pairs),
        "n_stereo": collector.n_stereo,
        "n_temporal": collector.n_temporal,
    }

    logger.info(
        "Two-regime pairs: %d submaps, %d seq-links, %d LCs | "
        "TemporalStitch=%d (window=%d, stride=%d, %d joints) | "
        "LCStitch=%d (window=%d, stride=%d, %d joints) | "
        "total=%d unique (%d stereo + %d temporal)",
        summary["n_submaps"], n_seq_links, len(loop_closures),
        n_ts_total, ts_window, ts_stride, summary["n_ts_joints"],
        n_lc_total, lc_window, lc_stride, summary["n_lc_joints"],
        len(collector.pairs), collector.n_stereo, collector.n_temporal,
    )
    return PairResult(collector.pairs, joints, summary)
