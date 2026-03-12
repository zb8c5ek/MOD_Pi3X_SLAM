"""UTILREPORT_global_remap -- Plotly HTML report for global remap.

Generates an interactive 3D HTML report showing:
  1. SLAM submap centroids and keyframe trajectories (colored per submap)
  2. Submap adjacency edges (sequential neighbors + loop closures)
  3. MASt3R match connectivity between images, aggregated to submap level

Inputs:
  - scene_graph.json  (submap structure, keyframe poses, edges)
  - pairs.txt         (verified image pairs from MASt3R + pycolmap)
  - filelist_relpath   (staged image relpaths, for frame_idx extraction)

Pure visualization: no model loading, no GPU.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_FRAME_IDX_RE = re.compile(r"(\d{6})_")
_CAM_ANGLE_RE = re.compile(r"_(cam\d+)_(p[+-]\d+_y[+-]\d+_r[+-]\d+)\.")


def _extract_frame_idx(filename: str) -> Optional[int]:
    m = _FRAME_IDX_RE.search(filename)
    return int(m.group(1)) if m else None


def _load_scene_graph(sg_path: Path) -> Dict[str, Any]:
    with open(sg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_submap_data(sg: Dict) -> Dict[str, Any]:
    """Extract per-submap centroids and per-keyframe positions.

    Returns dict with:
      submap_centroids: {submap_id: (x,y,z)}
      kf_positions:     {frame_idx: (x,y,z)}  (first cam only, deduped)
      kf_cam_positions: {(frame_idx, cam_name): (x,y,z)}  per-camera poses
      submap_kf_fidxs:  {submap_id: [frame_idx, ...]}
      fidx_to_submap:   {frame_idx: submap_id}  (first submap seen)
    """
    submap_centroids: Dict[int, np.ndarray] = {}
    kf_positions: Dict[int, np.ndarray] = {}
    kf_cam_positions: Dict[Tuple[int, str], np.ndarray] = {}
    submap_kf_fidxs: Dict[int, List[int]] = {}
    fidx_to_submap: Dict[int, int] = {}

    for sm in sg["submaps"]:
        sid = sm["submap_id"]
        positions = []
        fidxs_seen: Set[int] = set()
        fidx_list: List[int] = []

        for kf in sm["keyframes"]:
            fidx = kf.get("frame_idx")
            if fidx is None:
                continue
            pose = kf["pose_cam2world_global"]
            pos = np.array([pose[0][3], pose[1][3], pose[2][3]])

            img_name = kf.get("image_name", "")
            m = _CAM_NAME_RE.search(img_name)
            cam = m.group(1) if m else "cam0"
            kf_cam_positions[(fidx, cam)] = pos

            if fidx not in fidxs_seen:
                fidxs_seen.add(fidx)
                fidx_list.append(fidx)
                positions.append(pos)

                if fidx not in kf_positions:
                    kf_positions[fidx] = pos
                if fidx not in fidx_to_submap:
                    fidx_to_submap[fidx] = sid

        submap_kf_fidxs[sid] = sorted(fidx_list)
        if positions:
            submap_centroids[sid] = np.mean(positions, axis=0)

    return {
        "submap_centroids": submap_centroids,
        "kf_positions": kf_positions,
        "kf_cam_positions": kf_cam_positions,
        "submap_kf_fidxs": submap_kf_fidxs,
        "fidx_to_submap": fidx_to_submap,
    }


def _parse_pairs_txt(pairs_txt: Path) -> List[Tuple[str, str]]:
    """Read verified pairs from pairs.txt (space-separated relpath pairs)."""
    pairs = []
    if not pairs_txt.is_file():
        return pairs
    with open(pairs_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def _aggregate_matches_to_submaps(
    verified_pairs: List[Tuple[str, str]],
    fidx_to_submap: Dict[int, int],
) -> Dict[Tuple[int, int], int]:
    """Count verified matches per submap pair.

    Returns {(sm_a, sm_b): count} with sm_a <= sm_b.
    """
    counts: Dict[Tuple[int, int], int] = {}
    for rp_a, rp_b in verified_pairs:
        fname_a = rp_a.rsplit("/", 1)[-1]
        fname_b = rp_b.rsplit("/", 1)[-1]
        fidx_a = _extract_frame_idx(fname_a)
        fidx_b = _extract_frame_idx(fname_b)
        if fidx_a is None or fidx_b is None:
            continue
        sm_a = fidx_to_submap.get(fidx_a)
        sm_b = fidx_to_submap.get(fidx_b)
        if sm_a is None or sm_b is None:
            continue
        key = (min(sm_a, sm_b), max(sm_a, sm_b))
        counts[key] = counts.get(key, 0) + 1
    return counts


_CAM_NAME_RE = re.compile(r"(cam\d+)")


def _cam_from_subfolder(sf: str) -> str:
    m = _CAM_NAME_RE.search(sf)
    return m.group(1) if m else sf


def _build_joint_graph_figure(
    joint_label: str,
    joint_pairs: List[Tuple[str, str]],
    kf_positions: Dict[int, np.ndarray],
    kf_cam_positions: Dict[Tuple[int, str], np.ndarray],
    fidx_to_submap: Dict[int, int],
    sm_color: Dict[int, str],
) -> "go.Figure":
    """Build a 3D Plotly figure showing the pair graph for one joint.

    Nodes use real per-camera 3D poses from the scene graph (each camera on
    the rig has a distinct position/orientation).  Edges are colored by type:
    green=stereo (same timestamp, diff cam), blue=temporal (same cam, diff
    timestamp), orange=cross-cam (diff timestamp AND diff cam).
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Collect unique (frame_idx, camera) nodes referenced by pairs
    involved_fidxs: Set[int] = set()
    involved_nodes: Set[Tuple[int, str]] = set()
    for rp_a, rp_b in joint_pairs:
        for rp in (rp_a, rp_b):
            fidx = _extract_frame_idx(rp.rsplit("/", 1)[-1])
            sf = rp.split("/")[0] if "/" in rp else ""
            cam = _cam_from_subfolder(sf)
            if fidx is not None:
                involved_fidxs.add(fidx)
                involved_nodes.add((fidx, cam))

    def _node_pos(fidx: int, cam: str) -> Optional[np.ndarray]:
        """Look up real camera position; fall back to rig-centre."""
        pos = kf_cam_positions.get((fidx, cam))
        if pos is not None:
            return pos
        return kf_positions.get(fidx)

    sorted_cams = sorted({c for _, c in involved_nodes})

    # Keyframe positions as nodes — one marker set per (submap, camera)
    sm_cam_groups: Dict[Tuple[int, str], List[Tuple[int, np.ndarray]]] = {}
    for fidx, cam in sorted(involved_nodes):
        sid = fidx_to_submap.get(fidx)
        if sid is None:
            continue
        pos = _node_pos(fidx, cam)
        if pos is not None:
            sm_cam_groups.setdefault((sid, cam), []).append((fidx, pos))

    cam_marker = {"cam0": "circle", "cam1": "diamond",
                  "cam2": "square", "cam3": "cross"}
    for (sid, cam), entries in sorted(sm_cam_groups.items()):
        xs = [p[0] for _, p in entries]
        ys = [p[1] for _, p in entries]
        zs = [p[2] for _, p in entries]
        texts = [f"f{fidx} sm{sid} {cam}" for fidx, _ in entries]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=4, color=sm_color.get(sid, "gray"),
                        symbol=cam_marker.get(cam, "circle")),
            text=texts, hoverinfo="text",
            name=f"sm{sid:03d}/{cam}", legendgroup=f"sm{sid}",
            showlegend=(cam == sorted_cams[0]),
        ))

    # Classify and draw pair edges: stereo / temporal / cross-camera
    edge_bins: Dict[str, List[List[float]]] = {
        "stereo": [[], [], []], "temporal": [[], [], []], "cross": [[], [], []],
    }
    for rp_a, rp_b in joint_pairs:
        fa = _extract_frame_idx(rp_a.rsplit("/", 1)[-1])
        fb = _extract_frame_idx(rp_b.rsplit("/", 1)[-1])
        if fa is None or fb is None:
            continue
        sf_a = rp_a.split("/")[0] if "/" in rp_a else ""
        sf_b = rp_b.split("/")[0] if "/" in rp_b else ""
        cam_a, cam_b = _cam_from_subfolder(sf_a), _cam_from_subfolder(sf_b)
        pa = _node_pos(fa, cam_a)
        pb = _node_pos(fb, cam_b)
        if pa is None or pb is None:
            continue

        same_ts = (fa == fb)
        same_cam = (cam_a == cam_b)

        if same_ts and not same_cam:
            key = "stereo"
        elif not same_ts and same_cam:
            key = "temporal"
        else:
            key = "cross"

        edge_bins[key][0].extend([pa[0], pb[0], None])
        edge_bins[key][1].extend([pa[1], pb[1], None])
        edge_bins[key][2].extend([pa[2], pb[2], None])

    edge_style = {
        "temporal": ("temporal", dict(width=1, color="rgba(50,100,220,0.3)")),
        "stereo":  ("stereo",   dict(width=2, color="rgba(0,180,0,0.5)")),
        "cross":   ("cross-cam", dict(width=1.5, color="rgba(220,120,0,0.5)")),
    }
    for key in ("temporal", "stereo", "cross"):
        lx, ly, lz = edge_bins[key]
        if not lx:
            continue
        label, style = edge_style[key]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines", line=style,
            name=label, hoverinfo="skip",
        ))

    def _count(bins: List[List[float]]) -> int:
        return len([x for x in bins[0] if x is not None]) // 2

    n_stereo = _count(edge_bins["stereo"])
    n_temporal = _count(edge_bins["temporal"])
    n_cross = _count(edge_bins["cross"])
    fig.update_layout(
        title=dict(
            text=(f"<b>{joint_label}</b><br>"
                  f"<span style='font-size:12px'>"
                  f"{len(involved_fidxs)} timestamps | "
                  f"{len(joint_pairs)} pairs "
                  f"({n_stereo} stereo + {n_temporal} temporal + {n_cross} cross-cam)"
                  f"</span>"),
            x=0.5,
        ),
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        height=500, margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def generate_report(
    scene_graph_json: Path,
    pairs_txt: Path,
    output_html: Path,
    custom_pairs: Optional[List[Tuple[str, str]]] = None,
    joints: Optional[List[Dict[str, Any]]] = None,
    pairs_per_joint_dir: Optional[Path] = None,
    title: str = "Global Remap Report",
) -> Path:
    """Generate interactive Plotly HTML report.

    Args:
        scene_graph_json: Path to scene_graph.json.
        pairs_txt:        Path to pairs.txt (verified MASt3R pairs).
        output_html:      Where to write the HTML file.
        custom_pairs:     Original input pairs (before matching), for
                          showing attempted vs verified connectivity.
        joints:           Per-submap-joint pair metadata from PairResult.
        pairs_per_joint_dir: Directory containing per-joint pair .txt files.
        title:            Page title.

    Returns:
        Path to the written HTML file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sg = _load_scene_graph(scene_graph_json)
    sd = _build_submap_data(sg)
    centroids = sd["submap_centroids"]
    kf_positions = sd["kf_positions"]
    kf_cam_positions = sd["kf_cam_positions"]
    submap_kf_fidxs = sd["submap_kf_fidxs"]
    fidx_to_submap = sd["fidx_to_submap"]

    verified_pairs = _parse_pairs_txt(pairs_txt)
    sm_match_counts = _aggregate_matches_to_submaps(verified_pairs, fidx_to_submap)

    n_submaps = len(centroids)
    summary = sg.get("summary", {})

    # ── Color palette (one per submap) ──
    import plotly.express as px
    palette = px.colors.qualitative.Alphabet
    sm_ids = sorted(centroids.keys())
    sm_color = {sid: palette[i % len(palette)] for i, sid in enumerate(sm_ids)}

    # ── Figure: 3D scene ──
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Submap Trajectories", "Match Connectivity"],
        horizontal_spacing=0.02,
    )

    # --- Panel 1: Submap trajectories ---
    for sid in sm_ids:
        fidxs = submap_kf_fidxs.get(sid, [])
        xs, ys, zs, texts = [], [], [], []
        for fidx in fidxs:
            pos = kf_positions.get(fidx)
            if pos is not None:
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])
                texts.append(f"sm{sid} f{fidx}")

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            marker=dict(size=2, color=sm_color[sid]),
            line=dict(width=2, color=sm_color[sid]),
            name=f"submap_{sid:03d}",
            text=texts,
            hoverinfo="text+name",
            legendgroup=f"sm{sid}",
        ), row=1, col=1)

    # Submap centroids (larger markers)
    cx = [centroids[s][0] for s in sm_ids]
    cy = [centroids[s][1] for s in sm_ids]
    cz = [centroids[s][2] for s in sm_ids]
    ctxt = [f"submap_{s:03d} ({len(submap_kf_fidxs.get(s, []))} KFs)" for s in sm_ids]
    ccolors = [sm_color[s] for s in sm_ids]

    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="markers+text",
        marker=dict(size=8, color=ccolors, symbol="diamond",
                    line=dict(width=1, color="black")),
        text=[f"sm{s}" for s in sm_ids],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=ctxt,
        hoverinfo="text",
        name="centroids",
        showlegend=False,
    ), row=1, col=1)

    # Submap adjacency edges from scene graph
    for sm_a in sm_ids:
        for sm_b in sm_ids:
            if sm_a >= sm_b:
                continue
            fidxs_a = set(submap_kf_fidxs.get(sm_a, []))
            fidxs_b = set(submap_kf_fidxs.get(sm_b, []))
            if fidxs_a & fidxs_b:
                ca, cb = centroids[sm_a], centroids[sm_b]
                fig.add_trace(go.Scatter3d(
                    x=[ca[0], cb[0]], y=[ca[1], cb[1]], z=[ca[2], cb[2]],
                    mode="lines",
                    line=dict(width=1, color="rgba(150,150,150,0.4)", dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=1)

    # --- Panel 2: Match connectivity ---
    # Redraw centroids
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="markers+text",
        marker=dict(size=8, color=ccolors, symbol="diamond",
                    line=dict(width=1, color="black")),
        text=[f"sm{s}" for s in sm_ids],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=ctxt,
        hoverinfo="text",
        name="centroids",
        showlegend=False,
    ), row=1, col=2)

    # Match edges (width proportional to match count)
    max_count = max(sm_match_counts.values()) if sm_match_counts else 1
    for (sm_a, sm_b), count in sorted(sm_match_counts.items()):
        if sm_a not in centroids or sm_b not in centroids:
            continue
        ca, cb = centroids[sm_a], centroids[sm_b]
        width = 1 + 6 * (count / max_count)
        is_intra = (sm_a == sm_b)
        color = "rgba(0,150,0,0.6)" if is_intra else "rgba(220,50,50,0.6)"
        label = f"sm{sm_a}-sm{sm_b}: {count} matches"
        if is_intra:
            label = f"sm{sm_a} intra: {count} matches"
        fig.add_trace(go.Scatter3d(
            x=[ca[0], cb[0]], y=[ca[1], cb[1]], z=[ca[2], cb[2]],
            mode="lines",
            line=dict(width=width, color=color),
            hovertext=[label, label],
            hoverinfo="text",
            showlegend=False,
        ), row=1, col=2)

    # ── Layout ──
    scene_layout = dict(
        aspectmode="data",
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:12px'>"
                f"{n_submaps} submaps | "
                f"{summary.get('num_keyframes', len(kf_positions))} keyframes | "
                f"{len(verified_pairs)} verified pairs | "
                f"{len(sm_match_counts)} submap connections"
                f"</span>"
            ),
            x=0.5,
        ),
        scene=scene_layout,
        scene2=scene_layout,
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=10),
        ),
    )

    # ── Stats table below ──
    stats_rows = []
    for sm_a, sm_b in sorted(sm_match_counts.keys()):
        count = sm_match_counts[(sm_a, sm_b)]
        link_type = "intra" if sm_a == sm_b else "inter"
        stats_rows.append((f"sm{sm_a:03d}", f"sm{sm_b:03d}", link_type, count))

    if stats_rows:
        header = "<tr><th>Submap A</th><th>Submap B</th><th>Type</th><th>Verified Matches</th></tr>"
        rows_html = "".join(
            f"<tr><td>{a}</td><td>{b}</td><td>{t}</td><td>{c}</td></tr>"
            for a, b, t, c in stats_rows
        )
        table_html = (
            '<div style="max-width:600px;margin:20px auto;font-family:sans-serif">'
            '<h3>Match Connectivity by Submap Pair</h3>'
            f'<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;width:100%">'
            f'{header}{rows_html}</table>'
            f'<p>Total verified pairs: {len(verified_pairs)} | '
            f'Submap connections: {len(sm_match_counts)}</p>'
            '</div>'
        )
    else:
        table_html = '<div style="margin:20px auto"><p>No verified matches found.</p></div>'

    # ── Joints table ──
    if joints:
        j_header = ("<tr><th>Joint</th><th>Regime</th><th>Submaps</th>"
                     "<th>Timestamps</th><th>Pairs</th><th>Stereo</th><th>Temporal</th></tr>")
        j_rows = []
        for j in joints:
            regime_color = "#e3f2fd" if j["regime"] == "temporal_stitch" else "#fce4ec"
            j_rows.append(
                f'<tr style="background:{regime_color}">'
                f'<td><code>{j["label"]}</code></td>'
                f'<td>{j["regime"]}</td>'
                f'<td>{j["submaps"]}</td>'
                f'<td>{j["n_timestamps"]}</td>'
                f'<td><b>{j["n_pairs"]}</b></td>'
                f'<td>{j.get("n_stereo", 0)}</td>'
                f'<td>{j.get("n_temporal", 0)}</td>'
                f'</tr>'
            )
        total_pairs = sum(j["n_pairs"] for j in joints)
        joints_html = (
            '<div style="max-width:900px;margin:20px auto;font-family:sans-serif">'
            '<h3>Pair Generation by Submap Joint</h3>'
            f'<table border="1" cellpadding="4" cellspacing="0" '
            f'style="border-collapse:collapse;width:100%;font-size:13px">'
            f'{j_header}{"".join(j_rows)}</table>'
            f'<p>Total joints: {len(joints)} | '
            f'TemporalStitch: {sum(1 for j in joints if j["regime"] == "temporal_stitch")} | '
            f'LCStitch: {sum(1 for j in joints if j["regime"] == "lc_stitch")} | '
            f'Total attempted pairs: {total_pairs}</p>'
            '</div>'
        )
    else:
        joints_html = ""

    # ── Joint graph plots (one representative per regime) ──
    joint_graphs_html = ""
    if joints and pairs_per_joint_dir and Path(pairs_per_joint_dir).is_dir():
        ts_joints = [j for j in joints if j["regime"] == "temporal_stitch"]
        lc_joints = [j for j in joints if j["regime"] == "lc_stitch"]

        representative_joints = []
        if ts_joints:
            representative_joints.append(
                max(ts_joints, key=lambda j: j["n_pairs"]))
        if lc_joints:
            representative_joints.append(
                max(lc_joints, key=lambda j: j["n_pairs"]))

        graph_figs = []
        for jmeta in representative_joints:
            pair_file = Path(pairs_per_joint_dir) / f"{jmeta['label']}.txt"
            if not pair_file.is_file():
                continue
            jpairs = _parse_pairs_txt(pair_file)
            if not jpairs:
                continue
            jfig = _build_joint_graph_figure(
                jmeta["label"], jpairs,
                kf_positions, kf_cam_positions,
                fidx_to_submap, sm_color,
            )
            graph_figs.append(jfig.to_html(full_html=False, include_plotlyjs=False))

        if graph_figs:
            joint_graphs_html = (
                '<div style="max-width:1200px;margin:20px auto;font-family:sans-serif">'
                '<h3>Representative Joint Pair Graphs</h3>'
                '<p style="font-size:12px;color:#666">'
                'Green edges = stereo pairs (same timestamp, different camera). '
                'Blue edges = temporal pairs (same camera, different timestamp). '
                'Orange edges = cross-camera pairs (different timestamp AND different camera). '
                'One representative shown per regime (largest joint).</p>'
                + "".join(graph_figs)
                + '</div>'
            )

    # Write HTML with plotly figure + stats table + joints table + joint graphs
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body style="margin:0;padding:0;background:#fafafa">
{plot_html}
{joints_html}
{joint_graphs_html}
{table_html}
</body></html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info("Report written: %s", output_html)
    return output_html
