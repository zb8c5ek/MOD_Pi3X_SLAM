"""
util_pipeline_report -- Self-contained HTML summary for VO-en-SLAM pipeline.

Generates ``pipeline_summary.html`` at the VO-en-SLAM output root with:

- SLAM 3D trajectory (Plotly)
- VO group summary table
- Dual-edge registration results
- SLAM submap quality (RMSD, loop closures)
- Keyframe agreement (WAFT vs LK)
"""

import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _traj_from_scene_graph(sg: dict) -> List[dict]:
    """Extract submap centroid trajectory from scene graph."""
    traj = []
    for sm in sg.get("submaps", []):
        kfs = sm.get("keyframes", [])
        if not kfs:
            continue
        positions = []
        for kf in kfs:
            pose = kf.get("pose_cam2world_global")
            if pose is not None:
                t = np.array(pose, dtype=np.float64)[:3, 3]
                positions.append(t)
        if positions:
            centroid = np.mean(positions, axis=0)
            traj.append({
                "submap_id": sm["submap_id"],
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]),
                "num_kf": len(kfs),
            })
    return traj


def _kf_traj_from_scene_graph(sg: dict) -> List[dict]:
    """Extract individual keyframe trajectory from scene graph."""
    traj = []
    for sm in sg.get("submaps", []):
        for kf in sm.get("keyframes", []):
            pose = kf.get("pose_cam2world_global")
            if pose is not None:
                t = np.array(pose, dtype=np.float64)[:3, 3]
                traj.append({
                    "x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
                    "submap_id": sm["submap_id"],
                    "frame_idx": kf.get("frame_idx"),
                })
    return traj


def _dual_edge_rows(reg: Dict[str, dict]) -> List[dict]:
    """Build table rows from dual-edge registration results."""
    rows = []
    for gname in sorted(reg.keys()):
        r = reg[gname]
        te = r.get("temporal_edge") or {}
        sa = r.get("submap_anchor_edge") or {}
        rows.append({
            "group": gname,
            "temporal_rmsd": f"{te['kabsch_rmsd']:.4f}" if "kabsch_rmsd" in te else "--",
            "temporal_status": te.get("status", "--"),
            "temporal_corr": te.get("n_correspondences", "--"),
            "anchor_rmsd": f"{sa['kabsch_rmsd']:.4f}" if "kabsch_rmsd" in sa else "--",
            "anchor_status": sa.get("status", "--"),
            "anchor_target": sa.get("target_submap", "--"),
            "selected": r.get("selected_edge", "none"),
            "reason": r.get("selection_reason", ""),
        })
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VO-en-SLAM Pipeline Summary</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
         box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.stats {{ display: flex; gap: 15px; flex-wrap: wrap; }}
.stat {{ background: #e8f4f8; border-radius: 8px; padding: 15px 20px; min-width: 160px; text-align: center; }}
.stat .val {{ font-size: 28px; font-weight: bold; color: #0a3d62; }}
.stat .lbl {{ font-size: 12px; color: #666; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #16213e; color: white; padding: 8px 12px; text-align: left; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f0f8ff; }}
.good {{ color: #27ae60; font-weight: bold; }}
.warning {{ color: #f39c12; font-weight: bold; }}
.bad {{ color: #e74c3c; font-weight: bold; }}
.failed {{ color: #c0392b; font-weight: bold; }}
.plot {{ width: 100%; height: 500px; }}
.meta {{ color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>VO-en-SLAM Pipeline Summary</h1>
<p class="meta">Generated: {generated_at} &nbsp;|&nbsp; Output: {output_dir}</p>

<div class="stats">
  <div class="stat"><div class="val">{n_groups}</div><div class="lbl">VO Groups</div></div>
  <div class="stat"><div class="val">{n_submaps}</div><div class="lbl">SLAM Submaps</div></div>
  <div class="stat"><div class="val">{n_keyframes}</div><div class="lbl">Keyframes</div></div>
  <div class="stat"><div class="val">{n_loop_closures}</div><div class="lbl">Loop Closures</div></div>
  <div class="stat"><div class="val">{total_images}</div><div class="lbl">Total Images</div></div>
  <div class="stat"><div class="val">{elapsed_s:.1f}s</div><div class="lbl">Elapsed</div></div>
</div>

<h2>SLAM Trajectory</h2>
<div class="card">
  <div id="traj-plot" class="plot"></div>
</div>

{kf_agreement_html}

<h2>VO Groups</h2>
<div class="card">
<table>
<tr><th>Group</th><th>Frames</th><th>Range</th><th>KF Pre</th><th>KF Post</th></tr>
{vo_group_rows}
</table>
</div>

{dual_edge_html}

<h2>SLAM Submaps</h2>
<div class="card">
<table>
<tr><th>Submap</th><th>Keyframes</th><th>Stitch RMSD</th><th>Loop Closure</th></tr>
{slam_submap_rows}
</table>
</div>

<script>
var smTraj = {sm_traj_json};
var kfTraj = {kf_traj_json};

var traces = [];
if (kfTraj.length > 0) {{
    traces.push({{
        x: kfTraj.map(p => p.x), y: kfTraj.map(p => p.y), z: kfTraj.map(p => p.z),
        mode: 'markers', type: 'scatter3d', name: 'Keyframes',
        marker: {{ size: 2, color: kfTraj.map(p => p.submap_id), colorscale: 'Viridis', opacity: 0.5 }},
        text: kfTraj.map(p => 'sm' + p.submap_id + ' f' + p.frame_idx),
    }});
}}
if (smTraj.length > 0) {{
    traces.push({{
        x: smTraj.map(p => p.x), y: smTraj.map(p => p.y), z: smTraj.map(p => p.z),
        mode: 'lines+markers', type: 'scatter3d', name: 'Submap centroids',
        marker: {{ size: 6, color: '#e74c3c' }},
        line: {{ color: '#e74c3c', width: 3 }},
        text: smTraj.map(p => 'submap_' + p.submap_id + ' (' + p.num_kf + ' kf)'),
    }});
}}
Plotly.newPlot('traj-plot', traces, {{
    scene: {{ aspectmode: 'data',
              xaxis: {{ title: 'X' }}, yaxis: {{ title: 'Y' }}, zaxis: {{ title: 'Z' }} }},
    margin: {{ l: 0, r: 0, t: 30, b: 0 }},
}});
</script>
</body>
</html>"""


def _status_class(status: str) -> str:
    return status if status in ("good", "warning", "bad", "failed") else ""


def generate_pipeline_summary_html(
    output_dir: Path,
    scene_graph: dict,
    exported_groups: list,
    registration_results: Dict[str, dict],
    manifest_data: Optional[dict],
    slam_report: Optional[dict],
    kf_agreement: Optional[dict],
    elapsed_s: float,
    total_images: int,
) -> Path:
    """Generate ``pipeline_summary.html`` at the output root."""

    sg = scene_graph or {}
    summary = sg.get("summary", {})

    sm_traj = _traj_from_scene_graph(sg)
    kf_traj = _kf_traj_from_scene_graph(sg)

    # VO group rows
    groups_data = (manifest_data or {}).get("groups", [])
    vo_rows = []
    for g in groups_data:
        fi = g.get("frame_indices", [])
        frange = f"{fi[0]}–{fi[-1]}" if fi else "--"
        kf_pre = g.get("kf_window_pre", [])
        kf_post = g.get("kf_window_post", [])
        vo_rows.append(
            f"<tr><td>{g['name']}</td><td>{g.get('num_frames', 0)}</td>"
            f"<td>{frange}</td>"
            f"<td>{len(kf_pre)} {kf_pre if kf_pre else ''}</td>"
            f"<td>{len(kf_post)} {kf_post if kf_post else ''}</td></tr>"
        )
    vo_group_rows = "\n".join(vo_rows) if vo_rows else "<tr><td colspan=5>No groups</td></tr>"

    # SLAM submap rows
    slam_rows = []
    for sm in sg.get("submaps", []):
        sid = sm["submap_id"]
        n_kf = sm.get("num_keyframes", 0)
        stitch = sm.get("stitch_info", {})
        rmsd = stitch.get("kabsch_rmsd")
        rmsd_str = f"{rmsd:.4f}" if rmsd is not None else "--"
        has_lc = "Yes" if sm.get("has_loop_closure") else "No"
        slam_rows.append(
            f"<tr><td>submap_{sid:03d}</td><td>{n_kf}</td>"
            f"<td>{rmsd_str}</td><td>{has_lc}</td></tr>"
        )
    slam_submap_rows = "\n".join(slam_rows) if slam_rows else "<tr><td colspan=4>No submaps</td></tr>"

    # Dual-edge table
    if registration_results:
        de_rows = _dual_edge_rows(registration_results)
        de_html_rows = []
        for r in de_rows:
            ts = _status_class(r["temporal_status"])
            as_ = _status_class(r["anchor_status"])
            de_html_rows.append(
                f"<tr><td>{r['group']}</td>"
                f"<td class='{ts}'>{r['temporal_rmsd']}</td><td class='{ts}'>{r['temporal_status']}</td>"
                f"<td>{r['temporal_corr']}</td>"
                f"<td class='{as_}'>{r['anchor_rmsd']}</td><td class='{as_}'>{r['anchor_status']}</td>"
                f"<td>{r['anchor_target']}</td>"
                f"<td><b>{r['selected']}</b></td></tr>"
            )
        dual_edge_html = (
            "<h2>Dual-Edge Registration</h2>\n<div class='card'>\n<table>\n"
            "<tr><th>Group</th><th>Temporal RMSD</th><th>Status</th><th>Corr</th>"
            "<th>Anchor RMSD</th><th>Status</th><th>Target</th><th>Selected</th></tr>\n"
            + "\n".join(de_html_rows) + "\n</table>\n</div>"
        )
    else:
        dual_edge_html = ""

    # Keyframe agreement
    if kf_agreement and kf_agreement.get("total_timestamps", 0) > 0:
        ka = kf_agreement
        kf_agreement_html = (
            "<h2>Keyframe Agreement</h2>\n<div class='card'>\n"
            f"<p><b>{ka.get('primary_method','?')}</b> (primary) vs "
            f"<b>{ka.get('shadow_method','?')}</b> (shadow)</p>\n"
            "<div class='stats'>\n"
            f"  <div class='stat'><div class='val'>{ka.get('agreement_pct', 0):.1f}%</div>"
            f"<div class='lbl'>Agreement</div></div>\n"
            f"  <div class='stat'><div class='val'>{ka.get('total_keyframes', 0)}</div>"
            f"<div class='lbl'>Keyframes</div></div>\n"
            f"  <div class='stat'><div class='val'>{ka.get('agree_kf', 0)}</div>"
            f"<div class='lbl'>Agree KF</div></div>\n"
            f"  <div class='stat'><div class='val'>{ka.get('primary_only_kf', 0)}</div>"
            f"<div class='lbl'>{ka.get('primary_method','')} Only</div></div>\n"
            f"  <div class='stat'><div class='val'>{ka.get('shadow_only_kf', 0)}</div>"
            f"<div class='lbl'>{ka.get('shadow_method','')} Only</div></div>\n"
            "</div>\n</div>"
        )
    else:
        kf_agreement_html = ""

    n_kf = summary.get("num_keyframes", sum(
        sm.get("num_keyframes", 0) for sm in sg.get("submaps", [])))

    html = HTML_TEMPLATE.format(
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        output_dir=str(output_dir),
        n_groups=len(exported_groups),
        n_submaps=summary.get("num_submaps", len(sg.get("submaps", []))),
        n_keyframes=n_kf,
        n_loop_closures=summary.get("num_loop_closures", 0),
        total_images=total_images,
        elapsed_s=elapsed_s,
        kf_agreement_html=kf_agreement_html,
        vo_group_rows=vo_group_rows,
        dual_edge_html=dual_edge_html,
        slam_submap_rows=slam_submap_rows,
        sm_traj_json=json.dumps(sm_traj),
        kf_traj_json=json.dumps(kf_traj),
    )

    report_path = output_dir / "pipeline_summary.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path
