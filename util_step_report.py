"""
util_step_report - Per-step HTML report with Plotly 3D trajectory visualization.

Generates a self-contained HTML report after each submap registration step.
Shows:
  - 3D trajectory of all submaps (primary alignment)
  - 3D trajectory of shadow SL(4) alignment (when enabled)
  - Side-by-side comparison of primary vs shadow
  - Per-step timing breakdown
  - Stitch metrics table (scale, rotation, translation, inliers, RMSD)

Report naming: {step_idx}_{essn_name}_{datetime}.html
"""

import datetime
import json
import os
import html as html_mod
from typing import Dict, List, Optional

import numpy as np


def _extract_translation(T: np.ndarray) -> np.ndarray:
    """Extract translation vector from a 4x4 transform."""
    return T[:3, 3].copy()


def _build_trajectory_data(graph, map_store, shadow_transforms=None):
    """Extract trajectory positions from graph and optionally shadow transforms.

    Returns:
        primary_traj: list of dicts with x, y, z, submap_id
        shadow_traj: list of dicts with x, y, z, submap_id (or empty list)
    """
    primary_traj = []
    submap_ids = sorted(map_store.submaps.keys())

    for sid in submap_ids:
        try:
            T = graph.get_submap_transform(sid)
            pos = _extract_translation(T)
            primary_traj.append({
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "submap_id": int(sid),
            })
        except Exception:
            continue

    shadow_traj = []
    if shadow_transforms:
        for sid in submap_ids:
            if sid in shadow_transforms:
                T = shadow_transforms[sid]
                pos = _extract_translation(T)
                shadow_traj.append({
                    "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                    "submap_id": int(sid),
                })

    return primary_traj, shadow_traj


def _format_records_table(records: List[dict], label: str) -> str:
    """Build an HTML table from stitch records."""
    if not records:
        return f"<p>No {label} records.</p>"

    rows = []
    for r in records:
        backend = r.get('backend', '?')
        status_class = "good"
        if abs(r.get('sim3_s', r.get('s', 1.0)) - 1.0) > 0.1:
            status_class = "warn"
        if abs(r.get('sim3_s', r.get('s', 1.0)) - 1.0) > 0.3:
            status_class = "bad"

        s_val = r.get('sim3_s', r.get('s', 1.0))
        rows.append(f"""
        <tr class="{status_class}">
            <td>{html_mod.escape(str(r.get('edge', '?')))}</td>
            <td>{r.get('submap_prev', '?')}</td>
            <td>{r.get('submap_curr', '?')}</td>
            <td>{backend}</td>
            <td>{s_val:.6f}</td>
            <td>{r.get('rot_deg', 0):.4f}</td>
            <td>{r.get('t_norm', 0):.6f}</td>
            <td>{r.get('inliers', 0)} / {r.get('total_pts', 0)}</td>
            <td>{r.get('kabsch_rmsd', float('nan')):.6f}</td>
            <td>{'LC' if r.get('is_lc') else ''}</td>
        </tr>""")

    return f"""
    <table>
      <thead>
        <tr>
          <th>Edge</th><th>Prev</th><th>Curr</th><th>Backend</th>
          <th>Scale</th><th>Rot (deg)</th><th>|t|</th>
          <th>Inliers</th><th>Kabsch RMSD</th><th>LC</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def generate_step_report(
    step_idx: int,
    essn_name: str,
    output_dir: str,
    graph,
    map_store,
    stitch_records: List[dict],
    shadow_records: Optional[List[dict]] = None,
    shadow_transforms: Optional[Dict[int, np.ndarray]] = None,
    profiler_summary: Optional[str] = None,
) -> str:
    """Generate a per-step HTML report with 3D trajectories.

    Args:
        step_idx: Submap sequence index (0-based).
        essn_name: ESSN module name (e.g. 'essn_submap').
        output_dir: Directory to write the report into.
        graph: PoseGraph instance.
        map_store: GraphMap instance.
        stitch_records: List of primary stitch records so far.
        shadow_records: List of shadow SL4 stitch records (or None).
        shadow_transforms: Dict mapping submap_id -> 4x4 shadow global transform.
        profiler_summary: Optional text summary of profiling stages.

    Returns:
        Path to the written HTML file.
    """
    if not output_dir:
        return ""
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{step_idx:04d}_{essn_name}_{ts}.html"
    filepath = os.path.join(output_dir, filename)

    primary_traj, shadow_traj = _build_trajectory_data(
        graph, map_store, shadow_transforms)

    has_shadow = bool(shadow_traj)

    # Build JSON data for Plotly
    primary_json = json.dumps(primary_traj)
    shadow_json = json.dumps(shadow_traj) if has_shadow else "[]"

    # Build comparison table if shadow exists
    comparison_html = ""
    if has_shadow and stitch_records and shadow_records:
        comp_rows = []
        n = min(len(stitch_records), len(shadow_records))
        for i in range(n):
            pr = stitch_records[i]
            sr = shadow_records[i]
            ps = pr.get('sim3_s', pr.get('s', 1.0))
            ss = sr.get('sim3_s', sr.get('s', 1.0))
            comp_rows.append(f"""
            <tr>
                <td>{html_mod.escape(str(pr.get('edge', '?')))}</td>
                <td>{ps:.6f}</td><td>{ss:.6f}</td><td>{abs(ps - ss):.6f}</td>
                <td>{pr.get('rot_deg', 0):.4f}</td><td>{sr.get('rot_deg', 0):.4f}</td>
                <td>{pr.get('t_norm', 0):.6f}</td><td>{sr.get('t_norm', 0):.6f}</td>
                <td>{pr.get('kabsch_rmsd', float('nan')):.6f}</td>
                <td>{sr.get('kabsch_rmsd', float('nan')):.6f}</td>
            </tr>""")
        comparison_html = f"""
        <h2>Primary vs Shadow Comparison</h2>
        <table>
          <thead>
            <tr>
              <th>Edge</th>
              <th>SIM3 Scale</th><th>SL4 Scale</th><th>&Delta; Scale</th>
              <th>SIM3 Rot</th><th>SL4 Rot</th>
              <th>SIM3 |t|</th><th>SL4 |t|</th>
              <th>SIM3 RMSD</th><th>SL4 RMSD</th>
            </tr>
          </thead>
          <tbody>{"".join(comp_rows)}</tbody>
        </table>"""

    # Profiler section
    profiler_html = ""
    if profiler_summary:
        profiler_html = f"""
        <h2>Timing Profile</h2>
        <pre class="profiler">{html_mod.escape(profiler_summary)}</pre>"""

    # Stitch records
    primary_table = _format_records_table(stitch_records, "primary")
    shadow_table = _format_records_table(
        shadow_records, "shadow") if shadow_records else ""

    # Compute trajectory distance stats
    def _traj_stats(traj):
        if len(traj) < 2:
            return "N/A"
        pts = np.array([[p["x"], p["y"], p["z"]] for p in traj])
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(np.sum(dists))
        return f"total_dist={total:.3f}, steps={len(traj)}, avg_step={total/max(len(dists),1):.4f}"

    primary_stats = _traj_stats(primary_traj)
    shadow_stats = _traj_stats(shadow_traj) if has_shadow else "N/A"

    # Two-panel or one-panel layout
    if has_shadow:
        plot_layout = """
        <div class="chart-row">
          <div class="chart-box"><div id="traj_primary" style="height:500px;"></div></div>
          <div class="chart-box"><div id="traj_shadow" style="height:500px;"></div></div>
        </div>
        <div class="chart-box" style="margin-top:16px;"><div id="traj_overlay" style="height:550px;"></div></div>
        """
    else:
        plot_layout = """
        <div class="chart-box"><div id="traj_primary" style="height:600px;"></div></div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Step {step_idx} &mdash; {html_mod.escape(essn_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
    --good: #238636; --warn: #d29922; --bad: #da3633;
    --sim3-color: #58a6ff; --sl4-color: #f778ba;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin: 28px 0 12px; color: var(--accent); }}
  .subtitle {{ color: var(--text2); font-size: 0.9rem; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }}
  .card .label {{ font-size: 0.8rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.3rem; font-weight: 600; margin-top: 4px; font-family: monospace; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .chart-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }}
  @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  table {{
    width: 100%; border-collapse: collapse;
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    font-size: 0.82rem; overflow: hidden; margin-bottom: 16px;
  }}
  th {{ background: #1c2128; text-align: left; padding: 10px 8px; font-weight: 600; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px; border-bottom: 1px solid var(--border); font-family: monospace; font-size: 0.8rem; }}
  tr.good td:first-child {{ border-left: 3px solid var(--good); }}
  tr.warn td:first-child {{ border-left: 3px solid var(--warn); }}
  tr.bad td:first-child {{ border-left: 3px solid var(--bad); }}
  .profiler {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; font-size: 0.8rem; overflow-x: auto; color: var(--text2);
    white-space: pre-wrap;
  }}
  .badge {{ display: inline-block; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }}
  .badge-sim3 {{ background: #1a2740; color: var(--sim3-color); }}
  .badge-sl4 {{ background: #2d1b2e; color: var(--sl4-color); }}
</style>
</head>
<body>

<h1>Step {step_idx} Report <span class="badge badge-sim3">Primary</span>
    {"<span class='badge badge-sl4'>Shadow SL4</span>" if has_shadow else ""}</h1>
<p class="subtitle">{html_mod.escape(essn_name)} &mdash; {html_mod.escape(ts)} &mdash;
    {len(primary_traj)} submaps</p>

<div class="cards">
  <div class="card"><div class="label">Step</div><div class="value">{step_idx}</div></div>
  <div class="card"><div class="label">Submaps</div><div class="value">{len(primary_traj)}</div></div>
  <div class="card"><div class="label">Stitches</div><div class="value">{len(stitch_records)}</div></div>
  <div class="card"><div class="label">Shadow</div><div class="value">{'ON' if has_shadow else 'OFF'}</div></div>
  <div class="card"><div class="label">Primary Traj</div><div class="value" style="font-size:0.85rem;">{primary_stats}</div></div>
  {"<div class='card'><div class='label'>Shadow Traj</div><div class='value' style='font-size:0.85rem;'>" + shadow_stats + "</div></div>" if has_shadow else ""}
</div>

<h2>3D Trajectories</h2>
{plot_layout}

<h2>Primary Stitch Records</h2>
{primary_table}

{"<h2>Shadow SL4 Stitch Records</h2>" + shadow_table if shadow_table else ""}

{comparison_html}

{profiler_html}

<script>
const plotBg = '#161b22';
const gridColor = '#30363d';
const textColor = '#8b949e';

const primaryData = {primary_json};
const shadowData = {shadow_json};

function makeTrajTrace(data, name, color, symbol, size) {{
    if (!data || data.length === 0) return [];
    const xs = data.map(p => p.x);
    const ys = data.map(p => p.y);
    const zs = data.map(p => p.z);
    const labels = data.map(p => 's' + p.submap_id);
    const hovers = data.map(p =>
        '<b>Submap ' + p.submap_id + '</b><br>' +
        'x=' + p.x.toFixed(4) + ' y=' + p.y.toFixed(4) + ' z=' + p.z.toFixed(4)
    );
    return [
        {{ x: xs, y: ys, z: zs, type: 'scatter3d', mode: 'lines',
           line: {{ color: color, width: 4 }}, name: name + ' path',
           showlegend: false, hoverinfo: 'skip' }},
        {{ x: xs, y: ys, z: zs, type: 'scatter3d', mode: 'markers+text',
           marker: {{ color: color, size: size, symbol: symbol, line: {{ width: 1, color: 'white' }} }},
           text: labels, textposition: 'top center',
           textfont: {{ size: 9, color: color }},
           name: name, hovertext: hovers, hoverinfo: 'text' }},
    ];
}}

const scene3d = {{
    xaxis: {{ title: 'X', backgroundcolor: plotBg, gridcolor: gridColor, color: textColor }},
    yaxis: {{ title: 'Y', backgroundcolor: plotBg, gridcolor: gridColor, color: textColor }},
    zaxis: {{ title: 'Z', backgroundcolor: plotBg, gridcolor: gridColor, color: textColor }},
    aspectmode: 'data',
}};
const layoutBase = {{
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    font: {{ color: textColor, size: 11 }},
    margin: {{ l: 0, r: 0, t: 40, b: 0 }},
    scene: scene3d,
}};

// Primary trajectory
Plotly.newPlot('traj_primary',
    makeTrajTrace(primaryData, 'Primary (SIM3)', '#58a6ff', 'circle', 5),
    {{ ...layoutBase, title: 'Primary Trajectory' }},
    {{ responsive: true }});

// Shadow trajectory (if present)
if (shadowData.length > 0 && document.getElementById('traj_shadow')) {{
    Plotly.newPlot('traj_shadow',
        makeTrajTrace(shadowData, 'Shadow (SL4)', '#f778ba', 'diamond', 5),
        {{ ...layoutBase, title: 'Shadow SL4 Trajectory' }},
        {{ responsive: true }});

    // Overlay
    const overlayTraces = [
        ...makeTrajTrace(primaryData, 'Primary (SIM3)', '#58a6ff', 'circle', 5),
        ...makeTrajTrace(shadowData, 'Shadow (SL4)', '#f778ba', 'diamond', 5),
    ];
    Plotly.newPlot('traj_overlay', overlayTraces,
        {{ ...layoutBase, title: 'Primary vs Shadow Overlay', showlegend: true,
           legend: {{ x: 0, y: 1, font: {{ size: 11 }} }} }},
        {{ responsive: true }});
}}
</script>

</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath
