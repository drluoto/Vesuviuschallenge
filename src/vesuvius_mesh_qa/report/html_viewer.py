"""Generate interactive HTML review page for mesh quality inspection.

Creates a standalone HTML file with an embedded three.js 3D viewer.
Sheet switching clusters are highlighted with per-cluster cross-section
profiles and boundary proximity warnings to help human reviewers
distinguish real sheet switches from false positives.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricResult
from vesuvius_mesh_qa.metrics.normals import _build_face_adjacency_sparse
from vesuvius_mesh_qa.metrics.winding_angle import compute_winding_angles_bfs, load_umbilicus

# Maximum faces for the HTML viewer (decimate larger meshes)
MAX_VIEWER_FACES = 200_000
# Max clusters shown in sidebar (sorted by face count)
MAX_DISPLAY_CLUSTERS = 30
# Nearby faces for cross-section context
CROSS_SECTION_RADIUS = 500


def _decimate_if_needed(
    mesh: o3d.geometry.TriangleMesh, max_faces: int = MAX_VIEWER_FACES
) -> tuple[o3d.geometry.TriangleMesh, float]:
    """Decimate mesh if it exceeds max_faces. Returns (mesh, ratio)."""
    n_faces = len(mesh.triangles)
    if n_faces <= max_faces:
        return mesh, 1.0
    ratio = max_faces / n_faces
    decimated = mesh.simplify_quadric_decimation(max_faces)
    return decimated, ratio


def _encode_array(arr: np.ndarray) -> str:
    """Encode numpy array as base64 string."""
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _compute_deviation_angles(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Compute per-face normal deviation angles (degrees) from 8-ring smoothed normals."""
    triangles = np.asarray(mesh.triangles)
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals).copy()

    adj, _, _ = _build_face_adjacency_sparse(triangles)
    a2 = adj.dot(adj); a2.data[:] = 1.0
    a4 = a2.dot(a2); a4.data[:] = 1.0
    adj_k = a4.dot(a4); adj_k.data[:] = 1.0

    smoothed = adj_k.dot(normals)
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    smoothed /= np.maximum(norms, 1e-12)

    dots = np.clip(np.sum(normals * smoothed, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _find_boundary_faces(mesh: o3d.geometry.TriangleMesh) -> set[int]:
    """Find faces adjacent to mesh boundary edges."""
    triangles = np.asarray(mesh.triangles)
    n_faces = len(triangles)

    # Build edge -> face mapping
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        tri = triangles[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            edge = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            edge_faces.setdefault(edge, []).append(fi)

    # Boundary edges have only 1 face
    boundary_faces = set()
    for faces in edge_faces.values():
        if len(faces) == 1:
            boundary_faces.add(faces[0])
    return boundary_faces


def _build_vertex_colors_winding_angle(
    mesh: o3d.geometry.TriangleMesh,
    winding_angles: np.ndarray,
) -> np.ndarray:
    """Build per-vertex rainbow colors from accumulated winding angles.

    Maps winding angle range to HSV hue (0-360), producing a rainbow gradient
    around the scroll. Discontinuities = color jumps = sheet switches.
    """
    n_verts = len(mesh.vertices)
    # All finite angles are valid (0.0 is a valid angle — it's the BFS seed).
    # Only NaN/inf (unvisited vertices in disconnected components) are invalid.
    valid = np.isfinite(winding_angles)

    if not np.any(valid):
        return np.full((n_verts, 3), 128, dtype=np.uint8)

    # Normalize to [0, 1] over the observed range
    wa_min = winding_angles[valid].min()
    wa_max = winding_angles[valid].max()
    wa_range = wa_max - wa_min
    if wa_range < 1e-6:
        return np.full((n_verts, 3), 128, dtype=np.uint8)

    t = np.clip((winding_angles - wa_min) / wa_range, 0.0, 1.0)

    # HSV to RGB: hue = t * 300° (red→yellow→green→cyan→blue→magenta, skip wrap)
    hue = np.clip(t * 300.0, 0.0, 299.99)  # clamp to avoid boundary
    h_prime = hue / 60.0

    colors = np.zeros((n_verts, 3), dtype=np.float64)
    for sector in range(5):  # 0-4 covers 0-300°
        mask = (h_prime >= sector) & (h_prime < sector + 1)
        frac = h_prime[mask] - sector
        if sector == 0:    # red to yellow
            colors[mask] = np.column_stack([np.ones(mask.sum()), frac, np.zeros(mask.sum())])
        elif sector == 1:  # yellow to green
            colors[mask] = np.column_stack([1 - frac, np.ones(mask.sum()), np.zeros(mask.sum())])
        elif sector == 2:  # green to cyan
            colors[mask] = np.column_stack([np.zeros(mask.sum()), np.ones(mask.sum()), frac])
        elif sector == 3:  # cyan to blue
            colors[mask] = np.column_stack([np.zeros(mask.sum()), 1 - frac, np.ones(mask.sum())])
        elif sector == 4:  # blue to magenta
            colors[mask] = np.column_stack([frac, np.zeros(mask.sum()), np.ones(mask.sum())])

    # Invalid vertices get gray
    colors[~valid] = [0.5, 0.5, 0.5]

    return (colors * 255).astype(np.uint8)


def _build_vertex_colors_fiber(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
) -> np.ndarray | None:
    """Build per-vertex colors from fiber coherence class data.

    Colors: horizontal fibers = blue, vertical = red, unsampled = gray.
    Vertices where class flips occur get highlighted in yellow.
    Returns None if no fiber_coherence result with class data is found.
    """
    fiber_result = None
    for r in results:
        if r.name == "fiber_coherence" and "fiber_class" in r.details:
            fiber_result = r
            break
    if fiber_result is None:
        return None

    n_verts = len(mesh.vertices)
    colors = np.full((n_verts, 3), 128, dtype=np.uint8)  # gray default

    sample_indices = fiber_result.details["sample_indices"]
    fiber_class = fiber_result.details["fiber_class"]

    for i, vi in enumerate(sample_indices):
        if vi >= n_verts:
            continue
        cls = fiber_class[i]
        if cls == 1:  # horizontal = blue
            colors[vi] = [50, 100, 220]
        elif cls == 2:  # vertical = red
            colors[vi] = [220, 50, 50]

    # Highlight problem faces (class flips) in yellow
    if fiber_result.problem_faces is not None:
        triangles = np.asarray(mesh.triangles)
        for fi in fiber_result.problem_faces:
            if fi < len(triangles):
                for vi in triangles[fi]:
                    colors[vi] = [255, 220, 50]

    return colors


def _build_vertex_colors_heatmap(
    mesh: o3d.geometry.TriangleMesh,
    deviation_deg: np.ndarray,
) -> np.ndarray:
    """Build per-vertex colors as deviation angle heatmap (green->yellow->red)."""
    triangles = np.asarray(mesh.triangles)
    n_verts = len(mesh.vertices)
    n_faces = len(triangles)

    # Map deviation angle to color: 0°=green, 20°=yellow, 40°+=red
    t = np.clip(deviation_deg / 45.0, 0.0, 1.0)
    face_colors = np.zeros((n_faces, 3), dtype=np.float64)
    # Green to yellow (0->0.5): R increases, G stays
    # Yellow to red (0.5->1.0): G decreases
    face_colors[:, 0] = np.clip(t * 2.0, 0, 1) * 255  # R
    face_colors[:, 1] = np.clip(2.0 - t * 2.0, 0, 1) * 255  # G
    face_colors[:, 2] = 0  # B

    vertex_colors = np.zeros((n_verts, 3), dtype=np.float64)
    vertex_counts = np.zeros(n_verts, dtype=np.float64)
    for fi in range(n_faces):
        for vi in triangles[fi]:
            vertex_colors[vi] += face_colors[fi]
            vertex_counts[vi] += 1.0
    nonzero = vertex_counts > 0
    vertex_colors[nonzero] /= vertex_counts[nonzero, np.newaxis]
    return vertex_colors.astype(np.uint8)


def _build_vertex_colors(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
) -> np.ndarray:
    """Build per-vertex color array (Uint8, RGB) from metric results."""
    from vesuvius_mesh_qa.report.visualize import METRIC_COLORS, GOOD_COLOR

    triangles = np.asarray(mesh.triangles)
    n_verts = len(mesh.vertices)
    n_faces = len(triangles)

    face_colors = np.tile(GOOD_COLOR, (n_faces, 1)).astype(np.float64)
    face_priority = np.zeros(n_faces, dtype=np.float64)

    for r in results:
        if r.problem_faces is None or len(r.problem_faces) == 0:
            continue
        color = METRIC_COLORS.get(r.name, np.array([255, 0, 0]))
        valid = r.problem_faces[r.problem_faces < n_faces]
        if len(valid) == 0:
            continue
        mask = r.weight > face_priority[valid]
        faces_to_paint = valid[mask]
        face_colors[faces_to_paint] = color
        face_priority[faces_to_paint] = r.weight

    vertex_colors = np.zeros((n_verts, 3), dtype=np.float64)
    vertex_counts = np.zeros(n_verts, dtype=np.float64)
    for fi in range(n_faces):
        for vi in triangles[fi]:
            vertex_colors[vi] += face_colors[fi]
            vertex_counts[vi] += 1.0
    nonzero = vertex_counts > 0
    vertex_colors[nonzero] /= vertex_counts[nonzero, np.newaxis]
    return vertex_colors.astype(np.uint8)


def _extract_clusters_with_diagnostics(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    deviation_deg: np.ndarray,
    boundary_faces: set[int],
) -> list[dict]:
    """Extract sheet switching clusters with cross-section and boundary data."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    face_centroids = vertices[triangles].mean(axis=1)

    sheet_result = None
    for r in results:
        if r.name == "sheet_switching":
            sheet_result = r
            break
    if sheet_result is None or "problem_regions" not in sheet_result.details:
        return []

    problem_set = set()
    if sheet_result.problem_faces is not None:
        problem_set = set(sheet_result.problem_faces.tolist())

    clusters = []
    for i, region in enumerate(sheet_result.details["problem_regions"]):
        centroid = np.array(region["centroid"])
        fc = region["face_count"]

        # Find nearby faces for cross-section
        dists = np.linalg.norm(face_centroids - centroid, axis=1)
        n_nearby = min(CROSS_SECTION_RADIUS, len(face_centroids))
        nearby_idx = np.argsort(dists)[:n_nearby]

        nearby_centroids = face_centroids[nearby_idx]
        nearby_z = nearby_centroids[:, 2].tolist()
        nearby_x = nearby_centroids[:, 0].tolist()
        nearby_y = nearby_centroids[:, 1].tolist()
        nearby_flagged = [int(nearby_idx[j]) in problem_set for j in range(len(nearby_idx))]

        # Deviation stats for this cluster's faces
        cluster_devs = deviation_deg[nearby_idx[nearby_flagged]]
        mean_dev = float(np.mean(cluster_devs)) if len(cluster_devs) > 0 else 0.0
        max_dev = float(np.max(cluster_devs)) if len(cluster_devs) > 0 else 0.0

        # Boundary proximity: what fraction of flagged faces are boundary faces
        flagged_nearby = nearby_idx[nearby_flagged]
        n_boundary = sum(1 for f in flagged_nearby if int(f) in boundary_faces)
        boundary_frac = n_boundary / max(len(flagged_nearby), 1)

        # Z-range analysis: is there a Z-discontinuity?
        flagged_z = np.array([nearby_z[j] for j in range(len(nearby_idx)) if nearby_flagged[j]])
        good_z = np.array([nearby_z[j] for j in range(len(nearby_idx)) if not nearby_flagged[j]])

        z_jump = 0.0
        if len(flagged_z) > 0 and len(good_z) > 0:
            # Check if flagged faces have different Z from nearby good faces
            z_jump = abs(float(np.median(flagged_z) - np.median(good_z)))

        # Subsample cross-section data to keep HTML small (max 200 points)
        step = max(1, len(nearby_idx) // 200)
        cs_x = nearby_x[::step]
        cs_y = nearby_y[::step]
        cs_z = nearby_z[::step]
        cs_flagged = nearby_flagged[::step]

        is_boundary = boundary_frac > 0.3

        clusters.append({
            "id": i,
            "face_count": fc,
            "centroid": region["centroid"],
            "mean_dev": round(mean_dev, 1),
            "max_dev": round(max_dev, 1),
            "boundary_frac": round(boundary_frac, 2),
            "is_boundary": is_boundary,
            "z_jump": round(z_jump, 1),
            "cross_section": {
                "x": [round(v, 1) for v in cs_x],
                "y": [round(v, 1) for v in cs_y],
                "z": [round(v, 1) for v in cs_z],
                "flagged": cs_flagged,
            },
        })

    # Sort by face count descending, limit display
    clusters.sort(key=lambda c: c["face_count"], reverse=True)
    return clusters[:MAX_DISPLAY_CLUSTERS]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mesh QA Review: %(title)s</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0; overflow: hidden; }
#container { display: flex; height: 100vh; }
#viewer { flex: 1; position: relative; }
#sidebar { width: 380px; background: #16213e; border-left: 1px solid #333;
           overflow-y: auto; padding: 16px; }
h1 { font-size: 16px; margin-bottom: 12px; color: #fff; }
h2 { font-size: 13px; margin: 16px 0 8px; color: #aaa; text-transform: uppercase;
     letter-spacing: 1px; }
.score-box { background: #0f3460; border-radius: 8px; padding: 12px;
             margin-bottom: 12px; }
.score-label { font-size: 11px; color: #888; }
.score-value { font-size: 28px; font-weight: bold; }
.score-a { color: #4caf50; } .score-b { color: #2196f3; }
.score-c { color: #ff9800; } .score-d, .score-f { color: #f44336; }
.metric-row { display: flex; justify-content: space-between; padding: 4px 0;
              font-size: 12px; border-bottom: 1px solid #1a1a3e; }
.metric-name { color: #aaa; } .metric-score { font-weight: bold; }
.toggle-btn { background: #0f3460; border: 1px solid #335; color: #aaa;
              padding: 6px 12px; border-radius: 4px; cursor: pointer;
              font-size: 11px; margin: 4px 2px; }
.toggle-btn.active { background: #1a4a7a; color: #fff; border-color: #4488bb; }
.cluster-card { background: #0f3460; border-radius: 6px; padding: 10px;
                margin-bottom: 8px; cursor: pointer; transition: all 0.2s;
                border-left: 3px solid transparent; }
.cluster-card:hover { background: #1a4a7a; transform: translateX(4px); }
.cluster-card.active { border-left-color: #f44336; background: #1a2a5a; }
.cluster-header { display: flex; justify-content: space-between; align-items: center; }
.cluster-id { font-weight: bold; color: #f44336; font-size: 13px; }
.cluster-badge { font-size: 10px; padding: 2px 6px; border-radius: 3px;
                 font-weight: bold; }
.badge-boundary { background: #ff9800; color: #000; }
.badge-suspicious { background: #f44336; color: #fff; }
.cluster-detail { font-size: 11px; color: #888; margin-top: 4px; }
.cluster-chart { margin-top: 8px; background: #0a1a3a; border-radius: 4px;
                 padding: 4px; }
.legend { margin-top: 16px; }
.legend-item { display: flex; align-items: center; margin: 4px 0; font-size: 11px; }
.legend-swatch { width: 14px; height: 14px; border-radius: 3px; margin-right: 8px;
                 flex-shrink: 0; }
#info-overlay { position: absolute; bottom: 16px; left: 16px; font-size: 11px;
                color: #666; pointer-events: none; }
.no-clusters { color: #666; font-size: 12px; font-style: italic; padding: 8px; }
.help-text { font-size: 11px; color: #666; margin: 8px 0; line-height: 1.4; }
</style>
</head>
<body>
<div id="container">
  <div id="viewer">
    <canvas id="canvas"></canvas>
    <div id="info-overlay">
      Scroll to zoom | Drag to rotate | Right-drag to pan
    </div>
  </div>
  <div id="sidebar">
    <h1>%(title)s</h1>
    <div class="score-box">
      <div class="score-label">Aggregate Score</div>
      <div class="score-value %(grade_class)s">%(aggregate_fmt)s</div>
      <div class="score-label">Grade: %(grade)s | %(n_faces)s faces</div>
    </div>
    <h2>Metrics</h2>
    <div id="metrics">%(metrics_html)s</div>
    <h2>View Mode</h2>
    <div>%(view_buttons)s</div>
    <h2>Sheet Switching Clusters (%(n_clusters)s found)</h2>
    <div class="help-text">
      Click a cluster to zoom in. Cross-section shows Z-height profile
      (red=flagged, green=good). A Z-jump suggests a real sheet switch.
      Boundary clusters are often false positives.
    </div>
    <div id="clusters">%(clusters_html)s</div>
    <div class="legend">
      <h2>Legend</h2>
      <div class="legend-item"><div class="legend-swatch" style="background:#b4dcb4"></div>Good</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ff0000"></div>Sheet switching</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ff00ff"></div>Self-intersections</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#0080ff"></div>Noise</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ffa500"></div>Triangle quality</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ffff00"></div>Normal consistency</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#3264dc"></div>Horizontal fibers (fiber view)</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#dc3232"></div>Vertical fibers (fiber view)</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#ffdc32"></div>Fiber class flip (fiber view)</div>
    </div>
  </div>
</div>

<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/" }}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const posB64 = "%(positions_b64)s";
const idxB64 = "%(indices_b64)s";
const colMetricB64 = "%(colors_metric_b64)s";
const colHeatmapB64 = "%(colors_heatmap_b64)s";
const colWindingB64 = "%(colors_winding_b64)s";
const colFiberB64 = "%(colors_fiber_b64)s";
const clusters = %(clusters_json)s;

function b64ToFloat32(b64) {
  const bin = atob(b64); const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}
function b64ToUint32(b64) {
  const bin = atob(b64); const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Uint32Array(buf);
}
function b64ToUint8(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf;
}

const positions = b64ToFloat32(posB64);
const indices = b64ToUint32(idxB64);
const colMetricRaw = b64ToUint8(colMetricB64);
const colHeatmapRaw = b64ToUint8(colHeatmapB64);
const colWindingRaw = colWindingB64 ? b64ToUint8(colWindingB64) : null;
const colFiberRaw = colFiberB64 ? b64ToUint8(colFiberB64) : null;

// Pre-convert color sets to float
const colMetric = new Float32Array(colMetricRaw.length);
const colHeatmap = new Float32Array(colHeatmapRaw.length);
const colWinding = colWindingRaw ? new Float32Array(colWindingRaw.length) : null;
const colFiber = colFiberRaw ? new Float32Array(colFiberRaw.length) : null;
for (let i = 0; i < colMetricRaw.length; i++) {
  colMetric[i] = colMetricRaw[i] / 255.0;
  colHeatmap[i] = colHeatmapRaw[i] / 255.0;
}
if (colWindingRaw) {
  for (let i = 0; i < colWindingRaw.length; i++) colWinding[i] = colWindingRaw[i] / 255.0;
}
if (colFiberRaw) {
  for (let i = 0; i < colFiberRaw.length; i++) colFiber[i] = colFiberRaw[i] / 255.0;
}

// Setup
const canvas = document.getElementById('canvas');
const viewer = document.getElementById('viewer');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x1a1a2e);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setIndex(new THREE.BufferAttribute(indices, 1));
const colorAttr = new THREE.BufferAttribute(colMetric.slice(), 3);
geometry.setAttribute('color', colorAttr);
geometry.computeVertexNormals();

const material = new THREE.MeshPhongMaterial({
  vertexColors: true, side: THREE.DoubleSide, shininess: 20, specular: 0x222222
});
const meshObj = new THREE.Mesh(geometry, material);
scene.add(meshObj);

geometry.computeBoundingBox();
const bbox = geometry.boundingBox;
const center = new THREE.Vector3(); bbox.getCenter(center);
const bsize = new THREE.Vector3(); bbox.getSize(bsize);
const maxDim = Math.max(bsize.x, bsize.y, bsize.z);

scene.add(new THREE.AmbientLight(0x404040, 1.5));
const dl1 = new THREE.DirectionalLight(0xffffff, 1.0);
dl1.position.set(center.x + maxDim, center.y + maxDim, center.z + maxDim);
scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0xffffff, 0.5);
dl2.position.set(center.x - maxDim, center.y - maxDim, center.z + maxDim * 0.5);
scene.add(dl2);

camera.position.set(center.x + maxDim * 0.8, center.y + maxDim * 0.5, center.z + maxDim * 0.8);
camera.lookAt(center);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true; controls.dampingFactor = 0.1;
controls.update();

function onResize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize); onResize();

// Cluster markers
const markerGroup = new THREE.Group(); scene.add(markerGroup);
const markerSize = maxDim * 0.008;
clusters.forEach((c, i) => {
  const geo = new THREE.SphereGeometry(markerSize, 12, 12);
  const color = c.is_boundary ? 0xff8800 : 0xff2222;
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 });
  const s = new THREE.Mesh(geo, mat);
  s.position.set(c.centroid[0], c.centroid[1], c.centroid[2]);
  markerGroup.add(s);
});

// View mode toggle
window.setViewMode = function(mode) {
  let src = colMetric;
  if (mode === 'heatmap') src = colHeatmap;
  else if (mode === 'winding' && colWinding) src = colWinding;
  else if (mode === 'fiber' && colFiber) src = colFiber;
  const arr = colorAttr.array;
  for (let i = 0; i < arr.length; i++) arr[i] = src[i];
  colorAttr.needsUpdate = true;
  document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn-' + mode);
  if (btn) btn.classList.add('active');
};

// Draw cross-section chart on a canvas
function drawCrossSection(canvasId, data) {
  const cvs = document.getElementById(canvasId);
  if (!cvs) return;
  const ctx = cvs.getContext('2d');
  const W = cvs.width, H = cvs.height;
  ctx.fillStyle = '#0a1a3a'; ctx.fillRect(0, 0, W, H);

  const { x, y, z, flagged } = data;
  if (!z || z.length === 0) return;

  // Use the axis with more spread for X
  const xRange = Math.max(...x) - Math.min(...x);
  const yRange = Math.max(...y) - Math.min(...y);
  const horiz = xRange >= yRange ? x : y;
  const horizLabel = xRange >= yRange ? 'X' : 'Y';

  const hMin = Math.min(...horiz), hMax = Math.max(...horiz);
  const zMin = Math.min(...z), zMax = Math.max(...z);
  const hSpan = Math.max(hMax - hMin, 1);
  const zSpan = Math.max(zMax - zMin, 1);

  const pad = 30;
  const pw = W - pad * 2, ph = H - pad * 2;

  // Axes
  ctx.strokeStyle = '#335'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad); ctx.stroke();
  ctx.fillStyle = '#666'; ctx.font = '9px sans-serif';
  ctx.fillText(horizLabel, W - pad + 2, H - pad + 4);
  ctx.fillText('Z', pad - 2, pad - 4);
  ctx.fillText(zMin.toFixed(0), 2, H - pad);
  ctx.fillText(zMax.toFixed(0), 2, pad + 10);

  // Plot points: good first (small, transparent), then flagged (bigger)
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < z.length; i++) {
      if (pass === 0 && flagged[i]) continue;
      if (pass === 1 && !flagged[i]) continue;
      const px = pad + ((horiz[i] - hMin) / hSpan) * pw;
      const py = H - pad - ((z[i] - zMin) / zSpan) * ph;
      ctx.beginPath();
      if (flagged[i]) {
        ctx.fillStyle = 'rgba(255, 60, 60, 0.8)';
        ctx.arc(px, py, 3, 0, Math.PI * 2);
      } else {
        ctx.fillStyle = 'rgba(100, 200, 100, 0.3)';
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }
}

// Focus camera on cluster
function focusCluster(idx) {
  if (idx < 0 || idx >= clusters.length) return;
  const c = clusters[idx];
  const target = new THREE.Vector3(c.centroid[0], c.centroid[1], c.centroid[2]);
  const dist = maxDim * 0.15;
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  const newPos = target.clone().add(dir.multiplyScalar(dist));

  const startPos = camera.position.clone();
  const startTarget = controls.target.clone();
  const startTime = performance.now();
  function anim(time) {
    const t = Math.min((time - startTime) / 600, 1);
    const ease = t * (2 - t);
    camera.position.lerpVectors(startPos, newPos, ease);
    controls.target.lerpVectors(startTarget, target, ease);
    controls.update();
    if (t < 1) requestAnimationFrame(anim);
  }
  requestAnimationFrame(anim);

  document.querySelectorAll('.cluster-card').forEach(el => el.classList.remove('active'));
  const card = document.getElementById('cluster-' + idx);
  if (card) card.classList.add('active');

  // Draw cross-section
  if (c.cross_section) drawCrossSection('chart-' + idx, c.cross_section);
}

// Attach click handlers
document.querySelectorAll('.cluster-card').forEach(el => {
  el.addEventListener('click', () => focusCluster(parseInt(el.dataset.idx)));
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>"""


def export_html_review(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    aggregate: float,
    grade: str,
    output_path: Path,
    title: str = "Segment",
    umbilicus: str | tuple[float, float] | np.ndarray | None = None,
) -> None:
    """Export an interactive HTML review page for the mesh.

    Args:
        umbilicus: Optional umbilicus data for winding angle rainbow view.
            Can be a file path, (x, z) tuple, or numpy array.
    """
    n_faces_orig = len(mesh.triangles)

    # Compute diagnostics on the ORIGINAL mesh before decimation
    deviation_deg = _compute_deviation_angles(mesh)
    boundary_faces = _find_boundary_faces(mesh)
    clusters = _extract_clusters_with_diagnostics(mesh, results, deviation_deg, boundary_faces)

    # Compute winding angles if umbilicus is available
    winding_angles = None
    if umbilicus is not None:
        try:
            umb_func = load_umbilicus(umbilicus)
            verts = np.asarray(mesh.vertices)
            tris = np.asarray(mesh.triangles)
            winding_angles = compute_winding_angles_bfs(verts, tris, umb_func)
        except Exception:
            pass  # Gracefully degrade — no winding angle view

    # Build fiber class colors (from metric results, no volume access needed)
    colors_fiber_orig = _build_vertex_colors_fiber(mesh, results)

    # Decimate for viewer
    view_mesh, ratio = _decimate_if_needed(mesh)

    if ratio < 1.0:
        import scipy.spatial
        orig_verts = np.asarray(mesh.vertices)
        dec_verts = np.asarray(view_mesh.vertices)

        orig_colors_metric = _build_vertex_colors(mesh, results)
        orig_colors_heatmap = _build_vertex_colors_heatmap(mesh, deviation_deg)

        tree = scipy.spatial.cKDTree(orig_verts)
        _, nearest = tree.query(dec_verts)
        colors_metric = orig_colors_metric[nearest]
        colors_heatmap = orig_colors_heatmap[nearest]

        if winding_angles is not None:
            orig_colors_winding = _build_vertex_colors_winding_angle(mesh, winding_angles)
            colors_winding = orig_colors_winding[nearest]
        else:
            colors_winding = None

        colors_fiber = colors_fiber_orig[nearest] if colors_fiber_orig is not None else None
    else:
        colors_metric = _build_vertex_colors(view_mesh, results)
        colors_heatmap = _build_vertex_colors_heatmap(view_mesh, deviation_deg)
        if winding_angles is not None:
            colors_winding = _build_vertex_colors_winding_angle(view_mesh, winding_angles)
        else:
            colors_winding = None
        colors_fiber = colors_fiber_orig  # Same mesh, no remapping needed

    vertices = np.asarray(view_mesh.vertices).astype(np.float32)
    triangles = np.asarray(view_mesh.triangles).astype(np.uint32)

    positions_b64 = _encode_array(vertices.ravel())
    indices_b64 = _encode_array(triangles.ravel())
    colors_metric_b64 = _encode_array(colors_metric.astype(np.uint8).ravel())
    colors_heatmap_b64 = _encode_array(colors_heatmap.astype(np.uint8).ravel())
    colors_winding_b64 = _encode_array(colors_winding.astype(np.uint8).ravel()) if colors_winding is not None else ""
    colors_fiber_b64 = _encode_array(colors_fiber.astype(np.uint8).ravel()) if colors_fiber is not None else ""

    clusters_json = json.dumps(clusters)

    # Build metrics HTML
    metrics_html = ""
    for r in results:
        color = "#4caf50" if r.score >= 0.9 else "#2196f3" if r.score >= 0.75 else "#ff9800" if r.score >= 0.5 else "#f44336"
        metrics_html += (
            f'<div class="metric-row">'
            f'<span class="metric-name">{r.name}</span>'
            f'<span class="metric-score" style="color:{color}">{r.score:.3f}</span>'
            f'</div>'
        )

    # Build clusters HTML with cross-section canvases
    n_total_clusters = 0
    for r in results:
        if r.name == "sheet_switching":
            n_total_clusters = r.details.get("n_switch_regions", 0)

    if clusters:
        clusters_html = ""
        for i, c in enumerate(clusters):
            cx, cy, cz = c["centroid"]
            badge = ""
            if c["is_boundary"]:
                badge = '<span class="cluster-badge badge-boundary">BOUNDARY</span>'
            elif c["z_jump"] > 50:
                badge = '<span class="cluster-badge badge-suspicious">Z-JUMP</span>'

            clusters_html += (
                f'<div class="cluster-card" id="cluster-{i}" data-idx="{i}">'
                f'<div class="cluster-header">'
                f'<span class="cluster-id">Cluster {i+1}</span>{badge}</div>'
                f'<div class="cluster-detail">{c["face_count"]} faces | '
                f'dev: {c["mean_dev"]:.0f}-{c["max_dev"]:.0f}deg | '
                f'Z-jump: {c["z_jump"]:.0f}</div>'
                f'<div class="cluster-detail">boundary: {c["boundary_frac"]*100:.0f}%% | '
                f'at ({cx:.0f}, {cy:.0f}, {cz:.0f})</div>'
                f'<div class="cluster-chart">'
                f'<canvas id="chart-{i}" width="330" height="120"></canvas></div>'
                f'</div>'
            )
        if n_total_clusters > len(clusters):
            clusters_html += (
                f'<div class="help-text">Showing top {len(clusters)} of '
                f'{n_total_clusters} clusters by size</div>'
            )
    else:
        clusters_html = '<div class="no-clusters">No sheet switching detected</div>'

    grade_class = f"score-{grade.lower()}"

    # Build dynamic view mode buttons
    view_buttons = (
        '<button class="toggle-btn active" id="btn-metric" onclick="setViewMode(\'metric\')">Metric Colors</button>'
        '<button class="toggle-btn" id="btn-heatmap" onclick="setViewMode(\'heatmap\')">Deviation Heatmap</button>'
    )
    if colors_fiber is not None:
        view_buttons += '<button class="toggle-btn" id="btn-fiber" onclick="setViewMode(\'fiber\')">Fiber Classes</button>'
    if colors_winding is not None:
        view_buttons += '<button class="toggle-btn" id="btn-winding" onclick="setViewMode(\'winding\')">Winding Angle</button>'

    html = HTML_TEMPLATE % {
        "title": title,
        "aggregate_fmt": f"{aggregate:.3f}",
        "grade": grade,
        "grade_class": grade_class,
        "n_faces": f"{n_faces_orig:,}",
        "metrics_html": metrics_html,
        "n_clusters": n_total_clusters,
        "clusters_html": clusters_html,
        "positions_b64": positions_b64,
        "indices_b64": indices_b64,
        "colors_metric_b64": colors_metric_b64,
        "colors_heatmap_b64": colors_heatmap_b64,
        "colors_winding_b64": colors_winding_b64,
        "colors_fiber_b64": colors_fiber_b64,
        "clusters_json": clusters_json,
        "view_buttons": view_buttons,
    }

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
