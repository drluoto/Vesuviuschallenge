"""Generate interactive HTML review page for mesh quality inspection.

Creates a standalone HTML file with an embedded three.js 3D viewer.
Sheet switching clusters are highlighted with per-cluster cross-section
profiles and boundary proximity warnings to help human reviewers
distinguish real sheet switches from false positives.
"""

from __future__ import annotations

import base64
import html as html_mod
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricResult
from vesuvius_mesh_qa.metrics.normals import _build_face_adjacency_sparse
from vesuvius_mesh_qa.metrics.winding_angle import compute_winding_angles_bfs, load_umbilicus

logger = logging.getLogger(__name__)

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
    a2 = adj.dot(adj)
    a2.data[:] = 1.0
    a4 = a2.dot(a2)
    a4.data[:] = 1.0
    adj_k = a4.dot(a4)
    adj_k.data[:] = 1.0

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
    Each unsampled vertex is colored by its nearest classified sample
    using a KD-tree, so the visualization covers the entire mesh.
    Vertices where class flips occur get highlighted in yellow.
    Returns None if no fiber_coherence result with class data is found.
    """
    from scipy.spatial import KDTree

    fiber_result = None
    for r in results:
        if r.name == "fiber_coherence" and "fiber_class" in r.details:
            fiber_result = r
            break
    if fiber_result is None:
        return None

    vertices = np.asarray(mesh.vertices)
    n_verts = len(vertices)
    colors = np.full((n_verts, 3), 128, dtype=np.uint8)  # gray default

    sample_indices = fiber_result.details["sample_indices"]
    fiber_class = np.asarray(fiber_result.details["fiber_class"])

    # Build map of classified sample positions
    classified_mask = fiber_class > 0
    classified_local = np.where(classified_mask)[0]

    if len(classified_local) >= 2:
        # Spread color: assign every vertex the class of its nearest
        # classified sample using a KD-tree
        classified_vertex_ids = np.array([
            sample_indices[i] for i in classified_local
            if sample_indices[i] < n_verts
        ])
        classified_classes = np.array([
            fiber_class[i] for i in classified_local
            if sample_indices[i] < n_verts
        ])
        classified_positions = vertices[classified_vertex_ids]
        tree = KDTree(classified_positions)
        _, nn_idx = tree.query(vertices)
        nearest_class = classified_classes[nn_idx]

        # Color by nearest class
        hz_mask = nearest_class == 1
        vt_mask = nearest_class == 2
        colors[hz_mask] = [50, 100, 220]   # horizontal = blue
        colors[vt_mask] = [220, 50, 50]    # vertical = red
    else:
        # Too few samples to spread — color only sampled vertices
        for i, vi in enumerate(sample_indices):
            if vi >= n_verts:
                continue
            cls = fiber_class[i]
            if cls == 1:
                colors[vi] = [50, 100, 220]
            elif cls == 2:
                colors[vi] = [220, 50, 50]

    # Highlight problem faces (class flips) in yellow
    if fiber_result.problem_faces is not None:
        triangles = np.asarray(mesh.triangles)
        for fi in fiber_result.problem_faces:
            if fi < len(triangles):
                for vi in triangles[fi]:
                    colors[vi] = [255, 220, 50]

    return colors


def _build_vertex_colors_ct_texture(
    mesh: o3d.geometry.TriangleMesh,
    volume_url: str,
) -> np.ndarray | None:
    """Build per-vertex colors from CT volume.

    Uses scale=1 (~15.8µm/voxel) on the original mesh (not decimated).
    Scale=0 gives the same visual result since the mesh vertex spacing
    (~80µm) is the limiting factor for visible detail, not voxel size.
    Scale=1 is 3× faster for the same output.

    Returns None if volume is unavailable or has no data at mesh positions.
    """
    from collections import defaultdict
    from vesuvius_mesh_qa.volume import VolumeAccessor

    try:
        vol = VolumeAccessor(volume_url, scale=1, cache_chunks=256)
    except Exception:
        logger.debug("Failed to open volume at %s", volume_url, exc_info=True)
        return None

    vertices = np.asarray(mesh.vertices)
    n_verts = len(vertices)
    intensities = np.full(n_verts, -1.0, dtype=np.float32)
    shape = vol.shape  # (Z, Y, X)
    chunks = vol.chunks  # (cz, cy, cx)
    scale = vol.scale_factor

    # Convert all vertices to volume indices at once
    voxels_x = np.round(vertices[:, 0] / scale).astype(np.int64)
    voxels_y = np.round(vertices[:, 1] / scale).astype(np.int64)
    voxels_z = np.round(vertices[:, 2] / scale).astype(np.int64)

    # Filter to in-bounds vertices
    in_bounds = (
        (voxels_z >= 0) & (voxels_z < shape[0]) &
        (voxels_y >= 0) & (voxels_y < shape[1]) &
        (voxels_x >= 0) & (voxels_x < shape[2])
    )

    # Group in-bounds vertices by chunk key for batch reads
    chunk_groups = defaultdict(list)
    for i in np.where(in_bounds)[0]:
        cz = int(voxels_z[i] // chunks[0])
        cy = int(voxels_y[i] // chunks[1])
        cx = int(voxels_x[i] // chunks[2])
        chunk_groups[(cz, cy, cx)].append(i)

    # Batch-read each chunk and extract all vertex values via numpy fancy indexing
    n_valid = 0
    for (cz, cy, cx), vert_indices in chunk_groups.items():
        try:
            chunk_data = vol._fetch_chunk(cz, cy, cx)
        except Exception:
            logger.debug("Failed to fetch chunk (%d,%d,%d)", cz, cy, cx, exc_info=True)
            continue
        gz0 = cz * chunks[0]
        gy0 = cy * chunks[1]
        gx0 = cx * chunks[2]

        vis = np.array(vert_indices, dtype=np.intp)
        lz = voxels_z[vis].astype(np.intp) - gz0
        ly = voxels_y[vis].astype(np.intp) - gy0
        lx = voxels_x[vis].astype(np.intp) - gx0

        # Clamp to chunk bounds
        valid = (
            (lz >= 0) & (lz < chunk_data.shape[0]) &
            (ly >= 0) & (ly < chunk_data.shape[1]) &
            (lx >= 0) & (lx < chunk_data.shape[2])
        )
        if not np.any(valid):
            continue

        vis_ok = vis[valid]
        vals = chunk_data[lz[valid], ly[valid], lx[valid]]
        intensities[vis_ok] = vals
        n_valid += int(np.sum(vals > 0))

    if n_valid < 10:
        return None

    # Normalize to 0-1 range using non-zero percentiles
    nonzero = intensities[intensities > 0]
    if len(nonzero) < 10:
        return None
    p2, p98 = np.percentile(nonzero, [2, 98])
    if p98 <= p2:
        return None
    norm = np.clip((intensities - p2) / (p98 - p2), 0.0, 1.0)

    # High-contrast grayscale for maximum texture visibility.
    # MeshBasicMaterial (no lighting) renders these values exactly.
    colors = np.zeros((n_verts, 3), dtype=np.uint8)
    # Masked/zero regions -> very dark
    zero_mask = intensities <= 0
    colors[zero_mask] = [10, 8, 6]
    # Non-zero: high-contrast warm grayscale (dark brown → cream)
    nz = ~zero_mask
    colors[nz, 0] = (30 + norm[nz] * 225).astype(np.uint8)   # R: 30-255
    colors[nz, 1] = (20 + norm[nz] * 210).astype(np.uint8)   # G: 20-230
    colors[nz, 2] = (10 + norm[nz] * 175).astype(np.uint8)   # B: 10-185

    return colors


def _scatter_face_to_vertex(
    triangles: np.ndarray, face_colors: np.ndarray, n_verts: int
) -> np.ndarray:
    """Average face colors onto vertices using sparse incidence matrix."""
    from scipy import sparse

    n_faces = len(triangles)
    rows = triangles.ravel()
    cols = np.repeat(np.arange(n_faces), 3)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_faces)
    )
    vertex_colors = np.zeros((n_verts, 3), dtype=np.float64)
    for ch in range(3):
        vertex_colors[:, ch] = incidence.dot(face_colors[:, ch])
    vertex_counts = np.array(incidence.sum(axis=1)).ravel()
    mask = vertex_counts > 0
    vertex_colors[mask] /= vertex_counts[mask, np.newaxis]
    return vertex_colors


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

    vertex_colors = _scatter_face_to_vertex(triangles, face_colors, n_verts)
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

    vertex_colors = _scatter_face_to_vertex(triangles, face_colors, n_verts)
    return vertex_colors.astype(np.uint8)


def _cluster_faces_bfs(
    face_indices: set[int],
    triangles: np.ndarray,
    min_cluster_faces: int = 5,
) -> list[list[int]]:
    """Cluster a set of face indices into connected components via BFS.

    Uses edge-based face adjacency: two faces are adjacent if they share an edge.
    Returns clusters with at least `min_cluster_faces` faces.
    """
    from collections import deque

    if not face_indices:
        return []

    # Build face adjacency for the subset
    n_faces = len(triangles)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi in face_indices:
        if fi >= n_faces:
            continue
        tri = triangles[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            edge = (min(int(tri[a]), int(tri[b])), max(int(tri[a]), int(tri[b])))
            edge_to_faces.setdefault(edge, []).append(fi)

    # Build adjacency from shared edges
    adj: dict[int, set[int]] = {fi: set() for fi in face_indices}
    for faces in edge_to_faces.values():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                adj[faces[i]].add(faces[j])
                adj[faces[j]].add(faces[i])

    # BFS clustering
    remaining = set(face_indices)
    clusters: list[list[int]] = []
    while remaining:
        seed = next(iter(remaining))
        cluster: list[int] = []
        queue: deque[int] = deque([seed])
        remaining.discard(seed)
        while queue:
            current = queue.popleft()
            cluster.append(current)
            for nb in adj.get(current, set()):
                if nb in remaining:
                    remaining.discard(nb)
                    queue.append(nb)
        if len(cluster) >= min_cluster_faces:
            clusters.append(cluster)

    return clusters


# Map metric names to short source tags for the cluster panel.
# CT is excluded: structure tensor is too noisy (~25-30° baseline) for
# reliable spatial detection. CT contributes via its global score only.
# Spatial detection uses community-validated approaches: fiber patterns
# and winding angle consistency.
_METRIC_SOURCE_TAGS: dict[str, str] = {
    "sheet_switching": "GEOM",
    "fiber_coherence": "FIBER",
    "winding_angle": "WINDING",
}


def _extract_clusters_with_diagnostics(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    deviation_deg: np.ndarray,
    boundary_faces: set[int],
) -> list[dict]:
    """Extract problem clusters from ALL metrics with cross-section and boundary data.

    Gathers problem_faces from sheet_switching (GEOM), ct_sheet_switching (CT),
    fiber_coherence (FIBER), and winding_angle (WINDING) metrics. Clusters each
    source's faces spatially, then merges overlapping clusters from different
    sources so a single region can carry multiple source tags.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    face_centroids = vertices[triangles].mean(axis=1)

    # --- Gather per-source clusters ---
    # Each entry: (cluster_faces, source_tag)
    tagged_clusters: list[tuple[list[int], set[str]]] = []

    for r in results:
        tag = _METRIC_SOURCE_TAGS.get(r.name)
        if tag is None:
            continue

        if r.name == "sheet_switching":
            # Use pre-computed problem_regions for GEOM (they have centroid info)
            if "problem_regions" not in r.details:
                continue
            problem_set = set()
            if r.problem_faces is not None:
                problem_set = set(r.problem_faces.tolist())
            for region in r.details["problem_regions"]:
                # Reconstruct face list from problem_faces near this region centroid
                centroid = np.array(region["centroid"])
                fc = region["face_count"]
                # Find the fc closest problem faces to this centroid
                if problem_set:
                    pf_arr = np.array(list(problem_set))
                    pf_dists = np.linalg.norm(face_centroids[pf_arr] - centroid, axis=1)
                    closest = pf_arr[np.argsort(pf_dists)[:fc]]
                    tagged_clusters.append((closest.tolist(), {tag}))
        else:
            # CT, FIBER, WINDING: cluster their problem_faces spatially
            if r.problem_faces is None or len(r.problem_faces) == 0:
                continue
            pf_set = set(r.problem_faces.tolist())
            spatial_clusters = _cluster_faces_bfs(pf_set, triangles, min_cluster_faces=5)
            for cluster in spatial_clusters:
                tagged_clusters.append((cluster, {tag}))

    if not tagged_clusters:
        return []

    # --- Merge overlapping clusters from different sources ---
    # Two clusters overlap if they share any face indices
    # Build face -> cluster index mapping
    face_to_cluster: dict[int, int] = {}
    merged: list[tuple[set[int], set[str]]] = []

    for faces, tags in tagged_clusters:
        face_set = set(faces)
        # Find all existing clusters this overlaps with
        overlapping_ids: set[int] = set()
        for f in face_set:
            if f in face_to_cluster:
                overlapping_ids.add(face_to_cluster[f])

        if not overlapping_ids:
            # New cluster
            idx = len(merged)
            merged.append((face_set, set(tags)))
            for f in face_set:
                face_to_cluster[f] = idx
        else:
            # Merge into the first overlapping cluster
            target_id = min(overlapping_ids)
            target_faces, target_tags = merged[target_id]
            target_faces.update(face_set)
            target_tags.update(tags)
            # Merge other overlapping clusters into target
            for oid in overlapping_ids:
                if oid != target_id:
                    of, ot = merged[oid]
                    target_faces.update(of)
                    target_tags.update(ot)
                    merged[oid] = (set(), set())  # empty placeholder
                    for f in of:
                        face_to_cluster[f] = target_id
            for f in face_set:
                face_to_cluster[f] = target_id

    # Filter out empty placeholders and small clusters
    final_clusters = [(faces, tags) for faces, tags in merged if len(faces) >= 5]

    # --- Build enriched cluster dicts ---
    all_problem_faces = set()
    for faces, _ in final_clusters:
        all_problem_faces.update(faces)

    clusters = []
    for i, (face_set, tags) in enumerate(final_clusters):
        face_list = sorted(face_set)
        face_arr = np.array(face_list)
        centroid = face_centroids[face_arr].mean(axis=0)
        fc = len(face_list)

        # Find nearby faces for cross-section
        dists = np.linalg.norm(face_centroids - centroid, axis=1)
        n_nearby = min(CROSS_SECTION_RADIUS, len(face_centroids))
        nearby_idx = np.argsort(dists)[:n_nearby]

        nearby_centroids = face_centroids[nearby_idx]
        nearby_z = nearby_centroids[:, 2].tolist()
        nearby_x = nearby_centroids[:, 0].tolist()
        nearby_y = nearby_centroids[:, 1].tolist()
        nearby_flagged = [int(nearby_idx[j]) in all_problem_faces for j in range(len(nearby_idx))]

        # Deviation stats for this cluster's faces
        cluster_devs = deviation_deg[face_arr]
        mean_dev = float(np.mean(cluster_devs)) if len(cluster_devs) > 0 else 0.0
        max_dev = float(np.max(cluster_devs)) if len(cluster_devs) > 0 else 0.0

        # Boundary proximity
        n_boundary = sum(1 for f in face_list if f in boundary_faces)
        boundary_frac = n_boundary / max(fc, 1)

        # Z-range analysis
        flagged_z = np.array([nearby_z[j] for j in range(len(nearby_idx)) if nearby_flagged[j]])
        good_z = np.array([nearby_z[j] for j in range(len(nearby_idx)) if not nearby_flagged[j]])
        z_jump = 0.0
        if len(flagged_z) > 0 and len(good_z) > 0:
            z_jump = abs(float(np.median(flagged_z) - np.median(good_z)))

        # Subsample cross-section data (max 200 points)
        step = max(1, len(nearby_idx) // 200)
        cs_x = nearby_x[::step]
        cs_y = nearby_y[::step]
        cs_z = nearby_z[::step]
        cs_flagged = nearby_flagged[::step]

        is_boundary = boundary_frac > 0.3

        clusters.append({
            "id": i,
            "face_count": fc,
            "centroid": centroid.tolist(),
            "sources": sorted(tags),
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

    # Sort by number of sources (multi-detector hits first), then face count
    clusters.sort(key=lambda c: (len(c["sources"]), c["face_count"]), reverse=True)
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
.viewmode-info { background: #0a1a3a; border: 1px solid #1a3a6a; border-radius: 4px;
                 padding: 10px 12px; margin-top: 6px; font-size: 11px; color: #8ab;
                 line-height: 1.5; display: none; max-height: 200px; overflow-y: auto; }
.viewmode-info.visible { display: block; }
.viewmode-info strong { color: #bcd; }
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
.badge-source { font-size: 9px; padding: 1px 5px; border-radius: 3px;
                font-weight: bold; margin-left: 3px; }
.badge-GEOM { background: #f44336; color: #fff; }
.badge-CT { background: #9c27b0; color: #fff; }
.badge-FIBER { background: #2196f3; color: #fff; }
.badge-WINDING { background: #4caf50; color: #fff; }
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
      Scroll to zoom | Drag to rotate | Right-drag to pan | R = reset | 1-5 = view modes
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
    <div class="viewmode-info visible" id="viewmode-info"></div>
    <h2>Problem Clusters (%(n_clusters)s found)</h2>
    <div class="help-text">
      Click a cluster to zoom in. Tags show which detectors flagged each region:
      GEOM (geometry), CT (volume), FIBER (coherence), WINDING (angle).
      Multi-tag clusters are highest confidence. Boundary clusters may be false positives.
    </div>
    <div id="clusters">%(clusters_html)s</div>
    <div class="legend">
      <h2>Legend</h2>
      <div id="legend-content">
        <div class="legend-item"><div class="legend-swatch" style="background:#b4dcb4"></div>Good</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ff0000"></div>Sheet switching</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ff00ff"></div>Self-intersections</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#0080ff"></div>Noise</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ffa500"></div>Triangle quality</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#ffff00"></div>Normal consistency</div>
      </div>
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

if (!THREE || !THREE.Scene) {
  document.getElementById('viewer').textContent = 'Failed to load Three.js from CDN. Check your internet connection.';
  throw new Error('Three.js failed to load');
}

const posB64 = "%(positions_b64)s";
const idxB64 = "%(indices_b64)s";
const colMetricB64 = "%(colors_metric_b64)s";
const colHeatmapB64 = "%(colors_heatmap_b64)s";
const colWindingB64 = "%(colors_winding_b64)s";
const colFiberB64 = "%(colors_fiber_b64)s";
const colCtB64 = "%(colors_ct_b64)s";
const ctPosB64 = "%(ct_positions_b64)s";
const ctIdxB64 = "%(ct_indices_b64)s";
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
const colCtRaw = colCtB64 ? b64ToUint8(colCtB64) : null;

// Pre-convert color sets to float
const colMetric = new Float32Array(colMetricRaw.length);
const colHeatmap = new Float32Array(colHeatmapRaw.length);
const colWinding = colWindingRaw ? new Float32Array(colWindingRaw.length) : null;
const colFiber = colFiberRaw ? new Float32Array(colFiberRaw.length) : null;
const colCt = colCtRaw ? new Float32Array(colCtRaw.length) : null;
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
if (colCtRaw) {
  for (let i = 0; i < colCtRaw.length; i++) colCt[i] = colCtRaw[i] / 255.0;
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

const matLit = new THREE.MeshPhongMaterial({
  vertexColors: true, side: THREE.DoubleSide, shininess: 20, specular: 0x222222
});
const matUnlit = new THREE.MeshBasicMaterial({
  vertexColors: true, side: THREE.DoubleSide
});
const meshObj = new THREE.Mesh(geometry, matLit);
scene.add(meshObj);

// CT texture uses separate full-resolution geometry for detail
let ctMeshObj = null;
if (ctPosB64 && colCtRaw) {
  const ctPositions = b64ToFloat32(ctPosB64);
  const ctIndices = b64ToUint32(ctIdxB64);
  const ctGeom = new THREE.BufferGeometry();
  ctGeom.setAttribute('position', new THREE.BufferAttribute(ctPositions, 3));
  ctGeom.setIndex(new THREE.BufferAttribute(ctIndices, 1));
  const ctColorArr = new Float32Array(colCtRaw.length);
  for (let i = 0; i < colCtRaw.length; i++) ctColorArr[i] = colCtRaw[i] / 255.0;
  ctGeom.setAttribute('color', new THREE.BufferAttribute(ctColorArr, 3));
  ctGeom.computeVertexNormals();
  ctMeshObj = new THREE.Mesh(ctGeom, matUnlit);
  ctMeshObj.visible = false;
  scene.add(ctMeshObj);
}

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

// Save initial camera state for reset
const initialCamPos = camera.position.clone();
const initialTarget = controls.target.clone();

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
  // Hide cluster markers in CT mode (they obscure texture)
  markerGroup.visible = (mode !== 'ct');
  if (mode === 'ct' && ctMeshObj) {
    // CT mode: show full-resolution CT mesh, hide decimated mesh
    meshObj.visible = false;
    ctMeshObj.visible = true;
  } else {
    // Other modes: show decimated mesh with appropriate colors
    meshObj.visible = true;
    if (ctMeshObj) ctMeshObj.visible = false;
    let src = colMetric;
    if (mode === 'heatmap') src = colHeatmap;
    else if (mode === 'winding' && colWinding) src = colWinding;
    else if (mode === 'fiber' && colFiber) src = colFiber;
    const arr = colorAttr.array;
    for (let i = 0; i < arr.length; i++) arr[i] = src[i];
    colorAttr.needsUpdate = true;
    meshObj.material = matLit;
  }
  document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn-' + mode);
  if (btn) btn.classList.add('active');
  // Update info box with detailed explanations for scrollprize experts
  const descs = {
    metric: 'Composite quality overlay — each face is colored by its worst-scoring metric. '
      + 'Green = passes all checks. Red = sheet switching (surface normal flips >15deg between '
      + 'adjacent faces, suggesting the mesh jumped between papyrus layers). Magenta = self-intersecting '
      + 'triangles (mesh folds through itself, common in autogen segments). Blue = geometric noise '
      + '(vertex positions deviate from local surface fit, e.g. from CT reconstruction artifacts). '
      + 'Orange = poor triangle quality (degenerate aspect ratios or angles <10deg). '
      + 'Yellow = normal inconsistency (face normals disagree with neighbors). '
      + 'When metrics overlap, higher-weighted ones take priority.',
    heatmap: 'Normal deviation angle for each face relative to its 8-ring neighborhood average. '
      + 'This reveals geometric anomalies independently of the metric thresholds. Green = well-aligned '
      + '(<5deg deviation, typical for smooth papyrus following a single layer). Yellow = moderate '
      + 'deviation (15-25deg, could indicate a fold, crinkle, or gradual layer transition). '
      + 'Red = sharp deviation (>40deg, strong indicator of a sheet switch, tear, or segmentation '
      + 'artifact). Unlike the metric view, this shows continuous angles rather than binary pass/fail, '
      + 'so subtle surface kinks that pass metric thresholds are still visible.',
    ct: 'Raw micro-CT intensity mapped onto the full-resolution mesh (not decimated). Shows the '
      + 'actual papyrus structure as captured by synchrotron scanning. Dark regions = masked or empty '
      + 'volume (outside the scanned region or masked during reconstruction). Brightness = CT density '
      + '(higher density appears brighter). Use this to verify the mesh follows real papyrus — if '
      + 'the texture shows clean fiber patterns, the segmentation is tracking a real layer. Blurry or '
      + 'discontinuous texture may indicate the surface is between layers or in a damaged region. '
      + 'Resolution depends on mesh vertex density vs. the ~8um voxel size.',
    fiber: 'Fiber orientation classification from CT volume analysis. Each vertex is classified based on '
      + 'the dominant fiber direction in its local CT neighborhood. Blue = horizontal fibers (parallel to '
      + 'papyrus sheet plane, typically the writing surface). Red = vertical fibers (perpendicular, typical '
      + 'of the backing layer in cross-laminated papyrus). Yellow = orientation flip between adjacent '
      + 'vertices — this is a strong sheet switching indicator, since real papyrus has consistent fiber '
      + 'direction within a single layer. Gray = no CT data or ambiguous orientation. '
      + 'Note: if using structure tensor (labeled "approx"), classification is noisier than nnUNet — '
      + 'expect ~5-10%% baseline flip rate even on good segments.',
    winding: 'Winding angle around the scroll umbilicus (center axis), computed via BFS traversal from '
      + 'ThaumatoAnakalyptor\'s method. Rainbow coloring encodes angular position: each color band '
      + 'represents ~60deg of rotation around the scroll center. Adjacent faces with the same color are on '
      + 'the same wrapping layer. Sharp color transitions between neighboring faces indicate the surface '
      + 'jumped to a different layer (sheet switch) — the angular gap tells you how many layers apart. '
      + 'This is the most reliable sheet switching detector for parallel-layer switches that geometry '
      + 'alone cannot catch, but requires an accurate umbilicus position. Smooth gradients = good.'
  };
  const info = document.getElementById('viewmode-info');
  if (info && descs[mode]) { info.textContent = descs[mode]; info.classList.add('visible'); }
  else if (info) { info.classList.remove('visible'); }
  // Update legend for current mode
  const legendEl = document.getElementById('legend-content');
  if (legendEl) {
    const legends = {
      metric: [['#b4dcb4','Good'],['#ff0000','Sheet switching'],['#ff00ff','Self-intersections'],['#0080ff','Noise'],['#ffa500','Triangle quality'],['#ffff00','Normal consistency']],
      heatmap: [['#00ff00','Aligned (0deg)'],['#ffff00','Moderate (20deg)'],['#ff0000','Sharp (40deg+)']],
      ct: [['#1e140a','Masked/empty'],['#ffe6b9','High density']],
      fiber: [['#3264dc','Horizontal'],['#dc3232','Vertical'],['#ffdc32','Class flip'],['#808080','No data']],
      winding: [['#ff0000','0deg'],['#ffff00','60deg'],['#00ff00','120deg'],['#00ffff','180deg'],['#0000ff','240deg'],['#ff00ff','300deg']]
    };
    const items = legends[mode] || legends.metric;
    while (legendEl.firstChild) legendEl.removeChild(legendEl.firstChild);
    items.forEach(([color, label]) => {
      const row = document.createElement('div');
      row.className = 'legend-item';
      const swatch = document.createElement('div');
      swatch.className = 'legend-swatch';
      swatch.style.background = color;
      const text = document.createTextNode(label);
      row.appendChild(swatch);
      row.appendChild(text);
      legendEl.appendChild(row);
    });
  }
  if (typeof render === 'function') render();
};
// Set initial info for default mode
window.setViewMode('metric');
document.getElementById('btn-metric').classList.add('active');

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
    render();
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

// Render cross-section charts on load
clusters.forEach((c, i) => {
  if (c.cross_section) drawCrossSection('chart-' + i, c.cross_section);
});

// Render CT cross-section slices
function drawCtSlice(canvasId, sliceB64, sliceSize, pts) {
  const cvs = document.getElementById(canvasId);
  if (!cvs) return;
  const ctx = cvs.getContext('2d');
  const raw = Uint8Array.from(atob(sliceB64), c => c.charCodeAt(0));
  const imgData = ctx.createImageData(sliceSize, sliceSize);
  for (let i = 0; i < raw.length; i++) {
    imgData.data[i*4] = raw[i];
    imgData.data[i*4+1] = raw[i];
    imgData.data[i*4+2] = raw[i];
    imgData.data[i*4+3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  if (pts && pts.x) {
    ctx.fillStyle = 'rgba(0, 255, 100, 0.5)';
    for (let i = 0; i < pts.x.length; i++) {
      const px = pts.x[i], py = pts.y[i];
      if (px >= 0 && px < sliceSize && py >= 0 && py < sliceSize) {
        ctx.fillRect(px - 0.5, py - 0.5, 1, 1);
      }
    }
  }
}
clusters.forEach((c, i) => {
  if (c.ct_slice_b64) drawCtSlice('ct-slice-' + i, c.ct_slice_b64, c.ct_slice_size || 128, c.ct_slice_pts);
});

// On-demand rendering (no animation loop — saves power)
function render() { renderer.render(scene, camera); }
controls.addEventListener('change', render);
window.addEventListener('resize', () => { onResize(); render(); });
render();

// Reset view
window.resetView = function() {
  camera.position.copy(initialCamPos);
  controls.target.copy(initialTarget);
  controls.update();
  render();
};

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toLowerCase();
  if (key === 'r') resetView();
  else if (key === '1') setViewMode('metric');
  else if (key === '2') setViewMode('heatmap');
  else if (key === '3') setViewMode('ct');
  else if (key === '4') setViewMode('fiber');
  else if (key === '5') setViewMode('winding');
});
</script>
</body>
</html>"""


def _add_ct_slices_to_clusters(
    clusters: list[dict],
    mesh: o3d.geometry.TriangleMesh,
    volume_url: str,
    slice_size: int = 128,
) -> None:
    """Add CT cross-section slice data to cluster dicts (in place).

    For each cluster, samples a Z-aligned slice from the CT volume at the
    cluster centroid, finds mesh vertices near that Z level, and adds
    the slice image and mesh intersection points to the cluster dict.
    """
    from vesuvius_mesh_qa.volume import VolumeAccessor

    try:
        vol = VolumeAccessor(volume_url, scale=1, cache_chunks=64)
    except Exception:
        logger.debug("Failed to open volume for CT slices", exc_info=True)
        return

    vertices = np.asarray(mesh.vertices)
    half = slice_size // 2

    for cluster in clusters[:10]:  # Limit to first 10 clusters
        cx, cy, cz = cluster["centroid"]
        s = vol.scale_factor
        iz = int(round(cz / s))
        iy = int(round(cy / s))
        ix = int(round(cx / s))

        # Check bounds
        shape = vol.shape
        if not (half <= iz < shape[0] - 1 and
                half <= iy < shape[1] - half and
                half <= ix < shape[2] - half):
            continue

        try:
            # Sample a Z-slice (Y-X plane at iz)
            raw = np.array(
                vol._vol[iz, iy - half:iy + half, ix - half:ix + half],
                dtype=np.float32,
            )
            if raw.shape != (slice_size, slice_size):
                continue
            if raw.max() <= 0:
                continue

            # Normalize to 0-255
            p2, p98 = np.percentile(raw[raw > 0], [2, 98]) if np.any(raw > 0) else (0, 1)
            if p98 <= p2:
                continue
            norm = np.clip((raw - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

            # Encode as base64
            slice_b64 = base64.b64encode(norm.tobytes()).decode("ascii")

            # Find mesh vertices near this Z-slice (within ±2 voxels)
            z_tol = 2.0 * s
            near_z = np.abs(vertices[:, 2] - cz) < z_tol
            near_verts = vertices[near_z]
            # Project to slice-local coordinates
            pts_x = ((near_verts[:, 0] / s - (ix - half))).tolist()
            pts_y = ((near_verts[:, 1] / s - (iy - half))).tolist()

            cluster["ct_slice_b64"] = slice_b64
            cluster["ct_slice_size"] = slice_size
            cluster["ct_slice_pts"] = {"x": pts_x, "y": pts_y}
        except Exception:
            logger.debug("Failed to sample CT slice for cluster", exc_info=True)
            continue


def export_html_review(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    aggregate: float,
    grade: str,
    output_path: Path,
    title: str = "Segment",
    umbilicus: str | tuple[float, float] | np.ndarray | None = None,
    volume_url: str | None = None,
) -> None:
    """Export an interactive HTML review page for the mesh.

    Args:
        umbilicus: Optional umbilicus data for winding angle rainbow view.
            Can be a file path, (x, z) tuple, or numpy array.
        volume_url: Optional Zarr URL for CT texture view mode.
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
            logger.debug("Winding angle computation failed", exc_info=True)

    # Enrich clusters with CT cross-section slices if volume available
    if volume_url and clusters:
        _add_ct_slices_to_clusters(clusters, mesh, volume_url)

    # Build fiber class colors (from metric results, no volume access needed)
    colors_fiber_orig = _build_vertex_colors_fiber(mesh, results)

    # CT texture is built on the decimated mesh (faster for remote volumes)
    # We defer it to after decimation below.

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

    # Build CT texture on the ORIGINAL mesh (full resolution for texture detail)
    # Cap at 500K faces to keep HTML file size manageable
    MAX_CT_FACES = 500_000
    colors_ct = None
    ct_positions_b64 = ""
    ct_indices_b64 = ""
    if volume_url:
        colors_ct = _build_vertex_colors_ct_texture(mesh, volume_url)
        if colors_ct is not None:
            ct_mesh = mesh
            ct_colors = colors_ct
            n_ct_faces = len(mesh.triangles)
            if n_ct_faces > MAX_CT_FACES:
                ct_mesh, _ = _decimate_if_needed(mesh, MAX_CT_FACES)
                import scipy.spatial
                tree = scipy.spatial.cKDTree(np.asarray(mesh.vertices))
                _, nearest = tree.query(np.asarray(ct_mesh.vertices))
                ct_colors = colors_ct[nearest]
            ct_verts = np.asarray(ct_mesh.vertices).astype(np.float32)
            ct_tris = np.asarray(ct_mesh.triangles).astype(np.uint32)
            ct_positions_b64 = _encode_array(ct_verts.ravel())
            ct_indices_b64 = _encode_array(ct_tris.ravel())
            colors_ct = ct_colors

    vertices = np.asarray(view_mesh.vertices).astype(np.float32)
    triangles = np.asarray(view_mesh.triangles).astype(np.uint32)

    positions_b64 = _encode_array(vertices.ravel())
    indices_b64 = _encode_array(triangles.ravel())
    colors_metric_b64 = _encode_array(colors_metric.astype(np.uint8).ravel())
    colors_heatmap_b64 = _encode_array(colors_heatmap.astype(np.uint8).ravel())
    colors_winding_b64 = _encode_array(colors_winding.astype(np.uint8).ravel()) if colors_winding is not None else ""
    colors_fiber_b64 = _encode_array(colors_fiber.astype(np.uint8).ravel()) if colors_fiber is not None else ""
    colors_ct_b64 = _encode_array(colors_ct.astype(np.uint8).ravel()) if colors_ct is not None else ""

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

    # Build clusters HTML with cross-section canvases and source tags
    n_total_clusters = len(clusters)

    if clusters:
        clusters_html = ""
        for i, c in enumerate(clusters):
            cx, cy, cz = c["centroid"]

            # Source tags (GEOM, CT, FIBER, WINDING)
            source_badges = ""
            for src in c.get("sources", []):
                source_badges += f'<span class="badge-source badge-{src}">{src}</span>'

            # Status badges
            status_badge = ""
            if c["is_boundary"]:
                status_badge = '<span class="cluster-badge badge-boundary">BOUNDARY</span>'
            elif c["z_jump"] > 50:
                status_badge = '<span class="cluster-badge badge-suspicious">Z-JUMP</span>'

            clusters_html += (
                f'<div class="cluster-card" id="cluster-{i}" data-idx="{i}">'
                f'<div class="cluster-header">'
                f'<span class="cluster-id">Cluster {i+1}</span>'
                f'<span>{source_badges}{status_badge}</span></div>'
                f'<div class="cluster-detail">{c["face_count"]} faces | '
                f'dev: {c["mean_dev"]:.0f}-{c["max_dev"]:.0f}deg | '
                f'Z-jump: {c["z_jump"]:.0f}</div>'
                f'<div class="cluster-detail">boundary: {c["boundary_frac"]*100:.0f}% | '
                f'at ({cx:.0f}, {cy:.0f}, {cz:.0f})</div>'
                f'<div class="cluster-chart">'
                f'<canvas id="chart-{i}" width="330" height="120"></canvas></div>'
                + (f'<div class="cluster-chart"><canvas id="ct-slice-{i}" width="128" height="128" style="image-rendering:pixelated"></canvas></div>'
                   if c.get("ct_slice_b64") else '')
                + '</div>'
            )
    else:
        clusters_html = '<div class="no-clusters">No problem clusters detected</div>'

    grade_class = f"score-{grade.lower()}"

    # Build dynamic view mode buttons
    view_buttons = (
        '<button class="toggle-btn active" id="btn-metric" onclick="setViewMode(\'metric\')">Metric Colors</button>'
        '<button class="toggle-btn" id="btn-heatmap" onclick="setViewMode(\'heatmap\')">Deviation Heatmap</button>'
    )
    if colors_ct is not None:
        view_buttons += '<button class="toggle-btn" id="btn-ct" onclick="setViewMode(\'ct\')">CT Texture</button>'
    if colors_fiber is not None:
        fiber_label = "Fiber Classes"
        fiber_result = next((r for r in results if r.name == "fiber_coherence"), None)
        if fiber_result and fiber_result.details.get("method", "").startswith("structure_tensor"):
            fiber_label = "Fiber (approx)"
        view_buttons += f'<button class="toggle-btn" id="btn-fiber" onclick="setViewMode(\'fiber\')">{fiber_label}</button>'
    if colors_winding is not None:
        view_buttons += '<button class="toggle-btn" id="btn-winding" onclick="setViewMode(\'winding\')">Winding Angle</button>'
    view_buttons += '<button class="toggle-btn" onclick="resetView()">Reset View</button>'
    view_buttons += '<button class="toggle-btn" disabled title="Coming soon">Compare</button>'

    html = HTML_TEMPLATE % {
        "title": html_mod.escape(title),
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
        "colors_ct_b64": colors_ct_b64,
        "ct_positions_b64": ct_positions_b64,
        "ct_indices_b64": ct_indices_b64,
        "clusters_json": clusters_json,
        "view_buttons": view_buttons,
    }

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")


def export_comparison_html() -> None:
    """Export a comparison view of two meshes (not yet implemented)."""
    raise NotImplementedError("Comparison mode is not yet implemented")
