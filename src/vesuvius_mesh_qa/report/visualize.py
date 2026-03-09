"""Export colored PLY meshes highlighting problem regions per metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricResult


# Per-metric colors (RGB 0-255): red-ish tones for different problem types
METRIC_COLORS = {
    "triangle_quality": np.array([255, 165, 0]),    # orange
    "topology":         np.array([128, 0, 128]),     # purple
    "normal_consistency": np.array([255, 255, 0]),   # yellow
    "sheet_switching":  np.array([255, 0, 0]),        # red
    "self_intersections": np.array([255, 0, 255]),   # magenta
    "noise":            np.array([0, 128, 255]),      # blue
    "ct_sheet_switching": np.array([0, 255, 255]),    # cyan
}

GOOD_COLOR = np.array([180, 220, 180])  # light green


def export_visualization(
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    output_path: Path,
) -> None:
    """Export a PLY mesh with per-vertex colors showing problem regions.

    Each metric's problem faces are colored with a distinct color.
    Non-problem faces are light green. When multiple metrics flag the
    same face, the metric with higher weight wins.

    Args:
        mesh: The original mesh.
        results: List of MetricResult with problem_faces populated.
        output_path: Path to write the .ply file.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_verts = len(vertices)
    n_faces = len(triangles)

    # Start with all faces as "good"
    face_colors = np.tile(GOOD_COLOR, (n_faces, 1)).astype(np.float64)
    face_priority = np.zeros(n_faces, dtype=np.float64)

    # Paint problem faces per metric (higher weight = higher priority)
    for r in results:
        if r.problem_faces is None or len(r.problem_faces) == 0:
            continue
        color = METRIC_COLORS.get(r.name, np.array([255, 0, 0]))
        mask = r.weight > face_priority[r.problem_faces]
        faces_to_paint = r.problem_faces[mask]
        face_colors[faces_to_paint] = color
        face_priority[faces_to_paint] = r.weight

    # Convert face colors to vertex colors by averaging incident face colors
    from scipy import sparse

    rows = triangles.ravel()
    cols = np.repeat(np.arange(n_faces), 3)
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_verts, n_faces)
    )
    vertex_colors = np.zeros((n_verts, 3), dtype=np.float64)
    for ch in range(3):
        vertex_colors[:, ch] = incidence.dot(face_colors[:, ch])
    vertex_counts = np.array(incidence.sum(axis=1)).ravel()
    nonzero = vertex_counts > 0
    vertex_colors[nonzero] /= vertex_counts[nonzero, np.newaxis]

    # Build output mesh
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(vertices)
    out.triangles = o3d.utility.Vector3iVector(triangles)
    out.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)

    # Copy normals if available
    if mesh.has_triangle_normals():
        out.triangle_normals = mesh.triangle_normals
    if mesh.has_vertex_normals():
        out.vertex_normals = mesh.vertex_normals

    o3d.io.write_triangle_mesh(str(output_path), out)
