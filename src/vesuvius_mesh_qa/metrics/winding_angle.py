"""Winding angle consistency metric for sheet switch detection.

Adapts ThaumatoAnakalyptor's winding angle concept for post-hoc QA.
On a correct single-layer mesh, the winding angle (angular position
around the scroll's center axis / umbilicus) varies smoothly.  A sheet
switch creates a discontinuity in the winding angle field.

The umbilicus is the 3D curve through the scroll center.  For each mesh
vertex, the winding angle is computed via BFS traversal that accumulates
angular differences edge-by-edge — exactly as ThaumatoAnakalyptor's
MeshSplitter.compute_uv_with_bfs() does.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Union

import numpy as np
import open3d as o3d
from scipy import interpolate

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


# ---------------------------------------------------------------------------
# Umbilicus loading
# ---------------------------------------------------------------------------

UmbilicusFunc = Callable[[float], tuple[float, float]]


def load_umbilicus(
    source: Union[str, Path, np.ndarray, tuple[float, float]],
) -> UmbilicusFunc:
    """Load umbilicus data and return an interpolation function.

    The umbilicus is the 3D curve through the scroll center.  This function
    returns a callable that takes a Y-coordinate and returns (x, z) of the
    umbilicus at that height.

    Args:
        source: One of:
            - Path to a text file with rows of "x y z" (ThaumatoAnakalyptor format)
            - (N, 3) numpy array of [x, y, z] samples along the axis
            - (x, z) tuple for scrolls with a straight center axis

    Returns:
        Callable that maps y -> (x, z).
    """
    if isinstance(source, (tuple, list)) and len(source) == 2:
        # Simple (x, z) center point — constant along Y
        cx, cz = float(source[0]), float(source[1])
        return lambda y: (cx, cz)

    if isinstance(source, np.ndarray):
        data = source
    else:
        path = Path(source)
        data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Umbilicus data must have shape (N, 3), got {data.shape}")

    # Sort by Y coordinate
    sort_idx = np.argsort(data[:, 1])
    data = data[sort_idx]

    y_vals = data[:, 1]
    x_vals = data[:, 0]
    z_vals = data[:, 2]

    if len(y_vals) < 2:
        # Single point — constant
        cx, cz = float(x_vals[0]), float(z_vals[0])
        return lambda y: (cx, cz)

    # Linear interpolation with extrapolation for out-of-range Y values
    interp_x = interpolate.interp1d(y_vals, x_vals, kind="linear", fill_value="extrapolate")
    interp_z = interpolate.interp1d(y_vals, z_vals, kind="linear", fill_value="extrapolate")

    def _lookup(y: float) -> tuple[float, float]:
        return (float(interp_x(y)), float(interp_z(y)))

    return _lookup


# ---------------------------------------------------------------------------
# BFS winding angle computation
# ---------------------------------------------------------------------------


def _build_vertex_adjacency(triangles: np.ndarray, n_vertices: int) -> list[list[int]]:
    """Build vertex adjacency list from triangle array."""
    adj: list[list[int]] = [[] for _ in range(n_vertices)]
    for tri in triangles:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        adj[v0].append(v1)
        adj[v0].append(v2)
        adj[v1].append(v0)
        adj[v1].append(v2)
        adj[v2].append(v0)
        adj[v2].append(v1)
    # Deduplicate
    for i in range(n_vertices):
        adj[i] = list(set(adj[i]))
    return adj


def _angle_between_vertices(
    v1_pos: np.ndarray,
    v2_pos: np.ndarray,
    umbilicus: UmbilicusFunc,
) -> float:
    """Compute angular difference between two vertices around the umbilicus.

    Projects both vertices to the 2D plane (relative to umbilicus at each
    vertex's Y-level) and returns the signed angle difference in degrees,
    normalized to [-180, 180].

    This matches ThaumatoAnakalyptor's MeshSplitter.angle_between_vertices().
    """
    # Get umbilicus position at each vertex's Y level
    ux1, uz1 = umbilicus(v1_pos[1])
    ux2, uz2 = umbilicus(v2_pos[1])

    # Vector from umbilicus to vertex in XZ plane
    dx1 = v1_pos[0] - ux1
    dz1 = v1_pos[2] - uz1
    dx2 = v2_pos[0] - ux2
    dz2 = v2_pos[2] - uz2

    # Angle of each vertex around umbilicus
    angle1 = np.degrees(np.arctan2(dz1, dx1))
    angle2 = np.degrees(np.arctan2(dz2, dx2))

    # Signed difference, normalized to [-180, 180]
    diff = angle2 - angle1
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0

    return diff


def compute_winding_angles_bfs(
    vertices: np.ndarray,
    triangles: np.ndarray,
    umbilicus: UmbilicusFunc,
    start_vertex: int = 0,
) -> np.ndarray:
    """Compute per-vertex winding angles via BFS traversal.

    Starting from a seed vertex (angle=0), walks the mesh graph
    accumulating angular differences edge-by-edge.  This gives
    consistent winding angles that increase monotonically around
    the scroll, unlike per-vertex atan2 which wraps at ±180°.

    Args:
        vertices: (N, 3) vertex positions.
        triangles: (M, 3) triangle indices.
        umbilicus: Callable y -> (x, z) for umbilicus position.
        start_vertex: Index of the seed vertex (angle = 0).

    Returns:
        (N,) array of winding angles in degrees.  NaN for unreachable vertices.
    """
    n_vertices = len(vertices)
    adj = _build_vertex_adjacency(triangles, n_vertices)

    angles = np.full(n_vertices, np.nan, dtype=np.float64)
    angles[start_vertex] = 0.0

    queue: deque[int] = deque([start_vertex])
    visited = np.zeros(n_vertices, dtype=bool)
    visited[start_vertex] = True

    while queue:
        current = queue.popleft()
        current_angle = angles[current]

        for neighbor in adj[current]:
            if visited[neighbor]:
                continue

            diff = _angle_between_vertices(
                vertices[current], vertices[neighbor], umbilicus
            )
            angles[neighbor] = current_angle + diff
            visited[neighbor] = True
            queue.append(neighbor)

    return angles


# ---------------------------------------------------------------------------
# Edge gradient computation and scoring
# ---------------------------------------------------------------------------


def _compute_edge_angle_gradients(
    vertices: np.ndarray,
    triangles: np.ndarray,
    winding_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute winding angle gradient across all mesh edges.

    Returns:
        edge_v1: (E,) array of vertex indices for edge start.
        edge_v2: (E,) array of vertex indices for edge end.
        edge_grad: (E,) array of |angle_diff| / edge_length in degrees per unit.
    """
    # Extract unique edges from triangles
    edges_a = np.column_stack([triangles[:, 0], triangles[:, 1]])
    edges_b = np.column_stack([triangles[:, 1], triangles[:, 2]])
    edges_c = np.column_stack([triangles[:, 2], triangles[:, 0]])
    all_edges = np.vstack([edges_a, edges_b, edges_c])

    # Canonicalize: always (min, max)
    edge_min = np.minimum(all_edges[:, 0], all_edges[:, 1])
    edge_max = np.maximum(all_edges[:, 0], all_edges[:, 1])
    edge_keys = edge_min.astype(np.int64) * (vertices.shape[0] + 1) + edge_max.astype(np.int64)
    _, unique_idx = np.unique(edge_keys, return_index=True)

    edge_v1 = edge_min[unique_idx]
    edge_v2 = edge_max[unique_idx]

    # Angle difference across each edge
    angle_diff = np.abs(winding_angles[edge_v2] - winding_angles[edge_v1])

    # Edge length
    edge_lengths = np.linalg.norm(
        vertices[edge_v2] - vertices[edge_v1], axis=1
    )
    edge_lengths = np.maximum(edge_lengths, 1e-12)

    edge_grad = angle_diff / edge_lengths

    return edge_v1, edge_v2, edge_grad


# ---------------------------------------------------------------------------
# Metric class
# ---------------------------------------------------------------------------


class WindingAngleMetric(MetricComputer):
    """Detect sheet switches via winding angle discontinuities.

    On a correct single-layer mesh, the winding angle field is smooth.
    A sheet switch creates an abrupt jump in the accumulated winding angle.
    This catches both angular and parallel-layer switches.

    Requires umbilicus data (the scroll's center axis).
    """

    name: str = "winding_angle"
    weight: float = 0.15

    # Edges with angle gradient above this (degrees per mesh unit) are flagged
    _gradient_threshold_deg_per_unit: float = 2.0
    # Alternative: absolute angle jump threshold (degrees)
    _absolute_threshold_deg: float = 15.0
    # Fraction of flagged edges that maps to score=0
    _max_flagged_fraction: float = 0.05

    def __init__(self, umbilicus: UmbilicusFunc):
        self._umbilicus = umbilicus

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        # Step 1: BFS winding angle computation
        winding_angles = compute_winding_angles_bfs(
            vertices, triangles, self._umbilicus
        )

        n_valid = int(np.sum(~np.isnan(winding_angles)))
        if n_valid == 0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "no reachable vertices"},
            )

        # Step 2: Compute edge gradients
        edge_v1, edge_v2, edge_grad = _compute_edge_angle_gradients(
            vertices, triangles, winding_angles
        )

        # Filter out edges with NaN angles
        valid_mask = ~np.isnan(winding_angles[edge_v1]) & ~np.isnan(winding_angles[edge_v2])
        edge_v1 = edge_v1[valid_mask]
        edge_v2 = edge_v2[valid_mask]
        edge_grad = edge_grad[valid_mask]

        n_edges = len(edge_v1)
        if n_edges == 0:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={"n_edges": 0, "n_discontinuous_edges": 0},
            )

        # Step 3: Flag discontinuous edges using absolute angle jump
        angle_diffs = np.abs(winding_angles[edge_v2] - winding_angles[edge_v1])
        flagged_mask = angle_diffs > self._absolute_threshold_deg

        n_flagged = int(flagged_mask.sum())
        flagged_fraction = n_flagged / n_edges

        # Step 4: Score — linear mapping: 0 flagged = 1.0, max_flagged_fraction = 0.0
        score = float(np.clip(
            1.0 - flagged_fraction / self._max_flagged_fraction, 0.0, 1.0
        ))

        # Step 5: Identify problem faces (faces containing flagged edges)
        problem_vertices = set()
        if n_flagged > 0:
            flagged_v1 = edge_v1[flagged_mask]
            flagged_v2 = edge_v2[flagged_mask]
            problem_vertices.update(flagged_v1.tolist())
            problem_vertices.update(flagged_v2.tolist())

        # Map problem vertices to faces
        problem_faces = []
        if problem_vertices:
            for fi in range(len(triangles)):
                tri = triangles[fi]
                if int(tri[0]) in problem_vertices or int(tri[1]) in problem_vertices or int(tri[2]) in problem_vertices:
                    problem_faces.append(fi)

        problem_faces_arr = (
            np.array(sorted(problem_faces), dtype=np.int64)
            if problem_faces else None
        )

        # Winding angle statistics
        valid_angles = winding_angles[~np.isnan(winding_angles)]
        angle_range = float(valid_angles.max() - valid_angles.min()) if len(valid_angles) > 0 else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_edges": n_edges,
                "n_discontinuous_edges": n_flagged,
                "flagged_edge_fraction": flagged_fraction,
                "angle_range_deg": angle_range,
                "n_reachable_vertices": n_valid,
                "absolute_threshold_deg": self._absolute_threshold_deg,
            },
            problem_faces=problem_faces_arr,
        )
