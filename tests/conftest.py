"""Synthetic test meshes for vesuvius-mesh-qa tests."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest


def _make_grid_mesh(rows: int, cols: int, z_func=None) -> o3d.geometry.TriangleMesh:
    """Create a triangulated grid mesh on the XY plane.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        z_func: Optional callable (x, y) -> z for vertex displacement.

    Returns:
        Open3D TriangleMesh with normals computed.
    """
    vertices = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            x, y = float(j), float(i)
            z = z_func(x, y) if z_func else 0.0
            vertices.append([x, y, z])

    triangles = []
    for i in range(rows):
        for j in range(cols):
            v0 = i * (cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (cols + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


@pytest.fixture
def perfect_plane():
    """A perfect 50x50 grid mesh — should score ~1.0 on all metrics."""
    return _make_grid_mesh(50, 50)


@pytest.fixture
def sheet_switch_mesh():
    """Two parallel planes connected by a strip that jumps between them.

    Simulates the sheet switching failure mode.
    """
    vertices = []
    triangles = []

    # Bottom plane: z=0, 50x50 grid
    rows, cols = 50, 50
    for i in range(rows + 1):
        for j in range(cols + 1):
            vertices.append([float(j), float(i), 0.0])

    n_verts_per_plane = (rows + 1) * (cols + 1)

    for i in range(rows):
        for j in range(cols):
            v0 = i * (cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (cols + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Top plane: z=5, same grid but shifted up
    offset = n_verts_per_plane
    for i in range(rows + 1):
        for j in range(cols + 1):
            vertices.append([float(j), float(i), 5.0])

    for i in range(rows):
        for j in range(cols):
            v0 = offset + i * (cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (cols + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Connecting strip: 50 quads jumping from z=0 to z=5 along y=25
    # This creates a vertical wall that represents a sheet switch
    strip_row = 25
    for j in range(cols):
        # Bottom edge vertices (z=0 plane)
        b0 = strip_row * (cols + 1) + j
        b1 = b0 + 1
        # Top edge vertices (z=5 plane)
        t0 = offset + strip_row * (cols + 1) + j
        t1 = t0 + 1
        triangles.append([b0, b1, t0])
        triangles.append([b1, t1, t0])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


@pytest.fixture
def spiked_mesh():
    """A plane with 5% of vertices displaced far from the surface."""
    mesh = _make_grid_mesh(50, 50)
    vertices = np.asarray(mesh.vertices).copy()
    n = len(vertices)
    rng = np.random.default_rng(42)
    spike_indices = rng.choice(n, size=int(0.05 * n), replace=False)
    vertices[spike_indices, 2] += rng.uniform(5, 10, size=len(spike_indices))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


@pytest.fixture
def non_manifold_mesh():
    """A mesh with T-junction non-manifold edges."""
    # Two triangles sharing an edge, plus a third triangle creating a T-junction
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [0.5, -1, 0],
        [0.5, 0.5, 0],  # T-junction vertex on the shared edge
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],  # main triangle
        [0, 3, 1],  # adjacent triangle
        [0, 4, 1],  # T-junction triangle (shares edge 0-1 creating non-manifold)
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh
