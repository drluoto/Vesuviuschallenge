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
    """A curved scroll-like surface that jumps between two layers.

    Simulates real sheet switching: a surface follows a gentle curve (like
    a section of papyrus scroll), then abruptly transitions to a parallel
    layer offset in the normal direction, then continues on that layer.
    """
    rows, cols = 80, 80
    vertices = []
    # Generate a gently curved surface (like a section of scroll)
    # with a sheet switch: at y=40, the surface jumps from one layer to
    # an adjacent layer (offset by 5 units in z)
    for i in range(rows + 1):
        for j in range(cols + 1):
            x = float(j)
            y = float(i)
            # Gentle curvature to simulate scroll surface
            z = 2.0 * np.sin(x * np.pi / cols)
            # Sheet switch: at y=40, surface abruptly jumps to adjacent layer
            # Over just 2 rows (steep ramp), z shifts by 5 units
            if i > 41:
                z += 5.0
            elif i == 41:
                z += 3.5
            elif i == 40:
                z += 1.5
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
def self_intersecting_mesh():
    """Two triangulated planes that cross through each other.

    Creates a known self-intersection: a horizontal plane at z=0 and a
    tilted plane that passes through it, guaranteeing intersections
    between nearby non-adjacent faces.
    """
    vertices = []
    triangles = []

    # Plane 1: z=0, 20x20 grid
    rows, cols = 20, 20
    for i in range(rows + 1):
        for j in range(cols + 1):
            vertices.append([float(j), float(i), 0.0])

    n1 = (rows + 1) * (cols + 1)
    for i in range(rows):
        for j in range(cols):
            v0 = i * (cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (cols + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Plane 2: tilted, passes through plane 1 at y=10
    # z = (y - 10) * 0.5, so z<0 for y<10 and z>0 for y>10
    for i in range(rows + 1):
        for j in range(cols + 1):
            x, y = float(j), float(i)
            z = (y - 10.0) * 0.5
            vertices.append([x, y, z])

    for i in range(rows):
        for j in range(cols):
            v0 = n1 + i * (cols + 1) + j
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
