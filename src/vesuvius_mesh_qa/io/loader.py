"""Mesh loading via Open3D."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d


def load_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    """Load an OBJ (or PLY/STL) mesh and prepare it for analysis.

    Computes vertex and face normals if not present.

    Args:
        path: Path to the mesh file.

    Returns:
        Open3D TriangleMesh with normals computed.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the mesh has no triangles.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = o3d.io.read_triangle_mesh(str(path))

    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh
