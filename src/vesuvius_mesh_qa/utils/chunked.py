"""Spatial partitioning utilities for large meshes."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def voxel_partition_faces(
    mesh: o3d.geometry.TriangleMesh,
    grid_size: int = 8,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Partition mesh faces into spatial grid cells by centroid.

    Args:
        mesh: The triangle mesh to partition.
        grid_size: Number of grid divisions per axis.

    Returns:
        Dict mapping (ix, iy, iz) cell indices to arrays of face indices.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    centroids = vertices[triangles].mean(axis=1)  # (n_faces, 3)

    bbox_min = centroids.min(axis=0)
    bbox_max = centroids.max(axis=0)
    extent = bbox_max - bbox_min
    extent = np.where(extent == 0, 1.0, extent)  # avoid division by zero

    # Normalize to [0, grid_size)
    normalized = (centroids - bbox_min) / extent * (grid_size - 1e-9)
    cell_indices = normalized.astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_size - 1)

    cells: dict[tuple[int, int, int], list[int]] = {}
    for face_idx in range(len(triangles)):
        key = tuple(cell_indices[face_idx])
        cells.setdefault(key, []).append(face_idx)

    return {k: np.array(v) for k, v in cells.items()}
