"""Self-intersection detection for mesh quality assessment.

Uses spatial hash partitioning with AABB overlap testing for efficient
O(n log n) self-intersection detection instead of brute-force O(n^2).
"""

from __future__ import annotations

from itertools import product

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.utils.chunked import voxel_partition_faces


def _triangles_share_vertex(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    """Check whether two triangles (each an array of 3 vertex indices) share a vertex."""
    return len(np.intersect1d(tri_a, tri_b)) > 0


def _aabb_overlap(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> bool:
    """Check if two axis-aligned bounding boxes strictly overlap in 3D.

    Uses strict inequality to avoid false positives from coplanar triangles
    that touch at a boundary but don't actually intersect.
    """
    return bool(
        np.all(min_a < max_b) and np.all(min_b < max_a)
    )


def _check_intersections(
    vertices: np.ndarray,
    triangles: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    sample_size: int = 50_000,
) -> tuple[int, float, bool]:
    """Detect self-intersecting triangle pairs using spatial hashing and AABB tests.

    Parameters
    ----------
    vertices : np.ndarray
        (V, 3) vertex positions.
    triangles : np.ndarray
        (F, 3) face vertex indices.
    mesh : o3d.geometry.TriangleMesh
        The original mesh, passed through for voxel partitioning.
    sample_size : int
        Maximum number of faces to test when the mesh is large.

    Returns
    -------
    n_intersecting_pairs : int
        Number of detected intersecting (AABB-overlapping, non-adjacent) pairs.
    intersection_fraction : float
        Fraction of tested faces involved in at least one intersection.
    was_subsampled : bool
        Whether random subsampling was applied.
    """
    n_faces = len(triangles)
    was_subsampled = n_faces > sample_size

    if was_subsampled:
        rng = np.random.default_rng(42)
        sampled_indices = set(rng.choice(n_faces, sample_size, replace=False).tolist())
    else:
        sampled_indices = None  # use all

    # Precompute per-face AABBs for ALL faces (needed for neighbor lookups).
    tri_verts = vertices[triangles]  # (F, 3, 3)
    bbox_min = tri_verts.min(axis=1)  # (F, 3)
    bbox_max = tri_verts.max(axis=1)  # (F, 3)

    # Spatial partitioning
    cells = voxel_partition_faces(mesh, grid_size=8)

    # Precompute neighbor offsets (26-connectivity + self)
    neighbor_offsets = list(product((-1, 0, 1), repeat=3))

    intersecting_faces: set[int] = set()
    n_intersecting_pairs = 0

    # For each cell, test faces against faces in the same cell and 26 neighbors.
    visited_cell_pairs: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    for cell_coord, face_indices_a in cells.items():
        for offset in neighbor_offsets:
            neighbor_coord = (
                cell_coord[0] + offset[0],
                cell_coord[1] + offset[1],
                cell_coord[2] + offset[2],
            )

            if neighbor_coord not in cells:
                continue

            # Avoid testing the same cell pair twice (order-independent).
            pair_key = (
                min(cell_coord, neighbor_coord),
                max(cell_coord, neighbor_coord),
            )
            if pair_key in visited_cell_pairs:
                continue
            visited_cell_pairs.add(pair_key)

            face_indices_b = cells[neighbor_coord]

            for i in face_indices_a:
                # If subsampled, only test faces in our sample set.
                if sampled_indices is not None and i not in sampled_indices:
                    continue

                for j in face_indices_b:
                    if i >= j:
                        # Avoid self-comparison and duplicate pairs.
                        continue

                    # Skip adjacent triangles (sharing a vertex).
                    if _triangles_share_vertex(triangles[i], triangles[j]):
                        continue

                    # AABB overlap test as proxy for intersection.
                    if _aabb_overlap(
                        bbox_min[i], bbox_max[i],
                        bbox_min[j], bbox_max[j],
                    ):
                        n_intersecting_pairs += 1
                        intersecting_faces.add(i)
                        intersecting_faces.add(j)

    # Compute fraction of tested faces that are involved in intersections.
    n_tested = sample_size if was_subsampled else n_faces
    intersection_fraction = len(intersecting_faces) / max(n_tested, 1)

    # If subsampled, scale the pair count estimate to full mesh.
    if was_subsampled:
        scale_factor = n_faces / sample_size
        n_intersecting_pairs = int(n_intersecting_pairs * scale_factor)

    return n_intersecting_pairs, intersection_fraction, was_subsampled


class SelfIntersectionMetric(MetricComputer):
    """Detects self-intersecting triangles in a mesh.

    Uses spatial hash partitioning (8x8x8 grid) with per-triangle AABB overlap
    as a proxy for intersection. For large meshes (>200K faces), a random
    subsample of 50K faces is tested and results are scaled.

    Score mapping:
        0% intersecting faces -> 1.0
        5%+ intersecting faces -> 0.0
        Linear interpolation in between.
    """

    name: str = "self_intersections"
    weight: float = 0.15

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name,
                score=1.0,
                weight=self.weight,
                details={
                    "n_intersecting_pairs": 0,
                    "intersection_fraction": 0.0,
                    "was_subsampled": False,
                    "n_faces": 0,
                },
            )

        n_intersecting_pairs, intersection_fraction, was_subsampled = (
            _check_intersections(vertices, triangles, mesh, sample_size=50_000)
        )

        # Score: 1.0 at 0% intersections, 0.0 at 5%+ intersections.
        clamped = float(np.clip(intersection_fraction, 0.0, 0.05))
        score = 1.0 - clamped * 20.0

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_intersecting_pairs": n_intersecting_pairs,
                "intersection_fraction": intersection_fraction,
                "was_subsampled": was_subsampled,
                "n_faces": len(triangles),
            },
        )
