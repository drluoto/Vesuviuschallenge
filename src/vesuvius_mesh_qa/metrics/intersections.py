"""Self-intersection detection for mesh quality assessment.

Uses random sampling with vectorized AABB overlap testing for efficient
self-intersection detection on large meshes.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.utils.chunked import voxel_partition_faces


def _check_intersections_vectorized(
    vertices: np.ndarray,
    triangles: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    sample_size: int = 10_000,
    neighbors_per_sample: int = 50,
) -> tuple[int, float, bool]:
    """Detect self-intersections by sampling faces and testing nearby non-adjacent faces.

    For each sampled face, find spatially nearby faces (by centroid distance),
    exclude adjacent faces, and check AABB overlap.
    """
    n_faces = len(triangles)
    was_subsampled = n_faces > sample_size

    # Compute per-face data
    tri_verts = vertices[triangles]  # (F, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (F, 3)
    bbox_min = tri_verts.min(axis=1)  # (F, 3)
    bbox_max = tri_verts.max(axis=1)  # (F, 3)

    # Build vertex-to-face adjacency for fast neighbor exclusion
    # Two faces are adjacent if they share any vertex
    vert_to_faces: dict[int, list[int]] = {}
    for fi in range(n_faces):
        for vi in triangles[fi]:
            vert_to_faces.setdefault(int(vi), []).append(fi)

    # Sample faces
    rng = np.random.default_rng(42)
    if was_subsampled:
        sample_indices = rng.choice(n_faces, sample_size, replace=False)
    else:
        sample_indices = np.arange(n_faces)

    # Use spatial partitioning to find nearby faces efficiently
    cells = voxel_partition_faces(mesh, grid_size=16)

    # Build cell lookup: face_index -> cell_coord
    face_to_cell: dict[int, tuple[int, int, int]] = {}
    for coord, face_arr in cells.items():
        for fi in face_arr:
            face_to_cell[int(fi)] = coord

    intersecting_faces: set[int] = set()
    n_intersecting_pairs = 0

    for si in sample_indices:
        si = int(si)
        cell = face_to_cell.get(si)
        if cell is None:
            continue

        # Gather faces in same cell and 26 neighbors
        candidate_faces: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nc = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if nc in cells:
                        candidate_faces.extend(cells[nc].tolist())

        if not candidate_faces:
            continue

        candidates = np.array(candidate_faces)

        # Exclude self and adjacent faces (faces sharing a vertex)
        adjacent = set()
        adjacent.add(si)
        for vi in triangles[si]:
            for fi in vert_to_faces.get(int(vi), []):
                adjacent.add(fi)

        mask = np.array([c not in adjacent for c in candidates])
        candidates = candidates[mask]

        if len(candidates) == 0:
            continue

        # Limit candidates to avoid quadratic blowup
        if len(candidates) > neighbors_per_sample:
            # Pick closest by centroid distance
            dists = np.linalg.norm(centroids[candidates] - centroids[si], axis=1)
            top_k = np.argpartition(dists, neighbors_per_sample)[:neighbors_per_sample]
            candidates = candidates[top_k]

        # Vectorized AABB overlap: strict inequality
        overlap = np.all(bbox_min[si] < bbox_max[candidates], axis=1) & \
                  np.all(bbox_min[candidates] < bbox_max[si], axis=1)

        n_overlapping = int(overlap.sum())
        if n_overlapping > 0:
            n_intersecting_pairs += n_overlapping
            intersecting_faces.add(si)
            for ci in candidates[overlap]:
                intersecting_faces.add(int(ci))

    n_tested = len(sample_indices)
    intersection_fraction = len(intersecting_faces) / max(n_tested, 1)

    if was_subsampled:
        scale = n_faces / n_tested
        n_intersecting_pairs = int(n_intersecting_pairs * scale)

    return n_intersecting_pairs, intersection_fraction, was_subsampled


class SelfIntersectionMetric(MetricComputer):
    """Detects self-intersecting triangles in a mesh.

    Uses spatial partitioning with vectorized AABB overlap testing.
    For large meshes, samples faces and tests nearby non-adjacent faces.

    Score mapping:
        0% intersecting faces -> 1.0
        5%+ intersecting faces -> 0.0
    """

    name: str = "self_intersections"
    weight: float = 0.15

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={
                    "n_intersecting_pairs": 0,
                    "intersection_fraction": 0.0,
                    "was_subsampled": False,
                    "n_faces": 0,
                },
            )

        n_intersecting_pairs, intersection_fraction, was_subsampled = (
            _check_intersections_vectorized(vertices, triangles, mesh)
        )

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
