"""Normal-based mesh quality metrics.

Implements normal consistency checking and sheet-switching detection -- the
most critical metric for Vesuvius Challenge scroll segmentation quality.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import open3d as o3d
from scipy import sparse

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


# ---------------------------------------------------------------------------
# Shared helpers (vectorized for large meshes)
# ---------------------------------------------------------------------------


def _build_face_adjacency_sparse(triangles: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build face adjacency as a sparse matrix using vectorized numpy.

    Handles both manifold edges (2 faces) and non-manifold edges (3+ faces).

    Returns:
        adj: Sparse CSR binary adjacency matrix (n_faces x n_faces) with self-loops.
        edge_face_i: Array of face indices for each unique adjacent pair.
        edge_face_j: Array of face indices for each unique adjacent pair.
    """
    n_faces = len(triangles)

    # Build all edges: 3 edges per face, each as sorted (v_min, v_max)
    face_indices = np.repeat(np.arange(n_faces), 3)
    v_a = triangles[:, [0, 1, 2]].ravel()
    v_b = triangles[:, [1, 2, 0]].ravel()
    edge_min = np.minimum(v_a, v_b)
    edge_max = np.maximum(v_a, v_b)

    # Encode edges as unique integers for grouping
    max_v = max(int(edge_max.max()), 1) + 1
    edge_keys = edge_min.astype(np.int64) * max_v + edge_max.astype(np.int64)

    # Sort by edge key to group same edges together
    sort_idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_idx]
    sorted_faces = face_indices[sort_idx]

    # Find all consecutive pairs sharing the same edge key.
    # For manifold edges (2 faces): one pair. For non-manifold (3+ faces): all pairs.
    same_as_next = sorted_keys[:-1] == sorted_keys[1:]
    pair_indices = np.nonzero(same_as_next)[0]

    face_i = sorted_faces[pair_indices]
    face_j = sorted_faces[pair_indices + 1]

    # For non-manifold edges with 3+ faces sharing an edge, we also need
    # pairs between non-consecutive entries in the same group.
    # Detect groups of 3+: where same_as_next[i] and same_as_next[i+1] are both True
    if len(same_as_next) > 1:
        triple_mask = same_as_next[:-1] & same_as_next[1:]
        triple_indices = np.nonzero(triple_mask)[0]
        if len(triple_indices) > 0:
            # For each triple, pair first with third: sorted_faces[i] <-> sorted_faces[i+2]
            extra_i = sorted_faces[triple_indices]
            extra_j = sorted_faces[triple_indices + 2]
            face_i = np.concatenate([face_i, extra_i])
            face_j = np.concatenate([face_j, extra_j])

    # Build sparse adjacency matrix (symmetric + self-loops)
    rows = np.concatenate([face_i, face_j, np.arange(n_faces)])
    cols = np.concatenate([face_j, face_i, np.arange(n_faces)])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))

    return adj, face_i, face_j


def _compute_face_areas(
    vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Compute per-face areas (vectorized)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


# ---------------------------------------------------------------------------
# Metric 1: Normal Consistency
# ---------------------------------------------------------------------------


class NormalConsistencyMetric(MetricComputer):
    """Fraction of adjacent face pairs with small dihedral angle."""

    name: str = "normal_consistency"
    weight: float = 0.15

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        triangles = np.asarray(mesh.triangles)
        if len(triangles) == 0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)

        _, face_i, face_j = _build_face_adjacency_sparse(triangles)

        if len(face_i) == 0:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={"note": "no adjacent pairs found"},
            )

        # Vectorized dihedral angle computation
        dots = np.sum(normals[face_i] * normals[face_j], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dots))

        threshold_deg = 30.0
        fraction_consistent = float(np.mean(angles_deg < threshold_deg))
        mean_dihedral = float(np.mean(angles_deg))

        return MetricResult(
            name=self.name,
            score=fraction_consistent,
            weight=self.weight,
            details={
                "mean_dihedral_angle_deg": mean_dihedral,
                "fraction_consistent": fraction_consistent,
            },
        )


# ---------------------------------------------------------------------------
# Metric 2: Sheet Switching Detection
# ---------------------------------------------------------------------------


class SheetSwitchingMetric(MetricComputer):
    """Detect regions where the mesh surface jumps between scroll layers.

    Sheet switching is the primary failure mode in automated scroll
    segmentation: a fitted surface follows one papyrus layer and then
    abruptly transitions to an adjacent layer.

    Algorithm:
    1. Compute per-face normals.
    2. Build sparse face adjacency matrix.
    3. Raise adjacency to the 8th power via repeated squaring for
       8-ring neighborhoods (wide enough to capture layer transitions).
    4. Smooth normals by averaging over 8-ring neighborhoods.
    5. Flag faces deviating > 35 deg from smoothed normal.
    6. Cluster flagged faces into connected components.
    7. Filter clusters < 20 faces.
    8. Score = 1 - (flagged area / total area).

    Note: This detects angular surface anomalies — regions where the mesh
    bends sharply relative to its neighborhood. This catches sheet switches
    that create angular transitions (e.g. where layers diverge), but cannot
    detect switches between tightly packed parallel layers where normals
    stay similar. Detecting the latter requires CT volume context.
    """

    name: str = "sheet_switching"
    weight: float = 0.30

    _deviation_threshold_deg: float = 35.0
    _min_cluster_faces: int = 20

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        n_faces = len(triangles)

        # Step 1: per-face normals
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals).copy()

        # Step 2: sparse adjacency matrix (with self-loops)
        adj, _, _ = _build_face_adjacency_sparse(triangles)

        # Step 3: 8-ring neighbourhood via repeated squaring (3 mults vs 7)
        a2 = adj.dot(adj); a2.data[:] = 1.0
        a4 = a2.dot(a2); a4.data[:] = 1.0
        adj_k = a4.dot(a4); adj_k.data[:] = 1.0

        # Step 4: smoothed normals
        smoothed = adj_k.dot(normals)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        smoothed /= norms

        # Step 5: deviation angle per face
        dots = np.sum(normals * smoothed, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        deviation_deg = np.degrees(np.arccos(dots))

        flagged_mask = deviation_deg > self._deviation_threshold_deg
        flagged_indices = set(np.nonzero(flagged_mask)[0].tolist())

        if not flagged_indices:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={
                    "n_switch_regions": 0,
                    "total_switch_area_fraction": 0.0,
                    "problem_regions": [],
                },
            )

        # Step 6: cluster flagged faces via BFS using sparse adjacency
        clusters: list[list[int]] = []
        remaining = set(flagged_indices)
        while remaining:
            seed = next(iter(remaining))
            cluster: list[int] = []
            queue: deque[int] = deque([seed])
            remaining.discard(seed)
            while queue:
                current = queue.popleft()
                cluster.append(current)
                row_start = adj.indptr[current]
                row_end = adj.indptr[current + 1]
                neighbors = adj.indices[row_start:row_end]
                for neighbor in neighbors:
                    if neighbor in remaining:
                        remaining.discard(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)

        # Step 7: filter small clusters
        valid_clusters = [c for c in clusters if len(c) >= self._min_cluster_faces]

        # Step 8: compute areas and score
        face_areas = _compute_face_areas(vertices, triangles)
        total_area = float(face_areas.sum())

        if total_area <= 0.0:
            return MetricResult(
                name=self.name, score=0.0, weight=self.weight,
                details={"error": "mesh has zero total area"},
            )

        all_problem_faces: list[int] = []
        problem_regions: list[dict[str, Any]] = []

        for cluster in valid_clusters:
            cluster_arr = np.array(cluster)
            all_problem_faces.extend(cluster)

            face_centroids = vertices[triangles[cluster_arr]].mean(axis=1)
            cluster_areas = face_areas[cluster_arr]
            total_cluster_area = float(cluster_areas.sum())
            if total_cluster_area > 0:
                centroid = (
                    (face_centroids * cluster_areas[:, np.newaxis]).sum(axis=0)
                    / total_cluster_area
                )
            else:
                centroid = face_centroids.mean(axis=0)

            problem_regions.append({
                "face_count": len(cluster),
                "centroid": centroid.tolist(),
            })

        flagged_area = float(face_areas[np.array(all_problem_faces)].sum()) if all_problem_faces else 0.0
        switch_area_fraction = flagged_area / total_area
        score = float(np.clip(1.0 - switch_area_fraction, 0.0, 1.0))

        problem_faces_arr = (
            np.array(sorted(all_problem_faces), dtype=np.int64)
            if all_problem_faces else None
        )

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_switch_regions": len(valid_clusters),
                "total_switch_area_fraction": switch_area_fraction,
                "problem_regions": problem_regions,
            },
            problem_faces=problem_faces_arr,
        )
