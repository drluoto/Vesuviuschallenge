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


def _detect_edge_length_outliers(
    vertices: np.ndarray, triangles: np.ndarray, adj: sparse.csr_matrix,
    std_threshold: float = 2.0, min_cluster_faces: int = 10,
) -> tuple[np.ndarray, int]:
    """Detect faces with abnormally long edges relative to local neighborhood.

    Sheet switches between tightly packed parallel layers create stretched
    triangles where the mesh bridges between layers. These faces have edges
    significantly longer than their neighbors.

    Uses per-face max edge length compared to the 4-ring neighborhood median.
    Faces where max_edge > median + std_threshold * MAD are flagged.

    Returns:
        (flagged_face_indices, n_regions)
    """
    n_faces = len(triangles)
    if n_faces == 0:
        return np.array([], dtype=np.int64), 0

    # Compute max edge length per face
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    e_lens = np.stack([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ], axis=1)  # (F, 3)
    max_edge = e_lens.max(axis=1)  # (F,)

    # 4-ring neighborhood via repeated squaring
    a2 = adj.dot(adj); a2.data[:] = 1.0
    adj4 = a2.dot(a2); adj4.data[:] = 1.0

    # Compute local median edge length per face over 4-ring
    # For efficiency, compute smoothed mean and MAD instead of true median
    neighbor_counts = np.array(adj4.sum(axis=1)).ravel()
    neighbor_counts = np.maximum(neighbor_counts, 1.0)
    local_mean = np.array(adj4.dot(max_edge)).ravel() / neighbor_counts

    # Local MAD approximation: mean of |x - local_mean|
    abs_dev = np.abs(max_edge - local_mean)
    local_mad = np.array(adj4.dot(abs_dev)).ravel() / neighbor_counts
    local_mad = np.maximum(local_mad, 1e-8)

    # Flag faces where max edge is far above local mean
    z_scores = (max_edge - local_mean) / local_mad
    flagged_mask = z_scores > std_threshold

    flagged_indices = set(np.nonzero(flagged_mask)[0].tolist())
    if not flagged_indices:
        return np.array([], dtype=np.int64), 0

    # Cluster flagged faces via BFS
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
            for nb in adj.indices[row_start:row_end]:
                if nb in remaining:
                    remaining.discard(nb)
                    queue.append(nb)
        clusters.append(cluster)

    valid = [c for c in clusters if len(c) >= min_cluster_faces]
    if not valid:
        return np.array([], dtype=np.int64), 0

    all_faces = []
    for c in valid:
        all_faces.extend(c)
    return np.array(sorted(all_faces), dtype=np.int64), len(valid)


class SheetSwitchingMetric(MetricComputer):
    """Detect regions where the mesh surface jumps between scroll layers.

    Sheet switching is the primary failure mode in automated scroll
    segmentation: a fitted surface follows one papyrus layer and then
    abruptly transitions to an adjacent layer.

    Uses two complementary detectors:
    1. **Normal deviation** — flags faces where the normal deviates >35 deg
       from 8-ring smoothed normals. Catches switches that create angular
       bends (e.g. when layers diverge).
    2. **Edge length outliers** — flags faces with abnormally long edges
       relative to their 4-ring neighborhood. Catches switches between
       tightly packed parallel layers where the mesh stretches to bridge
       between layers (common case, no angular signature).

    The union of both detectors' flagged faces determines the score.
    """

    name: str = "sheet_switching"
    weight: float = 0.30

    _deviation_threshold_deg: float = 35.0
    _edge_std_threshold: float = 2.0
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

        # Per-face normals
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals).copy()

        # Sparse adjacency matrix (with self-loops)
        adj, _, _ = _build_face_adjacency_sparse(triangles)

        # --- Detector 1: Normal deviation (8-ring) ---
        a2 = adj.dot(adj); a2.data[:] = 1.0
        a4 = a2.dot(a2); a4.data[:] = 1.0
        adj_k = a4.dot(a4); adj_k.data[:] = 1.0

        smoothed = adj_k.dot(normals)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        smoothed /= norms

        dots = np.sum(normals * smoothed, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        deviation_deg = np.degrees(np.arccos(dots))

        normal_flagged = set(np.nonzero(deviation_deg > self._deviation_threshold_deg)[0].tolist())

        # --- Detector 2: Edge length outliers (4-ring) ---
        edge_faces, n_edge_regions = _detect_edge_length_outliers(
            vertices, triangles, adj,
            std_threshold=self._edge_std_threshold,
            min_cluster_faces=self._min_cluster_faces,
        )
        edge_flagged = set(edge_faces.tolist()) if len(edge_faces) > 0 else set()

        # --- Union of both detectors ---
        all_flagged = normal_flagged | edge_flagged

        if not all_flagged:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={
                    "n_switch_regions": 0,
                    "n_normal_flagged": 0,
                    "n_edge_flagged": 0,
                    "total_switch_area_fraction": 0.0,
                    "problem_regions": [],
                },
            )

        # Cluster all flagged faces via BFS
        clusters: list[list[int]] = []
        remaining = set(all_flagged)
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
                for nb in adj.indices[row_start:row_end]:
                    if nb in remaining:
                        remaining.discard(nb)
                        queue.append(nb)
            clusters.append(cluster)

        valid_clusters = [c for c in clusters if len(c) >= self._min_cluster_faces]

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

            # Tag whether this region was found by normal, edge, or both
            cluster_set = set(cluster)
            has_normal = bool(cluster_set & normal_flagged)
            has_edge = bool(cluster_set & edge_flagged)
            detector = "both" if (has_normal and has_edge) else ("normal" if has_normal else "edge_length")

            problem_regions.append({
                "face_count": len(cluster),
                "centroid": centroid.tolist(),
                "detector": detector,
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
                "n_normal_flagged": len(normal_flagged),
                "n_edge_flagged": len(edge_flagged),
                "total_switch_area_fraction": switch_area_fraction,
                "problem_regions": problem_regions,
            },
            problem_faces=problem_faces_arr,
        )
