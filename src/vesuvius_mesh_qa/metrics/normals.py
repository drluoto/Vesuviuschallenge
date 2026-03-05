"""Normal-based mesh quality metrics.

Implements normal consistency checking and sheet-switching detection -- the
most critical metric for Vesuvius Challenge scroll segmentation quality.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import open3d as o3d
from scipy import sparse

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_face_adjacency(
    triangles: np.ndarray,
) -> dict[int, set[int]]:
    """Build a face adjacency map from a triangle array.

    Two faces are adjacent if they share an edge (i.e. two vertices).

    Args:
        triangles: (N, 3) int array of vertex indices per face.

    Returns:
        Dictionary mapping face index to the set of adjacent face indices.
    """
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_idx, tri in enumerate(triangles):
        # Each triangle has three edges; store with sorted vertex order.
        for i in range(3):
            edge = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edge_to_faces[edge].append(face_idx)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            adjacency[faces[0]].add(faces[1])
            adjacency[faces[1]].add(faces[0])
        elif len(faces) > 2:
            # Non-manifold edge — still record all pairings.
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    adjacency[faces[i]].add(faces[j])
                    adjacency[faces[j]].add(faces[i])

    return adjacency


def _compute_face_areas(
    vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Compute per-face areas (vectorized).

    Args:
        vertices: (V, 3) float array of vertex positions.
        triangles: (N, 3) int array of vertex indices per face.

    Returns:
        (N,) float array of triangle areas.
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _adjacency_to_sparse(
    adjacency: dict[int, set[int]], n_faces: int
) -> sparse.csr_matrix:
    """Convert adjacency dict to a sparse CSR binary matrix."""
    rows: list[int] = []
    cols: list[int] = []
    for face_i, neighbors in adjacency.items():
        for face_j in neighbors:
            rows.append(face_i)
            cols.append(face_j)
    # Include self-connections so each face's own normal is part of the average.
    diag = np.arange(n_faces)
    rows.extend(diag.tolist())
    cols.extend(diag.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))


# ---------------------------------------------------------------------------
# Metric 1: Normal Consistency
# ---------------------------------------------------------------------------


class NormalConsistencyMetric(MetricComputer):
    """Fraction of adjacent face pairs with small dihedral angle.

    A high score means the mesh surface is locally smooth — adjacent
    triangles have similar orientations.
    """

    name: str = "normal_consistency"
    weight: float = 0.15

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        triangles = np.asarray(mesh.triangles)
        if len(triangles) == 0:
            return MetricResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)

        adjacency = _build_face_adjacency(triangles)

        # Collect unique adjacent pairs.
        seen: set[tuple[int, int]] = set()
        angles: list[float] = []
        for face_i, neighbors in adjacency.items():
            for face_j in neighbors:
                pair = (min(face_i, face_j), max(face_i, face_j))
                if pair in seen:
                    continue
                seen.add(pair)
                dot = np.clip(np.dot(normals[face_i], normals[face_j]), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(dot)))

        if not angles:
            return MetricResult(
                name=self.name,
                score=1.0,
                weight=self.weight,
                details={"note": "no adjacent pairs found"},
            )

        angles_arr = np.array(angles)
        threshold_deg = 30.0
        fraction_consistent = float(np.mean(angles_arr < threshold_deg))
        mean_dihedral = float(np.mean(angles_arr))

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
    abruptly transitions to an adjacent layer.  These transitions manifest
    as clusters of faces whose normals deviate sharply from the local
    surface orientation.

    Algorithm
    ---------
    1. Compute per-face normals.
    2. Build sparse face adjacency matrix.
    3. Raise adjacency to the 3rd power to obtain 3-ring neighborhoods.
    4. Smooth normals by averaging over 3-ring neighborhoods.
    5. Flag faces whose normal deviates > 45 deg from the smoothed normal.
    6. Cluster flagged faces into connected components.
    7. Filter clusters with < 20 faces (noise).
    8. Score = 1 - (flagged area in valid clusters / total area), clamped [0, 1].
    """

    name: str = "sheet_switching"
    weight: float = 0.30

    # Tuneable thresholds
    _deviation_threshold_deg: float = 45.0
    _min_cluster_faces: int = 20

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        n_faces = len(triangles)

        # Step 1: per-face normals
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals).copy()  # (N, 3)

        # Step 2: sparse adjacency matrix (including self-loops)
        adjacency_dict = _build_face_adjacency(triangles)
        adj = _adjacency_to_sparse(adjacency_dict, n_faces)

        # Step 3: 3-ring neighbourhood via matrix power
        # A^3 gives reachability within 3 hops.  We binarise at each step
        # to avoid numerical blow-up and to keep the semantics of
        # "reachable within 3 edges".
        adj_k = adj.copy()
        for _ in range(2):  # already have A^1; multiply twice more
            adj_k = adj_k.dot(adj)
            adj_k.data[:] = 1.0  # binarise

        # Step 4: smoothed normals via neighbourhood averaging
        smoothed = adj_k.dot(normals)  # (N, 3) — sum of neighbour normals
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        # Avoid division by zero for isolated faces.
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
                name=self.name,
                score=1.0,
                weight=self.weight,
                details={
                    "n_switch_regions": 0,
                    "total_switch_area_fraction": 0.0,
                    "problem_regions": [],
                },
            )

        # Step 6: cluster flagged faces via BFS connected components
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
                for neighbor in adjacency_dict.get(current, set()):
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
                name=self.name,
                score=0.0,
                weight=self.weight,
                details={"error": "mesh has zero total area"},
            )

        all_problem_faces: list[int] = []
        problem_regions: list[dict[str, Any]] = []

        for cluster in valid_clusters:
            cluster_arr = np.array(cluster)
            all_problem_faces.extend(cluster)

            # Centroid of the cluster: area-weighted average of face centroids.
            face_centroids = vertices[triangles[cluster_arr]].mean(axis=1)  # (C, 3)
            cluster_areas = face_areas[cluster_arr]
            total_cluster_area = float(cluster_areas.sum())
            if total_cluster_area > 0:
                centroid = (
                    (face_centroids * cluster_areas[:, np.newaxis]).sum(axis=0)
                    / total_cluster_area
                )
            else:
                centroid = face_centroids.mean(axis=0)

            problem_regions.append(
                {
                    "face_count": len(cluster),
                    "centroid": centroid.tolist(),
                }
            )

        flagged_area = float(face_areas[np.array(all_problem_faces)].sum()) if all_problem_faces else 0.0
        switch_area_fraction = flagged_area / total_area
        score = float(np.clip(1.0 - switch_area_fraction, 0.0, 1.0))

        problem_faces_arr = (
            np.array(sorted(all_problem_faces), dtype=np.int64)
            if all_problem_faces
            else None
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
