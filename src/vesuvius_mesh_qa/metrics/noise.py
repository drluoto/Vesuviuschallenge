"""Noise/spike detection metric for mesh quality assessment.

Uses mesh-aware outlier detection: excludes boundary neighborhoods from
statistical outlier analysis to avoid false positives.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


def _boundary_neighborhood(triangles: np.ndarray, n_verts: int, hops: int = 3) -> set[int]:
    """Find vertices within `hops` of a mesh boundary edge.

    Boundary edges are shared by only 1 face. Returns the set of vertex
    indices within the specified hop count of any boundary vertex.
    """
    # Build edge counts
    edge_count: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for k in range(3):
            e = (min(int(tri[k]), int(tri[(k + 1) % 3])),
                 max(int(tri[k]), int(tri[(k + 1) % 3])))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_verts: set[int] = set()
    for (v0, v1), count in edge_count.items():
        if count == 1:
            boundary_verts.add(v0)
            boundary_verts.add(v1)

    if not boundary_verts or hops == 0:
        return boundary_verts

    # Build vertex adjacency
    vert_adj: list[set[int]] = [set() for _ in range(n_verts)]
    for tri in triangles:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            vert_adj[a].add(b)
            vert_adj[b].add(a)

    # BFS expansion
    exclude = set(boundary_verts)
    frontier = set(boundary_verts)
    for _ in range(hops):
        next_frontier: set[int] = set()
        for v in frontier:
            for nb in vert_adj[v]:
                if nb not in exclude:
                    next_frontier.add(nb)
                    exclude.add(nb)
        frontier = next_frontier
        if not frontier:
            break

    return exclude


class NoiseMetric(MetricComputer):
    """Detects statistical outlier vertices (noise/spikes) in a mesh.

    Excludes vertices within 3 hops of mesh boundaries before running
    statistical outlier detection. Boundary neighborhoods have fewer
    point cloud neighbors and would otherwise produce false positives.
    """

    name = "noise"
    weight = 0.10

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        n_total = len(vertices)

        if n_total == 0 or len(triangles) == 0:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={"n_outliers": 0, "n_total_vertices": 0, "outlier_fraction": 0.0},
            )

        # Exclude boundary neighborhoods
        exclude = _boundary_neighborhood(triangles, n_total, hops=3)
        interior_mask = np.ones(n_total, dtype=bool)
        if exclude:
            interior_mask[list(exclude)] = False
        interior_indices = np.where(interior_mask)[0]
        n_interior = len(interior_indices)

        if n_interior < 20:
            return MetricResult(
                name=self.name, score=1.0, weight=self.weight,
                details={
                    "n_outliers": 0, "n_total_vertices": n_total,
                    "outlier_fraction": 0.0, "n_boundary_excluded": len(exclude),
                },
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices[interior_indices])

        _, inlier_local = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=5.0
        )

        inlier_local_set = set(inlier_local)
        n_outliers = n_interior - len(inlier_local_set)
        outlier_fraction = n_outliers / n_total if n_total > 0 else 0.0

        # Score: 0% outliers -> 1.0, 8%+ outliers -> 0.0
        # Real papyrus surfaces have natural texture (fibers, ridges) that
        # causes ~0.5-1% false positive outliers. The 8% threshold avoids
        # penalising surface detail while still flagging genuinely noisy meshes.
        clamped = max(0.0, min(outlier_fraction, 0.08))
        score = 1.0 - clamped * 12.5

        # Map outlier indices back to global vertex indices
        outlier_global: set[int] = set()
        for i in range(n_interior):
            if i not in inlier_local_set:
                outlier_global.add(int(interior_indices[i]))

        # Find faces containing at least one outlier vertex
        if outlier_global and len(triangles) > 0:
            mask = np.isin(triangles, list(outlier_global)).any(axis=1)
            problem_faces = np.where(mask)[0]
        else:
            problem_faces = np.array([], dtype=np.int64)

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_outliers": n_outliers,
                "n_total_vertices": n_total,
                "outlier_fraction": outlier_fraction,
            },
            problem_faces=problem_faces,
        )
