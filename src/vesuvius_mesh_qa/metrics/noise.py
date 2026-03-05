"""Noise/spike detection metric for mesh quality assessment."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


class NoiseMetric(MetricComputer):
    """Detects statistical outlier vertices (noise/spikes) in a mesh."""

    name = "noise"
    weight = 0.10

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        n_total = len(vertices)

        # Build point cloud from mesh vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

        # Statistical outlier removal
        _, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=4.0
        )

        inlier_set = set(inlier_indices)
        n_outliers = n_total - len(inlier_set)
        outlier_fraction = 1.0 - len(inlier_set) / n_total if n_total > 0 else 0.0

        # Score: 0% outliers -> 1.0, 5%+ outliers -> 0.0
        # Using 5% threshold since open meshes have boundary vertices that
        # appear as statistical outliers due to fewer neighbors.
        clamped = max(0.0, min(outlier_fraction, 0.05))
        score = 1.0 - clamped * 20.0

        # Find faces containing at least one outlier vertex
        outlier_vertex_set = set(range(n_total)) - inlier_set
        if outlier_vertex_set and len(triangles) > 0:
            mask = np.isin(triangles, list(outlier_vertex_set)).any(axis=1)
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
