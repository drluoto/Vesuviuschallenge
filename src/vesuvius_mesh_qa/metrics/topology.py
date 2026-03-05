"""Topology metrics for mesh quality assessment."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


class TopologyMetric(MetricComputer):
    """Evaluates topological properties of a triangle mesh.

    Sub-metrics:
        - Manifoldness (edge and vertex)
        - Connected components
        - Boundary edges
    """

    name = "topology"
    weight = 0.10

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        """Compute topology metrics for the given mesh.

        Parameters
        ----------
        mesh : o3d.geometry.TriangleMesh
            Input triangle mesh to evaluate.

        Returns
        -------
        MetricResult
            Aggregated topology score with detailed sub-metric breakdown.
        """
        # --- Manifoldness ---
        is_edge_manifold = mesh.is_edge_manifold()
        edge_manifold_score = 1.0 if is_edge_manifold else 0.0

        vertex_manifold_result = mesh.is_vertex_manifold()
        # Open3D may return a bool (single value) or a per-vertex vector
        if isinstance(vertex_manifold_result, bool):
            vertex_manifold_fraction = 1.0 if vertex_manifold_result else 0.0
        else:
            vertex_manifold_mask = np.asarray(vertex_manifold_result)
            n_vertices = len(vertex_manifold_mask)
            if n_vertices > 0:
                vertex_manifold_fraction = float(vertex_manifold_mask.sum()) / n_vertices
            else:
                vertex_manifold_fraction = 1.0

        manifold_score = 0.5 * edge_manifold_score + 0.5 * vertex_manifold_fraction

        # --- Connected components ---
        cluster_indices, num_per_cluster, area_per_cluster = (
            mesh.cluster_connected_triangles()
        )
        n_components = len(num_per_cluster)
        component_score = 1.0 / n_components if n_components > 0 else 0.0

        # --- Boundary edges ---
        boundary_edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        n_boundary_edges = len(boundary_edges)

        triangles = np.asarray(mesh.triangles)
        n_faces = len(triangles)
        total_edges = max(int(n_faces * 3 / 2), 1)

        boundary_score = float(np.clip(1.0 - n_boundary_edges / total_edges, 0.0, 1.0))

        # --- Final score ---
        score = 0.4 * manifold_score + 0.3 * component_score + 0.3 * boundary_score

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "manifold_score": manifold_score,
                "component_score": component_score,
                "boundary_score": boundary_score,
                "is_edge_manifold": is_edge_manifold,
                "vertex_manifold_fraction": vertex_manifold_fraction,
                "n_components": n_components,
                "n_boundary_edges": n_boundary_edges,
            },
        )
