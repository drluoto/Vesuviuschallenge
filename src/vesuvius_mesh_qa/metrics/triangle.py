"""Triangle quality metrics for mesh quality assessment."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult


class TriangleQualityMetric(MetricComputer):
    """Evaluates triangle shape quality: aspect ratio, minimum angle, and area uniformity."""

    name = "triangle_quality"
    weight = 0.10

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)  # (V, 3)
        triangles = np.asarray(mesh.triangles)  # (F, 3)

        if len(triangles) == 0:
            return MetricResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                details={"error": "mesh has no triangles"},
            )

        # Gather triangle vertex positions: (F, 3, 3)
        v0 = vertices[triangles[:, 0]]  # (F, 3)
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        # Edge vectors
        e0 = v1 - v0  # edge opposite v2
        e1 = v2 - v1  # edge opposite v0
        e2 = v0 - v2  # edge opposite v1

        # Edge lengths: (F,)
        len0 = np.linalg.norm(e0, axis=1)
        len1 = np.linalg.norm(e1, axis=1)
        len2 = np.linalg.norm(e2, axis=1)

        edge_lengths = np.stack([len0, len1, len2], axis=1)  # (F, 3)

        # --- Aspect ratio ---
        longest = edge_lengths.max(axis=1)
        shortest = edge_lengths.min(axis=1)
        # Avoid division by zero for degenerate triangles
        safe_shortest = np.where(shortest > 0, shortest, 1.0)
        aspect_ratios = longest / safe_shortest
        # Degenerate triangles get infinite aspect ratio
        aspect_ratios = np.where(shortest > 0, aspect_ratios, np.inf)

        aspect_ratio_score = float(np.mean(aspect_ratios < 5.0))
        mean_aspect_ratio = float(np.nanmean(aspect_ratios[np.isfinite(aspect_ratios)]))

        # --- Minimum angle ---
        # Interior angles at each vertex using dot products of adjacent edges.
        # At v0: angle between edges (v1-v0) and (v2-v0)
        # At v1: angle between edges (v0-v1) and (v2-v1)
        # At v2: angle between edges (v0-v2) and (v1-v2)
        def _angles_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Angle in radians between edge vectors a and b (row-wise)."""
            dot = np.sum(a * b, axis=1)
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            denom = na * nb
            denom = np.where(denom > 0, denom, 1.0)
            cos_angle = np.clip(dot / denom, -1.0, 1.0)
            return np.arccos(cos_angle)

        angle0 = _angles_between(v1 - v0, v2 - v0)  # at v0
        angle1 = _angles_between(v0 - v1, v2 - v1)  # at v1
        angle2 = _angles_between(v0 - v2, v1 - v2)  # at v2

        angles = np.stack([angle0, angle1, angle2], axis=1)  # (F, 3)
        min_angles = angles.min(axis=1)  # (F,)
        min_angles_deg = np.degrees(min_angles)

        min_angle_score = float(np.mean(min_angles_deg > 15.0))
        mean_min_angle_deg = float(np.mean(min_angles_deg))

        # --- Area uniformity ---
        # Triangle area = 0.5 * |e0 x e1| (using cross product of two edges from same vertex)
        cross = np.cross(e0, e2 * -1)  # e0 = v1-v0, -e2 = v2-v0
        areas = 0.5 * np.linalg.norm(cross, axis=1)  # (F,)

        mean_area = np.mean(areas)
        if mean_area > 0:
            cv = float(np.std(areas) / mean_area)
        else:
            cv = float("inf")
        area_uniformity_score = max(0.0, 1.0 - cv)

        # --- Final score ---
        score = (
            0.4 * aspect_ratio_score
            + 0.4 * min_angle_score
            + 0.2 * area_uniformity_score
        )

        # Identify problem faces: aspect ratio >= 5 OR min angle <= 15 deg
        problem_mask = (aspect_ratios >= 5.0) | (min_angles_deg <= 15.0)
        problem_faces = np.where(problem_mask)[0].astype(np.int64)

        return MetricResult(
            name=self.name,
            score=float(score),
            weight=self.weight,
            details={
                "aspect_ratio_score": aspect_ratio_score,
                "min_angle_score": min_angle_score,
                "area_uniformity_score": area_uniformity_score,
                "mean_aspect_ratio": mean_aspect_ratio,
                "mean_min_angle_deg": mean_min_angle_deg,
                "area_cv": cv,
                "num_triangles": len(triangles),
                "num_problem_faces": int(len(problem_faces)),
            },
            problem_faces=problem_faces if len(problem_faces) > 0 else None,
        )
