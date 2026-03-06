"""CT-informed sheet switching metric.

Compares mesh vertex normals against CT-derived structure tensor normals
to detect regions where the mesh surface has switched between papyrus layers.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.ct_normals import compute_ct_normal
from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.volume import VolumeAccessor


class CTSheetSwitchingMetric(MetricComputer):
    """Detect sheet switching by comparing mesh normals to CT structure normals."""

    name: str = "ct_sheet_switching"
    weight: float = 0.20

    def __init__(
        self,
        volume_accessor: VolumeAccessor,
        *,
        n_samples: int = 500,
        half_size: int = 16,
        sigma: float = 3.0,
        misalignment_threshold_deg: float = 45.0,
        anisotropy_threshold: float = 0.1,
    ) -> None:
        self._volume = volume_accessor
        self._n_samples = n_samples
        self._half_size = half_size
        self._sigma = sigma
        self._misalignment_threshold_deg = misalignment_threshold_deg
        self._anisotropy_threshold = anisotropy_threshold

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        """Compute sheet switching score by sampling mesh vertices."""
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        # Filter to in-bounds vertices
        in_bounds_mask = np.array(
            [self._volume.vertex_in_bounds(v, margin=self._half_size) for v in vertices]
        )
        valid_indices = np.where(in_bounds_mask)[0]

        if len(valid_indices) == 0:
            return MetricResult(
                name=self.name,
                score=1.0,
                weight=self.weight,
                details={
                    "n_sampled": 0,
                    "mean_angle_deg": 0.0,
                    "median_angle_deg": 0.0,
                    "fraction_misaligned": 0.0,
                    "mean_anisotropy": 0.0,
                    "n_problem_vertices": 0,
                },
            )

        # Sample up to n_samples random vertices
        rng = np.random.default_rng(42)
        if len(valid_indices) > self._n_samples:
            sample_indices = rng.choice(valid_indices, self._n_samples, replace=False)
        else:
            sample_indices = valid_indices

        angles = np.zeros(len(sample_indices))
        anisotropies = np.zeros(len(sample_indices))
        problem_vertices: list[int] = []

        for i, vi in enumerate(sample_indices):
            vertex = vertices[vi]
            mesh_normal = normals[vi]
            mesh_normal_norm = np.linalg.norm(mesh_normal)
            if mesh_normal_norm < 1e-10:
                angles[i] = 0.0
                anisotropies[i] = 0.0
                continue
            mesh_normal = mesh_normal / mesh_normal_norm

            # Fetch CT neighborhood and compute structure tensor normal
            chunk = self._volume.sample_neighborhood(vertex, half_size=self._half_size)
            ct_normal_zyx, anisotropy = compute_ct_normal(chunk, sigma=self._sigma)

            # Reorder ZYX -> XYZ to match mesh convention
            ct_normal_xyz = ct_normal_zyx[[2, 1, 0]]
            ct_norm = np.linalg.norm(ct_normal_xyz)
            if ct_norm < 1e-10:
                angles[i] = 0.0
                anisotropies[i] = anisotropy
                continue
            ct_normal_xyz = ct_normal_xyz / ct_norm

            # Use abs(dot) to handle sign ambiguity (normal can point either way)
            dot = np.abs(np.dot(mesh_normal, ct_normal_xyz))
            dot = np.clip(dot, 0.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(dot)))

            angles[i] = angle_deg
            anisotropies[i] = anisotropy

            # Track problem vertices: high anisotropy AND large angle
            if anisotropy > self._anisotropy_threshold and angle_deg > self._misalignment_threshold_deg:
                problem_vertices.append(int(vi))

        score = self._compute_score(angles, anisotropies)

        # Compute fraction_misaligned among structured vertices
        structured_mask = anisotropies > self._anisotropy_threshold
        if structured_mask.any():
            fraction_misaligned = float(
                np.sum(angles[structured_mask] > self._misalignment_threshold_deg)
                / np.sum(structured_mask)
            )
        else:
            fraction_misaligned = 0.0

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_sampled": len(sample_indices),
                "mean_angle_deg": float(np.mean(angles)) if len(angles) > 0 else 0.0,
                "median_angle_deg": float(np.median(angles)) if len(angles) > 0 else 0.0,
                "fraction_misaligned": fraction_misaligned,
                "mean_anisotropy": float(np.mean(anisotropies)) if len(anisotropies) > 0 else 0.0,
                "n_problem_vertices": len(problem_vertices),
            },
            problem_faces=np.array(problem_vertices) if problem_vertices else None,
        )

    def _compute_score(
        self, angles: np.ndarray, anisotropies: np.ndarray
    ) -> float:
        """Compute score from angles and anisotropies.

        Only vertices with anisotropy above threshold (clear structure) count.
        Score = 1.0 - fraction_bad among structured vertices.
        """
        if len(angles) == 0:
            return 1.0

        structured_mask = anisotropies > self._anisotropy_threshold
        total_structured = int(np.sum(structured_mask))

        if total_structured == 0:
            return 1.0

        n_bad = int(np.sum(angles[structured_mask] > self._misalignment_threshold_deg))
        fraction_bad = n_bad / total_structured
        return 1.0 - fraction_bad
