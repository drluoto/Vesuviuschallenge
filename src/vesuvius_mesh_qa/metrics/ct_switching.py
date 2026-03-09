"""CT-informed sheet switching metric.

Compares mesh vertex normals against CT-derived structure tensor normals
to detect regions where the mesh surface has switched between papyrus layers.

Baseline: even correctly-segmented papyrus has ~25-30° median alignment due
to structure tensor noise and multi-layer interference. The scoring uses
continuous angular distance rather than a hard threshold to maximize
dynamic range.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.ct_normals import compute_ct_normal
from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.volume import VolumeAccessor

# Expected median angle on correctly-segmented papyrus (~25-30°).
# Angles significantly above this suggest sheet switching.
_BASELINE_MEDIAN_DEG = 25.0
# Angle at which a vertex is considered severely misaligned.
_SEVERE_ANGLE_DEG = 60.0


class CTSheetSwitchingMetric(MetricComputer):
    """Detect sheet switching by comparing mesh normals to CT structure normals."""

    name: str = "ct_sheet_switching"
    weight: float = 0.10

    def __init__(
        self,
        volume_accessor: VolumeAccessor,
        *,
        n_samples: int = 500,
        half_size: int = 16,
        sigma: float = 3.0,
    ) -> None:
        self._volume = volume_accessor
        self._n_samples = n_samples
        self._half_size = half_size
        self._sigma = sigma

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        """Compute CT alignment score by sampling mesh vertices.

        This metric provides a statistical signal (score) comparing mesh
        normals to CT structure tensor normals. Useful as a global quality
        indicator but NOT for spatial detection — the structure tensor is
        too noisy (~25-30° baseline) for reliable per-vertex flagging.

        For spatial sheet-switch detection, use fiber_coherence and
        winding_angle metrics instead (community-validated approaches).
        """
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
                    "fraction_severe": 0.0,
                },
            )

        # Sample random vertices, sorted by chunk for cache efficiency
        rng = np.random.default_rng(42)
        if len(valid_indices) > self._n_samples:
            valid_indices = rng.choice(valid_indices, self._n_samples, replace=False)
        sample_indices = self._volume.sort_by_chunk(vertices, valid_indices)

        angles = np.zeros(len(sample_indices))
        problem_vertices: list[int] = []

        for i, vi in enumerate(sample_indices):
            vertex = vertices[vi]
            mesh_normal = normals[vi]
            mesh_normal_norm = np.linalg.norm(mesh_normal)
            if mesh_normal_norm < 1e-10:
                angles[i] = 0.0
                continue
            mesh_normal = mesh_normal / mesh_normal_norm

            chunk = self._volume.sample_neighborhood(vertex, half_size=self._half_size)
            ct_normal_zyx, _ = compute_ct_normal(chunk, sigma=self._sigma)

            # Reorder ZYX -> XYZ to match mesh convention
            ct_normal_xyz = ct_normal_zyx[[2, 1, 0]]
            ct_norm = np.linalg.norm(ct_normal_xyz)
            if ct_norm < 1e-10:
                angles[i] = 0.0
                continue
            ct_normal_xyz = ct_normal_xyz / ct_norm

            dot = np.abs(np.dot(mesh_normal, ct_normal_xyz))
            dot = np.clip(dot, 0.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(dot)))

            angles[i] = angle_deg

            if angle_deg > _SEVERE_ANGLE_DEG:
                problem_vertices.append(int(vi))

        score = self._compute_score(angles)

        fraction_severe = float(np.sum(angles > _SEVERE_ANGLE_DEG) / len(angles))

        # Convert problem vertices to problem faces (vectorized)
        problem_face_indices: list[int] = []
        if problem_vertices:
            triangles = np.asarray(mesh.triangles)
            problem_set = set(problem_vertices)
            mask = np.array([
                tri[0] in problem_set or tri[1] in problem_set or tri[2] in problem_set
                for tri in triangles
            ])
            problem_face_indices = np.where(mask)[0].tolist()

        return MetricResult(
            name=self.name,
            score=score,
            weight=self.weight,
            details={
                "n_sampled": len(sample_indices),
                "mean_angle_deg": float(np.mean(angles)) if len(angles) > 0 else 0.0,
                "median_angle_deg": float(np.median(angles)) if len(angles) > 0 else 0.0,
                "fraction_severe": fraction_severe,
            },
            problem_faces=np.array(problem_face_indices, dtype=np.int64) if problem_face_indices else None,
        )

    @staticmethod
    def _compute_score(angles: np.ndarray) -> float:
        """Compute continuous score from per-vertex angles.

        Uses mean cosine similarity: score = mean(cos(angle)).
        This gives smooth, continuous scoring with natural dynamic range:
          - All at 0° → 1.000
          - Median ~25° (good baseline) → ~0.900
          - Median ~35° (some issues) → ~0.820
          - Median ~50° (severe) → ~0.640
          - All at 90° → 0.000
        """
        if len(angles) == 0:
            return 1.0
        radians = np.radians(angles)
        return float(np.mean(np.cos(radians)))
