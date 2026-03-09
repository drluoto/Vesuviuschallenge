"""Layer distance consistency metric.

Measures inter-layer distance consistency by casting rays along mesh normals
through the CT volume and detecting papyrus layer peaks. Consistent inter-peak
spacing indicates a well-segmented surface following a single layer, while
irregular spacing suggests sheet switching or other issues.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.volume import VolumeAccessor


class LayerDistanceMetric(MetricComputer):
    """Measure inter-layer distance consistency from CT volume data."""

    name: str = "layer_distance"
    weight: float = 0.10

    def __init__(
        self,
        volume_accessor: VolumeAccessor,
        *,
        n_samples: int = 300,
        ray_length: int = 200,
    ) -> None:
        self._volume = volume_accessor
        self._n_samples = n_samples
        self._ray_length = ray_length

    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        # Filter to in-bounds vertices
        in_bounds_mask = np.array(
            [self._volume.vertex_in_bounds(v, margin=self._ray_length) for v in vertices]
        )
        valid_indices = np.where(in_bounds_mask)[0]

        if len(valid_indices) == 0:
            return MetricResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                details={"n_sampled": 0, "reason": "no_in_bounds_vertices"},
            )

        # Sample random vertices, sorted by chunk for cache efficiency
        rng = np.random.default_rng(42)
        if len(valid_indices) > self._n_samples:
            valid_indices = rng.choice(valid_indices, self._n_samples, replace=False)
        sample_indices = self._volume.sort_by_chunk(vertices, valid_indices)

        all_distances: list[float] = []
        per_vertex_distances: dict[int, list[float]] = {}

        for vi in sample_indices:
            vertex = vertices[vi]
            normal = normals[vi]
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-10:
                continue
            normal = normal / normal_norm

            distances = self._sample_layer_distances(vertex, normal)
            if distances:
                all_distances.extend(distances)
                per_vertex_distances[int(vi)] = distances

        if len(all_distances) < 5:
            return MetricResult(
                name=self.name,
                score=0.0,
                weight=self.weight,
                details={
                    "n_sampled": len(sample_indices),
                    "n_distances": len(all_distances),
                    "reason": "insufficient_peaks",
                },
            )

        all_distances_arr = np.array(all_distances)
        mean_dist = float(np.mean(all_distances_arr))
        std_dist = float(np.std(all_distances_arr))
        cv = std_dist / mean_dist if mean_dist > 0 else 1.0
        score = 1.0 - min(cv / 0.5, 1.0)

        # Problem faces: vertices where local distance deviates >2 sigma from median
        median_dist = float(np.median(all_distances_arr))
        problem_vertices: list[int] = []
        for vi, dists in per_vertex_distances.items():
            for d in dists:
                if abs(d - median_dist) > 2 * std_dist:
                    problem_vertices.append(vi)
                    break

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
                "n_distances": len(all_distances),
                "mean_distance_voxels": mean_dist,
                "std_distance_voxels": std_dist,
                "cv": cv,
                "median_distance_voxels": median_dist,
                "n_problem_vertices": len(problem_vertices),
            },
            problem_faces=np.array(problem_face_indices, dtype=np.int64) if problem_face_indices else None,
        )

    def _sample_layer_distances(
        self, vertex_xyz: np.ndarray, normal_xyz: np.ndarray
    ) -> list[float]:
        """Cast ray along +/- normal, find peaks, return inter-peak distances."""
        s = self._volume.scale_factor
        # Convert mesh XYZ normal to volume ZYX direction
        normal_zyx = np.array([normal_xyz[2], normal_xyz[1], normal_xyz[0]]) / s

        # Center in volume coords
        center_zyx = np.array([
            vertex_xyz[2] / s,
            vertex_xyz[1] / s,
            vertex_xyz[0] / s,
        ])

        # Sample along ray: -ray_length to +ray_length voxel positions
        n_steps = 2 * self._ray_length + 1
        t_values = np.arange(-self._ray_length, self._ray_length + 1, dtype=np.float64)

        # Normalize direction so each step is 1 voxel
        dir_norm = np.linalg.norm(normal_zyx)
        if dir_norm < 1e-10:
            return []
        direction = normal_zyx / dir_norm

        profile = np.zeros(n_steps, dtype=np.float32)
        shape = self._volume.shape

        for i, t in enumerate(t_values):
            pos = center_zyx + t * direction
            iz, iy, ix = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
            if 0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]:
                # Use sample_neighborhood with half_size=1 to leverage cache
                try:
                    v_xyz = np.array([ix * s, iy * s, iz * s], dtype=np.float64)
                    chunk = self._volume.sample_neighborhood(v_xyz, half_size=1)
                    profile[i] = chunk[0, 0, 0]
                except (IndexError, ValueError):
                    profile[i] = 0.0
            else:
                profile[i] = 0.0

        # Find peaks using median as height threshold
        height_threshold = float(np.percentile(profile, 50))
        peaks, _ = find_peaks(profile, height=height_threshold, distance=8)

        if len(peaks) < 2:
            return []

        # Compute inter-peak distances in voxels
        return [float(peaks[i + 1] - peaks[i]) for i in range(len(peaks) - 1)]
