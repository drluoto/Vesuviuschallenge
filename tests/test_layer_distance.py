"""Tests for layer distance consistency metric."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from vesuvius_mesh_qa.metrics.layer_distance import LayerDistanceMetric


class MockVolumeAccessor:
    """Mock VolumeAccessor that returns synthetic intensity profiles."""

    def __init__(
        self,
        *,
        shape: tuple[int, int, int] = (1000, 1000, 1000),
        chunks: tuple[int, int, int] = (128, 128, 128),
        scale: int = 0,
        profile_fn=None,
    ):
        self._shape = shape
        self._chunks = chunks
        self._scale_factor = 2 ** scale
        self._profile_fn = profile_fn or self._default_profile

    @property
    def shape(self):
        return self._shape

    @property
    def chunks(self):
        return self._chunks

    @property
    def scale_factor(self):
        return self._scale_factor

    @staticmethod
    def _default_profile(iz, iy, ix):
        return 100.0

    def vertex_in_bounds(self, vertex_xyz, margin=16):
        s = self._scale_factor
        x, y, z = vertex_xyz
        iz, iy, ix = int(round(z / s)), int(round(y / s)), int(round(x / s))
        return (margin <= iz < self._shape[0] - margin and
                margin <= iy < self._shape[1] - margin and
                margin <= ix < self._shape[2] - margin)

    def sample_neighborhood(self, vertex_xyz, half_size=16):
        s = self._scale_factor
        x, y, z = vertex_xyz
        iz, iy, ix = int(round(z / s)), int(round(y / s)), int(round(x / s))
        h = half_size
        result = np.zeros((2 * h, 2 * h, 2 * h), dtype=np.float32)
        for dz in range(2 * h):
            for dy in range(2 * h):
                for dx in range(2 * h):
                    result[dz, dy, dx] = self._profile_fn(
                        iz - h + dz, iy - h + dy, ix - h + dx
                    )
        return result

    def sort_by_chunk(self, vertices_xyz, indices):
        return indices


def _make_mesh(n_vertices: int = 50, center: float = 500.0) -> o3d.geometry.TriangleMesh:
    """Create a simple planar mesh with normals along Z axis."""
    rng = np.random.default_rng(123)
    vertices = np.zeros((n_vertices, 3))
    vertices[:, 0] = center + rng.uniform(-10, 10, n_vertices)  # X
    vertices[:, 1] = center + rng.uniform(-10, 10, n_vertices)  # Y
    vertices[:, 2] = center  # Z (flat plane)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Create triangles connecting consecutive vertices
    triangles = []
    for i in range(n_vertices - 2):
        triangles.append([i, i + 1, i + 2])
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    # Set normals along Z
    normals = np.zeros((n_vertices, 3))
    normals[:, 2] = 1.0  # normal along Z
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    return mesh


class TestLayerDistanceConsistentLayers:
    """Test with regularly spaced peaks -> high score."""

    def test_consistent_layers_high_score(self):
        spacing = 20  # consistent 20-voxel spacing

        def regular_peaks(iz, iy, ix):
            # Peaks every `spacing` voxels along Z
            if iz % spacing < 3:
                return 200.0
            return 50.0

        mock_vol = MockVolumeAccessor(profile_fn=regular_peaks)
        metric = LayerDistanceMetric(mock_vol, n_samples=30, ray_length=100)
        mesh = _make_mesh(n_vertices=40, center=500.0)

        result = metric.compute(mesh)
        assert result.name == "layer_distance"
        assert result.score > 0.8, f"Expected high score for consistent layers, got {result.score}"
        assert result.details["n_distances"] > 0


class TestLayerDistanceIrregularLayers:
    """Test with irregularly spaced peaks -> low score."""

    def test_irregular_layers_low_score(self):
        # Pre-generate irregular peak positions
        rng = np.random.default_rng(99)
        peak_positions = sorted(rng.choice(range(0, 1000, 1), size=40, replace=False))

        def irregular_peaks(iz, iy, ix):
            for p in peak_positions:
                if abs(iz - p) < 2:
                    return 200.0
            return 50.0

        mock_vol = MockVolumeAccessor(profile_fn=irregular_peaks)
        metric = LayerDistanceMetric(mock_vol, n_samples=30, ray_length=100)
        mesh = _make_mesh(n_vertices=40, center=500.0)

        result = metric.compute(mesh)
        assert result.score < 0.5, f"Expected low score for irregular layers, got {result.score}"


class TestLayerDistanceNoPeaks:
    """Test with flat profile (no peaks) -> score 0.0."""

    def test_no_peaks_score_zero(self):
        def flat_profile(iz, iy, ix):
            return 100.0  # constant -> no peaks

        mock_vol = MockVolumeAccessor(profile_fn=flat_profile)
        metric = LayerDistanceMetric(mock_vol, n_samples=30, ray_length=100)
        mesh = _make_mesh(n_vertices=40, center=500.0)

        result = metric.compute(mesh)
        assert result.score == 0.0, f"Expected score 0.0 for no peaks, got {result.score}"
        assert "insufficient_peaks" in result.details.get("reason", "")


class TestLayerDistanceOutOfBounds:
    """Test with vertices outside volume bounds."""

    def test_out_of_bounds_handled(self):
        def some_profile(iz, iy, ix):
            return 100.0

        # Small volume so vertices at center=500 are out of bounds
        mock_vol = MockVolumeAccessor(
            shape=(100, 100, 100),
            profile_fn=some_profile,
        )
        metric = LayerDistanceMetric(mock_vol, n_samples=30, ray_length=200)
        mesh = _make_mesh(n_vertices=40, center=500.0)

        result = metric.compute(mesh)
        assert result.score == 0.0
        assert result.details["n_sampled"] == 0
        assert result.details.get("reason") == "no_in_bounds_vertices"
