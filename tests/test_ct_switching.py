"""Tests for CT sheet switching metric."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import open3d as o3d
import pytest

from vesuvius_mesh_qa.metrics.ct_switching import CTSheetSwitchingMetric


def _make_mesh_with_normals(
    vertices: np.ndarray, normals: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """Create a minimal mesh with given vertices and normals."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    n = len(vertices)
    if n >= 3:
        triangles = [[i, (i + 1) % n, (i + 2) % n] for i in range(0, n - 2)]
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


class TestComputeScore:
    """Test _compute_score with continuous cosine scoring."""

    def test_perfect_alignment(self):
        angles = np.array([0.0, 0.0, 0.0, 0.0])
        score = CTSheetSwitchingMetric._compute_score(angles)
        assert score == 1.0

    def test_all_perpendicular(self):
        angles = np.array([90.0, 90.0, 90.0])
        score = CTSheetSwitchingMetric._compute_score(angles)
        assert abs(score) < 1e-6

    def test_mixed_angles(self):
        # mean(cos([10, 20, 30, 40]°)) ≈ mean([0.985, 0.940, 0.866, 0.766]) ≈ 0.889
        angles = np.array([10.0, 20.0, 30.0, 40.0])
        score = CTSheetSwitchingMetric._compute_score(angles)
        assert 0.85 < score < 0.92

    def test_high_angles_score_low(self):
        angles = np.array([60.0, 70.0, 80.0, 90.0])
        score = CTSheetSwitchingMetric._compute_score(angles)
        assert score < 0.4

    def test_empty_arrays(self):
        score = CTSheetSwitchingMetric._compute_score(np.array([]))
        assert score == 1.0

    def test_baseline_range(self):
        # Typical good mesh: median ~25°
        angles = np.random.default_rng(42).uniform(10, 40, 500)
        score = CTSheetSwitchingMetric._compute_score(angles)
        assert 0.80 < score < 0.95


class TestCTSheetSwitchingMetric:
    """Integration tests with mocked VolumeAccessor."""

    def _make_mock_accessor(self) -> MagicMock:
        accessor = MagicMock()
        accessor.vertex_in_bounds.return_value = True
        accessor.sample_neighborhood.return_value = np.zeros(
            (32, 32, 32), dtype=np.float32
        )
        return accessor

    def test_well_aligned_mesh_scores_high(self, monkeypatch):
        """When mesh normals match CT normals, score should be high."""
        n = 20
        ct_normal_zyx = np.array([1.0, 0.0, 0.0])
        # After ZYX->XYZ reorder, ct_normal becomes [0, 0, 1] -> aligned with mesh Z
        mesh_normals = np.array([[0.0, 0.0, 1.0]] * n)
        rng = np.random.default_rng(123)
        for i in range(n):
            perturb = rng.normal(0, 0.15, 3)
            mesh_normals[i] += perturb
            mesh_normals[i] /= np.linalg.norm(mesh_normals[i])

        vertices = rng.uniform(100, 900, (n, 3))
        mesh = _make_mesh_with_normals(vertices, mesh_normals)
        accessor = self._make_mock_accessor()

        def mock_compute_ct_normal(chunk, sigma=3.0):
            return ct_normal_zyx.copy(), 0.6

        monkeypatch.setattr(
            "vesuvius_mesh_qa.metrics.ct_switching.compute_ct_normal",
            mock_compute_ct_normal,
        )

        metric = CTSheetSwitchingMetric(accessor, n_samples=n)
        result = metric.compute(mesh)

        assert result.score > 0.9
        assert result.name == "ct_sheet_switching"
        assert result.details["n_sampled"] == n

    def test_misaligned_region_scores_low(self, monkeypatch):
        """Perpendicular normals should produce low score."""
        n = 20
        ct_normal_zyx = np.array([1.0, 0.0, 0.0])
        # All mesh normals perpendicular to CT (X-pointing vs Z-pointing CT)
        mesh_normals = np.array([[1.0, 0.0, 0.0]] * n)

        rng = np.random.default_rng(99)
        vertices = rng.uniform(100, 900, (n, 3))
        mesh = _make_mesh_with_normals(vertices, mesh_normals)
        accessor = self._make_mock_accessor()

        def mock_compute_ct_normal(chunk, sigma=3.0):
            return ct_normal_zyx.copy(), 0.6

        monkeypatch.setattr(
            "vesuvius_mesh_qa.metrics.ct_switching.compute_ct_normal",
            mock_compute_ct_normal,
        )

        metric = CTSheetSwitchingMetric(accessor, n_samples=n)
        result = metric.compute(mesh)

        assert result.score < 0.2

    def test_weight_is_point_one(self):
        accessor = self._make_mock_accessor()
        metric = CTSheetSwitchingMetric(accessor)
        assert metric.weight == 0.10
