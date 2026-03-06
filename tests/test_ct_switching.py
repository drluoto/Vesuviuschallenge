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
    # Need at least one triangle for it to be a valid mesh
    n = len(vertices)
    if n >= 3:
        triangles = [[i, (i + 1) % n, (i + 2) % n] for i in range(0, n - 2)]
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


class TestComputeScoreDirectly:
    """Test _compute_score in isolation without mesh or volume."""

    def _make_metric(self) -> CTSheetSwitchingMetric:
        metric = CTSheetSwitchingMetric.__new__(CTSheetSwitchingMetric)
        metric.name = "ct_sheet_switching"
        metric.weight = 0.20
        metric._misalignment_threshold_deg = 45.0
        metric._anisotropy_threshold = 0.1
        return metric

    def test_all_well_aligned(self):
        metric = self._make_metric()
        angles = np.array([5.0, 10.0, 15.0, 20.0])
        anisotropies = np.array([0.5, 0.6, 0.7, 0.8])
        score = metric._compute_score(angles, anisotropies)
        assert score == 1.0  # all below threshold

    def test_all_misaligned(self):
        metric = self._make_metric()
        angles = np.array([60.0, 70.0, 80.0, 90.0])
        anisotropies = np.array([0.5, 0.6, 0.7, 0.8])
        score = metric._compute_score(angles, anisotropies)
        assert score == 0.0  # all above threshold

    def test_mixed_alignment(self):
        metric = self._make_metric()
        # 7 good + 3 bad = 30% bad -> score 0.7
        angles = np.array([10.0] * 7 + [60.0] * 3)
        anisotropies = np.full(10, 0.5)
        score = metric._compute_score(angles, anisotropies)
        assert abs(score - 0.7) < 1e-6

    def test_low_anisotropy_ignored(self):
        metric = self._make_metric()
        # Bad angles but all low anisotropy -> no structured vertices -> 1.0
        angles = np.array([60.0, 70.0, 80.0])
        anisotropies = np.array([0.01, 0.05, 0.09])
        score = metric._compute_score(angles, anisotropies)
        assert score == 1.0

    def test_empty_arrays(self):
        metric = self._make_metric()
        score = metric._compute_score(np.array([]), np.array([]))
        assert score == 1.0


class TestCTSheetSwitchingMetric:
    """Integration tests with mocked VolumeAccessor."""

    def _make_mock_accessor(
        self,
        ct_normals_zyx: list[np.ndarray],
        anisotropies: list[float],
    ) -> MagicMock:
        """Create a mock VolumeAccessor that returns synthetic neighborhoods."""
        accessor = MagicMock()
        accessor.vertex_in_bounds.return_value = True
        # sample_neighborhood returns a dummy chunk; we mock compute_ct_normal instead
        accessor.sample_neighborhood.return_value = np.zeros(
            (32, 32, 32), dtype=np.float32
        )
        # Store the normals/anisotropies to be returned by the patched compute_ct_normal
        accessor._ct_normals_zyx = ct_normals_zyx
        accessor._anisotropies = anisotropies
        return accessor

    def test_well_aligned_mesh_scores_high(self, monkeypatch):
        """When mesh normals match CT normals, score should be high."""
        n = 20
        # CT normals in ZYX that point along Z
        ct_normal_zyx = np.array([1.0, 0.0, 0.0])
        # Mesh normals in XYZ: Z-pointing = [0, 0, 1]
        # After ZYX->XYZ reorder, ct_normal becomes [0, 0, 1] -> aligned
        mesh_normals = np.array([[0.0, 0.0, 1.0]] * n)
        # Add small perturbation (5-20 deg)
        rng = np.random.default_rng(123)
        for i in range(n):
            perturb = rng.normal(0, 0.15, 3)
            mesh_normals[i] += perturb
            mesh_normals[i] /= np.linalg.norm(mesh_normals[i])

        vertices = rng.uniform(100, 900, (n, 3))
        mesh = _make_mesh_with_normals(vertices, mesh_normals)

        accessor = self._make_mock_accessor(
            ct_normals_zyx=[ct_normal_zyx] * n,
            anisotropies=[0.6] * n,
        )

        call_idx = {"i": 0}

        def mock_compute_ct_normal(chunk, sigma=3.0):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return accessor._ct_normals_zyx[idx], accessor._anisotropies[idx]

        monkeypatch.setattr(
            "vesuvius_mesh_qa.metrics.ct_switching.compute_ct_normal",
            mock_compute_ct_normal,
        )

        metric = CTSheetSwitchingMetric(accessor, n_samples=n)
        result = metric.compute(mesh)

        assert result.score > 0.8
        assert result.name == "ct_sheet_switching"
        assert result.details["n_sampled"] == n

    def test_misaligned_region_scores_low(self, monkeypatch):
        """30% misaligned vertices should produce score < 0.8."""
        n = 20
        n_good = 14
        n_bad = 6

        ct_normal_zyx = np.array([1.0, 0.0, 0.0])

        # Good: Z-pointing mesh normals (aligned with CT after reorder)
        good_normals = np.array([[0.0, 0.0, 1.0]] * n_good)
        # Bad: X-pointing mesh normals (perpendicular to CT)
        bad_normals = np.array([[1.0, 0.0, 0.0]] * n_bad)
        mesh_normals = np.vstack([good_normals, bad_normals])

        rng = np.random.default_rng(99)
        vertices = rng.uniform(100, 900, (n, 3))
        mesh = _make_mesh_with_normals(vertices, mesh_normals)

        accessor = self._make_mock_accessor(
            ct_normals_zyx=[ct_normal_zyx] * n,
            anisotropies=[0.6] * n,
        )

        call_idx = {"i": 0}

        def mock_compute_ct_normal(chunk, sigma=3.0):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return accessor._ct_normals_zyx[idx], accessor._anisotropies[idx]

        monkeypatch.setattr(
            "vesuvius_mesh_qa.metrics.ct_switching.compute_ct_normal",
            mock_compute_ct_normal,
        )

        metric = CTSheetSwitchingMetric(accessor, n_samples=n)
        result = metric.compute(mesh)

        assert result.score < 0.8
        assert result.details["n_problem_vertices"] > 0

    def test_low_anisotropy_ignored(self, monkeypatch):
        """Bad angles with low anisotropy should still score high."""
        n = 20
        # CT normals in ZYX pointing Z
        ct_normal_zyx = np.array([1.0, 0.0, 0.0])
        # Mesh normals perpendicular (X-pointing) -> would be 90 deg misaligned
        mesh_normals = np.array([[1.0, 0.0, 0.0]] * n)

        rng = np.random.default_rng(77)
        vertices = rng.uniform(100, 900, (n, 3))
        mesh = _make_mesh_with_normals(vertices, mesh_normals)

        accessor = self._make_mock_accessor(
            ct_normals_zyx=[ct_normal_zyx] * n,
            anisotropies=[0.05] * n,  # low anisotropy -> no clear structure
        )

        call_idx = {"i": 0}

        def mock_compute_ct_normal(chunk, sigma=3.0):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return accessor._ct_normals_zyx[idx], accessor._anisotropies[idx]

        monkeypatch.setattr(
            "vesuvius_mesh_qa.metrics.ct_switching.compute_ct_normal",
            mock_compute_ct_normal,
        )

        metric = CTSheetSwitchingMetric(accessor, n_samples=n)
        result = metric.compute(mesh)

        assert result.score > 0.8
        assert result.details["n_problem_vertices"] == 0
