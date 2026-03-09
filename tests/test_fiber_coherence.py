"""Tests for fiber coherence metric."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import open3d as o3d
import pytest

from vesuvius_mesh_qa.metrics.fiber_coherence import (
    FiberCoherenceMetric,
    _compute_fiber_orientation_manual,
    _find_nnunet_model_dir,
)


def _make_grid_mesh(rows=20, cols=20):
    """Simple grid mesh for testing."""
    vertices = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            vertices.append([float(j), float(i), 0.0])
    triangles = []
    for i in range(rows):
        for j in range(cols):
            v0 = i * (cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (cols + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _mock_volume_with_fibers(horizontal=True):
    """Create a mock volume accessor that returns synthetic fiber-like patches.

    Generates patches with dominant gradient in one direction to simulate
    fiber orientation.
    """
    vol = MagicMock()
    vol.vertex_in_bounds.return_value = True

    def _sample(vertex, half_size=16):
        size = 2 * half_size
        patch = np.random.default_rng(42).normal(100, 10, (size, size, size)).astype(np.float32)
        # Add strong directional gradient to simulate fibers
        if horizontal:
            # Horizontal fibers: strong gradient in X direction
            for i in range(size):
                patch[:, :, i] += 50.0 * np.sin(i * np.pi / 4)
        else:
            # Vertical fibers: strong gradient in Z direction
            for i in range(size):
                patch[i, :, :] += 50.0 * np.sin(i * np.pi / 4)
        return patch

    vol.sample_neighborhood.side_effect = _sample
    return vol


class TestFiberCoherenceMetric:
    def test_metric_name_and_weight(self):
        vol = MagicMock()
        metric = FiberCoherenceMetric(vol)
        assert metric.name == "fiber_coherence"
        assert metric.weight == 0.10

    def test_empty_mesh(self):
        vol = MagicMock()
        mesh = o3d.geometry.TriangleMesh()
        metric = FiberCoherenceMetric(vol)
        result = metric.compute(mesh)
        assert result.score == 0.0

    def test_no_in_bounds_vertices(self):
        vol = MagicMock()
        vol.vertex_in_bounds.return_value = False
        mesh = _make_grid_mesh(10, 10)
        metric = FiberCoherenceMetric(vol, n_samples=50)
        result = metric.compute(mesh)
        assert result.score == 1.0
        assert result.details["n_sampled"] == 0

    def test_consistent_fibers_score_high(self):
        """Mesh where all vertices see same fiber orientation should score high."""
        vol = _mock_volume_with_fibers(horizontal=True)
        mesh = _make_grid_mesh(10, 10)
        metric = FiberCoherenceMetric(vol, n_samples=30, half_size=8, n_rings=2)
        result = metric.compute(mesh)
        # With consistent fiber orientation, should have few/no flips
        assert result.score >= 0.8
        assert result.details["n_sampled"] == 30

    def test_details_present(self):
        vol = _mock_volume_with_fibers(horizontal=True)
        mesh = _make_grid_mesh(10, 10)
        metric = FiberCoherenceMetric(vol, n_samples=20, half_size=8)
        result = metric.compute(mesh)
        assert "n_sampled" in result.details
        assert "n_compared" in result.details
        assert "n_class_flips" in result.details
        assert "flip_fraction" in result.details


class TestStructureTensorComputation:
    def test_manual_fallback_produces_output(self):
        """The manual scipy-based structure tensor should produce valid output."""
        vol = _mock_volume_with_fibers(horizontal=True)
        vertices = np.array([[5.0, 5.0, 0.0], [10.0, 10.0, 0.0]])
        sample_indices = np.array([0, 1])
        fiber_dirs, fiber_class, anisotropy = _compute_fiber_orientation_manual(
            vol, vertices, sample_indices, half_size=8
        )
        assert fiber_dirs.shape == (2, 3)
        assert len(fiber_class) == 2
        assert len(anisotropy) == 2
        # At least one should be classified
        assert np.any(fiber_class > 0)


class TestNnUNetModelDiscovery:
    def test_find_model_dir(self, tmp_path):
        """Should find the nnUNet trainer directory."""
        # Create mock directory structure
        d040 = tmp_path / "Dataset040_newHorizontals" / "nnUNetTrainer__nnUNetResEncUNetPlans_16G__3d_fullres" / "fold_0"
        d040.mkdir(parents=True)
        (d040.parent / "dataset.json").write_text("{}")
        (d040.parent / "plans.json").write_text("{}")

        result = _find_nnunet_model_dir(str(tmp_path), "Dataset040")
        assert result is not None
        assert "Dataset040" in result
        assert "16G" in result

    def test_find_model_dir_missing(self, tmp_path):
        """Should return None if dataset not found."""
        result = _find_nnunet_model_dir(str(tmp_path), "Dataset040")
        assert result is None

    def test_prefers_16g_over_40g(self, tmp_path):
        """Should prefer 16G variant over 40G for lower memory."""
        for variant in ["nnUNetTrainer__nnUNetResEncUNetPlans_16G__3d_fullres",
                        "nnUNetTrainer__nnUNetResEncUNetPlans_40G__3d_fullres"]:
            d = tmp_path / "Dataset040_newHorizontals" / variant / "fold_0"
            d.mkdir(parents=True)

        result = _find_nnunet_model_dir(str(tmp_path), "Dataset040")
        assert "16G" in result
