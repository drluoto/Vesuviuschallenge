"""Unit tests for individual mesh quality metrics."""

from __future__ import annotations

import numpy as np


class TestTriangleQuality:
    def test_perfect_plane_scores_high(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.triangle import TriangleQualityMetric
        result = TriangleQualityMetric().compute(perfect_plane)
        assert result.score > 0.8
        assert result.name == "triangle_quality"
        assert result.weight == 0.10

    def test_details_present(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.triangle import TriangleQualityMetric
        result = TriangleQualityMetric().compute(perfect_plane)
        assert "aspect_ratio_score" in result.details
        assert "min_angle_score" in result.details
        assert "area_uniformity_score" in result.details


class TestTopology:
    def test_perfect_plane_scores_high(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.topology import TopologyMetric
        result = TopologyMetric().compute(perfect_plane)
        assert result.score > 0.7
        assert result.name == "topology"

    def test_non_manifold_scores_lower(self, non_manifold_mesh):
        from vesuvius_mesh_qa.metrics.topology import TopologyMetric
        result = TopologyMetric().compute(non_manifold_mesh)
        # Non-manifold mesh should score lower than perfect
        assert result.details["is_edge_manifold"] is False or result.score < 1.0


class TestNormalConsistency:
    def test_perfect_plane_scores_high(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.normals import NormalConsistencyMetric
        result = NormalConsistencyMetric().compute(perfect_plane)
        assert result.score > 0.95
        assert result.name == "normal_consistency"

    def test_details_present(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.normals import NormalConsistencyMetric
        result = NormalConsistencyMetric().compute(perfect_plane)
        assert "mean_dihedral_angle_deg" in result.details
        assert "fraction_consistent" in result.details


class TestSheetSwitching:
    def test_perfect_plane_no_switching(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.normals import SheetSwitchingMetric
        result = SheetSwitchingMetric().compute(perfect_plane)
        assert result.score > 0.95
        assert result.name == "sheet_switching"
        assert result.details["n_switch_regions"] == 0

    def test_sheet_switch_mesh_detected(self, sheet_switch_mesh):
        from vesuvius_mesh_qa.metrics.normals import SheetSwitchingMetric
        result = SheetSwitchingMetric().compute(sheet_switch_mesh)
        # Should detect switching and score lower
        assert result.score < 1.0
        # The layer transition zone should be detected as a switch region
        assert result.details["n_switch_regions"] >= 1
        assert result.details["total_switch_area_fraction"] > 0


class TestNoise:
    def test_perfect_plane_no_noise(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.noise import NoiseMetric
        result = NoiseMetric().compute(perfect_plane)
        assert result.score > 0.9
        assert result.name == "noise"

    def test_spiked_mesh_detects_outliers(self, spiked_mesh):
        from vesuvius_mesh_qa.metrics.noise import NoiseMetric
        result = NoiseMetric().compute(spiked_mesh)
        assert result.details["n_outliers"] > 0
        assert result.score < 1.0


class TestSelfIntersections:
    def test_perfect_plane_no_intersections(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.intersections import SelfIntersectionMetric
        result = SelfIntersectionMetric().compute(perfect_plane)
        assert result.score > 0.9
        assert result.name == "self_intersections"
