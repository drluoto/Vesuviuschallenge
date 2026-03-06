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

    def test_parallel_layer_switch_detected(self, parallel_layer_switch_mesh):
        """Edge-length detector catches switches between parallel layers."""
        from vesuvius_mesh_qa.metrics.normals import SheetSwitchingMetric
        result = SheetSwitchingMetric().compute(parallel_layer_switch_mesh)
        assert result.score < 1.0
        assert result.details["n_edge_flagged"] > 0


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

    def test_intersecting_mesh_detected(self, self_intersecting_mesh):
        from vesuvius_mesh_qa.metrics.intersections import SelfIntersectionMetric
        result = SelfIntersectionMetric().compute(self_intersecting_mesh)
        assert result.score < 1.0
        assert result.details["n_intersecting_pairs"] > 0

    def test_cross_validate_no_intersection_with_open3d(self, perfect_plane):
        """Cross-validate: Open3D and our metric agree on no intersections."""
        from vesuvius_mesh_qa.metrics.intersections import SelfIntersectionMetric
        o3d_result = perfect_plane.is_self_intersecting()
        our_result = SelfIntersectionMetric().compute(perfect_plane)
        assert o3d_result is False
        assert our_result.details["n_intersecting_pairs"] == 0

    def test_cross_validate_intersection_with_open3d(self, self_intersecting_mesh):
        """Cross-validate: Open3D and our metric both detect intersections."""
        from vesuvius_mesh_qa.metrics.intersections import SelfIntersectionMetric
        o3d_result = self_intersecting_mesh.is_self_intersecting()
        our_result = SelfIntersectionMetric().compute(self_intersecting_mesh)
        assert o3d_result is True
        assert our_result.details["n_intersecting_pairs"] > 0


class TestVisualization:
    def test_export_ply(self, perfect_plane, tmp_path):
        """Visualization exports a valid PLY file with vertex colors."""
        import open3d as o3d
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        from vesuvius_mesh_qa.report.visualize import export_visualization

        results = compute_all_metrics(perfect_plane)
        out_path = tmp_path / "viz.ply"
        export_visualization(perfect_plane, results, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Reload and check it has vertex colors
        loaded = o3d.io.read_triangle_mesh(str(out_path))
        assert loaded.has_vertex_colors()
        assert len(loaded.vertices) == len(perfect_plane.vertices)
        assert len(loaded.triangles) == len(perfect_plane.triangles)
