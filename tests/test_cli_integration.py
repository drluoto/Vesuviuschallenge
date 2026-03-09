"""Tests for CLI integration of CT sheet switching metric."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest


class TestComputeAllMetricsWithVolume:
    """Test that compute_all_metrics handles volume_url parameter correctly."""

    def test_without_volume_url_returns_6_metrics(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        results = compute_all_metrics(perfect_plane)
        assert len(results) == 6

    def test_with_volume_url_returns_7_metrics(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics

        # Mock VolumeAccessor and CTSheetSwitchingMetric
        mock_accessor = MagicMock()
        mock_accessor.vertex_in_bounds.return_value = False  # No vertices in bounds

        with patch("vesuvius_mesh_qa.metrics.summary.VolumeAccessor", return_value=mock_accessor) as mock_va_cls, \
             patch("vesuvius_mesh_qa.metrics.summary.CTSheetSwitchingMetric") as mock_ct_cls:
            from vesuvius_mesh_qa.metrics.base import MetricResult
            mock_ct_instance = MagicMock()
            mock_ct_instance.name = "ct_sheet_switching"
            mock_ct_instance.weight = 0.20
            mock_ct_instance.compute.return_value = MetricResult(
                name="ct_sheet_switching",
                score=1.0,
                weight=0.20,
                details={"n_sampled": 0},
            )
            mock_ct_cls.return_value = mock_ct_instance

            results = compute_all_metrics(perfect_plane, volume_url="s3://fake/volume")
            assert len(results) == 9  # 6 base + CT + fiber_coherence + layer_distance
            metric_names = [r.name for r in results]
            assert "ct_sheet_switching" in metric_names
            mock_va_cls.assert_called_once_with("s3://fake/volume")
            mock_ct_cls.assert_called_once_with(mock_accessor)

    def test_volume_url_none_does_not_import_ct(self, perfect_plane):
        """When volume_url is None, CT metric classes should not be instantiated."""
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics

        with patch("vesuvius_mesh_qa.metrics.summary.VolumeAccessor") as mock_va_cls:
            results = compute_all_metrics(perfect_plane, volume_url=None)
            assert len(results) == 6
            mock_va_cls.assert_not_called()

    def test_progress_callback_total_with_volume(self, perfect_plane):
        """Progress callback should report total=7 when volume is provided."""
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        from vesuvius_mesh_qa.metrics.base import MetricResult

        mock_accessor = MagicMock()
        totals_seen = []

        def on_progress(name, idx, total):
            totals_seen.append(total)

        with patch("vesuvius_mesh_qa.metrics.summary.VolumeAccessor", return_value=mock_accessor), \
             patch("vesuvius_mesh_qa.metrics.summary.CTSheetSwitchingMetric") as mock_ct_cls:
            mock_ct_instance = MagicMock()
            mock_ct_instance.name = "ct_sheet_switching"
            mock_ct_instance.weight = 0.20
            mock_ct_instance.compute.return_value = MetricResult(
                name="ct_sheet_switching", score=1.0, weight=0.20, details={},
            )
            mock_ct_cls.return_value = mock_ct_instance

            compute_all_metrics(
                perfect_plane, volume_url="s3://fake/volume", on_progress=on_progress,
            )

        # All progress reports should show total=9 (6 base + CT + fiber + layer_distance)
        assert all(t == 9 for t in totals_seen), f"Expected all totals=9, got {totals_seen}"

    def test_progress_callback_total_without_volume(self, perfect_plane):
        """Progress callback should report total=6 when no volume."""
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        totals_seen = []

        def on_progress(name, idx, total):
            totals_seen.append(total)

        compute_all_metrics(perfect_plane, on_progress=on_progress)
        assert all(t == 6 for t in totals_seen)

    def test_weight_override_applies_to_ct_metric(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        from vesuvius_mesh_qa.metrics.base import MetricResult

        mock_accessor = MagicMock()

        with patch("vesuvius_mesh_qa.metrics.summary.VolumeAccessor", return_value=mock_accessor), \
             patch("vesuvius_mesh_qa.metrics.summary.CTSheetSwitchingMetric") as mock_ct_cls:
            mock_ct_instance = MagicMock()
            mock_ct_instance.name = "ct_sheet_switching"
            mock_ct_instance.weight = 0.20
            mock_ct_instance.compute.return_value = MetricResult(
                name="ct_sheet_switching", score=0.8, weight=0.20, details={},
            )
            mock_ct_cls.return_value = mock_ct_instance

            results = compute_all_metrics(
                perfect_plane,
                volume_url="s3://fake/volume",
                weight_overrides={"ct_sheet_switching": 0.5},
            )
            ct_result = [r for r in results if r.name == "ct_sheet_switching"][0]
            # Weight override should have been applied to the instance
            assert mock_ct_instance.weight == 0.5


class TestCLIVolumeOption:
    """Test that the CLI score command accepts --volume option."""

    def test_cli_score_has_volume_option(self):
        from vesuvius_mesh_qa.cli import score
        param_names = [p.name for p in score.params]
        assert "volume" in param_names

    def test_cli_score_volume_option_is_optional(self):
        from vesuvius_mesh_qa.cli import score
        volume_param = [p for p in score.params if p.name == "volume"][0]
        assert volume_param.default is None
        assert not volume_param.required


class TestCLIScrollConfig:
    """Test that the CLI accepts --scroll-config option."""

    def test_cli_score_has_scroll_config_option(self):
        from vesuvius_mesh_qa.cli import score
        param_names = [p.name for p in score.params]
        assert "scroll_config" in param_names

    def test_cli_batch_has_scroll_config_option(self):
        from vesuvius_mesh_qa.cli import batch
        param_names = [p.name for p in batch.params]
        assert "scroll_config" in param_names

    def test_cli_batch_has_volume_and_umbilicus(self):
        from vesuvius_mesh_qa.cli import batch
        param_names = [p.name for p in batch.params]
        assert "volume" in param_names
        assert "umbilicus" in param_names

    def test_cli_score_has_fiber_options(self):
        from vesuvius_mesh_qa.cli import score
        param_names = [p.name for p in score.params]
        assert "fiber_model" in param_names
        assert "fiber_predictions" in param_names

    def test_cli_batch_has_fiber_options(self):
        from vesuvius_mesh_qa.cli import batch
        param_names = [p.name for p in batch.params]
        assert "fiber_model" in param_names
        assert "fiber_predictions" in param_names


class TestTierWeights:
    """Test that tier-based weight redistribution works correctly."""

    def test_tier1_weights_sum_to_one(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        results = compute_all_metrics(perfect_plane)
        total = sum(r.weight for r in results)
        assert abs(total - 1.0) < 0.01, f"Tier 1 weights sum to {total}, expected 1.0"

    def test_tier1_has_6_metrics(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        results = compute_all_metrics(perfect_plane)
        assert len(results) == 6

    def test_tier1_sheet_switching_weight(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        results = compute_all_metrics(perfect_plane)
        ss = [r for r in results if r.name == "sheet_switching"][0]
        assert ss.weight == 0.30
