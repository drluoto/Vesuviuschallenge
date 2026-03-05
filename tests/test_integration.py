"""Integration tests for the full scoring pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d


class TestFullPipeline:
    def test_compute_all_metrics(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics, aggregate_score, letter_grade
        results = compute_all_metrics(perfect_plane)
        assert len(results) == 6
        agg = aggregate_score(results)
        assert 0.0 <= agg <= 1.0
        grade = letter_grade(agg)
        assert grade in ("A", "B", "C", "D", "F")

    def test_perfect_plane_high_aggregate(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics, aggregate_score, letter_grade
        results = compute_all_metrics(perfect_plane)
        agg = aggregate_score(results)
        assert agg > 0.7
        grade = letter_grade(agg)
        assert grade in ("A", "B")

    def test_json_report(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics, aggregate_score, letter_grade
        from vesuvius_mesh_qa.report.json_report import build_json_report
        results = compute_all_metrics(perfect_plane)
        agg = aggregate_score(results)
        grade = letter_grade(agg)
        report = build_json_report(Path("test.obj"), perfect_plane, results, agg, grade)
        # Verify JSON-serializable
        json_str = json.dumps(report)
        assert "aggregate_score" in json_str
        assert "metrics" in report
        assert len(report["metrics"]) == 6

    def test_weight_overrides(self, perfect_plane):
        from vesuvius_mesh_qa.metrics.summary import compute_all_metrics
        overrides = {"triangle_quality": 0.5, "sheet_switching": 0.0}
        results = compute_all_metrics(perfect_plane, weight_overrides=overrides)
        tq = [r for r in results if r.name == "triangle_quality"][0]
        ss = [r for r in results if r.name == "sheet_switching"][0]
        assert tq.weight == 0.5
        assert ss.weight == 0.0


class TestDiscovery:
    def test_discover_in_flat_dir(self, perfect_plane, tmp_path):
        from vesuvius_mesh_qa.io.discovery import discover_segments
        # Save mesh to tmp dir
        obj_path = tmp_path / "test_segment.obj"
        o3d.io.write_triangle_mesh(str(obj_path), perfect_plane)
        segments = discover_segments(tmp_path)
        assert len(segments) == 1
        assert segments[0].segment_id == "test_segment"

    def test_discover_volpkg_convention(self, perfect_plane, tmp_path):
        from vesuvius_mesh_qa.io.discovery import discover_segments
        # Create volpkg-style directory
        seg_dir = tmp_path / "paths" / "20231001000000"
        seg_dir.mkdir(parents=True)
        obj_path = seg_dir / "20231001000000.obj"
        o3d.io.write_triangle_mesh(str(obj_path), perfect_plane)
        segments = discover_segments(tmp_path)
        assert len(segments) == 1
        assert segments[0].segment_id == "20231001000000"


class TestLoader:
    def test_load_mesh(self, perfect_plane, tmp_path):
        from vesuvius_mesh_qa.io.loader import load_mesh
        obj_path = tmp_path / "test.obj"
        o3d.io.write_triangle_mesh(str(obj_path), perfect_plane)
        loaded = load_mesh(obj_path)
        assert len(loaded.triangles) == len(perfect_plane.triangles)

    def test_load_nonexistent_raises(self):
        from vesuvius_mesh_qa.io.loader import load_mesh
        import pytest
        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/file.obj")
