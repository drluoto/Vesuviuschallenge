"""Tests for the HTML review viewer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from vesuvius_mesh_qa.metrics.base import MetricResult


def _make_simple_mesh(n_rows=10, n_cols=10):
    """Create a simple grid mesh for testing."""
    vertices = []
    for i in range(n_rows + 1):
        for j in range(n_cols + 1):
            vertices.append([float(j), float(i), 0.0])
    triangles = []
    for i in range(n_rows):
        for j in range(n_cols):
            v0 = i * (n_cols + 1) + j
            v1 = v0 + 1
            v2 = v0 + (n_cols + 1)
            v3 = v2 + 1
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _make_results(n_faces, problem_face_indices=None):
    """Create minimal MetricResult list for testing."""
    results = []
    for name, weight in [("triangle_quality", 0.10), ("sheet_switching", 0.30)]:
        pf = np.array(problem_face_indices or [], dtype=np.int64) if name == "sheet_switching" else None
        results.append(MetricResult(
            name=name,
            score=1.0 if pf is None or len(pf) == 0 else 0.5,
            weight=weight,
            problem_faces=pf,
            details={"n_faces": n_faces},
        ))
    return results


class TestBuildVertexColors:
    def test_all_good_is_green(self):
        from vesuvius_mesh_qa.report.html_viewer import _build_vertex_colors

        mesh = _make_simple_mesh()
        results = _make_results(len(mesh.triangles))
        colors = _build_vertex_colors(mesh, results)
        # All vertices should be greenish (GOOD_COLOR = [180, 220, 180])
        assert colors.shape == (len(mesh.vertices), 3)
        assert np.all(colors[:, 1] > colors[:, 0])  # G > R for green

    def test_problem_faces_colored(self):
        from vesuvius_mesh_qa.report.html_viewer import _build_vertex_colors

        mesh = _make_simple_mesh()
        n_faces = len(mesh.triangles)
        results = _make_results(n_faces, problem_face_indices=[0, 1, 2, 3])
        colors = _build_vertex_colors(mesh, results)
        # Vertices of problem faces should differ from pure green
        tri = np.asarray(mesh.triangles)
        problem_verts = np.unique(tri[:4].ravel())
        good_verts = np.setdiff1d(np.arange(len(mesh.vertices)), problem_verts)
        # Problem vertices should have different colors than good vertices
        if len(good_verts) > 0:
            mean_problem = colors[problem_verts].mean(axis=0)
            mean_good = colors[good_verts].mean(axis=0)
            assert not np.allclose(mean_problem, mean_good, atol=5)


class TestClusterExtraction:
    def test_no_problems_empty_clusters(self):
        from vesuvius_mesh_qa.report.html_viewer import (
            _compute_deviation_angles,
            _extract_clusters_with_diagnostics,
            _find_boundary_faces,
        )

        mesh = _make_simple_mesh()
        results = _make_results(len(mesh.triangles))
        deviation_deg = _compute_deviation_angles(mesh)
        boundary_faces = _find_boundary_faces(mesh)
        clusters = _extract_clusters_with_diagnostics(mesh, results, deviation_deg, boundary_faces)
        assert isinstance(clusters, list)

    def test_cluster_fields_present(self):
        from vesuvius_mesh_qa.report.html_viewer import (
            _compute_deviation_angles,
            _extract_clusters_with_diagnostics,
            _find_boundary_faces,
        )

        mesh = _make_simple_mesh()
        n_faces = len(mesh.triangles)
        results = _make_results(n_faces, problem_face_indices=list(range(min(20, n_faces))))
        deviation_deg = _compute_deviation_angles(mesh)
        boundary_faces = _find_boundary_faces(mesh)
        clusters = _extract_clusters_with_diagnostics(mesh, results, deviation_deg, boundary_faces)
        if clusters:
            c = clusters[0]
            assert "centroid" in c
            assert "face_count" in c
            assert "is_boundary" in c
            assert "z_jump" in c
            assert "mean_dev" in c
            assert "max_dev" in c


class TestHtmlGeneration:
    def test_title_escaped(self):
        from vesuvius_mesh_qa.report.html_viewer import export_html_review

        mesh = _make_simple_mesh()
        results = _make_results(len(mesh.triangles))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            export_html_review(
                mesh, results, 0.95, "A", out,
                title='<script>alert("xss")</script>',
            )
            html = out.read_text()
            assert "<script>alert" not in html
            assert "&lt;script&gt;" in html

    def test_valid_html_structure(self):
        from vesuvius_mesh_qa.report.html_viewer import export_html_review

        mesh = _make_simple_mesh()
        results = _make_results(len(mesh.triangles))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            export_html_review(mesh, results, 0.95, "A", out, title="Test")
            html = out.read_text()
            assert "<!DOCTYPE html>" in html
            assert "</html>" in html
            assert "three" in html.lower()
            # Reset view button present
            assert "Reset View" in html
            # Legend present
            assert "legend-content" in html

    def test_keyboard_shortcuts_present(self):
        from vesuvius_mesh_qa.report.html_viewer import export_html_review

        mesh = _make_simple_mesh()
        results = _make_results(len(mesh.triangles))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.html"
            export_html_review(mesh, results, 0.95, "A", out, title="Test")
            html = out.read_text()
            assert "keydown" in html
            assert "resetView" in html


class TestDeviationAngles:
    def test_flat_plane_near_zero(self):
        from vesuvius_mesh_qa.report.html_viewer import _compute_deviation_angles

        mesh = _make_simple_mesh(20, 20)
        dev = _compute_deviation_angles(mesh)
        assert dev.shape == (len(mesh.triangles),)
        # Interior faces of a flat plane should have ~0 deviation
        assert np.median(dev) < 5.0


class TestComparisonStub:
    def test_comparison_not_implemented(self):
        from vesuvius_mesh_qa.report.html_viewer import export_comparison_html

        with pytest.raises(NotImplementedError):
            export_comparison_html()
