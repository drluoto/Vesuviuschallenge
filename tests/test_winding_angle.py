"""Tests for winding angle consistency metric."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from vesuvius_mesh_qa.metrics.winding_angle import (
    WindingAngleMetric,
    load_umbilicus,
    compute_winding_angles_bfs,
)


def _make_cylinder_mesh(
    n_rows: int = 40,
    n_cols: int = 60,
    radius: float = 100.0,
    height: float = 200.0,
    angle_range: float = 1.5,  # radians, how much of the cylinder to cover
) -> o3d.geometry.TriangleMesh:
    """Create a mesh patch on a cylinder surface (like a scroll segment).

    The cylinder axis is along Y.  The umbilicus is at (0, *, 0).
    """
    vertices = []
    for i in range(n_rows + 1):
        for j in range(n_cols + 1):
            theta = angle_range * j / n_cols  # angular position
            y = height * i / n_rows
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            vertices.append([x, y, z])

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


def _make_sheet_switch_cylinder(
    n_rows: int = 40,
    n_cols: int = 60,
    radius: float = 100.0,
    height: float = 200.0,
    angle_range: float = 1.5,
    switch_row: int = 20,
    angular_offset_deg: float = 25.0,  # angular jump simulating winding switch
) -> o3d.geometry.TriangleMesh:
    """Cylinder mesh that switches to a different winding at switch_row.

    At the switch row, the angular position jumps (simulating a sheet
    switch to an adjacent papyrus layer at a different winding position).
    In a real scroll, adjacent layers are at different accumulated winding
    angles — a switch creates an angular discontinuity in the BFS field.
    """
    angular_offset_rad = np.radians(angular_offset_deg)
    vertices = []
    for i in range(n_rows + 1):
        for j in range(n_cols + 1):
            theta = angle_range * j / n_cols
            if i >= switch_row:
                theta += angular_offset_rad
            y = height * i / n_rows
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            vertices.append([x, y, z])

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


@pytest.fixture
def cylinder_mesh():
    """Clean cylinder patch — should score high."""
    return _make_cylinder_mesh()


@pytest.fixture
def sheet_switch_cylinder():
    """Cylinder with radial layer jump — should score lower."""
    return _make_sheet_switch_cylinder()


@pytest.fixture
def simple_umbilicus():
    """Umbilicus at origin along Y axis: (x=0, z=0) at all Y."""
    return np.array([[0.0, 0.0, 0.0], [0.0, 200.0, 0.0]])


class TestLoadUmbilicus:
    def test_load_from_array(self):
        """Load umbilicus from (N, 3) array."""
        data = np.array([[10.0, 0.0, 20.0], [10.0, 100.0, 20.0]])
        umb = load_umbilicus(data)
        # Should interpolate to get (x, z) at any y
        xz = umb(50.0)
        assert abs(xz[0] - 10.0) < 1e-6
        assert abs(xz[1] - 20.0) < 1e-6

    def test_load_from_file(self, tmp_path):
        """Load umbilicus from text file."""
        f = tmp_path / "umbilicus.txt"
        f.write_text("10.0 0.0 20.0\n10.0 100.0 20.0\n10.0 200.0 20.0\n")
        umb = load_umbilicus(str(f))
        xz = umb(100.0)
        assert abs(xz[0] - 10.0) < 1e-6
        assert abs(xz[1] - 20.0) < 1e-6

    def test_load_from_simple_pair(self):
        """Load umbilicus from simple (x, z) center point."""
        umb = load_umbilicus((50.0, 30.0))
        # Should return same (x, z) for any y
        xz = umb(0.0)
        assert abs(xz[0] - 50.0) < 1e-6
        assert abs(xz[1] - 30.0) < 1e-6
        xz = umb(999.0)
        assert abs(xz[0] - 50.0) < 1e-6
        assert abs(xz[1] - 30.0) < 1e-6


class TestComputeWindingAnglesBFS:
    def test_smooth_on_cylinder(self, cylinder_mesh, simple_umbilicus):
        """Winding angles should be smooth on a clean cylinder patch."""
        umb = load_umbilicus(simple_umbilicus)
        vertices = np.asarray(cylinder_mesh.vertices)
        triangles = np.asarray(cylinder_mesh.triangles)
        angles = compute_winding_angles_bfs(vertices, triangles, umb)

        assert len(angles) == len(vertices)
        assert not np.any(np.isnan(angles))

        # On a cylinder, angles should span the angle_range (~1.5 rad = ~86 deg)
        angle_span = angles.max() - angles.min()
        assert angle_span > 50.0  # at least 50 degrees
        assert angle_span < 120.0  # not more than expected

    def test_detects_jump_on_sheet_switch(self, sheet_switch_cylinder, simple_umbilicus):
        """BFS winding angles should show a jump at the sheet switch boundary."""
        umb = load_umbilicus(simple_umbilicus)
        vertices = np.asarray(sheet_switch_cylinder.vertices)
        triangles = np.asarray(sheet_switch_cylinder.triangles)
        angles = compute_winding_angles_bfs(vertices, triangles, umb)

        assert len(angles) == len(vertices)
        assert not np.any(np.isnan(angles))


class TestWindingAngleMetric:
    def test_clean_cylinder_scores_high(self, cylinder_mesh, simple_umbilicus):
        """Clean cylinder should get a high winding angle score."""
        umb = load_umbilicus(simple_umbilicus)
        metric = WindingAngleMetric(umb)
        result = metric.compute(cylinder_mesh)

        assert result.name == "winding_angle"
        assert result.score > 0.9
        assert "n_discontinuous_edges" in result.details

    def test_sheet_switch_scores_lower(self, sheet_switch_cylinder, simple_umbilicus):
        """Sheet-switch cylinder should score lower than clean cylinder."""
        umb = load_umbilicus(simple_umbilicus)
        metric = WindingAngleMetric(umb)

        clean = metric.compute(_make_cylinder_mesh())
        switched = metric.compute(sheet_switch_cylinder)

        assert switched.score < clean.score

    def test_weight_default(self, simple_umbilicus):
        """Default weight should be 0.15."""
        umb = load_umbilicus(simple_umbilicus)
        metric = WindingAngleMetric(umb)
        assert metric.weight == 0.15
