"""CSV report generation for batch scoring."""

from __future__ import annotations

from typing import Any

import open3d as o3d

from vesuvius_mesh_qa.io.discovery import SegmentInfo
from vesuvius_mesh_qa.metrics.base import MetricResult


def build_csv_row(
    seg: SegmentInfo,
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    aggregate: float,
    grade: str,
) -> dict[str, Any]:
    """Build a dict representing one CSV row for a scored segment."""
    row: dict[str, Any] = {
        "segment_id": seg.segment_id,
        "obj_path": str(seg.obj_path),
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.triangles),
    }
    for r in results:
        row[r.name] = round(r.score, 4)
    row["aggregate_score"] = round(aggregate, 4)
    row["grade"] = grade
    return row
