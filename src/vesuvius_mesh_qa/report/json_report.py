"""JSON report generation for single-segment scoring."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricResult

# Known Vesuvius Challenge scroll identifiers
_SCROLL_PATTERN = re.compile(r"(PHerc\d+[A-Z]?|PHercParis\d+(?:Fr\d+)?)")
_SEGMENT_PATTERN = re.compile(r"(\d{14})")

SCHEMA_VERSION = "1.0"


def _extract_identifiers(mesh_path: Path) -> dict[str, str | None]:
    """Extract scroll_id and segment_id from the file path.

    Recognizes Vesuvius Challenge naming conventions:
    - Scroll IDs: PHerc0332, PHerc1667, PHercParis4, etc.
    - Segment IDs: 14-digit timestamps like 20231210121321
    """
    path_str = str(mesh_path)

    scroll_match = _SCROLL_PATTERN.search(path_str)
    scroll_id = scroll_match.group(1) if scroll_match else None

    segment_match = _SEGMENT_PATTERN.search(mesh_path.stem)
    segment_id = segment_match.group(1) if segment_match else mesh_path.stem

    return {"scroll_id": scroll_id, "segment_id": segment_id}


def build_json_report(
    mesh_path: Path,
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    aggregate: float,
    grade: str,
) -> dict[str, Any]:
    """Build a JSON-serializable quality report."""
    ids = _extract_identifiers(mesh_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "scroll_id": ids["scroll_id"],
        "segment_id": ids["segment_id"],
        "file": str(mesh_path),
        "mesh_stats": {
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.triangles),
        },
        "metrics": {r.name: _metric_to_dict(r) for r in results},
        "aggregate_score": round(aggregate, 4),
        "grade": grade,
    }


def _metric_to_dict(r: MetricResult) -> dict[str, Any]:
    details = {}
    for k, v in r.details.items():
        if isinstance(v, (np.integer, np.floating)):
            details[k] = float(v)
        elif isinstance(v, np.ndarray):
            details[k] = v.tolist()
        else:
            details[k] = v

    d: dict[str, Any] = {
        "name": r.name,
        "score": round(r.score, 4),
        "weight": r.weight,
        "weighted_score": round(r.weighted_score, 4),
        "details": details,
    }
    if r.problem_faces is not None:
        d["n_problem_faces"] = len(r.problem_faces)
    return d
