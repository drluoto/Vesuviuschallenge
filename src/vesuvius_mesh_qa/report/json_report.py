"""JSON report generation for single-segment scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricResult


def build_json_report(
    mesh_path: Path,
    mesh: o3d.geometry.TriangleMesh,
    results: list[MetricResult],
    aggregate: float,
    grade: str,
) -> dict[str, Any]:
    """Build a JSON-serializable quality report."""
    return {
        "file": str(mesh_path),
        "mesh_stats": {
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.triangles),
        },
        "metrics": [_metric_to_dict(r) for r in results],
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
