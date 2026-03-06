"""Aggregate scoring across all metrics."""

from __future__ import annotations

import gc

import open3d as o3d

from vesuvius_mesh_qa.metrics.base import MetricComputer, MetricResult
from vesuvius_mesh_qa.metrics.triangle import TriangleQualityMetric
from vesuvius_mesh_qa.metrics.topology import TopologyMetric
from vesuvius_mesh_qa.metrics.normals import NormalConsistencyMetric, SheetSwitchingMetric
from vesuvius_mesh_qa.metrics.intersections import SelfIntersectionMetric
from vesuvius_mesh_qa.metrics.noise import NoiseMetric
from vesuvius_mesh_qa.metrics.ct_switching import CTSheetSwitchingMetric
from vesuvius_mesh_qa.volume import VolumeAccessor


DEFAULT_METRICS: list[type[MetricComputer]] = [
    TriangleQualityMetric,
    TopologyMetric,
    NormalConsistencyMetric,
    SheetSwitchingMetric,
    SelfIntersectionMetric,
    NoiseMetric,
]


def compute_all_metrics(
    mesh: o3d.geometry.TriangleMesh,
    weight_overrides: dict[str, float] | None = None,
    on_progress: callable | None = None,
    volume_url: str | None = None,
) -> list[MetricResult]:
    """Compute all metrics on a mesh.

    Args:
        mesh: Open3D triangle mesh with normals computed.
        weight_overrides: Optional dict of {metric_name: new_weight}.
        on_progress: Optional callback(metric_name, index, total) called
            before each metric computation.
        volume_url: Optional OME-Zarr volume URL for CT-informed sheet
            switching detection. When provided, a CTSheetSwitchingMetric
            is appended to the default metrics.

    Returns:
        List of MetricResult from each metric.
    """
    # Build metrics list: default metrics + optional CT metric
    metrics: list[MetricComputer] = [cls() for cls in DEFAULT_METRICS]

    if volume_url is not None:
        accessor = VolumeAccessor(volume_url)
        ct_metric = CTSheetSwitchingMetric(accessor)
        metrics.append(ct_metric)

    n_metrics = len(metrics)
    results = []
    for i, metric in enumerate(metrics):
        if weight_overrides and metric.name in weight_overrides:
            metric.weight = weight_overrides[metric.name]
        if on_progress:
            on_progress(metric.name, i, n_metrics)
        result = metric.compute(mesh)
        results.append(result)
        gc.collect()
    return results


def aggregate_score(results: list[MetricResult]) -> float:
    """Compute weighted aggregate score from metric results."""
    total_weight = sum(r.weight for r in results)
    if total_weight == 0:
        return 0.0
    return sum(r.weighted_score for r in results) / total_weight


def letter_grade(score: float) -> str:
    """Convert a 0-1 score to a letter grade."""
    if score > 0.9:
        return "A"
    if score > 0.75:
        return "B"
    if score > 0.6:
        return "C"
    if score > 0.4:
        return "D"
    return "F"
