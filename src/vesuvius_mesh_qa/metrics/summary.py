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
from vesuvius_mesh_qa.metrics.fiber_coherence import FiberCoherenceMetric
from vesuvius_mesh_qa.metrics.layer_distance import LayerDistanceMetric
from vesuvius_mesh_qa.metrics.winding_angle import WindingAngleMetric, load_umbilicus
from vesuvius_mesh_qa.volume import VolumeAccessor


DEFAULT_METRICS: list[type[MetricComputer]] = [
    TriangleQualityMetric,
    TopologyMetric,
    NormalConsistencyMetric,
    SheetSwitchingMetric,
    SelfIntersectionMetric,
    NoiseMetric,
]

# Tier-specific weight tables from the design doc.
# Weights sum to 1.0 within each tier.
TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "tier1": {
        "triangle_quality": 0.10,
        "topology": 0.10,
        "normal_consistency": 0.10,
        "sheet_switching": 0.30,
        "self_intersections": 0.25,
        "noise": 0.15,
    },
    "tier2": {
        "triangle_quality": 0.10,
        "topology": 0.10,
        "normal_consistency": 0.10,
        "sheet_switching": 0.20,
        "self_intersections": 0.20,
        "noise": 0.10,
        "ct_sheet_switching": 0.00,
        "fiber_coherence": 0.10,
        "layer_distance": 0.10,
    },
    "tier3": {
        "triangle_quality": 0.05,
        "topology": 0.10,
        "normal_consistency": 0.05,
        "sheet_switching": 0.15,
        "self_intersections": 0.15,
        "noise": 0.05,
        "ct_sheet_switching": 0.00,
        "fiber_coherence": 0.10,
        "layer_distance": 0.10,
        "winding_angle": 0.25,
    },
}


def _suppress_noisy_metrics(results: list[MetricResult]) -> None:
    """Zero-weight metrics that lack reliable data, then redistribute."""
    # Suppress CT metrics when fiber coherence fell back to structure tensor
    fiber = next((r for r in results if r.name == "fiber_coherence"), None)
    if fiber is not None and fiber.details.get("method", "").startswith("structure_tensor"):
        for r in results:
            if r.name in ("ct_sheet_switching", "fiber_coherence"):
                r.weight = 0.0

    # Suppress any metric with too few samples
    for r in results:
        if r.details.get("n_sampled", float("inf")) < 50:
            r.weight = 0.0

    # Redistribute remaining weights to sum to 1.0
    total = sum(r.weight for r in results)
    if 0 < total < 1.0:
        scale = 1.0 / total
        for r in results:
            if r.weight > 0:
                r.weight *= scale


def _detect_tier(has_volume: bool, has_umbilicus: bool) -> str:
    if has_volume and has_umbilicus:
        return "tier3"
    if has_volume:
        return "tier2"
    return "tier1"


def compute_all_metrics(
    mesh: o3d.geometry.TriangleMesh,
    weight_overrides: dict[str, float] | None = None,
    on_progress: callable | None = None,
    volume_url: str | None = None,
    umbilicus: str | tuple[float, float] | None = None,
    fiber_model_path: str | None = None,
    fiber_predictions_url: str | None = None,
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
        umbilicus: Optional umbilicus data for winding angle metric.
            Can be a file path, (x, z) tuple, or "x,z" string.
        fiber_model_path: Optional path to nnUNet fiber model folder.
        fiber_predictions_url: Optional URL/path to pre-computed fiber predictions.

    Returns:
        List of MetricResult from each metric.
    """
    # Build metrics list: default metrics + optional CT/winding metrics
    metrics: list[MetricComputer] = [cls() for cls in DEFAULT_METRICS]

    if volume_url is not None:
        accessor = VolumeAccessor(volume_url)
        ct_metric = CTSheetSwitchingMetric(accessor)
        metrics.append(ct_metric)
        fiber_metric = FiberCoherenceMetric(
            accessor,
            fiber_model_path=fiber_model_path,
            fiber_predictions_url=fiber_predictions_url,
        )
        metrics.append(fiber_metric)
        layer_metric = LayerDistanceMetric(accessor)
        metrics.append(layer_metric)

    if umbilicus is not None:
        umb_func = load_umbilicus(umbilicus)
        winding_metric = WindingAngleMetric(umb_func)
        metrics.append(winding_metric)

    # Apply tier-appropriate default weights
    tier = _detect_tier(volume_url is not None, umbilicus is not None)
    tier_weights = TIER_WEIGHTS[tier]
    for metric in metrics:
        if metric.name in tier_weights:
            metric.weight = tier_weights[metric.name]

    n_metrics = len(metrics)
    results = []
    for i, metric in enumerate(metrics):
        # User overrides take priority over tier defaults
        if weight_overrides and metric.name in weight_overrides:
            metric.weight = weight_overrides[metric.name]
        if on_progress:
            on_progress(metric.name, i, n_metrics)
        result = metric.compute(mesh)
        results.append(result)
        gc.collect()
    _suppress_noisy_metrics(results)
    return results


def aggregate_score(results: list[MetricResult]) -> float:
    """Compute weighted aggregate score from metric results."""
    total_weight = sum(r.weight for r in results)
    if total_weight == 0:
        return 0.0
    return sum(r.weighted_score for r in results) / total_weight


def letter_grade(score: float) -> str:
    """Convert a 0-1 score to a letter grade."""
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"
