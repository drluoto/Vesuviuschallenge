"""Base classes for mesh quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d


@dataclass
class MetricResult:
    """Result from a single metric computation."""

    name: str
    score: float  # 0.0 (worst) to 1.0 (best)
    weight: float
    details: dict[str, Any] = field(default_factory=dict)
    problem_faces: np.ndarray | None = None  # indices of problematic faces

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


class MetricComputer(ABC):
    """Abstract base class for mesh quality metrics."""

    name: str
    weight: float

    @abstractmethod
    def compute(self, mesh: o3d.geometry.TriangleMesh) -> MetricResult:
        """Compute the metric on the given mesh.

        Args:
            mesh: An Open3D triangle mesh (with normals computed).

        Returns:
            MetricResult with score in [0, 1] and optional details.
        """
        ...
