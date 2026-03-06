"""CT volume accessor for mesh-to-volume coordinate mapping.

Handles remote OME-Zarr access, coordinate mapping (mesh XYZ -> volume ZYX),
and neighborhood sampling for structure tensor computation.
"""
from __future__ import annotations

import numpy as np
import zarr
import fsspec


class VolumeAccessor:
    """Lazy accessor for CT volume data at mesh vertex positions."""

    def __init__(self, zarr_url: str, scale: int = 0):
        store = fsspec.get_mapper(zarr_url)
        group = zarr.open(store, mode='r')
        self._vol = group[str(scale)]
        self._shape = self._vol.shape  # (Z, Y, X)
        self._scale_factor = 2 ** scale

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def _xyz_to_zyx(self, vertex_xyz: np.ndarray) -> tuple[int, int, int]:
        """Convert mesh (X, Y, Z) to volume index (iz, iy, ix)."""
        s = self._scale_factor
        x, y, z = vertex_xyz
        return int(round(z / s)), int(round(y / s)), int(round(x / s))

    def vertex_in_bounds(self, vertex_xyz: np.ndarray, margin: int = 16) -> bool:
        """Check if vertex + margin fits in volume."""
        iz, iy, ix = self._xyz_to_zyx(vertex_xyz)
        return (margin <= iz < self._shape[0] - margin and
                margin <= iy < self._shape[1] - margin and
                margin <= ix < self._shape[2] - margin)

    def sample_neighborhood(
        self, vertex_xyz: np.ndarray, half_size: int = 16
    ) -> np.ndarray:
        """Fetch a cubic neighborhood centered at a mesh vertex.

        Returns float32 array of shape (2*half_size, 2*half_size, 2*half_size).
        """
        iz, iy, ix = self._xyz_to_zyx(vertex_xyz)
        h = half_size
        chunk = self._vol[iz - h : iz + h, iy - h : iy + h, ix - h : ix + h]
        return np.array(chunk, dtype=np.float32)

    def sample_intensities(self, vertices_xyz: np.ndarray) -> np.ndarray:
        """Sample single-voxel intensities at each vertex position."""
        result = np.zeros(len(vertices_xyz), dtype=np.float32)
        for i, v in enumerate(vertices_xyz):
            iz, iy, ix = self._xyz_to_zyx(v)
            if (0 <= iz < self._shape[0] and
                0 <= iy < self._shape[1] and
                0 <= ix < self._shape[2]):
                result[i] = float(self._vol[iz, iy, ix])
        return result
